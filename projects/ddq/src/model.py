import torch
import torch.nn as nn
import torchvision
from torch import Tensor

from copy import deepcopy
import logging
from typing import Optional


logger = logging.getLogger("seallh.projects.src.ddq.model")

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")# if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def bbox_decode(distance, anchor_points, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


class DDQFCN(nn.Module):
    def __init__(self, backbone: nn.Module, neck: nn.Module, bbox_head: nn.Module, num_distinct_queries=300):
        super(DDQFCN, self).__init__()
        self.backbone = backbone
        self.neck = nn.Sequential(neck)
        self.bbox_head = bbox_head

        self.nc = self.bbox_head.num_classes
        self.num_distinct_queries = num_distinct_queries
        self.strides = self.bbox_head.strides

    def init_weights(self):
        initialize_module_weights(self.neck, validate=True)
        initialize_module_weights(self.bbox_head, validate=True)
        
    def forward(self, x: Tensor):
        out = self._forward(x)
        return out

    def _forward(self, x: Tensor):
        backbone_feats = self.backbone(x)
        neck_feat = self.neck(backbone_feats)
        outputs = self.bbox_head(neck_feat)
        if torch.onnx.is_in_onnx_export(): 
            scores, bboxes = outputs

            anchor_points, stride_tensor = make_anchors(bboxes, self.strides, grid_cell_offset=0.5)

            batch_size = bboxes[0].size(0)
            
            pred_bboxes = torch.cat([xi.view(batch_size, 4, -1) for xi in bboxes], 2).permute(0, 2, 1).contiguous()
            pred_scores = torch.cat([xi.view(batch_size, 1, -1) for xi in scores], 2).permute(0, 2, 1).contiguous()
            
            pred_bboxes = bbox_decode(pred_bboxes, anchor_points)  # xyxy, (b, h*w, 4)
            pred_bboxes = pred_bboxes * stride_tensor
            
            return pred_scores, pred_bboxes
        
        return outputs

    def load_from_checkpoint(self, ckpt_fp, strict=True, coco2human=False, verbose=True):
        ckpt = torch.load(ckpt_fp, map_location='cpu')
        if coco2human:
            for k, v in ckpt.items():
                if k.startswith('bbox_head.conv_cls') or k.startswith('bbox_head.aux_conv_cls'):
                    ckpt[k] = v[0].unsqueeze(0)
        else:
            if 'model' in ckpt:
                ckpt = ckpt['model']

        self.load_state_dict(state_dict=ckpt, strict=strict)
        del ckpt

    def get_param_groups(self, no_decay_bn_filter_bias, wd):
        return parameter_list(self.named_parameters, weight_decay=wd, no_decay_bn_filter_bias=no_decay_bn_filter_bias)

    def postprocess(self, raw_output):
        main_result, _ = raw_output
        num_classes = self.nc
        num_distinct_queries = self.num_distinct_queries
        result_list = []

        cls_scores, bbox_preds = main_result
        for single_score, single_bbox_pred in zip(cls_scores, bbox_preds):
            single_score = single_score.flatten(0, 1)
            num_distinct_queries = min(num_distinct_queries,len(single_score))
            scores_per_img, topk_indices = single_score.topk(num_distinct_queries, sorted=True)

            labels_per_img = topk_indices % num_classes
            bboxes = single_bbox_pred[topk_indices // num_classes]
            bboxes = torch.cat([bboxes, scores_per_img[:, None]], dim=1)

            result_list.append((bboxes[:num_distinct_queries], labels_per_img[:num_distinct_queries]))

        return result_list



def initialize_weights(modules, conv_init_type, linear_init_type) -> None:
    """Helper function to initialize differnet layers in a model"""

    conv_std = None
    linear_std = 0.01

    norm_layers_tuple = (
        nn.BatchNorm2d,
        nn.SyncBatchNorm,
        nn.LayerNorm,
        nn.InstanceNorm2d,
        nn.GroupNorm,
    )

    def _g():
        yield 1
    GeneratorType = type(_g())

    if isinstance(modules, (nn.Sequential, nn.ModuleList, GeneratorType)):
        for m in modules:
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                initialize_conv_layer(
                    module=m, init_method=conv_init_type, std_val=conv_std
                )
            elif isinstance(m, norm_layers_tuple):
                initialize_norm_layers(module=m)
            elif isinstance(m, nn.Linear):
                initialize_fc_layer(
                    module=m, init_method=linear_init_type, std_val=linear_std
                )
    else:
        if isinstance(modules, (nn.Conv2d, nn.Conv3d)):
            initialize_conv_layer(
                module=modules, init_method=conv_init_type, std_val=conv_std
            )
        elif isinstance(modules, norm_layers_tuple):
            initialize_norm_layers(module=modules)
        elif isinstance(modules, nn.Linear):
            initialize_fc_layer(
                module=modules, init_method=linear_init_type, std_val=linear_std
            )

def _init_nn_layers(
    module,
    init_method: Optional[str] = "kaiming_normal",
    std_val: Optional[float] = None,
) -> None:
    """
    Helper function to initialize neural network module
    """
    supported_conv_inits = [
        "kaiming_normal",
        "kaiming_uniform",
        "xavier_normal",
        "xavier_uniform",
        "normal",
        "trunc_normal",
    ]

    init_method = init_method.lower()
    if init_method == "kaiming_normal":
        if module.weight is not None:
            nn.init.kaiming_normal_(module.weight, mode="fan_out")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif init_method == "kaiming_uniform":
        if module.weight is not None:
            nn.init.kaiming_uniform_(module.weight, mode="fan_out")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif init_method == "xavier_normal":
        if module.weight is not None:
            nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif init_method == "xavier_uniform":
        if module.weight is not None:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif init_method == "normal":
        if module.weight is not None:
            std = 1.0 / module.weight.size(1) if std_val is None else std_val
            nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif init_method == "trunc_normal":
        if module.weight is not None:
            std = 1.0 / module.weight.size(1) if std_val is None else std_val
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    else:
        supported_conv_message = "Supported initialization methods are:"
        for i, l in enumerate(supported_conv_inits):
            supported_conv_message += "\n \t {}) {}".format(i, l)
        logger.error("{} \n Got: {}".format(supported_conv_message, init_method))


def initialize_conv_layer(
    module,
    init_method: Optional[str] = "kaiming_normal",
    std_val: Optional[float] = 0.01,
) -> None:
    """Helper function to initialize convolution layers"""
    _init_nn_layers(module=module, init_method=init_method, std_val=std_val)


def initialize_fc_layer(
    module, init_method: Optional[str] = "normal", std_val: Optional[float] = 0.01
) -> None:
    """Helper function to initialize fully-connected layers"""
    if hasattr(module, "layer"):
        _init_nn_layers(module=module.layer, init_method=init_method, std_val=std_val)
    else:
        _init_nn_layers(module=module, init_method=init_method, std_val=std_val)


def initialize_norm_layers(module) -> None:
    """Helper function to initialize normalization layers"""

    def _init_fn(module):
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.ones_(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)

    _init_fn(module.layer) if hasattr(module, "layer") else _init_fn(module=module)


def initialize_module_weights(
    module: nn.Module,
    conv_init_type: str = "xavier_uniform",
    linear_init_type: str = "normal",
    validate: bool = False
):
    sd1 = None
    if validate:
        sd1 = deepcopy(module.state_dict())

    initialize_weights(module.modules(), conv_init_type, linear_init_type)

    if validate:
        sd2 = module.state_dict()

        identical = []
        changed = []

        ignore = ['running', 'var', 'mean', 'num_batches_tracked']
        for k1, v1 in sd1.items():
            v2 = sd2[k1]
            is_key_ignored = not any([x in k1 for x in ignore])
            if torch.allclose(v1, v2) and is_key_ignored:
                identical.append((k1, v1, v2))
            else:
                changed.append(k1)

        if len(identical):
            percentage = round(len(changed) / len(sd1) * 100.0)
            logger.warning(f'{percentage}% of weights changed after initialization of the module: '
                           f'{module.__class__.__name__}')


def parameter_list(
    named_parameters,
    weight_decay: Optional[float] = 0.0,
    no_decay_bn_filter_bias: Optional[bool] = False,
    *args,
    **kwargs
):
    with_decay = []
    without_decay = []
    if isinstance(named_parameters, list):
        for n_parameter in named_parameters:
            for p_name, param in n_parameter():
                if (
                    param.requires_grad
                    and len(param.shape) == 1
                    and no_decay_bn_filter_bias
                ):
                    # biases and normalization layer parameters are of len 1
                    without_decay.append(param)
                elif param.requires_grad:
                    with_decay.append(param)
    else:
        for p_name, param in named_parameters():
            if (
                param.requires_grad
                and len(param.shape) == 1
                and no_decay_bn_filter_bias
            ):
                # biases and normalization layer parameters are of len 1
                without_decay.append(param)
            elif param.requires_grad:
                with_decay.append(param)
    param_list = [{"params": with_decay, "weight_decay": weight_decay}]
    if len(without_decay) > 0:
        param_list.append({"params": without_decay, "weight_decay": 0.0})
    return param_list
