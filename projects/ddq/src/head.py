import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks import Scale
from mmcv.ops import batched_nms

from mmdet.models.task_modules.prior_generators import MlvlPointGenerator
from mmdet.structures.bbox import distance2bbox
from mmdet.models.utils import filter_scores_and_topk
from mmdet.models.utils import select_single_mlvl
from mmdet.models.utils import sigmoid_geometric_mean


def sigmoid_geometric_mean_export(x, y):
    x_sigmoid = x.sigmoid()
    y_sigmoid = y.sigmoid()
    z = (x_sigmoid * y_sigmoid).sqrt()
    return z


class StrideHead(nn.Module):
    def __init__(self, num_classes, in_channels, feat_channels, num_priors, stacked_convs=4):
        super(StrideHead, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.np = num_priors
        self.stacked_convs = stacked_convs

        self.conv_cfg = None
        self.norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        self.act_cfg = dict(type='Swish')

        self.conv_layers = nn.Sequential()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.conv_layers.append(
                ConvModule(chn, self.feat_channels, 3, stride=1, padding=3 // 2, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
            )

        self.objectness = nn.Sequential(
            nn.Conv2d(self.feat_channels, self.feat_channels // 4, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.feat_channels // 4, 1, 3, padding=3 // 2)
        )
        self.conv_cls = nn.Conv2d(self.feat_channels, self.np * self.num_classes, 3, padding=3 // 2)
        self.conv_reg = nn.Conv2d(self.feat_channels, self.np * 4, 3, padding=3 // 2)
        self.scale = Scale(1.0)

        self.aux_conv_objectness = nn.Sequential(
            nn.Conv2d(self.feat_channels, self.feat_channels // 4, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.feat_channels // 4, 1, 3, padding=3 // 2)
        )
        self.aux_conv_cls = nn.Conv2d(self.feat_channels, self.np * self.num_classes, 3, padding=3 // 2)
        self.aux_conv_reg = nn.Conv2d(self.feat_channels, self.np * 4, 3, padding=3 // 2)
        self.aux_scale = Scale(1.0)

    def forward(self, x):
        feat = self.conv_layers(x)

        cls_logits = self.conv_cls(feat)
        object_nesss = self.objectness(feat)

        if torch.onnx.is_in_onnx_export():
            cls_scores = sigmoid_geometric_mean_export(cls_logits, object_nesss)
        else:
            cls_scores = sigmoid_geometric_mean(cls_logits, object_nesss)

        bbox_preds = self.conv_reg(feat).exp()
        bbox_preds = self.scale(bbox_preds).float()

        main_results = (cls_scores, bbox_preds)

        if self.training:
            aux_cls_logits = self.aux_conv_cls(feat)
            aux_object_nesss = self.aux_conv_objectness(feat)
            aux_cls_scores = sigmoid_geometric_mean(aux_cls_logits, aux_object_nesss)

            aux_bbox_preds = self.aux_conv_reg(feat)
            aux_bbox_preds = self.aux_scale(aux_bbox_preds)

            aux_results = (aux_cls_scores, aux_bbox_preds)
        else:
            aux_results = (None, None)

        return main_results, aux_results


class DDQFCNHead(nn.Module):
    def __init__(
            self,
            num_classes,
            in_channels,
            feat_channels=256,
            stacked_convs=4,
            strides=(8, 16, 32, 64, 128),
            shuffle_channels=64,
            dqs_cfg=dict(type='nms', iou_threshold=0.7, nms_pre=1000),
            offset=0.5,
            num_distinct_queries=300,
            norm_cfg=None,      # todo:
    ):
        super(DDQFCNHead, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides

        self.prior_generator = MlvlPointGenerator(strides, offset=offset)
        self.num_base_priors = self.prior_generator.num_base_priors

        self.stride_heads = nn.ModuleList()
        for na in self.num_base_priors:
            head = StrideHead(self.num_classes, self.in_channels, self.feat_channels, na, self.stacked_convs)
            self.stride_heads.append(head)

        self.num_distinct_queries = num_distinct_queries
        self.dqs_cfg = dqs_cfg
        self.shuffle_channels = shuffle_channels

        # contains the tuple of level indices that will do the interaction
        self.fuse_lvl_list = []
        num_levels = len(self.prior_generator.strides)
        for lvl in range(num_levels):
            top_lvl = min(lvl + 1, num_levels - 1)
            dow_lvl = max(lvl - 1, 0)
            tar_lvl = lvl
            self.fuse_lvl_list.append((tar_lvl, top_lvl, dow_lvl))

        self.remain_chs = self.in_channels - self.shuffle_channels * 2

    def forward_base(self, inputs, **kwargs):
        shuffled_inputs = self._shuffle_features(inputs)

        cls_scores_list = []
        bbox_preds_list = []

        aux_cls_scores_list = []
        aux_bbox_preds_list = []

        for i, layer in enumerate(self.stride_heads):
            main_results, aux_results = layer(shuffled_inputs[i])
            cls_scores, bbox_preds = main_results
            aux_cls_scores, aux_bbox_preds = aux_results

            cls_scores_list.append(cls_scores)
            bbox_preds_list.append(bbox_preds)
            aux_cls_scores_list.append(aux_cls_scores)
            aux_bbox_preds_list.append(aux_bbox_preds)

        if torch.onnx.is_in_onnx_export():
            main_results = (cls_scores_list, bbox_preds_list)
            return main_results

        main_results = dict(
            cls_scores_list=cls_scores_list,
            bbox_preds_list=bbox_preds_list
        )

        aux_results = dict(
            cls_scores_list=aux_cls_scores_list,
            bbox_preds_list=aux_bbox_preds_list,
        )

        return main_results, aux_results

    def _shuffle_features(self, inputs):
        fused_inputs = []
        for fuse_lvl_tuple in self.fuse_lvl_list:
            tar_lvl, top_lvl, dow_lvl = fuse_lvl_tuple
            tar_input = inputs[tar_lvl]
            top_input = inputs[top_lvl]
            down_input = inputs[dow_lvl]
            remain = tar_input[:, :self.remain_chs]
            from_top = top_input[:, self.remain_chs:][:, self.shuffle_channels:]
            from_top = F.interpolate(from_top, size=tar_input.shape[-2:], mode='bilinear', align_corners=True)
            from_down = down_input[:, self.remain_chs:][:, :self.shuffle_channels]
            from_down = F.interpolate(from_down, size=tar_input.shape[-2:], mode='bilinear', align_corners=True)
            fused_inputs.append(torch.cat([remain, from_top, from_down], dim=1))
        return fused_inputs

    def forward(self, x):
        raw_results = self.forward_base(x)

        if torch.onnx.is_in_onnx_export():
            return raw_results

        # get dense-to-sparse predictions
        results = self.get_inputs(*raw_results)
        return results

    def get_inputs(self, main_results, aux_results, img_metas=None):

        mlvl_score = main_results['cls_scores_list']
        num_levels = len(mlvl_score)
        featmap_sizes = [mlvl_score[i].shape[-2:] for i in range(num_levels)]

        device = mlvl_score[0].device
        mlvl_priors = self.prior_generator.grid_priors(featmap_sizes, dtype=mlvl_score[0].dtype, device=device)

        all_cls_scores, all_bbox_preds, all_query_ids = self.pre_dqs(**main_results, mlvl_priors=mlvl_priors)
        # test stage
        if self.training:
            aux_cls_scores, aux_bbox_preds, all_query_ids = self.pre_dqs(**aux_results, mlvl_priors=mlvl_priors)
        else:
            (aux_cls_scores, aux_bbox_preds) = (None, None)

        nms_all_cls_scores, nms_all_bbox_preds = self.dqs(all_cls_scores, all_bbox_preds)

        return (nms_all_cls_scores, nms_all_bbox_preds), (aux_cls_scores, aux_bbox_preds)

    def dqs(self, all_mlvl_scores, all_mlvl_bboxes):
        ddq_bboxes = []
        ddq_scores = []
        for mlvl_bboxes, mlvl_scores in zip(all_mlvl_bboxes, all_mlvl_scores):
            if mlvl_bboxes.numel() == 0:
                return mlvl_bboxes, mlvl_scores

            det_bboxes, ddq_idxs = batched_nms(mlvl_bboxes, mlvl_scores.max(-1).values, torch.ones(len(mlvl_scores)), self.dqs_cfg)

            ddq_bboxes.append(mlvl_bboxes[ddq_idxs])
            ddq_scores.append(mlvl_scores[ddq_idxs])
        return ddq_scores, ddq_bboxes

    def pre_dqs(self, cls_scores_list=None, bbox_preds_list=None, mlvl_priors=None, img_metas=None, **kwargs):
        num_imgs = cls_scores_list[0].size(0)
        all_cls_scores = []
        all_bbox_preds = []
        all_query_ids = []
        for img_id in range(num_imgs):
            single_cls_score_list = select_single_mlvl(cls_scores_list, img_id, detach=False)
            sinlge_bbox_pred_list = select_single_mlvl(bbox_preds_list, img_id, detach=False)
            cls_score, bbox_pred, query_inds = self._get_topk(single_cls_score_list, sinlge_bbox_pred_list, mlvl_priors)
            all_cls_scores.append(cls_score)
            all_bbox_preds.append(bbox_pred)
            all_query_ids.append(query_inds)
        return all_cls_scores, all_bbox_preds, all_query_ids

    def _get_topk(self, cls_score_list, bbox_pred_list, mlvl_priors, **kwargs):
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_query_inds = []
        start_inds = 0
        for level_idx, (cls_score, bbox_pred, priors, stride) in enumerate(zip(cls_score_list, bbox_pred_list, mlvl_priors, self.prior_generator.strides)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.num_classes)

            binary_cls_score = cls_score.max(-1).values.reshape(-1, 1)
            if self.dqs_cfg:
                nms_pre = self.dqs_cfg.pop('nms_pre', 1000)
            else:
                if self.training:
                    nms_pre = len(binary_cls_score)
                else:
                    nms_pre = 1000
            results = filter_scores_and_topk(binary_cls_score, 0, nms_pre, dict(bbox_pred=bbox_pred, priors=priors, cls_score=cls_score))
            scores, labels, keep_idxs, filtered_results = results
            keep_idxs = keep_idxs + start_inds
            start_inds = start_inds + len(cls_score)
            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']
            cls_score = filtered_results['cls_score']
            bbox_pred = bbox_pred * stride[0]
            bbox_pred = distance2bbox(priors, bbox_pred)
            mlvl_bboxes.append(bbox_pred)
            mlvl_scores.append(cls_score)
            mlvl_query_inds.append(keep_idxs)

        return torch.cat(mlvl_scores), torch.cat(mlvl_bboxes), torch.cat(mlvl_query_inds)

    def get_bboxes(self, cls_scores, bbox_preds, img_metas=None, **kwargs):

        result_list = []
        for sinlge_score, single_bbox_pred, img_meta in zip(cls_scores, bbox_preds, img_metas):
            img_shape = img_meta['img_shape']
            single_bbox_pred[:, 0::2].clamp_(min=0, max=img_shape[1])
            single_bbox_pred[:, 1::2].clamp_(min=0, max=img_shape[0])
            single_bbox_pred = single_bbox_pred / single_bbox_pred.new_tensor(img_meta['scale_factor'])
            sinlge_score = sinlge_score.flatten(0, 1)
            num_distinct_queries = min(self.num_distinct_queries, len(sinlge_score))
            scores_per_img, topk_indices = sinlge_score.topk(num_distinct_queries, sorted=True)
            labels_per_img = topk_indices % self.num_classes
            bboxes = single_bbox_pred[topk_indices // self.num_classes]
            bboxes = torch.cat([bboxes, scores_per_img[:, None]], dim=1)

            result_list.append((bboxes[:num_distinct_queries], labels_per_img[:num_distinct_queries]))
        return result_list
