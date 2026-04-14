import torch
import torch.nn as nn
from mmdet.models.losses import DDQAuxLoss
from mmdet.models.task_modules.assigners import TopkHungarianAssigner

from mmengine import ConfigDict

from seallh.experiment.utils import import_class

# Monkey-patch mmdet bug: TopkHungarianAssigner sets labels=None when no
# predictions are matched (e.g. more GTs than queries after NMS), which
# crashes SamplingResult. The fix: never overwrite assigned_labels with None.
_original_assign = TopkHungarianAssigner.assign

def _patched_assign(self, *args, **kwargs):
    result = _original_assign(self, *args, **kwargs)
    if result.labels is None:
        num_bboxes = result.gt_inds.size(0)
        result.labels = result.gt_inds.new_full((num_bboxes,), -1)
    return result

TopkHungarianAssigner.assign = _patched_assign


class DDQLoss(nn.Module):

    def __init__(self, main_criterion, aux_criterion):
        
        super(DDQLoss, self).__init__()

        def _dictlike(value):
            return value if hasattr(value, 'get') else {}

        def _build_cfg(criterion, *, default_assigner_topk=None):
            args = getattr(criterion, 'args', None)
            args = _dictlike(args)

            cls_args = _dictlike(getattr(args, 'cls', None) if not isinstance(args, dict) else args.get('cls'))
            bbox_args = _dictlike(getattr(args, 'bbox', None) if not isinstance(args, dict) else args.get('bbox'))
            train_cfg_args = _dictlike(getattr(args, 'train_cfg', None) if not isinstance(args, dict) else args.get('train_cfg'))
            assigner_args = _dictlike(
                getattr(train_cfg_args, 'assigner', None)
                if not isinstance(train_cfg_args, dict)
                else train_cfg_args.get('assigner')
            )

            assigner = dict(type=assigner_args.get('type', 'TopkHungarianAssigner'))
            if 'topk' in assigner_args:
                assigner['topk'] = assigner_args.get('topk')
            elif default_assigner_topk is not None:
                assigner['topk'] = assigner_args.get('topk', default_assigner_topk)

            return ConfigDict(
                loss_cls=dict(
                    type=cls_args.get('type', 'QualityFocalLoss'),
                    use_sigmoid=cls_args.get('use_sigmoid', True),
                    activated=cls_args.get('activated', True),
                    beta=cls_args.get('beta', 2.0),
                    loss_weight=cls_args.get('loss_weight', 1.0),
                ),
                loss_bbox=dict(
                    type=bbox_args.get('type', 'GIoULoss'),
                    loss_weight=bbox_args.get('loss_weight', 2.0),
                ),
                train_cfg=dict(assigner=assigner),
            )

        main_criterion_cls = import_class(main_criterion["class"])
        self.main_criterion = main_criterion_cls(**_build_cfg(main_criterion))

        aux_criterion_cls = import_class(aux_criterion["class"])
        self.aux_criterion = aux_criterion_cls(**_build_cfg(aux_criterion, default_assigner_topk=8))

    def forward(self, raw_outputs, targets, device):

        def prepare_targets(targets):
            gt_labels = []
            gt_bboxes = []
            meta_list = []

            for item in targets:
                if not isinstance(item, dict):
                    raise TypeError('Each target must be a dict')

                labels = item.get('labels', [])
                bboxes = item.get('bboxes', [])

                if len(labels) == 0:
                    gt_labels.append(torch.empty((0,), dtype=torch.long))
                else:
                    gt_labels.append(torch.tensor(labels, dtype=torch.long))

                if len(bboxes) == 0:
                    gt_bboxes.append(torch.empty((0, 4), dtype=torch.float))
                else:
                    gt_bboxes.append(torch.tensor(bboxes, dtype=torch.float))

                shape = item.get('img_shape')
                if shape is None:
                    raise ValueError('img_shape missing in target item')

                h_w = list(shape)
                assert len(h_w) == 2, 'img_shape must be (height, width)'

                meta_list.append({
                    'img_shape': h_w,
                    'scale_factor': 1.0,
                })

            return gt_labels, gt_bboxes, meta_list

        gt_labels, gt_bboxes, meta_list = prepare_targets(targets)

        main_results, aux_results = raw_outputs

        gt_labels = [t.to(device) for t in gt_labels]
        gt_bboxes = [t.to(device) for t in gt_bboxes]

        main_cls_scores, _ = main_results
        if len(main_cls_scores) == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            loss_dict = {'loss_cls': zero, 'loss_bbox': zero}
        else:
            main_loss_dict = self.main_criterion.loss(*main_results, gt_bboxes, gt_labels, meta_list)
            loss_dict = {
                'loss_cls': torch.stack(main_loss_dict['aux_loss_cls']).mean(),
                'loss_bbox': torch.stack(main_loss_dict['aux_loss_bbox']).mean(),
            }

            aux_cls, aux_bbox = aux_results
            if aux_cls is not None and aux_bbox is not None:
                aux_loss_dict = self.aux_criterion.loss(*aux_results, gt_bboxes, gt_labels, meta_list)
                loss_dict['aux_loss_cls'] = torch.stack(aux_loss_dict['aux_loss_cls']).mean()
                loss_dict['aux_loss_bbox'] = torch.stack(aux_loss_dict['aux_loss_bbox']).mean()
            
        return loss_dict

    @property
    def num_train_losses(self):
        return 4
    
    @property
    def num_val_losses(self):
        return 2
