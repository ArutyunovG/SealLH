import torch
import torch.nn as nn
from mmdet.models.losses import DDQAuxLoss

from mmengine import ConfigDict

from seallh.experiment.utils import import_class

class AuxLoss(DDQAuxLoss):
    def forward(self, *args, **kwargs):
        # compatibility hack: modify img_shape to exclude channel dimension
        for i in range(len(args[4])):
            args[4][i]['img_shape'] = args[4][i]['img_shape'][:2]
        return self.loss(*args, **kwargs)


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

    def forward(self, raw_outputs, targets, meta):
        
        def prepare_targets(targets, meta):
            gt_labels = []
            gt_bboxes = []
            meta_list = []
            for i in range(len(meta['img_path'])):
                t = targets[targets[:, 0] == i]
                l = t[:, 1].long()
                b = t[:, 3:]
                gt_labels.append(l)
                gt_bboxes.append(b)

                meta_list.append({
                    'img_shape': [*meta['img1_shape'], 3],
                    'scale_factor': 1.0,
                })
            return gt_labels, gt_bboxes, meta_list

        gt_labels, gt_bboxes, meta_list = prepare_targets(targets, meta=meta)

        main_results, aux_results = raw_outputs
        main_loss_dict = self.main_criterion(*main_results, gt_bboxes, gt_labels, meta_list)

        loss_dict = {
            'loss_cls': torch.stack(main_loss_dict['aux_loss_cls']).mean(),
            'loss_bbox': torch.stack(main_loss_dict['aux_loss_bbox']).mean(),
        }

        if aux_results[0] is not None:
            aux_loss_dict = self.aux_criterion(*aux_results, gt_bboxes, gt_labels, meta_list)
            loss_dict['aux_loss_cls'] = torch.stack(aux_loss_dict['aux_loss_cls']).mean()
            loss_dict['aux_loss_bbox'] = torch.stack(aux_loss_dict['aux_loss_bbox']).mean()
            
        return loss_dict

    @property
    def num_train_losses(self):
        return 4
    
    @property
    def num_val_losses(self):
        return 2