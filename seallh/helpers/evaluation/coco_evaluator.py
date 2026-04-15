import torch.distributed as dist

import logging

try:
    import faster_coco_eval
    faster_coco_eval.init_as_pycocotools()
except ImportError:
    pass

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

logger = logging.getLogger("seallh.helpers.evaluation.coco_evaluator")


def xyxy2xywh(box):
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def predictions_to_coco(predictions, image_ids):
    scores_pred, labels_pred, bbox_pred = predictions
    batch_size = len(scores_pred)

    out_predictions = []
    for i in range(batch_size):
        scores, labels = scores_pred[i], labels_pred[i]
        boxes = bbox_pred[i]
        img_id = image_ids[i]

        if len(boxes) == 0:
            continue

        for s, l, bb in zip(scores, labels, boxes):

            p = {
                'image_id': img_id,
                'category_id': int(l),
                'bbox': xyxy2xywh(bb.tolist()),
                'score': float(s),
            }
            out_predictions.append(p)

    return out_predictions


def batched_all_gather(data, max_batch_size=10000):
    world_size = dist.get_world_size()
    if world_size == 1:
        return [data]
    
    # add padding to data
    curr_size = len(data)
    gathered_size = [None] * world_size
    dist.all_gather_object(gathered_size, curr_size)
    max_size = max(gathered_size)
    padding = [None for _ in range(max_size - curr_size)]
    data = [*data, *padding]
    
    # batched aggregation
    gathered_data = []
    for i in range(0, len(data), max_batch_size):
        batch = data[i : i + max_batch_size]
        gathered_batch = [None] * world_size
        dist.all_gather_object(gathered_batch, batch)
        for x in gathered_batch:
            if x is not None:
                x = [y for y in x if y is not None]
                gathered_data.extend(x)
    return gathered_data

    
class COCOEvaluator:

    _bbox_metrics = ['AP', 'AP@0.5', 'AP@0.75', 'AP@s', 'AP@m', 'AP@l', 'AR@1', 'AR@10', 'AR@100', 'AR@s', 'AR@m', 'AR@l']

    def __init__(self):
        self.category_names = None
        self.reset()

    @staticmethod
    def _targets_to_coco(targets, img_ids, img_shapes):
        """
        Converts targets to COCO format annotations
        """
        bboxes = targets['bboxes']

        images = []
        annotations = []

        bs = len(img_ids)

        for i in range(bs):
            bbs = bboxes[bboxes[:, 0] == i]
            img_id = img_ids[i]

            images.append({
                'id': img_id,
                'width': img_shapes[i][1],
                'height': img_shapes[i][0],
                'file_name': '',
                'license': 0,
                'flickr_url': '',
                'coco_url': '',
                'date_captured': '',
            })

            if len(bbs) == 0:
                continue

            for bb in bbs:
                cat_id = int(bb[1])
                bb = xyxy2xywh(bb[3:].tolist())
                _, _, w, h = bb
                annotations.append({
                    'id': -1,   # will be overridden during tmp json annotation creation while compute()
                    'image_id': img_id,
                    'category_id': cat_id,
                    'bbox': bb,
                    'area': w * h,
                    'iscrowd': 0,  # default coco tag (always constant for our tasks)
                })

        return images, annotations

    def add_batch(self, predictions, targets, img_shapes):
        if targets is None:
            raise ValueError('Targets must not be None')

        batch_size = len(img_shapes)
        image_ids = list(range(self._next_img_id, self._next_img_id + batch_size))
        self._next_img_id += batch_size

        coco_predictions = predictions_to_coco(predictions, image_ids)
        
        self.predictions.extend(coco_predictions)

        coco_images, coco_annotations = self._targets_to_coco(targets, image_ids, img_shapes)

        self.images.extend(coco_images)
        self.annotations.extend(coco_annotations)

    def _reduce_state(self):
        dist.barrier()
        
        # gather predictions across the workers
        predictions = batched_all_gather(self.predictions)
        self.predictions = predictions
        
        # gather annotations and images across the workers when no annotation file was provided
        gathered_images = batched_all_gather(self.images)
        self.images = gathered_images
        
        # gather images across the workers
        gathered_annotations = batched_all_gather(self.annotations)
        self.annotations = gathered_annotations

        dist.barrier()
    
    def _create_coco_gt(self):
        # init coco categories
        assert self.category_names is not None, "category_names must be set before creating coco_gt"
        categories = []
        for i in range(len(self.category_names)):
            cat = {
                'id': i,
                'name': self.category_names[i],
                'supercategory': None,
                'skeleton': []
            }
            categories.append(cat)

        # sort images and annotaions and assign correct ids for annotations
        self.images.sort(key=lambda x: x['id'])
        self.annotations.sort(key=lambda x: x['image_id'])
        for idx, ann in enumerate(self.annotations):
            ann['id'] = idx

        coco_annotations = {
            'categories': categories,
            'images': self.images,
            'annotations': self.annotations,
        }
        
        # init cocoGt annotations instance
        coco_gt = COCO()
        coco_gt.dataset = coco_annotations
        coco_gt.createIndex()
        return coco_gt
        
    def compute(self):
        # reduce annotations across the workers in DDP mode
        if dist.is_available() and dist.is_initialized():
            self._reduce_state()
            
        # use in-memory annotations
        coco_gt = self._create_coco_gt()

        # init cocoDt predictions instance
        if len(self.predictions) == 0:
            logger.warning('No predictions found')
            return {}
        coco_dt = coco_gt.loadRes(self.predictions)

        # compute COCO metrics
        self.metrics = self._compute(coco_gt, coco_dt)

        return self.metrics

    def _compute(self, cocoGt, cocoDt):
        metrics = {}
        annType = ['bbox']
        metrics_keys = [self._bbox_metrics]

        for ann, keys in zip(annType, metrics_keys):
            cocoEval = COCOeval(cocoGt, cocoDt, ann)
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

            metrics[ann] = dict(zip(keys, cocoEval.stats))

        result = metrics['bbox']
        return result

    def reset(self):
        self.metrics = {}
        self.predictions = []
        self.images = []
        self.annotations = []
        self._next_img_id = 0

    def set_category_names(self, category_names):
        assert isinstance(category_names, list), "category_names should be a list of category names"
        assert len(category_names) > 0, "category_names should not be empty"
        assert all(isinstance(name, str) for name in category_names), "All category names should be strings"
        self.category_names = category_names
