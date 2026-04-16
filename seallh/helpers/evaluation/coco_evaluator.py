import logging

try:
    import faster_coco_eval
    faster_coco_eval.init_as_pycocotools()
except ImportError:
    pass

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

logger = logging.getLogger("seallh.helpers.evaluation.coco_evaluator")


class COCOEvaluator:

    def __init__(self, 
                 bbox_metrics=None):

        self._bbox_metrics = bbox_metrics if bbox_metrics is not None else \
                             ['AP', 'AP@0.5', 'AP@0.75', 'AP@s', 'AP@m', 'AP@l', 'AR@1', 'AR@10', 'AR@100', 'AR@s', 'AR@m', 'AR@l']
        self.category_names = None
        self.reset()


    def compute(self):
        coco_gt = self._create_coco_gt()

        if len(self.predictions) == 0:
            logger.warning('No predictions found')
            return {}

        coco_dt = coco_gt.loadRes(self.predictions)
        self.metrics = self._compute(coco_gt, coco_dt)

        return self.metrics


    def add_batch(self, predictions, targets, img_shapes):
        if targets is None:
            raise ValueError('Targets must not be None')

        batch_size = len(img_shapes)
        image_ids = list(range(self._next_img_id, self._next_img_id + batch_size))
        self._next_img_id += batch_size

        coco_predictions = self._predictions_to_coco(predictions, image_ids)
        
        self.predictions.extend(coco_predictions)

        coco_images, coco_annotations = self._targets_to_coco(targets, image_ids, img_shapes)

        self.images.extend(coco_images)
        self.annotations.extend(coco_annotations)


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


    def _xyxy2xywh(self, box):
        x1, y1, x2, y2 = box
        return [x1, y1, x2 - x1, y2 - y1]


    def _predictions_to_coco(self, predictions, image_ids):
        scores_pred, labels_pred, bbox_pred = predictions
        out = []
        for scores, labels, boxes, img_id in zip(scores_pred, labels_pred, bbox_pred, image_ids):
            for s, l, bb in zip(scores, labels, boxes):
                out.append({
                    'image_id': img_id,
                    'category_id': int(l),
                    'bbox': self._xyxy2xywh(bb.tolist()),
                    'score': float(s),
                })
        return out


    def _targets_to_coco(self, targets, img_ids, img_shapes):

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
                bb = self._xyxy2xywh(bb[3:].tolist())
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


    def _create_coco_gt(self):

        assert self.category_names is not None, "category_names must be set before creating coco_gt"
        categories = [{'id': i, 'name': name, 'supercategory': None} for i, name in enumerate(self.category_names)]

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
        
        coco_gt = COCO()
        coco_gt.dataset = coco_annotations
        coco_gt.createIndex()
        return coco_gt
        
    def _compute(self, coco_gt, coco_dt):

        ann_types = {'bbox': self._bbox_metrics}

        metrics = {}
        for ann_type, keys in ann_types.items():
            coco_eval = COCOeval(coco_gt, coco_dt, ann_type)
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            metrics[ann_type] = dict(zip(keys, coco_eval.stats))

        return metrics
