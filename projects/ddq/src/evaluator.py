import sys
import faster_coco_eval
import torch.distributed as dist

from copy import deepcopy

import logging

# Replace pycocotools with faster_coco_eval
faster_coco_eval.init_as_pycocotools()
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

logger = logging.getLogger("seallh.projects.ddq.src.evaluator")

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def scale_coords(bboxes, original_shape, target_shape, pad_mode='center'):
    """
    Rescale bounding boxes from `original_shape` to `target_shape`.

    Parameters:
    - bboxes: Tensor of shape (N,4) in xyxy format.
    - original_shape: (height, width) of the source image.
    - target_shape: (height, width) of the target image.
    - pad_mode: 'center' (default) assumes padding is centered; 'top_left' assumes
      padding was added at the bottom/right (i.e., zero pad offsets).
    """
    h0, w0 = original_shape
    h1, w1 = target_shape

    # scale: target -> original
    scale_x = w0 / w1
    scale_y = h0 / h1

    scale = min(scale_x, scale_y)

    if pad_mode == 'top_left':
        pad_x, pad_y = 0.0, 0.0
    else:
        pad_x, pad_y = (w0 - w1 * scale) / 2.0, (h0 - h1 * scale) / 2.0

    scale_x, scale_y = scale, scale

    # box: original -> target, box_t = (box_o - pad_o) / scale_t2o
    bboxes[:, [0, 2]] -= pad_x
    bboxes[:, [1, 3]] -= pad_y
    bboxes[:, [0, 2]] /= scale_x
    bboxes[:, [1, 3]] /= scale_y

    clip_coords(bboxes, target_shape)

    return bboxes


def xyxy2xywh(box):
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def predictions_to_coco(predictions, image_ids, img1_shapes, img0_shapes, pad_mode='top_left'):
    scores_pred, labels_pred, bbox_pred = predictions
    batch_size = len(image_ids)

    out_predictions = []
    for i in range(batch_size):
        scores, labels = scores_pred[i], labels_pred[i]
        boxes = bbox_pred[i]
        img_id = image_ids[i]
        img0_shape, img1_shape = img0_shapes[i], img1_shapes[i]

        if len(boxes) == 0:
            continue

        boxes = scale_coords(boxes, original_shape=img1_shape, target_shape=img0_shape, pad_mode=pad_mode)

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

    
class Evaluator:

    _disable_coco_output = True
    _bbox_metrics = ['AP', 'AP@0.5', 'AP@0.75', 'AP@s', 'AP@m', 'AP@l', 'AR@1', 'AR@10', 'AR@100', 'AR@s', 'AR@m', 'AR@l']
    _kp_metrics = ['AP', 'AP@0.5', 'AP@0.75', 'AP@m', 'AP@l', 'AR', 'AR@50', 'AR@75', 'AR@m', 'AR@l']

    def __init__(self):
        self.category_names = None
        self.reset()

    @staticmethod
    def _targets_to_coco(targets, img_ids, img0_shapes):
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
                'width': img0_shapes[i][1],
                'height': img0_shapes[i][0],
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
                x, y, w, h = bb
                annotations.append({
                    'id': -1,   # will be overridden during tmp json annotation creation while compute()
                    'image_id': img_id,
                    'category_id': cat_id,
                    'bbox': bb,
                    'area': w * h,
                    'iscrowd': 0,  # default coco tag (always constant for our tasks)
                })

        return images, annotations

    def add_batch(self, predictions, targets, meta_data):
        image_ids = meta_data['img_id']
        img0_shapes = meta_data['img0_shape']
        img1_shapes = meta_data['img1_shape']

        if targets is None:
            raise ValueError('Targets must not be None in None ann_path mode')

        predictions = deepcopy(predictions)
        
        coco_predictions = predictions_to_coco(predictions, image_ids, img1_shapes, img0_shapes)
        
        self.predictions.extend(coco_predictions)

        coco_images, coco_annotations = self._targets_to_coco(targets, image_ids, img0_shapes)

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
        # disable default coco_eval script printing to stdout
        if self._disable_coco_output:
            sys.stdout = None
        
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
        
        # enable back stdout printing
        if self._disable_coco_output:
            sys.stdout = sys.__stdout__

        return self.metrics

    def _compute(self, cocoGt, cocoDt):
        metrics = {}
        annType = ['bbox']
        metrics_keys = [self._bbox_metrics]

        for ann, keys in zip(annType, metrics_keys):
            cocoEval = COCOeval(cocoGt, cocoDt, ann)

            # cocoEval.params.kpt_oks_sigmas = np.array(self.kp_oks_sigmas)
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

    def set_category_names(self, category_names):
        assert isinstance(category_names, list), "category_names should be a list of category names"
        assert len(category_names) > 0, "category_names should not be empty"
        assert all(isinstance(name, str) for name in category_names), "All category names should be strings"
        self.category_names = category_names
