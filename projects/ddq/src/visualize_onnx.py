import logging
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger("seallh.projects.ddq.visualize_onnx")


def _resize_fit_and_pad_center(img_bgr, target_h, target_w):
    orig_h, orig_w = img_bgr.shape[:2]
    scale = min(target_w / orig_w, target_h / orig_h)
    resized_w = int(scale * orig_w)
    resized_h = int(scale * orig_h)

    img_resized = cv2.resize(img_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_w = target_w - resized_w
    pad_h = target_h - resized_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2

    img_padded = cv2.copyMakeBorder(
        img_resized,
        pad_top, pad_h - pad_top,
        pad_left, pad_w - pad_left,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return img_padded, scale, pad_left, pad_top


def _to_rgb_chw(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    x = np.transpose(img_rgb, (2, 0, 1))
    return np.expand_dims(x, 0)


def _nms_xyxy(boxes, scores, iou_thresh):
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = np.argsort(-scores)
    keep = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou <= iou_thresh]

    return np.asarray(keep, dtype=np.int64)


def _postprocess(scores_out, bboxes_out, input_hw, conf_thresh=0.3,
                 nms_iou=0.7, nms_pre=1000, topk=300):
    scores_1d = scores_out[0, :, 0].astype(np.float32)
    boxes_xyxy = bboxes_out[0, :, :].astype(np.float32)

    in_h, in_w = input_hw
    strides = (8, 16, 32)
    level_counts = [int((in_h // s) * (in_w // s)) for s in strides]

    selected = []
    offset = 0
    for cnt in level_counts:
        sl = slice(offset, offset + cnt)
        lvl_scores = scores_1d[sl]
        k = min(nms_pre, cnt)
        if k > 0:
            idx = np.argpartition(-lvl_scores, kth=k - 1)[:k]
            idx = idx[np.argsort(-lvl_scores[idx])]
            selected.append(idx + offset)
        offset += cnt

    if not selected:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    cand_idx = np.concatenate(selected)
    cand_scores = scores_1d[cand_idx]
    cand_boxes = boxes_xyxy[cand_idx]

    keep = _nms_xyxy(cand_boxes, cand_scores, nms_iou)
    kept_idx = cand_idx[keep]

    kept_scores = scores_1d[kept_idx]
    order = np.argsort(-kept_scores)
    kept_idx = kept_idx[order][:topk]

    mask = scores_1d[kept_idx] >= conf_thresh
    kept_idx = kept_idx[mask]

    return boxes_xyxy[kept_idx], scores_1d[kept_idx]


def _map_boxes_back(boxes_xyxy, scale, pad_left, pad_top, orig_h, orig_w):
    if boxes_xyxy.size == 0:
        return boxes_xyxy
    boxes = boxes_xyxy.copy().astype(np.float32)
    boxes[:, 0::2] -= pad_left
    boxes[:, 1::2] -= pad_top
    boxes /= scale
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, orig_w - 1)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, orig_h - 1)
    return boxes


def visualize_exported_model(onnx_path, cfg, datasets_dict):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_info = sess.get_inputs()[0]
    in_name = in_info.name
    out_names = [o.name for o in sess.get_outputs()]

    in_shape = in_info.shape
    input_hw = (int(in_shape[2]), int(in_shape[3]))

    # Get a validation image from the first dataset
    primary_key = list(datasets_dict.keys())[0]
    val_ds = datasets_dict[primary_key].get("validation")
    if val_ds is None:
        logger.warning("No validation dataset for visualization")
        return None

    # Access the underlying raw dataset (unwrap Map transforms)
    raw_ds = val_ds
    while hasattr(raw_ds, "dataset"):
        raw_ds = raw_ds.dataset

    idx = 0
    sample = raw_ds[idx]

    # COCODataset returns {"image": np.array (RGB HWC), ...}
    img_rgb = sample["image"]
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    orig_h, orig_w = img_bgr.shape[:2]

    # Preprocess
    img_padded, scale, pad_left, pad_top = _resize_fit_and_pad_center(
        img_bgr, input_hw[0], input_hw[1]
    )
    x = _to_rgb_chw(img_padded)

    # Run inference
    outputs = sess.run(out_names, {in_name: x.astype(np.float32)})
    name_to_out = dict(zip(out_names, outputs))
    scores_out = name_to_out.get("scores", outputs[0])
    bboxes_out = name_to_out.get("bboxes", outputs[1])

    boxes, scores = _postprocess(scores_out, bboxes_out, input_hw,
                                   conf_thresh=cfg.inference.conf_thresh,
                                   nms_iou=cfg.inference.nms_iou,
                                   nms_pre=cfg.inference.nms_pre,
                                   topk=cfg.inference.topk)
    boxes = _map_boxes_back(boxes, scale, pad_left, pad_top, orig_h, orig_w)

    # Draw detections
    img_vis = img_bgr.copy()
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_vis, f"{score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Save
    export_dir = str(Path(onnx_path).parent)
    viz_path = str(Path(export_dir) / "ddq_visualization.png")
    cv2.imwrite(viz_path, img_vis)
    logger.info(f"Visualization saved: {viz_path} ({len(boxes)} detections)")

    return viz_path
