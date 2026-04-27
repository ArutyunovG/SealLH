"""Shared evaluation helpers used by DDQ training (validation) and testing."""

from seallh.helpers.sysinfo import get_allocated_gpu_mem_gb
from seallh.helpers.tqdm_loader_bar import tqdm_loader_bar

import torch
from torchmetrics import MeanMetric, MultioutputWrapper

import logging

logger = logging.getLogger("seallh.projects.ddq.evaluate")


def _run_eval_batch(model, image, targets, loss, device, evaluator, loss_avg, channels_last):
    """Run a single evaluation batch: forward, loss, postprocess, evaluator.add_batch.

    Returns the current loss_dict for progress-bar display.
    """
    image = image.to(device)
    if channels_last:
        image = image.contiguous(memory_format=torch.channels_last)

    raw_output = model(image)

    loss_dict = loss(raw_output, targets, device)
    items = torch.stack(list(loss_dict.values())[:loss.num_val_losses])
    loss_avg.update(items.unsqueeze(0))

    outputs = model.postprocess(raw_output)

    scores_list, labels_list, boxes_list = [], [], []
    for bboxes, labels in outputs:
        if bboxes is None or bboxes.numel() == 0:
            scores_list.append([])
            labels_list.append([])
            boxes_list.append([])
            continue
        scores_list.append(bboxes[:, -1].cpu())
        labels_list.append(labels.cpu())
        boxes_list.append(bboxes[:, :4].cpu())

    img_shapes = [list(image.shape[-2:])] * len(targets)

    bboxes_rows = []
    for i, t in enumerate(targets):
        for j, box in enumerate(t['bboxes']):
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            bboxes_rows.append([i, int(t['labels'][j].item()), 0, x1, y1, x2, y2])

    if len(bboxes_rows) == 0:
        targets_for_eval = {'bboxes': torch.empty((0, 7), dtype=torch.float32)}
    else:
        targets_for_eval = {'bboxes': torch.tensor(bboxes_rows, dtype=torch.float32)}

    predictions = (scores_list, labels_list, boxes_list)
    evaluator.add_batch(predictions, targets_for_eval, img_shapes)

    return loss_dict, image.shape[2:]


def evaluate_on_dataloaders(model, dataloaders, loss, evaluator, device,
                            clearml_task, cfg, mode, scalar_prefix, iteration,
                            epoch, max_epochs):
    """Run evaluation over all provided dataloaders and report losses/metrics.

    Args:
        model: eval-mode model with .postprocess().
        dataloaders: dict of {dataset_name: DataLoader}.
        loss: loss module (with .num_val_losses).
        evaluator: COCO-like evaluator (reset/add_batch/compute).
        device: torch device.
        clearml_task: ClearML task for scalar reporting.
        cfg: experiment config.
        mode: progress-bar mode string (e.g. 'val', 'test').
        scalar_prefix: prefix for ClearML scalar titles (e.g. 'val', 'test').
        iteration: iteration value used when reporting scalars.
        epoch, max_epochs: used by the progress bar.
    """
    channels_last = cfg.get("channels_last", False)

    with torch.no_grad():
        for ds_name, dataloader in dataloaders.items():
            logger.info(f"Evaluating on dataset '{ds_name}' ({mode})")

            evaluator.reset()
            loss_avg = MultioutputWrapper(base_metric=MeanMetric(),
                                          num_outputs=loss.num_val_losses).to(device)

            bar = tqdm_loader_bar(dataloader, mode=mode,
                                  epoch=epoch, max_epochs=max_epochs)

            loss_dict = None
            for image, targets in bar:
                loss_dict, img_size = _run_eval_batch(
                    model=model, image=image, targets=targets,
                    loss=loss, device=device, evaluator=evaluator,
                    loss_avg=loss_avg, channels_last=channels_last,
                )

                avg_losses = {name: f'{value:.4f}'
                              for name, value in zip(list(loss_dict.keys())[:loss.num_val_losses],
                                                     loss_avg.compute())}
                mem = get_allocated_gpu_mem_gb()
                bar.set_postfix({**avg_losses, 'gpu_mem': f'{mem:.2f}Gb', 'img_size': img_size})

            metrics = evaluator.compute()
            assert 'bbox' in metrics, f"No bbox keys found in metrics for dataset '{ds_name}'"
            logger.info(f"{mode.capitalize()} metrics for '{ds_name}': {metrics}")

            scalar_title = f"{scalar_prefix}/{ds_name}"
            avg_values = loss_avg.compute()
            if loss_dict is not None:
                for name, value in zip(list(loss_dict.keys())[:loss.num_val_losses], avg_values):
                    clearml_task.report_scalar(scalar_title, name, iteration=iteration, value=value.item())
            for name, value in metrics['bbox'].items():
                clearml_task.report_scalar(scalar_title, name, iteration=iteration, value=float(value))

            per_class = metrics.get('bbox_per_class', {})
            for class_name, class_metrics in per_class.items():
                per_class_title = f"{scalar_title}/per_class/{class_name}"
                for metric_name, metric_value in class_metrics.items():
                    clearml_task.report_scalar(per_class_title, metric_name,
                                               iteration=iteration, value=float(metric_value))
