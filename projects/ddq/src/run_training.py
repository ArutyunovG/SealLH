from projects.ddq.src.build_model import build_model
from projects.ddq.src.setup_dataloader import setup_dataloader

from seallh.experiment.utils import import_class
from seallh.helpers.sysinfo import get_allocated_gpu_mem_gb
from seallh.helpers.tqdm_loader_bar import tqdm_loader_bar

import torch
from torchmetrics import MeanMetric, MultioutputWrapper

import logging
import os

logger = logging.getLogger("seallh.projects.ddq.run_training")


def run_training(cfg, created_datasets, clearml_task):

    """
    DDQ-specific training function
    """
   
    model = build_model(cfg)

    dataloader = setup_dataloader(cfg=cfg,
                                  created_datasets=created_datasets,
                                  split='train')

    val_dataloader = None
    val_epoch = cfg.get("val_epoch", 0)
    if val_epoch > 0:
        val_dataloader = setup_dataloader(cfg=cfg,
                                          created_datasets=created_datasets,
                                          split='val')
    optimizer_cfg = cfg.optimizer
    logger.info(f"Setting up optimizer from config: {optimizer_cfg}")
    optimizer_cls = import_class(optimizer_cfg["class"])
    parameter_groups = model.get_param_groups(wd=optimizer_cfg.args.weight_decay, 
                                              no_decay_bn_filter_bias=optimizer_cfg.args.no_decay_bn_filter_bias)
    del optimizer_cfg.args.no_decay_bn_filter_bias
    optimizer = optimizer_cls(parameter_groups, **optimizer_cfg.args)
    logger.info("Optimizer created")

    scheduler_cfg = cfg.scheduler
    cfg.max_iters = len(dataloader)
    logger.info(f"Setting up learning rate scheduler from config {scheduler_cfg}")
    scheduler_cls = import_class(scheduler_cfg["class"])
    scheduler = scheduler_cls(optimizer, **scheduler_cfg.args)

    if scheduler_cfg.get("warmup", None) and scheduler_cfg.warmup.enabled:
        logger.info("Setting up warmup scheduler")
        warmup_scheduler_cls = import_class(scheduler_cfg.warmup["class"])
        warmup_scheduler = warmup_scheduler_cls(optimizer, **scheduler_cfg.warmup.args)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, scheduler], milestones=[scheduler_cfg.warmup.args.total_iters])
        logger.info(f'Apply warmup to lr scheduler') 
    else:
        logger.info("No warmup scheduler used")

    logger.info("Scheduler created")

    loss_cfg = cfg.loss
    logger.info(f"Setting up loss function from config: {loss_cfg}")

    loss_args = cfg.loss.args
    loss_cls = import_class(loss_cfg["class"])
    loss = loss_cls(**loss_args)
    logger.info("Loss function created")

    evaluator_cfg = cfg.evaluator
    logger.info(f"Setting up evaluator from config: {evaluator_cfg}")
    evaluator_cls = import_class(evaluator_cfg["class"])
    evaluator = evaluator_cls(**evaluator_cfg.args)
    if hasattr(evaluator, 'set_category_names'):
        evaluator.set_category_names(list(created_datasets.values())[0]['train'].categories)
    logger.info("Evaluator created")

    ema_cfg = cfg.ema
    logger.info(f"Setting up EMA model from config: {ema_cfg}")
    ema_cls = import_class(ema_cfg["class"])
    ema_model = ema_cls(model, **ema_cfg.args)
    logger.info("EMA model created")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    model.to(device)
    ema_model.to(device)

    global_step = 0

    for epoch in range(cfg.epochs):

        loader_bar = tqdm_loader_bar(dataloader, 
                                     mode='train',
                                     epoch=epoch + 1,
                                     max_epochs=cfg.epochs)
        
       
        model.train()
        loss_avg = MultioutputWrapper(base_metric=MeanMetric(),
                                      num_outputs=loss.num_train_losses).to(device)


        for image, targets in loader_bar:

            image = image.to(device)
            if cfg.get("channels_last", False):
                image = image.contiguous(memory_format=torch.channels_last)

            raw_output = model(image)

            loss_dict = loss(raw_output, targets, device)

            loss_val = sum(loss_dict.values())
            loss_items = torch.stack(list(loss_dict.values()))

            if torch.any(torch.isnan(loss_items)):
                logger.error('Nan Loss encountered')
                raise ValueError('NaN Loss encountered')
            else:
                loss_val.backward()
                loss_avg.update(loss_items.unsqueeze(0))

            if cfg.get("clip_grad_max_norm", None):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.clip_grad_max_norm)

            optimizer.step()
            optimizer.zero_grad()

            ema_model.update(model)

            scheduler.step()

            if global_step % cfg.loss_log_interval == 0:
                for name, value in loss_dict.items():
                    clearml_task.report_scalar("train", name, iteration=global_step, value=value.item())
                clearml_task.report_scalar("train", "lr", iteration=global_step, value=optimizer.param_groups[0]['lr'])

            global_step += 1

            avg_losses = {name: f'{value:.4f}'
                          for name, value in zip(loss_dict.keys(), loss_avg.compute())}
            mem = get_allocated_gpu_mem_gb()
            loader_bar.set_postfix({
                                    **avg_losses, 
                                    'gpu_mem': f'{mem:.2f}Gb',
                                    'img_size': image.shape[2:]
                                    })

        val_epoch = cfg.get("val_epoch", 0)
        if val_epoch > 0 and epoch % val_epoch == 0:

            logger.info(f"Epoch {epoch}: Running evaluation on validation set")

            assert val_dataloader is not None, "No validation dataloader available, set val_epoch == 0 to disable validation or provide a validation dataloader"

            eval_model = ema_model.ema
            eval_model.eval()
            
            evaluator.reset()
            val_loss_avg = MultioutputWrapper(base_metric=MeanMetric(),
                                                num_outputs=loss.num_val_losses).to(device)

            with torch.no_grad():

                val_bar = tqdm_loader_bar(val_dataloader,
                                          mode='val',
                                          epoch=epoch + 1,
                                          max_epochs=cfg.epochs)

                for image, targets in val_bar:
                    image = image.to(device)

                    raw_output = eval_model(image)

                    loss_dict = loss(raw_output, targets, device)

                    val_items = torch.stack(list(loss_dict.values())[:loss.num_val_losses])
                    val_loss_avg.update(val_items.unsqueeze(0))

                    outputs = eval_model.postprocess(raw_output)

                    scores_list = []
                    labels_list = []
                    boxes_list = []
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
                            row = [i, int(t['labels'][j].item()), 0,
                                    x1, y1, x2, y2]
                            bboxes_rows.append(row)

                    if len(bboxes_rows) == 0:
                        targets_for_eval = {'bboxes': torch.empty((0, 7), dtype=torch.float32)}
                    else:
                        targets_for_eval = {'bboxes': torch.tensor(bboxes_rows, dtype=torch.float32)}

                    predictions = (scores_list, labels_list, boxes_list)

                    evaluator.add_batch(predictions, targets_for_eval, img_shapes)

                    avg_losses = {name: f'{value:.4f}'
                                    for name, value in zip(list(loss_dict.keys())[:loss.num_val_losses], val_loss_avg.compute())}
                    mem = get_allocated_gpu_mem_gb()
                    val_bar.set_postfix({**avg_losses, 'gpu_mem': f'{mem:.2f}Gb', 'img_size': image.shape[2:]})

                metrics = evaluator.compute()
                assert 'bbox' in metrics, f"No bbox keys found in metrics"
                logger.info(f"Validation metrics: {metrics}")

                val_avg_values = val_loss_avg.compute()
                for name, value in zip(list(loss_dict.keys())[:loss.num_val_losses], val_avg_values):
                    clearml_task.report_scalar("val", name, iteration=epoch, value=value.item())
                for name, value in metrics['bbox'].items():
                    clearml_task.report_scalar("val", name, iteration=epoch, value=float(value))

        if cfg.get("checkpoint_epoch_interval", 0) > 0 and ((epoch + 1) % cfg.checkpoint_epoch_interval == 0):
            ckpt_dir = cfg.paths.checkpoint_dir
            os.makedirs(ckpt_dir, exist_ok=True)
            checkpoint_path = os.path.join(ckpt_dir, f'ddq_pytorch.pth')
            logger.info(f"Saving checkpoint to {checkpoint_path}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'ema_state_dict': ema_model.ema.state_dict(),
            }, checkpoint_path)
            logger.info("Checkpoint saved")

            clearml_task.upload_artifact(
                name=f'checkpoint_epoch_{epoch + 1}',
                artifact_object=checkpoint_path,
            )
