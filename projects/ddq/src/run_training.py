from projects.ddq.src.ema import is_parallel
from projects.ddq.src.model import DDQFCN
from projects.ddq.src.neck import RepFPN
from projects.ddq.src.head import DDQFCNHead

from projects.ddq.src.setup_dataloader import setup_dataloader

from seallh.experiment.utils import import_class
from seallh.helpers.dataset.map import Map as MapDataset
from seallh.helpers.sysinfo import get_allocated_gpu_mem_gb
from seallh.helpers.tqdm_loader_bar import tqdm_loader_bar

import timm
import torch
from torchmetrics import MeanMetric, MultioutputWrapper

import logging

logger = logging.getLogger("seallh.projects.ddq.run_training")


def run_training(cfg, created_datasets, clearml_task):

    """
    DDQ-specific training function
    """
    
    logger.info("Setting up model components based on config")

    logger.info(f"Building backbone from config: {cfg.model.backbone}")
    if cfg.model.backbone.type == "timm":
        backbone = timm.create_model(**cfg.model.backbone.args)
    else:
        raise ValueError(f"Unsupported backbone type: {cfg.model.backbone.type}")
    logger.info(f"Backbone created: {backbone.__class__.__name__}")

    logger.info(f"Building neck from config: {cfg.model.neck}")
    if cfg.model.neck.type == "RepFPN":
        neck = RepFPN(**cfg.model.neck.args)
    else:
        raise ValueError(f"Unsupported neck type: {cfg.model.neck.type}")
    logger.info(f"Neck created: {neck.__class__.__name__}")

    logger.info(f"Building head from config: {cfg.model.head}")
    if cfg.model.head.type == "DDQFCNHead":
        head = DDQFCNHead(**cfg.model.head.args)
    else:
        raise ValueError(f"Unsupported head type: {cfg.model.head.type}")
    logger.info(f"Head created: {head.__class__.__name__}")


    logger.info("Building DDQ model")
    model = DDQFCN(backbone=backbone,
                   neck=neck,
                   bbox_head=head)
    logger.info(f"DDQ model created: {model.__class__.__name__}")

    transform_cfg = cfg.transform
    logger.info(f"Building transform from config: {transform_cfg}")
    transform_cls = import_class(transform_cfg["class"])
    transform = transform_cls(**transform_cfg.args)
    logger.info("Transform created")

    logger.info("Mapping datasets")
    for dataset in created_datasets:
        pass
    logger.info("Dataset mapping done")

    dataloader = setup_dataloader(cfg=cfg,
                                  created_datasets=created_datasets,
                                  split='train')

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
    logger.info("Evaluator created")

    ema_cfg = cfg.ema
    logger.info(f"Setting up EMA model from config: {ema_cfg}")
    ema_cls = import_class(ema_cfg["class"])
    ema_model = ema_cls(model, **ema_cfg.args)
    logger.info("EMA model created")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.train()

    loss_avg = MultioutputWrapper(base_metric=MeanMetric(),
                                  num_outputs=loss.num_train_losses).to(device)

    for epoch in range(cfg.epochs):

        loader_bar = tqdm_loader_bar(dataloader, 
                                     mode='train',
                                     epoch=epoch,
                                     max_epochs=cfg.epochs)
        
        for batch in loader_bar:

            batch.to(device, channels_last=cfg.channels_last)
            img, targets, meta = batch['image'], batch['bboxes'], batch['meta']

            raw_output = model(img)
            
            loss_dict = loss(raw_output, targets, meta)

            loss = sum(loss_dict.values())
            loss_items = torch.stack(list(loss_dict.values()))

            if torch.any(torch.isnan(loss_items)):
                logger.error('Nan Loss encountered')
                raise ValueError('NaN Loss encountered')
            else:
                loss.backward()
                loss_avg.update(loss_items.unsqueeze(0))

            if cfg.get("clip_grad_max_norm", None):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.clip_grad_max_norm)

            optimizer.step()
            optimizer.zero_grad()

            ema_model.update(model)

            scheduler.step()

            avg_losses = {name: f'{value:.4f}'
                          for name, value in zip(loss_dict.keys(), loss_avg.compute())}
            mem = get_allocated_gpu_mem_gb()
            loader_bar.set_postfix({
                                    **avg_losses, 
                                    'gpu_mem': f'{mem:.2f}Gb',
                                    'img_size': meta['img1_shape']
                                    })

        val_epoch = cfg.get("val_epoch", 0)
        if val_epoch > 0 and epoch % val_epoch == 0:

            logger.info(f"Epoch {epoch}: Running evaluation on validation set")

            eval_model = ema_model.ema
            eval_model.eval()

            # todo

