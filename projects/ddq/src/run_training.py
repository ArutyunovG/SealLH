from projects.ddq.src.model import DDQFCN
from projects.ddq.src.neck import RepFPN
from projects.ddq.src.head import DDQFCNHead

from projects.ddq.src.setup_dataloader import setup_dataloader

from seallh.experiment.utils import import_class

import timm
import torch

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

    pass
