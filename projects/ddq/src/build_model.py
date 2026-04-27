import logging

import timm

from projects.ddq.src.model import DDQFCN
from projects.ddq.src.neck import RepFPN
from projects.ddq.src.head import DDQFCNHead

logger = logging.getLogger("seallh.projects.ddq.build_model")


def build_model(cfg):
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
    model = DDQFCN(backbone=backbone, neck=neck, bbox_head=head,
                   norm_mean=list(cfg.norm_mean), norm_std=list(cfg.norm_std))
    logger.info(f"DDQ model created: {model.__class__.__name__}")

    return model
