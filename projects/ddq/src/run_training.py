#from projects.ddq.src.model import DDQModel
from projects.ddq.src.neck import RepFPN

import timm
import logging

logger = logging.getLogger("projects.ddq.run_training")


def run_training(cfg, created_datasets, clearml_task):

    """
    DDQ-specific training function
    """
    
    if cfg.model.backbone.type == "timm":
        backbone = timm.create_model(**cfg.model.backbone.args)
    else:
        raise ValueError(f"Unsupported backbone type: {cfg.model.backbone.type}")

    if cfg.model.neck.type == "RepFPN":
        neck = RepFPN(**cfg.model.neck.args)
    else:
        raise ValueError(f"Unsupported neck type: {cfg.model.neck.type}")

    pass
