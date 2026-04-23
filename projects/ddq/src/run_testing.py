from projects.ddq.src.evaluate import evaluate_on_dataloaders
from projects.ddq.src.model_loader import load_model_for_export
from projects.ddq.src.setup_dataloader import setup_dataloaders_per_dataset

from seallh.experiment.utils import import_class

import torch

import logging
import os

logger = logging.getLogger("seallh.projects.ddq.run_testing")


def run_testing(cfg, created_datasets, clearml_task):
    """DDQ-specific testing function."""

    checkpoint_path = cfg.paths.checkpoint_path
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found at: {checkpoint_path}")
        logger.info("Testing phase skipped - no trained model checkpoint available")
        return

    logger.info(f"Loading DDQ model from checkpoint: {checkpoint_path}")
    model = load_model_for_export(checkpoint_path, cfg)

    test_dataloaders = setup_dataloaders_per_dataset(cfg=cfg,
                                                     created_datasets=created_datasets,
                                                     split='test')

    loss_cfg = cfg.loss
    logger.info(f"Setting up loss function from config: {loss_cfg}")
    loss_cls = import_class(loss_cfg["class"])
    loss = loss_cls(**loss_cfg.args)

    evaluator_cfg = cfg.evaluator
    logger.info(f"Setting up evaluator from config: {evaluator_cfg}")
    evaluator_cls = import_class(evaluator_cfg["class"])
    evaluator = evaluator_cls(**evaluator_cfg.args)
    if hasattr(evaluator, 'set_category_names'):
        evaluator.set_category_names(list(created_datasets.values())[0]['test'].categories)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    model.to(device)
    model.eval()

    evaluate_on_dataloaders(
        model=model,
        dataloaders=test_dataloaders,
        loss=loss,
        evaluator=evaluator,
        device=device,
        clearml_task=clearml_task,
        cfg=cfg,
        mode='test',
        scalar_prefix='test',
        iteration=0,
        epoch=1,
        max_epochs=1,
    )
