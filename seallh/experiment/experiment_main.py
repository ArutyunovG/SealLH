from omegaconf import DictConfig, OmegaConf
import inspect
import importlib
import logging

from seallh.experiment.setup_logging import setup_logging
from seallh.experiment.run_training import run_training
from seallh.experiment.run_testing import run_testing
from seallh.experiment.run_export import run_export
from seallh.experiment.external_repository_setuper import setup_repositories
from seallh.experiment.prepare_clearml_datasets import prepare_clearml_datasets
from seallh._clearml.task import ClearMLTask


def _run_phase(func_cfg_key, default_func, cfg, created_datasets, clearml_task, pl_loggers):
    """Invoke a phase, optionally dispatching to a user-configured function by import path."""
    logger = logging.getLogger("seallh.experiment")
    func_path = cfg.get(func_cfg_key, None)
    if not func_path:
        default_func(cfg, created_datasets, clearml_task, pl_loggers)
        return

    assert isinstance(func_path, str), f"{func_cfg_key} must be a string with function import path"
    try:
        module_path, func_name = func_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        func = getattr(module, func_name)
        logger.info(f"Calling configured {func_cfg_key}: {func_path}")
        num_params = len(inspect.signature(func).parameters)
        all_func_params_num = 4
        if num_params == all_func_params_num:
            func(cfg, created_datasets, clearml_task, pl_loggers)
        elif num_params == all_func_params_num - 1:
            func(cfg, created_datasets, clearml_task)
        else:
            raise RuntimeError(
                f"{func.__name__} must accept {all_func_params_num - 1} or {all_func_params_num} "
                f"parameters, got {num_params}"
            )
    except Exception as e:
        logger.exception(f"Failed to call {func_cfg_key} '{func_path}'; Error: {e}")
        raise


def experiment_main(cfg: DictConfig) -> None:
    """Experiment main, receives config from main.py"""
    logger = logging.getLogger("seallh.experiment")
    logger.info("Entered experiment_main")

    clearml_task = ClearMLTask(clearml_config=cfg.clearml)

    pl_loggers = setup_logging(cfg)

    logger.info('Running experiment with config:')
    logger.info('============================================================')
    try:
        # Try to resolve and print the config
        resolved_config = OmegaConf.to_yaml(cfg, resolve=True)
        logger.info(resolved_config)
    except Exception as e:
        # Print unresolved config first, then reraise the exception
        logger.info(OmegaConf.to_yaml(cfg, resolve=False))
        logger.warning(f"Could not fully resolve config for logging: {e}")
        raise
    logger.info('============================================================')

    # Set up external repositories (clone and install dependencies)
    external_repositories = setup_repositories(cfg)
    if external_repositories:
        logger.info(f"External repositories ready: {list(external_repositories.keys())}")

    # Prepare datasets from ClearML dataset configurations
    created_datasets = prepare_clearml_datasets(cfg)

    # Check which phases are enabled and orchestrate them
    phases = cfg.get("phases", {"training": False, "testing": False, "export": False})
    
    logger.info("Experiment phases configuration:")
    logger.info(f"  Training: {'+' if phases['training'] else '-'}")
    logger.info(f"  Testing: { '+' if phases['testing']  else '-'}")
    logger.info(f"  Export: {  '+' if phases['export']   else '-'}")
    
    phase_specs = [
        ("training", "run_training_func", run_training),
        ("testing",  "run_testing_func",  run_testing),
        ("export",   "run_export_func",   run_export),
    ]

    for phase_name, func_cfg_key, default_func in phase_specs:
        if not phases[phase_name]:
            logger.info(f"{phase_name.capitalize()} phase skipped")
            continue
        logger.info(f"=== {phase_name.upper()} PHASE ===")
        _run_phase(func_cfg_key, default_func,
                   cfg, created_datasets, clearml_task, pl_loggers)
        logger.info(f"{phase_name.capitalize()} phase completed!")
    
    logger.info("All experiment phases completed!")
