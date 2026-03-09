from pathlib import Path
import subprocess
import sys
import logging


logger = logging.getLogger("projects.ddq.run_training")


def run_training(cfg, created_datasets, clearml_task):

    """
    DDQ-specific training function
    """
    k = 1
    pass
