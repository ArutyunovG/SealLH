from pathlib import Path
import subprocess
import sys
import logging
from typing import Optional


logger = logging.getLogger("projects.nanodet.run_training")


def _find_nanodet_repo() -> Optional[Path]:
    # Look for external_repos/nanodet relative to this file and upward parents
    p = Path(__file__).resolve()
    for depth in range(1, 7):
        candidate = p.parents[depth - 1].joinpath("external_repos", "nanodet")
        if candidate.exists():
            return candidate.resolve()

    # Fallback: cwd/external_repos/nanodet
    cand2 = Path.cwd() / "external_repos" / "nanodet"
    if cand2.exists():
        return cand2.resolve()

    return None


def run_training(cfg, created_datasets, clearml_task, pl_loggers=None):
    """Wrapper that launches external Nanodet training script.

    Respects the following config keys (optional):
      - cfg.run_training_config: path to nanodet YAML config file
      - cfg.run_training_args: list of extra CLI args to append

    If no config is provided, the wrapper will try a sensible default
    under the nanodet repo: `config/nanodet-plus-m_320.yml`.
    """
    # locate repo
    repo = _find_nanodet_repo()
    if repo is None:
        raise RuntimeError("Could not locate external_repos/nanodet repository")

    # determine config path
    cfg_path = None
    if hasattr(cfg, "run_training_config") and cfg.run_training_config:
        cfg_path = str(cfg.run_training_config)
    else:
        candidate = repo / "config" / "nanodet-plus-m_320.yml"
        if candidate.exists():
            cfg_path = str(candidate)

    if not cfg_path:
        raise RuntimeError("No nanodet config provided and no default config found in nanodet repo")

    train_script = repo / "tools" / "train.py"
    if not train_script.exists():
        raise RuntimeError(f"Nanodet train script not found at {train_script}")

    cmd = [sys.executable, str(train_script), cfg_path]

    # extra args
    extra = None
    if hasattr(cfg, "run_training_args") and cfg.run_training_args:
        extra = list(cfg.run_training_args)
    # allow passing list via plain attr or OmegaConf ListConfig
    if extra:
        cmd.extend([str(x) for x in extra])

    logger.info(f"Launching nanodet training: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
