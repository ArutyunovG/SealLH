import logging

import torch

from projects.ddq.src.build_model import build_model

logger = logging.getLogger("seallh.projects.ddq.model_loader")


def load_model_for_export(checkpoint_path, cfg):
    model = build_model(cfg)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("ema_state_dict", ckpt.get("model_state_dict"))
    model.load_state_dict(state_dict)
    model.eval()

    logger.info(f"DDQ model loaded from {checkpoint_path} (key: {'ema_state_dict' if 'ema_state_dict' in ckpt else 'model_state_dict'})")
    return model
