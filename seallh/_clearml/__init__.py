import logging
import os
import socket
from urllib.parse import urlparse

logger = logging.getLogger("seallh._clearml")

def __collect_remote_clearml_info() -> None:

    def __resolve_ip(url: str) -> str:
        try:
            host = urlparse(url).hostname or url
            return socket.gethostbyname(host)
        except Exception:
            return "unresolved"

    try:
        from clearml.backend_api.session.client import APIClient
    except Exception as e:
        raise RuntimeError("ClearML SDK import failed. Install/configure ClearML.") from e

    try:
        client = APIClient()
        _ = client.projects.get_all()
    except Exception as e:
        raise RuntimeError(
            "ClearML server not reachable or credentials missing. "
            "Run `clearml-init` or set CLEARML_* env vars."
        ) from e
    
    logger.info(f"ClearML server is reachable on ip {__resolve_ip(client.auth.session.host)}")


# auto-check at import unless explicitly skipped
# Prefer new SEALLH_SKIP_CLEARML_CHECK env var, fall back to legacy SEALLH_SKIP_CLEARML_CHECK
skip_val = os.getenv("SEALLH_SKIP_CLEARML_CHECK", os.getenv("SEALLH_SKIP_CLEARML_CHECK", "0"))
if skip_val not in ("1", "true", "True", "TRUE", "ON"):
    __collect_remote_clearml_info()

import seallh._clearml.dataset
