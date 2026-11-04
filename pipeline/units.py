"""Config loader: load_config() reads <repo>/example_config.yaml, fills the ${MODELS} / ${TOOLS} / ${PKG} placeholders from
pipeline/ctx.py and points training.base_full at this book's starting weights. The sha256 returned is that of the file as
shipped (placeholders unexpanded).
"""
import hashlib
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline import ctx  # noqa: E402

CONFIG_PATH = os.path.join(ROOT, "example_config.yaml")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _expand(v):
    if isinstance(v, str):
        return v.replace("${MODELS}", ctx.models_dir()).replace("${TOOLS}", ctx.tools_dir()).replace("${PKG}", ctx.PKG)
    if isinstance(v, list):
        return [_expand(x) for x in v]
    if isinstance(v, dict):
        return {k: _expand(x) for k, x in v.items()}
    return v


def load_config():
    raw = open(CONFIG_PATH, "rb").read()
    cfg = _expand(yaml.safe_load(raw))
    if os.environ.get("CBI_BOOK_OUT"):
        cfg["training"]["base_full"] = ctx.base_model()
    return cfg, sha256_bytes(raw)
