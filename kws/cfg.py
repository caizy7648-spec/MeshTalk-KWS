"""cfg.py - reads configs/train.yaml (the one place for every parameter) and
gives deterministic random generators.

    from kws.cfg import load, rng_for, ROOT
    C = load()                       # dict, paths made absolute
    rng = rng_for(C, "tts", 17)      # numpy Generator, same stream on every run
"""
import hashlib
import os
import random

import numpy as np
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT = os.path.join(ROOT, "configs", "train.yaml")


def load(path=None):
    with open(path or os.environ.get("KWS_CONFIG", DEFAULT)) as f:
        C = yaml.safe_load(f)
    C["paths"] = {k: os.path.join(ROOT, v) for k, v in C["paths"].items()}
    for v in C["paths"].values():
        os.makedirs(v, exist_ok=True)
    return C


def stable_hash(s):
    """int from a string, the same on every machine / Python run"""
    return int.from_bytes(hashlib.md5(str(s).encode()).digest()[:8], "little")


def rng_for(C, *keys):
    """numpy Generator seeded by the global seed and any keys (str / int)"""
    return np.random.default_rng([C["seed"]] + [stable_hash(k) & 0xFFFFFFFF if isinstance(k, str) else int(k) for k in keys])


def speaker_split(C, speaker):
    """'train' / 'val' / 'test' from the speaker id (hash), see split: in train.yaml"""
    h = stable_hash("split:" + str(speaker)) % 100
    if h < C["split"]["test_pct"]:
        return "test"
    if h < C["split"]["test_pct"] + C["split"]["val_pct"]:
        return "val"
    return "train"


def seed_all(C, extra=0):
    import torch
    random.seed(C["seed"] + extra)
    np.random.seed((C["seed"] + extra) % 2**32)
    torch.manual_seed(C["seed"] + extra)
    torch.use_deterministic_algorithms(True)
