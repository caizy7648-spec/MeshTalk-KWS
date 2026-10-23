"""data.py - phase B data: configs/spk.yaml + the clips of configs/train.yaml.

    from spk.data import load_cfg, SpkTrain
    C, P = load_cfg()                    # C = train.yaml (paths, window, ...), P = spk.yaml
    ds = SpkTrain(Data(C), C, P, "train", n)   # (logmel [frames, 40], speaker index), real voices only

A training window: one clip of the speaker, augmented like the device hears
it (kws.dataset.Builder with spk.yaml augment:, no speed / pitch), placed in
the 81-frame window; running-speech chunks longer than the window are cut.
"""
import os

import numpy as np
import torch
import yaml

from kws.cfg import ROOT, load, stable_hash
from kws.dataset import Builder
from kws.features import logmel


def load_cfg(path=None):
    C = load()
    with open(path or os.environ.get("SPK_CONFIG", os.path.join(ROOT, "configs", "spk.yaml"))) as f:
        P = yaml.safe_load(f)
    assert P["net"]["frames"] == C["window"]["T"], "spk frames must equal the keyword window T (verify gets that window)"
    return C, P


def speakers(D, split, exclude=()):
    """{speaker: [clip index]} of real voices (spk_use) in the split"""
    df = D.df[(D.df.spk_use == True) & (D.df.split == split) & (~D.df.group.isin(list(exclude)))]  # noqa: E712
    return {s: g.index.to_numpy() for s, g in df.groupby("speaker")}


class SpkTrain(torch.utils.data.Dataset):
    def __init__(self, D, C, P, split, n, tag="spkB"):
        self.D, self.C, self.P, self.n, self.tag = D, C, P, n, tag
        sp = speakers(D, split)
        self.names = sorted(sp)
        self.by = [sp[s] for s in self.names]
        self.B = Builder(D, C, split, P["augment"])
        self.epoch = 0

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng([self.P["seed"], stable_hash(self.tag) & 0xFFFFFFFF, self.epoch, i])
        while True:
            k = int(rng.integers(len(self.names)))
            idx = self.by[k]
            x, s, e = self.D.clip(idx[rng.integers(len(idx))])
            fits = (e - s) <= self.B.N - 2 * self.B.margin
            w = self.B.window(rng, x, s, e, keyword=fits)
            if w is not None:
                return torch.from_numpy(logmel(w)), k
