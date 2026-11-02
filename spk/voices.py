"""voices.py - phase B2 data: clips.npy (MSWC, LibriSpeech) + spk_extra.npy (VCTK, People's Speech),
genders, the distillation bank (teacher vectors) and the two training streams.

    V = Voices(C)                               # stores, genders, noise banks (kws.dataset.Data)
    aam = AamWindows(V, C, P, n)                # (logmel, speaker index): labelled voices, gender-balanced
    dis = DistillWindows(V, C, P, n)            # (logmel, teacher vector): every bank segment, gender-balanced
Both pass the audio through spk.device.chain (finetune.device) - the teacher
vector belongs to the clean audio.
"""
import os

import numpy as np
import pandas as pd
import torch

from kws.audio import to_float
from kws.cfg import stable_hash
from kws.dataset import Data
from kws.features import logmel

from . import device


class Voices:
    def __init__(self, C):
        self.C = C
        self.D = Data(C)                                        # clips + noise banks
        cache = C["paths"]["cache"]
        ex = pd.read_csv(os.path.join(cache, "spk_extra.csv"), keep_default_na=False)
        ex["group"] = ex.source
        ex["label"] = "_other"
        ex["text"] = ex.utt
        self.stores = {"clips": (self.D.df, None), "spk_extra": (ex, np.load(os.path.join(cache, "spk_extra.npy"), mmap_mode="r"))}
        g = pd.read_csv(os.path.join(cache, "spk_gender.csv"))
        self.gender = dict(zip(g.speaker, g.gender))
        self.gender_source = dict(zip(g.speaker, g.source))

    def noise_seg(self, rng, split, n):
        return self.D.noise_seg(rng, split, n)

    def clip_of(self, item):
        name, i = item
        if name == "clips":
            return self.D.clip(i)
        df, a = self.stores[name]
        r = df.iloc[i]
        return to_float(np.array(a[r.offset:r.offset + r.length])), int(r.kw_start), int(r.kw_end)


def balanced(rng, items):
    """an item, female and male equally likely"""
    g = "FEMALE" if rng.random() < 0.5 else "MALE"
    pool = items[g] if items[g] else items["MALE" if g == "FEMALE" else "FEMALE"]
    return pool[rng.integers(len(pool))]


class AamWindows(torch.utils.data.Dataset):
    """labelled real voices: train-split speakers of clips (spk_use, max_clips_per_speaker) + VCTK train speakers"""

    def __init__(self, V, C, P, n, tag="aamB2"):
        self.V, self.P, self.n, self.tag = V, P, n, tag
        F = P["finetune"]
        cap = F["train"]["max_clips_per_speaker"]
        by = {}
        for name, (df, _) in V.stores.items():
            g = df[(df.split == "train") & (df.speaker != "")]
            if name == "clips":
                g = g[g.spk_use.astype(str) == "True"]
            for s, idx in g.groupby("speaker").groups.items():
                by.setdefault(s, []).extend((name, i) for i in sorted(idx, key=lambda i: stable_hash("capB:%d" % i))[:cap])
        self.names = sorted(by)
        self.items = [by[s] for s in self.names]
        self.by_gender = {"MALE": [], "FEMALE": []}
        for k, s in enumerate(self.names):
            self.by_gender[V.gender.get(s, "MALE")].append(k)
        self.Q = F["device"]
        self.epoch = 0

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng([self.P["seed"], stable_hash(self.tag) & 0xFFFFFFFF, self.epoch, i])
        k = balanced(rng, self.by_gender)
        it = self.items[k][rng.integers(len(self.items[k]))]
        x, s, e = self.V.clip_of(it)
        return torch.from_numpy(logmel(device.chain(rng, x, s, e, self.V, self.Q, "train"))), k


class DistillWindows(torch.utils.data.Dataset):
    """bank segments (labelled or not) with the teacher vector of their clean audio;
    with finetune.train.loss_consistency > 0 also a second view as recorded at enrolment"""

    def __init__(self, V, C, P, n, tag="disB2"):
        self.V, self.P, self.n, self.tag = V, P, n, tag
        cache = C["paths"]["cache"]
        b = pd.read_csv(os.path.join(cache, "distill_bank.csv"), keep_default_na=False)
        t = np.load(os.path.join(cache, "distill_teacher.npy"))
        keep = (b.split == "train").to_numpy()
        self.bank = b[keep].reset_index(drop=True)
        t = t[keep]
        self.teacher = (t / np.linalg.norm(t, axis=1, keepdims=True)).astype(np.float32)
        self.by_gender = {"MALE": [], "FEMALE": []}
        for k, g in enumerate(self.bank.gender):
            self.by_gender[g if g in self.by_gender else "MALE"].append(k)
        self.T = P["finetune"]["teacher"]
        self.Q = P["finetune"]["device"]
        self.Qc = P["finetune"]["eval"]["conditions"]["enrol"]
        self.two_views = P["finetune"]["train"].get("loss_consistency", 0) > 0
        self.epoch = 0

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng([self.P["seed"], stable_hash(self.tag) & 0xFFFFFFFF, self.epoch, i])
        k = balanced(rng, self.by_gender)
        r = self.bank.iloc[k]
        x, s, e = self.V.clip_of((r.store, int(r.row)))
        n = int(self.T["max_s"] * 16000)
        if len(x) > n:                                   # the part the teacher heard
            c = (s + e) // 2
            o = min(max(0, c - n // 2), len(x) - n)
            x, s, e = x[o:o + n], max(0, s - o), min(n, e - o)
        dev = logmel(device.chain(rng, x, s, e, self.V, self.Q, "train"))
        if not self.two_views:
            return torch.from_numpy(dev), torch.from_numpy(self.teacher[k])
        # second view: the same audio as recorded at enrolment (close, quiet room) - for the consistency loss
        cln = logmel(device.chain(rng, x, s, e, self.V, self.Qc, "train"))
        return torch.from_numpy(dev), torch.from_numpy(cln), torch.from_numpy(self.teacher[k])
