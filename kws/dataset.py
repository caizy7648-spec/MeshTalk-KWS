"""dataset.py - training / validation windows of T frames, features ONLY from
kws.features.logmel (the firmware front-end).

    D = Data(C)                                   # loads data/cache (memory-mapped)
    ds = KwWindows(D, C, "train", n=48000)        # torch Dataset: (feat [T,40], label)
    ds.epoch = 3                                  # different (but reproducible) windows per epoch
    sp = SpkWindows(D, C, "train", n=40000)       # (feat, speaker index), real voices only

Every window's random choices come from numpy Generator([seed, tag, epoch, i]),
so the data is identical whatever the number of DataLoader workers.
A window is built like the device hears it:
  clip (speed / pitch) -> level -> shouting -> placed in the window (keyword
  fully inside) + context speech -> reverb -> noise at SNR + mic noise floor
  -> high-pass -> int16 (clipping) -> logmel.
"""
import os

import numpy as np
import pandas as pd
import torch

from . import augment as aug
from .audio import to_float, to_int16
from .cfg import stable_hash
from .features import FRAME, HOP, logmel


def n_samples(T):
    return FRAME + (T - 1) * HOP


def worker_init(_):
    """DataLoader workers: one thread each (numpy BLAS / torch would otherwise oversubscribe the CPU)"""
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    torch.set_num_threads(1)


class Data:
    """clips + noise banks from data/cache, grouped for sampling"""

    def __init__(self, C):
        cache = C["paths"]["cache"]
        self.C = C
        # clips.npy (02_prepare.py) and, when present, hardneg*.npy (12_mine.py): one table, store = which file
        frames, self.audios = [], []
        # r4 (Phase C): + hardneg_r4 (13_mine_r4.py), own / vocal / chat (06_prepare_r4.py)
        for k, name in enumerate(("clips", "hardneg", "hardneg_r3", "hardneg_r4", "own", "vocal", "chat")):
            if not os.path.exists(os.path.join(cache, name + ".npy")):
                continue
            frames.append(pd.read_csv(os.path.join(cache, name + ".csv"), keep_default_na=False).assign(store=len(self.audios)))
            self.audios.append(np.load(os.path.join(cache, name + ".npy"), mmap_mode="r"))
        self.df = pd.concat(frames, ignore_index=True)
        self.noise = {}
        for s in ("train", "test"):
            self.noise[s] = (pd.read_csv(os.path.join(cache, "noise_%s.csv" % s)),
                             np.load(os.path.join(cache, "noise_%s.npy" % s), mmap_mode="r"))
        df = self.df
        self.off = df.offset.to_numpy()
        self.len = df.length.to_numpy()
        self.ks = df.kw_start.to_numpy()
        self.ke = df.kw_end.to_numpy()
        self.store = df.store.to_numpy()
        self.pools = {}
        for (lab, grp, split), g in df.groupby(["label", "group", "split"]):
            self.pools[(lab, grp, split)] = g.index.to_numpy()
        # "help me" joined from the same LibriSpeech speaker's "help" and "me"
        self.join = {}
        for split in ("train", "val", "test"):
            h = df[(df.group == "libri_help") & (df.split == split)].groupby("speaker").groups
            m = df[(df.group == "libri_me") & (df.split == split)].groupby("speaker").groups
            sp = sorted(set(h) & set(m))
            self.join[split] = [(np.asarray(h[s]), np.asarray(m[s])) for s in sp]

    def pool(self, label, groups, split):
        out = [self.pools.get((label, g, split), np.zeros(0, int)) for g in groups]
        return np.concatenate(out) if out else np.zeros(0, int)

    def clip(self, i):
        o = self.off[i]
        return to_float(np.array(self.audios[self.store[i]][o:o + self.len[i]])), int(self.ks[i]), int(self.ke[i])

    def noise_seg(self, rng, split, n):
        df, a = self.noise[split]
        r = df.iloc[rng.integers(len(df))]
        if r.length >= n:
            o = r.offset + rng.integers(r.length - n + 1)
            return to_float(np.array(a[o:o + n]))
        x = to_float(np.array(a[r.offset:r.offset + r.length]))
        return np.tile(x, n // max(1, len(x)) + 1)[:n]

    def joined_help_me(self, rng, split, gap_ms):
        """help + gap + me of the same speaker -> (audio, span start, span end)"""
        hs, ms = self.join[split][rng.integers(len(self.join[split]))]
        h, hs0, he0 = self.clip(hs[rng.integers(len(hs))])
        m, ms0, me0 = self.clip(ms[rng.integers(len(ms))])
        t = int(0.02 * aug.RATE)
        h = h[:min(len(h), he0 + t)]
        m = m[max(0, ms0 - t):]
        m = m * (aug.rms(h[hs0:he0]) / aug.rms(m[t:t + (me0 - ms0)]))
        gap = np.zeros(int(rng.uniform(*gap_ms) / 1000 * aug.RATE), np.float32)
        x = np.concatenate([h, gap, m]).astype(np.float32)
        return x, hs0, len(h) + len(gap) + (me0 - ms0) + t


class Builder:
    """turns a clip into one window (int16) with the augmentation of train.yaml"""

    def __init__(self, D, C, split, A=None):
        self.D, self.C, self.split = D, C, split
        self.A = A if A is not None else C["augment"]
        self.T = C["window"]["T"]
        self.N = n_samples(self.T)
        self.margin = C["window"]["margin_frames"] * HOP
        self.p_edge = C["window"].get("p_right_edge", 0.0)
        self.edge = C["window"].get("right_edge_s", 0.0)
        self.ctx_pool = D.pool("_other", ["libri_chunk"], split)

    def place(self, rng, x, s, e, keyword):
        """offset of the clip start inside the window, or None (keyword does not fit)"""
        N, m, L = self.N, self.margin, len(x)
        lo, hi = m - s, N - m - e
        if keyword:
            if lo > hi:
                return None
            if rng.random() < self.p_edge:
                # keyword end close to the window end: the windows the device runs on in a quiet room
                # (the -55 dBFS gate only lets the CNN run while the word is in its last 240 ms)
                return int(max(lo, hi - int(rng.uniform(0, self.edge) * aug.RATE)))
            return int(rng.integers(lo, hi + 1))
        if L >= N:
            return int(rng.integers(N - L, 1))
        return int(rng.integers(min(lo, 0), max(hi, N - L) + 1))

    def paste(self, y, x, o):
        a, b = max(0, o), min(len(y), o + len(x))
        if b > a:
            y[a:b] += x[a - o:b - o]

    def window(self, rng, x, s, e, keyword):
        """int16 window or None"""
        A, N = self.A, self.N
        if keyword or e - s > 0:
            if rng.random() < A.get("p_speed", 0):
                f = rng.uniform(*A["speed"])
                x = aug.speed(x, f)
                s, e = int(s / f), min(len(x), int(e / f))
            if rng.random() < A.get("p_pitch", 0):
                x = aug.pitch(x, rng.uniform(*A["pitch_semitones"]))
        o = self.place(rng, x, s, e, keyword)
        if o is None:
            return None
        ref = x[s:e] if e - s > 160 else x
        x = aug.set_rms(x, A["ref_level_dbfs"] + rng.uniform(*A["gain_db"]), ref)
        if rng.random() < A.get("p_shout", 0):
            x = aug.shout(x, rng.uniform(*A["shout_gain_db"]))
        y = np.zeros(N, np.float32)
        self.paste(y, x, o)
        if keyword and rng.random() < A.get("p_context", 0) and len(self.ctx_pool):
            gap = int(0.05 * aug.RATE)
            lvl = aug.rms(x[s:e]) * aug.db(rng.uniform(*A["context_level_db"]))
            c, _, _ = self.D.clip(self.ctx_pool[rng.integers(len(self.ctx_pool))])
            c = c * (lvl / aug.rms(c))
            left, right = o + s - gap, N - (o + e + gap)
            if left > 1600 and rng.random() < 0.5:
                y[:left] += c[-left:] if len(c) >= left else np.pad(c, (left - len(c), 0))
            elif right > 1600:
                y[N - right:] += c[:right] if len(c) >= right else np.pad(c, (0, right - len(c)))
        sp = y[max(0, o + s):max(1, min(N, o + e))]
        if rng.random() < A.get("p_reverb", 0):
            y = aug.reverb(y, rng, rng.uniform(*A["rt60_s"]), A["rir_len_s"], rng.uniform(*A["reverb_wet"]))
        if rng.random() < A.get("p_noise", 0):
            y = aug.add_noise(y, self.D.noise_seg(rng, "train" if self.split == "train" else "test", N),
                              rng.uniform(*A["snr_db"]), sp if aug.rms(sp) > 1e-6 else y)
        y = aug.noise_floor(y, rng, rng.uniform(*A["noise_floor_dbfs"]))
        if rng.random() < A.get("p_highpass", 0):
            y = aug.highpass(y, rng.uniform(*A["highpass_hz"]))
        return to_int16(y)

    def noise_window(self, rng, silence=False):
        A = self.A
        y = np.zeros(self.N, np.float32)
        if not silence:
            y = aug.set_rms(self.D.noise_seg(rng, "train" if self.split == "train" else "test", self.N),
                            A["ref_level_dbfs"] + rng.uniform(*A["gain_db"]))
        y = aug.noise_floor(y, rng, rng.uniform(*A["noise_floor_dbfs"]))
        if rng.random() < A.get("p_highpass", 0):
            y = aug.highpass(y, rng.uniform(*A["highpass_hz"]))
        return to_int16(y)


class KwWindows(torch.utils.data.Dataset):
    """keyword head: (logmel [T,40] float32, class index)"""

    def __init__(self, D, C, split, n, tag="kw"):
        self.D, self.C, self.split, self.n, self.tag = D, C, split, n, tag
        self.labels = C["labels"]
        K = C["train_kw"]
        self.B = Builder(D, C, split)
        # r4: the owner's clips (device microphone) get the lighter augment_own
        self.B_own = Builder(D, C, split, dict(C["augment"], **C.get("augment_own", {})))
        self.own_share = K.get("own_share", {})
        self.class_p = np.array([K["class_mix"][l] for l in self.labels], float)
        self.class_p /= self.class_p.sum()
        self.other = [(g, p) for g, p in K["other_mix"].items() if p > 0]
        self.epoch = 0
        P = {}
        tts_split = split
        P["help", "real"] = D.pool("help", ["mswc_help", "libri_help"], split)
        P["help", "tts"] = D.pool("help", ["tts"], tts_split)
        P["help me", "real"] = D.pool("help me", ["libri_help_me"], split)
        P["help me", "tts"] = D.pool("help me", ["tts"], tts_split)
        P["sos", "tts"] = D.pool("sos", ["tts"], tts_split)
        for lab in ("help", "help me", "sos"):
            P[lab, "own"] = D.pool(lab, ["own"], split)
        for g, _ in self.other:
            if g == "libri_words":
                P[g] = D.pool("_other", ["libri_words", "libri_me"], split)
            elif g not in ("noise_only", "silence"):
                P[g] = D.pool("_other", [g], split)
        # big _other groups: a fixed subset (hash order) - each epoch only uses a few thousand of them, and
        # the whole cache does not fit in memory (random reads from disk would slow training ~10x)
        cap = K.get("max_other_per_group")
        if cap:
            for g in list(P):
                if isinstance(g, str) and len(P[g]) > cap:
                    P[g] = np.array(sorted(P[g], key=lambda i: stable_hash("cap:%d" % i))[:cap])
        # a group without clips in this split (hard negatives exist for train only) is left out
        self.other = [(g, p) for g, p in self.other if g in ("noise_only", "silence") or len(P[g])]
        self.other_p = np.array([p for _, p in self.other], float)
        self.other_p /= self.other_p.sum()
        self.P = P

    def __len__(self):
        return self.n

    def rng(self, i):
        return np.random.default_rng([self.C["seed"], stable_hash(self.tag + self.split) & 0xFFFFFFFF, self.epoch, i])

    def pick(self, rng, k):
        """(audio, s, e) of one clip of class k (or "noise_only" / "silence")"""
        return self.pick2(rng, k)[0]

    def pick2(self, rng, k):
        """(clip, builder): the owner's clips use B_own"""
        lab = self.labels[k]
        own = self.P.get((lab, "own"), [])
        if len(own) and rng.random() < self.own_share.get(lab, 0.0):
            return self.D.clip(own[rng.integers(len(own))]), self.B_own
        return self.pick1(rng, k), self.B

    def pick1(self, rng, k):
        K = self.C["train_kw"]
        lab = self.labels[k]
        if lab.startswith("_"):
            g = self.other[rng.choice(len(self.other), p=self.other_p)][0]
            if g in ("noise_only", "silence"):
                return g
            p = self.P[g]
            return self.D.clip(p[rng.integers(len(p))])
        real = rng.random() < K["real_vs_tts"].get(lab, 0.0)
        if lab == "help me" and real:
            src = K["help_me_sources"]
            if rng.random() < src["joined_same_speaker"] / (src["joined_same_speaker"] + src["libri_real"]) and self.D.join[self.split]:
                return self.D.joined_help_me(rng, self.split, K["join_gap_ms"])
        p = self.P[lab, "real" if real and len(self.P.get((lab, "real"), [])) else "tts"]
        return self.D.clip(p[rng.integers(len(p))])

    def make(self, rng, k):
        kw = not self.labels[k].startswith("_")
        for _ in range(20):
            c, B = self.pick2(rng, k)
            if isinstance(c, str):
                return self.B.noise_window(rng, silence=(c == "silence"))
            w = B.window(rng, *c, keyword=kw)
            if w is not None:
                return w
        raise RuntimeError("no clip fits the window for class %s" % self.labels[k])

    def __getitem__(self, i):
        rng = self.rng(i)
        k = int(rng.choice(len(self.labels), p=self.class_p))
        return torch.from_numpy(logmel(self.make(rng, k))), k


class FixedKwWindows(KwWindows):
    """validation: the same number of windows per class, the same every time"""

    def __init__(self, D, C, split, per_class, tag="kwval"):
        super().__init__(D, C, split, per_class * len(C["labels"]), tag)
        self.per_class = per_class

    def __getitem__(self, i):
        rng = self.rng(i)
        k = i // self.per_class
        return torch.from_numpy(logmel(self.make(rng, k))), k


class SpkWindows(torch.utils.data.Dataset):
    """speaker head: real voices only (spk_use), (logmel, speaker index)"""

    def __init__(self, D, C, split, n, A, tag="spk"):
        self.D, self.C, self.split, self.n, self.tag = D, C, split, n, tag
        df = D.df[(D.df.spk_use == True) & (D.df.split == split)]  # noqa: E712
        self.speakers = sorted(df.speaker.unique())
        self.by_spk = [g.index.to_numpy() for _, g in df.groupby("speaker")]
        self.B = Builder(D, C, split, A)
        self.epoch = 0

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng([self.C["seed"], stable_hash(self.tag + self.split) & 0xFFFFFFFF, self.epoch, i])
        while True:
            k = int(rng.integers(len(self.speakers)))
            idx = self.by_spk[k]
            w = self.B.window(rng, *self.D.clip(idx[rng.integers(len(idx))]), keyword=True)
            if w is not None:
                return torch.from_numpy(logmel(w)), k
