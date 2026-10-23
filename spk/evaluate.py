"""evaluate.py - speaker model, device protocol (CONTRACT_SPK.md).

Enrolment: N takes of one speaker (quiet room, normal voice); take i asks
for class enrol_classes[i % 3]; the take's vector = spk embedding of the
window centred on the word; template of a class = sum of its take vectors.
Test: another take of the same speaker (target) or of other speakers
(impostor), clean / in noise (SNR 5-20 dB) / shouted; cosine with the
template of the take's class (help -> help, help me -> help me, other words
-> help, like a keyword hit of class help); class without takes -> all takes.

Real data has few repeated keywords per speaker, so most speakers enrol with
other single words and are tested with their "help" / "help me" (when they
have them) and other words: a text-independent, pessimistic estimate.
"""
import numpy as np
import pandas as pd
import torch

from kws import augment as aug
from kws.audio import to_int16
from kws.cfg import rng_for, stable_hash
from kws.dataset import n_samples
from kws.features import logmel

from .data import speakers


def window(rng, x, s, e, frames, level, floor, noise=None, snr=None, shout_db=None):
    """int16 window of `frames` frames with the word in the middle"""
    n = n_samples(frames)
    x = aug.set_rms(x, level, x[s:e] if e - s > 160 else x)
    if shout_db is not None:
        x = aug.shout(x, shout_db)
    o = n // 2 - (s + e) // 2
    y = np.zeros(n, np.float32)
    a, b = max(0, o), min(n, o + len(x))
    y[a:b] = x[a - o:b - o]
    if noise is not None:
        y = aug.add_noise(y, noise, snr, y[max(0, o + s):max(1, min(n, o + e))])
    return to_int16(aug.noise_floor(y, rng, floor))


def embed(net, wins, bs=512):
    out = []
    with torch.no_grad():
        for a in range(0, len(wins), bs):
            out.append(net(torch.from_numpy(np.stack([logmel(w) for w in wins[a:a + bs]]))).numpy())
    return np.concatenate(out) if out else np.zeros((0, net.cfg["dim"]), np.float32)


def classes(C, P):
    lab = C["labels"]
    return [lab.index(c) for c in P["eval"]["enrol_classes"]]


def enrol(net, D, C, P, split, n_takes, n_speakers, conds=("clean", "noisy", "shout"), tag="spkeval"):
    """templates {speaker: [ncls, dim] sums}, test vectors DataFrame(speaker, cls, cond, keyword, emb)"""
    E = P["eval"]
    F = net.cfg["frames"]
    kc = classes(C, P)
    sp = speakers(D, split, exclude=P["data"]["exclude_groups_eval"])
    names = sorted((s for s, idx in sp.items() if len(idx) >= n_takes + 3), key=lambda s: stable_hash("spkB:" + s))[:n_speakers]
    lab = C["labels"]
    wins, meta = [], []
    for s in names:
        rng = rng_for(C, tag, split, s, n_takes)
        idx = sorted(sp[s], key=lambda i: stable_hash("clipB:%d" % i))
        kw = [i for i in idx if D.df.label[i] in ("help", "help me")]
        other = [i for i in idx if i not in kw]
        en = other[:n_takes]
        te = (kw + other[n_takes:])[:E["test_takes_max"]]
        for j, i in enumerate(en):
            x, a, b = D.clip(i)
            wins.append(window(rng, x, a, b, F, rng.uniform(*E["enrol_level_dbfs"]), E["noise_floor_dbfs"]))
            meta.append(("enrol", s, kc[j % len(kc)], "clean", False))
        for i in te:
            x, a, b = D.clip(i)
            k = lab.index(D.df.label[i]) if D.df.label[i] in ("help", "help me") else lab.index("help")
            for cond in conds:
                kw_ = {}
                if cond == "noisy":
                    kw_ = dict(noise=D.noise_seg(rng, "test", n_samples(F)), snr=rng.uniform(*E["test_snr_db"]))
                if cond == "shout":
                    kw_ = dict(shout_db=rng.uniform(*E["shout_gain_db"]))
                wins.append(window(rng, x, a, b, F, rng.uniform(*E["test_level_dbfs"]), E["noise_floor_dbfs"], **kw_))
                meta.append(("test", s, k, cond, D.df.label[i] in ("help", "help me")))
    emb = embed(net, wins)
    tmpl = {s: np.zeros((len(lab), net.cfg["dim"])) for s in names}
    tests = []
    for (kind, s, k, cond, iskw), e in zip(meta, emb):
        if kind == "enrol":
            tmpl[s][k] += e
        else:
            tests.append(dict(speaker=s, cls=k, cond=cond, keyword=iskw, emb=e))
    return tmpl, pd.DataFrame(tests)


def template(t, k):
    """the class template if it has takes, else the sum of all (kws_voice_similarity)"""
    return t[k] if np.any(t[k]) else t.sum(axis=0)


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def scores(tmpl, tests, n_imp, seed_key):
    names = sorted(tmpl)
    rows = []
    for n, t in enumerate(tests.itertuples()):
        rng = np.random.default_rng([stable_hash(seed_key) & 0xFFFFFFFF, n])
        rows.append(dict(cond=t.cond, keyword=t.keyword, target=True, score=cos(template(tmpl[t.speaker], t.cls), t.emb)))
        imp = [names[j] for j in rng.permutation(len(names)) if names[j] != t.speaker][:n_imp]
        for s in imp:
            rows.append(dict(cond=t.cond, keyword=t.keyword, target=False, score=cos(template(tmpl[s], t.cls), t.emb)))
    return pd.DataFrame(rows)


def eer(target, impostor):
    t, i = np.sort(target), np.sort(impostor)
    th = np.unique(np.concatenate([t, i]))
    frr = np.searchsorted(t, th, side="left") / len(t)
    far = 1.0 - np.searchsorted(i, th, side="left") / len(i)
    j = int(np.argmin(np.abs(frr - far)))
    return float((frr[j] + far[j]) / 2), float(th[j])


def rates(target, impostor, thr):
    return float(np.mean(np.asarray(target) < thr)), float(np.mean(np.asarray(impostor) >= thr))
