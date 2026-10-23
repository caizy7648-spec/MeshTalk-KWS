"""evaluate.py - measurements exactly as the device decides (kws.detect.Detector).

The CNN outputs are computed in batches for the frames where the Detector
runs (every 12 frames once T frames are there) and handed to Detector through
run_fn, so Detector itself (window gate, 3-run average, thr + cls_off,
refractory time, hit counting) is used unchanged; detect(check=True) verifies
that the cached window is the one the Detector would pass.

    st = Stream(net, pcm_int16, T)              # features, loudness, cached CNN outputs
    events = st.hits(keyword_mask, thr, hits=3, hits_ms=20000, cls_off=[...])

Used by scripts/20_eval.py (system-level evaluation) and scripts/12_mine.py
(hard negatives).
"""
import numpy as np
import torch

from .detect import EV_HIT, STEP, Detector
from .features import FRAME, HOP, logmel

RATE = 16000


# ------------------------------------------------------------------ helpers
def frame_dbfs(pcm):
    """loudness of the 320 new samples of each feature frame (kws_dbfs)"""
    x = np.asarray(pcm, np.float64)
    n = 0 if len(x) < FRAME else 1 + (len(x) - FRAME) // HOP
    idx = (FRAME - HOP) + HOP * np.arange(n)
    seg = x[idx[:, None] + np.arange(HOP)[None, :]]
    p = np.mean(seg * seg, axis=1) / 32768.0 ** 2
    return np.where(p > 0, 10.0 * np.log10(np.maximum(p, 1e-30)), -120.0)


def run_frames(T, n):
    """frame indices at which Detector calls run_fn"""
    return np.arange(T + STEP - 2, n, STEP)


def batch_run(net, feats, frames, T, bs=2048):
    """probs [len(frames), ncls], emb [len(frames), d] for the windows ending at frames"""
    P, E = [], []
    sw = np.lib.stride_tricks.sliding_window_view(feats, (T, feats.shape[1]))[:, 0]   # [n-T+1, T, 40]
    with torch.no_grad():
        for a in range(0, len(frames), bs):
            w = torch.from_numpy(np.ascontiguousarray(sw[frames[a:a + bs] - T + 1]))
            lg, emb = net(w)
            P.append(torch.softmax(lg, 1).numpy())
            E.append(emb.numpy() if emb is not None else np.zeros((len(w), 0), np.float32))
    if not P:
        return np.zeros((0, net.cfg["ncls"]), np.float32), np.zeros((0, 0), np.float32)
    return np.concatenate(P), np.concatenate(E)


def detect(feats, dbfs, probs, frames, T, keyword, thr, hits, hits_ms, check=False, cls_off=None):
    """runs kws.detect.Detector over a stream; returns list of (t_ms, events, class)"""
    cache = {int(f): k for k, f in enumerate(frames)}
    cur = [0]

    def run_fn(window):
        k = cache[cur[0]]
        if check:
            assert np.array_equal(window, feats[cur[0] - T + 1:cur[0] + 1]), "cached window differs"
        return probs[k], None

    det = Detector(run_fn, T, keyword, thr, cls_off=cls_off, hits=hits, hits_ms=hits_ms)
    out = []
    for i in range(len(feats)):
        cur[0] = i
        ev = det.frame(feats[i], dbfs[i], i * 20)
        if ev & EV_HIT:
            out.append((i * 20, ev, det.hit_cls))
    return out


class Stream:
    """one audio stream: features, loudness and the cached CNN outputs"""

    def __init__(self, net, pcm, T):
        self.pcm = pcm
        self.feats = logmel(pcm)
        self.dbfs = frame_dbfs(pcm)[:len(self.feats)]
        self.frames = run_frames(T, len(self.feats))
        self.probs, self.emb = batch_run(net, self.feats, self.frames, T)
        self.T = T

    def hits(self, keyword, thr, hits=3, hits_ms=20000, check=False, cls_off=None):
        return detect(self.feats, self.dbfs, self.probs, self.frames, self.T, keyword, thr, hits, hits_ms, check, cls_off)
