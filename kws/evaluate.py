"""evaluate.py - measurements exactly as the device decides (kws.detect.Detector).

Keyword head (Step 2):
  * every test keyword is put into its own stream (context_s of background
    before and after, level and SNR from train.yaml eval:), the stream goes
    frame by frame through kws.detect.Detector -> hit or miss;
  * running speech (LibriSpeech dev / test) and noise (MUSAN / ESC-50 test
    parts) go through Detector as long streams -> false hits per hour and
    false SOS (hits within hits_ms) per hour.
  The CNN outputs are computed in batches for the frames where the Detector
  runs (every 12 frames once T frames are there) and handed to Detector
  through run_fn, so Detector itself (gate, smoothing, refractory time, hit
  counting) is used unchanged. check_cache() verifies that the cached window
  is the one the Detector would pass.

Speaker head (Step 3):
  * device enrolment: 5 takes of 2 s, classes help, help me, sos, help, help me;
    for each take the window with the highest probability of its class
    (40 ms steps, like kws_best_window) gives the speaker vector; template of
    a class = sum of its vectors;
  * a test clip's vector (best window for its class) is compared by cosine with
    the template of that class (own speaker = target, other speakers = impostor).
"""
import numpy as np
import torch

from . import augment as aug
from .audio import to_int16
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


def detect(feats, dbfs, probs, frames, T, keyword, thr, hits, hits_ms, check=False):
    """runs kws.detect.Detector over a stream; returns list of (t_ms, events, class)"""
    cache = {int(f): k for k, f in enumerate(frames)}
    cur = [0]

    def run_fn(window):
        k = cache[cur[0]]
        if check:
            assert np.array_equal(window, feats[cur[0] - T + 1:cur[0] + 1]), "cached window differs"
        return probs[k], None

    det = Detector(run_fn, T, keyword, thr, hits=hits, hits_ms=hits_ms)
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

    def hits(self, keyword, thr, hits=3, hits_ms=20000, check=False):
        return detect(self.feats, self.dbfs, self.probs, self.frames, self.T, keyword, thr, hits, hits_ms, check)


# ------------------------------------------------------------------ keyword streams
def positive_stream(rng, x, s, e, E, noise_fn, noisy):
    """clip in the middle of context_s of background; returns (int16, clip start, clip end) in samples"""
    ctx = int(E["context_s"] * RATE)
    lvl = rng.uniform(*E["test_level_dbfs"])
    x = aug.set_rms(x, lvl, x[s:e] if e - s > 160 else x)
    y = np.concatenate([np.zeros(ctx, np.float32), x, np.zeros(ctx, np.float32)])
    if noisy:
        y = aug.add_noise(y, noise_fn(rng, len(y)), rng.uniform(*E["test_snr_db"]), x[s:e])
    y = aug.noise_floor(y, rng, E["noise_floor_dbfs"])
    return to_int16(y), ctx + s, ctx + e


def hit_in(events, a_ms, b_ms):
    """first hit inside [a_ms, b_ms] -> class or -1"""
    for t, ev, k in events:
        if a_ms <= t <= b_ms:
            return k
    return -1


def fa_count(events, guard, hits_needed, hits_ms):
    """false hits (not near a real keyword) and false SOS from a list of hits"""
    fh, sos, times = 0, 0, []
    for t, ev, k in events:
        if any(a <= t <= b for a, b in guard):
            continue
        fh += 1
        times = [u for u in times if t - u < hits_ms] + [t]      # same rule as Detector / kws_detect.c
        if len(times) >= hits_needed:
            sos += 1
            times = []
    return fh, sos


# ------------------------------------------------------------------ speaker
def best_window(net, feats, k, T):
    """kws_best_window: every window at 40 ms steps, the one where class k is most likely -> (p, emb)"""
    starts = np.arange(0, len(feats) - T + 1, 2)
    probs, emb = batch_run(net, feats, starts + T - 1, T)
    j = int(np.argmax(probs[:, k]))
    return float(probs[j, k]), emb[j]


def take(rng, x, s, e, seconds, level, floor_dbfs, noise=None, snr=None, shout_db=None):
    """a 2 s enrolment / test take with the word in the middle"""
    n = int(seconds * RATE)
    x = aug.set_rms(x, level, x[s:e] if e - s > 160 else x)
    if shout_db is not None:
        x = aug.shout(x, shout_db)
    mid = (s + e) // 2
    o = n // 2 - mid
    y = np.zeros(n, np.float32)
    a, b = max(0, o), min(n, o + len(x))
    y[a:b] = x[a - o:b - o]
    if noise is not None:
        seg = y[max(0, o + s):max(1, min(n, o + e))]
        y = aug.add_noise(y, noise, snr, seg)
    y = aug.noise_floor(y, rng, floor_dbfs)
    return to_int16(y)


def eer(target, impostor):
    """equal error rate and its threshold"""
    t, i = np.sort(target), np.sort(impostor)
    th = np.unique(np.concatenate([t, i]))
    frr = np.searchsorted(t, th, side="left") / len(t)
    far = 1.0 - np.searchsorted(i, th, side="left") / len(i)
    j = int(np.argmin(np.abs(frr - far)))
    return float((frr[j] + far[j]) / 2), float(th[j])


def rates(target, impostor, thr):
    """false reject, false accept at a cosine threshold"""
    return float(np.mean(np.asarray(target) < thr)), float(np.mean(np.asarray(impostor) >= thr))
