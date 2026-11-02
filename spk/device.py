"""device.py - what the device microphone does to a voice (phase B2 augmentation and test conditions).

    y = chain(rng, x, s, e, D, Q, split)     # float clip, spoken part s..e -> int16 window of `frames` frames

Order (as in a room): voice (shouting: drive + soft clipping) -> placed in
the window -> room at a distance (synthetic impulse response: direct sound +
exponentially decaying tail; the direct-to-reverberant ratio falls with
distance: DRR = 20 log10(critical distance / distance)) -> background noise
at an SNR -> INMP441 + case (2nd-order high-pass, low and high shelf EQ)
-> level (speech RMS, dBFS) -> int16 with hard clipping at full scale ->
microphone self-noise. Every value comes from Q (a dict from configs/spk.yaml
finetune.device or finetune.eval.conditions).
"""
import numpy as np
from scipy.signal import fftconvolve, sosfilt

from kws import augment as aug
from kws.audio import to_int16
from kws.dataset import n_samples

RATE = 16000


def u(rng, r):
    """uniform from [lo, hi] (or a fixed number)"""
    return float(rng.uniform(*r)) if isinstance(r, (list, tuple)) else float(r)


def shelf(x, f0, gain_db, high):
    """RBJ cookbook shelving filter (slope 1)"""
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / RATE
    alpha = np.sin(w0) / 2 * np.sqrt(2)
    c, sA = np.cos(w0), 2 * np.sqrt(A) * alpha
    if high:
        b = [A * ((A + 1) + (A - 1) * c + sA), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - sA)]
        a = [(A + 1) - (A - 1) * c + sA, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - sA]
    else:
        b = [A * ((A + 1) - (A - 1) * c + sA), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - sA)]
        a = [(A + 1) + (A - 1) * c + sA, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - sA]
    sos = np.array([[b[0] / a[0], b[1] / a[0], b[2] / a[0], 1.0, a[1] / a[0], a[2] / a[0]]])
    return sosfilt(sos, x).astype(np.float32)


def room(rng, y, rt60, dist, dcrit, length_s):
    """impulse response at `dist` metres: direct sound + tail with DRR = 20 log10(dcrit / dist)"""
    n = int(length_s * RATE)
    t = np.arange(n) / RATE
    tail = rng.standard_normal(n) * np.exp(-6.908 * t / rt60)
    tail[:int(rng.uniform(0.002, 0.008) * RATE)] = 0.0
    tail /= np.sqrt(np.sum(tail ** 2))
    h = tail
    h[0] = 10 ** (20 * np.log10(dcrit / dist) / 20)        # direct amplitude for that DRR (tail energy 1)
    return fftconvolve(y, h)[:len(y)].astype(np.float32)


def place(rng, x, s, e, N, margin, center):
    """offset of the clip in a window of N samples: word in the middle (center) or anywhere inside"""
    if e - s >= N and not center:                      # longer than the window: a random part of the speech
        return int(rng.integers(N - e, -s + 1))
    if center or e - s > N - 2 * margin:
        return N // 2 - (s + e) // 2
    return int(rng.integers(margin - s, N - margin - e + 1))


def chain(rng, x, s, e, D, Q, split, frames=81, center=False):
    N = n_samples(frames)
    margin = 2 * 320
    if rng.random() < Q.get("p_shout", 0):
        x = aug.shout(aug.set_rms(x, -20.0, x[s:e] if e - s > 160 else x), u(rng, Q["shout_gain_db"]))
    o = place(rng, x, s, e, N, margin, center)
    y = np.zeros(N, np.float32)
    a, b = max(0, o), min(N, o + len(x))
    if b > a:
        y[a:b] = x[a - o:b - o]
    sp = slice(max(0, o + s), max(1, min(N, o + e)))
    if rng.random() < Q.get("p_room", 0):
        y = room(rng, y, u(rng, Q["rt60_s"]), u(rng, Q["distance_m"]), u(rng, Q["critical_distance_m"]), Q["rir_len_s"])
    if rng.random() < Q.get("p_noise", 0):
        y = aug.add_noise(y, D.noise_seg(rng, "train" if split == "train" else "test", N), u(rng, Q["snr_db"]), y[sp])
    y = aug.highpass(y, u(rng, Q["highpass_hz"]))
    y = shelf(y, Q["low_shelf_hz"], u(rng, Q["low_shelf_db"]), False)
    y = shelf(y, Q["high_shelf_hz"], u(rng, Q["high_shelf_db"]), True)
    ref = y[sp] if np.any(y[sp]) else y
    y = aug.set_rms(y, u(rng, Q["level_dbfs"]), ref)
    y = np.clip(y, -1.0, 1.0)                                   # the ADC clips loud shouts
    return to_int16(aug.noise_floor(y, rng, u(rng, Q["noise_floor_dbfs"])))
