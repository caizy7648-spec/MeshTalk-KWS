"""voice_mod.py - Phase D voice changes (numpy / scipy only, no extra packages).

    y = f0_shift(x, semitones)      # pitch (F0) changed, formants (vocal tract) kept   -> the SAME speaker
    y = vtl_shift(x, factor)        # speed perturbation: F0 AND formants move together  -> a NEW speaker

Why the two are treated differently (configs/spk.yaml phase_d):
  * a person can change the pitch of his / her voice at will (shouting, imitating a lower voice),
    but not the length of the vocal tract, which places the formants. Training with F0-shifted
    copies labelled as the same speaker makes the voice vector ignore pitch alone; an imitator who
    only lowers her pitch keeps her own formants and is not pulled towards the owner.
  * speed perturbation moves the whole spectrum (like a longer / shorter vocal tract); its copies
    are labelled as new speakers ("speaker augmentation", Yamamoto et al., Interspeech 2019), so the
    vector has to follow the formants.

f0_shift is LPC source-filter pitch shifting: the signal is inverse-filtered frame by frame with
its own LPC envelope (order `order`, 25 ms Hann frames, 50 % overlap), the residual (excitation,
spectrally flat) is pitch-shifted (kws.augment.pitch: phase vocoder + resampling), and each frame
is filtered again with the ORIGINAL envelope. The level is restored to the input RMS.
"""
import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import lfilter

from kws import augment as aug

RATE = 16000


def lpc(frame, order):
    """LPC polynomial a (a[0] = 1) of one windowed frame (autocorrelation method)"""
    r = np.correlate(frame, frame, "full")[len(frame) - 1:len(frame) + order]
    if r[0] <= 1e-10:
        return np.r_[1.0, np.zeros(order)]
    r = r.copy()
    r[0] *= 1.0 + 1e-6                                    # white-noise correction: always stable
    a = solve_toeplitz(r[:order], -r[1:order + 1])
    return np.r_[1.0, a]


def _frames(n, L, hop):
    return range(0, max(1, n - L + hop), hop)


def f0_shift(x, semitones, order=18, frame_s=0.025, pad_s=0.0125):
    """pitch changed by `semitones`, spectral envelope (formants) of every frame kept; same length"""
    x = np.asarray(x, np.float32)
    if abs(semitones) < 1e-3 or len(x) < 2 * int(frame_s * RATE):
        return x.copy()
    L = int(frame_s * RATE)
    hop = L // 2
    pad = int(pad_s * RATE)
    w = np.hanning(L + 2)[1:-1].astype(np.float64)          # periodic-ish Hann: sum of 50 % overlaps ~ 1
    n = len(x)
    xp = np.pad(x.astype(np.float64), (pad, L + pad))
    A, res = [], np.zeros(n + L + 2 * pad)
    for i in _frames(n + L, L, hop):
        seg = xp[i + pad:i + pad + L]
        a = lpc(seg * w, order)
        A.append((i, a))
        # residual of this frame (filter state warmed up over `pad` samples before it)
        e = lfilter(a, [1.0], xp[i:i + pad + L])[pad:]
        res[i + pad:i + pad + L] += e * w
    e2 = aug.pitch(res.astype(np.float32), semitones).astype(np.float64)
    y = np.zeros_like(res)
    for i, a in A:
        s = e2[i:i + pad + L]
        y[i + pad:i + pad + L] += lfilter([1.0], a, s)[pad:] * w
    y = y[pad:pad + n]
    r_in, r_out = np.sqrt(np.mean(x.astype(np.float64) ** 2)), np.sqrt(np.mean(y ** 2))
    if r_out > 1e-9:
        y *= r_in / r_out
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def vtl_shift(x, factor):
    """speed perturbation: factor > 1 = shorter, F0 and formants higher by `factor` (a different voice)"""
    return aug.speed(np.asarray(x, np.float32), factor)


# ------------------------------------------------------------------ measurements (tests, report)
def f0_estimate(x, lo=60.0, hi=400.0):
    """median F0 (Hz) over voiced 40 ms frames, autocorrelation; nan if nothing voiced"""
    L, hop = int(0.04 * RATE), int(0.01 * RATE)
    out = []
    lag_lo, lag_hi = int(RATE / hi), int(RATE / lo)
    e = np.array([np.sum(x[i:i + L] ** 2) for i in range(0, len(x) - L, hop)]) if len(x) > L else np.zeros(0)
    if not len(e):
        return float("nan")
    gate = e.max() * 0.1
    for k, i in enumerate(range(0, len(x) - L, hop)):
        if e[k] < gate:
            continue
        f = x[i:i + L] - np.mean(x[i:i + L])
        r = np.correlate(f, f, "full")[L - 1:]
        if r[0] <= 0:
            continue
        r = r / r[0]
        lag = lag_lo + int(np.argmax(r[lag_lo:lag_hi]))
        if r[lag] > 0.3:
            out.append(RATE / lag)
    return float(np.median(out)) if out else float("nan")


def formants(x, order=18, n=3):
    """first n formant frequencies (Hz) of the loudest 25 ms frame (LPC roots)"""
    L = int(0.025 * RATE)
    if len(x) <= L:
        return [float("nan")] * n
    i = max(range(0, len(x) - L, L // 2), key=lambda i: np.sum(x[i:i + L] ** 2))
    a = lpc(x[i:i + L].astype(np.float64) * np.hanning(L), order)
    r = [z for z in np.roots(a) if np.imag(z) > 0 and np.abs(z) > 0.85]
    f = sorted(np.angle(r) * RATE / (2 * np.pi))
    f = [v for v in f if v > 150]
    return (f + [float("nan")] * n)[:n]
