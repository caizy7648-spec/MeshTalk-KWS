"""features.py - the log-mel front-end, EXACTLY as the firmware computes it
(MeshTalk_ESP32 Ai/kws_frontend.c). Do not change one without the other:
Tests/test_kws.c and check_c.py compare the two.

    16 kHz mono int16 -> /32768
    frame 480 samples (30 ms), hop 320 (20 ms): frame i = x[i*320 : i*320+480]
    frame mean removed, periodic Hann window, 512 point FFT, power |X|^2
    40 mel bands (HTK mel), triangles from 60 to 7600 Hz, peak 1 (not area-normalised)
    feature = ln(band energy + 1e-6)
"""
import numpy as np

RATE = 16000
FRAME = 480
HOP = 320
NFFT = 512
BANDS = 40
FMIN = 60.0
FMAX = 7600.0
LOG_EPS = 1e-6


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_matrix():
    """[BANDS, NFFT//2+1] float32, the same triangles as the C code."""
    nb = NFFT // 2 + 1
    m0, m1 = _hz_to_mel(FMIN), _hz_to_mel(FMAX)
    edge = _mel_to_hz(m0 + (m1 - m0) * np.arange(BANDS + 2) / (BANDS + 1))
    f = np.arange(nb) * RATE / NFFT
    w = np.zeros((BANDS, nb))
    for b in range(BANDS):
        lo, ce, hi = edge[b], edge[b + 1], edge[b + 2]
        up = (f > lo) & (f <= ce) & (f < hi)
        dn = (f > ce) & (f < hi)
        w[b, up] = (f[up] - lo) / (ce - lo)
        w[b, dn] = (hi - f[dn]) / (hi - ce)
    return w.astype(np.float32)


_MEL = mel_matrix()
_WIN = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(FRAME) / FRAME)).astype(np.float32)


def n_frames(n_samples):
    return 0 if n_samples < FRAME else 1 + (n_samples - FRAME) // HOP


def logmel(pcm):
    """pcm: int16 array (or float -1..1). Returns float32 [frames, 40]."""
    x = np.asarray(pcm)
    x = x.astype(np.float32) / 32768.0 if x.dtype == np.int16 else x.astype(np.float32)
    n = n_frames(len(x))
    if n == 0:
        return np.zeros((0, BANDS), np.float32)
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    fr = x[idx]
    fr = fr - fr.mean(axis=1, keepdims=True)
    spec = np.fft.rfft(fr * _WIN, NFFT, axis=1)
    p = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    return np.log(p @ _MEL.T + LOG_EPS).astype(np.float32)


def dbfs(pcm):
    x = np.asarray(pcm, np.float64)
    if x.size == 0 or not np.any(x):
        return -120.0
    return 10.0 * np.log10(np.mean(x * x) / 32768.0 ** 2)
