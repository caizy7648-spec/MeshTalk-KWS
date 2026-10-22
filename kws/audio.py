"""audio.py - small audio helpers shared by the scripts (no parameters here:
callers pass the values from configs/train.yaml).
"""
import numpy as np
from scipy.signal import resample_poly

RATE = 16000


def to_float(x):
    x = np.asarray(x)
    return x.astype(np.float32) / 32768.0 if x.dtype == np.int16 else x.astype(np.float32)


def to_int16(x):
    return np.clip(np.round(np.asarray(x, np.float64) * 32768.0), -32768, 32767).astype(np.int16)


def resample(x, sr_in, sr_out=RATE):
    if sr_in == sr_out:
        return x
    g = np.gcd(int(sr_in), int(sr_out))
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def rms_db(x):
    x = to_float(x).astype(np.float64)
    return 10.0 * np.log10(np.mean(x * x) + 1e-12)


def speech_span(x, rel_db=-35.0, hop=160, win=400):
    """(start, end) samples of the part louder than (loudest frame + rel_db)"""
    x = to_float(x)
    if len(x) < win:
        return 0, len(x)
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    e = 10.0 * np.log10(np.mean(x[idx] ** 2, axis=1) + 1e-12)
    on = np.where(e > e.max() + rel_db)[0]
    return int(on[0] * hop), int(min(len(x), on[-1] * hop + win))
