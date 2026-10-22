"""augment.py - data augmentation for the training windows (Step 2).

Every function takes float32 audio (-1..1, 16 kHz) and a numpy Generator;
the values (ranges, probabilities) come from configs/train.yaml augment:,
nothing is hard-coded here except the sample rate and the FFT sizes of the
phase vocoder.

    gain        level change in dB
    add_noise   noise at a given SNR (relative to the speech part)
    noise_floor white microphone self-noise at an absolute level
    reverb      synthetic room impulse response (exponential decay, RT60)
    speed       resampling: tempo and pitch together
    pitch       phase-vocoder pitch shift, tempo kept
    shout       gain + soft clipping (tanh): a shouted / overdriven voice
    highpass    2nd-order Butterworth high-pass (microphone / case)
"""
from fractions import Fraction

import numpy as np
from scipy.signal import butter, fftconvolve, resample_poly, sosfilt

RATE = 16000


def db(x):
    return 10.0 ** (x / 20.0)


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2) + 1e-12))


def gain(x, g_db):
    return x * db(g_db)


def set_rms(x, level_dbfs, ref=None):
    """scale x so that rms(ref or x) = level_dbfs"""
    return x * (db(level_dbfs) / rms(x if ref is None else ref))


def add_noise(x, noise, snr_db, ref):
    """noise (same length) scaled so that rms(ref) / rms(noise) = snr_db"""
    return x + noise * (rms(ref) / (rms(noise) * db(snr_db)))


def noise_floor(x, rng, level_dbfs):
    return x + rng.standard_normal(len(x)).astype(np.float32) * db(level_dbfs)


def synth_rir(rng, rt60, length_s, rate=RATE):
    """direct path + exponentially decaying noise tail (decays 60 dB in rt60 seconds)"""
    n = int(length_s * rate)
    t = np.arange(n) / rate
    h = rng.standard_normal(n) * np.exp(-6.908 * t / rt60)
    pre = int(rng.uniform(0.002, 0.01) * rate)          # first reflections a few ms after the direct sound
    h[:pre] = 0.0
    h[0] = 1.0
    return (h / np.sqrt(np.sum(h * h))).astype(np.float32)


def reverb(x, rng, rt60, length_s, wet):
    h = synth_rir(rng, rt60, length_s)
    y = fftconvolve(x, h)[:len(x)]
    y = y * (rms(x) / rms(y))
    return ((1.0 - wet) * x + wet * y).astype(np.float32)


def speed(x, factor):
    """factor > 1: faster and higher. Returns the resampled audio."""
    f = Fraction(factor).limit_denominator(50)
    return resample_poly(x, f.denominator, f.numerator).astype(np.float32)


def _stretch(x, rate, n_fft=512, hop=128):
    """phase vocoder: rate > 1 makes x shorter, pitch unchanged"""
    win = np.hanning(n_fft).astype(np.float32)
    pad = np.concatenate([np.zeros(n_fft, np.float32), x, np.zeros(n_fft, np.float32)])
    nf = 1 + (len(pad) - n_fft) // hop
    S = np.fft.rfft(np.stack([pad[i * hop:i * hop + n_fft] * win for i in range(nf)]), axis=1)
    steps = np.arange(0, nf - 1, rate)
    omega = 2 * np.pi * hop * np.arange(n_fft // 2 + 1) / n_fft
    phase = np.angle(S[0])
    out = np.zeros(len(steps) * hop + n_fft, np.float32)
    norm = np.zeros_like(out)
    for k, s in enumerate(steps):
        i = int(s)
        frac = s - i
        mag = (1 - frac) * np.abs(S[i]) + frac * np.abs(S[i + 1])
        frame = np.fft.irfft(mag * np.exp(1j * phase), n_fft).astype(np.float32) * win
        out[k * hop:k * hop + n_fft] += frame
        norm[k * hop:k * hop + n_fft] += win * win
        dphi = np.angle(S[i + 1]) - np.angle(S[i]) - omega
        dphi -= 2 * np.pi * np.round(dphi / (2 * np.pi))
        phase = phase + omega + dphi
    out = out / np.maximum(norm, 1e-3)
    return out[n_fft:n_fft + int(round(len(x) / rate))]


def pitch(x, semitones):
    """pitch shift by semitones, same length (stretch, then resample back)"""
    r = 2.0 ** (semitones / 12.0)
    y = _stretch(x, 1.0 / r)                 # longer by r
    y = speed(y, r)                          # back to the original length, pitch * r
    n = len(x)
    return np.pad(y, (0, max(0, n - len(y))))[:n].astype(np.float32)


def shout(x, g_db):
    """drive into a soft clipper: tanh(g * x), peak stays below 1"""
    g = db(g_db)
    return np.tanh(x * g).astype(np.float32)


def highpass(x, fc):
    sos = butter(2, fc, btype="highpass", fs=RATE, output="sos")
    return sosfilt(sos, x).astype(np.float32)
