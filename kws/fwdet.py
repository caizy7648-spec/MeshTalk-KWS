"""fwdet.py - the firmware v1.6.7 keyword detector (c_ref_v167/) from Python.

Phase C (r4) measures everything with the logic the device really runs:
Svc/kws.c feeds 20 ms blocks -> kws_dbfs -> C log-mel front end ->
kws_det_frame (sound segments, words mode, 2 calls in 10 s, confirm drop /
word slack, cls_extra / no_relax of v1.6.7 ...). kws/detect.py is the older
v1.4.1 copy and is NOT used for the r4 numbers.

    python -m kws.fwdet --build                       # builds build/fwdet.so (gcc)
    python -m kws.fwdet --check                       # C front end == kws.features.logmel, scripted detector cases

    det = FwDet(net, thr=[0.65, 0.55, 0.50], cls_off=[-0.10, 0, 0.05, 0])  # PyTorch model gives the probabilities
    events = det.run(pcm_int16)          # [(t_ms from the stream start, flags, class, score), ...]
    det.reset()                          # next independent stream

The defaults are the device defaults of v1.6.7: medium sensitivity, rule
"2 calls within 10 s" with words mode, single "help" +0.05 and no relaxation
(help_extra). native=True uses the int8 C engine with the model compiled into
the library instead of the PyTorch model (build with --model out/kws_model.c).
"""
import argparse
import ctypes
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "c_ref_v167")
BUILD = os.path.join(ROOT, "build")
EV_HIT, EV_REJECT, EV_SOS = 2, 4, 8
HOP = 320
CB = ctypes.CFUNCTYPE(None, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
                      ctypes.POINTER(ctypes.c_float))


def lib_path(model_c=None):
    return os.path.join(BUILD, "fwdet.so" if model_c is None else "fwdet_native.so")


def build(model_c=None):
    """compile c_ref_v167 + harness.c -> build/fwdet.so (model: c_ref_v167/kws_model.c = r3, or model_c)"""
    os.makedirs(BUILD, exist_ok=True)
    out = lib_path(model_c)
    srcs = [os.path.join(SRC, f) for f in ("harness.c", "kws_frontend.c", "kws_engine.c", "kws_detect.c")]
    srcs.append(os.path.abspath(model_c) if model_c else os.path.join(SRC, "kws_model.c"))
    cmd = ["gcc", "-O2", "-shared", "-fPIC", "-I" + SRC, "-o", out] + srcs + ["-lm"]
    subprocess.run(cmd, check=True)
    return out


def _lib(model_c=None):
    path = lib_path(model_c)
    if not os.path.exists(path):
        build(model_c)
    L = ctypes.CDLL(path)
    L.h_new.restype = ctypes.c_void_p
    L.h_new.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_int, CB]
    L.h_feed.restype = ctypes.c_int
    L.h_feed.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    L.h_reset.argtypes = [ctypes.c_void_p]
    L.h_free.argtypes = [ctypes.c_void_p]
    L.h_now.restype = ctypes.c_uint32
    L.h_now.argtypes = [ctypes.c_void_p]
    L.h_logmel.restype = ctypes.c_int
    L.h_logmel.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    return L


def c_logmel(pcm):
    """C front end features [n, 40] of int16 audio"""
    L = _lib()
    pcm = np.ascontiguousarray(pcm, np.int16)
    n = len(pcm) // HOP
    out = np.zeros((n, 40), np.float32)
    k = L.h_logmel(pcm.ctypes.data, len(pcm), out.ctypes.data, n)
    return out[:k]


class FwDet:
    def __init__(self, net=None, thr=None, cls_off=None, labels=("help", "help me", "sos", "_other"), T=None,
                 sens=1, rule=0, help_extra=0.05, native=False, model_c=None, probs_fn=None):
        """net: KwsNet (or probs_fn(window[T,40]) -> probs); thr: [low, medium, high];
        cls_off: per class; rule 0 = 2 calls within 10 s (words mode on, v1.6.7), 1 = 3 within 20 s,
        2 = 2 calls within 10 s with words mode OFF (each sound counts once - a firmware option to test)"""
        self.L = _lib(model_c if native else None)
        self.labels = list(labels)
        ncls = len(self.labels)
        self.net, self.probs_fn = net, probs_fn
        if net is not None:
            import torch
            self.torch = torch
            T = net.cfg["T"]
        self.T = T
        help_k = [i for i, l in enumerate(self.labels) if l == "help"]
        extra = np.zeros(8, np.float32)
        norel = np.zeros(8, np.uint8)
        for i in help_k:
            extra[i] = help_extra
            norel[i] = 1 if help_extra is not None and help_extra > 0 else 0
        self._keep = []
        thr_a = None if thr is None else np.asarray(thr, np.float32)
        off_a = None if cls_off is None else np.pad(np.asarray(cls_off, np.float32), (0, 8 - ncls))
        self._keep += [extra, norel, thr_a, off_a]
        self._cb = CB(self._probs)
        self.h = self.L.h_new(1 if native else 0, int(T or 0), ncls, "|".join(self.labels).encode(),
                              None if thr_a is None else thr_a.ctypes.data,
                              None if off_a is None else off_a.ctypes.data,
                              extra.ctypes.data, norel.ctypes.data, int(sens), 3 if rule == 1 else 2,
                              20000 if rule == 1 else 10000, 1 if rule == 0 else 0, self._cb)
        self.runs = 0
        self.ev = np.zeros(4 * 4096, np.int32)

    def _probs(self, now, win, T, word, probs):
        w = np.ctypeslib.as_array(win, shape=(T * 40,)).reshape(1, T, 40).copy()
        if self.probs_fn is not None:
            p = np.asarray(self.probs_fn(w[0]), np.float32)
        else:
            with self.torch.no_grad():
                lg, _ = self.net(self.torch.from_numpy(w))
                p = self.torch.softmax(lg, 1)[0].numpy()
        for k in range(len(p)):
            probs[k] = float(p[k])
        self.runs += 1

    def run(self, pcm, chunk_s=60):
        """events of one stream: [(t_ms from the start of pcm, flags, class, score)];
        the detector state goes on from the previous call (use reset() between independent streams)"""
        pcm = np.ascontiguousarray(pcm, np.int16)
        t0 = self.L.h_now(self.h)
        step = int(chunk_s * 16000) // HOP * HOP
        out = []
        for pos in range(0, len(pcm), step):
            seg = np.ascontiguousarray(pcm[pos:pos + step])
            n = self.L.h_feed(self.h, seg.ctypes.data, len(seg), self.ev.ctypes.data, 4096)
            for i in range(n):
                t, f, c, s = (int(v) for v in self.ev[4 * i:4 * i + 4])
                out.append((t - t0, f, c, s / 1000.0))
        return out

    def reset(self):
        self.L.h_reset(self.h)

    def __del__(self):
        try:
            self.L.h_free(self.h)
        except Exception:  # noqa: BLE001
            pass


# ------------------------------------------------------------------ checks
def check():
    """C front end == kws.features.logmel; scripted detector cases behave as the firmware tests say"""
    from .features import logmel
    rng = np.random.default_rng(1)
    t = np.arange(16000 * 3) / 16000.0
    x = (0.3 * np.sin(2 * np.pi * 440 * t) * (t > 1) + 0.01 * rng.standard_normal(len(t))) * 32767
    pcm = np.clip(x, -32768, 32767).astype(np.int16)
    a, b = c_logmel(pcm), logmel(pcm)
    n = min(len(a), len(b))
    d = float(np.abs(a[:n] - b[:n]).max())
    ok = n > 100 and d < 1e-3
    print("C front end vs kws.features.logmel: %d frames, max diff %.2e  %s" % (n, d, "ok" if ok else "FAIL"))
    fails = 0 if ok else 1

    # scripted probabilities: two loud 0.3 s sounds 2.5 s apart, "help" at p
    def stream():
        s = np.zeros(16000 * 9, np.float64)
        for at in (2.0, 4.5):
            i = int(at * 16000)
            s[i:i + 4800] = 0.25 * np.sin(2 * np.pi * 300 * np.arange(4800) / 16000)
        s += 0.0005 * rng.standard_normal(len(s))
        return np.clip(s * 32767, -32768, 32767).astype(np.int16)

    def scripted(p1, p2, cls):
        state = {"n": 0}

        def fn(w):
            loud = w.max() > 0.0                            # the sound is in the window (log-mel > 0)
            p = np.array([0.02, 0.02, 0.02, 0.94], np.float32)
            if loud:
                p[:] = 0.02
                v = p1 if state["n"] < 1 else p2
                p[cls] = v
                p[3] = 1 - v - 0.04
            return p
        return fn, state

    pcm = stream()
    for p1, p2, cls, extra, want, what in [
            (0.70, 0.45, 0, 0.0, 1, "help 0.70 + 0.45, as before v1.6.7 -> SOS"),
            (0.70, 0.45, 0, 0.05, 0, "help 0.70 + 0.45, v1.6.7 (+0.05, no relax) -> no SOS"),
            (0.70, 0.45, 2, 0.05, 1, "sos 0.70 + 0.45, v1.6.7 -> SOS (help only is stricter)")]:
        fn, st = scripted(p1, p2, cls)

        def counting(w, fn=fn, st=st):
            p = fn(w)
            return p
        det = FwDet(probs_fn=counting, T=81, thr=[0.5, 0.5, 0.5], cls_off=[0, 0, 0, 0], help_extra=extra)
        evs = []
        # the first sound's runs give p1, the second's p2: switch when the first hit has been reported
        for k in range(0, len(pcm), 3200):
            e = det.run(pcm[k:k + 3200])
            evs += e
            if any(f & EV_HIT for _, f, _, _ in e):
                st["n"] += 1
        sos = sum(1 for _, f, _, _ in evs if f & EV_SOS)
        good = (sos > 0) == bool(want)
        fails += 0 if good else 1
        print("  %-55s hits %d, SOS %d  %s" % (what, sum(1 for _, f, _, _ in evs if f & EV_HIT), sos, "ok" if good else "FAIL"))
    print("%d failed" % fails)
    return fails


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--model", help="a kws_model.c for the native (int8 C engine) library")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.build:
        print("built", build(None))
        if a.model:
            print("built", build(a.model))
    if a.check:
        sys.exit(check())
