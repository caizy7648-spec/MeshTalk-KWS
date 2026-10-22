"""03_tts.py - synthetic "help" / "help me" / "SOS" (and synthetic _other
words) with Kokoro-82M (Apache-2.0, kokoro-onnx).

    python scripts/03_tts.py [--procs 4]

Output: data/cache/tts/{train,val,test}.npy (int16, 16 kHz, clips one after
the other) + .csv (label, text, voice, speed, offset, length, kw_start, kw_end).
Voices, speeds, texts and counts: configs/train.yaml (tts:). Every clip's
voice / text / speed comes from rng_for(seed, "tts", split, label, i), so a
re-run gives the same clips. Synthetic voices are used for the keyword head
ONLY (never for the speaker head).
"""
import argparse
import csv
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kws.audio import resample, speech_span, to_int16  # noqa: E402
from kws.cfg import load, rng_for  # noqa: E402

C = load()
T = C["tts"]
RAW = os.path.join(C["paths"]["raw"], "kokoro")
OUT = os.path.join(C["paths"]["cache"], "tts")
PAD = 0.10          # seconds kept around the spoken part
_k = None


def engine():
    global _k
    if _k is None:
        import onnxruntime as ort
        from kokoro_onnx import Kokoro
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        s = ort.InferenceSession(os.path.join(RAW, "kokoro-v1.0.onnx"), so, providers=["CPUExecutionProvider"])
        _k = Kokoro.from_session(s, os.path.join(RAW, "voices-v1.0.bin"))
    return _k


def reset():
    """each worker opens its own Kokoro (a shared voices file handle breaks)"""
    global _k
    _k = None


def voices():
    allv = sorted(engine().get_voices())
    eng = [v for v in allv if any(v.startswith(p) for p in T["voice_prefixes"])]
    hold = set(T["test_voices"]) | set(T["val_voices"])
    return {"train": [v for v in eng if v not in hold], "val": list(T["val_voices"]), "test": list(T["test_voices"])}


def jobs():
    """(split, label, i) for every clip"""
    out = []
    for lab, n in T["clips_per_class"].items():
        out += [("train", lab, i) for i in range(n)]
        nt = T["test_clips_per_class"] if lab != "_other" else T["test_clips_per_class"] * 2
        out += [("val", lab, i) for i in range(nt)]
        out += [("test", lab, i) for i in range(nt)]
    return out


def make(job):
    split, lab, i = job
    k = engine()
    V = voices()[split]
    rng = rng_for(C, "tts", split, lab, i)
    text = T["texts"][lab][rng.integers(len(T["texts"][lab]))]
    v1 = V[rng.integers(len(V))]
    lang = "en-gb" if v1[0] == "b" else "en-us"
    name, style = v1, v1
    if split == "train" and rng.random() < T["blend_prob"]:
        same = [v for v in V if v[0] == v1[0] and v != v1]
        v2 = same[rng.integers(len(same))]
        a = float(rng.uniform(0.3, 0.7))
        style = a * k.get_voice_style(v1) + (1 - a) * k.get_voice_style(v2)
        name = "%s*%.2f+%s" % (v1, a, v2)
    speed = float(rng.uniform(*T["speed"]))
    x, sr = k.create(text, voice=style, speed=speed, lang=lang)
    x = resample(np.asarray(x, np.float32), sr)
    s, e = speech_span(x)
    p = int(PAD * 16000)
    a, b = max(0, s - p), min(len(x), e + p)
    pk = np.max(np.abs(x[a:b])) + 1e-9
    y = to_int16(x[a:b] / pk * 0.5)
    return dict(split=split, label=lab, i=i, text=text, voice=name, speed=round(speed, 3),
                kw_start=s - a, kw_end=e - a), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    J = jobs()
    print("clips:", len(J), "voices:", {k: len(v) for k, v in voices().items()}, flush=True)
    res = {"train": [], "val": [], "test": []}
    with Pool(a.procs, initializer=reset) as pool:
        for n, (meta, y) in enumerate(pool.imap(make, J, chunksize=8)):
            res[meta["split"]].append((meta, y))
            if n % 500 == 0:
                print(n, meta, flush=True)
    for split, items in res.items():
        off, rows = 0, []
        for meta, y in items:
            rows.append(dict(meta, offset=off, length=len(y)))
            off += len(y)
        np.save(os.path.join(OUT, split + ".npy"), np.concatenate([y for _, y in items]))
        with open(os.path.join(OUT, split + ".csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(split, len(rows), "clips")


if __name__ == "__main__":
    main()
