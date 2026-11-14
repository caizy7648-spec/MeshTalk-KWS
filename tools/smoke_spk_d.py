"""smoke_spk_d.py - runs the whole Phase D speaker pipeline on made-up voices (a few minutes).

    python tools/smoke_spk_d.py          # PASS / FAIL per step

Builds a throw-away world in build/smoke_spk_d/ (own configs, synthetic speakers - vowels with a
speaker's own pitch and formants -, noise, VCTK-like extra voices, genders, AISHELL-like Mandarin
speakers, the owner's and the imitator's DataRec takes, a random "old" model) and runs:
voice_mod check -> 47_train_spk_d (d0 and d2, 2 tiny epochs) -> 48_eval_spk_d --split val ->
48_eval_spk_d --split test -> 42_export_spk --model spk_d2.pt (C engine == Python).
Nothing in data/, models/, out/ or docs/ of the repository is touched. The numbers mean nothing.
"""
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml
from scipy.signal import lfilter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
W = os.path.join(ROOT, "build", "smoke_spk_d")
RATE = 16000
rng = np.random.default_rng(11)
VOWELS = [(700, 1200, 2600), (300, 2300, 3000), (500, 900, 2500), (400, 1900, 2600)]


def vowel(f0, F, sec):
    n = int(sec * RATE)
    src, k = np.zeros(n), 0.0
    while k < n:
        src[int(k)] = 1.0
        k += RATE / (f0 * (1 + 0.03 * np.sin(k / 900)))
    y = src
    for f, b in zip(F, (80, 90, 120)):
        r = np.exp(-np.pi * b / RATE)
        th = 2 * np.pi * min(f, 7000) / RATE
        y = lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], y)
    return y * np.sin(np.pi * np.arange(n) / n) ** 0.5


def word(f0, scale, k):
    """a 'word' of 2-3 vowels of a speaker (pitch f0, vocal tract scale) + silence around"""
    parts = [np.zeros(int(0.3 * RATE))]
    for j in range(2 + k % 2):
        F = [f * scale for f in VOWELS[(k + j) % 4]]
        parts += [vowel(f0 * rng.uniform(0.95, 1.05), F, rng.uniform(0.15, 0.25)), np.zeros(int(0.04 * RATE))]
    parts.append(np.zeros(int(0.3 * RATE)))
    x = np.concatenate(parts)
    x = x / (np.abs(x).max() + 1e-9) * 0.3 + rng.standard_normal(len(x)) * 1e-4
    return (x * 32767).astype(np.int16)


def speaker(name, female):
    return dict(name=name, gender="FEMALE" if female else "MALE",
                f0=rng.uniform(190, 240) if female else rng.uniform(95, 140),
                scale=rng.uniform(1.1, 1.2) if female else rng.uniform(0.9, 1.0))


def world():
    shutil.rmtree(W, ignore_errors=True)
    os.makedirs(W)
    C = yaml.safe_load(open(os.path.join(ROOT, "configs", "train.yaml")))
    C["paths"] = {k: os.path.join(W, v) for k, v in C["paths"].items()}
    yaml.safe_dump(C, open(os.path.join(W, "smoke.yaml"), "w"))
    P = yaml.safe_load(open(os.path.join(ROOT, "configs", "spk.yaml")))
    D = P["phase_d"]
    D["train"].update(epochs=2, steps_per_epoch=3, batch=16, workers=2, torch_threads=2, val_every=1, norm_windows=48)
    D["eval"].update(speakers_val=8, speakers_test=8, cohort_speakers=10, asnorm_topk=5,
                     old_model=os.path.join(W, "models", "spk_best_v1.pt"), big_model=os.path.join(W, "models", "nope.pt"))
    D["eval"]["own"]["draws"] = 3
    P["finetune"]["eval"]["impostors_per_gender"] = 3
    yaml.safe_dump(P, open(os.path.join(W, "spk.yaml"), "w"))
    os.environ["KWS_CONFIG"] = os.path.join(W, "smoke.yaml")
    os.environ["SPK_CONFIG"] = os.path.join(W, "spk.yaml")
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    sys.path.insert(0, ROOT)
    prep = __import__("02_prepare")
    cache = os.path.join(W, C["paths"]["cache"])
    genders = []
    st = prep.Store("clips")
    ex = prep.Store("spk_extra")
    for split, n in (("train", 16), ("val", 8), ("test", 8)):
        for k in range(n):
            sp = speaker("mswc-%s-%d" % (split, k), k % 2 == 1)
            genders.append(dict(speaker=sp["name"], gender=sp["gender"], source="labels"))
            for j in range(16):
                x = word(sp["f0"], sp["scale"], j)
                st.add(x, source="mswc", group="mswc_help", label="help", word="help", speaker=sp["name"], gender="",
                       split=split, spk_use=True, text="", kw_start=int(0.3 * RATE), kw_end=len(x) - int(0.3 * RATE))
        for k in range(2):
            sp = speaker("vctk-%s-%d" % (split, k), k == 1)
            genders.append(dict(speaker=sp["name"], gender=sp["gender"], source="predicted"))
            for j in range(16):
                x = word(sp["f0"], sp["scale"], j)
                ex.add(x, source="vctk", speaker=sp["name"], gender="", split=split, utt="u%d" % j, kw_start=0, kw_end=len(x))
    st.save(prep.COLS)
    ex.save()
    pd.DataFrame(genders).to_csv(os.path.join(cache, "spk_gender.csv"), index=False)
    ch = prep.Store("chat")
    for k in range(3):
        sp = speaker("ais-S%04d" % k, k == 0)
        for j in range(6):
            x = word(sp["f0"], sp["scale"], j)
            ch.add(x, source="aishell", group="chat_zh", label="_other", word="", speaker=sp["name"], gender="", split="train",
                   spk_use=False, text="", kw_start=0, kw_end=len(x))
    ch.save(prep.COLS)
    own = prep.Store("own")
    a, b = speaker("own-A", False), speaker("own-B", True)
    for sp, n in ((a, 30), (b, 12)):
        for j in range(n):
            f0 = sp["f0"] * (0.7 if sp is b else 1.0)               # the imitator lowers her voice
            x = np.concatenate([word(f0, sp["scale"], j), np.zeros(RATE, np.int16)])[:40000]
            own.add(x, source="own", group="own", label=["help", "help me", "sos"][j % 3], word=["NORMAL", "NORMAL", "LOUD", "FAR"][(j // 3) % 4],
                    speaker=sp["name"], gender="", split="test", spk_use=False, text="", kw_start=int(0.3 * RATE), kw_end=int(1.2 * RATE))
    own.save(prep.COLS)
    for s in ("train", "test"):
        nz = prep.Store("noise_" + s)
        for k in range(3):
            nz.add((rng.standard_normal(RATE * 3) * 2000).astype(np.int16), source="musan_noise", file="n%d" % k)
        nz.save()
    import torch
    from spk.model import SpkNet
    net = SpkNet(**P["net"])
    os.makedirs(os.path.join(W, "models"), exist_ok=True)
    torch.save({"cfg": net.cfg, "state": net.state_dict()}, D["eval"]["old_model"])


VOICE_CHECK = """
import numpy as np
from tools.smoke_spk_d import vowel
from scipy.signal import welch
from spk.voice_mod import f0_shift, vtl_shift, f0_estimate
x = vowel(120, (700, 1200, 2600), 0.8).astype(np.float32) * 0.3
def f12(y):
    # F1, F2 from the cepstrally smoothed long-term spectrum
    f, p = welch(y, 16000, nperseg=2048); c = np.fft.irfft(10 * np.log10(p + 1e-12)); c[40:-40] = 0
    e = np.fft.rfft(c).real[:len(f)]
    out = []
    for t in (700, 1200):
        lo, hi = np.searchsorted(f, t * 0.8), np.searchsorted(f, t * 1.2)
        out.append(f[lo + np.argmax(e[lo:hi])])
    return np.array(out)
F = f12(x)
for st in (-6, -3, 4):
    y = f0_shift(x, st)
    r = f0_estimate(y) / f0_estimate(x); want = 2 ** (st / 12)
    q = f12(y) / F
    assert len(y) == len(x) and abs(r / want - 1) < 0.04 and np.all(np.abs(q - 1) < 0.05), (st, r, want, q)
    print("f0_shift %+d st: F0 ratio %.3f (want %.3f), F1/F2 ratio %.3f / %.3f (kept)" % (st, r, want, q[0], q[1]))
q = f12(vtl_shift(x, 0.9)) / F
print("vtl_shift 0.9: F1/F2 ratio %.3f / %.3f (moved)" % tuple(q))
assert q[1] < 0.95
"""


def main():
    t0 = time.time()
    print("building", W, flush=True)
    world()
    env = dict(os.environ, PYTHONPATH=ROOT)
    m = os.path.join(W, "models")
    steps = [
        ("voice_mod: F0 shift keeps the formants", [sys.executable, "-c", VOICE_CHECK]),
        ("47_train_spk_d --variant d0", [sys.executable, "scripts/47_train_spk_d.py", "--variant", "d0"]),
        ("47_train_spk_d --variant d2", [sys.executable, "scripts/47_train_spk_d.py", "--variant", "d2"]),
        ("48_eval_spk_d --split val", [sys.executable, "scripts/48_eval_spk_d.py", "--split", "val"]),
        ("48_eval_spk_d --split test (old, d2)", [sys.executable, "scripts/48_eval_spk_d.py", "--split", "test", "--models", "old", "d2"]),
        ("42_export_spk --model spk_d2.pt", [sys.executable, "scripts/42_export_spk.py", "--model", os.path.join(m, "spk_d2.pt"),
                                             "--name", "spk-smoke", "--threshold", "0.5"]),
    ]
    fails = 0
    for name, cmd in steps:
        t = time.time()
        r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        ok = r.returncode == 0
        fails += 0 if ok else 1
        print("%-45s %s  (%.0f s)" % (name, "PASS" if ok else "FAIL", time.time() - t), flush=True)
        if not ok:
            print(r.stdout[-3000:], r.stderr[-3000:])
    for f in ("spk_d_eval_val.json", "spk_d_eval_test.json"):
        p = os.path.join(m, f)
        if os.path.exists(p):
            import json
            d = json.load(open(p))
            print(f, "models:", [k for k in d if isinstance(d[k], dict) and "public" in d[k]], "best:", d.get("best_variant_val"))
    print("smoke test: %d failed (%.0f s)" % (fails, time.time() - t0))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
