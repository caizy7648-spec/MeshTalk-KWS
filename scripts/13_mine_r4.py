"""13_mine_r4.py - Phase C hard negatives: where r3 fires on the NEW training audio.

    python scripts/13_mine_r4.py          # -> data/cache/hardneg_r4.npy / .csv (group hard_neg_r4)

Looks at the training parts of chat.* (AMI conversation, AISHELL Mandarin) and vocal.*
(coughs, throat clearing, sneezes, sniffs, sighs, laughter) the way the device does (a window
every 240 ms) with mine_r4.model (r3). A window whose best keyword probability reaches
mine_r4.threshold becomes a 2 s _other clip ending mine_r4.clip_after_s after that window -
at most one per mine_r4.min_gap_s, the most confident mine_r4.max_per_source[source] kept.
A short clip is heard with 1 s of microphone noise before it and 0.5 s after, as on the device.
Statistics (what fired, how often, by sound type): data/cache/mine_stats_r4.json - the
"before" picture for REPORT.md.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kws.cfg import ROOT, load, rng_for  # noqa: E402
from kws.evaluate import batch_run, run_frames  # noqa: E402
from kws.features import HOP, logmel  # noqa: E402
from kws.model import KwsNet  # noqa: E402

prep = __import__("02_prepare")
C = load()
M = C["mine_r4"]
CACHE = C["paths"]["cache"]
RATE = 16000


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_net(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = KwsNet(**dict(ck["cfg"]))
    net.load_state_dict(ck["state"], strict=False)
    return net.eval()


def padded(x, rng):
    """1 s before, 0.5 s after, microphone noise floor everywhere (about -70 dBFS)"""
    y = np.concatenate([np.zeros(RATE, np.float32), x.astype(np.float32), np.zeros(RATE // 2, np.float32)])
    y += rng.standard_normal(len(y)).astype(np.float32) * (32768 * 10 ** (-70 / 20))
    return np.clip(np.round(y), -32768, 32767).astype(np.int16)


def mine(net, x, kmask):
    T = net.cfg["T"]
    f = logmel(x)
    if len(f) < T:
        return []
    fr = run_frames(T, len(f))
    p, _ = batch_run(net, f, fr, T)
    pk = p[:, kmask].max(1)
    kc = p[:, kmask].argmax(1)
    out, ends = [], []
    for j in np.argsort(-pk):
        if pk[j] < M["threshold"]:
            break
        end = int(fr[j]) * HOP + 480
        if any(abs(end - u) < M["min_gap_s"] * RATE for u in ends):
            continue
        ends.append(end)
        b = min(len(x), end + int(M["clip_after_s"] * RATE))
        a = max(0, b - int(M["clip_s"] * RATE))
        out.append((a, b, float(pk[j]), int(kc[j])))
    return out


def main():
    torch.set_num_threads(M["torch_threads"])
    net = load_net(os.path.join(ROOT, M["model"]))
    labels = C["labels"]
    kmask = np.array([not l.startswith("_") for l in labels])
    kw_names = [l for l in labels if not l.startswith("_")]
    rng = rng_for(C, "mine_r4")
    found = {}
    stats = {}
    t0 = time.time()
    for store in ("chat", "vocal"):
        path = os.path.join(CACHE, store + ".csv")
        if not os.path.exists(path):
            log("no %s - run 06_prepare_r4.py" % path)
            continue
        df = pd.read_csv(path, keep_default_na=False)
        a = np.load(os.path.join(CACHE, store + ".npy"), mmap_mode="r")
        df = df[df.split == "train"]
        for i, r in enumerate(df.itertuples()):
            x = padded(np.array(a[r.offset:r.offset + r.length]), rng)
            src = r.group if store == "chat" else "vocal"
            key = "%s|%s" % (src, r.word or "-")
            st = stats.setdefault(key, dict(clips=0, fired=0))
            st["clips"] += 1
            hits = mine(net, x, kmask)
            if hits:
                st["fired"] += 1
            for s, e, p, k in hits:
                found.setdefault(src, []).append((p, x[s:e], dict(
                    source=r.source, group="hard_neg_r4", label="_other", word=("%s:%s" % (r.word or src, kw_names[k]))[:60],
                    speaker=r.speaker, gender="", split="train", spk_use=False, text=r.text, kw_start=0, kw_end=e - s,
                    prob=round(p, 4))))
            if i % 2000 == 0:
                log("%s %d/%d, %d windows fired so far, %.0f min" % (store, i, len(df), sum(len(v) for v in found.values()),
                                                                   (time.time() - t0) / 60))
    hn = prep.Store("hardneg_r4")
    kept = {}
    for src, rows in found.items():
        rows.sort(key=lambda t: -t[0])
        rows = rows[:M["max_per_source"].get(src, 5000)]
        kept[src] = len(rows)
        for p, seg, meta in rows:
            hn.add(seg, **meta)
    hn.save(prep.COLS + ["prob"])
    for v in stats.values():
        v["fired_pct"] = round(100.0 * v["fired"] / max(1, v["clips"]), 2)
    out = dict(model=M["model"], threshold=M["threshold"], kept=kept, by_source_and_type=stats,
               minutes=round((time.time() - t0) / 60, 1))
    json.dump(out, open(os.path.join(CACHE, "mine_stats_r4.json"), "w"), indent=1)
    log(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
