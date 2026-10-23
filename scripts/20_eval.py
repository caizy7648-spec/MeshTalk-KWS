"""20_eval.py - SYSTEM-level evaluation of the keyword model with the device
logic (kws.detect.Detector: window gate, 3-run average, thr + cls_off,
1.2 s pause, 3 hits in 20 s -> SOS).

    python scripts/20_eval.py --model models/kw_r2.pt --split val --sweep   # dev: try thr x cls_off
    python scripts/20_eval.py --model models/best.pt  --split test          # final numbers (export: settings)

1. Trigger rate: one person calls N times within 20 s (N = 4 / 6 / 8, natural
   variation of level and speed), half the scenarios clean, half in noise
   (SNR 5-20 dB). Scenario sets (eval_sys.sets): real "help" only, real
   help / help me mixed, Kokoro help / help me / SOS mixed, Kokoro help only,
   Kokoro SOS only. A scenario "triggers" when Detector raises EV_SOS.
2. False triggers: LibriSpeech dev / test speech + the held-out speakers'
   speech from 12_mine.py (half with noise), and noise alone. False hits
   per hour, false SOS per 24 h, and the words being said at each false hit.
split val = validation speakers / voices + dev speech + half of the test
noise files; split test = test speakers / voices + test speech + the other
half. Thresholds are only ever chosen on val.
Results: models/eval_sys_<split>.json (+ _sweep), docs/sys_*.png.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kws import augment as aug  # noqa: E402
from kws.audio import to_float, to_int16  # noqa: E402
from kws.cfg import load, rng_for, stable_hash  # noqa: E402
from kws.dataset import Data  # noqa: E402
from kws.detect import EV_SOS  # noqa: E402
from kws.evaluate import Stream, detect  # noqa: E402
from kws.model import KwsNet  # noqa: E402

C = load()
E = C["eval"]
S = C["eval_sys"]
X = C["export"]
CACHE, MODELS, DOCS = C["paths"]["cache"], C["paths"]["models"], C["paths"]["docs"]
LABELS = C["labels"]
KEYWORD = [not l.startswith("_") for l in LABELS]
RATE = 16000
G = {}          # streams shared with the worker processes (fork)


def load_net(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = KwsNet(**ck["cfg"])
    net.load_state_dict(ck["state"], strict=False)
    return net.eval()


# ------------------------------------------------------------------ scenarios
def pools(D, split):
    """per scenario set: list of speakers, each a dict label -> list of clip makers"""
    df = D.df
    out = {}
    sp = {}
    for g in ("mswc_help", "libri_help"):
        for s, idx in df[(df.group == g) & (df.split == split)].groupby("speaker").groups.items():
            sp.setdefault(s, []).extend(list(idx))
    out["real_help"] = [{"help": [("clip", i) for i in sorted(v)]} for _, v in sorted(sp.items())]
    mixed = []
    joinable = {}
    for k, (h, m) in enumerate(D.join[split]):
        joinable[df.speaker[h[0]]] = k
    hm = df[(df.group == "libri_help_me") & (df.split == split)].groupby("speaker").groups
    for s in sorted(joinable):
        d = {"help": [("clip", i) for i in sorted(sp.get(s, []))]}
        d["help me"] = [("clip", i) for i in sorted(hm.get(s, []))] + [("join", joinable[s])] * 2
        mixed.append(d)
    out["real_mixed"] = mixed
    tts = df[(df.source == "tts") & (df.split == split) & (df.label != "_other")]
    voices = {}
    for (v, lab), g in tts.groupby(["speaker", "label"]):
        base = v.split("*")[0]
        voices.setdefault(base, {}).setdefault(lab, []).extend(("clip", i) for i in g.index)
    out["tts_mixed"] = [voices[v] for v in sorted(voices)]
    out["tts_help"] = [{"help": voices[v]["help"]} for v in sorted(voices)]
    out["tts_sos"] = [{"sos": voices[v]["sos"]} for v in sorted(voices)]
    return out


def call_audio(D, rng, maker, split):
    kind, i = maker
    if kind == "clip":
        return D.clip(i)
    h, m = D.join[split][i]
    return _join(D, rng, h, m)


def _join(D, rng, hs, ms):
    """the same speaker's help + me (like Data.joined_help_me, for one given speaker)"""
    h, hs0, he0 = D.clip(hs[rng.integers(len(hs))])
    m, ms0, me0 = D.clip(ms[rng.integers(len(ms))])
    t = int(0.02 * RATE)
    h = h[:min(len(h), he0 + t)]
    m = m[max(0, ms0 - t):]
    m = m * (aug.rms(h[hs0:he0]) / aug.rms(m[t:t + (me0 - ms0)]))
    gap = np.zeros(int(rng.uniform(*C["train_kw"]["join_gap_ms"]) / 1000 * RATE), np.float32)
    return np.concatenate([h, gap, m]).astype(np.float32), hs0, len(h) + len(gap) + (me0 - ms0) + t


def scenario(D, rng, speaker, n_calls, noisy, split):
    """(int16 stream, [(label, start_ms, end_ms)]) - one person calling n_calls times in 20 s"""
    total = int((S["lead_s"] + S["window_s"] + S["tail_s"]) * RATE)
    y = np.zeros(total, np.float32)
    level = rng.uniform(*E["test_level_dbfs"])
    slot = S["window_s"] / n_calls
    labs = sorted(speaker)
    calls, speech = [], []
    for c in range(n_calls):
        lab = labs[rng.integers(len(labs))]
        mk = speaker[lab][rng.integers(len(speaker[lab]))]
        x, s, e = call_audio(D, rng, mk, split)
        if rng.random() < S["p_speed"]:
            f = rng.uniform(*S["speed"])
            x = aug.speed(x, f)
            s, e = int(s / f), min(len(x), int(e / f))
        x = aug.set_rms(x, level + rng.uniform(*S["call_gain_db"]), x[s:e] if e - s > 160 else x)
        room = max(0.0, slot - len(x) / RATE)
        o = int((S["lead_s"] + c * slot + rng.uniform(0, room)) * RATE)
        n = min(len(x), total - o)
        y[o:o + n] += x[:n]
        calls.append((lab, (o + s) * 1000 // RATE, (o + e) * 1000 // RATE))
        speech.append(x[s:e])
    if noisy:
        ref = np.concatenate(speech)
        y = aug.add_noise(y, D.noise_seg(rng, "test", total), rng.uniform(*E["test_snr_db"]), ref)
    y = aug.noise_floor(y, rng, E["noise_floor_dbfs"])
    return to_int16(y), calls


def build_scenarios(net, D, split):
    T = net.cfg["T"]
    P = pools(D, split)
    out = []
    for name in S["sets"]:
        spk = P[name]
        if not spk:
            continue
        for n_calls in S["calls"]:
            for cond in ("clean", "noisy"):
                for k in range(S["scenarios"]):
                    rng = rng_for(C, "scen", split, name, n_calls, cond, k)
                    sp = spk[int(stable_hash("scspk:%s:%d:%s:%d" % (name, n_calls, cond, k)) % len(spk))]
                    pcm, calls = scenario(D, rng, sp, n_calls, cond == "noisy", split)
                    st = Stream(net, pcm, T)
                    st.pcm = None
                    out.append(dict(set=name, n=n_calls, cond=cond, calls=calls, st=st))
    return out


# ------------------------------------------------------------------ false-alarm streams
def fa_sources(split):
    name = "dev" if split == "val" else "test"
    out = []
    for base in ("fa_%s" % name, "fa_%s_extra" % name):
        p = os.path.join(CACHE, base + ".npy")
        if os.path.exists(p):
            out.append((base, pd.read_csv(os.path.join(CACHE, base + ".csv")), np.load(p, mmap_mode="r"),
                        pd.read_csv(os.path.join(CACHE, base + "_words.csv"), keep_default_na=False)))
    return out


def build_fa(net, D, split):
    """streams of ~fa_chunk_min minutes: speech (with words) and noise"""
    T = net.cfg["T"]
    rng = rng_for(C, "fa", split)
    out = []
    kwset = set(C["sources"]["librispeech"]["keyword_words"])
    for base, sp, a, words in fa_sources(split):
        wat = words["at"].to_numpy()
        cur, meta, pos = [], [], 0
        for r in sp.itertuples():
            x = to_float(np.array(a[r.offset:r.offset + r.length]))
            x = aug.set_rms(x, rng.uniform(*E["test_level_dbfs"]))
            if rng.random() < E["fa_noise_mix_prob"]:
                x = aug.add_noise(x, D.noise_seg(rng, "test", len(x)), rng.uniform(*E["fa_noise_snr_db"]), x)
            x = aug.noise_floor(x, rng, E["noise_floor_dbfs"])
            meta.append((pos, r.offset, r.length))
            cur.append(x)
            pos += len(x)
            if pos > E["fa_chunk_min"] * 60 * RATE or r.Index == len(sp) - 1:
                # words of this stream, in stream time (ms)
                w = []
                for p0, off, ln in meta:
                    j0, j1 = np.searchsorted(wat, off), np.searchsorted(wat, off + ln)
                    for wd in words.iloc[j0:j1].itertuples():
                        w.append((wd.word, (p0 + wd.at - off) * 1000 // RATE, (p0 + wd.end - off) * 1000 // RATE))
                guard = [(s, e + int(E["keyword_guard_s"] * 1000)) for wd, s, e in w if wd in kwset]
                st = Stream(net, to_int16(np.concatenate(cur)), T)
                hours = len(st.feats) * 0.02 / 3600
                st.pcm = None
                out.append(dict(kind="speech", src=base, hours=hours, guard=guard, words=w, st=st))
                cur, meta, pos = [], [], 0
                if G.get("quick"):
                    break
    nd, na = D.noise["test"]
    keep = [r for r in nd.itertuples() if (stable_hash("noisesplit:" + r.file) % 2 == 0) == (split == "val")]
    cur, pos = [], 0
    for k, r in enumerate(keep):
        x = to_float(np.array(na[r.offset:r.offset + r.length]))
        cur.append(aug.noise_floor(aug.set_rms(x, rng.uniform(*E["noise_level_dbfs"])), rng, E["noise_floor_dbfs"]))
        pos += len(x)
        if pos > E["fa_chunk_min"] * 60 * RATE or k == len(keep) - 1:
            st = Stream(net, to_int16(np.concatenate(cur)), T)
            st.pcm = None
            out.append(dict(kind="noise", src="noise", hours=len(st.feats) * 0.02 / 3600, guard=[], words=[], st=st))
            cur, pos = [], 0
    return out


# ------------------------------------------------------------------ one setting
def run_setting(setting):
    """(thr, cls_off) -> metrics; uses the streams in G (forked)"""
    thr, off = setting
    off = list(off)
    res = dict(thr=thr, cls_off=off)
    rows = []
    for sc in G["scen"]:
        ev = detect(sc["st"].feats, sc["st"].dbfs, sc["st"].probs, sc["st"].frames, sc["st"].T, KEYWORD, thr,
                    X["hits"], X["hits_ms"], cls_off=off)
        trig = any(e & EV_SOS for t, e, k in ev)
        hit = [any(a <= t <= b + E["hit_after_ms"] for t, e, k in ev) for lab, a, b in sc["calls"]]
        rows.append((sc["set"], sc["n"], sc["cond"], trig, hit, [lab for lab, a, b in sc["calls"]]))
    trig = {}
    for (s, n, c), g in pd.DataFrame(rows, columns=["set", "n", "cond", "trig", "hit", "labs"]).groupby(["set", "n", "cond"]):
        trig[f"{s}|{n}|{c}"] = round(float(g.trig.mean()), 4)
    calls = Counter()
    for s, n, c, t, hit, labs in rows:
        for h, lab in zip(hit, labs):
            calls[(s, lab, "n")] += 1
            calls[(s, lab, "h")] += h
    res["trigger"] = trig
    res["call_hit"] = {f"{s}|{lab}": round(calls[(s, lab, "h")] / calls[(s, lab, "n")], 4)
                       for (s, lab, k) in calls if k == "n"}
    fa = {"speech": [0, 0, 0.0], "noise": [0, 0, 0.0]}
    words = Counter()
    for f in G["fa"]:
        st = f["st"]
        ev = detect(st.feats, st.dbfs, st.probs, st.frames, st.T, KEYWORD, thr, X["hits"], X["hits_ms"], cls_off=off)
        times = []
        for t, e, k in ev:
            if any(a <= t <= b for a, b in f["guard"]):
                continue
            fa[f["kind"]][0] += 1
            times = [u for u in times if t - u < X["hits_ms"]] + [t]
            if len(times) >= X["hits"]:
                fa[f["kind"]][1] += 1
                times = []
            if f["words"]:
                said = [w for w, a, b in f["words"] if b >= t - S["blame_s"] * 1000 and a <= t]
                words[(LABELS[k], " ".join(said[-S["blame_words"]:]))] += 1
            else:
                words[(LABELS[k], "<noise>")] += 1
        fa[f["kind"]][2] += f["hours"]
    hrs = sum(v[2] for v in fa.values())
    res["fa"] = {k: dict(hours=round(v[2], 2), false_hits=v[0], false_hits_per_h=round(v[0] / v[2], 3),
                         false_sos=v[1], false_sos_per_24h=round(24 * v[1] / v[2], 3)) for k, v in fa.items()}
    res["fa"]["all"] = dict(hours=round(hrs, 2), false_hits_per_h=round(sum(v[0] for v in fa.values()) / hrs, 3),
                            false_sos=sum(v[1] for v in fa.values()),
                            false_sos_per_24h=round(24 * sum(v[1] for v in fa.values()) / hrs, 3))
    res["fa_top"] = [dict(cls=c, said=w, n=n) for (c, w), n in words.most_common(S["blame_top"])]
    return res


def settings_grid():
    import itertools
    out = []
    names = list(S["sweep_off"])
    for thr in S["sweep_thr"]:
        for vals in itertools.product(*[S["sweep_off"][n] for n in names]):
            off = [0.0] * len(LABELS)
            for n, v in zip(names, vals):
                off[LABELS.index(n)] = v
            out.append((thr, tuple(off)))
    return out


def export_settings():
    off = [float(X["cls_off"].get(l, 0.0)) for l in LABELS]
    names = ["low", "medium", "high"]
    return {n: (t, tuple(off)) for n, t in zip(names, X["thr"])}


def summary(r):
    t = r["trigger"]
    m6 = np.mean([t[k] for k in t if k.split("|")[1] == "6"])
    return "thr %.2f off %s | trig@6 %s | all@6 %.3f | FA hits/h %.2f falseSOS/24h %.2f" % (
        r["thr"], r["cls_off"], {s: round(np.mean([t[k] for k in t if k.startswith(s + "|6|")]), 3) for s in S["sets"]
                                  if any(k.startswith(s + "|6|") for k in t)}, m6,
        r["fa"]["all"]["false_hits_per_h"], r["fa"]["all"]["false_sos_per_24h"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(MODELS, "best.pt"))
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--sweep", action="store_true", help="try the thr x cls_off grid of eval_sys (val only)")
    ap.add_argument("--quick", action="store_true", help="smoke test: 2 scenarios each, first false-alarm stream only")
    a = ap.parse_args()
    if a.quick:
        S["scenarios"] = 2
        G["quick"] = True
    torch.set_num_threads(S["torch_threads"])
    t0 = time.time()
    net = load_net(a.model)
    import hashlib
    import pickle
    key = hashlib.md5(open(a.model, "rb").read() + json.dumps([E, S, a.split, a.quick], sort_keys=True,
                                                               default=str).encode()).hexdigest()[:12]
    cpath = os.path.join(CACHE, "evalcache_%s.pkl" % key)
    if os.path.exists(cpath):
        G.update(pickle.load(open(cpath, "rb")))
        print("streams from", cpath, flush=True)
    else:
        D = Data(C)
        G["scen"] = build_scenarios(net, D, a.split)
        # the cached CNN outputs give exactly the windows Detector asks for
        sc = G["scen"][0]["st"]
        detect(sc.feats, sc.dbfs, sc.probs, sc.frames, sc.T, KEYWORD, 0.5, X["hits"], X["hits_ms"], check=True)
        print("scenarios: %d (%.0f s)" % (len(G["scen"]), time.time() - t0), flush=True)
        G["fa"] = build_fa(net, D, a.split)
        print("false-alarm streams: %.1f h (%.0f s)" % (sum(f["hours"] for f in G["fa"]), time.time() - t0), flush=True)
        # Detector only stores the feature rows and hands the window to run_fn, which uses the
        # cached CNN outputs instead -> the rows can be zeros (saves memory, same decisions)
        for x in G["scen"] + G["fa"]:
            x["st"].feats = np.broadcast_to(np.zeros(40, np.float32), x["st"].feats.shape)
        pickle.dump({k: G[k] for k in ("scen", "fa")}, open(cpath, "wb"))
    sets = settings_grid() if a.sweep else list(export_settings().values())
    with Pool(S["procs"]) as pool:
        res = pool.map(run_setting, sets, chunksize=1)
    out = dict(split=a.split, model=os.path.basename(a.model), scenarios=len(G["scen"]),
               fa_hours={k: round(sum(f["hours"] for f in G["fa"] if f["kind"] == k), 2) for k in ("speech", "noise")},
               results=res)
    if not a.sweep:
        out["sensitivity"] = dict(zip(export_settings().keys(), res))
    name = "eval_sys_%s%s%s.json" % (a.split, "_sweep" if a.sweep else "", "_quick" if a.quick else "")
    json.dump(out, open(os.path.join(MODELS, name), "w"), indent=1)
    for r in res:
        print(summary(r))
    plot(out, a.split, a.sweep)
    print("done in %.0f min" % ((time.time() - t0) / 60))


def plot(out, split, sweep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = out["results"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    groups = {}
    for r in res:
        groups.setdefault(tuple(r["cls_off"]), []).append(r)
    for off, rs in groups.items():
        rs = sorted(rs, key=lambda r: r["thr"])
        lab = "cls_off help %+.2f sos %+.2f" % (off[LABELS.index("help")], off[LABELS.index("sos")])
        x = [r["fa"]["all"]["false_hits_per_h"] for r in rs]
        y = [100 * np.mean([v for k, v in r["trigger"].items() if k.split("|")[1] == "6"]) for r in rs]
        ax[0].plot(x, y, "o-", ms=3, label=lab)
        x = [r["fa"]["all"]["false_sos_per_24h"] for r in rs]
        ax[1].plot(x, y, "o-", ms=3, label=lab)
    ax[0].set_xscale("symlog", linthresh=0.1)
    ax[0].set_xlabel("false single hits per hour")
    ax[1].set_xscale("symlog", linthresh=0.1)
    ax[1].set_xlabel("false SOS per 24 h")
    ax[1].axvline(S["target_false_sos_24h"], color="k", lw=.8)
    for a_ in ax:
        a_.set_ylabel("trigger rate % (6 calls in 20 s, all sets)")
        a_.axhline(100 * S["target_trigger_6"], color="k", lw=.8)
        a_.grid(alpha=.3)
    ax[0].legend(fontsize=6)
    fig.suptitle("system level, kws.detect.Detector (%s, %.0f h false-alarm audio)" % (split, sum(out["fa_hours"].values())))
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "sys_trigger_vs_fa_%s%s.png" % (split, "_sweep" if sweep else "")), dpi=130)


if __name__ == "__main__":
    main()
