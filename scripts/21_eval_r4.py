"""21_eval_r4.py - Phase C evaluation with the FIRMWARE v1.6.7 detector (kws/fwdet.py).

    python scripts/21_eval_r4.py --model models/kw_r3.pt --split test --tag r3      # "before"
    python scripts/21_eval_r4.py --model models/kw_r4.pt --split val --sweep        # choose thresholds (val only)
    python scripts/21_eval_r4.py --model models/kw_r4.pt --split test --tag r4 --from-export export_r4
    python scripts/21_eval_r4.py --native out/kws_model.c --split test --tag r4_int8 # the exported int8 model, C engine
    python scripts/21_eval_r4.py --plot r3 r4                                       # docs/kw_r4_*.png from the two tests
    python scripts/21_eval_r4.py --model models/kw_r4.pt --split val --sweep --rule 1   # the "3 calls in 20 s" setting
    (--quick: a small smoke test of everything)

Every stream goes through the C detector exactly as Svc/kws.c feeds it (eval_r4.device:
medium, 2 calls within 10 s with words mode, single "help" +0.05 and no relaxation).
Measured (split = val or test; nothing of test is ever used for a choice):
  calls   one call: the clip with eval_r4.lead_s of microphone noise before and tail_s after
          -> heard (any keyword) / heard as the right keyword; pairs: two calls of the same
          set (and speaker) eval_r4.pair_gap_s apart -> SOS. Sets: own_help / own_help_me /
          own_sos (the owner's recordings, as recorded), real_help (MSWC + LibriSpeech),
          real_help_me (LibriSpeech), tts_sos (Kokoro) - the public ones at call_level_dbfs.
  talk    AMI meetings (English conversation), AISHELL-1 (Mandarin), LibriSpeech (the old
          false-alarm speech): false hits per hour, false SOS per 24 h; half of the streams with
          test noise (fa_noise_prob). A hit within keyword_guard_s of a real "help" is not false.
  vocal   coughs, throat clearing, sneezes, sniffs, sighs, laughter (VocalSound + ESC-50) in a
          row with pauses: false hits per 100 sounds, by type, and false SOS.
Results: models/eval_r4_<tag>_<split>.json (+ _sweep_val.json with the recommended setting).
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kws import augment as aug  # noqa: E402
from kws.audio import speech_span, to_float, to_int16  # noqa: E402
from kws.dataset import cut_tail  # noqa: E402
from kws.cfg import load, rng_for, stable_hash  # noqa: E402
from kws.fwdet import EV_HIT, EV_SOS, FwDet  # noqa: E402
from kws.model import KwsNet  # noqa: E402

C = load()
E = C["eval_r4"]
P = C["r4_prepare"]
CACHE, MODELS, DOCS = C["paths"]["cache"], C["paths"]["models"], C["paths"]["docs"]
LABELS = C["labels"]
RATE = 16000
G = {}                                    # shared with the worker processes (fork)


# ------------------------------------------------------------------ data
def store(name):
    p = os.path.join(CACHE, name + ".csv")
    if not os.path.exists(p):
        return None, None
    return pd.read_csv(p, keep_default_na=False), np.load(os.path.join(CACHE, name + ".npy"), mmap_mode="r")


def floor_noise(rng, n):
    return rng.standard_normal(n).astype(np.float32) * 10 ** (E["noise_floor_dbfs"] / 20)


def call_sets(split, quick):
    """{set: [(audio float, label, speaker)]}"""
    out = {}
    own, own_a = store("own")
    if own is not None:
        for lab in ("help", "help me", "sos"):
            d = own[(own.label == lab) & (own.split == split)]
            out["own_" + lab.replace(" ", "_")] = [(to_float(np.array(own_a[r.offset:r.offset + r.length])), lab, r.speaker, False)
                                                   for r in d.itertuples()]
    cl, cl_a = store("clips")
    if cl is not None:
        for name, lab, groups in (("real_help", "help", ["mswc_help", "libri_help"]),
                                  ("real_help_me", "help me", ["libri_help_me"]), ("tts_sos", "sos", ["tts"])):
            d = cl[(cl.group.isin(groups)) & (cl.label == lab) & (cl.split == split)]
            d = d.loc[sorted(d.index, key=lambda i: stable_hash("evr4:%d" % i))[:E["max_per_set"]]]
            out[name] = [(to_float(np.array(cl_a[r.offset:r.offset + r.length])), lab, r.speaker, True) for r in d.itertuples()]
    # r4b: the same "help" calls with the end of the word cut off ("hel", no final /p/): every hit as "help" is false
    Q = C.get("cut_help")
    if Q:
        for src in ("own_help", "real_help"):
            cut = []
            for i, (x, lab, spk, public) in enumerate(out.get(src, [])):
                s0, e0 = speech_span(x, C["prepare"]["span_rel_db"])
                y, _, _ = cut_tail(x, s0, e0, np.random.default_rng(stable_hash("evcut:%s:%d" % (src, i))), Q)
                cut.append((y, lab, spk, public))
            out["cut_" + src] = cut
    if quick:
        out = {k: v[:6] for k, v in out.items()}
    return {k: v for k, v in out.items() if v}


def talk_streams(split, quick):
    """[(name, source, audio int16, guard spans [(a, b)] samples, noisy)]"""
    out = []
    fa, fa_a = store("fa4_" + split)
    ev = pd.read_csv(os.path.join(CACHE, "fa4_%s_events.csv" % split)) if fa is not None else None
    guard = int(P["keyword_guard_s"] * RATE)
    if fa is not None:
        for r in fa.itertuples():
            if r.source == "vocal":
                continue
            g = [(a - guard, b + guard) for a, b in ev[(ev.stream == r.stream) & (ev.kind == "keyword")][["at", "end"]].values]
            out.append((r.stream, r.source, (r.offset, r.length, "fa4_" + split), g))
    # the old false-alarm speech (LibriSpeech dev / test + held-out speakers), in streams of ~30 min
    old = "fa_dev" if split == "val" else "fa_test"
    cap = E["libri_hours"][split] * 3600 * RATE
    for name in (old, old + "_extra"):
        d, _ = store(name)
        if d is None:
            continue
        words = pd.read_csv(os.path.join(CACHE, name + "_words.csv"), keep_default_na=False)
        kw = words[words.word.isin(C["sources"]["librispeech"]["keyword_words"])][["at", "end"]].values
        pos, n, k = 0, 0, 0
        total = int(d.length.sum())
        while pos < total and n < cap:
            ln = min(30 * 60 * RATE, total - pos, int(cap - n))
            g = [(a - guard - pos, b + guard - pos) for a, b in kw if a >= pos and a < pos + ln]
            out.append(("%s-%d" % (name, k), "libri", (pos, ln, name), g))
            pos += ln
            n += ln
            k += 1
    for i, s in enumerate(out):
        out[i] = s + (stable_hash("noisy:" + s[0]) % 1000 < 1000 * E["fa_noise_prob"],)
    if quick:
        out = [(a, b, (c[0], min(c[1], 60 * RATE), c[2]), d, e) for a, b, c, d, e in out][:4]
    return out


def vocal_streams(split, quick):
    fa, _ = store("fa4_" + split)
    if fa is None:
        return []
    ev = pd.read_csv(os.path.join(CACHE, "fa4_%s_events.csv" % split))
    out = []
    for r in fa[fa.source == "vocal"].itertuples():
        e = ev[(ev.stream == r.stream) & (ev.kind == "vocal")]
        out.append((r.stream, (r.offset, r.length, "fa4_" + split), list(zip(e.type, e["at"], e["end"]))))
    return out[:3] if quick else out


# ------------------------------------------------------------------ the detector in a worker
def make_det(setting):
    net = G.get("net")
    if G.get("native"):
        return FwDet(native=True, model_c=G["native"], thr=setting.get("thr"), cls_off=setting.get("cls_off"),
                     sens=E["device"]["sens"], rule=setting.get("rule", E["device"]["rule"]), help_extra=setting["help_extra"])
    return FwDet(net, thr=setting["thr"], cls_off=setting["cls_off"], sens=E["device"]["sens"],
                 rule=setting.get("rule", E["device"]["rule"]), help_extra=setting["help_extra"])


def audio_of(ref):
    off, ln, name = ref
    if name not in G["arrays"]:
        G["arrays"][name] = np.load(os.path.join(CACHE, name + ".npy"), mmap_mode="r")
    return np.array(G["arrays"][name][off:off + ln])


def with_noise(x, rng):
    """test noise at an SNR relative to the stream's loud part"""
    nd, na = G["noise"]
    y = to_float(x)
    r = nd.iloc[rng.integers(len(nd))]
    n = to_float(np.array(na[r.offset:r.offset + r.length]))
    n = np.tile(n, len(y) // max(1, len(n)) + 1)[:len(y)]
    loud = y[np.abs(y) > np.percentile(np.abs(y), 90)] if len(y) else y
    return to_int16(aug.add_noise(y, n, rng.uniform(*E["fa_noise_snr_db"]), loud if len(loud) else y))


def job(arg):
    kind, setting, payload = arg
    det = make_det(setting)
    rng = rng_for(C, "evr4", kind, str(payload[0]), str(payload[-1]))
    if kind == "calls":                              # payload: (set name, [items], pairs or None, chunk)
        name, items, pairs, _ = payload
        res = []
        lead, tail = int(E["lead_s"] * RATE), int(E["tail_s"] * RATE)
        groups = [[it] for it in items] if not pairs else pairs
        for grp in groups:
            parts = [floor_noise(rng, lead)]
            for j, (x, lab, spk, public) in enumerate(grp):
                if public:
                    x = aug.set_rms(x, rng.uniform(*E["call_level_dbfs"]))
                if j:
                    parts.append(floor_noise(rng, int(rng.uniform(*E["pair_gap_s"]) * RATE)))
                parts.append(x + floor_noise(rng, len(x)))
            parts.append(floor_noise(rng, tail))
            ev = det.run(to_int16(np.concatenate(parts)))
            det.reset()
            lab = grp[0][1]
            hits = [(t, c, s) for t, f, c, s in ev if f & EV_HIT]
            res.append(dict(heard=bool(hits), right=any(LABELS[c] == lab for _, c, _ in hits),
                            sos=any(f & EV_SOS for _, f, _, _ in ev), n_hits=len(hits),
                            score=max([s for _, _, s in hits], default=0.0)))
        return kind, name, res
    if kind == "talk":
        name, src, ref, guards, noisy = payload
        x = audio_of(ref)
        if noisy:
            x = with_noise(x, rng)
        ev = det.run(x)
        fh, fs = 0, 0
        for t, f, c, s in ev:
            at = t * RATE // 1000
            if any(a <= at <= b for a, b in guards):
                continue
            fh += bool(f & EV_HIT)
            fs += bool(f & EV_SOS)
        return kind, name, dict(source=src, hours=len(x) / RATE / 3600, false_hits=fh, false_sos=fs, noisy=bool(noisy))
    if kind == "vocal":
        name, ref, evs = payload
        x = audio_of(ref)
        ev = det.run(x)
        per = {}
        for typ, a, b in evs:
            d = per.setdefault(typ, dict(sounds=0, false_hits=0))
            d["sounds"] += 1
        fs = 0
        for t, f, c, s in ev:
            at = t * RATE // 1000
            if f & EV_SOS:
                fs += 1
            if not f & EV_HIT:
                continue
            owner = [typ for typ, a, b in evs if a <= at <= b + int(1.8 * RATE)]
            typ = owner[-1] if owner else "between"
            d = per.setdefault(typ, dict(sounds=0, false_hits=0))
            d["false_hits"] += 1
        return kind, name, dict(per_type=per, false_sos=fs)
    raise ValueError(kind)


def init_worker(model_path, native):
    torch.set_num_threads(1)
    G["arrays"] = {}
    G["native"] = native
    if model_path:
        ck = torch.load(model_path, map_location="cpu", weights_only=False)
        net = KwsNet(**dict(ck["cfg"]))
        net.load_state_dict(ck["state"], strict=False)
        G["net"] = net.eval()
    nd = pd.read_csv(os.path.join(CACHE, "noise_test.csv"))
    G["noise"] = (nd, np.load(os.path.join(CACHE, "noise_test.npy"), mmap_mode="r"))


# ------------------------------------------------------------------ one setting
def evaluate(setting, split, quick, pool):
    sets = G["sets"]
    jobs = []
    for name, items in sets.items():
        for k in range(0, len(items), 25):
            jobs.append(("calls", setting, (name, items[k:k + 25], None, k)))
        # pairs: two calls of the same speaker (own) / of the same set (public), hash order
        by = {}
        for it in items:
            by.setdefault(it[2] if name.startswith("own") else "all", []).append(it)
        # a person calling repeatedly: as many calls as the rule needs for an SOS
        # (rule 0: 2 within 10 s -> "@pair"; rule 1: 3 within 20 s -> 4 calls, "@x4")
        rule = setting.get("rule", E["device"]["rule"])
        n_seq = E["sos_calls"][rule]
        tag = "@pair" if n_seq == 2 else "@x%d" % n_seq
        pairs = []
        for spk, lst in sorted(by.items()):
            lst = sorted(lst, key=lambda it: stable_hash("pair:%d" % len(it[0])))
            pairs += [tuple(lst[i:i + n_seq]) for i in range(0, len(lst) - n_seq + 1, n_seq)]
        pairs = pairs[:E["pairs_per_set"] if not quick else 3]
        for k in range(0, len(pairs), 20):
            jobs.append(("calls", setting, (name + tag, None, pairs[k:k + 20], k)))
    jobs += [("talk", setting, s) for s in G["talk"]]
    jobs += [("vocal", setting, s) for s in G["vocal"]]
    calls, talk, vocal = {}, [], []
    for kind, name, res in pool.imap_unordered(job, jobs):
        if kind == "calls":
            calls.setdefault(name, []).extend(res)
        elif kind == "talk":
            talk.append(res)
        else:
            vocal.append(res)
    return summarise(setting, calls, talk, vocal)


def summarise(setting, calls, talk, vocal):
    out = dict(setting=setting, calls={}, talk={}, vocal={})
    for name, res in sorted(calls.items()):
        if "@" in name:
            out["calls"][name] = dict(n=len(res), sos_rate=round(float(np.mean([r["sos"] for r in res])), 4))
        else:
            sc = [r["score"] for r in res]
            out["calls"][name] = dict(n=len(res), heard=round(float(np.mean([r["heard"] for r in res])), 4),
                                      right=round(float(np.mean([r["right"] for r in res])), 4),
                                      score_median=round(float(np.median(sc)), 3),      # highest hit score of a call (0 = no hit)
                                      score_p90=round(float(np.percentile(sc, 90)), 3))
    for src in sorted({t["source"] for t in talk}) + ["all"]:
        tt = [t for t in talk if src == "all" or t["source"] == src]
        h = sum(t["hours"] for t in tt)
        if h <= 0:
            continue
        fh, fs = sum(t["false_hits"] for t in tt), sum(t["false_sos"] for t in tt)
        out["talk"][src] = dict(hours=round(h, 2), false_hits=fh, false_sos=fs, false_hits_per_h=round(fh / h, 3),
                                false_sos_per_24h=round(24 * fs / h, 3))
    per = {}
    fs = 0
    for v in vocal:
        fs += v["false_sos"]
        for typ, d in v["per_type"].items():
            p = per.setdefault(typ, dict(sounds=0, false_hits=0))
            p["sounds"] += d["sounds"]
            p["false_hits"] += d["false_hits"]
    n = sum(p["sounds"] for p in per.values())
    for typ, p in per.items():
        p["per_100"] = round(100.0 * p["false_hits"] / max(1, p["sounds"]), 3) if p["sounds"] else None
    out["vocal"] = dict(sounds=n, false_hits=sum(p["false_hits"] for p in per.values()), false_sos=fs,
                        false_hits_per_100=round(100.0 * sum(p["false_hits"] for p in per.values()) / max(1, n), 3),
                        false_sos_per_100=round(100.0 * fs / max(1, n), 3), per_type=per)
    # cut_* sets ("hel") are false accepts, not calls: not in the mean hit rate
    singles = [v["heard"] for k, v in out["calls"].items() if "@" not in k and not k.startswith("cut_")]
    out["mean_heard"] = round(float(np.mean(singles)), 4) if singles else None
    return out


def show(r):
    s = r["setting"]
    print("thr %s help_extra %.2f  mean heard %s" % (s.get("thr"), s["help_extra"], r["mean_heard"]), flush=True)
    for k, v in r["calls"].items():
        print("   %-18s %s" % (k, v), flush=True)
    for k, v in r["talk"].items():
        print("   talk %-10s %s" % (k, v), flush=True)
    v = r["vocal"]
    print("   vocal: %d sounds, %.2f false hits /100, %.2f false SOS /100  %s" % (
        v["sounds"], v["false_hits_per_100"], v["false_sos_per_100"],
        {t: p["per_100"] for t, p in v["per_type"].items()}), flush=True)


# ------------------------------------------------------------------ plots
def plot(tags):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = [json.load(open(os.path.join(MODELS, "eval_r4_%s_test.json" % t))) for t in tags]
    # 1. calls
    names = [k for k in R[0]["calls"] if "@" not in k and not k.startswith("cut_")]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    w = 0.8 / len(R)
    for i, (t, r) in enumerate(zip(tags, R)):
        ax[0].bar(np.arange(len(names)) + i * w, [100 * r["calls"][n]["heard"] for n in names], w, label=t)
    ax[0].set_xticks(np.arange(len(names)) + w * (len(R) - 1) / 2, names, rotation=30, ha="right")
    ax[0].set_ylabel("one call heard (%)"), ax[0].set_ylim(0, 100), ax[0].legend(), ax[0].set_title("real calls (test)")
    src = [k for k in R[0]["talk"]]
    for i, (t, r) in enumerate(zip(tags, R)):
        ax[1].bar(np.arange(len(src)) + i * w, [r["talk"].get(s, {}).get("false_hits_per_h", 0) for s in src], w, label=t)
    ax[1].set_xticks(np.arange(len(src)) + w * (len(R) - 1) / 2, src)
    ax[1].set_ylabel("false hits per hour"), ax[1].legend(), ax[1].set_title("people talking (test)")
    typs = sorted(R[0]["vocal"]["per_type"])
    for i, (t, r) in enumerate(zip(tags, R)):
        ax[2].bar(np.arange(len(typs)) + i * w, [r["vocal"]["per_type"].get(y, {}).get("per_100") or 0 for y in typs], w, label=t)
    ax[2].set_xticks(np.arange(len(typs)) + w * (len(R) - 1) / 2, typs, rotation=30, ha="right")
    ax[2].set_ylabel("false hits per 100 sounds"), ax[2].legend(), ax[2].set_title("coughs and other vocal sounds (test)")
    fig.suptitle("firmware v1.6.7 detector, medium sensitivity, 2 calls in 10 s: " + " vs ".join(tags))
    fig.tight_layout()
    png = "kw_r4_compare_test.png" if list(tags) == ["r3", "r4"] else "kw_%s_compare_test.png" % "_".join(tags)
    fig.savefig(os.path.join(DOCS, png), dpi=130)
    print("docs/" + png)
    # r4b: "hel" (help with the end cut off) heard as help, and "hel ... hel" -> SOS
    cuts = [k for k in R[0]["calls"] if k.startswith("cut_") and "@" not in k]
    if cuts and all(k in r["calls"] for r in R for k in cuts):
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, (t, r) in enumerate(zip(tags, R)):
            ax.bar(np.arange(len(cuts)) + i * w, [100 * r["calls"][k]["right"] for k in cuts], w, label=t)
        ax.set_xticks(np.arange(len(cuts)) + w * (len(R) - 1) / 2, cuts)
        ax.set_ylabel('heard as "help" (%)'), ax.legend(), ax.set_title('"help" with the end cut off ("hel"), test')
        fig.tight_layout()
        png = "kw_%s_cut_help_test.png" % "_".join(tags)
        fig.savefig(os.path.join(DOCS, png), dpi=130)
        print("docs/" + png)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--native", help="kws_model.c: the int8 C engine instead of the PyTorch model")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--thr", type=float, nargs=3, help="low medium high (default: export: of r3)")
    ap.add_argument("--cls-off", type=float, nargs=4, help="per class (default: export:)")
    ap.add_argument("--help-extra", type=float, default=None)
    ap.add_argument("--from-export", default="export", help="config section for thr / cls_off / help_extra (export = r3, export_r4)")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--rule", type=int, choices=[0, 1, 2], default=None,
                    help="0 = 2 calls within 10 s (words mode, v1.6.7 default), 1 = 3 calls within 20 s, 2 = 2 calls within 10 s, words mode off (default: eval_r4.device.rule)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plot", nargs="+")
    a = ap.parse_args()
    if a.plot:
        return plot(a.plot)
    if a.sweep and a.split != "val":
        sys.exit("--sweep only on val")
    X = C[a.from_export]
    thr = a.thr or X["thr"]
    off = a.cls_off or [X["cls_off"].get(l, 0.0) for l in LABELS]
    hx = a.help_extra if a.help_extra is not None else X.get("help_extra", E["device"]["help_extra"])
    rule = E["device"]["rule"] if a.rule is None else a.rule
    t0 = time.time()
    from kws.fwdet import build, lib_path
    if not os.path.exists(lib_path()):
        build()
    if a.native:
        build(a.native)                              # always: the model file may have changed
    G["sets"] = call_sets(a.split, a.quick)
    G["talk"] = talk_streams(a.split, a.quick)
    G["vocal"] = vocal_streams(a.split, a.quick)
    print("sets: %s; talk %d streams; vocal %d streams" % ({k: len(v) for k, v in G["sets"].items()}, len(G["talk"]),
                                                           len(G["vocal"])), flush=True)
    pool = Pool(E["procs"] if not a.quick else 2, initializer=init_worker, initargs=(a.model, a.native))
    tag = a.tag or (os.path.splitext(os.path.basename(a.model or a.native))[0])
    if a.sweep:
        rows = []
        for t in E["sweep_med_thr"]:
            for h in E["sweep_help_extra"]:
                st = dict(thr=[t + E["low_minus_med"], t, t + E["high_minus_med"]], cls_off=off, help_extra=h, rule=rule)
                r = evaluate(st, a.split, a.quick, pool)
                show(r)
                rows.append(r)
        ok = [r for r in rows if r["talk"].get("all", {}).get("false_sos_per_24h", 0) <= E["target_false_sos_24h"]]
        if ok:
            best = max(ok, key=lambda r: (r["mean_heard"] or 0, r["setting"]["thr"][1]))
        else:   # no row meets the target: the fewest false SOS (ties: the higher hit rate) - never the noisiest row
            best = min(rows, key=lambda r: (r["talk"].get("all", {}).get("false_sos_per_24h", 1e9), -(r["mean_heard"] or 0)))
        out = dict(model=a.model, split=a.split, rows=rows, recommended=best["setting"],
                   target_met=bool(ok), rule_calls=rule,
                   rule="highest mean single-call hit rate on val with talk false SOS <= %.1f / 24 h (ties: higher thr); "
                        "if no row meets it: the fewest talk false SOS (ties: higher hit rate)"
                        % E["target_false_sos_24h"], minutes=round((time.time() - t0) / 60, 1))
        path = os.path.join(MODELS, "eval_r4_sweep_val%s%s.json" % ("" if rule == 0 else "_rule%d" % rule,
                                                                   "_" + a.tag if a.tag else ""))
        json.dump(out, open(path, "w"), indent=1)
        print("recommended:", best["setting"], "->", path)
    else:
        st = dict(thr=None if (a.native and not a.thr) else thr, cls_off=None if (a.native and not a.cls_off) else off,
                  help_extra=hx, rule=rule)
        r = evaluate(st, a.split, a.quick, pool)
        show(r)
        r.update(model=a.model or a.native, split=a.split, minutes=round((time.time() - t0) / 60, 1))
        path = os.path.join(MODELS, "eval_r4_%s_%s.json" % (tag, a.split))
        json.dump(r, open(path, "w"), indent=1)
        print("->", path)
    pool.close()


if __name__ == "__main__":
    main()
