"""48_eval_spk_d.py - Phase D: every speaker model on the SAME trials (public speakers + the owner's recordings).

    python scripts/48_eval_spk_d.py --split val                 # old, big120 (if present), d0, d1, d2 (those trained)
    python scripts/48_eval_spk_d.py --split test
    python scripts/48_eval_spk_d.py --split test --models old d2

Per model: device conditions clean / far / shout / noisy, imitation mimic_f2m / mimic_m2f (pitch moved
to the other gender, formants kept) and plain_f2m / plain_m2f (the same takes unchanged), each with
the scoring methods of spk/phase_d.py (cls, all, max = firmware now, asn = AS-norm, pair = two calls
averaged, pasn); EER, false accepts at 5 % owner rejection. The owner's real recordings: own-A
(owner, 10 NORMAL takes enrolled, 30 random enrolments) vs own-B (the woman who imitated him):
EER, score percentiles, owner acceptance per style, imitator SOS chance within 2 / 6 / 10 / 20 tries.
Firmware thresholds: for every model the thresholds whose same-gender false-accept rate (val / test,
clean, max) equals the old model's at 0.46 / 0.52 / 0.60.
Writes models/spk_d_eval_<split>.json, docs/spk_d_<split>.png, docs/spk_d_own_<split>.png.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from spk import phase_d  # noqa: E402
from spk.data import load_cfg  # noqa: E402
from spk.export import load as load_spk  # noqa: E402
from spk.voices import Voices  # noqa: E402

C, P = load_cfg()
Q = P["phase_d"]
E = Q["eval"]
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MODELS, DOCS = C["paths"]["models"], C["paths"]["docs"]


def model_paths(names):
    out = {}
    for n in names:
        if n == "old":
            p = os.path.join(ROOT, E["old_model"])
        elif n == "big120":
            p = os.path.join(ROOT, E["big_model"])
        else:
            p = os.path.join(MODELS, "spk_%s.pt" % n)
        if os.path.exists(p):
            out[n] = p
        else:
            print("skip %s: %s not found" % (n, p), flush=True)
    return out


def fw_thresholds(sc_old, sc_new, thr_old):
    """new-model thresholds with the same same-gender false-accept rate (clean, max) as the old model at thr_old"""
    io = sc_old[(sc_old.cond == "clean") & (sc_old.kind == "same_gender")]["max"].dropna().to_numpy()
    im = sc_new[(sc_new.cond == "clean") & (sc_new.kind == "same_gender")]["max"].dropna().to_numpy()
    out = {}
    for t in thr_old:
        far = float(np.mean(io >= t))
        out["%.2f" % t] = dict(far_same_gender=round(far, 4), new_threshold=round(float(np.quantile(im, 1 - far)), 4))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--models", nargs="*", default=["old", "big120"] + sorted(Q["variants"]))
    ap.add_argument("--speakers", type=int, help="override (smoke test)")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    V = Voices(C)
    n = a.speakers or E["speakers_%s" % a.split]
    tr = phase_d.make_trials(V, P, a.split, n)
    cw = phase_d.cohort_windows(V, P, E["cohort_speakers"])
    OX, om = phase_d.own_windows(V, P)
    print("%s: %d speakers (%d female), %d windows; cohort %d; own takes %s (%.0f s)" % (
        a.split, len(tr.names), sum(g == "FEMALE" for g in tr.gender.values()), len(tr.X), len(cw),
        om.speaker.value_counts().to_dict(), time.time() - t0), flush=True)
    res, scs, owns = {}, {}, {}
    for name, path in model_paths(a.models).items():
        try:
            net = load_spk(path)
        except Exception as ex:  # noqa: BLE001 - a checkpoint of another format (e.g. big120 from the gpu-train branch)
            print("skip %s: cannot load %s (%s)" % (name, path, ex), flush=True)
            continue
        t1 = time.time()
        coh = phase_d.cohort_vectors(net, cw, dev)
        summ, sc = phase_d.score_all(tr, phase_d.embed(net, tr.X, dev), coh, P)
        own, osc = phase_d.own_eval(net, OX, om, coh, P, dev)
        res[name] = dict(path=os.path.relpath(path, ROOT), macs=int(net.macs()), select_metric=round(phase_d.select_metric(summ), 4),
                         select_metric_asn=round(phase_d.select_metric(summ, "asn"), 4), public=summ, own=own)
        scs[name], owns[name] = sc, osc
        line = {c: summ[c]["max"]["eer"] for c in ("clean", "far", "shout", "noisy", "mimic_f2m", "plain_f2m") if c in summ}
        print(name, "select %.4f" % res[name]["select_metric"], json.dumps(line),
              "own EER max %s / pair %s" % (own.get("max", {}).get("eer"), own.get("pair", {}).get("eer")),
              "(%.0f s)" % (time.time() - t1), flush=True)
    if "old" in scs:
        for name in scs:
            if name != "old":
                res[name]["firmware_thresholds"] = fw_thresholds(scs["old"], scs[name], E["firmware_thresholds"])
    cand = {k: v["select_metric"] for k, v in res.items() if k in Q["variants"]}
    if cand and a.split == "val":
        res["best_variant_val"] = min(cand, key=cand.get)
    res["split"], res["speakers"] = a.split, len(tr.names)
    json.dump(res, open(os.path.join(MODELS, "spk_d_eval_%s.json" % a.split), "w"), indent=1)
    plot(res, a.split)
    plot_own(owns, res, a.split)
    print("done in %.0f min -> models/spk_d_eval_%s.json" % ((time.time() - t0) / 60, a.split), flush=True)


def plot(res, split):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [k for k in res if isinstance(res[k], dict) and "public" in res[k]]
    conds = ["clean", "far", "shout", "noisy", "mimic_f2m", "plain_f2m", "mimic_m2f"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
    w = 0.8 / max(1, len(names))
    for i, n in enumerate(names):
        s = res[n]["public"]
        ax[0].bar(np.arange(len(conds)) + i * w, [100 * s.get(c, {}).get("max", {}).get("eer", np.nan) for c in conds], w, label=n)
    ax[0].set_xticks(np.arange(len(conds)) + w * (len(names) - 1) / 2, conds, fontsize=8)
    ax[0].set_ylabel("EER % (firmware scoring: max)"), ax[0].legend(fontsize=7), ax[0].grid(alpha=.3, axis="y")
    meths = list(phase_d.METHODS)
    for i, n in enumerate(names):
        s = res[n]["public"]
        ax[1].bar(np.arange(len(meths)) + i * w,
                  [100 * np.nanmean([s.get(c, {}).get(m, {}).get("eer", np.nan) for c in ("clean", "far", "shout", "noisy", "mimic_f2m")])
                   for m in meths], w, label=n)
    ax[1].set_xticks(np.arange(len(meths)) + w * (len(names) - 1) / 2, meths)
    ax[1].set_ylabel("mean EER % (4 conditions + imitation)"), ax[1].set_xlabel("scoring"), ax[1].grid(alpha=.3, axis="y")
    fig.suptitle("speaker models, %s speakers (%d)" % (split, res["speakers"]))
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "spk_d_%s.png" % split), dpi=130)


def plot_own(owns, res, split):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [n for n, sc in owns.items() if sc is not None]
    if not names:
        return
    fig, ax = plt.subplots(2, len(names), figsize=(3.6 * len(names), 6), squeeze=False)
    for j, n in enumerate(names):
        sc = owns[n]
        for i, meth in enumerate(("max", "pair")):
            for who, col in (("owner", "C2"), ("imitator", "C3")):
                v = sc[sc.who == who][meth].dropna()
                ax[i, j].hist(v, bins=40, alpha=.55, density=True, color=col, label=who)
            r = res[n]["own"].get(meth, {})
            if r:
                ax[i, j].axvline(r["thr_at_frr"], color="k", lw=.8)
                ax[i, j].set_title("%s / %s: EER %.1f%%, imitator SOS in 10 tries %.0f%%" % (
                    n, meth, 100 * r["eer"], 100 * r["imitator_sos_within"].get("10", np.nan)), fontsize=8)
            ax[i, j].grid(alpha=.3)
    ax[0, 0].legend(fontsize=7)
    fig.suptitle("owner (own-A) vs the woman imitating him (own-B), device recordings; line = 5 % owner rejection")
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "spk_d_own_%s.png" % split), dpi=120)


if __name__ == "__main__":
    main()
