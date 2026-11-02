"""46_compare_spk.py - old (spk-20261023) vs new speaker model on the SAME test takes (phase B2).

    python scripts/46_compare_spk.py [--split test]

Device protocol (spk/eval_dev.py): 10 enrolment takes, a template per keyword
class + the template of all takes, cosine; test takes clean / far (0.8-1 m,
echo) / shouted (clipping) / noisy; impostors of the same and the other gender.
The windows are generated from fixed seeds, so both models see identical audio.
Reports per model and condition: EER, false reject / false accept at the
firmware "Normal" threshold (0.52), score percentiles, and for the new model
the threshold that keeps its same-gender false-accept rate at the old model's
value (false reject compared there too). Decision (finetune.deliver_if): the
new model is delivered only if it is clearly better.
Writes models/spk_compare_<split>.json and docs/spk_compare_<split>.png.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from spk import eval_dev  # noqa: E402
from spk.data import load_cfg  # noqa: E402
from spk.export import load as load_spk  # noqa: E402
from spk.voices import Voices  # noqa: E402

C, P = load_cfg()
FT = P["finetune"]
E = FT["eval"]
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CONDS = ["clean", "far", "shout", "noisy"]


def matched_thr(sc, far_target, col="score_cls"):
    """threshold at which the same-gender false-accept rate equals far_target"""
    im = np.sort(sc[sc.kind == "same_gender"][col].to_numpy())
    return float(np.quantile(im, 1 - far_target)) if len(im) else 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(4)
    t0 = time.time()
    V = Voices(C)
    res, scs = {}, {}
    for name, path in (("old", FT["init"]), ("new", FT["out"])):
        net = load_spk(os.path.join(ROOT, path))
        tm, tests, g = eval_dev.build(net, V, P, a.split, E["speakers"], CONDS, tag="evB2" + a.split)
        sc = eval_dev.scores(tm, tests, g, E["impostors_per_gender"], a.split)
        scs[name] = sc
        res[name] = dict(model=path, speakers=len(tm), female=sum(v == "FEMALE" for v in g.values()),
                         gender_predicted=sum(V.gender_source.get(s) == "predicted" for s in g),
                         per_class_template=eval_dev.summary(sc, E["threshold"], "score_cls"),
                         all_template=eval_dev.summary(sc, E["threshold"], "score_all"),
                         max_of_both=eval_dev.summary(sc, E["threshold"], "score_max"))
        print(name, json.dumps({c: (v["eer"], v["frr"], v["far"]) for c, v in res[name]["per_class_template"].items()}), flush=True)
    # new model at the threshold that matches the old model's same-gender false accept (all conditions pooled)
    old_far = float(np.mean(scs["old"][scs["old"].kind == "same_gender"].score_cls >= E["threshold"]))
    thr = matched_thr(scs["new"], old_far)
    res["new"]["matched_threshold"] = round(thr, 4)
    res["new"]["at_matched_threshold"] = eval_dev.summary(scs["new"], thr, "score_cls")
    mean = {k: float(np.mean([res[k]["per_class_template"][c]["eer"] for c in CONDS])) for k in ("old", "new")}
    d = FT["deliver_if"]
    better = (mean["new"] <= (1 - d["min_rel_eer_gain"]) * mean["old"]
              and all(res["new"]["per_class_template"][c]["eer"] <= res["old"]["per_class_template"][c]["eer"] + d["max_eer_loss"]
                      for c in CONDS)
              and np.mean([res["new"]["at_matched_threshold"][c]["far_cross_gender"] for c in CONDS])
              <= np.mean([res["old"]["per_class_template"][c]["far_cross_gender"] for c in CONDS]) + d["max_far_loss"])
    res["mean_eer"] = {k: round(v, 4) for k, v in mean.items()}
    res["deliver"] = bool(better)
    json.dump(res, open(os.path.join(C["paths"]["models"], "spk_compare_%s.json" % a.split), "w"), indent=1)
    print(json.dumps(dict(mean_eer=res["mean_eer"], matched_threshold=thr, deliver=res["deliver"])), flush=True)
    plot(scs, res, a.split)
    print("done in %.0f min" % ((time.time() - t0) / 60))


def plot(scs, res, split):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 4, figsize=(16, 6.5), sharex=True)
    bins = np.linspace(-0.3, 1.0, 66)
    for i, name in enumerate(("old", "new")):
        sc = scs[name]
        for j, c in enumerate(CONDS):
            g = sc[sc.cond == c]
            for kind, col in (("owner", "C2"), ("same_gender", "C3"), ("cross_gender", "C0")):
                ax[i, j].hist(g[g.kind == kind].score_cls, bins=bins, alpha=.5, density=True, color=col, label=kind)
            ax[i, j].axvline(E["threshold"], color="k", lw=.8)
            r = res[name]["per_class_template"][c]
            ax[i, j].set_title("%s / %s: EER %.1f%%, FRR %.0f%% FAR %.1f%% @%.2f" % (
                name, c, 100 * r["eer"], 100 * r["frr"], 100 * r["far"], E["threshold"]), fontsize=8)
            ax[i, j].grid(alpha=.3)
    ax[0, 0].legend(fontsize=7)
    for j in range(4):
        ax[1, j].set_xlabel("cosine to the enrolled template (class)")
    fig.suptitle("speaker model: old spk-20261023 (top) vs fine-tuned (bottom), %s speakers, 10 enrolment takes" % split)
    fig.tight_layout()
    fig.savefig(os.path.join(C["paths"]["docs"], "spk_compare_%s.png" % split), dpi=120)


if __name__ == "__main__":
    main()
