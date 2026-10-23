"""20_eval.py - hit rate / false alarms per hour with kws.detect (Step 2) and
speaker EER with the device enrolment (Step 3).

    python scripts/20_eval.py kw  --model models/kw.pt   --split val    # dev: choose thresholds
    python scripts/20_eval.py kw  --model models/best.pt --split test   # final numbers
    python scripts/20_eval.py spk --model models/best.pt --split test

split val  = validation speakers / voices + LibriSpeech dev  (used for choices)
split test = test speakers / voices + LibriSpeech test + test noise (never used for choices)
Results: models/eval_<part>_<split>.json, figures in docs/.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kws import augment as aug  # noqa: E402
from kws.audio import to_float, to_int16  # noqa: E402
from kws.cfg import load, rng_for, stable_hash  # noqa: E402
from kws.dataset import Data  # noqa: E402
from kws.evaluate import Stream, best_window, eer, fa_count, hit_in, positive_stream, rates, take  # noqa: E402
from kws.features import logmel  # noqa: E402
from kws.model import KwsNet  # noqa: E402

C = load()
E = C["eval"]
CACHE, MODELS, DOCS = C["paths"]["cache"], C["paths"]["models"], C["paths"]["docs"]


def load_net(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = KwsNet(**ck["cfg"])
    net.load_state_dict(ck["state"])
    net.eval()
    return net


def noise_fn(D):
    return lambda rng, n: D.noise_seg(rng, "test", n)


# ------------------------------------------------------------------ keyword positives
def positives(D, split):
    """[(class, source, audio, s, e)] of the split"""
    out = []
    df = D.df
    cap = E["max_per_source"]

    def add(lab, src, idx):
        idx = sorted(idx, key=lambda i: stable_hash("evalpos:%d" % i))[:cap]
        for i in idx:
            out.append((lab, src) + D.clip(i))

    add("help", "real MSWC", df[(df.group == "mswc_help") & (df.split == split)].index)
    add("help", "real LibriSpeech", df[(df.group == "libri_help") & (df.split == split)].index)
    add("help", "TTS (held-out voices)", df[(df.group == "tts") & (df.label == "help") & (df.split == split)].index)
    add("help me", "real LibriSpeech", df[(df.group == "libri_help_me") & (df.split == split)].index)
    if D.join[split]:
        rng = rng_for(C, "evaljoin", split)
        for _ in range(E["joined_help_me"]):
            out.append(("help me", "real joined help+me") + D.joined_help_me(rng, split, C["train_kw"]["join_gap_ms"]))
    add("help me", "TTS (held-out voices)", df[(df.group == "tts") & (df.label == "help me") & (df.split == split)].index)
    add("sos", "TTS (held-out voices)", df[(df.group == "tts") & (df.label == "sos") & (df.split == split)].index)
    return out


def fa_streams(D, split, T):
    """running speech (+ noise on a share of the files) and the test noise, cut into ~30 min streams"""
    name = "dev" if split == "val" else "test"
    sp = pd.read_csv(os.path.join(CACHE, "fa_%s.csv" % name))
    a = np.load(os.path.join(CACHE, "fa_%s.npy" % name), mmap_mode="r")
    kw = pd.read_csv(os.path.join(CACHE, "fa_%s_kw.csv" % name))
    rng = rng_for(C, "fa", split)
    chunks, cur, guard, base = [], [], [], 0
    for r in sp.itertuples():
        x = to_float(np.array(a[r.offset:r.offset + r.length]))
        x = aug.set_rms(x, rng.uniform(*E["test_level_dbfs"]))
        if rng.random() < E["fa_noise_mix_prob"]:
            x = aug.add_noise(x, D.noise_seg(rng, "test", len(x)), rng.uniform(*E["fa_noise_snr_db"]), x)
        x = aug.noise_floor(x, rng, E["noise_floor_dbfs"])
        for k in kw[(kw["at"] >= r.offset) & (kw["at"] < r.offset + r.length)].itertuples():
            a0 = base + (k.at - r.offset)
            guard.append((a0 * 1000 // 16000, (base + k.end - r.offset) * 1000 // 16000 + int(E["keyword_guard_s"] * 1000)))
        cur.append(x)
        base += len(x)
        if base > E["fa_chunk_min"] * 60 * 16000:
            chunks.append(("speech", to_int16(np.concatenate(cur)), guard))
            cur, guard, base = [], [], 0
    if cur:
        chunks.append(("speech", to_int16(np.concatenate(cur)), guard))
    nd, na = D.noise["test"]
    cur, base = [], 0
    for r in nd.itertuples():
        x = to_float(np.array(na[r.offset:r.offset + r.length]))
        x = aug.set_rms(x, rng.uniform(*E["noise_level_dbfs"]))
        cur.append(aug.noise_floor(x, rng, E["noise_floor_dbfs"]))
        base += len(x)
        if base > E["fa_chunk_min"] * 60 * 16000:
            chunks.append(("noise", to_int16(np.concatenate(cur)), []))
            cur, base = [], 0
    if cur:
        chunks.append(("noise", to_int16(np.concatenate(cur)), []))
    return chunks


def eval_kw(net, split, tag, kind="device"):
    T = net.cfg["T"]
    labels = C["labels"]
    keyword = [not l.startswith("_") for l in labels]
    grid = E["thr_grid"]
    X = C["export"]
    D = Data(C)
    nf = noise_fn(D)
    t0 = time.time()
    # --- positives
    pos = positives(D, split)
    rows = []
    for n, (lab, src, x, s, e) in enumerate(pos):
        for cond in ("clean", "noisy"):
            rng = rng_for(C, "evalstream", split, n, cond)
            pcm, a, b = positive_stream(rng, x, s, e, E, nf, cond == "noisy")
            st = Stream(net, pcm, T)
            r = dict(label=lab, source=src, cond=cond)
            for thr in grid:
                k = hit_in(st.hits(keyword, thr, check=(n == 0), kind=kind), a * 1000 // 16000, b * 1000 // 16000 + E["hit_after_ms"])
                r["hit@%.2f" % thr] = k >= 0
                r["cls@%.2f" % thr] = k == labels.index(lab)
            rows.append(r)
    P = pd.DataFrame(rows)
    print("positives: %d streams (%.0f s)" % (len(P), time.time() - t0), flush=True)
    # --- false alarms
    fa = {"speech": dict(hours=0.0), "noise": dict(hours=0.0)}
    for thr in grid:
        for k in fa:
            fa[k]["hits@%.2f" % thr] = 0
            fa[k]["sos@%.2f" % thr] = 0
    for src, pcm, guard in fa_streams(D, split, T):
        st = Stream(net, pcm, T)
        fa[src]["hours"] += len(pcm) / 16000 / 3600
        for thr in grid:
            fh, sos = fa_count(st.hits(keyword, thr, X["hits"], X["hits_ms"], kind=kind), guard, X["hits"], X["hits_ms"])
            fa[src]["hits@%.2f" % thr] += fh
            fa[src]["sos@%.2f" % thr] += sos
        print("  %s %.2f h done (%.0f s)" % (src, fa[src]["hours"], time.time() - t0), flush=True)
    # --- summary per threshold
    sfx = "" if kind == "device" else "_" + kind
    res = dict(split=split, model=tag, detector=kind, streams=len(P), fa_hours={k: round(v["hours"], 2) for k, v in fa.items()}, per_thr={})
    res["positives"] = {f"{a}|{b}": int(v) for (a, b), v in P[P.cond == "clean"].groupby(["label", "source"]).size().items()}
    for thr in grid:
        h, c = "hit@%.2f" % thr, "cls@%.2f" % thr
        d = {}
        for (lab, src, cond), g in P.groupby(["label", "source", "cond"]):
            d[f"{lab}|{src}|{cond}"] = dict(hit=round(g[h].mean(), 4), correct_class=round(g[c].mean(), 4))
        for lab, g in P.groupby("label"):
            d[f"{lab}|ALL|ALL"] = dict(hit=round(g[h].mean(), 4), correct_class=round(g[c].mean(), 4))
        hrs = sum(v["hours"] for v in fa.values())
        f = {k: dict(false_hits_per_h=round(v["hits@%.2f" % thr] / max(v["hours"], 1e-9), 3),
                     false_sos_per_h=round(v["sos@%.2f" % thr] / max(v["hours"], 1e-9), 3),
                     false_hits=v["hits@%.2f" % thr], false_sos=v["sos@%.2f" % thr]) for k, v in fa.items()}
        f["all"] = dict(false_hits_per_h=round(sum(v["hits@%.2f" % thr] for v in fa.values()) / hrs, 3),
                        false_sos=sum(v["sos@%.2f" % thr] for v in fa.values()))
        res["per_thr"]["%.2f" % thr] = dict(hit=d, fa=f)
    json.dump(res, open(os.path.join(MODELS, "eval_kw_%s%s.json" % (split, sfx)), "w"), indent=1)
    plot_kw(res, split, sfx)
    print("done in %.0f min" % ((time.time() - t0) / 60))
    return res


def plot_kw(res, split, sfx=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    thr = sorted(res["per_thr"], key=float)
    for lab, col in zip(C["labels"][:-1], ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        xs = [res["per_thr"][t]["fa"]["all"]["false_hits_per_h"] for t in thr]
        ys = [100 * res["per_thr"][t]["hit"][f"{lab}|ALL|ALL"]["hit"] for t in thr]
        ax.plot(xs, ys, "o-", color=col, label=lab, ms=3)
    for t in thr[::3]:
        ax.annotate(t, (res["per_thr"][t]["fa"]["all"]["false_hits_per_h"], 100 * res["per_thr"][t]["hit"]["help|ALL|ALL"]["hit"]),
                    fontsize=7, xytext=(3, -8), textcoords="offset points")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("false hits per hour (running speech + noise, %s)" % ("LibriSpeech dev" if split == "val" else "LibriSpeech test + test noise"))
    ax.set_ylabel("hit rate % (clean + noisy)")
    ax.grid(alpha=.3)
    ax.legend()
    ax.set_title("hit rate vs false alarms, %s (%s)" % ("kws.detect.Detector" if not sfx else "proposed window gate", split))
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "kw_hit_vs_fa_%s%s.png" % (split, sfx)), dpi=130)


# ------------------------------------------------------------------ speaker
KW_WORDS = ("help", "help me", "me")


def trial_speakers(D, split, text_dependent=False):
    df = D.df[(D.df.spk_use == True) & (D.df.split == split)]  # noqa: E712
    if text_dependent:      # only "help" / "help me" / "me" clips (LibriSpeech): enrolment and test say the keywords
        df = df[df.word.isin(KW_WORDS)]
    S = C["train_spk"]
    need = S["enrol_takes"] + S["test_clips_min"]
    sp = [s for s, g in df.groupby("speaker") if len(g) >= need]
    sp = sorted(sp, key=lambda s: stable_hash("trial:" + s))[:S["trial_speakers"]]
    return {s: sorted(df[df.speaker == s].index, key=lambda i: stable_hash("clip:%d" % i)) for s in sp}


def speaker_vectors(net, D, split, conds=("clean", "noisy", "shout"), text_dependent=False):
    """enrolment templates and test vectors per speaker (device protocol)
    text_dependent=False: enrol with 5 other words, test with the speaker's keywords first (pessimistic)
    text_dependent=True:  enrol and test with help / help me / me clips only"""
    S = C["train_spk"]
    T = net.cfg["T"]
    labels = C["labels"]
    kcls = [labels.index(c) for c in S["enrol_classes"]]
    spk = trial_speakers(D, split, text_dependent)
    tmpl, tests = {}, []
    for s, idx in spk.items():
        rng = rng_for(C, "spk", split, s, int(text_dependent))
        if text_dependent:
            enrol, test = idx[:S["enrol_takes"]], idx[S["enrol_takes"]:]
        else:
            kwi = [i for i in idx if D.df.label[i] in ("help", "help me")]
            oth = [i for i in idx if i not in kwi]
            enrol = oth[:S["enrol_takes"]]                   # keywords are kept for testing
            test = kwi + oth[S["enrol_takes"]:S["enrol_takes"] + S["test_clips_max"] - len(kwi)]
        tm = np.zeros((len(labels), net.cfg["spk_dim"]))
        for k, i in zip(kcls, enrol):
            x, a, b = D.clip(i)
            pcm = take(rng, x, a, b, S["take_s"], rng.uniform(*E["test_level_dbfs"]), E["noise_floor_dbfs"])
            tm[k] += best_window(net, logmel(pcm), k, T)[1]
        tmpl[s] = tm
        for i in test[:S["test_clips_max"]]:
            x, a, b = D.clip(i)
            lab = D.df.label[i]
            k = labels.index(lab) if lab in ("help", "help me") else 0
            for cond in conds:
                kw = {}
                if cond == "noisy":
                    kw = dict(noise=D.noise_seg(rng, "test", int(S["take_s"] * 16000)), snr=rng.uniform(*E["test_snr_db"]))
                if cond == "shout":
                    kw = dict(shout_db=rng.uniform(*C["augment"]["shout_gain_db"]))
                pcm = take(rng, x, a, b, S["take_s"], rng.uniform(*E["test_level_dbfs"]), E["noise_floor_dbfs"], **kw)
                tests.append(dict(speaker=s, k=k, cond=cond, keyword=lab in ("help", "help me"),
                                  emb=best_window(net, logmel(pcm), k, T)[1]))
    return tmpl, tests


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def spk_scores(tmpl, tests, split, n_imp):
    names = sorted(tmpl)
    out = []
    for n, t in enumerate(tests):
        rng = rng_for(C, "imp", split, n)
        imp = [names[j] for j in rng.permutation(len(names))[:n_imp + 1] if names[j] != t["speaker"]][:n_imp]
        out.append(dict(cond=t["cond"], keyword=t["keyword"], target=True, score=cos(tmpl[t["speaker"]][t["k"]], t["emb"])))
        for s in imp:
            out.append(dict(cond=t["cond"], keyword=t["keyword"], target=False, score=cos(tmpl[s][t["k"]], t["emb"])))
    return pd.DataFrame(out)


def eval_spk(net, split, tag, spk_thr=None):
    S = C["train_spk"]
    t0 = time.time()
    D = Data(C)
    spk_thr = C["export"]["spk_thr"] if spk_thr is None else spk_thr
    res = dict(split=split, model=tag, spk_thr=spk_thr)
    for proto, td in (("text_independent", False), ("text_dependent", True)):
        tmpl, tests = speaker_vectors(net, D, split, text_dependent=td)
        sc = spk_scores(tmpl, tests, split, S["impostors_per_test"])
        r0 = dict(speakers=len(tmpl), cond={})
        for cond, g in sc.groupby("cond"):
            tg, im = g[g.target].score.to_numpy(), g[~g.target].score.to_numpy()
            e, th = eer(tg, im)
            frr, far = rates(tg, im, spk_thr)
            r0["cond"][cond] = dict(trials_target=len(tg), trials_impostor=len(im), eer=round(e, 4), eer_thr=round(th, 4),
                                    frr_at_spk_thr=round(frr, 4), far_at_spk_thr=round(far, 4))
        res[proto] = r0
        sc.to_csv(os.path.join(MODELS, "eval_spk_%s_%s_scores.csv" % (split, proto)), index=False)
        plot_spk(sc, r0, spk_thr, split, proto)
        print(proto, json.dumps(r0), flush=True)
    json.dump(res, open(os.path.join(MODELS, "eval_spk_%s.json" % split), "w"), indent=1)
    print("done in %.0f min" % ((time.time() - t0) / 60))
    return res


def plot_spk(sc, res, spk_thr, split, proto):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    th = np.linspace(-0.2, 1.0, 241)
    for cond, ls in (("clean", "-"), ("noisy", "--")):
        g = sc[sc.cond == cond]
        tg, im = g[g.target].score.to_numpy(), g[~g.target].score.to_numpy()
        ax[0].plot(th, [100 * np.mean(tg < t) for t in th], "r" + ls, label="false reject (%s)" % cond)
        ax[0].plot(th, [100 * np.mean(im >= t) for t in th], "b" + ls, label="false accept (%s)" % cond)
    ax[0].axvline(spk_thr, color="k", lw=.8)
    ax[0].set_xlabel("cosine threshold spk_thr"), ax[0].set_ylabel("%"), ax[0].grid(alpha=.3), ax[0].legend(fontsize=7)
    ax[0].set_title("EER clean %.1f %%, noisy %.1f %%" % (100 * res["cond"]["clean"]["eer"], 100 * res["cond"]["noisy"]["eer"]))
    g = sc[sc.cond == "clean"]
    ax[1].hist(g[g.target].score, bins=50, alpha=.6, density=True, label="same speaker")
    ax[1].hist(g[~g.target].score, bins=50, alpha=.6, density=True, label="other speakers")
    ax[1].axvline(spk_thr, color="k", lw=.8)
    ax[1].set_xlabel("cosine to the enrolled template"), ax[1].legend(), ax[1].grid(alpha=.3)
    fig.suptitle("speaker head, device enrolment (5 takes), %s, %d %s speakers" % (proto.replace("_", "-"), res["speakers"], split))
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "spk_eer_%s_%s.png" % (split, proto)), dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["kw", "spk"])
    ap.add_argument("--model", default=os.path.join(MODELS, "best.pt"))
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--detector", default="device", choices=["device", "window_gate"],
                    help="device = kws.detect.Detector (the firmware today); window_gate = proposed gate change")
    a = ap.parse_args()
    torch.set_num_threads(C["train_kw"]["torch_threads"])
    net = load_net(a.model)
    if a.part == "kw":
        eval_kw(net, a.split, os.path.basename(a.model), a.detector)
    else:
        eval_spk(net, a.split, os.path.basename(a.model))


if __name__ == "__main__":
    main()
