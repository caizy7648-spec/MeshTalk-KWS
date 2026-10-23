"""41_eval_spk.py - speaker model: per-take EER (device enrolment protocol) and
SYSTEM numbers with the keyword model and the 3-hits-in-20-s rule.

    python scripts/41_eval_spk.py --split val     # choose spk threshold (written to models/eval_spk_val.json)
    python scripts/41_eval_spk.py --split test    # final numbers at export.threshold of configs/spk.yaml

1. Per take: enrol N = 5 / 10 takes, test takes clean / noisy / shouted,
   EER and false reject / false accept at the threshold.
2. System: a speaker who has real "help" recordings enrols 10 takes (other
   words: the only real takes there are), then
     owner:  calls "help" 6 times within 20 s (clean / noisy / shouted)
     other:  another speaker calls 6 times, compared with the owner's voice
   through kws.detect.Detector(only_me=True, verify=speaker model): trigger
   rate = scenarios ending in EV_SOS. Keyword model and its thresholds: models/best.pt
   and configs/train.yaml export:.
On val the threshold is swept (the similarity of every keyword hit does not
depend on it) and chosen: highest owner trigger rate with other-person
trigger rate <= target; on test only the chosen threshold is run.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kws import augment as aug  # noqa: E402
from kws.audio import to_int16  # noqa: E402
from kws.cfg import rng_for, stable_hash  # noqa: E402
from kws.dataset import Data  # noqa: E402
from kws.detect import EV_HIT, EV_REJECT, EV_SOS, Detector  # noqa: E402
from kws.evaluate import Stream  # noqa: E402
from spk import evaluate as ev  # noqa: E402
from spk.data import load_cfg, speakers  # noqa: E402
from spk.export import load as load_spk  # noqa: E402

C, P = load_cfg()
E, X, S = P["eval"], C["export"], C["eval_sys"]
MODELS, DOCS = C["paths"]["models"], C["paths"]["docs"]
LABELS = C["labels"]
KEYWORD = [not l.startswith("_") for l in LABELS]
RATE = 16000
kw_eval = __import__("20_eval")


def per_take(net, D, split, thr):
    out = {}
    curves = {}
    for n in E["enrol_takes"]:
        tmpl, tests = ev.enrol(net, D, C, P, split, n, E["speakers"])
        sc = ev.scores(tmpl, tests, E["impostors"], "%s%d" % (split, n))
        r = dict(speakers=len(tmpl), cond={})
        for cond, g in sc.groupby("cond"):
            tg, im = g[g.target].score.to_numpy(), g[~g.target].score.to_numpy()
            e, th = ev.eer(tg, im)
            frr, far = ev.rates(tg, im, thr)
            r["cond"][cond] = dict(target=len(tg), impostor=len(im), eer=round(e, 4), eer_thr=round(th, 4),
                                   frr=round(frr, 4), far=round(far, 4))
            gk = g[g.keyword]
            if gk.target.sum() >= 20:
                r["cond"][cond]["eer_keyword_tests"] = round(ev.eer(gk[gk.target].score.to_numpy(), gk[~gk.target].score.to_numpy())[0], 4)
        out[n] = r
        curves[n] = sc
    return out, curves


def calls_stream(D, rng, clips, n_calls, cond):
    """one person calls n_calls times (their own clips) within 20 s; clean / noisy / shouted"""
    total = int((S["lead_s"] + S["window_s"] + S["tail_s"]) * RATE)
    y = np.zeros(total, np.float32)
    level = rng.uniform(*C["eval"]["test_level_dbfs"])
    slot = S["window_s"] / n_calls
    speech = []
    for c in range(n_calls):
        x, s, e = D.clip(clips[rng.integers(len(clips))])
        if rng.random() < S["p_speed"]:
            f = rng.uniform(*S["speed"])
            x = aug.speed(x, f)
            s, e = int(s / f), min(len(x), int(e / f))
        x = aug.set_rms(x, level + rng.uniform(*S["call_gain_db"]), x[s:e] if e - s > 160 else x)
        if cond == "shout":
            x = aug.shout(x, rng.uniform(*E["shout_gain_db"]))
        o = int((S["lead_s"] + c * slot + rng.uniform(0, max(0.0, slot - len(x) / RATE))) * RATE)
        n = min(len(x), total - o)
        y[o:o + n] += x[:n]
        speech.append(x[s:e])
    if cond == "noisy":
        y = aug.add_noise(y, D.noise_seg(rng, "test", total), rng.uniform(*C["eval"]["test_snr_db"]), np.concatenate(speech))
    return to_int16(aug.noise_floor(y, rng, C["eval"]["noise_floor_dbfs"]))


def system(kwnet, snet, D, split, thr_list):
    """owner / other trigger rates for every spk threshold in thr_list (hit similarities recorded once)"""
    df = D.df
    helps = {}
    for g in ("mswc_help", "libri_help"):
        for s, idx in df[(df.group == g) & (df.split == split)].groupby("speaker").groups.items():
            helps.setdefault(s, []).extend(list(idx))
    sp = speakers(D, split, exclude=P["data"]["exclude_groups_eval"])
    owners = sorted((s for s in helps if s in sp and len([i for i in sp[s] if df.label[i] == "_other"]) >= E["system_enrol_takes"]),
                    key=lambda s: stable_hash("sysB:" + s))
    kc = ev.classes(C, P)
    tm = {}
    for s in owners:
        rng = rng_for(C, "sysenrol", split, s)
        en = sorted((i for i in sp[s] if df.label[i] == "_other"), key=lambda i: stable_hash("clipB:%d" % i))[:E["system_enrol_takes"]]
        wins = [ev.window(rng, *D.clip(i), snet.cfg["frames"], rng.uniform(*E["enrol_level_dbfs"]), E["noise_floor_dbfs"]) for i in en]
        t = np.zeros((len(LABELS), snet.cfg["dim"]))
        for j, e in enumerate(ev.embed(snet, wins)):
            t[kc[j % len(kc)]] += e
        tm[s] = t
    thr_kw = X["thr"][1]
    off = [float(X["cls_off"].get(l, 0.0)) for l in LABELS]
    res = {}
    for cond in ("clean", "noisy", "shout"):
        for who in ("owner", "other"):
            sims_all = []
            for k in range(E["system_scenarios"]):
                rng = rng_for(C, "sysB", split, cond, who, k)
                owner = owners[k % len(owners)]
                caller = owner if who == "owner" else owners[(k + 1 + int(rng.integers(len(owners) - 1))) % len(owners)]
                pcm = calls_stream(D, rng, helps[caller], E["system_calls"], cond)
                st = Stream(kwnet, pcm, kwnet.cfg["T"])
                cache = {int(f): j for j, f in enumerate(st.frames)}
                cur = [0]

                def run_fn(w):
                    return st.probs[cache[cur[0]]], None

                def verify(win, cls):
                    with torch.no_grad():
                        e = snet(torch.from_numpy(np.ascontiguousarray(win, np.float32))[None])[0].numpy()
                    return ev.cos(ev.template(tm[owner], cls), e)

                single = len(thr_list) == 1        # one threshold: Detector itself rejects / counts
                det = Detector(run_fn, kwnet.cfg["T"], KEYWORD, thr_kw, cls_off=off, hits=X["hits"], hits_ms=X["hits_ms"],
                               verify=verify, verify_thr=thr_list[0] if single else -2.0, only_me=True)
                hits, sos = [], False
                for i in range(len(st.feats)):
                    cur[0] = i
                    evt = det.frame(st.feats[i], st.dbfs[i], i * 20)
                    if evt & (EV_HIT | EV_REJECT):
                        hits.append((i * 20, det.hit_spk))
                    sos |= bool(evt & EV_SOS)
                sims_all.append((hits, sos))
            if len(thr_list) == 1:
                res.setdefault("%.3f" % thr_list[0], {})["%s|%s" % (who, cond)] = round(np.mean([s_ for h, s_ in sims_all]), 4)
                continue
            for thr in thr_list:
                n = 0
                for hits, _ in sims_all:    # Detector: a rejected hit does not count, everything else is the same
                    times = []
                    for t, sim in hits:
                        if sim < thr:
                            continue
                        times = [u for u in times if t - u < X["hits_ms"]] + [t]
                        if len(times) >= X["hits"]:
                            n += 1
                            break
                res.setdefault("%.3f" % thr, {})["%s|%s" % (who, cond)] = round(n / len(sims_all), 4)
    return res, len(owners)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--model", default=os.path.join(MODELS, "spk_best.pt"))
    ap.add_argument("--kw-model", default=os.path.join(MODELS, "best.pt"))
    a = ap.parse_args()
    torch.set_num_threads(P["train"]["torch_threads"] * 2)
    t0 = time.time()
    snet = load_spk(a.model)
    kwnet = kw_eval.load_net(a.kw_model)
    D = Data(C)
    grid = list(np.round(np.arange(E["thr_grid"][0], E["thr_grid"][1] + 1e-9, E["thr_grid"][2]), 3))
    if a.split == "val":
        sysres, n_owners = system(kwnet, snet, D, "val", grid)
        ok = [(float(t), r) for t, r in sysres.items()
              if max(r["other|%s" % c] for c in ("clean", "noisy", "shout")) <= E["target_other"]]
        thr = max(ok, key=lambda tr: (np.mean([tr[1]["owner|%s" % c] for c in ("clean", "noisy", "shout")]), -tr[0]))[0] if ok \
            else float(min(sysres, key=lambda t: max(sysres[t]["other|%s" % c] for c in ("clean", "noisy", "shout"))))
    else:
        thr = P["export"]["threshold"]
        sysres, n_owners = system(kwnet, snet, D, "test", [thr])
    takes, curves = per_take(snet, D, a.split, thr)
    out = dict(split=a.split, model=os.path.basename(a.model), kw_model=os.path.basename(a.kw_model), threshold=thr,
               owners=n_owners, per_take=takes, system=sysres, system_at_threshold=sysres["%.3f" % thr])
    json.dump(out, open(os.path.join(MODELS, "eval_spk_%s.json" % a.split), "w"), indent=1)
    print(json.dumps(dict(threshold=thr, per_take=takes, system=sysres["%.3f" % thr]), indent=1))
    plot(curves, sysres, thr, a.split)
    print("done in %.0f min" % ((time.time() - t0) / 60))


def plot(curves, sysres, thr, split):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    th = np.linspace(-0.2, 1.0, 241)
    n = max(curves)
    sc = curves[n]
    for cond, col in (("clean", "C0"), ("noisy", "C1"), ("shout", "C2")):
        g = sc[sc.cond == cond]
        tg, im = g[g.target].score.to_numpy(), g[~g.target].score.to_numpy()
        e, _ = ev.eer(tg, im)
        ax[0].plot(th, [100 * np.mean(tg < t) for t in th], "-", color=col, label="false reject %s (EER %.1f%%)" % (cond, 100 * e))
        ax[0].plot(th, [100 * np.mean(im >= t) for t in th], "--", color=col, label="false accept %s" % cond)
    ax[0].axvline(thr, color="k", lw=.8)
    ax[0].set_xlabel("cosine threshold"), ax[0].set_ylabel("% per take"), ax[0].legend(fontsize=6), ax[0].grid(alpha=.3)
    ax[0].set_title("per take, %d enrolment takes" % n)
    ts = sorted(sysres, key=float)
    if len(ts) > 1:
        for cond, col in (("clean", "C0"), ("noisy", "C1"), ("shout", "C2")):
            ax[1].plot([100 * sysres[t]["other|%s" % cond] for t in ts], [100 * sysres[t]["owner|%s" % cond] for t in ts],
                       "o-", ms=3, color=col, label=cond)
        ax[1].axvline(100 * E["target_other"], color="k", lw=.8)
        ax[1].axhline(100 * E["target_owner"], color="k", lw=.8)
        ax[1].set_xlabel("other person calling 6x: trigger %"), ax[1].set_ylabel("owner calling 6x: trigger %")
        ax[1].legend(), ax[1].grid(alpha=.3)
        ax[1].set_title("system (keyword model + 3 hits / 20 s + Only My Voice)")
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "spk_eer_%s.png" % split), dpi=130)


if __name__ == "__main__":
    main()
