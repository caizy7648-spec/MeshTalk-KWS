"""eval_dev.py - speaker model, device protocol with device conditions (phase B2).

Per speaker: 10 enrolment takes recorded like on the device (close, quiet room,
normal voice), take i belongs to class enrol_classes[i % 3]; template of a class
= sum of its take vectors, plus the template of all takes. Test takes (other
recordings of the speaker - LibriSpeech: other chapters first = other day /
session) in the conditions of finetune.eval.conditions (clean, far, shout,
noisy), each compared by cosine with
  * score_cls: the template of its class (what kws_voice_similarity does), and
  * score_all: the template of all takes,
  * score_max: the larger of the two (a firmware option),
for the speaker itself (target) and for impostors of the same and of the other
gender. Gender: data/cache/spk_gender.csv (true labels; predicted where missing).
"""
import numpy as np
import pandas as pd
import torch

from kws.cfg import rng_for, stable_hash
from kws.features import logmel

from . import device


def speakers(D, E, split, gender, exclude):
    """{speaker: [(store, row)]} with enough single-word takes"""
    out = {}
    for name, (df, _) in D.stores.items():
        g = df[(df.split == split) & (df.speaker != "") & (~df.group.isin(exclude))]
        if name == "clips":
            g = g[g.spk_use.astype(str) == "True"]
        for s, idx in g.groupby("speaker").groups.items():
            if s in gender:
                out.setdefault(s, []).extend((name, i) for i in idx)
    need = E["enrol_takes"] + E["test_takes_min"]
    keep = sorted((s for s, v in out.items() if len(v) >= need), key=lambda s: stable_hash("evB2:" + s))
    return {s: out[s] for s in keep}


def session(D, item):
    """recording session of a clip: LibriSpeech chapter, else the recording itself"""
    name, i = item
    df = D.stores[name][0]
    t = str(df.text[i]) if "text" in df else ""
    s = df.speaker[i]
    if s.startswith("ls-") and t.count("-") == 2:
        return t.rsplit("-", 1)[0]
    return "%s:%d" % (name, i)


def build(net, D, P, split, n_speakers, conds, tag="evB2"):
    """templates and test vectors; returns (templates {spk: [ncls+1, dim]}, tests DataFrame, gender dict)"""
    E = P["finetune"]["eval"]
    gender = D.gender
    sp = speakers(D, E, split, gender, P["data"]["exclude_groups_eval"])
    names = list(sp)[:n_speakers]
    ncls = len(E["enrol_classes"])
    wins, meta = [], []
    for s in names:
        rng = rng_for(D.C, tag, split, s)
        items = sorted(sp[s], key=lambda it: stable_hash("%s:%d" % it))
        en = items[:E["enrol_takes"]]
        en_sess = {session(D, it) for it in en}
        rest = items[E["enrol_takes"]:]
        rest = sorted(rest, key=lambda it: session(D, it) in en_sess)      # other sessions first
        te = rest[:E["test_takes"]]
        for j, it in enumerate(en):
            x, a, b = D.clip_of(it)
            wins.append(device.chain(rng, x, a, b, D, E["conditions"]["enrol"], split, center=True))
            meta.append(("enrol", s, j % ncls, "enrol", False))
        for j, it in enumerate(te):
            x, a, b = D.clip_of(it)
            other = session(D, it) not in en_sess
            for c in conds:
                wins.append(device.chain(rng, x, a, b, D, E["conditions"][c], split, center=True))
                meta.append(("test", s, j % ncls, c, other))
    emb = []
    with torch.no_grad():
        for k in range(0, len(wins), 512):
            emb.append(net(torch.from_numpy(np.stack([logmel(w) for w in wins[k:k + 512]]))).numpy())
    emb = np.concatenate(emb)
    tm = {s: np.zeros((ncls + 1, emb.shape[1])) for s in names}
    tests = []
    for (kind, s, k, c, other), e in zip(meta, emb):
        if kind == "enrol":
            tm[s][k] += e
            tm[s][ncls] += e
        else:
            tests.append(dict(speaker=s, cls=k, cond=c, other_session=other, emb=e))
    return tm, pd.DataFrame(tests), {s: gender[s] for s in names}


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def scores(tm, tests, gender, n_imp, key):
    """target and impostor trials (n_imp of the same gender + n_imp of the other gender)"""
    names = sorted(tm)
    by_g = {g: [s for s in names if gender[s] == g] for g in ("MALE", "FEMALE")}
    rows = []
    for n, t in enumerate(tests.itertuples()):
        g = gender[t.speaker]
        a, b = cos(tm[t.speaker][t.cls], t.emb), cos(tm[t.speaker][-1], t.emb)
        rows.append(dict(cond=t.cond, kind="owner", gender=g, other_session=t.other_session,
                         score_cls=a, score_all=b, score_max=max(a, b)))
        rng = np.random.default_rng([stable_hash(key) & 0xFFFFFFFF, n])
        for kind, pool in (("same_gender", by_g[g]), ("cross_gender", by_g["FEMALE" if g == "MALE" else "MALE"])):
            pool = [s for s in pool if s != t.speaker]
            for j in rng.permutation(len(pool))[:n_imp]:
                s = pool[j]
                a, b = cos(tm[s][t.cls], t.emb), cos(tm[s][-1], t.emb)
                rows.append(dict(cond=t.cond, kind=kind, gender=g, other_session=t.other_session,
                                 score_cls=a, score_all=b, score_max=max(a, b)))
    return pd.DataFrame(rows)


def eer(tg, im):
    t, i = np.sort(tg), np.sort(im)
    th = np.unique(np.concatenate([t, i]))
    frr = np.searchsorted(t, th, side="left") / len(t)
    far = 1.0 - np.searchsorted(i, th, side="left") / len(i)
    j = int(np.argmin(np.abs(frr - far)))
    return float((frr[j] + far[j]) / 2)


def summary(sc, thr, col="score_cls"):
    out = {}
    for c, g in sc.groupby("cond"):
        tg = g[g.kind == "owner"][col].to_numpy()
        im = g[g.kind != "owner"][col].to_numpy()
        r = dict(eer=round(eer(tg, im), 4), frr=round(float(np.mean(tg < thr)), 4), far=round(float(np.mean(im >= thr)), 4))
        for k in ("same_gender", "cross_gender"):
            v = g[g.kind == k][col].to_numpy()
            r["far_" + k] = round(float(np.mean(v >= thr)), 4)
            r["eer_" + k] = round(eer(tg, v), 4)
        for k, v in (("owner", tg), ("same_gender", g[g.kind == "same_gender"][col].to_numpy()),
                     ("cross_gender", g[g.kind == "cross_gender"][col].to_numpy())):
            r["p%s" % k] = [round(float(q), 3) for q in np.percentile(v, [5, 25, 50, 75, 95])]
        os_ = g[(g.kind == "owner") & g.other_session][col].to_numpy()
        if len(os_):
            r["frr_other_session"] = round(float(np.mean(os_ < thr)), 4)
            r["n_other_session"] = int(len(os_))
        r["n_owner"], r["n_impostor"] = int(len(tg)), int(len(im))
        out[c] = r
    return out
