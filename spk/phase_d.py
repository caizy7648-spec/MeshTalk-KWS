"""phase_d.py - Phase D speaker model: training windows, trials, scoring, the owner's real recordings.

    V = Voices(C)
    ds = DWindows(V, C, P, "d2", n)            # (logmel, label): speaker augmentation / F0 invariance per variant
    tr = make_trials(V, P, "val", 150)         # windows of enrolment / test / mimic takes (fixed seeds, built once)
    emb = embed(net, tr.X, device)             # voice vectors of every window
    res = score_all(tr, emb, coh, P)           # EER etc. per condition and scoring method
    own = own_eval(V, net, P)                  # owner (own-A) vs female imitator (own-B), real device recordings

Scoring methods (the firmware currently uses "max"):
  cls   cosine to the template of the call's keyword class
  all   cosine to the template of all enrolment takes
  max   max(cls, all)
  asn   adaptive symmetric score normalisation (AS-norm) of max: a cohort of training speakers'
        vectors; s' = ((s - mu_t) / sd_t + (s - mu_e) / sd_e) / 2 with mean / sd of the top-K
        cohort cosines of the template and of the test vector
  pair  two calls of the same person averaged (the firmware needs two calls for an SOS anyway),
        cosine to the template of all takes
  pasn  pair + AS-norm
Conditions: clean / far / shout / noisy (spk.device chain, configs/spk.yaml finetune.eval.conditions)
and the imitation tests "mimic_f2m" (a woman's takes with ONLY the pitch lowered by f2m_semitones,
formants kept - spk.voice_mod.f0_shift - scored against men's templates; owner trials = the men's
clean takes) and "mimic_m2f" (the other way round). "plain_f2m" = the same women's takes unchanged,
for reference (how much the lowered pitch helps the imitator).
"""
import numpy as np
import pandas as pd
import torch

from kws.cfg import rng_for, stable_hash
from kws.dataset import n_samples
from kws.features import logmel
from kws.audio import to_int16

from . import device, eval_dev
from .voice_mod import f0_shift, vtl_shift
from .voices import balanced

RATE = 16000
METHODS = ("cls", "all", "max", "asn", "pair", "pasn")


# ------------------------------------------------------------------ training windows
def crop(rng, x, s, e, max_s=2.5):
    """at most max_s seconds of a long clip, around the spoken part (the window is 1.62 s)"""
    n = int(max_s * RATE)
    if len(x) <= n:
        return x, s, e
    c = (s + e) // 2 + int(rng.integers(-RATE // 4, RATE // 4 + 1))
    o = int(min(max(0, c - n // 2), len(x) - n))
    return x[o:o + n], max(0, s - o), min(n, e - o)


class DWindows(torch.utils.data.Dataset):
    """labelled real voices (B2 data: MSWC, LibriSpeech, VCTK; + AISHELL-1 train speakers), gender-balanced,
    through the device chain; variant switches: speaker_aug (speed-perturbed copies = new speakers),
    f0_aug (pitch-only changes = same speaker)"""

    def __init__(self, V, C, P, variant, n, tag="phaseD"):
        self.V, self.C, self.P, self.n, self.tag = V, C, P, n, tag
        Q = P["phase_d"]
        self.sw = Q["variants"][variant]
        cap = Q["data"]["max_clips_per_speaker"]
        by = {}
        for name, (df, _) in V.stores.items():
            g = df[(df.split == "train") & (df.speaker != "")]
            if name == "clips":
                g = g[g.spk_use.astype(str) == "True"]
            for s, idx in g.groupby("speaker").groups.items():
                by.setdefault(s, []).extend((name, i) for i in sorted(idx, key=lambda i: stable_hash("capB:%d" % i))[:cap])
        names = sorted(by)
        # AISHELL-1 (Mandarin) training speakers from the r4 cache (group chat_zh): no gender labels -> own pool
        df = V.D.df
        ais = df[(df.group == "chat_zh") & (df.split == "train")]
        aby = {s: [("clips", i) for i in sorted(idx, key=lambda i: stable_hash("capB:%d" % i))[:cap]]
               for s, idx in ais.groupby("speaker").groups.items()}
        self.names = names + sorted(aby)
        self.items = [by[s] for s in names] + [aby[s] for s in sorted(aby)]
        self.by_gender = {"MALE": [], "FEMALE": []}
        for k, s in enumerate(names):
            self.by_gender[V.gender.get(s, "MALE")].append(k)
        self.ais = list(range(len(names), len(self.names)))
        self.p_ais = Q["data"]["aishell_p"] if self.ais else 0.0
        self.K = len(self.names)
        self.factors = list(Q["speaker_aug"]["factors"]) if self.sw["speaker_aug"] else []
        self.fp = np.asarray(Q["speaker_aug"]["p"], float)[:1 + len(self.factors)]
        self.fp = self.fp / self.fp.sum()
        self.n_classes = self.K * (1 + len(self.factors))
        self.F0 = Q["f0_aug"]
        self.Q = P["finetune"]["device"]
        self.epoch = 0

    def __len__(self):
        return self.n

    def __getstate__(self):
        """DataLoader workers are started with "spawn" (forking a process that already uses CUDA / OpenMP threads
        deadlocks): do not pickle Voices - its memory-mapped audio would be copied in full into every worker;
        each worker opens the cache itself"""
        d = dict(self.__dict__)
        d["V"] = None
        return d

    def __getitem__(self, i):
        if self.V is None:
            from .voices import Voices
            self.V = Voices(self.C)
        ep = self.epoch
        if isinstance(i, tuple):                          # (epoch, index) from EpochSampler (persistent workers)
            ep, i = i
        rng = np.random.default_rng([self.P["seed"], stable_hash(self.tag) & 0xFFFFFFFF, ep, i])
        if self.ais and rng.random() < self.p_ais:
            k = self.ais[rng.integers(len(self.ais))]
        else:
            k = balanced(rng, self.by_gender)
        it = self.items[k][rng.integers(len(self.items[k]))]
        x, s, e = self.V.clip_of(it)
        x, s, e = crop(rng, x, s, e)
        j = int(rng.choice(len(self.fp), p=self.fp)) if self.factors else 0
        if j:                                             # speed perturbation: another vocal tract -> another speaker
            f = self.factors[j - 1]
            x = vtl_shift(x, f)
            s, e = int(s / f), min(len(x), max(int(s / f) + 1, int(e / f)))
        if self.sw["f0_aug"] and rng.random() < self.F0["p"]:
            x = f0_shift(x, float(rng.uniform(*self.F0["semitones"])))   # pitch only -> the same speaker
        w = device.chain(rng, x, s, e, self.V, self.Q, "train")
        return torch.from_numpy(logmel(w)), k + self.K * j


class EpochSampler(torch.utils.data.Sampler):
    """shuffled (epoch, index) pairs: the epoch reaches workers that live through the whole training"""

    def __init__(self, n, seed):
        self.n, self.seed, self.epoch = n, seed, 0

    def __len__(self):
        return self.n

    def __iter__(self):
        g = np.random.default_rng([self.seed, self.epoch])
        return iter([(self.epoch, int(i)) for i in g.permutation(self.n)])


# ------------------------------------------------------------------ trials (built once, embedded per model)
class Trials:
    pass


def make_trials(V, P, split, n_speakers, conds=("clean", "far", "shout", "noisy"), tag="evD"):
    """windows (logmel) of every enrolment / test / mimic take of n_speakers speakers of the split"""
    E = P["finetune"]["eval"]
    M = P["phase_d"]["mimic"]
    sp = eval_dev.speakers(V, E, split, V.gender, P["data"]["exclude_groups_eval"])
    names = list(sp)[:n_speakers]
    ncls = len(E["enrol_classes"])
    X, meta = [], []
    for s in names:
        rng = rng_for(V.C, tag, split, s)
        items = sorted(sp[s], key=lambda it: stable_hash("%s:%d" % it))
        en = items[:E["enrol_takes"]]
        en_sess = {eval_dev.session(V, it) for it in en}
        rest = sorted(items[E["enrol_takes"]:], key=lambda it: eval_dev.session(V, it) in en_sess)
        te = rest[:E["test_takes"]]
        for j, it in enumerate(en):
            x, a, b = V.clip_of(it)
            X.append(logmel(device.chain(rng, x, a, b, V, E["conditions"]["enrol"], split, center=True)))
            meta.append(("enrol", s, j % ncls, "enrol", j))
        g = V.gender[s]
        for j, it in enumerate(te):
            x, a, b = V.clip_of(it)
            for c in conds:
                X.append(logmel(device.chain(rng, x, a, b, V, E["conditions"][c], split, center=True)))
                meta.append(("test", s, j % ncls, c, j))
            # imitation: the same take with only the pitch moved towards the other gender
            st = rng.uniform(*(M["f2m_semitones"] if g == "FEMALE" else M["m2f_semitones"]))
            xm, a2, b2 = crop(rng, x, a, b)
            X.append(logmel(device.chain(rng, f0_shift(xm, float(st)), a2, b2, V, E["conditions"]["clean"], split, center=True)))
            meta.append(("mimic", s, j % ncls, "mimic_f2m" if g == "FEMALE" else "mimic_m2f", j))
    tr = Trials()
    tr.X = np.stack(X).astype(np.float32)
    tr.meta = pd.DataFrame(meta, columns=["kind", "speaker", "cls", "cond", "take"])
    tr.names, tr.gender, tr.ncls, tr.conds = names, {s: V.gender[s] for s in names}, ncls, list(conds)
    tr.n_imp = E["impostors_per_gender"]
    tr.split = split
    return tr


def cohort_windows(V, P, n_speakers, per=3, tag="cohortD"):
    """clean enrolment-like windows of n_speakers TRAIN speakers (AS-norm cohort), [n_speakers, per, F, 40]"""
    E = P["finetune"]["eval"]
    by = {}
    df = V.D.df
    g = df[(df.split == "train") & (df.speaker != "") & (df.spk_use.astype(str) == "True")]
    for s, idx in g.groupby("speaker").groups.items():
        if len(idx) >= per:
            by[s] = list(idx)
    names = sorted(by, key=lambda s: stable_hash(tag + s))[:n_speakers]
    out = []
    for s in names:
        rng = rng_for(V.C, tag, s)
        idx = sorted(by[s], key=lambda i: stable_hash("%s:%d" % (tag, i)))[:per]
        out.append(np.stack([logmel(device.chain(rng, *V.D.clip(i), V, E["conditions"]["enrol"], "train", center=True))
                             for i in idx]))
    return np.stack(out).astype(np.float32)


def embed(net, X, dev="cpu", batch=1024):
    net = net.to(dev).eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(X), batch):
            out.append(net(torch.from_numpy(X[k:k + batch]).to(dev)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1), np.float32)


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def cohort_vectors(net, CW, dev="cpu"):
    """one vector per cohort speaker (mean of its windows), unit length"""
    n, per = CW.shape[:2]
    e = embed(net, CW.reshape((n * per,) + CW.shape[2:]), dev).reshape(n, per, -1)
    return unit(unit(e).sum(1))


# ------------------------------------------------------------------ scoring
def eer(tg, im):
    return eval_dev.eer(np.asarray(tg), np.asarray(im)) if len(tg) and len(im) else float("nan")


def far_at_frr(tg, im, frr=0.05):
    """false-accept rate at the threshold that rejects `frr` of the owner's calls (and that threshold)"""
    if not len(tg) or not len(im):
        return float("nan"), float("nan")
    thr = float(np.quantile(np.asarray(tg), frr))
    return float(np.mean(np.asarray(im) >= thr)), thr


class Scorer:
    """templates of every speaker and the scoring methods (cosine, AS-norm, pairs)"""

    def __init__(self, tr, emb, coh, P):
        self.tr, self.emb, self.coh = tr, unit(emb), coh
        self.K = P["phase_d"]["eval"]["asnorm_topk"]
        m = tr.meta
        self.tm = {}
        for s in tr.names:
            e = m[(m.kind == "enrol") & (m.speaker == s)]
            t = np.zeros((tr.ncls + 1, self.emb.shape[1]))
            for r in e.itertuples():
                t[r.cls] += self.emb[r.Index]
                t[tr.ncls] += self.emb[r.Index]
            self.tm[s] = unit(t)
        self.tstat = {s: self._stats(self.tm[s]) for s in tr.names}

    def _stats(self, v):
        """mean / sd of the top-K cohort cosines of each row of v"""
        if self.coh is None or not len(self.coh):
            return np.zeros(len(v)), np.ones(len(v))
        c = np.sort(v @ self.coh.T, axis=1)[:, -self.K:]
        return c.mean(1), c.std(1) + 1e-6

    def score(self, s, e, cls):
        """{method: score} of test vector(s) e (1 = single call, 2 rows = pair) against speaker s's templates"""
        t = self.tm[s]
        mu_t, sd_t = self.tstat[s]
        out = {}
        if e.ndim == 1:
            a, b = float(t[cls] @ e), float(t[-1] @ e)
            mu_e, sd_e = self._stats(e[None])
            na = 0.5 * ((a - mu_t[cls]) / sd_t[cls] + (a - mu_e[0]) / sd_e[0])
            nb = 0.5 * ((b - mu_t[-1]) / sd_t[-1] + (b - mu_e[0]) / sd_e[0])
            out.update(cls=a, all=b, max=max(a, b), asn=max(na, nb))
        else:
            p = unit(e.mean(0))
            b = float(t[-1] @ p)
            mu_e, sd_e = self._stats(p[None])
            out.update(pair=b, pasn=0.5 * ((b - mu_t[-1]) / sd_t[-1] + (b - mu_e[0]) / sd_e[0]))
        return out


def score_all(tr, emb, coh, P):
    """per condition: owner / impostor (same, cross gender) / mimic scores of every method -> summary dict"""
    S = Scorer(tr, emb, coh, P)
    m = tr.meta
    E = P["phase_d"]["eval"]
    by_g = {g: [s for s in tr.names if tr.gender[s] == g] for g in ("MALE", "FEMALE")}
    other = {"MALE": "FEMALE", "FEMALE": "MALE"}
    rows = []

    def impostors(rng, s, pool):
        pool = [x for x in pool if x != s]
        return [pool[j] for j in rng.permutation(len(pool))[:tr.n_imp]]

    test = m[m.kind == "test"]
    for (s, c), g in test.groupby(["speaker", "cond"]):
        rng = np.random.default_rng([stable_hash("imp:%s:%s:%s" % (tr.split, s, c)) & 0xFFFFFFFF])
        same = impostors(rng, s, by_g[tr.gender[s]])
        cross = impostors(rng, s, by_g[other[tr.gender[s]]])
        idx = g.index.to_numpy()
        for r in g.itertuples():
            e = S.emb[r.Index]
            rows.append(dict(cond=c, kind="owner", gender=tr.gender[s], **S.score(s, e, r.cls)))
            for kind, pool in (("same_gender", same), ("cross_gender", cross)):
                for o in pool:
                    rows.append(dict(cond=c, kind=kind, gender=tr.gender[s], **S.score(o, e, r.cls)))
        for k in range(0, len(idx) - 1, 2):                 # pairs of calls
            e2 = S.emb[idx[k:k + 2]]
            rows.append(dict(cond=c, kind="owner", gender=tr.gender[s], **S.score(s, e2, 0)))
            for kind, pool in (("same_gender", same), ("cross_gender", cross)):
                for o in pool:
                    rows.append(dict(cond=c, kind=kind, gender=tr.gender[s], **S.score(o, e2, 0)))
    # imitation: women with a lowered pitch against men's templates (and the other way); plain = unchanged takes
    mim = m[m.kind == "mimic"]
    clean = m[(m.kind == "test") & (m.cond == "clean")]
    for src, g in ((mim, "mimic"), (clean, "plain")):
        for s, gg in src.groupby("speaker"):
            gs = tr.gender[s]
            cond = "%s_%s" % (g, "f2m" if gs == "FEMALE" else "m2f")
            rng = np.random.default_rng([stable_hash("mim:%s:%s" % (tr.split, s)) & 0xFFFFFFFF])
            targets = impostors(rng, s, by_g[other[gs]])
            idx = gg.index.to_numpy()
            for r in gg.itertuples():
                for o in targets:
                    rows.append(dict(cond=cond, kind="impostor", gender=gs, **S.score(o, S.emb[r.Index], r.cls)))
            for k in range(0, len(idx) - 1, 2):
                for o in targets:
                    rows.append(dict(cond=cond, kind="impostor", gender=gs, **S.score(o, S.emb[idx[k:k + 2]], 0)))
    sc = pd.DataFrame(rows)
    return summarise(sc, E), sc


def summarise(sc, E):
    """{cond: {method: {eer, eer_same, eer_cross, far@frr, thr@frr, n}}}; mimic conds: owner = the target gender's clean calls"""
    out = {}
    frr = E["owner_frr_point"]
    for c in sorted(sc.cond.unique()):
        g = sc[sc.cond == c]
        if c.startswith(("mimic", "plain")):
            tgt_gender = "MALE" if c.endswith("f2m") else "FEMALE"
            own = sc[(sc.cond == "clean") & (sc.kind == "owner") & (sc.gender == tgt_gender)]
        else:
            own = g[g.kind == "owner"]
        out[c] = {}
        for meth in METHODS:
            tg = own[meth].dropna().to_numpy()
            im = g[g.kind != "owner"][meth].dropna().to_numpy()
            if not len(tg) or not len(im):
                continue
            far, thr = far_at_frr(tg, im, frr)
            r = dict(eer=round(eer(tg, im), 4), far_at_frr=round(far, 4), thr_at_frr=round(thr, 4), n_owner=int(len(tg)), n_imp=int(len(im)))
            for k in ("same_gender", "cross_gender"):
                v = g[g.kind == k][meth].dropna().to_numpy()
                if len(v):
                    r["eer_" + k] = round(eer(tg, v), 4)
            out[c][meth] = r
    return out


def select_metric(summ, method="max", conds=("clean", "far", "shout", "noisy")):
    """the number a variant is chosen by (val): mean EER of the 4 device conditions and the f2m imitation"""
    v = [summ[c][method]["eer"] for c in conds if c in summ and method in summ[c]]
    if "mimic_f2m" in summ and method in summ["mimic_f2m"]:
        v.append(summ["mimic_f2m"][method]["eer"])
    return float(np.mean(v)) if v else float("nan")


# ------------------------------------------------------------------ the owner's real recordings (DataRec)
def own_windows(V, P):
    """windows of the owner's and the imitator's DataRec takes (as the device hands them to the speaker check)"""
    O = P["phase_d"]["eval"]["own"]
    df = V.D.df
    g = df[(df.group == "own") & df.speaker.isin([O["owner"], O["imitator"]])]
    N = n_samples(P["net"]["frames"])
    X, meta = [], []
    for i in g.index:
        x, s, e = V.D.clip(i)
        o = device.place(None, x, s, e, N, 2 * 320, True)
        y = np.zeros(N, np.float32)
        a, b = max(0, o), min(N, o + len(x))
        if b > a:
            y[a:b] = x[a - o:b - o]
        X.append(logmel(to_int16(y)))
        meta.append((df.speaker[i], df.label[i], df.word[i]))
    m = pd.DataFrame(meta, columns=["speaker", "label", "style"])
    return (np.stack(X).astype(np.float32) if X else np.zeros((0, P["net"]["frames"], 40), np.float32)), m


def attack(scores, thr, n, rng, trials=2000, pair=False):
    """probability that an imitator gets an SOS within n attempts (two accepted calls in a row of three
    attempts ~ "2 calls within 10 s"; pair: two consecutive calls averaged must pass once)"""
    if not len(scores):
        return float("nan")
    ok = 0
    k = max(1, n - 1) if pair else n                     # n calls give n - 1 consecutive pairs
    for _ in range(trials):
        s = np.asarray(scores)[rng.integers(len(scores), size=k)]
        acc = s >= thr
        if pair:
            ok += bool(np.any(acc))
        else:
            ok += any(acc[i] and acc[i + 1:i + 3].any() for i in range(k - 1))
    return ok / trials


def own_eval(net, X, m, coh, P, dev="cpu"):
    """owner (enrol 10 NORMAL takes, the device cycle help / help me / sos) vs the imitator, `draws` random enrolments"""
    O = P["phase_d"]["eval"]["own"]
    K = P["phase_d"]["eval"]["asnorm_topk"]
    cls_of = {"help": 0, "help me": 1, "sos": 2}
    if not len(X) or not (m.speaker == O["owner"]).any():
        return {"note": "no own recordings"}, None
    E = unit(embed(net, X, dev))

    def stats(v):
        if coh is None or not len(coh):
            return 0.0, 1.0
        c = np.sort(coh @ v)[-K:]
        return float(c.mean()), float(c.std() + 1e-6)

    own_i = m.index[m.speaker == O["owner"]].to_numpy()
    imp_i = m.index[m.speaker == O["imitator"]].to_numpy()
    res = {"n_owner_takes": int(len(own_i)), "n_imitator_takes": int(len(imp_i))}
    pool = {k: [i for i in own_i if m.label[i] == lab and m["style"][i] == O["enrol_style"]] for lab, k in cls_of.items()}
    rows = []
    for d in range(O["draws"]):
        rng = np.random.default_rng([stable_hash("ownD") & 0xFFFFFFFF, d])
        en = []
        for j in range(O["enrol_takes"]):
            p = [i for i in pool[j % 3] if i not in en] or [i for i in own_i if i not in en]
            en.append(p[rng.integers(len(p))])
        t = np.zeros((4, E.shape[1]))
        for j, i in enumerate(en):
            t[cls_of.get(m.label[i], j % 3)] += E[i]
            t[3] += E[i]
        t = unit(t)
        ts = [stats(v) for v in t]
        test_own = [i for i in own_i if i not in en]
        for who, idx in (("owner", test_own), ("imitator", imp_i)):
            for i in idx:
                k = cls_of.get(m.label[i], 0)
                a, b = float(t[k] @ E[i]), float(t[3] @ E[i])
                me, se = stats(E[i])
                na = 0.5 * ((a - ts[k][0]) / ts[k][1] + (a - me) / se)
                nb = 0.5 * ((b - ts[3][0]) / ts[3][1] + (b - me) / se)
                rows.append(dict(draw=d, who=who, style=m["style"][i], label=m.label[i], cls=a, all=b, max=max(a, b), asn=max(na, nb)))
            # pairs of calls (two takes of the same person, in recording order)
            for k in range(0, len(idx) - 1, 2):
                p = unit(E[idx[k]] + E[idx[k + 1]])
                b = float(t[3] @ p)
                me, se = stats(p)
                rows.append(dict(draw=d, who=who, style="pair", label="pair", pair=b,
                                 pasn=0.5 * ((b - ts[3][0]) / ts[3][1] + (b - me) / se)))
    sc = pd.DataFrame(rows)
    rng = np.random.default_rng(7)
    for meth in METHODS:
        tg = sc[sc.who == "owner"][meth].dropna().to_numpy()
        im = sc[sc.who == "imitator"][meth].dropna().to_numpy()
        if not len(tg) or not len(im):
            continue
        far, thr = far_at_frr(tg, im, P["phase_d"]["eval"]["owner_frr_point"])
        r = dict(eer=round(eer(tg, im), 4), far_at_frr=round(far, 4), thr_at_frr=round(thr, 4),
                 owner_pct=[round(float(q), 3) for q in np.percentile(tg, [5, 25, 50, 75, 95])],
                 imitator_pct=[round(float(q), 3) for q in np.percentile(im, [5, 25, 50, 75, 95])],
                 imitator_sos_within=({str(n): round(attack(im, thr, n, rng, pair=meth in ("pair", "pasn")), 4)
                                       for n in O["attempts"]}))
        if meth in ("cls", "all", "max", "asn"):
            r["owner_accept_by_style"] = {st: round(float(np.mean(v >= thr)), 3)
                                          for st, v in sc[(sc.who == "owner") & (sc["style"] != "pair")].groupby("style")[meth]}
        res[meth] = r
    return res, sc
