"""45_finetune_spk.py - warm-start fine-tuning of the speaker model (phase B2).

    python scripts/45_finetune_spk.py                     # B2: models/spk_best_v1.pt -> models/spk_ft.pt
    python scripts/45_finetune_spk.py --stage big --trial # B3: larger SpkNet from scratch, 1/4 of the training -> models/spk_big_trial.pt
    python scripts/45_finetune_spk.py --stage big         # B3: full training -> models/spk_big.pt

Starts from spk-20261023 (same SpkNet: 40 log-mel x 81 frames -> 128).
  * the first layer (convs[0], bns[0]) is frozen, BN statistics included
    (freeze_bn_stats: the running statistics of every BN layer are kept);
  * every step: a batch of labelled windows -> AAM-softmax (weights start at the
    speakers' mean vectors of the old model, so the first steps do not shake the
    network), and a batch of bank windows -> 1 - cos(proj(student), teacher),
    proj = Linear(128, 192) used only in training (not exported);
    optional (weights in finetune.train): relational distillation (the batch's
    pairwise cosine matrix like the teacher's) and consistency (the device view
    and the enrolment-like view of the same audio give the same vector);
  * all windows go through the device chain (spk/device.py), genders balanced;
  * small learning rate, cosine schedule; after every epoch the device-protocol
    EER on validation speakers (clean / far / shout / noisy) - the best epoch is kept.
Parameters: configs/spk.yaml finetune.train / device / eval.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kws.dataset import worker_init  # noqa: E402
from kws.features import logmel  # noqa: E402
from spk import device, eval_dev  # noqa: E402
from spk.data import load_cfg  # noqa: E402
from spk.export import load as load_spk  # noqa: E402
from spk.model import SpkNet  # noqa: E402
from spk.voices import AamWindows, DistillWindows, Voices  # noqa: E402

C, P = load_cfg()
import argparse  # noqa: E402

_ap = argparse.ArgumentParser()
_ap.add_argument("--stage", default="ft", choices=["ft", "big"], help="ft: warm start (B2); big: larger SpkNet from scratch (B3)")
_ap.add_argument("--trial", action="store_true", help="big only: big.trial_epochs instead of the full training")
ARGS = _ap.parse_args()
FT = P["finetune"]
if ARGS.stage == "big":                 # B3: values of big.train replace finetune.train (the datasets read them too)
    FT["train"] = dict(FT["train"], **P["big"]["train"])
    if ARGS.trial:
        FT["train"]["epochs"] = P["big"]["trial_epochs"]
T = FT["train"]
TAG = "" if ARGS.stage == "ft" else ("_big_trial" if ARGS.trial else "_big")
OUT = FT["out"] if ARGS.stage == "ft" else "models/spk%s.pt" % TAG
SCORE = T.get("score", "score_cls")     # big: score_max = max(class template, all template), as the firmware
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODELS, DOCS = C["paths"]["models"], C["paths"]["docs"]


class AAM(torch.nn.Module):
    def __init__(self, W, m, s):
        super().__init__()
        self.W = torch.nn.Parameter(W)
        self.m, self.s = m, s

    def forward(self, emb, y):
        cos = F.linear(emb, F.normalize(self.W)).clamp(-1 + 1e-7, 1 - 1e-7)
        tgt = torch.cos(torch.acos(cos) + self.m)
        logits = self.s * torch.where(F.one_hot(y, cos.shape[1]).bool(), tgt, cos)
        return F.cross_entropy(logits, y), (cos.argmax(1) == y).float().mean()


def centroids(net, V, aam):
    """mean vector of every training speaker (clean, centred windows) with the starting model"""
    Q = FT["eval"]["conditions"]["enrol"]
    W = torch.zeros(len(aam.names), net.cfg["dim"])
    with torch.no_grad():
        for k0 in range(0, len(aam.names), 256):
            wins, owner = [], []
            for k in range(k0, min(len(aam.names), k0 + 256)):
                rng = np.random.default_rng([P["seed"], k])
                for it in aam.items[k][:T["centroid_clips"]]:
                    x, s, e = V.clip_of(it)
                    wins.append(logmel(device.chain(rng, x, s, e, V, Q, "train", center=True)))
                    owner.append(k)
            emb = net(torch.from_numpy(np.stack(wins)))
            W.index_add_(0, torch.tensor(owner), emb)
    return F.normalize(W, dim=1)


def val_eer(net, V):
    conds = T["select_conds"]
    tm, tests, g = eval_dev.build(net, V, P, "val", T["val_speakers"], conds, tag="evB2val")
    sc = eval_dev.scores(tm, tests, g, FT["eval"]["impostors_per_gender"], "val")
    s = eval_dev.summary(sc, FT["eval"]["threshold"], SCORE)
    return float(np.mean([s[c]["eer"] for c in conds])), s


def main():
    torch.manual_seed(P["seed"])
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(T["torch_threads"])
    t0 = time.time()
    old = load_spk(os.path.join(ROOT, FT["init"]))
    V = Voices(C)
    aam = AamWindows(V, C, P, T["steps_per_epoch"] * T["batch_aam"])
    dis = DistillWindows(V, C, P, T["steps_per_epoch"] * T["batch_distill"])
    print("AAM speakers %d (F %d / M %d), bank %d (F %d / M %d)" % (
        len(aam.names), len(aam.by_gender["FEMALE"]), len(aam.by_gender["MALE"]), len(dis.bank),
        len(dis.by_gender["FEMALE"]), len(dis.by_gender["MALE"])), flush=True)
    old.eval()
    e0, s0 = val_eer(old, V)
    print("spk-20261023: val mean EER %.4f %s" % (e0, json.dumps({c: v["eer"] for c, v in s0.items()})), flush=True)
    if ARGS.stage == "ft":
        net = old
        head = AAM(centroids(net, V, aam), T["aam_margin"], T["aam_scale"])
    else:                               # from scratch: new sizes, in_mean / in_std from training windows
        net = SpkNet(**P["big"]["net"])
        aam.epoch = 999999
        X = torch.stack([aam[i][0] for i in range(T["norm_windows"])])
        net.in_mean.copy_(X.mean(dim=(0, 1)))
        net.in_std.copy_(X.std(dim=(0, 1)))
        print("SpkNet %s: %d multiply-adds" % (net.cfg, net.macs()), flush=True)
        head = AAM(torch.randn(len(aam.names), net.cfg["dim"]) * 0.01, T["aam_margin"], T["aam_scale"])
        e0 = 9.0                        # keep the best epoch of the new model
    proj = torch.nn.Linear(net.cfg["dim"], dis.teacher.shape[1])
    frozen = []
    for i in range(T["freeze_layers"]):
        for p_ in list(net.convs[i].parameters()) + list(net.bns[i].parameters()):
            p_.requires_grad = False
        frozen.append(net.bns[i])
    params = [p_ for p_ in net.parameters() if p_.requires_grad]
    opt = torch.optim.AdamW([{"params": params, "lr": T["lr"]},
                             {"params": list(head.parameters()) + list(proj.parameters()), "lr": T["lr_heads"]}],
                            weight_decay=T["weight_decay"])
    total = T["epochs"] * T["steps_per_epoch"]
    warm = int(T["warmup_frac"] * total)

    def lr_at(step):
        if step < warm:
            return (step + 1) / warm
        q = (step - warm) / max(1, total - warm)
        return T["final_lr_frac"] + (1 - T["final_lr_frac"]) * 0.5 * (1 + np.cos(np.pi * q))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    log = [dict(epoch=0, val_eer=e0, **{"eer_" + c: v["eer"] for c, v in s0.items()})]
    best = e0
    for ep in range(T["epochs"]):
        aam.epoch = dis.epoch = ep
        mk = lambda ds, b, sd: iter(torch.utils.data.DataLoader(  # noqa: E731
            ds, batch_size=b, shuffle=True, num_workers=T["workers"] // 2, drop_last=True, worker_init_fn=worker_init,
            generator=torch.Generator().manual_seed(P["seed"] + 100 * ep + sd)))
        ia, idd = mk(aam, T["batch_aam"], 1), mk(dis, T["batch_distill"], 2)
        net.train()
        for bn in (list(net.bns) + [net.bn_emb]) if T.get("freeze_bn_stats") else frozen:
            bn.eval()                                    # keep the running statistics of spk-20261023
        t1, la, ld, acc, n, lr_, lc_ = time.time(), 0.0, 0.0, 0.0, 0, 0.0, 0.0
        for _ in range(T["steps_per_epoch"]):
            xa, ya = next(ia)
            batch = next(idd)
            xd, td = batch[0], batch[-1]
            views = [xa, xd] + ([batch[1]] if len(batch) == 3 else [])
            emb = net(torch.cat(views))
            na, nd = len(ya), len(xd)
            ed = emb[na:na + nd]
            l_aam, a_ = head(emb[:na], ya)
            l_dis = (1 - F.cosine_similarity(proj(ed), td)).mean()
            loss = T["loss_aam"] * l_aam + T["loss_distill"] * l_dis
            if T.get("loss_relational", 0) > 0:          # pairwise similarities of the batch like the teacher's
                off = ~torch.eye(nd, dtype=torch.bool)
                l_rel = ((ed @ ed.T - td @ td.T)[off] ** 2).mean()
                loss = loss + T["loss_relational"] * l_rel
                lr_ += l_rel.item()
            if len(batch) == 3:                           # device view == enrolment view of the same audio
                l_con = (1 - F.cosine_similarity(ed, emb[na + nd:])).mean()
                loss = loss + T["loss_consistency"] * l_con
                lc_ += l_con.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            la, ld, acc, n = la + l_aam.item(), ld + l_dis.item(), acc + a_.item(), n + 1
        net.eval()
        e, s = val_eer(net, V)
        row = dict(epoch=ep + 1, lr=opt.param_groups[0]["lr"], aam_loss=la / n, distill_loss=ld / n, aam_acc=acc / n,
                   relational_loss=lr_ / n, consistency_loss=lc_ / n,
                   val_eer=e, sec=time.time() - t1, **{"eer_" + c: v["eer"] for c, v in s.items()})
        log.append(row)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
        if e < best:
            best = e
            torch.save({"cfg": net.cfg, "state": net.state_dict(), "epoch": ep + 1, "val_eer": e, "stage": ARGS.stage},
                       os.path.join(ROOT, OUT))
    pd.DataFrame(log).to_csv(os.path.join(MODELS, "spk%s_log.csv" % (TAG or "_ft")), index=False)
    print("best val mean EER %.4f (start %.4f), %.0f min" % (best, e0, (time.time() - t0) / 60))
    plot(log)


def plot(log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ep = [r["epoch"] for r in log[1:]]
    ax[0].plot(ep, [r["distill_loss"] for r in log[1:]], label="distillation 1-cos")
    ax[0].plot(ep, [r["aam_loss"] / 10 for r in log[1:]], label="AAM-softmax / 10")
    ax[0].set_xlabel("epoch"), ax[0].legend(), ax[0].grid(alpha=.3)
    for c in T["select_conds"]:
        ax[1].plot([r["epoch"] for r in log], [100 * r["eer_" + c] for r in log], "o-", ms=3, label=c)
    ax[1].set_xlabel("epoch (0 = spk-20261023)"), ax[1].set_ylabel("validation EER %"), ax[1].legend(), ax[1].grid(alpha=.3)
    fig.suptitle("speaker model fine-tuning (warm start, distillation from ECAPA)")
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "spk%s_training_curves.png" % (TAG or "_ft")), dpi=130)


if __name__ == "__main__":
    main()
