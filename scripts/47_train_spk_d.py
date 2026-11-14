"""47_train_spk_d.py - Phase D: train one variant of the speaker model from scratch (configs/spk.yaml phase_d).

    python scripts/47_train_spk_d.py --variant d2        # -> models/spk_d2.pt, models/spk_d2_log.csv, docs/spk_d2_training.png

Same SpkNet as spk-20261023 (configs/spk.yaml net: 40 log-mel x 81 frames -> 128, 12.8 M MAC, same C engine).
AAM-softmax over the training speakers (with speaker_aug: x3 classes, the speed-perturbed copies),
windows through the device chain (spk/device.py, finetune.device), genders balanced; F0-shifted
copies (f0_aug) keep their speaker label. Every val_every epochs: validation speakers (fixed windows,
built once) clean / far / shout / noisy + imitation (mimic_f2m) - the epoch with the lowest
phase_d.select_metric (mean of the 5 EERs, firmware scoring "max") is kept.
"""
import argparse
import faulthandler
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kws.dataset import worker_init  # noqa: E402
from spk import phase_d  # noqa: E402
from spk.data import load_cfg  # noqa: E402
from spk.model import SpkNet  # noqa: E402
from spk.voices import Voices  # noqa: E402

C, P = load_cfg()
Q = P["phase_d"]
MODELS, DOCS = C["paths"]["models"], C["paths"]["docs"]


class AAM(torch.nn.Module):
    """additive angular margin softmax: s * cos(theta + m) for the true class"""

    def __init__(self, dim, n, m, s):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(n, dim) * 0.01)
        self.m, self.s = m, s

    def forward(self, emb, y):
        cos = F.linear(emb, F.normalize(self.W)).clamp(-1 + 1e-7, 1 - 1e-7)
        tgt = torch.cos(torch.acos(cos) + self.m)
        oh = F.one_hot(y, cos.shape[1]).bool()
        logits = self.s * torch.where(oh, tgt, cos)
        return F.cross_entropy(logits, y), (cos.argmax(1) == y).float().mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(Q["variants"]))
    ap.add_argument("--epochs", type=int, help="override (smoke test)")
    ap.add_argument("--steps", type=int, help="override steps per epoch (smoke test)")
    ap.add_argument("--fresh", action="store_true", help="ignore models/spk_<variant>_last.pt and start again")
    a = ap.parse_args()
    faulthandler.enable()                                  # a crash in C code prints the Python stack of every thread
    T = dict(Q["train"])
    if a.epochs:
        T["epochs"] = a.epochs
    if a.steps:
        T["steps_per_epoch"] = a.steps
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if T["device"] == "auto" else T["device"]
    torch.manual_seed(P["seed"])
    np.random.seed(P["seed"] % 2 ** 32)
    torch.set_num_threads(T["torch_threads"])
    t0 = time.time()
    V = Voices(C)
    n = T["steps_per_epoch"] * T["batch"]
    ds = phase_d.DWindows(V, C, P, a.variant, n)
    print("variant %s %s: %d speakers (%d AISHELL), %d classes, device %s" % (
        a.variant, json.dumps(Q["variants"][a.variant]), ds.K, len(ds.ais), ds.n_classes, dev), flush=True)
    net = SpkNet(**P["net"])
    last = os.path.join(MODELS, "spk_%s_last.pt" % a.variant)   # resume point, written every epoch (not committed)
    sig = json.dumps(dict(variant=Q["variants"][a.variant], train=T, net=P["net"], n_classes=ds.n_classes), sort_keys=True)
    R = None
    if os.path.exists(last) and not a.fresh:
        R = torch.load(last, map_location="cpu", weights_only=False)
        if R.get("sig") != sig:
            print("%s is from other settings - starting again (--fresh to silence)" % last, flush=True)
            R = None
    if R is None:
        ds.epoch = 999999
        X = torch.stack([ds[i][0] for i in range(min(T["norm_windows"], n))])
        net.in_mean.copy_(X.mean(dim=(0, 1)))
        net.in_std.copy_(X.std(dim=(0, 1)))
        print("in_mean / in_std (%.0f s), %d multiply-adds per vector" % (time.time() - t0, net.macs()), flush=True)
    t1 = time.time()
    vt = phase_d.make_trials(V, P, "val", Q["eval"]["speakers_val"])
    cw = phase_d.cohort_windows(V, P, Q["eval"]["cohort_speakers"])
    print("validation: %d speakers, %d windows; cohort %d (%.0f s)" % (len(vt.names), len(vt.X), len(cw), time.time() - t1), flush=True)
    net = net.to(dev)
    head = AAM(P["net"]["dim"], ds.n_classes, T["aam_margin"], T["aam_scale"]).to(dev)
    opt = torch.optim.AdamW(list(net.parameters()) + list(head.parameters()), lr=T["lr"], weight_decay=T["weight_decay"])
    total = T["epochs"] * T["steps_per_epoch"]
    warm = max(1, int(T["warmup_frac"] * total))

    def lr_at(step):
        if step < warm:
            return T["lr"] * (step + 1) / warm
        q = (step - warm) / max(1, total - warm)
        lo = T["lr"] * T["final_lr_frac"]
        return lo + (T["lr"] - lo) * 0.5 * (1 + math.cos(math.pi * q))

    out = os.path.join(MODELS, "spk_%s.pt" % a.variant)
    log, best, step, ep0 = [], float("inf"), 0, 0
    if R is not None:                                      # continue after a crash: same weights, optimiser, step, log
        net.load_state_dict(R["net"])
        head.load_state_dict(R["head"])
        opt.load_state_dict(R["opt"])
        log, best, step, ep0 = R["log"], R["best"], R["step"], R["epoch"]
        print("resumed from %s after epoch %d (best %.4f)" % (last, ep0, best), flush=True)
    # one DataLoader for the whole run: workers started once with "spawn" (fork after CUDA / OpenMP init can
    # deadlock - it did twice at the start of an epoch), kept alive (persistent_workers); the epoch travels
    # with every index (EpochSampler), so the windows are the same as with a new loader per epoch
    sampler = phase_d.EpochSampler(n, P["seed"])
    ds.epoch = 0
    kw = dict(multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=4) if T["workers"] > 0 else {}
    dl = torch.utils.data.DataLoader(ds, batch_size=T["batch"], sampler=sampler, num_workers=T["workers"], drop_last=True,
                                     worker_init_fn=worker_init, pin_memory=False, **kw)   # no pin-memory thread (WSL segfaults)
    for ep in range(ep0, T["epochs"]):
        sampler.epoch = ep
        net.train()
        t1, tl, ta, k = time.time(), 0.0, 0.0, 0
        for x, y in dl:
            for pg in opt.param_groups:
                pg["lr"] = lr_at(step)
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            loss, acc = head(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            tl += loss.item() * len(y)
            ta += acc.item() * len(y)
            k += len(y)
        row = dict(epoch=ep + 1, lr=lr_at(step - 1), train_loss=tl / k, train_acc=ta / k, sec=time.time() - t1)
        if (ep + 1) % T["val_every"] == 0 or ep + 1 == T["epochs"]:
            net.eval()
            coh = phase_d.cohort_vectors(net, cw, dev)
            summ, _ = phase_d.score_all(vt, phase_d.embed(net, vt.X, dev), coh, P)
            sel = phase_d.select_metric(summ)
            row.update(val_select=sel, **{"eer_%s" % c: summ[c]["max"]["eer"] for c in
                                          ("clean", "far", "shout", "noisy", "mimic_f2m", "plain_f2m") if c in summ})
            if "mimic_f2m" in summ:
                row["far_mimic_f2m"] = summ["mimic_f2m"]["max"]["far_at_frr"]
            if sel < best:
                best = sel
                torch.save({"cfg": net.cfg, "state": {k2: v.cpu() for k2, v in net.state_dict().items()}, "epoch": ep + 1,
                            "val_select": sel, "variant": a.variant, "switches": Q["variants"][a.variant]}, out)
                row["saved"] = 1
        log.append(row)
        print(json.dumps({k2: round(v, 4) if isinstance(v, float) else v for k2, v in row.items()}), flush=True)
        pd.DataFrame(log).to_csv(os.path.join(MODELS, "spk_%s_log.csv" % a.variant), index=False)
        torch.save(dict(sig=sig, net=net.state_dict(), head=head.state_dict(), opt=opt.state_dict(), log=log, best=best,
                        step=step, epoch=ep + 1), last + ".tmp")
        os.replace(last + ".tmp", last)                    # atomic: a crash while saving keeps the previous one
    print("best validation select metric %.4f -> %s, %.0f min" % (best, out, (time.time() - t0) / 60), flush=True)
    plot(log, a.variant)


def plot(log, variant):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ep = [r["epoch"] for r in log]
    ax[0].plot(ep, [r["train_loss"] for r in log])
    ax[0].set_xlabel("epoch"), ax[0].set_ylabel("AAM-softmax loss (train)"), ax[0].grid(alpha=.3)
    v = [r for r in log if "val_select" in r]
    for key, lab in (("val_select", "select (mean of 5)"), ("eer_clean", "clean"), ("eer_shout", "shout"),
                     ("eer_mimic_f2m", "imitation f2m"), ("eer_plain_f2m", "women, unchanged")):
        if v and key in v[0]:
            ax[1].plot([r["epoch"] for r in v], [100 * r[key] for r in v], label=lab)
    ax[1].set_xlabel("epoch"), ax[1].set_ylabel("validation EER %"), ax[1].grid(alpha=.3), ax[1].legend(fontsize=7)
    fig.suptitle("speaker model phase D, variant %s" % variant)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "spk_%s_training.png" % variant), dpi=130)


if __name__ == "__main__":
    main()
