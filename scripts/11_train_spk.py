"""11_train_spk.py - speaker head (Step 3), REAL voices only (MSWC + LibriSpeech
speakers with spk_use; synthetic voices are never used here).

    python scripts/11_train_spk.py           # models/kw.pt -> models/best.pt (+ spk_log.csv)

mode frozen_trunk: only spk (Linear c3 -> spk_dim) is trained; the keyword
part of the network stays bit-for-bit what step 2 produced.
mode multitask: the trunk is trained too, with the keyword loss on keyword
windows added (weight multitask_kw_weight) so the keyword head is kept.
Loss: AAM-softmax (additive angular margin) over the training speakers.
The epoch with the lowest validation EER (device enrolment protocol,
kws/evaluate.py) is kept.
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
from kws.cfg import load, seed_all  # noqa: E402
from kws.dataset import Data, KwWindows, SpkWindows  # noqa: E402
from kws.evaluate import eer  # noqa: E402
from kws.model import KwsNet  # noqa: E402

C = load()
S = C["train_spk"]
MODELS = C["paths"]["models"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ev = __import__("20_eval")


class AAM(torch.nn.Module):
    """additive angular margin softmax: s * cos(theta + m) for the true speaker"""

    def __init__(self, dim, n, m, s):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(n, dim) * 0.01)
        self.m, self.s = m, s

    def forward(self, emb, y):
        cos = F.linear(F.normalize(emb), F.normalize(self.W)).clamp(-1 + 1e-7, 1 - 1e-7)
        th = torch.acos(cos)
        tgt = torch.cos(th + self.m)
        oh = F.one_hot(y, cos.shape[1]).bool()
        return F.cross_entropy(self.s * torch.where(oh, tgt, cos), y)


def spk_augment():
    A = dict(C["augment"])
    for k in ("p_speed", "p_pitch", "p_shout", "p_context"):
        A[k] = 0.0                                   # these change the voice / add other voices
    A.update(S["augment"])
    return A


def val_eer(net, D):
    tmpl, tests = ev.speaker_vectors(net, D, "val", conds=("clean",))
    sc = ev.spk_scores(tmpl, tests, "val", S["impostors_per_test"])
    return eer(sc[sc.target].score.to_numpy(), sc[~sc.target].score.to_numpy())[0]


def main():
    seed_all(C, 1)
    torch.set_num_threads(C["train_kw"]["torch_threads"])
    ck = torch.load(os.path.join(MODELS, "kw.pt"), weights_only=False)
    net = KwsNet(**ck["cfg"])
    net.load_state_dict(ck["state"])
    D = Data(C)
    tr = SpkWindows(D, C, "train", S["windows_per_epoch"], spk_augment())
    print("speaker head: %d training speakers" % len(tr.speakers), flush=True)
    head = AAM(net.cfg["spk_dim"], len(tr.speakers), S["aam_margin"], S["aam_scale"])
    frozen = S["mode"] == "frozen_trunk"
    if frozen:
        for n_, p in net.named_parameters():
            p.requires_grad = n_.startswith("spk.")
        params = list(net.spk.parameters()) + list(head.parameters())
    else:
        params = list(net.parameters()) + list(head.parameters())
        kw_ds = KwWindows(D, C, "train", S["windows_per_epoch"], tag="spk_mt")
    opt = torch.optim.AdamW(params, lr=S["lr"], weight_decay=S["weight_decay"])
    steps = S["epochs"] * (S["windows_per_epoch"] // S["batch"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=S["lr"], total_steps=steps, pct_start=C["train_kw"]["warmup_frac"],
                                                div_factor=C["train_kw"]["div_factor"], final_div_factor=C["train_kw"]["final_div_factor"])
    g = torch.Generator()
    log, best, t0 = [], 9.0, time.time()
    for ep in range(S["epochs"]):
        tr.epoch = ep
        g.manual_seed(C["seed"] + 1000 + ep)
        dl = torch.utils.data.DataLoader(tr, batch_size=S["batch"], shuffle=True, num_workers=C["train_kw"]["workers"],
                                         generator=g, drop_last=True)
        if not frozen:
            kw_ds.epoch = ep
            kw_it = iter(torch.utils.data.DataLoader(kw_ds, batch_size=S["batch"], shuffle=True, num_workers=1,
                                                     generator=torch.Generator().manual_seed(C["seed"] + 2000 + ep), drop_last=True))
        net.train(not frozen)
        head.train()
        tl, tn, t1 = 0.0, 0, time.time()
        for x, y in dl:
            _, emb = net(x)
            loss = head(emb, y)
            if not frozen:
                xk, yk = next(kw_it)
                lg, _ = net(xk)
                loss = loss + S["multitask_kw_weight"] * F.cross_entropy(lg, yk, label_smoothing=C["train_kw"]["label_smoothing"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item() * len(y)
            tn += len(y)
        net.eval()
        e = val_eer(net, D)
        row = dict(epoch=ep + 1, train_loss=tl / tn, val_eer=e, sec=time.time() - t1)
        log.append(row)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
        if e < best:
            best = e
            torch.save({"cfg": net.cfg, "state": net.state_dict(), "labels": C["labels"], "kw_epoch": ck["epoch"],
                        "spk_epoch": ep + 1, "val_eer": e, "spk_mode": S["mode"]}, os.path.join(MODELS, "best.pt"))
    pd.DataFrame(log).to_csv(os.path.join(MODELS, "spk_log.csv"), index=False)
    print("best validation EER %.4f, %.0f min" % (best, (time.time() - t0) / 60))
    plot(log)


def plot(log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ep = [r["epoch"] for r in log]
    ax[0].plot(ep, [r["train_loss"] for r in log])
    ax[0].set_xlabel("epoch"), ax[0].set_ylabel("AAM-softmax loss (train)"), ax[0].grid(alpha=.3)
    ax[1].plot(ep, [100 * r["val_eer"] for r in log])
    ax[1].set_xlabel("epoch"), ax[1].set_ylabel("validation EER %"), ax[1].grid(alpha=.3)
    fig.suptitle("speaker head training (%s)" % S["mode"])
    fig.tight_layout()
    fig.savefig(os.path.join(C["paths"]["docs"], "spk_training_curves.png"), dpi=130)


if __name__ == "__main__":
    main()
