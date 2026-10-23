"""40_train_spk.py - the separate speaker model of phase B (CONTRACT_SPK.md).

    python scripts/40_train_spk.py           # -> models/spk_best.pt, models/spk_log.csv, docs/spk_training_curves.png

Real voices only (MSWC + LibriSpeech speakers, train split), AAM-softmax over
the training speakers, all parameters in configs/spk.yaml. After every epoch
a quick validation EER (device protocol, 5 enrolment takes, clean) picks the
epoch that is kept.
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
from kws.dataset import Data, worker_init  # noqa: E402
from spk import evaluate as ev  # noqa: E402
from spk.data import SpkTrain, load_cfg  # noqa: E402
from spk.model import SpkNet  # noqa: E402

C, P = load_cfg()
T = P["train"]
MODELS, DOCS = C["paths"]["models"], C["paths"]["docs"]


class AAM(torch.nn.Module):
    """additive angular margin softmax: s * cos(theta + m) for the true speaker"""

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


def val_eer(net, D):
    tmpl, tests = ev.enrol(net, D, C, P, "val", 5, T["val_speakers"], conds=("clean",), tag="spkval")
    sc = ev.scores(tmpl, tests, P["eval"]["impostors"], "val")
    return ev.eer(sc[sc.target].score.to_numpy(), sc[~sc.target].score.to_numpy())[0]


def main():
    torch.manual_seed(P["seed"])
    np.random.seed(P["seed"] % 2 ** 32)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(T["torch_threads"])
    D = Data(C)
    tr = SpkTrain(D, C, P, "train", T["windows_per_epoch"])
    print("speakers: %d train" % len(tr.names), flush=True)
    net = SpkNet(**P["net"])
    t0 = time.time()
    tr.epoch = 999999
    X = torch.stack([tr[i][0] for i in range(T["norm_windows"])])
    net.in_mean.copy_(X.mean(dim=(0, 1)))
    net.in_std.copy_(X.std(dim=(0, 1)))
    print("in_mean / in_std (%.0f s), %d multiply-adds per vector" % (time.time() - t0, net.macs()), flush=True)
    head = AAM(P["net"]["dim"], len(tr.names), T["aam_margin"], T["aam_scale"])
    opt = torch.optim.AdamW(list(net.parameters()) + list(head.parameters()), lr=T["lr"], weight_decay=T["weight_decay"])
    steps = T["epochs"] * (T["windows_per_epoch"] // T["batch"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=T["lr"], total_steps=steps, pct_start=T["warmup_frac"],
                                                div_factor=T["div_factor"], final_div_factor=T["final_div_factor"])
    log, best = [], 9.0
    for ep in range(T["epochs"]):
        tr.epoch = ep
        g = torch.Generator()
        g.manual_seed(P["seed"] + ep)
        dl = torch.utils.data.DataLoader(tr, batch_size=T["batch"], shuffle=True, num_workers=T["workers"], generator=g,
                                         drop_last=True, worker_init_fn=worker_init)
        net.train()
        t1, tl, ta, n = time.time(), 0.0, 0.0, 0
        for x, y in dl:
            loss, acc = head(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item() * len(y)
            ta += acc.item() * len(y)
            n += len(y)
        net.eval()
        e = val_eer(net, D)
        row = dict(epoch=ep + 1, lr=opt.param_groups[0]["lr"], train_loss=tl / n, train_acc=ta / n, val_eer=e, sec=time.time() - t1)
        log.append(row)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
        if e < best:
            best = e
            torch.save({"cfg": net.cfg, "state": net.state_dict(), "epoch": ep + 1, "val_eer": e},
                       os.path.join(MODELS, "spk_best.pt"))
    pd.DataFrame(log).to_csv(os.path.join(MODELS, "spk_log.csv"), index=False)
    print("best validation EER %.4f, %.0f min" % (best, (time.time() - t0) / 60))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ep = [r["epoch"] for r in log]
    ax[0].plot(ep, [r["train_loss"] for r in log])
    ax[0].set_xlabel("epoch"), ax[0].set_ylabel("AAM-softmax loss (train)"), ax[0].grid(alpha=.3)
    ax[1].plot(ep, [100 * r["val_eer"] for r in log])
    ax[1].set_xlabel("epoch"), ax[1].set_ylabel("validation EER % (5 takes, clean)"), ax[1].grid(alpha=.3)
    fig.suptitle("speaker model training")
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "spk_training_curves.png"), dpi=130)


if __name__ == "__main__":
    main()
