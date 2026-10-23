"""10_train_kw.py - keyword head (Step 2): help / help me / sos / _other.

    python scripts/10_train_kw.py            # -> models/kw.pt, models/kw_log.csv, docs/kw_*.png

All parameters: configs/train.yaml (window, net, augment, train_kw).
The network is kws.model.KwsNet (layer types fixed by the firmware); only
its sizes come from net:. in_mean / in_std are set from training features.
The checkpoint with the best validation balanced accuracy is kept.
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kws.cfg import load, seed_all  # noqa: E402
from kws.dataset import worker_init, Data, FixedKwWindows, KwWindows  # noqa: E402
from kws.model import KwsNet  # noqa: E402

C = load()
K, N = C["train_kw"], C["net"]
MODELS, DOCS = C["paths"]["models"], C["paths"]["docs"]


def loader(ds, shuffle, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=K["batch"], shuffle=shuffle, num_workers=K["workers"],
                                       generator=g, drop_last=shuffle, persistent_workers=False,
                                       worker_init_fn=worker_init)


def norm_stats(ds, n):
    """per-band mean / std of the training features (sets in_mean / in_std)"""
    ds.epoch = 999999                                       # windows not used for training epochs
    X = torch.stack([ds[i][0] for i in range(n)])          # [n, T, 40]
    return X.mean(dim=(0, 1)), X.std(dim=(0, 1))


def evaluate(net, dl, ncls):
    net.eval()
    cm = np.zeros((ncls, ncls), int)
    loss, n = 0.0, 0
    with torch.no_grad():
        for x, y in dl:
            lg, _ = net(x)
            loss += F.cross_entropy(lg, y, reduction="sum").item()
            n += len(y)
            for t, p in zip(y.numpy(), lg.argmax(1).numpy()):
                cm[t, p] += 1
    acc = np.diag(cm).sum() / cm.sum()
    bal = np.mean(np.diag(cm) / np.maximum(cm.sum(1), 1))
    return loss / n, acc, bal, cm


def main():
    seed_all(C)
    torch.set_num_threads(K["torch_threads"])
    labels = C["labels"]
    D = Data(C)
    tr = KwWindows(D, C, "train", K["windows_per_epoch"])
    va = FixedKwWindows(D, C, "val", K["val_windows_per_class"])
    net = KwsNet(ncls=len(labels), T=C["window"]["T"], **N)
    t0 = time.time()
    m, s = norm_stats(tr, K["norm_windows"])
    net.in_mean.copy_(m)
    net.in_std.copy_(s)
    print("in_mean / in_std from %d windows (%.0f s)" % (K["norm_windows"], time.time() - t0), flush=True)
    params = [p for n_, p in net.named_parameters() if not n_.startswith("spk.")]
    opt = torch.optim.AdamW(params, lr=K["lr"], weight_decay=K["weight_decay"])
    steps = K["epochs"] * (K["windows_per_epoch"] // K["batch"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=K["lr"], total_steps=steps, pct_start=K["warmup_frac"],
                                                anneal_strategy="cos", div_factor=K["div_factor"],
                                                final_div_factor=K["final_div_factor"])
    w = torch.tensor(K["class_weights"], dtype=torch.float32)
    dl_va = loader(va, False, C["seed"])
    log, best = [], -1.0
    for ep in range(K["epochs"]):
        tr.epoch = ep
        net.train()
        t1, tl, tc, tn = time.time(), 0.0, 0, 0
        for x, y in loader(tr, True, C["seed"] + ep):
            lg, _ = net(x)
            loss = F.cross_entropy(lg, y, weight=w, label_smoothing=K["label_smoothing"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item() * len(y)
            tc += (lg.argmax(1) == y).sum().item()
            tn += len(y)
        vl, vacc, vbal, cm = evaluate(net, dl_va, len(labels))
        row = dict(epoch=ep + 1, lr=opt.param_groups[0]["lr"], train_loss=tl / tn, train_acc=tc / tn,
                   val_loss=vl, val_acc=vacc, val_bal_acc=vbal, sec=time.time() - t1)
        log.append(row)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
        if vbal > best:
            best = vbal
            torch.save({"cfg": net.cfg, "state": net.state_dict(), "labels": labels, "epoch": ep + 1,
                        "val_bal_acc": vbal, "val_cm": cm.tolist()}, os.path.join(MODELS, "kw.pt"))
    import pandas as pd
    pd.DataFrame(log).to_csv(os.path.join(MODELS, "kw_log.csv"), index=False)
    print("best val balanced accuracy %.4f, total %.0f min" % (best, (time.time() - t0) / 60))
    plots(log, labels)


def plots(log, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ck = torch.load(os.path.join(MODELS, "kw.pt"), weights_only=False)
    ep = [r["epoch"] for r in log]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(ep, [r["train_loss"] for r in log], label="train")
    ax[0].plot(ep, [r["val_loss"] for r in log], label="validation")
    ax[0].set_xlabel("epoch"), ax[0].set_ylabel("cross-entropy"), ax[0].legend(), ax[0].grid(alpha=.3)
    ax[1].plot(ep, [r["train_acc"] for r in log], label="train accuracy")
    ax[1].plot(ep, [r["val_acc"] for r in log], label="validation accuracy")
    ax[1].plot(ep, [r["val_bal_acc"] for r in log], "--", label="validation balanced")
    ax[1].set_xlabel("epoch"), ax[1].set_ylabel("accuracy"), ax[1].legend(), ax[1].grid(alpha=.3)
    fig.suptitle("keyword head training (best epoch %d)" % ck["epoch"])
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "kw_training_curves.png"), dpi=130)
    cm = np.array(ck["val_cm"])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    r = cm / cm.sum(1, keepdims=True)
    ax.imshow(r, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, "%d\n%.1f%%" % (cm[i, j], 100 * r[i, j]), ha="center", va="center",
                    color="white" if r[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xticks(range(len(labels)), labels), ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted"), ax.set_ylabel("true")
    ax.set_title("validation confusion matrix (single window)")
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "kw_confusion_val.png"), dpi=130)


if __name__ == "__main__":
    main()
