"""30_export.py - thresholds -> kws.export -> kws.check_c (Step 4).

    python scripts/30_export.py              # models/<export.model> -> models/best.pt -> out/kws_model.c, out/kws_model_test.c

The chosen round (export.model, e.g. kw_r3.pt) is copied to models/best.pt
without the unused round-1 speaker head (spk_dim = 0, v1.4.1). Thresholds
(thr low / medium / high, cls_off per class), hits, hits_ms and the name come
from configs/train.yaml export: (chosen on the dev set with 20_eval.py --sweep,
see REPORT.md). Also checks on real validation windows that the int8 model
(export.reference = the C arithmetic) agrees with the float model, and
writes the model's size, work memory and multiply-adds to models/export_check.json.
"""
import json
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kws import export  # noqa: E402
from kws.cfg import ROOT, load  # noqa: E402
from kws.dataset import Data, FixedKwWindows  # noqa: E402
from kws.model import KwsNet  # noqa: E402

C = load()
X = C["export"]


def best_pt():
    """models/<export.model> -> models/best.pt with spk_dim = 0"""
    src = torch.load(os.path.join(C["paths"]["models"], X["model"]), map_location="cpu", weights_only=False)
    cfg = dict(src["cfg"], spk_dim=0)
    net = KwsNet(**cfg)
    net.load_state_dict({k: v for k, v in src["state"].items() if not k.startswith("spk.")})
    path = os.path.join(C["paths"]["models"], "best.pt")
    torch.save({"cfg": net.cfg, "state": net.state_dict(), "labels": C["labels"], "from": X["model"],
                "epoch": src.get("epoch")}, path)
    return path


def budget(net, L):
    """weights (bytes), work memory (floats, as kws_engine_work) and multiply-adds of one run"""
    c = net.cfg
    T, B = c["T"], 40
    macs = T * B * c["c1"] * 9 + (T // 2) * (B // 2) * c["c1"] * c["c2"] * 9 + (T // 4) * (B // 4) * c["c2"] * c["c3"] * 9 \
        + c["c3"] * c["hid"] + c["hid"] * c["ncls"]
    wbytes = sum(L[k][0].size + 4 * (1 + L[k][2].size) for k in L) + 8 * B
    work = T * B + c["c1"] * (T // 2) * (B // 2) + c["c2"] * (T // 4) * (B // 4)       # kws_engine_work()
    return dict(macs=int(macs), weight_bytes=int(wbytes), work_floats=int(work), work_bytes=int(4 * work),
                detector_buffers_bytes=int(2 * 4 * T * B))                              # ring + win of kws_det_t


def int8_net(net, L):
    """a KwsNet that computes what the C engine computes: BatchNorm folded, int8 weights x scale
    (saved as models/best_int8sim.pt; 20_eval.py --model models/best_int8sim.pt gives device-exact numbers)"""
    q = KwsNet(**net.cfg)
    q.load_state_dict(net.state_dict())
    with torch.no_grad():
        for i, (conv, bn) in enumerate([(q.conv1, q.bn1), (q.conv2, q.bn2), (q.conv3, q.bn3)], 1):
            w8, sc, b = L["%d" % i]
            conv.weight.copy_(torch.from_numpy(w8.astype(np.float32) * sc))
            conv.bias.copy_(torch.from_numpy(b))
            bn.running_mean.zero_()
            bn.running_var.fill_(1.0 - bn.eps)
            bn.weight.fill_(1.0)
            bn.bias.zero_()
        for k, lin in (("4", q.fc1), ("5", q.fc2)):
            w8, sc, b = L[k]
            lin.weight.copy_(torch.from_numpy(w8.astype(np.float32) * sc))
            lin.bias.copy_(torch.from_numpy(b))
    return q.eval()


def main():
    model = best_pt()
    out = C["paths"]["out"]
    labels = "|".join(C["labels"])
    off = ",".join("%s=%.4f" % (k, v) for k, v in X["cls_off"].items())
    cmd = [sys.executable, "-m", "kws.export", model, out, "--labels", labels,
           "--thr"] + ["%.4f" % t for t in X["thr"]] + ["--cls-off", off, "--spk-thr", "%.4f" % X["spk_thr"],
           "--hits", str(X["hits"]), "--hits-ms", str(X["hits_ms"]), "--name", X["name"]]
    print(" ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    # int8 (C arithmetic) against float on real validation windows
    ck = torch.load(model, map_location="cpu", weights_only=False)
    net = KwsNet(**ck["cfg"])
    net.load_state_dict(ck["state"])
    net.eval()
    L = export.layers(net)
    q = int8_net(net, L)
    torch.save({"cfg": q.cfg, "state": q.state_dict(), "labels": C["labels"], "note": "int8-equivalent of best.pt"},
               os.path.join(C["paths"]["models"], "best_int8sim.pt"))
    va = FixedKwWindows(Data(C), C, "val", X["check_windows_per_class"], tag="exportcheck")
    dp, same, ecos, dq = [], 0, [], []
    for i in range(len(va)):
        f, _ = va[i]
        p8, e8 = export.reference(net, L, f.numpy())
        with torch.no_grad():
            lg, emb = net(f[None])
        pf = torch.softmax(lg, 1)[0].numpy()
        dp.append(np.abs(pf - p8).max())
        with torch.no_grad():
            dq.append(float(np.abs(torch.softmax(q(f[None])[0], 1)[0].numpy() - p8).max()))
        same += int(pf.argmax() == p8.argmax())
        if emb is not None:
            ecos.append(float(np.dot(emb[0].numpy(), e8)))
    r = dict(**budget(net, L), windows=len(va), max_prob_diff=float(np.max(dp)), mean_prob_diff=float(np.mean(dp)),
             argmax_agree=same / len(va), int8sim_vs_reference_max_diff=float(np.max(dq)), emb_cos_min=float(np.min(ecos)) if ecos else None,
             emb_cos_mean=float(np.mean(ecos)) if ecos else None)
    json.dump(r, open(os.path.join(C["paths"]["models"], "export_check.json"), "w"), indent=1)
    print("int8 vs float:", json.dumps(r), flush=True)
    rc = subprocess.run([sys.executable, "-m", "kws.check_c", out], cwd=ROOT).returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()
