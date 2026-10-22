"""30_export.py - thresholds -> kws.export -> kws.check_c (Step 4).

    python scripts/30_export.py              # models/best.pt -> out/kws_model.c, out/kws_model_test.c

Thresholds (thr low / medium / high, spk_thr), hits, hits_ms and the model
name come from configs/train.yaml export: (chosen in step 4 from the dev
curves, see TRAINING.md). Also checks on real validation windows that the
int8 model (export.reference = the C arithmetic) agrees with the float model.
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


def main():
    model = os.path.join(C["paths"]["models"], "best.pt")
    out = C["paths"]["out"]
    labels = "|".join(C["labels"])
    cmd = [sys.executable, "-m", "kws.export", model, out, "--labels", labels,
           "--thr"] + ["%.4f" % t for t in X["thr"]] + ["--spk-thr", "%.4f" % X["spk_thr"],
           "--hits", str(X["hits"]), "--hits-ms", str(X["hits_ms"]), "--name", X["name"]]
    print(" ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    # int8 (C arithmetic) against float on real validation windows
    ck = torch.load(model, map_location="cpu", weights_only=False)
    net = KwsNet(**ck["cfg"])
    net.load_state_dict(ck["state"])
    net.eval()
    L = export.layers(net)
    va = FixedKwWindows(Data(C), C, "val", X["check_windows_per_class"], tag="exportcheck")
    dp, same, ecos = [], 0, []
    for i in range(len(va)):
        f, _ = va[i]
        p8, e8 = export.reference(net, L, f.numpy())
        with torch.no_grad():
            lg, emb = net(f[None])
        pf = torch.softmax(lg, 1)[0].numpy()
        dp.append(np.abs(pf - p8).max())
        same += int(pf.argmax() == p8.argmax())
        if emb is not None:
            ecos.append(float(np.dot(emb[0].numpy(), e8)))
    r = dict(windows=len(va), max_prob_diff=float(np.max(dp)), mean_prob_diff=float(np.mean(dp)),
             argmax_agree=same / len(va), emb_cos_min=float(np.min(ecos)) if ecos else None,
             emb_cos_mean=float(np.mean(ecos)) if ecos else None)
    json.dump(r, open(os.path.join(C["paths"]["models"], "export_check.json"), "w"), indent=1)
    print("int8 vs float:", json.dumps(r), flush=True)
    rc = subprocess.run([sys.executable, "-m", "kws.check_c", out], cwd=ROOT).returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()
