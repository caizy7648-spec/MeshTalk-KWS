"""42_export_spk.py - speaker model -> out/spk_model.c + out/spk_model_test.c,
then the C test (spk/check_c.py: spk_engine.c == Python).

    python scripts/42_export_spk.py
    python scripts/42_export_spk.py --model models/spk_d2.pt --name spk-d2-YYYYMMDD --threshold 0.52   # Phase D

Threshold, enrolment takes and name: configs/spk.yaml export:.
Also copies spk/c/spk_engine.c and spk/c/spk_model.h (-> firmware Ai/ with
spk_model.c) and the C test spk/c/spk_test.c (-> Tests/ with spk_model_test.c) to out/.
"""
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kws.cfg import ROOT  # noqa: E402
from spk import export  # noqa: E402
from spk.data import load_cfg  # noqa: E402

C, P = load_cfg()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(C["paths"]["models"], "spk_best.pt"), help="Phase D: models/spk_d2.pt etc.")
    ap.add_argument("--name", help="spk_model name (default: export.name of configs/spk.yaml)")
    ap.add_argument("--threshold", type=float, help="default: export.threshold")
    a = ap.parse_args()
    out = C["paths"]["out"]
    net = export.load(a.model)
    L = export.layers(net)
    x = dict(P["export"])
    if a.name:
        x["name"] = a.name
    if a.threshold is not None:
        x["threshold"] = a.threshold
    mid = export.write_model(net, L, os.path.join(out, "spk_model.c"), x["name"], x["threshold"], x["enrol_takes"])
    export.write_test(net, L, os.path.join(out, "spk_model_test.c"))
    for f in ("spk_engine.c", "spk_model.h", "spk_test.c"):
        shutil.copy(os.path.join(ROOT, "spk", "c", f), out)
    info = dict(id="%08X" % mid, name=x["name"], model=os.path.relpath(a.model, ROOT), **export.sizes(net, L),
                threshold=x["threshold"], enrol_takes=x["enrol_takes"])
    json.dump(info, open(os.path.join(C["paths"]["models"], "spk_export.json"), "w"), indent=1)
    print("spk_model.c:", info, flush=True)
    sys.exit(subprocess.run([sys.executable, "-m", "spk.check_c", out], cwd=ROOT).returncode)


if __name__ == "__main__":
    main()
