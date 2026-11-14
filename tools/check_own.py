"""check_own.py - r3 listens to every one of the owner's takes (firmware v1.6.7 detector).

    python tools/check_own.py                    # after 04_import_own.py
    python tools/check_own.py --model models/kw_r4.pt            # thresholds of export_r4

Each take is played once (1 s of microphone noise before, 2 s after) through kws/fwdet.py
with the device settings (medium, help +0.05). Written to data/own/check_r3.csv:
  file, label, style, split, heard_as (keywords heard, in order), best score, verdict
verdict: ok | missed (nothing heard) | other_keyword (only a different keyword) | two_calls.
A take that r3 hears as a DIFFERENT keyword with a high score, or as two calls, may be a wrong
take (said the wrong word, two calls in one take) - show those to the owner before training.
"missed" takes are normal for a model that has never heard this microphone: that is exactly
what r4 should learn. Also prints the per-label / per-style hit rate of r3 on the owner's voice
(all splits): the first "before" number.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kws.audio import to_float, to_int16  # noqa: E402
from kws.cfg import ROOT, load  # noqa: E402
from kws.fwdet import EV_HIT, FwDet  # noqa: E402
from kws.model import KwsNet  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/kw_r3.pt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--from-export", default=None, help="thresholds from this config section (default: export for r3, export_r4 for r4)")
    a = ap.parse_args()
    C = load()
    b = os.path.basename(a.model)
    sec = a.from_export or ("export_r4c" if "r4c" in b else "export_r4b" if "r4b" in b else "export_r4" if "r4" in b else "export")
    X, E = C[sec], C["eval_r4"]
    own = os.path.join(ROOT, C["own"]["dir"])
    man = pd.read_csv(os.path.join(own, "manifest.csv"), keep_default_na=False)
    ck = torch.load(os.path.join(ROOT, a.model) if not os.path.isabs(a.model) else a.model, map_location="cpu", weights_only=False)
    net = KwsNet(**dict(ck["cfg"]))
    net.load_state_dict(ck["state"], strict=False)
    net.eval()
    torch.set_num_threads(4)
    labels = C["labels"]
    det = FwDet(net, thr=X["thr"], cls_off=[X["cls_off"].get(l, 0.0) for l in labels], help_extra=X.get("help_extra", E["device"]["help_extra"]))
    rng = np.random.default_rng(0)
    rows = []
    import soundfile as sf
    for r in man.itertuples():
        x, sr = sf.read(os.path.join(own, r.path), dtype="int16")
        y = np.concatenate([np.zeros(16000), to_float(x), np.zeros(32000)]).astype(np.float32)
        y += rng.standard_normal(len(y)).astype(np.float32) * 10 ** (E["noise_floor_dbfs"] / 20)
        ev = det.run(to_int16(y))
        det.reset()
        hits = [(labels[c], s) for t, f, c, s in ev if f & EV_HIT]
        heard = [h for h, _ in hits]
        if not hits:
            v = "missed"
        elif len(hits) > 1:
            v = "two_calls"
        elif r.label not in heard:
            v = "other_keyword"
        else:
            v = "ok"
        rows.append(dict(file=r.path, label=r.label, style=r.style, speaker=r.speaker, split=r.split,
                         heard_as=" + ".join(heard), best=round(max([s for _, s in hits], default=0.0), 3), verdict=v))
    df = pd.DataFrame(rows)
    out = a.out or os.path.join(own, "check_%s.csv" % os.path.splitext(os.path.basename(a.model))[0].replace("kw_", ""))
    df.to_csv(out, index=False)
    print("written", out)
    print("\nverdicts:\n", df.groupby(["label", "verdict"]).size().unstack(fill_value=0).to_string())
    print("\nheard as the right keyword (%):\n",
          (100 * df.assign(ok=df.verdict.eq("ok")).groupby(["label", "style"]).ok.mean()).round(1).unstack().to_string())
    sus = df[df.verdict.isin(["other_keyword", "two_calls"])]
    if len(sus):
        print("\nmay be wrong takes (listen to them):")
        print(sus.to_string(index=False))


if __name__ == "__main__":
    main()
