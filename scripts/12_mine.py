"""12_mine.py - hard negatives (Step A2) + more held-out running speech for the
false-alarm test.

    python scripts/12_mine.py                  # round 1 model -> hardneg.*      (configs/train.yaml mine:)
    python scripts/12_mine.py --stage r3       # round 2 model -> hardneg_r3.*   (mine: with mine_r3: replacing values)

Streams LibriSpeech training shards (the same files as 01_download.py, SHA-256
checked, deleted after use) and, per utterance, by the speaker split:
  * train speakers: the keyword model (mine.model) looks at the speech the way
    the device does (a window every 240 ms); windows where a keyword class
    reaches mine.threshold (and no real "help" is said there) become hard
    negatives -> data/cache/hardneg.npy / .csv (group hard_neg, label _other);
  * test / val speakers (never trained on): whole utterances are kept, with
    every word, as extra false-alarm speech -> data/cache/fa_{test,dev}_extra.*
    (up to mine.fa_extra_hours; independent of the model).
The training noise bank is mined the same way (group hard_neg_noise).
"""
import io
import json
import os
import sys
import threading
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kws.cfg import ROOT, load, speaker_split  # noqa: E402
from kws.evaluate import batch_run, run_frames  # noqa: E402
from kws.features import HOP, logmel  # noqa: E402
from kws.model import KwsNet  # noqa: E402

import argparse  # noqa: E402

_ap = argparse.ArgumentParser()
_ap.add_argument("--stage", default="r2", help="r2: hard negatives for round 2 (default), r3: for round 3")
STAGE = _ap.parse_args().stage
C = load()
M = dict(C["mine"], **C.get("mine_" + STAGE, {}))
L = C["sources"]["librispeech"]
CACHE = C["paths"]["cache"]
dl = __import__("01_download")
prep = __import__("02_prepare")
RATE = 16000


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_net(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])
    net = KwsNet(**cfg)
    net.load_state_dict(ck["state"])
    return net.eval()


def mine_audio(net, x, kw_spans, kmask):
    """windows of x (int16) where a keyword class >= threshold -> [(start, end, prob)] in samples"""
    T = net.cfg["T"]
    f = logmel(x)
    if len(f) < T:
        return []
    fr = run_frames(T, len(f))
    p, _ = batch_run(net, f, fr, T)
    pk = p[:, kmask].max(1)
    out, ends = [], []
    for j in np.argsort(-pk):
        if pk[j] < M["threshold"]:
            break
        end = int(fr[j]) * HOP + 480
        a = max(0, end - int(M["clip_s"] * RATE) + int(M["clip_after_s"] * RATE))
        b = min(len(x), a + int(M["clip_s"] * RATE))
        if any(s < b and e > a for s, e in kw_spans):
            continue
        if any(abs(end - u) < M["min_gap_s"] * RATE for u in ends):
            continue
        ends.append(end)
        out.append((a, b, float(pk[j])))
    return out


def shard_list():
    files = dl.hf_files(L["hf_repo"], "data")
    out = []
    for sub, n in M["shards"].items():
        skip = M.get("skip_shards", {}).get(sub, 0)
        fs = sorted(p for p in files if os.path.basename(p).startswith(sub + "-"))[skip:skip + n]
        out += [(p, files[p]) for p in fs]
    return out


def fetch(path, info, dst):
    h, n = dl.download(dl.hf_url(L["hf_repo"], path), dst, "sha256")
    dl.check(path, h, info.get("lfs", {}).get("oid"))


def main():
    torch.set_num_threads(M["torch_threads"])
    net = load_net(os.path.join(ROOT, M["model"]))
    kmask = np.array([not l.startswith("_") for l in C["labels"]])
    hn = prep.Store(M["out"])
    keep_extra = sum(M["fa_extra_hours"].values()) > 0
    extra = {"test": prep.Store("fa_test_extra"), "dev": prep.Store("fa_dev_extra")} if keep_extra else {}
    words = {"test": [], "dev": []}
    cap = {k: v * 3600 * RATE for k, v in M["fa_extra_hours"].items()}
    shards = shard_list()
    tmp = [os.path.join(CACHE, "mine_%d.parquet" % i) for i in range(2)]
    th = threading.Thread(target=fetch, args=(*shards[0], tmp[0]))
    th.start()
    t0, stats = time.time(), dict(utts=0, train_hours=0.0, hardneg=0)
    for i, (path, info) in enumerate(shards):
        th.join()
        cur = tmp[i % 2]
        if i + 1 < len(shards):
            th = threading.Thread(target=fetch, args=(*shards[i + 1], tmp[(i + 1) % 2]))
            th.start()
        rows = []
        for r in pd.read_parquet(cur, columns=["id", "audio", "words"]).itertuples():
            spk = "ls-" + r.id.split("-")[0]
            split = speaker_split(C, spk)
            x, sr = sf.read(io.BytesIO(r.audio["bytes"]), dtype="int16")
            W = [(w["word"].lower(), int(w["start"] * sr), int(w["end"] * sr)) for w in r.words if w["word"]]
            if split == "train":
                stats["utts"] += 1
                stats["train_hours"] += len(x) / RATE / 3600
                kw = [(s - int(0.3 * sr), e + int(0.3 * sr)) for w, s, e in W if w in L["keyword_words"]]
                for a, b, p in mine_audio(net, x, kw, kmask):
                    ws = " ".join(w for w, s, e in W if s < b and e > a)
                    rows.append((p, x[a:b], dict(source="librispeech", group=M["group"], label="_other", word=ws[:60],
                                                speaker=spk, gender="", split="train", spk_use=False, text=r.id,
                                                kw_start=0, kw_end=b - a, prob=round(p, 4))))
            else:
                name = "test" if split == "test" else "dev"
                if keep_extra and extra[name].pos < cap[name]:
                    base = extra[name].pos
                    words[name] += [dict(utt=r.id, word=w, at=base + s, end=base + e) for w, s, e in W]
                    extra[name].add(x, utt=r.id, speaker=spk, subset=os.path.basename(path).split("-")[0])
        rows.sort(key=lambda t: -t[0])
        for p, seg, meta in rows[:M["per_shard_max"]]:
            hn.add(seg, **meta)
        stats["hardneg"] = len(hn.rows)
        os.remove(cur)
        log("%d/%d %s: %d hard negatives (total %d), extra test %.1f h dev %.1f h, %.0f min"
            % (i + 1, len(shards), os.path.basename(path), min(len(rows), M["per_shard_max"]), len(hn.rows),
               *[extra[k].pos / RATE / 3600 if keep_extra else 0.0 for k in ("test", "dev")], (time.time() - t0) / 60))
    # training noise bank
    from kws.dataset import Data
    D = Data(C)
    nd, na = D.noise["train"]
    nn = 0
    for r in nd.itertuples():
        x = np.array(na[r.offset:r.offset + r.length])
        found = mine_audio(net, x, [], kmask)[:M["noise_per_file_max"]]
        for a, b, p in found:
            hn.add(x[a:b], source=r.source, group=M["group"] + "_noise", label="_other", word=r.file[:60], speaker="noise",
                   gender="", split="train", spk_use=False, text=r.file, kw_start=0, kw_end=b - a, prob=round(p, 4))
        nn += len(found)
    log("noise: %d hard negatives" % nn)
    hn.save(prep.COLS + ["prob"])
    for k in extra:
        extra[k].save()
        pd.DataFrame(words[k], columns=["utt", "word", "at", "end"]).to_csv(
            os.path.join(CACHE, "fa_%s_extra_words.csv" % k), index=False)
    stats.update(noise_hardneg=nn, fa_extra_hours={k: round(v.pos / RATE / 3600, 2) for k, v in extra.items()},
                 minutes=round((time.time() - t0) / 60, 1))
    json.dump(stats, open(os.path.join(CACHE, "mine_stats_%s.json" % STAGE), "w"), indent=1)
    log(json.dumps(stats))


if __name__ == "__main__":
    main()
