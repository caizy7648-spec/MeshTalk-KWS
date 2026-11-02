"""44_teacher.py - teacher voice vectors for distillation (phase B2) and gender labels.

    /home/user/venv_teacher/bin/python scripts/44_teacher.py     # separate venv: speechbrain + torch/torchaudio 2.11 CPU

Teacher: SpeechBrain spkrec-ecapa-voxceleb (ECAPA-TDNN, trained on VoxCeleb 1+2,
Apache-2.0), 192-dim vectors, computed on the CLEAN audio (no device effects):
the student later sees the same audio through the device chain and has to
land where the teacher puts the clean voice -> robust to distance, room, level.

Bank (configs/spk.yaml finetune.teacher):
  * labelled: train-split speakers of the speaker model (MSWC + LibriSpeech),
    per_speaker clips each (hash order), and VCTK train speakers;
  * unlabelled: People's Speech segments (distillation only);
  * for the gender model: gender_clips clips of every val / test speaker.
Gender: LibriSpeech SPEAKERS.TXT and MSWC labels are true labels; a logistic
regression on teacher vectors (trained on LibriSpeech train speakers, checked
on LibriSpeech val / test speakers) labels everyone else (VCTK, People's Speech
segments, MSWC speakers without a label).
Writes data/cache/distill_bank.csv + distill_teacher.npy, data/cache/spk_gender.csv,
models/teacher_gender.json (accuracy).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kws.cfg import ROOT, load, stable_hash  # noqa: E402

import yaml  # noqa: E402

C = load()
P = yaml.safe_load(open(os.path.join(ROOT, "configs", "spk.yaml")))
T = P["finetune"]["teacher"]
CACHE = C["paths"]["cache"]
RATE = 16000


def stores():
    out = {}
    for name in ("clips", "spk_extra"):
        df = pd.read_csv(os.path.join(CACHE, name + ".csv"), keep_default_na=False)
        out[name] = (df, np.load(os.path.join(CACHE, name + ".npy"), mmap_mode="r"))
    return out


def segment(df, a, i, max_s):
    """clean audio of row i, centred on its spoken part, at most max_s"""
    r = df.iloc[i]
    x = np.array(a[r.offset:r.offset + r.length]).astype(np.float32) / 32768.0
    n = int(max_s * RATE)
    if len(x) > n:
        c = (int(r.kw_start) + int(r.kw_end)) // 2
        o = min(max(0, c - n // 2), len(x) - n)
        x = x[o:o + n]
    return x


def pick(S):
    clips, _ = S["clips"]
    extra, _ = S["spk_extra"]
    rows = []
    sp = clips[clips.spk_use.astype(str) == "True"]
    for split, g in sp.groupby("split"):
        k = T["per_speaker"] if split == "train" else T["gender_clips"]
        for s, gg in g.groupby("speaker"):
            idx = sorted(gg.index, key=lambda i: stable_hash("tb:%d" % i))[:k]
            rows += [dict(store="clips", row=i, speaker=s, split=split, labelled=split == "train") for i in idx]
    for r in extra.itertuples():
        if r.source == "vctk":
            rows.append(dict(store="spk_extra", row=r.Index, speaker=r.speaker, split=r.split, labelled=r.split == "train"))
        else:
            rows.append(dict(store="spk_extra", row=r.Index, speaker="", split="train", labelled=False))
    return pd.DataFrame(rows)


def embed(model, S, bank):
    out = np.zeros((len(bank), 192), np.float32)
    lens = []
    segs = {}
    for j, r in enumerate(bank.itertuples()):
        df, a = S[r.store]
        segs[j] = segment(df, a, r.row, T["max_s"])
        lens.append(len(segs[j]))
    order = np.argsort(lens)
    t0 = time.time()
    for b in range(0, len(order), T["batch"]):
        ids = order[b:b + T["batch"]]
        n = max(lens[i] for i in ids)
        x = torch.zeros(len(ids), n)
        wl = torch.zeros(len(ids))
        for k, i in enumerate(ids):
            x[k, :lens[i]] = torch.from_numpy(segs[i])
            wl[k] = lens[i] / n
        with torch.no_grad():
            e = model.encode_batch(x, wl)[:, 0].numpy()
        out[ids] = e
        if b // T["batch"] % 50 == 0:
            print("teacher %d / %d (%.0f min)" % (b, len(order), (time.time() - t0) / 60), flush=True)
    return out


def genders(S, bank, emb):
    """true labels where known, else logistic regression on the teacher vectors"""
    clips, _ = S["clips"]
    lg = pd.read_csv(os.path.join(CACHE, "libri_gender.csv"))
    true = dict(zip(lg.speaker, lg.gender))
    for s, g in clips[clips.source == "mswc"].groupby("speaker"):
        v = g.gender.iloc[0]
        if v in ("MALE", "FEMALE"):
            true[s] = v
    e = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    y = np.array([{"MALE": 0.0, "FEMALE": 1.0}.get(true.get(s, ""), np.nan) for s in bank.speaker])
    lib = bank.speaker.str.startswith("ls-").to_numpy()
    tr = lib & (bank.split == "train").to_numpy() & ~np.isnan(y)
    te = lib & (bank.split != "train").to_numpy() & ~np.isnan(y)
    X = torch.from_numpy(e)
    w = torch.zeros(192, requires_grad=True)
    b0 = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b0], max_iter=200)
    Xt, yt = X[tr], torch.from_numpy(y[tr]).float()

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b0, yt) + 1e-3 * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    p = torch.sigmoid(X @ w + b0).detach().numpy()
    acc = float(np.mean((p[te] > 0.5) == (y[te] > 0.5)))
    mswc = (bank.speaker.str.startswith("mswc-") & ~np.isnan(y)).to_numpy()
    acc_mswc = float(np.mean((p[mswc] > 0.5) == (y[mswc] > 0.5)))
    rows = []
    for s, idx in bank[bank.speaker != ""].groupby("speaker").groups.items():
        if s in true:
            rows.append(dict(speaker=s, gender=true[s], source="label", p_female=float(np.mean(p[idx]))))
        else:
            q = float(np.mean(p[idx]))
            rows.append(dict(speaker=s, gender="FEMALE" if q > 0.5 else "MALE", source="predicted", p_female=q))
    seg_gender = np.where(p > 0.5, "FEMALE", "MALE")
    return pd.DataFrame(rows), seg_gender, dict(libri_heldout_segments=int(te.sum()), libri_heldout_accuracy=round(acc, 4),
                                                mswc_labelled_segments=int(mswc.sum()), mswc_accuracy=round(acc_mswc, 4))


def main():
    torch.set_num_threads(T["threads"])
    from speechbrain.inference.speaker import EncoderClassifier
    model = EncoderClassifier.from_hparams(source=T["model"], savedir=os.path.join(CACHE, "teacher_ecapa"),
                                           run_opts={"device": "cpu"})
    S = stores()
    bank = pick(S)
    print("bank:", dict(bank.groupby(["store", "split"]).size()), flush=True)
    emb = embed(model, S, bank)
    gtab, seg_gender, acc = genders(S, bank, emb)
    g = dict(zip(gtab.speaker, gtab.gender))
    bank["gender"] = [g.get(s, sg) if s else sg for s, sg in zip(bank.speaker, seg_gender)]
    bank.to_csv(os.path.join(CACHE, "distill_bank.csv"), index=False)
    np.save(os.path.join(CACHE, "distill_teacher.npy"), emb)
    gtab.to_csv(os.path.join(CACHE, "spk_gender.csv"), index=False)
    json.dump(dict(acc, teacher=T["model"], bank=len(bank),
                   gender_counts={k: int(v) for k, v in bank[bank.labelled].gender.value_counts().items()}),
              open(os.path.join(C["paths"]["models"], "teacher_gender.json"), "w"), indent=1)
    print(json.dumps(acc), flush=True)


if __name__ == "__main__":
    main()
