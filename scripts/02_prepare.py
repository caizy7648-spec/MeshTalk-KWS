"""02_prepare.py - data/raw -> data/cache (Step 1): decode, resample to 16 kHz
int16, find the spoken part, label, split by speaker.

    python scripts/02_prepare.py                    # everything
    python scripts/02_prepare.py --parts clips      # only clips.npy/.csv (noise / fa unchanged)

Writes (all int16, 16 kHz, clips one after the other + a CSV index):
  clips.npy / clips.csv     every short clip (MSWC, LibriSpeech cuts, Speech
                            Commands, Kokoro TTS) with label, group, speaker,
                            split, kw_start / kw_end (spoken part), spk_use
  noise_{train,test}.npy/.csv   noise banks (MUSAN noise + music, ESC-50, SC background)
  fa_{dev,test}.npy/.csv    running speech for false alarms (LibriSpeech dev / test)
  fa_{dev,test}_kw.csv      where a real "help" is said in that speech (not a false alarm)
  counts.json               numbers per class / source / split (for REPORT.md)
"""
import glob
import io
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kws.audio import resample, rms_db, speech_span, to_int16  # noqa: E402
from kws.cfg import load, speaker_split, stable_hash  # noqa: E402

C = load()
RAW, CACHE = C["paths"]["raw"], C["paths"]["cache"]
P = C["prepare"]
COLS = ["source", "group", "label", "word", "speaker", "gender", "split", "spk_use", "text", "offset", "length", "kw_start", "kw_end"]


class Store:
    """int16 clips one after the other, written straight to <name>.npy, + index rows"""

    def __init__(self, name):
        self.name, self.rows, self.pos = name, [], 0
        self.path = os.path.join(CACHE, name + ".npy")
        self.f = open(self.path + ".part", "wb")
        self.hlen = len(self._header(10 ** 12))
        self.f.write(b"\0" * self.hlen)

    @staticmethod
    def _header(n):
        b = io.BytesIO()
        np.lib.format.write_array_header_1_0(b, {"descr": "<i2", "fortran_order": False, "shape": (n,)})
        return b.getvalue()

    def add(self, x, **meta):
        x = np.asarray(x, np.int16)
        self.rows.append(dict(meta, offset=self.pos, length=len(x)))
        self.f.write(x.astype("<i2").tobytes())
        self.pos += len(x)

    def save(self, cols=None):
        h = self._header(self.pos)
        assert len(h) == self.hlen
        self.f.seek(0)
        self.f.write(h)
        self.f.close()
        os.replace(self.path + ".part", self.path)
        df = pd.DataFrame(self.rows)
        if cols:
            df = df.reindex(columns=cols)
        df.to_csv(os.path.join(CACHE, self.name + ".csv"), index=False)
        return df


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ MSWC
def mswc_decode(link):
    x, sr = sf.read(os.path.join(RAW, "mswc", "clips", link), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return to_int16(resample(x, sr))


def mswc(st):
    sel = pd.read_csv(os.path.join(RAW, "mswc", "selected.csv"))
    kws = set(C["sources"]["mswc"]["keyword_words"])
    pool = Pool(4)
    for r, x in zip(sel.itertuples(), pool.imap(mswc_decode, list(sel.LINK), chunksize=256)):
        if rms_db(x) < P["min_rms_db"]:
            continue
        s, e = speech_span(x, P["span_rel_db"])
        if r.WORD in kws:
            label, group = r.WORD, "mswc_help"
        else:
            label, group = "_other", ("mswc_lookalike" if r.role == "lookalike" else "mswc_random")
        st.add(x, source="mswc", group=group, label=label, word=r.WORD, speaker="mswc-" + r.SPEAKER[:16],
               gender=r.GENDER, split=speaker_split(C, "mswc-" + r.SPEAKER), text="", kw_start=s, kw_end=e)
    pool.close()
    log("mswc", len(sel))


# ------------------------------------------------------------------ LibriSpeech (cut by 01_download.py)
LS_KIND = {  # kind -> (label, group)
    "help": ("help", "libri_help"), "help_of_help_me": ("help", "libri_help"), "help_me": ("help me", "libri_help_me"),
    "me": ("_other", "libri_me"), "lookalike": ("_other", "libri_words"), "spk_word": ("_other", "libri_words"),
    "chunk": ("_other", "libri_chunk")}


def librispeech(st):
    n = 0
    for f in sorted(glob.glob(os.path.join(RAW, "librispeech", "cut", "*.csv"))):
        a = np.load(f.replace(".csv", ".npy"), mmap_mode="r")
        for r in pd.read_csv(f, keep_default_na=False).itertuples():
            x = np.array(a[r.offset:r.offset + r.length])
            if rms_db(x) < P["min_rms_db"]:
                continue
            label, group = LS_KIND[r.kind]
            st.add(x, source="librispeech", group=group, label=label, word=r.word, speaker=r.speaker, gender="",
                   split=speaker_split(C, r.speaker), text=r.utt, kw_start=r.kw_start, kw_end=r.kw_end)
            n += 1
    log("librispeech", n)


# ------------------------------------------------------------------ Speech Commands
def speech_commands(st):
    files = sorted(glob.glob(os.path.join(RAW, "speech_commands", "*", "*.wav")))
    files = [f for f in files if "_background_noise_" not in f]
    files = sorted(files, key=lambda f: stable_hash(os.path.relpath(f, RAW)))[:C["sources"]["speech_commands"]["other_clips"]]
    for f in files:
        x, sr = sf.read(f, dtype="int16")
        if rms_db(x) < P["min_rms_db"]:
            continue
        s, e = speech_span(x, P["span_rel_db"])
        spk = "sc-" + os.path.basename(f).split("_nohash_")[0]
        st.add(x, source="speech_commands", group="speech_commands", label="_other", word=os.path.basename(os.path.dirname(f)),
               speaker=spk, gender="", split=speaker_split(C, spk), text="", kw_start=s, kw_end=e)
    log("speech_commands", len(files))


# ------------------------------------------------------------------ TTS (03_tts.py)
def tts(st):
    for split in ("train", "val", "test"):
        a = np.load(os.path.join(CACHE, "tts", split + ".npy"), mmap_mode="r")
        for r in pd.read_csv(os.path.join(CACHE, "tts", split + ".csv"), keep_default_na=False).itertuples():
            st.add(np.array(a[r.offset:r.offset + r.length]), source="tts", group="tts" if r.label != "_other" else "tts_other",
                   label=r.label, word=r.label, speaker="tts-" + r.voice, gender="", split=split, text=r.text,
                   kw_start=r.kw_start, kw_end=r.kw_end)
    log("tts done")


# ------------------------------------------------------------------ noise banks
def noise():
    banks = {"train": Store("noise_train"), "test": Store("noise_test")}
    tp = C["sources"]["musan"]["test_pct"]
    for f in sorted(glob.glob(os.path.join(RAW, "musan", "*", "*", "*.wav"))):
        rel = os.path.relpath(f, os.path.join(RAW, "musan"))
        x, sr = sf.read(f, dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        split = "test" if stable_hash("musan:" + rel) % 100 < tp else "train"
        banks[split].add(to_int16(resample(x, sr)), source="musan_" + rel.split("/")[0], file=rel)
    tf = C["sources"]["esc50"]["test_fold"]
    for f in sorted(glob.glob(os.path.join(RAW, "esc50", "*.parquet"))):
        for r in pd.read_parquet(f).itertuples():
            x, sr = sf.read(io.BytesIO(r.audio["bytes"]), dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            banks["test" if r.fold == tf else "train"].add(to_int16(resample(x, sr)), source="esc50", file=r.filename)
    for f in sorted(glob.glob(os.path.join(RAW, "speech_commands", "_background_noise_", "*.wav"))):
        x, sr = sf.read(f, dtype="int16")
        banks["train"].add(x, source="sc_background", file=os.path.basename(f))
    out = {}
    for k, b in banks.items():
        df = b.save()
        out[k] = {s: round(g.length.sum() / 16000 / 3600, 2) for s, g in df.groupby("source")}
    log("noise hours", out)
    return out


# ------------------------------------------------------------------ false-alarm speech
def fa_speech():
    out = {}
    for name, subsets in P["fa_subsets"].items():
        st, kw = Store("fa_" + name), []
        for sub in subsets:
            for f in sorted(glob.glob(os.path.join(RAW, "librispeech", "eval", sub + "-*.parquet"))):
                for r in pd.read_parquet(f, columns=["id", "audio", "words"]).itertuples():
                    x, sr = sf.read(io.BytesIO(r.audio["bytes"]), dtype="int16")
                    base = st.pos
                    for w in r.words:
                        if w["word"] and w["word"].lower() in C["sources"]["librispeech"]["keyword_words"]:
                            kw.append(dict(utt=r.id, word=w["word"], at=base + int(w["start"] * sr), end=base + int(w["end"] * sr)))
                    st.add(x, utt=r.id, speaker="ls-" + r.id.split("-")[0], subset=sub)
        st.save()
        pd.DataFrame(kw, columns=["utt", "word", "at", "end"]).to_csv(os.path.join(CACHE, "fa_%s_kw.csv" % name), index=False)
        out[name] = dict(hours=round(st.pos / 16000 / 3600, 2), utterances=len(st.rows), real_keywords=len(kw))
        log("fa", name, out[name])
    return out


def clips(counts):
    st = Store("clips")
    mswc(st)
    librispeech(st)
    speech_commands(st)
    tts(st)
    df = st.save(COLS)
    # speaker head: real voices with enough clips (never TTS)
    real = df[df.source.isin(["mswc", "librispeech"])]
    n = real.groupby("speaker").size()
    good = set(n[n >= P["spk_min_clips"]].index)
    df["spk_use"] = df.source.isin(["mswc", "librispeech"]) & df.speaker.isin(good)
    df.to_csv(os.path.join(CACHE, "clips.csv"), index=False)
    counts["clips_by_label_split"] = {f"{a}|{b}": int(v) for (a, b), v in df.groupby(["label", "split"]).size().items()}
    counts["clips_by_group_split"] = {f"{a}|{b}": int(v) for (a, b), v in df.groupby(["group", "split"]).size().items()}
    counts["speakers_by_group_split"] = {f"{a}|{b}": int(v) for (a, b), v in df.groupby(["group", "split"]).speaker.nunique().items()}
    sp = df[df.spk_use]
    counts["speaker_head"] = {s: dict(speakers=int(g.speaker.nunique()), clips=len(g)) for s, g in sp.groupby("split")}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="clips,noise,fa")
    parts = ap.parse_args().parts.split(",")
    path = os.path.join(CACHE, "counts.json")
    counts = json.load(open(path)) if os.path.exists(path) else {}
    if "clips" in parts:
        clips(counts)
    if "noise" in parts:
        counts["noise_hours"] = noise()
    if "fa" in parts:
        counts["fa"] = fa_speech()
    json.dump(counts, open(path, "w"), indent=1)
    log(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
