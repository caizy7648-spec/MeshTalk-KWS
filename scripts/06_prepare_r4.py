"""06_prepare_r4.py - Phase C data -> data/cache (PLAN_R4.md).

    python scripts/06_prepare_r4.py                 # own, vocal, chat, fa4
    python scripts/06_prepare_r4.py --parts own     # only the owner's clips (after a new import)

Writes (int16, 16 kHz, clips one after the other + CSV, same columns as clips.csv):
  own.npy / own.csv       the owner's recordings (data/own/manifest.csv from 04_import_own.py),
                          group own, labels help / help me / sos, split from the manifest
  vocal.npy / vocal.csv   vocal sounds as _other clips: VocalSound (official speaker split) and
                          ESC-50 coughing / sneezing / breathing / laughing / snoring
                          (fold 5 test, 4 val, 1-3 train); one clip per event, a coughing fit
                          stays one clip; word = the sound type
  chat.npy / chat.csv     conversation as _other: AMI chunks (group chat_en, train / val meetings),
                          AISHELL-1 utterances (group chat_zh, train / dev speakers)
  fa4_{val,test}.npy/.csv false-alarm streams for 21_eval_r4.py: AMI meetings, AISHELL speakers'
                          utterances in a row, vocal sounds in a row (val / test speakers only)
  fa4_{val,test}_events.csv  where the vocal sounds are, and where somebody really says "help" in AMI
  counts_r4.json          numbers for REPORT.md
"""
import argparse
import glob
import io
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kws.audio import resample, rms_db, speech_span, to_int16  # noqa: E402
from kws.cfg import ROOT, load, rng_for, stable_hash  # noqa: E402

prep = __import__("02_prepare")                    # Store, COLS
C = load()
RAW, CACHE = C["paths"]["raw"], C["paths"]["cache"]
P = C["r4_prepare"]
R = C["r4_sources"]
RATE = 16000
SPLIT_NAME = {"tr": "train", "val": "val", "te": "test", "train": "train", "dev": "val", "test": "test"}


def log(*a):
    print(*a, flush=True)


def read_wav(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return to_int16(resample(x, sr))


def frames_db(x, hop=320):
    n = len(x) // hop
    if n == 0:
        return np.zeros(0)
    seg = x[:n * hop].astype(np.float64).reshape(n, hop) / 32768.0
    return 10 * np.log10(np.mean(seg * seg, axis=1) + 1e-12)


def events(x):
    """[(start, end)] samples of the vocal sounds in a recording (merged, padded, cut to event_max_s)"""
    e = frames_db(x)
    if len(e) == 0:
        return []
    floor = np.percentile(e, 20)
    on = (e > e.max() + P["event_rel_db"]) & (e > floor + P["event_min_above_floor_db"])
    segs, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            segs.append([i, j])
            i = j
        else:
            i += 1
    gap = int(P["event_merge_gap_s"] * 50)
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] <= gap:
            merged[-1][1] = s[1]
        else:
            merged.append(list(s))
    pad, out = int(P["event_pad_s"] * RATE), []
    for s, t in merged:
        if (t - s) * 0.02 < P["event_min_s"]:
            continue
        a, b = max(0, s * 320 - pad), min(len(x), t * 320 + pad)
        b = min(b, a + int(P["event_max_s"] * RATE))
        out.append((a, b))
    return out


# ------------------------------------------------------------------ the owner
def own():
    man = os.path.join(ROOT, C["own"]["dir"], "manifest.csv")
    if not os.path.exists(man):
        log("own: no %s - run scripts/04_import_own.py first" % man)
        return {}
    root = os.path.dirname(man)
    st = prep.Store("own")
    df = pd.read_csv(man, keep_default_na=False)
    for r in df.itertuples():
        x = read_wav(os.path.join(root, r.path))
        if rms_db(x) < C["prepare"]["min_rms_db"]:
            continue
        s, e = speech_span(x, C["prepare"]["span_rel_db"])
        st.add(x, source="own", group="own", label=r.label, word=r.style, speaker="own-" + str(r.speaker), gender="",
               split=r.split, spk_use=False, text=r.path, kw_start=s, kw_end=e)
    out = st.save(prep.COLS)
    c = {f"{a}|{b}": int(v) for (a, b), v in out.groupby(["label", "split"]).size().items()}
    log("own", c)
    return c


# ------------------------------------------------------------------ vocal sounds
def vocalsound_split():
    d = os.path.join(RAW, "vocalsound", "datafiles")
    out = {}
    for name in ("tr", "val", "te"):
        f = os.path.join(d, name + ".json")
        if os.path.exists(f):
            for r in json.load(open(f))["data"]:
                out[os.path.basename(r["wav"])] = SPLIT_NAME[name]
    return out


def vocal_items():
    """(audio int16, type, speaker, split, source) of every vocal recording"""
    sp = vocalsound_split()
    for f in sorted(glob.glob(os.path.join(RAW, "vocalsound", "audio_16k", "*.wav"))):
        b = os.path.basename(f)
        if b not in sp:
            continue
        typ = os.path.splitext(b)[0].split("_")[-1]
        yield read_wav(f), typ, "vs-" + b.split("_")[0], sp[b], "vocalsound", b
    E = R["esc50_vocal"]
    tf = C["sources"]["esc50"]["test_fold"]
    for f in sorted(glob.glob(os.path.join(RAW, "esc50", "*.parquet"))):
        for r in pd.read_parquet(f).itertuples():
            if r.category not in E["categories"]:
                continue
            x, sr = sf.read(io.BytesIO(r.audio["bytes"]), dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            split = "test" if r.fold == tf else "val" if r.fold == E["val_fold"] else "train"
            yield to_int16(resample(x, sr)), r.category, "esc-" + r.filename.split("-")[1], split, "esc50", r.filename


def vocal():
    st = prep.Store("vocal")
    n = {}
    for x, typ, spk, split, src, name in vocal_items():
        for a, b in events(x):
            seg = x[a:b]
            s, e = speech_span(seg, C["prepare"]["span_rel_db"])
            st.add(seg, source=src, group="vocal", label="_other", word=typ, speaker=spk, gender="", split=split,
                   spk_use=False, text=name, kw_start=s, kw_end=e)
            n[f"{typ}|{split}"] = n.get(f"{typ}|{split}", 0) + 1
    st.save(prep.COLS)
    log("vocal", n)
    return n


# ------------------------------------------------------------------ AMI / AISHELL
def ami_help_times(meeting):
    """[(start, end)] seconds where anybody says help / helps / helping ... in the meeting"""
    out = []
    for f in glob.glob(os.path.join(RAW, "ami", "words", meeting + ".*.words.xml")):
        for m in re.finditer(r'<w [^>]*starttime="([0-9.]+)"[^>]*endtime="([0-9.]+)"[^>]*>([^<]*)</w>', open(f, encoding="utf-8", errors="replace").read()):
            if m.group(3).strip().lower().startswith("help"):
                out.append((float(m.group(1)), float(m.group(2))))
    return out


def ami_meetings(split):
    return [m for m in R["ami"][split] if os.path.exists(os.path.join(RAW, "ami", m + ".wav"))]


def aishell_speakers(split):
    d = os.path.join(RAW, "aishell", split)
    return sorted(os.listdir(d)) if os.path.isdir(d) else []


def chat():
    st = prep.Store("chat")
    n = {}
    cl, every, guard = int(P["ami_chunk_s"] * RATE), P["ami_chunk_every_s"], P["keyword_guard_s"]
    for split in ("train", "val"):
        for mt in ami_meetings(split):
            x = read_wav(os.path.join(RAW, "ami", mt + ".wav"))
            helps = ami_help_times(mt)
            k = 0
            while (k + 1) * every * RATE <= len(x):
                a = int(k * every * RATE) + stable_hash("ami:%s:%d" % (mt, k)) % max(1, int(every * RATE) - cl)
                k += 1
                b = a + cl
                if any(s - guard < b / RATE and e + guard > a / RATE for s, e in helps):
                    continue
                seg = x[a:b]
                if rms_db(seg) < C["prepare"]["min_rms_db"] + 10:
                    continue
                st.add(seg, source="ami", group="chat_en", label="_other", word="", speaker="ami-" + mt, gender="",
                       split=split, spk_use=False, text="%s@%.1f" % (mt, a / RATE), kw_start=0, kw_end=cl)
                n["chat_en|" + split] = n.get("chat_en|" + split, 0) + 1
    for asplit, split in (("train", "train"), ("dev", "val")):
        for spk in aishell_speakers(asplit):
            for f in sorted(glob.glob(os.path.join(RAW, "aishell", asplit, spk, "*.wav"))):
                x = read_wav(f)
                s, e = speech_span(x, C["prepare"]["span_rel_db"])
                st.add(x, source="aishell", group="chat_zh", label="_other", word="", speaker="ais-" + spk, gender="",
                       split=split, spk_use=False, text=os.path.basename(f), kw_start=s, kw_end=e)
                n["chat_zh|" + split] = n.get("chat_zh|" + split, 0) + 1
    st.save(prep.COLS)
    log("chat", n)
    return n


# ------------------------------------------------------------------ false-alarm streams
def fa4():
    out = {}
    for split in ("val", "test"):
        st = prep.Store("fa4_" + split)
        ev = []
        rng = rng_for(C, "fa4", split)
        cap = P["fa_hours"][split]
        # AMI meetings, whole
        h = 0.0
        for mt in ami_meetings(split):
            if h >= cap["ami"]:
                break
            x = read_wav(os.path.join(RAW, "ami", mt + ".wav"))
            name = "ami-" + mt
            for s, e in ami_help_times(mt):
                ev.append(dict(stream=name, kind="keyword", type="help", at=int(s * RATE), end=int(e * RATE)))
            st.add(x, stream=name, source="ami", speaker=name)
            h += len(x) / RATE / 3600
        # AISHELL: each speaker's utterances one after the other
        h = 0.0
        for spk in aishell_speakers("dev" if split == "val" else "test"):
            if h >= cap["aishell"]:
                break
            parts = []
            for f in sorted(glob.glob(os.path.join(RAW, "aishell", "dev" if split == "val" else "test", spk, "*.wav"))):
                parts.append(read_wav(f))
                parts.append(np.zeros(int(rng.uniform(*P["aishell_gap_s"]) * RATE), np.int16))
            if parts:
                x = np.concatenate(parts)
                st.add(x, stream="ais-" + spk, source="aishell", speaker="ais-" + spk)
                h += len(x) / RATE / 3600
        # vocal sounds: per speaker (VocalSound) / per 10 clips (ESC-50), with pauses
        groups = {}
        for x, typ, spk, sp, src, name in vocal_items():
            if sp != split:
                continue
            key = spk if src == "vocalsound" else "esc-%s-%d" % (typ, stable_hash(name) % 4)
            for a, b in events(x):
                groups.setdefault(key, []).append((typ, src, x[a:b]))
        for key in sorted(groups):
            parts, pos, name = [], 0, "voc-" + key
            for typ, src, seg in groups[key]:
                g = np.zeros(int(rng.uniform(*P["vocal_gap_s"]) * RATE), np.int16)
                parts += [g, seg]
                ev.append(dict(stream=name, kind="vocal", type=typ, at=pos + len(g), end=pos + len(g) + len(seg)))
                pos += len(g) + len(seg)
            parts.append(np.zeros(int(P["vocal_gap_s"][1] * RATE), np.int16))
            st.add(np.concatenate(parts), stream=name, source="vocal", speaker=key)
        df = st.save()
        pd.DataFrame(ev, columns=["stream", "kind", "type", "at", "end"]).to_csv(
            os.path.join(CACHE, "fa4_%s_events.csv" % split), index=False)
        out[split] = {s: round(g.length.sum() / RATE / 3600, 2) for s, g in df.groupby("source")} if len(df) else {}
        out[split]["vocal_events"] = sum(1 for e in ev if e["kind"] == "vocal")
        log("fa4", split, out[split])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="own,vocal,chat,fa4")
    parts = ap.parse_args().parts.split(",")
    path = os.path.join(CACHE, "counts_r4.json")
    counts = json.load(open(path)) if os.path.exists(path) else {}
    for p in parts:
        counts[p] = globals()[p]()
    json.dump(counts, open(path, "w"), indent=1)
    log(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
