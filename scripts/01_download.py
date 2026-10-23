"""01_download.py - public datasets -> data/raw (Step 1).

    python scripts/01_download.py                 # everything (about 75 GB streamed, about 12 GB kept)
    python scripts/01_download.py --only mswc,kokoro

Sources, URLs, licences and checksums are in configs/train.yaml (sources:).
Big archives are streamed: MSWC English (35 GB) and the LibriSpeech training
shards (30 GB) are never stored whole - only the clips we need are kept.
Every download is checked (MD5 from Google Cloud Storage, SHA-256 from the
HuggingFace LFS pointer) and written to data/raw/MANIFEST.json.
A part that is complete has a data/raw/<part>/DONE file and is skipped.
"""
import argparse
import base64
import csv
import hashlib
import io
import json
import os
import sys
import tarfile
import time

import numpy as np
import requests
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kws.cfg import load, stable_hash  # noqa: E402

C = load()
RAW = C["paths"]["raw"]
S = C["sources"]
MANIFEST = os.path.join(RAW, "MANIFEST.json")


# ------------------------------------------------------------------ helpers
def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def manifest_add(key, **kw):
    m = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}
    m[key] = kw
    json.dump(m, open(MANIFEST, "w"), indent=1, sort_keys=True)


class Stream(io.RawIOBase):
    """HTTP GET as a file object; reconnects with a Range header when the
    connection breaks, and hashes everything it returns."""

    def __init__(self, url, algo="md5"):
        self.url, self.pos, self.h, self.r, self.it = url, 0, hashlib.new(algo), None, None
        self.size = None
        self._open()

    def _open(self):
        for attempt in range(8):
            try:
                hdr = {"Range": "bytes=%d-" % self.pos} if self.pos else {}
                self.r = requests.get(self.url, headers=hdr, stream=True, timeout=60, allow_redirects=True)
                self.r.raise_for_status()
                if self.size is None:
                    self.size = int(self.r.headers.get("Content-Length", 0)) or None
                self.it = self.r.iter_content(1 << 20)
                self.buf = b""
                return
            except Exception as e:  # noqa: BLE001
                log("  reconnect (%s), attempt %d" % (e, attempt + 1))
                time.sleep(2 ** attempt)
        raise RuntimeError("cannot download " + self.url)

    def readable(self):
        return True

    def readinto(self, b):
        while not self.buf:
            try:
                self.buf = next(self.it)
            except StopIteration:
                return 0
            except Exception as e:  # noqa: BLE001
                log("  connection lost at %.2f GB (%s), resuming" % (self.pos / 1e9, e))
                self._open()
        n = min(len(b), len(self.buf))
        b[:n] = self.buf[:n]
        self.h.update(self.buf[:n])
        self.buf = self.buf[n:]
        self.pos += n
        return n


def download(url, path, algo="md5"):
    """whole file -> path, returns hex digest"""
    s = Stream(url, algo)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        while True:
            b = s.read(1 << 22)
            if not b:
                break
            f.write(b)
    os.replace(tmp, path)
    return s.h.hexdigest(), s.pos


def b64_to_hex(b64):
    return base64.b64decode(b64).hex()


def hf_files(repo, prefix=""):
    r = requests.get("https://huggingface.co/api/datasets/%s/tree/main/%s" % (repo, prefix), timeout=60)
    r.raise_for_status()
    return {x["path"]: x for x in r.json() if x["type"] == "file"}


def hf_url(repo, path):
    return "https://huggingface.co/datasets/%s/resolve/main/%s" % (repo, path)


def done(part):
    return os.path.exists(os.path.join(RAW, part, "DONE"))


def mark(part, info=""):
    open(os.path.join(RAW, part, "DONE"), "w").write(info)


def check(name, got, want):
    if want and got != want:
        raise RuntimeError("%s: checksum %s, expected %s" % (name, got, want))
    log("  %s checksum ok (%s)" % (name, got))


def frac_keep(key, frac):
    return (stable_hash(key) % 100000) < frac * 100000


# ------------------------------------------------------------------ MSWC
def mswc():
    d = os.path.join(RAW, "mswc")
    os.makedirs(d, exist_ok=True)
    if done("mswc"):
        return log("mswc: done")
    m = S["mswc"]
    csv_path = os.path.join(d, "en_splits.csv")
    if not os.path.exists(csv_path):
        log("mswc: splits", m["splits_url"])
        tgz = os.path.join(d, "en_splits.tar.gz")
        h, n = download(m["splits_url"], tgz)
        check("mswc splits", h, b64_to_hex(m["splits_md5_b64"]))
        manifest_add("mswc_splits", url=m["splits_url"], bytes=n, md5=h, licence=m["licence"])
        with tarfile.open(tgz) as t:
            t.extract("en_splits.csv", d)
        os.remove(tgz)

    # --- which clips we keep (deterministic)
    import pandas as pd
    df = pd.read_csv(csv_path, usecols=["LINK", "WORD", "VALID", "SPEAKER", "GENDER"])
    df = df[df.VALID == True]  # noqa: E712
    df["h"] = [stable_hash(x) for x in df.LINK]
    sel = []
    kw = df[df.WORD.isin(m["keyword_words"])].assign(role="keyword")
    sel.append(kw)
    for w in m["lookalike_words"]:
        x = df[df.WORD == w].sort_values("h").head(m["lookalike_max_per_word"])
        sel.append(x.assign(role="lookalike"))
    rest = df[~df.WORD.isin(m["keyword_words"] + m["lookalike_words"])]
    sel.append(rest.sort_values("h").head(m["random_other_clips"]).assign(role="random"))
    cnt = df.groupby("SPEAKER").size()
    ok = cnt[cnt >= m["spk_min_clips"]]
    kw_spk = set(kw.SPEAKER)
    order = sorted(ok.index, key=lambda s: (s not in kw_spk, stable_hash("spk:" + s)))   # keyword speakers first
    spk = set(order[:m["spk_speakers"]])
    x = df[df.SPEAKER.isin(spk)].sort_values("h").groupby("SPEAKER").head(m["spk_clips_per_speaker"])
    sel.append(x.assign(role="speaker"))
    sel = pd.concat(sel).drop_duplicates("LINK").drop(columns=["h", "VALID"])
    sel.to_csv(os.path.join(d, "selected.csv"), index=False)
    want = set(sel.LINK)
    log("mswc: %d clips selected (%s)" % (len(want), dict(sel.role.value_counts())))

    # --- stream the 35 GB archive, keep the selected clips
    log("mswc: streaming", m["audio_url"])
    s = Stream(m["audio_url"], "md5")
    got, t0 = 0, time.time()
    with tarfile.open(fileobj=io.BufferedReader(s, 1 << 22), mode="r|gz") as t:
        for ti in t:
            if not ti.isfile():
                continue
            link = ti.name.split("/clips/", 1)[-1]
            if link in want:
                out = os.path.join(d, "clips", link)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with open(out, "wb") as f:
                    f.write(t.extractfile(ti).read())
                got += 1
                if got % 5000 == 0:
                    log("  %d / %d clips, %.1f GB read, %.0f MB/s" % (got, len(want), s.pos / 1e9, s.pos / 1e6 / (time.time() - t0)))
    # read to the end so the MD5 covers the whole file
    while s.read(1 << 22):
        pass
    check("mswc audio", s.h.hexdigest(), b64_to_hex(m["audio_md5_b64"]))
    manifest_add("mswc_audio", url=m["audio_url"], bytes=s.pos, md5=s.h.hexdigest(), licence=m["licence"],
                 clips_kept=got, clips_selected=len(want))
    mark("mswc", "%d clips" % got)
    os.remove(csv_path)      # 1.2 GB; selected.csv has everything we need
    log("mswc: %d clips kept" % got)


# ------------------------------------------------------------------ LibriSpeech
def ls_words(row):
    return [(w["word"].lower(), float(w["start"]), float(w["end"])) for w in row["words"] if w["word"]]


def ls_settings(sub):
    """librispeech: settings with the per-subset overrides of train.yaml"""
    L = dict(S["librispeech"])
    L.update(L.get("subset_overrides", {}).get(sub, {}))
    return L


def ls_cut_train(table, L):
    """the clips we keep from one training shard: (meta rows, int16 audio)"""
    look = set(S["mswc"]["lookalike_words"])
    ctx = L["context_s"]
    rows, audio, pos = [], [], 0

    def add(kind, word, spk, utt, x, a, b, ka, kb):
        nonlocal pos
        seg = x[a:b]
        rows.append(dict(kind=kind, word=word, speaker=spk, utt=utt, offset=pos, length=len(seg), kw_start=ka - a, kw_end=kb - a))
        audio.append(seg)
        pos += len(seg)

    for row in table.to_pylist():
        utt = row["id"]
        spk = "ls-" + utt.split("-")[0]
        x, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="int16")
        assert sr == 16000
        W = ls_words(row)
        n = len(x)

        def span(s0, e0):
            return max(0, int((s0 - ctx) * sr)), min(n, int((e0 + ctx) * sr)), int(s0 * sr), int(e0 * sr)

        for i, (w, s0, e0) in enumerate(W):
            key = "%s:%d" % (utt, i)
            nxt = W[i + 1] if i + 1 < len(W) else None
            if w in L["keyword_words"]:
                if nxt and nxt[0] == "me" and nxt[1] - e0 < 0.3:
                    add("help_me", "help me", spk, utt, x, *span(s0, nxt[2]))
                    add("help_of_help_me", w, spk, utt, x, *span(s0, e0))       # the "help" of a real "help me"
                else:
                    add("help", w, spk, utt, x, *span(s0, e0))
            elif w == "me" and frac_keep(key, L["me_keep_frac"]):
                add("me", w, spk, utt, x, *span(s0, e0))
            elif w in look and frac_keep(key, L["lookalike_keep_frac"]):
                add("lookalike", w, spk, utt, x, *span(s0, e0))
            elif e0 - s0 >= L["spk_word_min_s"] and frac_keep(key, L["spk_word_keep_frac"]):
                add("spk_word", w, spk, utt, x, *span(s0, e0))
        if frac_keep("chunk:" + utt, L["other_chunks_per_utt"]) and n > L["chunk_s"] * sr:
            cl = int(L["chunk_s"] * sr)
            a = stable_hash("chunkpos:" + utt) % (n - cl)
            # no keyword inside a running-speech chunk
            if not any(w in L["keyword_words"] and s0 * sr < a + cl and e0 * sr > a for w, s0, e0 in W):
                add("chunk", "", spk, utt, x, a, a + cl, a, a + cl)
    return rows, (np.concatenate(audio) if audio else np.zeros(0, np.int16))


def librispeech():
    import pyarrow.parquet as pq
    L = S["librispeech"]
    d = os.path.join(RAW, "librispeech")
    os.makedirs(os.path.join(d, "cut"), exist_ok=True)
    os.makedirs(os.path.join(d, "eval"), exist_ok=True)
    if done("librispeech"):
        return log("librispeech: done")
    files = hf_files(L["hf_repo"], "data")
    for path, info in sorted(files.items()):
        sub = os.path.basename(path).split("-")[0]
        sha = info.get("lfs", {}).get("oid")
        if sub in L["eval_subsets"]:
            out = os.path.join(d, "eval", os.path.basename(path))
            if not os.path.exists(out):
                log("librispeech eval:", path)
                h, n = download(hf_url(L["hf_repo"], path), out, "sha256")
                check(path, h, sha)
                manifest_add("librispeech/" + path, url=hf_url(L["hf_repo"], path), bytes=n, sha256=h, licence=L["licence"])
        elif sub in L["train_subsets"]:
            base = os.path.join(d, "cut", os.path.basename(path).replace(".parquet", ""))
            if os.path.exists(base + ".csv"):
                continue
            tmp = os.path.join(d, "tmp.parquet")
            log("librispeech train:", path, "%.2f GB" % (info["size"] / 1e9))
            h, n = download(hf_url(L["hf_repo"], path), tmp, "sha256")
            check(path, h, sha)
            pf = pq.ParquetFile(tmp)
            rows, audio = [], []
            off = 0
            for g in range(pf.num_row_groups):
                r, a = ls_cut_train(pf.read_row_group(g, columns=["id", "audio", "words"]), ls_settings(sub))
                for x in r:
                    x["offset"] += off
                off += len(a)
                rows += r
                audio.append(a)
            np.save(base + ".npy", np.concatenate(audio))
            with open(base + ".csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            os.remove(tmp)
            kinds = {}
            for x in rows:
                kinds[x["kind"]] = kinds.get(x["kind"], 0) + 1
            manifest_add("librispeech/" + path, url=hf_url(L["hf_repo"], path), bytes=n, sha256=h, licence=L["licence"],
                         kept=kinds, note="streamed; only the cut clips are kept")
            log("  kept", kinds)
    mark("librispeech")


# ------------------------------------------------------------------ Speech Commands
def speech_commands():
    m = S["speech_commands"]
    d = os.path.join(RAW, "speech_commands")
    os.makedirs(d, exist_ok=True)
    if done("speech_commands"):
        return log("speech_commands: done")
    log("speech_commands:", m["url"])
    s = Stream(m["url"], "md5")
    n = 0
    with tarfile.open(fileobj=io.BufferedReader(s, 1 << 22), mode="r|gz") as t:
        for ti in t:
            if not ti.isfile() or not ti.name.endswith(".wav"):
                continue
            name = ti.name.lstrip("./")
            # all background noise; words: a hashed share (about 2x what prepare uses)
            if name.startswith("_background_noise_") or frac_keep(name, 2.0 * m["other_clips"] / 105829):
                out = os.path.join(d, name)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                open(out, "wb").write(t.extractfile(ti).read())
                n += 1
    while s.read(1 << 22):
        pass
    check("speech_commands", s.h.hexdigest(), b64_to_hex(m["md5_b64"]))
    manifest_add("speech_commands", url=m["url"], bytes=s.pos, md5=s.h.hexdigest(), licence=m["licence"], files_kept=n)
    mark("speech_commands", "%d files" % n)


# ------------------------------------------------------------------ ESC-50
def esc50():
    m = S["esc50"]
    d = os.path.join(RAW, "esc50")
    os.makedirs(d, exist_ok=True)
    if done("esc50"):
        return log("esc50: done")
    for path, info in sorted(hf_files(m["hf_repo"], "data").items()):
        log("esc50:", path)
        out = os.path.join(d, os.path.basename(path))
        h, n = download(hf_url(m["hf_repo"], path), out, "sha256")
        check(path, h, info.get("lfs", {}).get("oid"))
        manifest_add("esc50/" + path, url=hf_url(m["hf_repo"], path), bytes=n, sha256=h, licence=m["licence"])
    mark("esc50")


# ------------------------------------------------------------------ MUSAN
def musan():
    m = S["musan"]
    d = os.path.join(RAW, "musan")
    os.makedirs(d, exist_ok=True)
    if done("musan"):
        return log("musan: done")
    log("musan:", m["url"])
    s = Stream(m["url"], "md5")
    n = 0
    with tarfile.open(fileobj=io.BufferedReader(s, 1 << 22), mode="r|gz") as t:
        for ti in t:
            parts = ti.name.split("/")
            if ti.isfile() and len(parts) > 2 and parts[1] in m["keep"] and ti.name.endswith(".wav") \
                    and (parts[1] != "music" or frac_keep(ti.name, m["music_keep_frac"])):
                out = os.path.join(d, *parts[1:])
                os.makedirs(os.path.dirname(out), exist_ok=True)
                open(out, "wb").write(t.extractfile(ti).read())
                n += 1
    while s.read(1 << 22):
        pass
    # openslr publishes no checksum file; we record ours
    manifest_add("musan", url=m["url"], bytes=s.pos, md5=s.h.hexdigest(), licence=m["licence"], files_kept=n)
    mark("musan", "%d files" % n)


# ------------------------------------------------------------------ Kokoro
def kokoro():
    m = S["kokoro"]
    d = os.path.join(RAW, "kokoro")
    os.makedirs(d, exist_ok=True)
    if done("kokoro"):
        return log("kokoro: done")
    for k in ("model_url", "voices_url"):
        out = os.path.join(d, os.path.basename(m[k]))
        log("kokoro:", m[k])
        h, n = download(m[k], out, "sha256")
        manifest_add("kokoro/" + os.path.basename(out), url=m[k], bytes=n, sha256=h, licence=m["licence"])
    mark("kokoro")


PARTS = dict(kokoro=kokoro, esc50=esc50, speech_commands=speech_commands, mswc=mswc, librispeech=librispeech, musan=musan)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(PARTS))
    a = ap.parse_args()
    for p in a.only.split(","):
        PARTS[p]()
    log("all done")
