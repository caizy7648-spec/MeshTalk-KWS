"""43_spk_extra.py - more voices for the speaker model (phase B2): VCTK and People's Speech.

    python scripts/43_spk_extra.py

Writes data/cache/spk_extra.npy (int16, 16 kHz) + spk_extra.csv
(source, speaker, gender, split, utt, offset, length, kw_start, kw_end) and
data/cache/libri_gender.csv (LibriSpeech SPEAKERS.TXT).
  * VCTK: utts_per_speaker utterances of every speaker are read straight out of
    the remote 10.7 GB zip with HTTP range requests (only those members are
    transferred), resampled 22.05 -> 16 kHz, silence trimmed, at most max_s.
    Speakers are labelled (AAM + distillation), split by the speaker hash.
  * People's Speech: a few training shards spread over the corpus, a seg_s
    segment from the middle of each utterance; no speaker labels (distillation
    only, split "train").
Gender of VCTK / People's Speech is not in these copies; 44_teacher.py predicts it.
Settings: configs/spk.yaml finetune.extra.
"""
import io
import os
import sys
import tarfile
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kws.audio import resample, speech_span, to_int16  # noqa: E402
from kws.cfg import speaker_split, stable_hash  # noqa: E402
from spk.data import load_cfg  # noqa: E402

C, P = load_cfg()
X = P["finetune"]["extra"]
CACHE, RAW = C["paths"]["cache"], C["paths"]["raw"]
prep = __import__("02_prepare")
dl = __import__("01_download")
RATE = 16000


class Remote(io.RawIOBase):
    """a seekable file over HTTP range requests (for zipfile)"""

    def __init__(self, url):
        r = requests.head(url, allow_redirects=True, timeout=60)
        self.url, self.size, self.pos = r.url, int(r.headers["Content-Length"]), 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, o, w=0):
        self.pos = o if w == 0 else (self.pos + o if w == 1 else self.size + o)
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        for attempt in range(6):
            try:
                d = requests.get(self.url, headers={"Range": "bytes=%d-%d" % (self.pos, self.pos + n - 1)}, timeout=60).content
                break
            except Exception:  # noqa: BLE001
                if attempt == 5:
                    raise
        b[:len(d)] = d
        self.pos += len(d)
        return len(d)


def trim(x, max_s):
    s, e = speech_span(x)
    p = int(0.1 * RATE)
    x = x[max(0, s - p):min(len(x), e + p)]
    return x[:int(max_s * RATE)]


def vctk(st):
    v = X["vctk"]
    z = zipfile.ZipFile(io.BufferedReader(Remote(v["url"]), 1 << 16))
    wavs = [n for n in z.namelist() if n.endswith(".wav")]
    by = {}
    for n in wavs:
        by.setdefault(n.split("/")[0], []).append(n)
    pick = []
    for spk, names in sorted(by.items()):
        pick += sorted(names, key=lambda n: stable_hash("vctk:" + n))[:v["utts_per_speaker"]]

    info = {i.filename: i for i in z.infolist()}
    url = z.fp.raw.url if hasattr(z.fp, "raw") else Remote(v["url"]).url

    def get(name):
        """one range request: local header + compressed data of this member"""
        i = info[name]
        n = 30 + len(i.filename.encode()) + 1024 + i.compress_size
        for attempt in range(6):
            try:
                b = requests.get(url, headers={"Range": "bytes=%d-%d" % (i.header_offset, i.header_offset + n - 1)},
                                 timeout=120).content
                break
            except Exception:  # noqa: BLE001
                if attempt == 5:
                    raise
        fl, el = int.from_bytes(b[26:28], "little"), int.from_bytes(b[28:30], "little")
        data = b[30 + fl + el:30 + fl + el + i.compress_size]
        raw = zlib.decompress(data, -15) if i.compress_type == zipfile.ZIP_DEFLATED else data
        x, sr = sf.read(io.BytesIO(raw), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        return name, trim(resample(x, sr), v["max_s"])

    with ThreadPoolExecutor(v["threads"]) as ex:
        for k, (name, x) in enumerate(ex.map(get, pick)):
            spk = "vctk-" + name.split("/")[0]
            st.add(to_int16(x), source="vctk", speaker=spk, gender="", split=speaker_split(C, spk), utt=name,
                   kw_start=0, kw_end=len(x))
            if k % 500 == 0:
                print("vctk", k, "/", len(pick), flush=True)
    print("vctk: %d utterances, %d speakers" % (len(pick), len(by)), flush=True)


def peoples_speech(st):
    v = X["peoples_speech"]
    files = dl.hf_files(v["repo"], "clean")
    train = sorted(p for p in files if "/train-" in p)
    seg = int(v["seg_s"] * RATE)
    for k in v["shards"]:
        path = train[k]
        tmp = os.path.join(CACHE, "ps_tmp.parquet")
        h, n = dl.download(dl.hf_url(v["repo"], path), tmp, "sha256")
        dl.check(path, h, files[path].get("lfs", {}).get("oid"))
        df = pd.read_parquet(tmp, columns=["id", "audio", "duration_ms"])
        df = df[df.duration_ms >= v["min_s"] * 1000]
        df = df.iloc[sorted(range(len(df)), key=lambda i: stable_hash("ps:" + df.id.iloc[i]))[:v["segs_per_shard"]]]
        for r in df.itertuples():
            x, sr = sf.read(io.BytesIO(r.audio["bytes"]), dtype="float32")
            x = resample(x if x.ndim == 1 else x.mean(axis=1), sr)
            a = max(0, len(x) // 2 - seg // 2)
            y = x[a:a + seg]
            if np.sqrt(np.mean(y ** 2) + 1e-12) < 1e-3:
                continue
            st.add(to_int16(y), source="peoples_speech", speaker="", gender="", split="train", utt=r.id,
                   kw_start=0, kw_end=len(y))
        os.remove(tmp)
        print("peoples_speech", path, len(st.rows), flush=True)


def libri_gender():
    tgz = os.path.join(RAW, "spk_extra", "raw-metadata.tar.gz")
    os.makedirs(os.path.dirname(tgz), exist_ok=True)
    if not os.path.exists(tgz):
        open(tgz, "wb").write(requests.get(X["librispeech_speakers_url"], timeout=120).content)
    with tarfile.open(tgz) as t:
        txt = t.extractfile("LibriSpeech/SPEAKERS.TXT").read().decode("utf-8", "replace")
    rows = []
    for line in txt.splitlines():
        if line.startswith(";") or "|" not in line:
            continue
        f = [x.strip() for x in line.split("|")]
        rows.append(dict(speaker="ls-" + f[0], gender={"F": "FEMALE", "M": "MALE"}.get(f[1], "")))
    pd.DataFrame(rows).to_csv(os.path.join(CACHE, "libri_gender.csv"), index=False)
    print("librispeech genders:", len(rows), flush=True)


def main():
    libri_gender()
    st = prep.Store("spk_extra")
    vctk(st)
    peoples_speech(st)
    df = st.save()
    print(df.groupby(["source", "split"]).agg(n=("length", "size"), hours=("length", lambda s: s.sum() / RATE / 3600)))


if __name__ == "__main__":
    main()
