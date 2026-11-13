"""05_download_r4.py - the public data of Phase C (r4) -> data/raw (PLAN_R4.md).

    python scripts/05_download_r4.py                      # all three
    python scripts/05_download_r4.py --only vocalsound,ami

  vocalsound  VocalSound 16 kHz (1.7 GB zip, CC BY-SA 4.0): cough, throat clearing, sneeze, sniff,
              sigh, laughter of 3 365 speakers -> data/raw/vocalsound/ (audio_16k/, datafiles/)
  ami         AMI meetings, headset mix (CC BY 4.0): the meetings named in r4_sources.ami
              (about 1.3 GB) + the word transcripts (ami_public_manual_1.6.2.zip, 22 MB)
              -> data/raw/ami/<meeting>.wav, data/raw/ami/words/
  aishell     AISHELL-1 (Apache-2.0, 15 GB, streamed - never stored whole): only the speakers /
              utterances of r4_sources.aishell are kept -> data/raw/aishell/<split>/<speaker>/*.wav

ESC-50 (the vocal classes) comes from the Step-1 download (01_download.py --only esc50).
Every file's size and SHA-256 / MD5 go to data/raw/MANIFEST.json; a finished part has a DONE
file and is skipped. A part that fails prints the URL and stops (download it by hand into the
folder named above and run again).
"""
import argparse
import io
import os
import sys
import tarfile
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
dl = __import__("01_download")                     # Stream, download, manifest_add, done, mark, log
from kws.cfg import load, stable_hash  # noqa: E402

C = load()
RAW = C["paths"]["raw"]
R = C["r4_sources"]
log = dl.log


# ------------------------------------------------------------------ VocalSound
def vocalsound_extract(zpath, d):
    """audio_16k/*.wav, datafiles/*.json, meta/*.json from the release zip"""
    n = 0
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            parts = name.replace("\\", "/").split("/")
            if len(parts) < 2 or name.endswith("/"):
                continue
            for keep in ("audio_16k", "datafiles", "meta"):
                if keep in parts:
                    rel = "/".join(parts[parts.index(keep):])
                    out = os.path.join(d, rel)
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    with z.open(name) as src, open(out, "wb") as dst:
                        dst.write(src.read())
                    n += 1
                    break
    return n


def vocalsound():
    m = R["vocalsound"]
    d = os.path.join(RAW, "vocalsound")
    os.makedirs(d, exist_ok=True)
    if dl.done("vocalsound"):
        return log("vocalsound: done")
    zpath = os.path.join(d, "vs_release_16k.zip")
    if not os.path.exists(zpath):
        log("vocalsound:", m["url"])
        h, n = dl.download(m["url"], zpath, "sha256")
        dl.manifest_add("vocalsound/vs_release_16k.zip", url=m["url"], bytes=n, sha256=h, licence=m["licence"])
    n = vocalsound_extract(zpath, d)
    wavs = len(os.listdir(os.path.join(d, "audio_16k")))
    log("  vocalsound: %d files, %d recordings" % (n, wavs))
    if wavs < 20000:
        raise RuntimeError("vocalsound: only %d recordings in the zip (expected 21 024)" % wavs)
    os.remove(zpath)                                  # 1.7 GB; the extracted audio is kept
    dl.mark("vocalsound", "%d recordings" % wavs)


# ------------------------------------------------------------------ AMI
def ami_words_extract(zpath, d):
    """words/*.words.xml of the annotation zip"""
    n = 0
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if name.endswith(".words.xml") and "/words/" in "/" + name:
                out = os.path.join(d, "words", os.path.basename(name))
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with z.open(name) as src, open(out, "wb") as dst:
                    dst.write(src.read())
                n += 1
    return n


def ami():
    m = R["ami"]
    d = os.path.join(RAW, "ami")
    os.makedirs(d, exist_ok=True)
    if dl.done("ami"):
        return log("ami: done")
    zpath = os.path.join(d, "ami_public_manual_1.6.2.zip")
    if not os.path.isdir(os.path.join(d, "words")):
        log("ami annotations:", m["annotations_url"])
        h, n = dl.download(m["annotations_url"], zpath, "sha256")
        dl.manifest_add("ami/ami_public_manual_1.6.2.zip", url=m["annotations_url"], bytes=n, sha256=h, licence=m["licence"])
        log("  %d word files" % ami_words_extract(zpath, d))
        os.remove(zpath)
    for split in ("train", "val", "test"):
        for mt in m[split]:
            out = os.path.join(d, mt + ".wav")
            if os.path.exists(out):
                continue
            url = m["audio_url"].format(m=mt)
            log("ami %s: %s" % (split, mt))
            try:
                h, n = dl.download(url, out, "sha256")
            except Exception as e:  # noqa: BLE001
                # a few meetings may be missing on the mirror: noted, the others are enough
                log("  missing (%s) - skipped" % e)
                continue
            dl.manifest_add("ami/" + mt + ".wav", url=url, bytes=n, sha256=h, licence=m["licence"])
    dl.mark("ami")


# ------------------------------------------------------------------ AISHELL-1
def aishell_from_stream(fileobj, d, A):
    """streams data_aishell.tgz: data_aishell/wav/<SPK>.tar.gz (each: <split>/<SPK>/<utt>.wav)."""
    want = A["speakers"]
    per = A["utts_per_speaker"]
    kept = {s: 0 for s in want}
    # every k-th speaker of a split (in archive order) until enough: spread over the split.
    # AISHELL-1 has 340 train, 40 dev, 20 test speakers.
    total = {"train": 340, "dev": 40, "test": 20}
    seen = {s: 0 for s in want}
    n_files = 0
    with tarfile.open(fileobj=fileobj, mode="r|gz") as t:
        for ti in t:
            if not ti.isfile() or not ti.name.endswith(".tar.gz") or "/wav/" not in "/" + ti.name:
                continue
            spk = os.path.basename(ti.name)[:-len(".tar.gz")]
            inner = io.BytesIO(t.extractfile(ti).read())
            with tarfile.open(fileobj=inner, mode="r:gz") as it:
                members = [x for x in it.getmembers() if x.isfile() and x.name.endswith(".wav")]
                if not members:
                    continue
                split = members[0].name.split("/")[0]
                if split not in want:
                    continue
                k = max(1, total.get(split, want[split]) // max(1, want[split]))
                idx = seen[split]
                seen[split] += 1
                if idx % k or kept[split] >= want[split]:
                    continue
                members.sort(key=lambda x: stable_hash("aishell-utt:" + x.name))
                for x in members[:per[split]]:
                    out = os.path.join(d, split, spk, os.path.basename(x.name))
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    open(out, "wb").write(it.extractfile(x).read())
                    n_files += 1
                kept[split] += 1
                log("  aishell %s %s: %d utterances (speakers kept: %s)" % (split, spk, min(len(members), per[split]), kept))
    return kept, n_files


def aishell():
    A = R["aishell"]
    d = os.path.join(RAW, "aishell")
    os.makedirs(d, exist_ok=True)
    if dl.done("aishell"):
        return log("aishell: done")
    last = None
    for url in A["urls"]:
        try:
            log("aishell:", url, "(15 GB, streamed)")
            s = dl.Stream(url, "md5")
            kept, n = aishell_from_stream(io.BufferedReader(s, 1 << 22), d, A)
            while s.read(1 << 22):
                pass
            dl.manifest_add("aishell/data_aishell.tgz", url=url, bytes=s.pos, md5=s.h.hexdigest(), licence=A["licence"],
                            speakers_kept=kept, files_kept=n, note="streamed; only the kept utterances are stored")
            dl.mark("aishell", "%d files" % n)
            return
        except Exception as e:  # noqa: BLE001
            last = e
            log("  failed (%s), next mirror" % e)
    raise RuntimeError("aishell: no mirror worked (%s)" % last)


PARTS = dict(vocalsound=vocalsound, ami=ami, aishell=aishell)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(PARTS))
    a = ap.parse_args()
    for p in a.only.split(","):
        PARTS[p]()
    log("r4 downloads done")
