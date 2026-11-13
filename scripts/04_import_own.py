"""04_import_own.py - import the owner's recordings (MeshTalk DataRec v1.0).

DataRec writes, on the microSD card:
    DATASET/INDEX.CSV                     file,category,speaker,style,word,ms,peak_dbfs,clipped,result,uptime_s,utc
    DATASET/<CAT>/<SPK>_<STYLE>_<NNNN>.WAV          CAT = HELP, HELPME, SOS
    (DATASET/WORDS/<SPK>_<WORD>_<STYLE>_<NNNN>.WAV, other folders: accepted if present)
16 kHz / 16 bit / mono, 2.5 s per take, recorded from the device microphone
through the same audio path as Voice SOS.

This script
  * takes the DATASET folder, a folder that contains it, or a .zip of either;
  * keeps only valid WAV files (16 kHz / 16 bit / mono, not empty) - a file
    deleted with "Delete last" is simply absent, its INDEX.CSV line is ignored;
  * maps folders to network labels (own.categories in configs/train.yaml);
  * splits: a speaker with fewer than own.min_clips_per_speaker clips goes to
    test as a whole (an unseen speaker), the others per file by a hash of the
    file name (own.test_pct / own.val_pct) - the same on every machine;
  * copies the clips to data/own/<label>/ and writes data/own/manifest.csv:
        path,label,category,speaker,style,word,split,ms,peak_dbfs,clipped
  * prints the counts per label x split x style.

data/own is not committed (the repository is public; this is a real voice).

    python scripts/04_import_own.py --src "C:/Users/Caizy/Desktop/MeshTalk_KWS_Data"
    python scripts/04_import_own.py --src DATASET.zip
"""
import argparse
import collections
import csv
import os
import shutil
import sys
import tempfile
import wave
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kws.cfg import load, stable_hash  # noqa: E402


def find_dataset(root):
    """the folder that holds the category folders (DATASET itself)"""
    for dirpath, dirnames, filenames in os.walk(root):
        names = {d.upper() for d in dirnames}
        if "INDEX.CSV" in {f.upper() for f in filenames} or names & {"HELP", "HELPME", "SOS"}:
            return dirpath
    return None


def parse_name(stem):
    """'A_LOUD_0012' -> (A, LOUD, '', 12); 'A_HELLO_NORMAL_0003' -> (A, NORMAL, HELLO, 3)"""
    parts = stem.split("_")
    if len(parts) < 3 or not parts[-1].isdigit():
        return None
    spk, style, num = parts[0], parts[-2], int(parts[-1])
    word = "_".join(parts[1:-2])
    return spk, style, word, num


def wav_info(path):
    """(ok, reason, ms)"""
    try:
        with wave.open(path, "rb") as w:
            if w.getnchannels() != 1 or w.getsampwidth() != 2:
                return False, "not 16 bit mono", 0
            n = w.getnframes()
            if n == 0:
                return False, "empty", 0
            return True, "", int(n * 1000 / w.getframerate()), w.getframerate()
    except (wave.Error, EOFError, OSError) as e:
        return False, "unreadable (%s)" % e, 0


def read_index(ds):
    meta = {}
    for name in os.listdir(ds):
        if name.upper() == "INDEX.CSV":
            with open(os.path.join(ds, name), newline="", encoding="utf-8", errors="replace") as f:
                for r in csv.DictReader(f):
                    if r.get("file"):
                        meta[r["file"].replace("\\", "/").upper()] = r
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="DATASET folder, a folder containing it, or a .zip")
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--keep", action="store_true", help="add to data/own instead of replacing it")
    a = ap.parse_args()
    C = load(a.config)
    O = C["own"]

    tmp = None
    src = a.src
    if src.lower().endswith(".zip"):
        tmp = tempfile.mkdtemp(prefix="own_zip_")
        with zipfile.ZipFile(src) as z:
            z.extractall(tmp)
        src = tmp
    ds = find_dataset(src)
    if not ds:
        sys.exit("no DATASET found in %s (expected HELP/, HELPME/, SOS/ or INDEX.CSV)" % a.src)
    meta = read_index(ds)

    # every valid clip
    clips, skipped = [], collections.Counter()
    for cat in sorted(os.listdir(ds)):
        cdir = os.path.join(ds, cat)
        if not os.path.isdir(cdir):
            continue
        label = O["categories"].get(cat.upper())
        if label is None:
            skipped["unknown folder " + cat] += len(os.listdir(cdir))
            continue
        for fn in sorted(os.listdir(cdir)):
            stem, ext = os.path.splitext(fn)
            if ext.upper() != ".WAV":
                continue
            p = parse_name(stem)
            if not p:
                skipped["bad file name"] += 1
                continue
            info = wav_info(os.path.join(cdir, fn))
            if not info[0]:
                skipped[info[1]] += 1
                continue
            if info[3] != O["rate"]:
                skipped["not %d Hz" % O["rate"]] += 1
                continue
            spk, style, word, num = p
            m = meta.get(("%s/%s" % (cat, fn)).upper(), {})
            clips.append(dict(src=os.path.join(cdir, fn), file=fn, label=label, category=cat.upper(), speaker=spk,
                              style=style, word=word, ms=info[2], peak_dbfs=m.get("peak_dbfs", ""),
                              clipped=m.get("clipped", "")))

    # split
    per_spk = collections.Counter(c["speaker"] for c in clips)
    for c in clips:
        if per_spk[c["speaker"]] < O["min_clips_per_speaker"]:
            c["split"] = "test"
        else:
            h = stable_hash("own:" + c["category"] + "/" + c["file"]) % 100
            c["split"] = "test" if h < O["test_pct"] else "val" if h < O["test_pct"] + O["val_pct"] else "train"

    # copy + manifest
    out = O["dir"]
    if os.path.isdir(out) and not a.keep:
        shutil.rmtree(out)
    rows = []
    man = os.path.join(out, "manifest.csv")
    if a.keep and os.path.exists(man):
        with open(man, newline="") as f:
            rows = list(csv.DictReader(f))
    have = {r["path"] for r in rows}
    for c in clips:
        rel = "%s/%s_%s" % (c["label"].replace(" ", "_"), c["category"], c["file"])
        if rel in have:
            continue
        os.makedirs(os.path.join(out, os.path.dirname(rel)), exist_ok=True)
        shutil.copyfile(c["src"], os.path.join(out, rel))
        rows.append(dict(path=rel, label=c["label"], category=c["category"], speaker=c["speaker"], style=c["style"],
                         word=c["word"], split=c["split"], ms=c["ms"], peak_dbfs=c["peak_dbfs"], clipped=c["clipped"]))
    os.makedirs(out, exist_ok=True)
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "category", "speaker", "style", "word", "split", "ms",
                                          "peak_dbfs", "clipped"])
        w.writeheader()
        w.writerows(rows)
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)

    # summary
    print("imported %d clips from %s -> %s" % (len(clips), ds, man))
    tab = collections.Counter((r["label"], r["split"]) for r in rows)
    print("%-10s %6s %6s %6s" % ("label", "train", "val", "test"))
    for lab in C["labels"]:
        print("%-10s %6d %6d %6d" % (lab, tab[(lab, "train")], tab[(lab, "val")], tab[(lab, "test")]))
    sty = collections.Counter((r["label"], r["style"]) for r in rows)
    print("styles:", ", ".join("%s/%s %d" % (k[0], k[1], v) for k, v in sorted(sty.items())))
    print("speakers:", dict(collections.Counter(r["speaker"] for r in rows)))
    clipped = sum(1 for r in rows if str(r["clipped"]).isdigit() and int(r["clipped"]) >= 8)
    if clipped:
        print("note: %d clips are clipped (too loud) - kept, listed in manifest.csv" % clipped)
    for k, v in skipped.items():
        print("skipped %d: %s" % (v, k))


if __name__ == "__main__":
    main()
