"""check_c.py - compile the speaker engine (spk/c/ + c_ref/spk_engine.h) with
an exported speaker model and check it gives the same vector as Python.

    python -m spk.check_c out        # out has spk_model.c + spk_model_test.c

Needs gcc. Exit code 0 = everything matches.
"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "out"
    tmp = tempfile.mkdtemp()
    for f in ("spk_engine.c", "spk_model.h", "spk_test.c"):
        shutil.copy(os.path.join(HERE, "c", f), tmp)
    shutil.copy(os.path.join(ROOT, "c_ref", "spk_engine.h"), tmp)
    for f in ("spk_model.c", "spk_model_test.c"):
        shutil.copy(os.path.join(out, f), tmp)
    cmd = ["gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-I.", "spk_test.c", "spk_engine.c", "spk_model.c",
           "spk_model_test.c", "-lm", "-o", "spk_test"]
    subprocess.run(cmd, cwd=tmp, check=True)
    r = subprocess.run(["./spk_test"], cwd=tmp)
    shutil.rmtree(tmp)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
