"""check_c.py - compile the firmware's C code (c_ref/, copied from
MeshTalk_ESP32 Ai/ and Tests/) with an exported model and check that it gives
the same results as Python.

    python -m kws.check_c out_dir      # out_dir has kws_model.c + kws_model_test.c

Needs gcc. Prints the C test's report; exit code 0 = everything matches.
"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CREF = os.path.join(HERE, "..", "c_ref")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else None
    tmp = tempfile.mkdtemp()
    for f in os.listdir(CREF):
        shutil.copy(os.path.join(CREF, f), tmp)
    extra = []
    if out:
        shutil.copy(os.path.join(out, "kws_model.c"), os.path.join(tmp, "kws_model.c"))
        shutil.copy(os.path.join(out, "kws_model_test.c"), os.path.join(tmp, "kws_model_test.c"))
        extra = ["-DHAVE_MODEL_TEST", "kws_model_test.c"]
    src = ["test_kws.c", "kws_fe_golden.c", "kws_test_model.c", "kws_test_model_test.c",
           "kws_frontend.c", "kws_engine.c", "kws_detect.c", "kws_model.c"]
    cmd = ["gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Wno-unused-parameter", "-I."] + extra + src + ["-lm", "-o", "test_kws"]
    subprocess.run(cmd, cwd=tmp, check=True)
    r = subprocess.run(["./test_kws"], cwd=tmp)
    shutil.rmtree(tmp)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
