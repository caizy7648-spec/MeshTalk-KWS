"""export.py - SpkNet -> spk_model.c (+ spk_model_test.c) for spk/c/spk_engine.c.

    python -m spk.export models/spk_best.pt out --name spk-YYYYMMDD --threshold 0.6 --enrol-takes 5

BatchNorm is folded into the convolution before it (and bn_emb into fc);
weights become int8 with one float scale per output channel (max |w| -> 127).
reference() repeats the C arithmetic in numpy (float32, same order), so the
C test can compare (max difference < 1e-4).
"""
import argparse
import time
import zlib

import numpy as np
import torch

from .model import STD_EPS, SpkNet


def _fold(w, b, bn):
    g = bn.weight.detach().double().numpy()
    beta = bn.bias.detach().double().numpy()
    mu = bn.running_mean.detach().double().numpy()
    var = bn.running_var.detach().double().numpy()
    k = g / np.sqrt(var + bn.eps)
    return w * k.reshape((-1,) + (1,) * (w.ndim - 1)), (b - mu) * k + beta


def _quant_rows(w):
    """int8 per output row (channel) + float32 scales"""
    w2 = w.reshape(w.shape[0], -1)
    m = np.max(np.abs(w2), axis=1)
    s = np.where(m > 0, m / 127.0, 1.0)
    q = np.clip(np.round(w2 / s[:, None]), -127, 127).astype(np.int8)
    return q.reshape(w.shape), s.astype(np.float32)


def layers(net):
    """[(int8 w [o, c, k], scales [o], bias [o], k, d)] for the convolutions, then (int8 w [dim, 2C], scales, bias)"""
    out = []
    for conv, bn in zip(net.convs, net.bns):
        w, b = _fold(conv.weight.detach().double().numpy(), conv.bias.detach().double().numpy(), bn)
        q, s = _quant_rows(w)
        out.append((q, s, b.astype(np.float32), conv.kernel_size[0], conv.dilation[0]))
    w, b = _fold(net.fc.weight.detach().double().numpy(), net.fc.bias.detach().double().numpy(), net.bn_emb)
    q, s = _quant_rows(w)
    return out, (q, s, b.astype(np.float32))


def reference(net, L, feat):
    """numpy copy of spk_embed(): feat [F, 40] -> emb [dim] (float32 like the C code)"""
    convs, (fq, fs, fb) = L
    f32 = np.float32
    x = ((feat.astype(f32) - net.in_mean.numpy()) / net.in_std.numpy()).T.astype(f32)    # [40, F]
    F_ = x.shape[1]
    for q, s, b, k, d in convs:
        pad = d * (k - 1) // 2
        xp = np.pad(x, ((0, 0), (pad, pad)))
        acc = np.zeros((q.shape[0], F_), f32)
        for j in range(k):
            acc += (q[:, :, j].astype(f32) @ xp[:, j * d:j * d + F_]).astype(f32)
        x = np.maximum(acc * s[:, None] + b[:, None], 0).astype(f32)
    mu = x.mean(axis=1, dtype=f32)
    sd = np.sqrt(((x - mu[:, None]) ** 2).mean(axis=1, dtype=f32) + f32(STD_EPS)).astype(f32)
    p = np.concatenate([mu, sd]).astype(f32)
    e = ((fq.astype(f32) @ p) * fs + fb).astype(f32)
    return (e / (np.sqrt(np.sum(e * e, dtype=f32)) + f32(1e-9))).astype(f32)


def _ints(name, a, ctype="int8_t"):
    a = np.asarray(a).ravel()
    rows = [",".join(str(int(v)) for v in a[i:i + 40]) for i in range(0, a.size, 40)]
    return "static const %s %s[%d] = {\n%s};\n" % (ctype, name, a.size, ",\n".join(rows))


def _floats(name, a, static=True):
    a = np.asarray(a, np.float32).ravel()
    return "%sconst float %s[%d] = {%s};\n" % ("static " if static else "", name, a.size, ",".join("%.9g" % v for v in a))


def write_model(net, L, path, name, threshold, enrol_takes):
    convs, (fq, fs, fb) = L
    blob = b"".join(q.tobytes() for q, *_ in convs) + fq.tobytes() + name.encode() + time.strftime("%Y%m%d%H%M%S").encode()
    mid = zlib.crc32(blob) & 0xFFFFFFFF
    s = ["/* spk_model.c - written by MeshTalk-KWS spk/export.py, %s. Do not edit. */\n" % time.strftime("%Y-%m-%d %H:%M"),
         '#include "spk_model.h"\n\n', _floats("in_mean", net.in_mean.numpy()), _floats("in_std", net.in_std.numpy())]
    for i, (q, sc, b, k, d) in enumerate(convs):
        s += [_ints("w%d" % i, q), _floats("s%d" % i, sc), _floats("b%d" % i, b)]
    s += [_ints("fw", fq), _floats("fs", fs), _floats("fb", fb)]
    c = net.cfg
    s.append("\nconst spk_model_t spk_model = {\n")
    s.append('    .name = "%s", .id = 0x%08XU,\n' % (name, mid))
    s.append("    .frames = %d, .nlayers = %d, .dim = %d,\n" % (c["frames"], len(convs), c["dim"]))
    s.append("    .ch = { %s }, .k = { %s }, .d = { %s },\n" % (", ".join(str(v) for v in c["channels"]),
                                                             ", ".join(str(v) for v in c["kernels"]),
                                                             ", ".join(str(v) for v in c["dilations"])))
    s.append("    .in_mean = in_mean, .in_std = in_std,\n")
    s.append("    .w = { %s }, .s = { %s }, .b = { %s },\n" % (", ".join("w%d" % i for i in range(len(convs))),
                                                           ", ".join("s%d" % i for i in range(len(convs))),
                                                           ", ".join("b%d" % i for i in range(len(convs)))))
    s.append("    .fw = fw, .fs = fs, .fb = fb,\n")
    s.append("    .threshold = %.4ff, .enrol_takes = %d,\n};\n" % (threshold, enrol_takes))
    open(path, "w").write("".join(s))
    return mid


def write_test(net, L, path, seed=1):
    rng = np.random.default_rng(seed)
    Fr = net.cfg["frames"]
    feat = (net.in_mean.numpy()[None, :] + rng.standard_normal((Fr, 40)).astype(np.float32)
            * net.in_std.numpy()[None, :]).astype(np.float32)
    e = reference(net, layers(net), feat)
    open(path, "w").write("/* spk_model_test.c - input and expected output of spk_model (spk/export.py). */\n"
                          "#include <stdint.h>\n" + _floats("spk_test_in", feat, False) + _floats("spk_test_emb", e, False))
    return feat, e


def sizes(net, L):
    convs, (fq, fs, fb) = L
    wbytes = sum(q.size + 4 * (s.size + b.size) for q, s, b, *_ in convs) + fq.size + 4 * (fs.size + fb.size) + 8 * 40
    cmax = max(net.cfg["channels"])
    work = 40 * net.cfg["frames"] + 2 * cmax * net.cfg["frames"] + 2 * net.cfg["channels"][-1] + net.cfg["dim"]
    return dict(weight_bytes=int(wbytes), work_floats=int(work), work_bytes=int(4 * work), macs=int(net.macs()))


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = SpkNet(**ck["cfg"])
    net.load_state_dict(ck["state"])
    return net.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out_dir")
    ap.add_argument("--name", default="spk-" + time.strftime("%Y%m%d"))
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--enrol-takes", type=int, default=5)
    a = ap.parse_args()
    net = load(a.model)
    L = layers(net)
    mid = write_model(net, L, a.out_dir + "/spk_model.c", a.name, a.threshold, a.enrol_takes)
    write_test(net, L, a.out_dir + "/spk_model_test.c")
    print("written spk_model.c / spk_model_test.c, id %08X, %s" % (mid, sizes(net, L)))


if __name__ == "__main__":
    main()
