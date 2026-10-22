"""export.py - KwsNet -> kws_model.c for the firmware (Ai/kws_model.c), plus
kws_model_test.c: an input and the outputs the C code must give for it.

    python -m kws.export model.pt out_dir --labels "help|help me|sos|_other" \
        --thr 0.90 0.80 0.65 --spk-thr 0.70 --hits 3 --hits-ms 20000

BatchNorm is folded into the convolutions, every weight tensor becomes int8
with one scale (symmetric, max |w| -> 127). reference() repeats the C
arithmetic in numpy with the int8 weights, so check_c.py can compare.
"""
import argparse
import time
import zlib
import numpy as np
import torch

from .model import KwsNet


def fold(conv, bn):
    w = conv.weight.detach().double().numpy()
    b = conv.bias.detach().double().numpy() if conv.bias is not None else np.zeros(w.shape[0])
    g = bn.weight.detach().double().numpy()
    beta = bn.bias.detach().double().numpy()
    mu = bn.running_mean.detach().double().numpy()
    var = bn.running_var.detach().double().numpy()
    k = g / np.sqrt(var + bn.eps)
    return w * k[:, None, None, None], (b - mu) * k + beta


def quant(w):
    m = float(np.max(np.abs(w)))
    s = m / 127.0 if m > 0 else 1.0
    return np.clip(np.round(w / s), -127, 127).astype(np.int8), np.float32(s)


def layers(net):
    """dict name -> (int8 weights, scale, float32 bias)"""
    out = {}
    for i, (c, b) in enumerate([(net.conv1, net.bn1), (net.conv2, net.bn2), (net.conv3, net.bn3)], 1):
        w, bias = fold(c, b)
        q, s = quant(w)
        out["%d" % i] = (q, s, bias.astype(np.float32))
    for name, lin in (("4", net.fc1), ("5", net.fc2), ("6", net.spk)):
        if lin is None:
            continue
        q, s = quant(lin.weight.detach().double().numpy())
        out[name] = (q, s, lin.bias.detach().float().numpy())
    return out


def reference(net, L, feat):
    """numpy copy of kws_engine_run(): feat [T, 40] -> (probs, emb)"""
    x = (feat.astype(np.float32) - net.in_mean.numpy()) / net.in_std.numpy()
    a = x[None].astype(np.float64)                     # [1, T, 40]

    def conv(a, q, s, b):
        cin, h, w = a.shape
        p = np.pad(a, ((0, 0), (1, 1), (1, 1)))
        out = np.zeros((q.shape[0], h, w))
        for dy in range(3):
            for dx in range(3):
                out += np.einsum("oc,chw->ohw", q[:, :, dy, dx].astype(np.float64), p[:, dy:dy + h, dx:dx + w])
        return out * float(s) + b[:, None, None]

    def pool(a):
        c, h, w = a.shape
        a = a[:, :h // 2 * 2, :w // 2 * 2]
        return a.reshape(c, h // 2, 2, w // 2, 2).max(axis=(2, 4))

    a = pool(np.maximum(conv(a, *L["1"]), 0))
    a = pool(np.maximum(conv(a, *L["2"]), 0))
    g = np.maximum(conv(a, *L["3"]), 0).mean(axis=(1, 2))
    q, s, b = L["4"]
    h = np.maximum(q.astype(np.float64) @ g * float(s) + b, 0)
    q, s, b = L["5"]
    z = q.astype(np.float64) @ h * float(s) + b
    p = np.exp(z - z.max())
    p /= p.sum()
    emb = None
    if "6" in L:
        q, s, b = L["6"]
        e = q.astype(np.float64) @ g * float(s) + b
        emb = e / (np.sqrt((e * e).sum()) + 1e-9)
    return p, emb


def c_floats(name, a):
    a = np.asarray(a, np.float32).ravel()
    body = ",".join("%.9g" % v for v in a)
    return "static const float %s[%d] = {%s};\n" % (name, a.size, body)


def c_int8(name, a):
    a = np.asarray(a, np.int8).ravel()
    lines = []
    for i in range(0, a.size, 40):
        lines.append(",".join(str(int(v)) for v in a[i:i + 40]))
    return "static const int8_t %s[%d] = {\n%s};\n" % (name, a.size, ",\n".join(lines))


def write_model(net, L, labels, thr, spk_thr, hits, hits_ms, name, path, var="kws_model"):
    cfg = net.cfg
    blob = b"".join(L[k][0].tobytes() for k in sorted(L)) + labels.encode()
    mid = zlib.crc32(blob) & 0xFFFFFFFF
    s = ["/* %s.c - written by MeshTalk-KWS export.py, %s. Do not edit. */\n" % (var, time.strftime("%Y-%m-%d %H:%M")),
         '#include "kws_engine.h"\n\n']
    s.append(c_floats("in_mean", net.in_mean.numpy()))
    s.append(c_floats("in_std", net.in_std.numpy()))
    for k in sorted(L):
        q, sc, b = L[k]
        s.append(c_int8("w%s" % k, q))
        s.append(c_floats("b%s" % k, b))
    s.append("\nconst kws_model_t %s = {\n" % var)
    s.append('    .name = "%s", .id = 0x%08XU,\n' % (name, mid))
    s.append("    .T = %d, .B = 40, .c1 = %d, .c2 = %d, .c3 = %d, .hid = %d, .ncls = %d, .spk_dim = %d,\n"
             % (cfg["T"], cfg["c1"], cfg["c2"], cfg["c3"], cfg["hid"], cfg["ncls"], cfg["spk_dim"] if "6" in L else 0))
    s.append('    .labels = "%s",\n    .in_mean = in_mean, .in_std = in_std,\n' % labels)
    for k in "123456":
        if k in L:
            s.append("    .w%s = w%s, .s%s = %.9gf, .b%s = b%s,\n" % (k, k, k, float(L[k][1]), k, k))
    s.append("    .thr = { %.4ff, %.4ff, %.4ff }, .spk_thr = %.4ff, .hits = %d, .hits_ms = %uu,\n"
             % (thr[0], thr[1], thr[2], spk_thr, hits, hits_ms))
    s.append("};\n")
    open(path, "w").write("".join(s))
    return mid


def write_test(net, L, path, var="kws_model", seed=1):
    rng = np.random.default_rng(seed)
    T = net.cfg["T"]
    feat = (net.in_mean.numpy()[None, :] + rng.standard_normal((T, 40)).astype(np.float32)
            * net.in_std.numpy()[None, :]).astype(np.float32)
    p, emb = reference(net, L, feat)
    s = ["/* %s_test.c - input and expected output of %s (export.py). */\n" % (var, var),
         '#include "kws_engine.h"\n\n',
         "const float %s_test_in[%d] = {%s};\n" % (var, feat.size, ",".join("%.9g" % v for v in feat.ravel())),
         "const float %s_test_probs[%d] = {%s};\n" % (var, p.size, ",".join("%.9g" % v for v in p)),
         "const float %s_test_emb[%d] = {%s};\n" % (var, emb.size if emb is not None else 1,
                                                  ",".join("%.9g" % v for v in emb) if emb is not None else "0")]
    open(path, "w").write("".join(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="torch.save({'cfg': net.cfg, 'state': net.state_dict()}, ...)")
    ap.add_argument("out_dir")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--thr", type=float, nargs=3, default=[0.9, 0.8, 0.65])
    ap.add_argument("--spk-thr", type=float, default=0.7)
    ap.add_argument("--hits", type=int, default=3)
    ap.add_argument("--hits-ms", type=int, default=20000)
    ap.add_argument("--name", default="kws-" + time.strftime("%Y%m%d"))
    a = ap.parse_args()
    ck = torch.load(a.model, map_location="cpu")
    net = KwsNet(**ck["cfg"])
    net.load_state_dict(ck["state"])
    net.eval()
    assert len(a.labels.split("|")) == net.cfg["ncls"], "labels do not match ncls"
    L = layers(net)
    mid = write_model(net, L, a.labels, a.thr, a.spk_thr, a.hits, a.hits_ms, a.name, a.out_dir + "/kws_model.c")
    write_test(net, L, a.out_dir + "/kws_model_test.c")
    print("written kws_model.c / kws_model_test.c, id %08X" % mid)


if __name__ == "__main__":
    main()
