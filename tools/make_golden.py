import sys, numpy as np, torch
import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kws import features, export
from kws.model import KwsNet
out = sys.argv[1]
torch.manual_seed(7)
net = KwsNet(ncls=4)
# make BN statistics non-trivial
with torch.no_grad():
    for bn in (net.bn1, net.bn2, net.bn3):
        bn.running_mean.uniform_(-0.2, 0.2); bn.running_var.uniform_(0.5, 2.0)
        bn.weight.uniform_(0.5, 1.5); bn.bias.uniform_(-0.1, 0.1)
    net.in_mean.copy_(torch.linspace(-8, -2, 40)); net.in_std.copy_(torch.linspace(2, 3, 40))
net.eval()
L = export.layers(net)
export.write_model(net, L, "help|help me|sos|_other", [0.9, 0.8, 0.65], 0.7, 3, 20000, "test-random", out + "/kws_test_model.c", var="kws_test_model")
export.write_test(net, L, out + "/kws_test_model_test.c", var="kws_test_model")
# check numpy reference against torch float forward (quantisation error only)
rng = np.random.default_rng(3)
feat = (net.in_mean.numpy() + rng.standard_normal((61, 40)).astype(np.float32) * net.in_std.numpy()).astype(np.float32)
p, e = export.reference(net, L, feat)
with torch.no_grad():
    lg, emb = net(torch.from_numpy(feat)[None])
pt = torch.softmax(lg, 1)[0].numpy()
print("torch vs int8 reference: probs diff %.4f, emb cos %.4f" % (abs(pt - p).max(), float((emb[0].numpy() * e).sum())))
# front-end golden
n = np.arange(16000)
x = 6000 * np.sin(2 * np.pi * (150 + 2500 * n / 16000) * n / 16000) + 1500 * np.sin(2 * np.pi * 3100 * n / 16000)
x += rng.standard_normal(16000) * 300
pcm = np.clip(np.round(x), -32768, 32767).astype(np.int16)
F = features.logmel(pcm)
with open(out + "/kws_fe_golden.c", "w") as f:
    f.write("/* kws_fe_golden.c - a test sound and its features from features.py */\n#include <stdint.h>\n")
    f.write("const int16_t kws_fe_golden_pcm[%d] = {%s};\n" % (pcm.size, ",".join(map(str, pcm))))
    f.write("const int kws_fe_golden_frames = %d;\n" % F.shape[0])
    f.write("const float kws_fe_golden_feat[%d] = {%s};\n" % (F.size, ",".join("%.9g" % v for v in F.ravel())))
print("frames", F.shape)
