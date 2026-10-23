"""model.py - SpkNet, the separate speaker (voice print) network of phase B
(CONTRACT_SPK.md). Its C twin is spk/c/spk_engine.c; spk/export.py writes
spk_model.c and repeats the C arithmetic in numpy (reference()).

    x: log-mel [N, F, 40] (kws.features.logmel, F = spk_frames) -> (x - in_mean) / in_std
    L1..Lk: Conv1d over time (kernel k_i, dilation d_i, 'same' zero padding) + BN + ReLU
    statistics pooling: mean and standard deviation over time  -> 2 * C_last
    emb: Linear(2 * C_last, dim) (+ BN folded into it at export), divided by its length

Sizes come from configs/spk.yaml (net:).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

BANDS = 40
STD_EPS = 1e-5


class SpkNet(nn.Module):
    def __init__(self, frames=81, channels=(128, 128, 128, 256), kernels=(5, 3, 3, 1), dilations=(1, 2, 3, 1), dim=128):
        super().__init__()
        self.cfg = dict(frames=frames, channels=list(channels), kernels=list(kernels), dilations=list(dilations), dim=dim)
        self.register_buffer("in_mean", torch.zeros(BANDS))
        self.register_buffer("in_std", torch.ones(BANDS))
        self.convs, self.bns = nn.ModuleList(), nn.ModuleList()
        cin = BANDS
        for c, k, d in zip(channels, kernels, dilations):
            self.convs.append(nn.Conv1d(cin, c, k, dilation=d, padding=d * (k - 1) // 2))
            self.bns.append(nn.BatchNorm1d(c))
            cin = c
        self.fc = nn.Linear(2 * cin, dim)
        self.bn_emb = nn.BatchNorm1d(dim)

    def forward(self, x):
        """x [N, F, 40] -> L2-normalised embedding [N, dim]"""
        x = ((x - self.in_mean) / self.in_std).transpose(1, 2)      # [N, 40, F]
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x)))
        mu = x.mean(dim=2)
        sd = torch.sqrt(((x - mu[:, :, None]) ** 2).mean(dim=2) + STD_EPS)
        e = self.bn_emb(self.fc(torch.cat([mu, sd], dim=1)))
        return F.normalize(e, dim=1)

    def macs(self):
        """multiply-adds of one embedding"""
        n, cin, f = 0, BANDS, self.cfg["frames"]
        for c, k in zip(self.cfg["channels"], self.cfg["kernels"]):
            n += f * cin * c * k
            cin = c
        return n + 2 * cin * self.cfg["dim"]
