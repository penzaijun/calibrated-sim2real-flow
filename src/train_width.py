"""Interval-width model: a small CNN that rescales the Gaussian-optimal half-width of every predicted element and shifts
the interval centre, fitted on out-of-fold residuals with a smoothed version of the scorer's SPS.

Per element (lead frame t, pixel, channel u / v) 20 features are computed from the input window X, the prediction mu and
its sigma only (no target): see FEATURES. u and v are stacked as channels (2 x 20) of a 32x64 image per lead frame; three
3x3 convolutions (dilation 2 after the first, SiLU) and a 1x1 layer output, per channel, eta and an offset o:

    half-width h = h*(sigma * exp(clamp(eta, -2.5, 2.5)))     h*(s): SPS-optimal half-width for a Gaussian error of std s
    centre       = mu + sigma * tanh(o)

The last layer starts at zero, i.e. at the plain Gaussian-optimal band h*(sigma). Several members (seeds) are averaged
on their raw (eta, o) maps. The fitted weights are written as JSON for the inference runtime.

usage: python train_width.py <data dir> <bundles dir> <output json> [steps] [members]"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as Fn
from scipy.special import erf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scoring as S

DATA, BUNDLES, OUT = sys.argv[1:4]
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 6000
MEMBERS = int(sys.argv[5]) if len(sys.argv) > 5 else 3
HIDDEN, N_CONV, DIL, BS, LR = 64, 3, 2, 12, 3e-3
NFOLD = 5
dev = "cuda"
SG = S.SIGMA_GLOBAL
FEATURES = ["const", "r_sd", "r_diff", "r_delta", "valid", "lead", "chan", "lsig", "r_sig3x3", "r_xsp3x3", "r_mutstd", "mulevel",
            "luref", "zero5x5", "r_sigt", "lead*r_sd", "lead*lsig", "mu_n", "xl_n", "r_sf"]

# h*(s) = argmax_h erf(h / (sqrt2 s)) * exp(-2h / SIGMA_GLOBAL), tabulated on a log-uniform grid of s
SIG_GRID = np.geomspace(1e-5, .5, 1024)
HG = np.linspace(1e-6, .6, 4000)
HSTAR = HG[np.argmax(erf(HG[None] / (np.sqrt(2) * SIG_GRID[:, None])) * np.exp(-2 * HG[None] / SG), axis=1)]
TABLE = torch.tensor(HSTAR, dtype=torch.float32, device=dev); NG = len(SIG_GRID); LO, HI = np.log(1e-5), np.log(.5)


def hstar_np(sigma, c=1.25):   # the plain band h*(c sigma), numpy
    return np.interp(np.log(np.clip(sigma * c, 1e-5, .5)), np.log(SIG_GRID), HSTAR).astype(np.float32)


def hstar_t(s):   # differentiable table lookup
    t = ((torch.log(s.clamp(1e-5, .5)) - LO) / (HI - LO) * (NG - 1)); i = t.floor().long().clamp(0, NG - 2); q = t - i
    return TABLE[i] * (1 - q) + TABLE[i + 1] * q


def accuracy(mu, y):   # the scorer's per-window accuracy prefactor of SPS
    d = S.rel_l2_per_sample(mu, y, 2); t = S.tke_rel_l2_per_sample(mu, y, 2); m = S.mvpe_rel_l2_per_sample(mu, y)
    return .5 * .5 / (.5 + d) + .3 * .5 / (.5 + t) + .2 * .5 / (.5 + m)


def pool(t, k):   # (B, T, H, W, C) -> spatial k x k mean (edge-aware)
    Bn, T, H, W, C = t.shape
    u = t.permute(0, 1, 4, 2, 3).reshape(-1, 1, H, W)
    return Fn.avg_pool2d(u, k, stride=1, padding=k // 2, count_include_pad=False).reshape(Bn, T, C, H, W).permute(0, 1, 3, 4, 2)


def features(X, mu, sigma):
    """X, mu, sigma (B, 20, H, W, 2) physical -> (B, 20, H, W, 2, 20). r_* = log ratio of a statistic to sigma; *_n =
    signed velocity over the window's mean |u|; r_sf = the input's structure function at the element's own lead."""
    sg = sigma.clamp(1e-5, .5); r = lambda a: torch.log((a + 1e-5) / sg).clamp(-4, 4)
    sd = X.std(dim=1, unbiased=False, keepdim=True); diff = X.diff(dim=1).square().mean(1, keepdim=True).sqrt()
    delta = (mu - X[:, -1:]).abs(); valid = (X != 0).float().mean(1, keepdim=True)
    lead = (torch.arange(20, device=X.device, dtype=torch.float32) / 19).view(1, 20, 1, 1, 1).expand_as(sg)
    chan = torch.arange(2, device=X.device, dtype=torch.float32).view(1, 1, 1, 1, 2).expand_as(sg)
    lsig = torch.log(sg / SG).clamp(-8, 3)
    sig3 = pool(sg, 3); xl = X[:, -1:]; xsp = (pool(xl * xl, 3) - pool(xl, 3) ** 2).clamp_min(0).sqrt()
    mutstd = mu.std(dim=1, unbiased=False, keepdim=True); mulevel = torch.log((mu.abs() + 1e-4) / SG).clamp(-8, 3)
    vmask = (X != 0).any(dim=4, keepdim=True).float()
    uref = (X[..., :1].abs() * vmask).sum(dim=(1, 2, 3, 4), keepdim=True) / vmask.sum(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1)
    luref = torch.log(uref / SG).clamp(-3, 4).expand_as(sg)
    zero5 = (1 - pool(vmask[:, -1:], 5)).expand_as(sg); sigt = sg.mean(dim=1, keepdim=True)
    sf = torch.stack([(X[:, l:] - X[:, :20 - l]).abs().mean(dim=1) for l in [min(j + 1, 19) for j in range(20)]], 1)
    f = [torch.ones_like(sg), r(sd), r(diff), r(delta), valid.expand_as(sg), lead, chan, lsig,
         r(sig3), r(xsp), r(mutstd), mulevel, luref, zero5, r(sigt), lead * r(sd), lead * lsig,
         (mu / (uref + 1e-4)).clamp(-5, 5), (xl / (uref + 1e-4)).clamp(-5, 5), r(sf)]
    return torch.stack([t.expand_as(sg) for t in f], -1)


class CNN(torch.nn.Module):
    def __init__(self, nfeat):
        super().__init__(); layers = []; cin = 2 * nfeat
        for i in range(N_CONV):
            d_ = DIL if i > 0 else 1
            layers += [torch.nn.Conv2d(cin, HIDDEN, 3, padding=d_, dilation=d_), torch.nn.SiLU()]; cin = HIDDEN
        layers += [torch.nn.Conv2d(cin, 4, 1)]; self.net = torch.nn.Sequential(*layers)
        torch.nn.init.zeros_(self.net[-1].weight); torch.nn.init.zeros_(self.net[-1].bias)   # start at the Gaussian-optimal band

    def raw(self, F):   # F (B, 20, H, W, 2, NF) -> (B, 20, H, W, chan, [eta, offset])
        Bn, T, H, Wd, C, nf = F.shape
        x = F.permute(0, 1, 4, 5, 2, 3).reshape(Bn * T, C * nf, H, Wd)
        return self.net(x).reshape(Bn, T, 2, 2, H, Wd).permute(0, 1, 4, 5, 2, 3)


def bands(o, sg):   # raw (eta, offset) -> half-width h, centre offset d
    return hstar_t(sg * torch.exp(o[..., 0].clamp(-2.5, 2.5))), sg * torch.tanh(o[..., 1])


def load():
    z = np.load(f"{DATA}/train_windows.npz"); folds = [np.load(f"{BUNDLES}/fold{k}.npz") for k in range(NFOLD)]
    idx = np.concatenate([f["idx"] for f in folds])
    d = {"X": z["X"][idx], "Y": z["Y"][idx], "mu": np.concatenate([f["mu"] for f in folds]), "sigma": np.concatenate([f["sigma"] for f in folds])}
    dz = np.load(f"{DATA}/val_windows.npz"); db = np.load(f"{BUNDLES}/dev.npz")
    dev_panel = {"X": dz["X"][db["idx"]], "Y": dz["Y"][db["idx"]], "mu": db["mu"], "sigma": db["sigma"]}
    return d, dev_panel


def batch(d, idx):
    Xb, Yb, mb, sb = [torch.as_tensor(d[k][idx], device=dev) for k in ("X", "Y", "mu", "sigma")]
    A = torch.as_tensor(accuracy(d["mu"][idx], d["Y"][idx]), device=dev, dtype=torch.float32)
    return features(Xb, mb, sb), sb.clamp(1e-5, .5), Yb - mb, (Yb != 0).float(), A


def fit(d, steps, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); train_idx = np.arange(len(d["X"]))
    m = CNN(len(FEATURES)).to(dev); opt = torch.optim.Adam(m.parameters(), lr=LR); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for it in range(steps):
        idx = np.sort(rng.choice(train_idx, BS, replace=False)); F, sg, e, valid, A = batch(d, idx)
        tau_rel = .12 - .09 * it / steps; h, dd = bands(m.raw(F), sg); tau = tau_rel * h.detach() + 1e-5   # soft indicator 0.12 h -> 0.03 h
        term = A.view(-1, 1, 1, 1, 1) * torch.sigmoid((h - (e - dd).abs()) / tau) * torch.exp(-2 * h / SG)
        loss = -(term * valid).sum() / valid.sum() + 1e-5 * sum(p.square().sum() for p in m.parameters())
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    bad = [k for k, v in m.state_dict().items() if v.is_floating_point() and not bool(torch.isfinite(v).all())]
    if bad:
        raise FloatingPointError(f"width CNN (seed {seed}): non-finite weights after {steps} steps, e.g. {bad[:3]}")
    return m.eval()


@torch.no_grad()
def sps_gain(members, p):   # SPS (0-100) of the width model's band on a panel, and its gain over the plain h*(1.25 sigma) band
    hs, ds = [], []
    for i in range(0, len(p["X"]), 8):
        Xb, mb, sb = [torch.as_tensor(p[k][i:i + 8], device=dev) for k in ("X", "mu", "sigma")]
        F = features(Xb, mb, sb); o = torch.stack([m.raw(F) for m in members]).mean(0)
        h, dd = bands(o, sb.clamp(1e-5, .5)); hs.append(h.cpu().numpy()); ds.append(dd.cpu().numpy())
    h, dd = np.concatenate(hs), np.concatenate(ds); mu, y = p["mu"], p["Y"]
    sps = 100 * S.aggregate_sps(mu, y, 2, mu + dd - h, mu + dd + h)[0]; h0 = hstar_np(p["sigma"])
    return sps, sps - 100 * S.aggregate_sps(mu, y, 2, mu - h0, mu + h0)[0]


t0 = time.time(); d, dev_panel = load(); print("out-of-fold windows", len(d["X"]), "%.0fs" % (time.time() - t0), flush=True)
members = []
for seed in range(MEMBERS):
    t1 = time.time(); members.append(fit(d, STEPS, seed))
    print("member %d (seed %d): dev SPS %.3f (%+.3f vs h*(1.25 sigma))  %.0fs" % (len(members), seed, *sps_gain(members[-1:], dev_panel), time.time() - t1), flush=True)
sps, gain = sps_gain(members, dev_panel)
print("ensemble of %d: dev SPS %.3f (%+.3f)" % (len(members), sps, gain), flush=True)
conv = lambda m: [[l.weight.detach().cpu().tolist(), l.bias.detach().cpu().tolist()] for l in m.net if isinstance(l, torch.nn.Conv2d)]
out = {"features": FEATURES, "dilation": DIL, "members": [conv(m) for m in members],
       "fit": {"steps": STEPS, "windows": len(d["X"]), "params": sum(p.numel() for p in members[0].parameters()), "dev_sps": sps, "dev_gain": gain}}
with open(OUT, "w") as f:
    json.dump(out, f)
print("exported", OUT, "WIDTH_DONE %.0fs" % (time.time() - t0), flush=True)
