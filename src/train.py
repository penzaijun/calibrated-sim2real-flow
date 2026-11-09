"""Train one point-token transformer: 20 input frames -> 20 output frames on the 32x64 grid, with a per-element sigma head.

Each training window is non-dimensionalised by its own velocity scale (masked mean of u over the 20 input frames); the
model sees z-scored velocities in those units plus three per-window channels: the standardised log of the scale, a
domain flag (1 for simulation windows) and log2 of the frame-skip rate. Every epoch mixes the real windows with as many
random windows from the simulation traces and half as many Reynolds-similarity windows (every r-th frame of a real trace,
velocities x r, r in {2, 3}). Loss: rel-L2 of the normalised tokens + tke_w x rel-L2 of the TKE map + nll_w x Gaussian
NLL of the sigma head (on the detached mean).

The run stops after --stop_epoch of a --epochs long OneCycle schedule and saves the EMA weights of that epoch
(<out>/ema_ep<N>.pt). With --fold k the cases of Reynolds-row fold k are left out of training (and out of the frame-skip
augmentation), for the out-of-fold predictions the width model is fitted on."""
import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model as M
import scoring as S

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)              # directory with train_windows / val_windows / real_traces / sim_traces .npz
p.add_argument("--out", required=True)               # run directory
p.add_argument("--config", default=os.path.join(HERE, "..", "configs", "model.json"))
p.add_argument("--seed", type=int, default=42); p.add_argument("--device", type=int, default=0)
p.add_argument("--fold", type=int, default=-1)       # >= 0: leave out Reynolds-row fold k (ROW_FOLD below)
p.add_argument("--epochs", type=int, default=70)     # length of the OneCycle schedule
p.add_argument("--stop_epoch", type=int, default=25) # stop and save after this epoch
p.add_argument("--bs", type=int, default=16); p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--wd", type=float, default=1e-5); p.add_argument("--beta2", type=float, default=0.99)
p.add_argument("--clip", type=float, default=10.0)
p.add_argument("--pct_start", type=float, default=0.3); p.add_argument("--div_factor", type=float, default=25.0)
p.add_argument("--tke_w", type=float, default=0.15); p.add_argument("--nll_w", type=float, default=0.1)
p.add_argument("--ema", type=float, default=0.999)
p.add_argument("--sim_frac", type=float, default=1.0)     # simulation windows per epoch, as a fraction of the real windows
p.add_argument("--skip_frac", type=float, default=0.5)    # frame-skip windows per epoch, as a fraction of the real windows
p.add_argument("--skip_rates", default="2,3"); p.add_argument("--skip_re_max", type=float, default=29000.0)   # only (case, r) with r x Re <= this
p.add_argument("--val_every", type=int, default=5)
p.add_argument("--max_windows", type=int, default=0)      # > 0: only the first N real windows (interface test)
p.add_argument("--compile", type=int, default=1)
a = p.parse_args()

torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = torch.device(f"cuda:{a.device}")
os.makedirs(a.out, exist_ok=True)
log = open(os.path.join(a.out, "log.txt"), "a")


def P(*s):
    m = " ".join(str(x) for x in s); print(m, flush=True); log.write(m + "\n"); log.flush()


P("args", vars(a))
# Reynolds-row index = round((Re - 3750) / 1275), 0..18; the five folds hold whole rows out (row 9 has no real data)
ROW_FOLD = {0: 0, 5: 0, 10: 0, 15: 0, 1: 1, 6: 1, 11: 1, 16: 1, 2: 2, 7: 2, 12: 2, 3: 3, 8: 3, 13: 3, 4: 4, 14: 4, 17: 4, 18: 4}


def row(re):
    return int(round((re - 3750) / 1275))


tr = np.load(f"{a.data}/train_windows.npz"); va = np.load(f"{a.data}/val_windows.npz")
Xtr, Ytr = tr["X"], tr["Y"]
held = None
if a.fold >= 0:
    cases = np.unique(tr["cond"], axis=0)
    held = np.array([c for c in cases if ROW_FOLD[row(c[0])] == a.fold])
    in_fold = np.array([any((c == h).all() for h in held) for c in tr["cond"]])
    keep = np.where(~in_fold)[0]; Xtr, Ytr = Xtr[keep], Ytr[keep]
    P("fold %d: held-out cases %s, %d training windows" % (a.fold, [tuple(int(v) for v in c) for c in held], len(keep)))
if a.max_windows:
    Xtr, Ytr = Xtr[:a.max_windows], Ytr[:a.max_windows]
torch.set_float32_matmul_precision("high")


def window_scale_np(X, chunk=512):   # (N, T, H, W, 2) -> (N,): mean of u over the measured (non-zero) elements of the window
    out = np.empty(len(X), np.float32)
    for i in range(0, len(X), chunk):
        u = X[i:i + chunk, ..., 0].astype(np.float64)
        out[i:i + chunk] = np.maximum(u.sum(axis=(1, 2, 3)) / np.maximum((u != 0).sum(axis=(1, 2, 3)), 1), 1e-3)
    return out


def window_scale(X):   # the same on a (B, T, H, W, 2) tensor -> (B, 1, 1, 1, 1)
    u = X[..., 0]; n = (u != 0).float().sum(dim=(1, 2, 3)).clamp_min(1.0)
    return (u.sum(dim=(1, 2, 3)) / n).clamp_min(1e-3).view(-1, 1, 1, 1, 1)


def channel_stats(X, sw, chunk=512):   # per-channel mean / std of X / sw, float64 accumulation
    s1 = np.zeros(2); s2 = np.zeros(2); n = 0
    for i in range(0, len(X), chunk):
        Z = X[i:i + chunk].astype(np.float64) / sw[i:i + chunk, None, None, None, None]
        s1 += Z.sum(axis=(0, 1, 2, 3)); s2 += (Z ** 2).sum(axis=(0, 1, 2, 3)); n += Z[..., 0].size
    m = s1 / n; s = np.sqrt(np.maximum(s2 / n - m ** 2, 1e-12))
    return torch.tensor(m, dtype=torch.float32).to(dev), torch.tensor(s, dtype=torch.float32).to(dev)


SW = window_scale_np(Xtr); mean, std = channel_stats(Xtr, SW)
lsw_m = float(np.log(SW).mean()); lsw_s = float(np.log(SW).std())
P("window scale min/median/max %.4f %.4f %.4f, log-scale mean/std %.4f %.4f" % (SW.min(), np.median(SW), SW.max(), lsw_m, lsw_s))
NR = len(Xtr)

sz = np.load(f"{a.data}/sim_traces.npz"); ST = sz["T"]
NS_EP = int(round(a.sim_frac * NR)); rng_s = np.random.default_rng(a.seed + 1000)
sim_p = np.ones(ST.shape[0]); sim_p = sim_p / sim_p.sum()   # uniform over the simulation cases
P("sim traces", ST.shape, ST.dtype, "-> %d random sim windows per epoch next to %d real" % (NS_EP, NR))

tz = np.load(f"{a.data}/real_traces.npz"); tcond = tz["cond"]; TR = [tz["c%03d" % i] for i in range(len(tcond))]
if held is not None:   # the fold's cases must not come back through the augmentation
    kt = [i for i, c in enumerate(tcond) if not any((c == h).all() for h in held)]; TR = [TR[i] for i in kt]; tcond = tcond[kt]
RATES = [int(r) for r in a.skip_rates.split(",") if r]
PAIRS = [(i, r) for i in range(len(TR)) for r in RATES if tcond[i, 0] * r <= a.skip_re_max and len(TR[i]) >= 40 * r]
NT_EP = int(round(a.skip_frac * NR)); rng_t = np.random.default_rng(a.seed + 2000)
P("frame-skip: %d traces, %d (case, r) pairs, %d windows per epoch" % (len(TR), len(PAIRS), NT_EP))

x, y = tr["x"], tr["y"]; H, W = x.shape; NP = H * W
pos = np.stack([2 * (x - x.min()) / (x.max() - x.min()) - 1, 2 * (y - y.min()) / (y.max() - y.min()) - 1], -1).reshape(1, NP, 2)
pos = torch.tensor(pos, dtype=torch.float32).to(dev)
VX, VY = va["X"], va["Y"]
P("train", Xtr.shape, "val", VX.shape)

cfg = json.load(open(a.config))["model"]
N_EXTRA = 3   # log window scale, domain flag, log2 frame skip
model = M.build(cfg, 40 + N_EXTRA, 80).to(dev)
P("params %.2fM" % (sum(q.numel() for q in model.parameters()) / 1e6))
ema = copy.deepcopy(model).eval().requires_grad_(False)
ema_step = 0


def ema_update():
    global ema_step
    ema_step += 1
    with torch.no_grad():
        d = min(a.ema, (1 + ema_step) / (10 + ema_step))
        ep_ = list(ema.parameters()); torch._foreach_mul_(ep_, d); torch._foreach_add_(ep_, [q.detach() for q in model.parameters()], alpha=1 - d)
        for be, bm in zip(ema.buffers(), model.buffers()):
            be.copy_(bm)


fmodel = torch.compile(model) if a.compile else model


def to_tokens(X):   # (B, 20, H, W, 2) in window units -> (B, NP, 40) z-scored
    X = (X - mean) / std
    return X.permute(0, 2, 3, 1, 4).reshape(X.shape[0], NP, -1)


def from_tokens(Z, m, s):
    return Z.reshape(Z.shape[0], H, W, 20, 2).permute(0, 3, 1, 2, 4) * s + m


def extra_tokens(sw, dom, skip):   # (B, NP, 3): standardised log window scale, domain flag, log2 frame skip
    return torch.stack([(torch.log(sw.view(-1)) - lsw_m) / lsw_s, dom, skip], -1)[:, None, :].expand(-1, NP, -1)


def kinetic(x):   # (B, T, H, W, 2) -> (B, H, W), the scorer's TKE map
    return 0.5 * (x[..., 0].var(dim=1, unbiased=False) + x[..., 1].var(dim=1, unbiased=False))


def rel_l2(pred, tgt):
    return ((pred - tgt).flatten(1).norm(dim=1) / tgt.flatten(1).norm(dim=1).clamp_min(1e-8)).mean()


def tke_rel_l2(pred, tgt):
    return ((kinetic(pred) - kinetic(tgt)).flatten(1).norm(dim=1) / kinetic(tgt).flatten(1).norm(dim=1).clamp_min(1e-8)).mean()


_HS = np.linspace(1e-6, 0.6, 3000); _SG = np.geomspace(1e-5, 0.5, 512); _ERF = np.vectorize(math.erf)
_HTAB = _HS[np.argmax(_ERF(_HS[None] / (math.sqrt(2) * _SG[:, None])) * np.exp(-2 * _HS[None] / S.SIGMA_GLOBAL), 1)]


def hstar(sig, c=1.25):   # SPS-optimal half-width for a Gaussian error of std c * sigma (validation log only)
    return np.interp(np.log(np.clip(c * sig, 1e-5, 0.5)), np.log(_SG), _HTAB).astype(np.float32)


@torch.no_grad()
def validate(net):
    preds = []; sigs = []
    for i in range(0, len(VX), 16):
        Xg = torch.as_tensor(VX[i:i + 16]).to(dev); sw = window_scale(Xg); Xg = Xg / sw
        xb = torch.cat([to_tokens(Xg), extra_tokens(sw, torch.zeros(len(Xg), device=dev), torch.zeros(len(Xg), device=dev))], -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = net(pos.expand(xb.shape[0], -1, -1), xb).float()
        preds.append((from_tokens(z[..., :40], mean, std) * sw).cpu().numpy())
        sigs.append((from_tokens(torch.exp(0.5 * z[..., 40:80].clamp(-12, 6)), 0.0, std) * sw).cpu().numpy())
    pred = np.concatenate(preds); sig = np.concatenate(sigs)
    p3 = np.concatenate([pred, np.zeros_like(pred[..., :1])], -1); t3 = np.concatenate([VY, np.zeros_like(VY[..., :1])], -1)
    r = float(S.rel_l2_per_sample(p3, t3, 2).mean()); k = float(S.tke_rel_l2_per_sample(p3, t3, 2).mean()); mv = S.mvpe_rel_l2(p3, t3)
    h = hstar(sig); sps, _ = S.aggregate_sps(p3, t3, 2, np.concatenate([pred - h, p3[..., 2:]], -1), np.concatenate([pred + h, p3[..., 2:]], -1))
    sc = {"rel_l2": r, "tke": k, "mvpe": mv, "rel_l2_score": S.score_error(r), "tke_score": S.score_error(k), "mvpe_score": S.score_error(mv), "sps_score": S.score_sps(sps)}
    sc["acc3"] = 0.31350542 * sc["rel_l2_score"] + 0.10178498 * sc["tke_score"] + 0.10178498 * sc["mvpe_score"]
    return sc


opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, a.beta2))
STEPS_EP = math.ceil((NR + NS_EP + NT_EP) / a.bs)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * STEPS_EP, pct_start=a.pct_start, div_factor=a.div_factor, anneal_strategy="cos")

for ep in range(1, a.epochs + 1):
    # this epoch's simulation windows (case, start) and frame-skip windows (case, r, start)
    sim_c = rng_s.choice(ST.shape[0], NS_EP, p=sim_p); sim_s = rng_s.integers(0, ST.shape[1] - 40 + 1, NS_EP)
    ks = rng_t.integers(0, len(PAIRS), NT_EP)
    skip = [(PAIRS[k][0], PAIRS[k][1], int(rng_t.integers(0, len(TR[PAIRS[k][0]]) - 40 * PAIRS[k][1] + 1))) for k in ks]
    model.train(); perm = np.random.permutation(NR + NS_EP + NT_EP); tl = 0.0; n = 0; parts = np.zeros(3); t0 = time.time(); gsum = 0.0; gmax = 0.0
    for i in range(0, len(perm), a.bs):
        idx = np.sort(perm[i:i + a.bs])   # positions < NR: real windows; then this epoch's sim windows; then its frame-skip windows
        js = idx[(idx >= NR) & (idx < NR + NS_EP)] - NR; jt = idx[idx >= NR + NS_EP] - NR - NS_EP; idx = idx[idx < NR]
        Ws = [ST[sim_c[j], sim_s[j]:sim_s[j] + 40].astype(np.float32) for j in js] + \
             [TR[skip[j][0]][skip[j][2]:skip[j][2] + 40 * skip[j][1]:skip[j][1]] * np.float32(skip[j][1]) for j in jt]
        Ws = np.stack(Ws) if Ws else np.zeros((0, 40) + Xtr.shape[2:], np.float32)
        Xg = torch.as_tensor(np.concatenate([Xtr[idx], Ws[:, :20]])).to(dev); Yg = torch.as_tensor(np.concatenate([Ytr[idx], Ws[:, 20:]])).to(dev)
        dom = torch.cat([torch.zeros(len(idx)), torch.ones(len(js)), torch.zeros(len(jt))]).to(dev)
        fskip = torch.cat([torch.zeros(len(idx) + len(js)), torch.tensor([math.log2(skip[j][1]) for j in jt], dtype=torch.float32)]).to(dev)
        sw = window_scale(Xg); Xg = Xg / sw; Yg = Yg / sw   # everything below is in the window's own velocity units
        xb = torch.cat([to_tokens(Xg), extra_tokens(sw, dom, fskip)], -1); yb = to_tokens(Yg)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = fmodel(pos.expand(len(Xg), -1, -1), xb)
        out = out.float(); mu = out[..., :40]; lv = out[..., 40:80].clamp(-12, 6)
        l_rel = rel_l2(mu, yb)
        l_tke = tke_rel_l2(from_tokens(mu, mean, std), from_tokens(yb, mean, std))
        l_nll = (0.5 * (torch.exp(-lv) * (mu.detach() - yb) ** 2 + lv)).mean()   # sigma head: learns the error scale without pulling on mu
        loss = l_rel + a.tke_w * l_tke + a.nll_w * l_nll
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip).item(); gsum += gn; gmax = max(gmax, gn)
        lval = loss.item()
        if not (math.isfinite(lval) and math.isfinite(gn)):   # stop before a non-finite step reaches the weights / the EMA
            raise FloatingPointError(f"non-finite training step at epoch {ep}, batch {i // a.bs}: loss {lval}, grad norm {gn}")
        opt.step(); sched.step(); ema_update()
        tl += lval; n += 1; parts += [l_rel.item(), l_tke.item(), l_nll.item()]
    P(f"ep {ep} train_loss {tl / n:.4f} parts {[round(float(v) / n, 4) for v in parts]} lr {sched.get_last_lr()[0]:.2e} gnorm {gsum / n:.3f}/{gmax:.2f} time {time.time() - t0:.0f}s")
    if ep % a.val_every == 0 or ep == a.stop_epoch:
        model.eval()
        P("VAL ep", ep, "raw", json.dumps({k: round(v, 4) for k, v in validate(fmodel).items()}))
        P("VAL ep", ep, "ema", json.dumps({k: round(v, 4) for k, v in validate(ema).items()}))
        model.train()
    if ep == a.stop_epoch:
        sd = ema.state_dict()
        bad = [k for k, v in sd.items() if v.is_floating_point() and not bool(torch.isfinite(v).all())]
        if bad:
            raise FloatingPointError(f"non-finite EMA weights at epoch {ep}: {bad[:3]}")
        torch.save({"model": sd, "epoch": ep, "ema": a.ema, "mean": mean.tolist(), "std": std.tolist(), "lsw": [lsw_m, lsw_s],
                    "pos": pos.cpu(), "c_in": 40 + N_EXTRA, "c_out": 80}, os.path.join(a.out, f"ema_ep{ep}.pt"))
        P("saved ema_ep%d.pt; STOP" % ep)
        break
P("peak GPU memory %.2f GB" % (torch.cuda.max_memory_allocated(dev) / 1e9))
P("DONE")
