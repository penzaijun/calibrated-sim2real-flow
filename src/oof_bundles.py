"""Out-of-fold predictions for the width model.

For each Reynolds-row fold k, the three seeds of the model trained without that fold predict every training window of
the fold's cases (eager bf16 inference, unmeasured pixels set to 0). Their mean mu gets the variance restoration used
for the width-model data (each cell's 20-frame variance moved a fraction VARFIX of the way back to the members' mean
variance); sigma is the members' mean. The deployed three-seed ensemble does the same on the validation windows ("dev",
a diagnostic panel for the width-model log).

usage: python oof_bundles.py <data dir> <runs dir> <out dir> <checkpoint name>
       (fold models: <runs dir>/fold<k>_s<seed>/<checkpoint name>; deployed: <runs dir>/all_s<seed>/<checkpoint name>)"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model as M

DATA, RUNS, OUT, CKPT = sys.argv[1:5]
SEEDS = (42, 43, 44)
NFOLD = 5
VARFIX = 0.75
ROW_FOLD = {0: 0, 5: 0, 10: 0, 15: 0, 1: 1, 6: 1, 11: 1, 16: 1, 2: 2, 7: 2, 12: 2, 3: 3, 8: 3, 13: 3, 4: 4, 14: 4, 17: 4, 18: 4}
CFG = json.load(open(os.path.join(HERE, "..", "configs", "model.json")))["model"]
torch.set_float32_matmul_precision("high")


@torch.inference_mode()
def predict(ckpt_path, X, batch=16):
    """one checkpoint on (N, 20, H, W, 2) physical windows -> mu (unmeasured pixels 0), sigma; numpy float32"""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = M.build(CFG, ck["c_in"], ck["c_out"]); net.load_state_dict(ck["model"]); net = net.cuda().eval()
    mean = torch.tensor(ck["mean"], dtype=torch.float32, device="cuda"); std = torch.tensor(ck["std"], dtype=torch.float32, device="cuda")
    lm, ls = ck["lsw"]; pos = ck["pos"].float().cuda()
    mu = np.empty_like(X); sig = np.empty_like(X)
    for i in range(0, len(X), batch):
        xb = torch.as_tensor(X[i:i + batch], device="cuda"); B, T, H, W, _ = xb.shape; NP = H * W
        u = xb[..., 0]; sw = (u.sum(dim=(1, 2, 3)) / (u != 0).float().sum(dim=(1, 2, 3)).clamp_min(1.0)).clamp_min(1e-3).view(-1, 1, 1, 1, 1)
        tk = ((xb / sw - mean) / std).permute(0, 2, 3, 1, 4).reshape(B, NP, 2 * T)
        tk = torch.cat([tk, ((torch.log(sw.view(-1)) - lm) / ls)[:, None, None].expand(-1, NP, 1)], -1)
        tk = torch.cat([tk, torch.zeros(B, NP, 2, device=tk.device, dtype=tk.dtype)], -1)   # domain flag, frame skip: 0 for real native-rate data
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = net(pos.expand(B, -1, -1), tk).float()
        if not torch.isfinite(z).all():   # reduced-precision overflow: redo in fp32
            z = net(pos.expand(B, -1, -1), tk).float()
        scale = std * sw
        m = z[..., :2 * T].reshape(B, H, W, T, 2).permute(0, 3, 1, 2, 4) * scale + mean * sw
        s = torch.exp(0.5 * z[..., 2 * T:4 * T].clamp(-12, 6)).reshape(B, H, W, T, 2).permute(0, 3, 1, 2, 4) * scale
        unmeasured = (xb == 0).all(dim=4).all(dim=1)[:, None, :, :, None]
        mu[i:i + batch] = m.masked_fill(unmeasured, 0).cpu().numpy(); sig[i:i + batch] = s.cpu().numpy()
    assert np.isfinite(mu).all() and np.isfinite(sig).all()
    return mu, sig


def export(ckpts, data, out, fold=None):
    z = np.load(data); cond = z["cond"]; case_id = np.array(["%d_%d" % (int(c[0]), int(c[1])) for c in cond])
    idx = np.arange(len(cond))
    if fold is not None:
        held = [c for c in np.unique(cond, axis=0) if ROW_FOLD[int(round((c[0] - 3750) / 1275))] == fold]
        idx = np.concatenate([np.flatnonzero(np.all(cond == c, axis=1)) for c in held])
    X = z["X"][idx]
    outs = [predict(c, X) for c in ckpts]
    mus = [o[0] for o in outs]; mu = np.mean(mus, 0); sigma = np.mean([o[1] for o in outs], 0)
    m_t = mu.mean(1, keepdims=True); f = mu - m_t
    v_ens = (f ** 2).mean(1, keepdims=True); v_mem = np.mean([((q - q.mean(1, keepdims=True)) ** 2).mean(1, keepdims=True) for q in mus], 0)
    mu = m_t + f * (1 + VARFIX * (np.sqrt((v_mem + 1e-12) / (v_ens + 1e-12)) - 1))
    np.savez(out, idx=idx, mu=mu, sigma=sigma, cond=cond[idx], case_id=case_id[idx])
    print(json.dumps({"export": out, "windows": int(len(idx)), "members": [str(c) for c in ckpts]}), flush=True)


os.makedirs(OUT, exist_ok=True)
jobs = [([f"{RUNS}/all_s{s}/{CKPT}" for s in SEEDS], f"{DATA}/val_windows.npz", f"{OUT}/dev.npz", None)]
jobs += [([f"{RUNS}/fold{k}_s{s}/{CKPT}" for s in SEEDS], f"{DATA}/train_windows.npz", f"{OUT}/fold{k}.npz", k) for k in range(NFOLD)]
for ckpts, data, out, fold in jobs:
    if os.path.exists(out):
        print("skip", out, "(exists)", flush=True); continue
    missing = [c for c in ckpts if not os.path.exists(c)]
    if missing:
        raise FileNotFoundError(f"{out}: missing checkpoints {missing}")
    export(ckpts, data, out, fold)
print("BUNDLES_DONE", flush=True)
