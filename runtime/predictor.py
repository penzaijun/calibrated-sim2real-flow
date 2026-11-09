"""Inference runtime of the trained package: predict(input_array) -> {"prediction", "lower", "upper"}.

Input (N, 20, 32, 64, 3) [u, v, p] in physical units; outputs of the same shape (p is returned as 0, its bounds are not
scored). Pixels that are zero in every input frame (airfoil body, outside the PIV field of view) are predicted as 0.

  mu     mean of the members' predictions, then each cell's 20-frame variance restored to the members' mean variance
  sigma  mean of the members' sigma heads
  band   [mu + d - h, mu + d + h] from the width CNN (width_net.py)

Speed (the competition timed predict at N = 5): the members run in bf16 without autocast (bf16 weights, Fourier
encoding of the fixed grid precomputed, one shared token tensor, the embedding GEMM padded to an aligned width, fused
QKV projection); residual add + LayerNorm run fused in fp32 with the fp32 LayerNorm weights when the kernels of
fused_ln.py load (the autocast numerics), otherwise as torch bf16 ops. The whole N = 5 stage including output assembly
is one CUDA graph, captured at import; the raw input array is uploaded as is and sliced / converted on the GPU, and the
outputs come back with a finite flag in one device-to-host copy. Untouched fp32 copies of the members are kept for the
rare non-finite result. Other batch sizes run the same stage eagerly in chunks of 32. Dependencies: torch + numpy."""
from __future__ import annotations

import copy
import json
import math
import os
import sys
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: E402

torch.set_num_threads(min(torch.get_num_threads(), 4))
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

import model as M  # noqa: E402
import width_net  # noqa: E402

CONFIG = json.load(open(os.path.join(_HERE, "package.json")))
WIDTH_MODEL = json.load(open(os.path.join(_HERE, CONFIG["width_model"])))
T_IN = T_OUT = 20
H, W = 32, 64
_BATCH = 32
SIGMA_GLOBAL = 0.0563870259   # the scorer's frozen sigma (scoring.py)
_S = {"members": None}

# h*(sigma) table on a log-uniform sigma grid, so a GPU linear interpolation in log(sigma) needs no search
_TAB_S0, _TAB_S1, _TAB_N = 1e-5, 0.5, 1024


def _hstar_table(c_sigma, n_h=4000):
    """h*(s) = argmax_h erf(h / (sqrt2 c_sigma s)) * exp(-2h / SIGMA_GLOBAL) on the log-uniform grid of s"""
    sig = np.geomspace(_TAB_S0, _TAB_S1, _TAB_N); ss = c_sigma * sig; hs = np.linspace(1e-6, 0.6, n_h)
    erf = np.vectorize(math.erf)
    val = erf(hs[None, :] / (math.sqrt(2.0) * ss[:, None])) * np.exp(-2.0 * hs[None, :] / SIGMA_GLOBAL)
    return hs[np.argmax(val, axis=1)].astype(np.float32)


def _lookup(sig, table):
    """linear interpolation of the table in log(sigma), clamped to the grid; a non-finite sigma gets the widest band"""
    dlog = (math.log(_TAB_S1) - math.log(_TAB_S0)) / (_TAB_N - 1)
    sig = torch.nan_to_num(sig, nan=_TAB_S1, posinf=_TAB_S1, neginf=_TAB_S0)
    t = (torch.log(sig.clamp(_TAB_S0, _TAB_S1)) - math.log(_TAB_S0)) / dlog
    i0 = t.floor().long().clamp_(0, _TAB_N - 2); f = (t - i0.to(t.dtype)).clamp(0.0, 1.0)
    return table[i0] * (1.0 - f) + table[i0 + 1] * f


def _build():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    cks = [torch.load(os.path.join(_HERE, n), map_location="cpu", weights_only=False) for n in CONFIG["members"]]
    for c in cks[1:]:   # the members are trained on the same windows, so they share the input statistics and one token tensor
        assert c["mean"] == cks[0]["mean"] and c["std"] == cks[0]["std"] and c["lsw"] == cks[0]["lsw"]
    nets = []
    for c in cks:
        net = M.build(CONFIG["model"], c["c_in"], c["c_out"]); net.load_state_dict(c["model"]); nets.append(net.to(device).eval())
    ck = cks[0]
    _S.update(device=device, pos=ck["pos"].float().to(device),
              mean=torch.tensor(ck["mean"], dtype=torch.float32, device=device), std=torch.tensor(ck["std"], dtype=torch.float32, device=device),
              lsw=tuple(ck["lsw"]), varfix=float(CONFIG["varfix"]), c_sigma=float(CONFIG["c_sigma"]),
              table=torch.from_numpy(_hstar_table(float(CONFIG["c_sigma"]))).to(device), fused=None)
    _S["fp32"] = nets
    pure = device == "cuda" and bool(torch.cuda.is_bf16_supported())
    _S["pure"] = pure
    if pure:
        members = [copy.deepcopy(n).to(torch.bfloat16) for n in nets]
        with torch.no_grad():
            _S["pe"] = nets[0].pos_enc(_S["pos"]).to(torch.bfloat16).contiguous()   # (1, NP, 6): the Fourier encoding in fp32, rounded once
            _S["qkv"] = {id(b): torch.cat([b.attn.q.weight, b.attn.k.weight, b.attn.v.weight], 0) for n in members for b in n.blocks}
            _S["embed_pad"] = {id(n): _embed_pad(n, _S["pe"], device) for n in members}
        try:
            import fused_ln
            if fused_ln.available():
                lnw = {}
                for n, n32 in zip(members, nets):
                    for m_, m32 in zip(n.modules(), n32.modules()):
                        if isinstance(m_, torch.nn.LayerNorm):
                            lnw[id(m_)] = (m32.weight.detach().float().contiguous(), m32.bias.detach().float().contiguous(), float(m32.eps))
                _S.update(fused=fused_ln, lnw=lnw)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write("fused kernels disabled: %s\n" % (str(e)[:120],))
        _S["members"] = members
    else:
        _S["members"] = nets


def _embed_pad(net, pe, device):
    """(zero-padded embedding weight, zero token block): the GEMM width 6 + c_in becomes a multiple of 8; the zeros add exactly 0"""
    lin = net.embed[0]; k = lin.in_features; kp = (k + 7) // 8 * 8
    if kp == k:
        return None
    w = torch.zeros((lin.out_features, kp), dtype=lin.weight.dtype, device=device); w[:, :k] = lin.weight.detach()
    return w.contiguous(), torch.zeros((1, pe.shape[1], kp - k), dtype=pe.dtype, device=device)


def _attn(blk, xn):   # self-attention with the fused QKV projection
    b, n, d = xn.shape; h = blk.attn.heads
    q, k, v = torch.nn.functional.linear(xn, _S["qkv"][id(blk)]).chunk(3, dim=-1)
    q = q.reshape(b, n, h, d // h).transpose(1, 2); k = k.reshape(b, n, h, d // h).transpose(1, 2); v = v.reshape(b, n, h, d // h).transpose(1, 2)
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    return blk.attn.out(o.transpose(1, 2).contiguous().view(b, n, d))


def _net_bf16(net, tk):
    """PointTokenTransformer.forward in pure bf16 on precomputed position features"""
    pe = _S["pe"]; B = tk.shape[0]; pad = _S["embed_pad"][id(net)]
    if pad is not None:
        w_pad, z_pad = pad
        t = net.embed[1](torch.nn.functional.linear(torch.cat([pe.expand(B, -1, -1), tk, z_pad.expand(B, -1, -1)], -1), w_pad, net.embed[0].bias))
    else:
        t = net.embed(torch.cat([pe.expand(B, -1, -1), tk], -1))
    fl = _S["fused"]
    x = net.proj_in(t)
    if fl is None:
        for blk in net.blocks:
            xn = blk.norm1(x)
            x = x + _attn(blk, xn)
            x = x + blk.mlp(blk.norm2(x))
        return net.head(net.proj_out(x))
    # residual adds fused into the LayerNorm kernels; per block: xn = norm1(x); a = attn(xn); x, h = x + a, mlp-LN(norm2(x + a));
    # f = MLP(h); the next block's norm1 is fused with x + f
    lnw = _S["lnw"]; blocks = net.blocks
    _, xn = fl.ln_single(x, None, *lnw[id(blocks[0].norm1)])
    for i, blk in enumerate(blocks):
        a = _attn(blk, xn)
        x, h = fl.ln_double(x, a, *lnw[id(blk.norm2)], *lnw[id(blk.mlp[0])])
        f = blk.mlp[3](blk.mlp[2](blk.mlp[1](h)))
        if i + 1 < len(blocks):
            x, xn = fl.ln_single(x, f, *lnw[id(blocks[i + 1].norm1)])
        else:
            x = x + f
    _, hn = fl.ln_single(net.proj_out(x), None, *lnw[id(net.head[0])])
    return net.head[3](net.head[2](net.head[1](hn)))


_CAPTURING = False


def _forward(xb):
    """xb (B, 20, H, W, 2) physical on the device -> mu, sigma (B, 20, H, W, 2) physical"""
    B, T = xb.shape[:2]; NP = H * W
    mean, std = _S["mean"], _S["std"]; lm, ls = _S["lsw"]
    u = xb[..., 0]; sw = (u.sum(dim=(1, 2, 3)) / (u != 0).float().sum(dim=(1, 2, 3)).clamp_min(1.0)).clamp_min(1e-3).view(-1, 1, 1, 1, 1)
    tk = ((xb / sw - mean) / std).permute(0, 2, 3, 1, 4).reshape(B, NP, 2 * T)
    tk = torch.cat([tk, ((torch.log(sw.view(-1)) - lm) / ls)[:, None, None].expand(-1, NP, 1)], -1)
    tk = torch.cat([tk, torch.zeros(B, NP, 2, device=tk.device, dtype=tk.dtype)], -1)   # domain flag, frame skip: 0 for real native-rate data
    tkb = tk.to(torch.bfloat16) if _S["pure"] else None
    zs = []
    for i, net in enumerate(_S["members"]):
        z = None
        if _S["pure"]:
            z = _net_bf16(net, tkb).float()
            if not _CAPTURING and not torch.isfinite(z).all():
                z = None
        if z is None:   # fp32 fallback (also the CPU path)
            z = _S["fp32"][i](_S["pos"].expand(B, -1, -1), tk).float()
        zs.append(z)
    scale = std * sw
    field = lambda t: t.reshape(B, H, W, T_OUT, 2).permute(0, 3, 1, 2, 4) * scale
    mus = [field(z[..., :2 * T_OUT]) + mean * sw for z in zs]
    mu = sum(mus) / len(mus)
    if len(mus) > 1 and _S["varfix"] > 0:   # averaging damps the fluctuations the members disagree on: restore each cell's 20-frame variance
        m_t = mu.mean(dim=1, keepdim=True); f = mu - m_t
        v_ens = (f * f).mean(dim=1, keepdim=True); v_mem = sum(((m_ - m_.mean(dim=1, keepdim=True)) ** 2).mean(dim=1, keepdim=True) for m_ in mus) / len(mus)
        mu = m_t + f * (1.0 + _S["varfix"] * (torch.sqrt((v_mem + 1e-12) / (v_ens + 1e-12)) - 1.0))
    sig = sum(field(torch.exp(0.5 * z[..., 2 * T_OUT:4 * T_OUT].clamp(-12, 6))) for z in zs) / len(zs)
    return mu, sig


def _stage(xb):
    """-> mu, lower half-width, upper half-width (each (B, 20, H, W, 2))"""
    mu, sig = _forward(xb)
    h, d = width_net.widths(xb, mu, sig, WIDTH_MODEL, _lookup, _S["table"], _S["c_sigma"], _S["fused"])
    return mu, h - d, h + d


def _stage_packed(xb):
    """the graph body: [prediction, lower, upper] (3, N, 20, H, W, 3) written into a persistent buffer whose p channel stays 0,
    followed by one finite flag"""
    mu, lo, hi = _stage(xb)
    n = xb.shape[0]
    packed = _S["out"][n]
    body = packed[:-1].view(3, n, T_OUT, H, W, 3)
    unmeasured = (xb == 0).all(dim=4).all(dim=1)[:, None, :, :, None]
    pred = torch.where(unmeasured, torch.zeros((), device=xb.device), mu)
    body[0, ..., :2] = pred
    body[1, ..., :2] = pred - lo
    body[2, ..., :2] = pred + torch.maximum(hi, -lo)
    packed[-1:].copy_(torch.isfinite(mu.sum() + lo.sum() + hi.sum()).float().reshape(1))
    return packed


_GRAPHS = {}   # batch size -> (static input, graph, packed output)
_RAW = {}      # (batch size, numpy dtype) -> device buffer for the raw (N, 20, H, W, 3) input


def _capture(sizes=(5,)):
    global _CAPTURING
    if _S["device"] != "cuda" or os.environ.get("S2R_NO_GRAPH"):
        return
    for n in sizes:
        try:
            static = torch.zeros((n, T_IN, H, W, 2), dtype=torch.float32, device="cuda")
            _S.setdefault("out", {})[n] = torch.zeros(3 * n * T_OUT * H * W * 3 + 1, dtype=torch.float32, device="cuda")
            _CAPTURING = True
            st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st), torch.inference_mode():
                for _ in range(2):
                    _stage_packed(static)
            torch.cuda.current_stream().wait_stream(st)
            g = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(g):
                out = _stage_packed(static)
            _GRAPHS[n] = (static, g, out)
            for dt in (np.float32, np.float64):
                _RAW[(n, np.dtype(dt).str)] = torch.zeros((n, T_IN, H, W, 3), dtype=torch.float32 if dt is np.float32 else torch.float64, device="cuda")
        except Exception as e:  # noqa: BLE001
            sys.stderr.write("graph capture failed for N=%d: %s\n" % (n, str(e)[:120])); _GRAPHS.pop(n, None)
        finally:
            _CAPTURING = False
    torch.cuda.synchronize()


def _upload(x, static, n):
    """raw input -> the graph's static (N, 20, H, W, 2) float32 buffer: one copy of the unmodified array, sliced and rounded to
    float32 on the GPU (same round-to-nearest as numpy)"""
    if isinstance(x, np.ndarray) and x.flags.c_contiguous:
        buf = _RAW.get((n, x.dtype.str))
        if buf is not None:
            buf.copy_(torch.from_numpy(x)); static.copy_(buf[..., :2])
            return
    static.copy_(torch.from_numpy(np.ascontiguousarray(np.asarray(x, dtype=np.float32)[..., :2])))


def predict(input_array, metadata=None):
    if _S["members"] is None:
        _build()
    device = _S["device"]
    shp = tuple(getattr(input_array, "shape", ()))
    entry = _GRAPHS.get(shp[0]) if len(shp) == 5 and shp[1:] == (T_IN, H, W, 3) else None
    if entry is not None:
        n = shp[0]; static, graph, packed = entry
        with torch.inference_mode():
            _upload(input_array, static, n)
            graph.replay()
            host = torch.empty(packed.shape, dtype=packed.dtype, device="cpu", pin_memory=True)
            host.copy_(packed, non_blocking=True)
            torch.cuda.current_stream().synchronize()
            host = host.numpy()
        if bool(host[-1]):
            out = host[:-1].reshape(3, n, T_OUT, H, W, 3)
            return {"prediction": out[0], "lower": out[1], "upper": out[2]}
        # a non-finite result: the eager path below (with its fp32 fallback)
    x = np.asarray(input_array, dtype=np.float32)
    N, T, Hh, Ww, C = x.shape
    with torch.inference_mode():
        xg = torch.from_numpy(np.ascontiguousarray(x[..., :2])).to(device, non_blocking=True)
        unmeasured = (xg == 0).all(dim=4).all(dim=1)[:, None, :, :, None]
        prediction = torch.zeros((N, T_OUT, Hh, Ww, C), dtype=torch.float32, device=device)
        lo = torch.zeros_like(prediction); hi = torch.zeros_like(prediction)   # lower = prediction - lo, upper = prediction + hi
        for i in range(0, N, _BATCH):
            mu, lo_, hi_ = _stage(xg[i:i + _BATCH])
            prediction[i:i + _BATCH, ..., :2] = mu; lo[i:i + _BATCH, ..., :2] = lo_; hi[i:i + _BATCH, ..., :2] = hi_
        prediction = torch.where(unmeasured, torch.zeros((), device=device), prediction)
        # the scorer zeroes every subscore on a non-finite output: a bad element costs one element, not the submission
        prediction = torch.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)
        lo = torch.nan_to_num(lo, nan=0.1, posinf=0.1, neginf=0.1); hi = torch.nan_to_num(hi, nan=0.1, posinf=0.1, neginf=0.1)
        hi = torch.maximum(hi, -lo)
        out = torch.stack([prediction, prediction - lo, prediction + hi], 0).cpu().numpy()
    if device == "cuda":
        torch.cuda.synchronize()
    return {"prediction": out[0], "lower": out[1], "upper": out[2]}


def _warmup():
    """build, run predict eagerly, capture the N = 5 graph, run it with both input dtypes, then keep the GPU busy for a
    moment so the first timed call does not run on a GPU that is still clocking up"""
    _build()
    rng = np.random.default_rng(0)
    for n in (5, 5):
        x = (rng.standard_normal((n, T_IN, H, W, 3)) * 0.05 + np.array([0.15, 0.0, 0.0])).astype(np.float32)
        x[:, :, :3, :5, :] = 0.0   # a few never-measured cells
        predict(x)
    _capture()
    if 5 in _GRAPHS:   # pinned host blocks of the output size into torch's cache
        blocks = [torch.empty(_GRAPHS[5][2].numel(), dtype=torch.float32, device="cpu", pin_memory=True) for _ in range(3)]
        del blocks
    for n, dt in ((5, np.float32), (5, np.float64), (5, np.float32)):
        x = (rng.standard_normal((n, T_IN, H, W, 3)) * 0.05 + np.array([0.15, 0.0, 0.0])).astype(dt); predict(x)
    if 5 in _GRAPHS:
        with torch.inference_mode():
            for _ in range(50):
                _GRAPHS[5][1].replay()
    if _S["device"] == "cuda":
        torch.cuda.synchronize()


try:
    _warmup()
except Exception as exc:  # never let a warm-up problem break the import; predict() builds lazily
    sys.stderr.write("warm-up failed: %s: %s\n" % (type(exc).__name__, exc))
    _GRAPHS.clear()
    _S["members"] = None
