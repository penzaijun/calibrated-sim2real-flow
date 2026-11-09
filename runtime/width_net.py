"""Inference side of the interval-width CNN (src/train_width.py): the same 20 features, computed on the GPU and assembled
directly as an NHWC bf16 image (channel index = c * 20 + feature), the members' convolutions merged into one pass
(layer 1 concatenated along the output channels, deeper layers as grouped convolutions), the members' final 1x1 layers
as one batched GEMM, and their raw (eta, offset) maps averaged in fp32."""
import torch
import torch.nn.functional as Fn

SG = 0.0563870259   # the scorer's SIGMA_GLOBAL
N_FEAT = 20
_cache = {}


def _pool(t, k):   # (B, T, H, W, C) -> spatial k x k mean, edge-aware
    Bn, T, H, W, C = t.shape
    u = t.permute(0, 1, 4, 2, 3).reshape(-1, 1, H, W)
    return Fn.avg_pool2d(u, k, stride=1, padding=k // 2, count_include_pad=False).reshape(Bn, T, C, H, W).permute(0, 1, 3, 4, 2)


def _const(name, device, make):
    key = (name, str(device))
    if key not in _cache:
        _cache[key] = make()
    return _cache[key]


def features(X, mu, sigma):
    """X, mu, sigma (B, 20, H, W, 2) -> (B * 20, 2 * 20, H, W) bf16 with channels_last strides"""
    sg = sigma.clamp(1e-5, .5)
    r = lambda a: torch.log((a + 1e-5) / sg).clamp(-4, 4)
    sd = X.std(dim=1, unbiased=False, keepdim=True); diff = X.diff(dim=1).square().mean(1, keepdim=True).sqrt()
    delta = (mu - X[:, -1:]).abs(); valid = (X != 0).float().mean(1, keepdim=True)
    lead = _const("lead", X.device, lambda: (torch.arange(20, device=X.device, dtype=torch.float32) / 19).view(1, 20, 1, 1, 1)).expand_as(sg)
    chan = _const("chan", X.device, lambda: torch.arange(2, device=X.device, dtype=torch.float32).view(1, 1, 1, 1, 2)).expand_as(sg)
    ones = _const("one", X.device, lambda: torch.ones((1, 1, 1, 1, 1), device=X.device, dtype=torch.float32)).expand_as(sg)
    lsig = torch.log(sg / SG).clamp(-8, 3)
    sig3 = _pool(sg, 3); xl = X[:, -1:]; xsp = (_pool(xl * xl, 3) - _pool(xl, 3) ** 2).clamp_min(0).sqrt()
    mutstd = mu.std(dim=1, unbiased=False, keepdim=True); mulevel = torch.log((mu.abs() + 1e-4) / SG).clamp(-8, 3)
    vmask = (X != 0).any(dim=4, keepdim=True).float()
    uref = (X[..., :1].abs() * vmask).sum(dim=(1, 2, 3, 4), keepdim=True) / vmask.sum(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1)
    luref = torch.log(uref / SG).clamp(-3, 4).expand_as(sg)
    zero5 = (1 - _pool(vmask[:, -1:], 5)).expand_as(sg); sigt = sg.mean(dim=1, keepdim=True)
    r_sd = r(sd)
    lags = [(X[:, l:] - X[:, :20 - l]).abs().mean(dim=1) for l in range(1, 20)]   # structure function at the element's own lead
    f = [ones, r_sd, r(diff), r(delta), valid, lead, chan, lsig, r(sig3), r(xsp), r(mutstd), mulevel, luref, zero5, r(sigt),
         lead * r_sd, lead * lsig, (mu / (uref + 1e-4)).clamp(-5, 5), (xl / (uref + 1e-4)).clamp(-5, 5), r(torch.stack(lags + [lags[-1]], 1))]
    Bn, T, H, W, C = sg.shape
    out = torch.empty((Bn, T, H, W, C, N_FEAT), dtype=torch.bfloat16, device=sg.device)
    torch.stack([t.expand_as(sg) for t in f], -1, out=out)
    return out.view(Bn * T, H, W, C * N_FEAT).permute(0, 3, 1, 2)


def _weights(model, device):
    key = ("w", str(device))
    if key not in _cache:
        members = [[(torch.tensor(w, dtype=torch.float32, device=device), torch.tensor(b, dtype=torch.float32, device=device)) for w, b in layers]
                   for layers in model["members"]]
        K = len(members); merged = []
        for i in range(len(members[0]) - 1):   # conv layers: (K * cout, cin, 3, 3), bf16, channels_last
            w = torch.cat([m[i][0] for m in members], 0); b = torch.cat([m[i][1] for m in members], 0)
            merged.append((w.to(torch.bfloat16).contiguous(memory_format=torch.channels_last), b.to(torch.bfloat16).contiguous()))
        w_last = torch.stack([m[-1][0][:, :, 0, 0].t() for m in members], 0).to(torch.bfloat16).contiguous()   # (K, cin, cout)
        b_last = torch.stack([m[-1][1] for m in members], 0).unsqueeze(1).to(torch.bfloat16).contiguous()     # (K, 1, cout)
        # without the fused bias + SiLU kernel: layer 1's bias folded into the centre tap of the all-ones feature (input
        # channel 0); the centre tap of a 3x3 convolution never reads padding, so it adds exactly b at every pixel
        w0 = torch.cat([m[0][0] for m in members], 0).clone(); w0[:, 0, 1, 1] += torch.cat([m[0][1] for m in members], 0)
        folded = w0.to(torch.bfloat16).contiguous(memory_format=torch.channels_last)
        _cache[key] = (K, merged, w_last, b_last, folded)
    return _cache[key]


def _cnn(model, x0, fused):
    """all members in one pass -> averaged raw maps (B * 20, H, W, 4) float32"""
    K, merged, w_last, b_last, folded = _weights(model, x0.device)
    dil = int(model["dilation"]); o = x0.to(dtype=torch.bfloat16, memory_format=torch.channels_last)
    if fused is not None:   # convolutions without bias, then bias + SiLU in one in-place kernel (bitwise equal to add_ then silu)
        for i, (w, b) in enumerate(merged):
            d_ = dil if i > 0 else 1
            o = fused.bias_silu_(Fn.conv2d(o, w, None, padding=(w.shape[-1] // 2) * d_, dilation=d_, groups=1 if i == 0 else K), b)
    else:
        o = Fn.silu(Fn.conv2d(o, folded, None, padding=folded.shape[-1] // 2))
        for w, b in merged[1:]:
            o = Fn.silu(Fn.conv2d(o, w, b, padding=(w.shape[-1] // 2) * dil, dilation=dil, groups=K))
    Bt, Ch, H, W = o.shape
    a = o.permute(0, 2, 3, 1).reshape(Bt * H * W, K, Ch // K).transpose(0, 1)   # (K, pixels, cin) view of the NHWC activations
    y = torch.baddbmm(b_last, a, w_last).float()
    y = y.mean(0) if K > 1 else y[0]
    return y.view(Bt, H, W, -1)


def widths(x, mu, sigma, model, lookup, table, c_sigma, fused=None):
    """x (B, 20, H, W, 2) physical input, mu / sigma (B, 20, H, W, 2) -> half-width h and centre offset d (same shape).
    lookup(s, table): the runtime's h*(c_sigma * s) table, so h = h*(sigma * exp(eta)) is lookup(sigma * exp(eta) / c_sigma)."""
    unmeasured = (x == 0).all(dim=4).all(dim=1)[:, None, :, :, None]
    mu = mu.masked_fill(unmeasured, 0)   # as in the returned prediction
    Bn, T, H, Wd, C = sigma.shape
    o = _cnn(model, features(x, mu, sigma), fused).view(Bn, T, H, Wd, 2, 2)   # (..., channel, [eta, offset])
    sg = sigma.clamp(1e-5, .5)
    h = lookup(sg * torch.exp(o[..., 0].clamp(-2.5, 2.5)) / c_sigma, table)
    return h, sg * torch.tanh(o[..., 1])
