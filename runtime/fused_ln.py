"""Fused residual-add + LayerNorm kernels for the transformer blocks and a fused bias + SiLU for the width CNN, launched
through the CUDA driver API.

The kernels (fused_ln.cu) ship precompiled: fused_ln_sm80.cubin for sm_80 (A100 / A800), fused_ln.ptx for other GPUs
(JIT-compiled by the driver). They are loaded with ctypes from the driver library -- no compiler is needed at run time.
Launches go to torch's current stream, so CUDA-graph capture records them like any other kernel. Everything is optional:
if the driver cannot load the image or a launch fails, the callers fall back to plain torch ops (see available()).

Numerics (those of bf16 autocast): residual sums are rounded to bf16, LayerNorm is computed in fp32 with fp32 gamma / beta,
and only the final LayerNorm output is rounded to bf16."""
import ctypes
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_state = {"ok": None}


def _driver():
    if sys.platform.startswith("linux"):
        return ctypes.CDLL("libcuda.so.1")
    if sys.platform == "win32":
        return ctypes.CDLL("nvcuda.dll")
    return None


def _load():
    if os.environ.get("S2R_FUSED_KERNELS", "1") != "1" or not torch.cuda.is_available():
        return False
    cu = _driver()
    if cu is None:
        return False
    torch.zeros(1, device="cuda"); torch.cuda.synchronize()        # primary context current on this thread
    ctx = ctypes.c_void_p()
    if cu.cuCtxGetCurrent(ctypes.byref(ctx)) != 0 or not ctx.value:
        return False
    images = (["fused_ln_sm80.cubin"] if torch.cuda.get_device_capability() == (8, 0) else []) + ["fused_ln.ptx"]
    mod = ctypes.c_void_p()
    for name in images:
        path = os.path.join(_HERE, name)
        if not os.path.exists(path):
            continue
        data = open(path, "rb").read()
        if name.endswith(".ptx"):
            data += b"\0"
        buf = ctypes.create_string_buffer(data, len(data))
        if cu.cuModuleLoadData(ctypes.byref(mod), buf) == 0 and mod.value:
            _state["image"] = buf
            break
        mod = ctypes.c_void_p()
    if not mod.value:
        return False
    f_single, f_double, f_bs = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    if (cu.cuModuleGetFunction(ctypes.byref(f_single), mod, b"fast_ln_single") != 0 or cu.cuModuleGetFunction(ctypes.byref(f_double), mod, b"fast_ln_double") != 0
            or cu.cuModuleGetFunction(ctypes.byref(f_bs), mod, b"fast_bias_silu") != 0):
        return False
    cu.cuLaunchKernel.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                                  ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    cu.cuLaunchKernel.restype = ctypes.c_int
    _state.update(cu=cu, mod=mod, single=f_single, double=f_double, bias_silu=f_bs)
    return True


def available():
    if _state["ok"] is None:
        try:
            _state["ok"] = bool(_load())
        except Exception as e:  # noqa: BLE001
            sys.stderr.write("fused kernels unavailable: %s\n" % (str(e)[:120],)); _state["ok"] = False
    return _state["ok"]


def _usable(*ts):
    for t in ts:
        if t is None:
            continue
        if not (t.is_cuda and t.is_contiguous() and t.data_ptr() % 16 == 0):
            return False
    return True


def _launch(fn, rows, args, threads=128, per_block=4):
    params = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    r = _state["cu"].cuLaunchKernel(fn, (rows + per_block - 1) // per_block, 1, 1, threads, 1, 1, 0, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), params, None)
    if r != 0:
        _state["ok"] = False
        sys.stderr.write("fused kernel launch failed (CUresult %d): torch fallback\n" % r)
    return r == 0


def _ref_single(x, add, g, b, eps):
    xo = x + add if add is not None else x
    return xo, torch.nn.functional.layer_norm(xo.float(), (x.shape[-1],), g, b, eps).to(x.dtype)


def _ref_double(x, add, g1, b1, e1, g2, b2, e2):
    xo = x + add
    h = torch.nn.functional.layer_norm(xo.float(), (x.shape[-1],), g1, b1, e1)
    return xo, torch.nn.functional.layer_norm(h, (x.shape[-1],), g2, b2, e2).to(x.dtype)


def _ok(x, add, *ws):
    return (_state["ok"] and x.dtype == torch.bfloat16 and x.shape[-1] == 256 and (add is None or (add.dtype == x.dtype and add.shape == x.shape))
            and all(w.dtype == torch.float32 and w.numel() == 256 for w in ws) and _usable(x, add, *ws))


def ln_single(x, add, g, b, eps):
    """-> (x + add rounded to bf16 (or x), bf16 LayerNorm of it)"""
    if _ok(x, add, g, b):
        rows = x.numel() // 256; y = torch.empty_like(x); xo = torch.empty_like(x) if add is not None else x
        args = [ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(add.data_ptr() if add is not None else None), ctypes.c_void_p(xo.data_ptr()),
                ctypes.c_void_p(y.data_ptr()), ctypes.c_void_p(g.data_ptr()), ctypes.c_void_p(b.data_ptr()), ctypes.c_int(rows), ctypes.c_float(eps)]
        if _launch(_state["single"], rows, args):
            return xo, y
    return _ref_single(x, add, g, b, eps)


def ln_double(x, add, g1, b1, e1, g2, b2, e2):
    """-> (x + add rounded to bf16, bf16 LN2(LN1(.)) with the intermediate in fp32)"""
    if _ok(x, add, g1, b1, g2, b2):
        rows = x.numel() // 256; y = torch.empty_like(x); xo = torch.empty_like(x)
        args = [ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(add.data_ptr()), ctypes.c_void_p(xo.data_ptr()), ctypes.c_void_p(y.data_ptr()),
                ctypes.c_void_p(g1.data_ptr()), ctypes.c_void_p(b1.data_ptr()), ctypes.c_void_p(g2.data_ptr()), ctypes.c_void_p(b2.data_ptr()),
                ctypes.c_int(rows), ctypes.c_float(e1), ctypes.c_float(e2)]
        if _launch(_state["double"], rows, args):
            return xo, y
    return _ref_double(x, add, g1, b1, e1, g2, b2, e2)


def bias_silu_(x, b):
    """In place on a channels-last bf16 conv output x (N, C, H, W) with bf16 bias b (C,): x = silu(x + b), each step rounded
    to bf16 like torch's add_ then silu. Returns x."""
    C = x.shape[1]
    if (_state["ok"] and x.dtype == torch.bfloat16 and b.dtype == torch.bfloat16 and x.dim() == 4 and C % 8 == 0 and b.numel() == C
            and x.is_contiguous(memory_format=torch.channels_last) and b.is_contiguous() and x.data_ptr() % 16 == 0 and b.data_ptr() % 16 == 0):
        n_vec = x.numel() // 8
        args = [ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(b.data_ptr()), ctypes.c_int(C), ctypes.c_longlong(n_vec)]
        if _launch(_state["bias_silu"], n_vec, args, threads=256, per_block=256):
            return x
    return torch.nn.functional.silu(x.add_(b.view(1, -1, 1, 1)))
