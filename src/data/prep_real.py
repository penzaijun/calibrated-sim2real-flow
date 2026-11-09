"""Real (PIV) data at the scoring resolution (32x64, every 2nd grid point of the release), channels [u, v]:

  train_windows.npz  20->20 windows, stride 10, of every train_real case (X, Y (N, 20, 32, 64, 2) float32,
                     cond (N, 2) = measured (Re, AoA), x, y (32, 64) grid coordinates)
  val_windows.npz    20->20 windows, stride 40, of six fixed cases; only for the validation lines of the training log
                     (those cases are in train_windows.npz as well)
  real_traces.npz    the full trace of every train_real case, c000..cNNN (T, 32, 64, 2) float32, cond (C, 2): the source
                     of the frame-skip windows

usage: python prep_real.py <raw release dir> <output dir>"""
import glob
import os
import sys

import h5py
import numpy as np

RAW, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
EXCLUDED = {"7575_0"}   # organizer notice: real 7575_0 duplicates 6300_0
VAL_CASES = {"5025_10", "8850_0", "12675_20", "17775_5", "21600_15", "25425_10"}
T_IN = T_OUT = 20


def case_files():
    for p in sorted(glob.glob(f"{RAW}/train_real/**/*.h5", recursive=True)):
        c = os.path.basename(p)[:-3]
        if c not in EXCLUDED:
            yield c, p


def load(path):
    with h5py.File(path) as h:
        u = h["u"][:, ::2, ::2].astype(np.float32); v = h["v"][:, ::2, ::2].astype(np.float32)
        x = h["x"][::2, ::2]; y = h["y"][::2, ::2]; re = int(h["re"][()]); aoa = int(h["aoa"][()])
    return np.stack([u, v], -1), x, y, re, aoa


def windows(name, stride, keep):
    X = []; Y = []; cond = []
    for c, p in case_files():
        if not keep(c):
            continue
        f, x, y, re, aoa = load(p)
        starts = range(0, f.shape[0] - T_IN - T_OUT + 1, stride)
        X.append(np.stack([f[s:s + T_IN] for s in starts])); Y.append(np.stack([f[s + T_IN:s + T_IN + T_OUT] for s in starts]))
        cond += [(re, aoa)] * len(starts)
    X = np.concatenate(X); Y = np.concatenate(Y)
    np.savez(f"{OUT}/{name}.npz", X=X, Y=Y, cond=np.array(cond, np.float32), x=x.astype(np.float32), y=y.astype(np.float32))
    print(name, X.shape, "GB %.1f" % (X.nbytes * 2 / 1e9), flush=True)


def traces():
    T = {}; cond = []
    for c, p in case_files():
        f, _, _, re, aoa = load(p)
        cond.append((re, aoa)); T["c%03d" % (len(cond) - 1)] = f
    np.savez(f"{OUT}/real_traces.npz", cond=np.array(cond, np.float32), **T)
    print("real_traces", len(cond), "cases, frames", sorted({t.shape[0] for t in T.values()}), flush=True)


windows("train_windows", 10, lambda c: True)
windows("val_windows", 40, lambda c: c in VAL_CASES)
traces()
print("PREP_REAL_DONE")
