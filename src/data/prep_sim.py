"""Simulation traces in the real data's physical units, masked like the PIV data, stored whole (float16) so that training
can draw random windows from them.

  u_phys = u_sim * U_inf(Re),  U_inf = K * Re   (the simulation is non-dimensional, u / U_inf; same grid and dt = 0.02 s
                                                 as the real data)
  K      = median over the cases present in both domains of <u_real> / <u_sim> / Re (means over the measured pixels)
  mask   = the static zero mask of the real data (airfoil body and outside the PIV field of view: pixels that are zero in
           every frame), taken from the real case with the same AoA nearest in Re and applied to every simulated frame

Output sim_traces.npz: T (C, L, 32, 64, 2) float16, re (C,), aoa (C,), K, names.

usage: python prep_sim.py <raw release dir> <output dir>"""
import glob
import os
import sys

import h5py
import numpy as np

RAW, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
EXCLUDED = {"7575_0"}


def load(path, n=None):
    with h5py.File(path) as h:
        u = h["u"][:n, ::2, ::2].astype(np.float32); v = h["v"][:n, ::2, ::2].astype(np.float32)
        return np.stack([u, v], -1), int(h["re"][()]), int(h["aoa"][()])


def name_key(c):
    return int(c.split("_")[0]), int(c.split("_")[1])


real = {os.path.basename(p)[:-3]: p for p in sorted(glob.glob(f"{RAW}/train_real/**/*.h5", recursive=True)) if os.path.basename(p)[:-3] not in EXCLUDED}
sim = {os.path.basename(p)[:-3]: p for p in sorted(glob.glob(f"{RAW}/train_sim/**/*.h5", recursive=True))}
print(len(real), "real cases,", len(sim), "sim cases", flush=True)

masks = {}; ratios = []   # real static masks keyed by the nominal (Re, AoA) of the file name
for c, p in real.items():
    f, _, _ = load(p, 200); m = (f == 0).all(axis=(0, 3)); masks[name_key(c)] = m
    if c in sim:
        g, re_s, _ = load(sim[c], 200); meas = ~m
        ratios.append(f[:, meas, 0][f[:, meas, 0] != 0].mean() / g[:, meas, 0].mean() / re_s)
K = float(np.median(ratios))
print("K = U_inf / Re: median %.4e over %d paired cases" % (K, len(ratios)), flush=True)

T = []; res = []; aoas = []; names = []
for c in sorted(sim, key=name_key):
    g, re_s, aoa = load(sim[c]); cand = [k for k in masks if k[1] == aoa]
    if not cand:
        print("skip", c, "(no real case with AoA", aoa, "for the mask)"); continue
    m = masks[min(cand, key=lambda k: abs(k[0] - re_s))]
    g = g * (K * re_s); g[:, m] = 0.0
    T.append(g.astype(np.float16)); res.append(re_s); aoas.append(aoa); names.append(c)
L = min(len(t) for t in T); T = np.stack([t[:L] for t in T])
np.savez(f"{OUT}/sim_traces.npz", T=T, re=np.array(res, np.float32), aoa=np.array(aoas, np.float32), K=K, names=np.array(names))
print("sim_traces", T.shape, T.dtype, "frames/case", L, flush=True)
print("PREP_SIM_DONE")
