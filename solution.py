"""RealPDE Track 1 (Sim2Real) training package.

train(data_dir, output_dir, device) trains everything from the raw release:

  1. data      20->20 windows and full traces at the scoring resolution (src/data/prep_real.py, src/data/prep_sim.py)
  2. models    3 seeds of the point-token transformer on all real cases (+ simulation and frame-skip windows); the deployed
               weights are the EMA (0.999) after epoch 25 of a 70-epoch OneCycle schedule (src/train.py)
  3. folds     the same recipe on 5 Reynolds-row folds x 3 seeds (15 runs), for out-of-fold residuals
  4. bundles   out-of-fold mu / sigma of every training window from the fold ensembles (src/oof_bundles.py)
  5. width     the interval-width CNN, 3 members, fitted on those residuals (src/train_width.py)
  6. package   weights + width model + inference runtime written to <output_dir>/package, then smoke-tested

Every run, epoch and seed is fixed in advance; nothing is selected on a validation score. Each stage runs as a subprocess
and logs to <output_dir>/logs/. A model run that fails (e.g. the non-finite-step guard in train.py) is restarted once
with the fixed backup seed seed + 1000; any other failure, or a second failure of the same run, raises.

load_predictor(output_dir, device) imports the packaged runtime: predict(x) takes (N, 20, 32, 64, 3) float32 in physical
units and returns {"prediction", "lower", "upper"} of the same shape (p is returned as 0)."""
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
RUNTIME = HERE / "runtime"
SEEDS = (42, 43, 44)
RETRY_SEED = 1000
NFOLD = 5
SMOKE = os.environ.get("S2R_SMOKE", "") == "1"   # interface check only: 64 windows, 1 epoch, 40 width steps; the result is not a usable model
STOP = 1 if SMOKE else 25
CKPT = f"ema_ep{STOP}.pt"
RECIPE = ["--epochs", "70", "--stop_epoch", str(STOP), "--val_every", "1" if SMOKE else "5"] + (["--max_windows", "64"] if SMOKE else [])
WIDTH_STEPS = "40" if SMOKE else "6000"
RUNTIME_FILES = ("predictor.py", "width_net.py", "fused_ln.py", "fused_ln.cu", "fused_ln_sm80.cubin", "fused_ln.ptx")


def _log(logs, msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S UTC ", time.gmtime()) + msg
    print(line, flush=True)
    with open(logs / "timeline.log", "a") as f:
        f.write(line + "\n")


def _run(cmd, env, logfile, logs, what):
    """run one stage, streaming its output to the console and to logfile; raise on a non-zero exit"""
    t0 = time.time(); _log(logs, f"START {what}")
    with open(logfile, "a") as lf:
        lf.write("$ " + " ".join(str(c) for c in cmd) + "\n"); lf.flush()
        p = subprocess.Popen([str(c) for c in cmd], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            lf.write(line); sys.stdout.write(line)
        rc = p.wait()
    _log(logs, f"END {what} rc={rc} {time.time() - t0:.0f}s")
    if rc != 0:
        raise RuntimeError(f"{what} failed with exit code {rc}; see {logfile}")


def assemble_package(pkg, members, width_json):
    """the inference package: runtime + model code + member checkpoints + width model"""
    pkg = Path(pkg)
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir(parents=True)
    for n in RUNTIME_FILES:
        shutil.copy2(RUNTIME / n, pkg / n)
    shutil.copy2(SRC / "model.py", pkg / "model.py")
    names = []
    for i, c in enumerate(members):
        names.append(f"member{i + 1}.pt"); shutil.copy2(c, pkg / names[-1])
    shutil.copy2(width_json, pkg / "width_model.json")
    cfg = json.loads((HERE / "configs" / "model.json").read_text())["model"]
    (pkg / "package.json").write_text(json.dumps({"members": names, "model": cfg, "c_sigma": 1.25, "varfix": 1.0, "width_model": "width_model.json"}, indent=1))


def train(data_dir, output_dir, device):
    t_start = time.time()
    out = Path(output_dir); logs = out / "logs"; work = out / "_work"; data = work / "data"; runs = work / "runs"; pkg = out / "package"
    for d in (logs, data, runs):
        d.mkdir(parents=True, exist_ok=True)
    dev_index = str(device).split(":")[1] if ":" in str(device) else "0"
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.setdefault("OMP_NUM_THREADS", "16")
    py = sys.executable
    _log(logs, f"train(data_dir={data_dir}, output_dir={output_dir}, device={device}) smoke={SMOKE} python={py}")

    # 1. data (outputs of an earlier attempt are reused)
    if not all((data / f).exists() for f in ("train_windows.npz", "val_windows.npz", "real_traces.npz")):
        _run([py, SRC / "data" / "prep_real.py", data_dir, data], env, logs / "prep_real.log", logs, "prep real windows and traces")
    if not (data / "sim_traces.npz").exists():
        _run([py, SRC / "data" / "prep_sim.py", data_dir, data], env, logs / "prep_sim.log", logs, "prep sim traces")

    # 2-3. the 3 deployed models, then the 15 Reynolds-row fold models; a failed run restarts once with seed + RETRY_SEED
    jobs = [(f"all_s{s}", s, []) for s in SEEDS] + [(f"fold{k}_s{s}", s, ["--fold", str(k)]) for s in SEEDS for k in range(NFOLD)]
    for name, seed, extra in jobs:
        rd = runs / name
        if (rd / CKPT).exists():
            _log(logs, f"SKIP {name} ({CKPT} exists)"); continue
        for attempt, s in enumerate((seed, seed + RETRY_SEED)):
            if rd.exists():
                shutil.rmtree(rd)   # an interrupted or failed attempt restarts from scratch
            try:
                _run([py, SRC / "train.py", "--data", data, "--out", rd, "--device", dev_index, "--seed", str(s)] + RECIPE + extra, env,
                     logs / f"train_{name}.log", logs, f"train {name} (seed {s})")
                break
            except RuntimeError:
                if attempt == 1:
                    raise
                _log(logs, f"RETRY {name}: seed {seed} failed, restarting with backup seed {seed + RETRY_SEED}")
        assert (rd / CKPT).exists(), f"{name}: {CKPT} was not written"

    # 4. out-of-fold bundles, 5. width model
    bundles = work / "bundles"
    _run([py, SRC / "oof_bundles.py", data, runs, bundles, CKPT], env, logs / "bundles.log", logs, "out-of-fold bundles")
    width_json = work / "width_model.json"
    if not width_json.exists():
        _run([py, SRC / "train_width.py", data, bundles, width_json, WIDTH_STEPS, "3"], env, logs / "width.log", logs, "width CNN (3 members)")

    # 6. package, then a smoke test in a fresh process: load, predict on validation windows, check shapes / finiteness / accuracy
    assemble_package(pkg, [runs / f"all_s{s}" / CKPT for s in SEEDS], width_json)
    _log(logs, "package " + json.dumps({n: hashlib.sha256((pkg / n).read_bytes()).hexdigest()[:16] for n in sorted(os.listdir(pkg)) if (pkg / n).is_file()}))
    check = ("import sys, json, numpy as np; sys.path.insert(0, sys.argv[1]); import solution\n"
             "p = solution.load_predictor(sys.argv[2], 'cuda'); z = np.load(sys.argv[3]); X, Y = z['X'][:16], z['Y'][:16]\n"
             "x = np.concatenate([X, np.zeros_like(X[..., :1])], -1).astype(np.float32); res = {}\n"
             "for n in (1, 5, 16):\n"
             "    o = p.predict(x[:n]); pr = o['prediction']; lo = o['lower']; hi = o['upper']\n"
             "    assert pr.shape == lo.shape == hi.shape == x[:n].shape, (pr.shape, x[:n].shape)\n"
             "    assert np.isfinite(pr).all() and np.isfinite(lo).all() and np.isfinite(hi).all()\n"
             "    assert (lo[..., :2] <= hi[..., :2]).all()\n"
             "    res[n] = float(np.linalg.norm(pr[..., :2] - Y[:n]) / np.linalg.norm(Y[:n]))\n"
             "print('PACKAGE_CHECK_OK rel_l2_by_batch', json.dumps(res))\n")
    _run([py, "-c", check, HERE, out, data / "val_windows.npz"], env, logs / "package_check.log", logs, "package check (fresh process)")

    # keep the logs of every run, drop the bulky intermediates
    for name, _, _ in jobs:
        shutil.copy2(runs / name / "log.txt", logs / f"run_{name}.txt")
    shutil.copy2(width_json, logs / "width_model.json")
    if os.environ.get("S2R_KEEP_WORK", "") != "1":
        shutil.rmtree(work)
    _log(logs, f"TRAIN_DONE in {(time.time() - t_start) / 3600:.2f} h")


class _Predictor:
    def __init__(self, module):
        self._m = module

    def predict(self, input_array):
        return self._m.predict(input_array)


def load_predictor(output_dir, device):
    pkg = Path(output_dir) / "package"
    for k in [k for k in sys.modules if k in ("model", "width_net", "fused_ln")]:
        del sys.modules[k]   # the package's own copies must win over anything imported before
    spec = importlib.util.spec_from_file_location("s2r_package_predictor", pkg / "predictor.py")
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)   # builds the 3-member ensemble + width model on the GPU and warms up (CUDA graph for N = 5)
    return _Predictor(mod)
