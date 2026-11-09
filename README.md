# Calibrated Sim-to-Real Flow Forecasting: Point-Token Transformer with Learned Uncertainty Intervals

Pengyuan Zhu and Yueming Lyu (Codabench: `g2404426g`). Code for Track 1 (Sim-to-Real Transfer Learning) of the
[NeurIPS 2026 RealPDE Competition](https://realpdecompetition.github.io/): forecasting 20 frames of the PIV-measured
velocity field around a NACA 4418 airfoil from the previous 20, with a per-element uncertainty interval.

| | Score |
|---|---|
| Final results (private test set; organizers' 3 trainings x 5 evaluations, mean +- std) | **82.1252 +- 0.0512**, 10th place |
| Development leaderboard (submission 948754) | 81.901 (Rel-L2 94.75, TKE 79.87, MVPE 94.18, Time 91.77, SPS 44.19) |

## Method

**Model.** Every grid point of the 32x64 scoring grid is one token: its 20 input frames of (u, v), three per-window
scalars (below) and a Fourier encoding of its (x, y). A linear + GELU embedding, 8 pre-LayerNorm self-attention blocks
over all 2048 tokens (d_model 256, 8 heads, MLP x4) and an MLP head give 80 outputs per point: 20 frames x (u, v) of
the mean and of the log-variance. 6.8 M parameters. Pixels that are zero in every input frame (airfoil body, outside the
PIV field of view) are predicted as 0.

**Non-dimensional inputs.** Each window is divided by its own velocity scale (the masked mean of u over its input
frames, roughly the free-stream speed, which grows with Re); the model works in those units and the prediction is scaled
back. The standardised log of the scale is one input channel, so the map is amplitude-equivariant and an unseen Reynolds
number is not an extrapolation in input magnitude.

**Training data.** All 81 `train_real` cases (`7575_0` duplicates `6300_0` and is excluded), cut into 20->20 windows with
stride 10 (6602 windows). Every epoch adds, 1:1, random 40-frame windows of the 100 `train_sim` cases converted to
physical units (u_sim x K x Re, K fitted on the cases present in both domains) and masked with the static PIV mask of
the nearest real case, plus 0.5x as many Reynolds-similarity windows: every r-th frame of a real trace with velocities
x r (r in {2, 3}, r x Re <= 29000), which is the same flow at r x Re on the same grid. A domain flag and log2(r) are
input channels (both 0 at inference).

**Loss and optimisation.** rel-L2 of the normalised tokens + 0.15 x rel-L2 of the TKE map + 0.1 x Gaussian NLL of the
sigma head on the detached mean. AdamW (lr 2e-3, wd 1e-5, betas 0.9 / 0.99), gradient clip 10, OneCycle over 70 epochs
(30 % warm-up, cosine), batch 16, bf16 autocast, `torch.compile`. Every model is the EMA (0.999) of its weights after
epoch 25; nothing is selected on a validation score. Three seeds (42, 43, 44) are averaged at inference.

**Uncertainty intervals.** The score rewards coverage times exp(-width / sigma_g) per element, scaled by the window's
accuracy.

1. sigma: the mean of the three members' sigma heads.
2. Gaussian-optimal half-width: h*(s) = argmax_h erf(h / (sqrt(2) s)) exp(-2h / sigma_g), the expected per-element
   reward for an error ~ N(0, s^2), tabulated once.
3. Width CNN: 20 features per element from the input window, mu and sigma only (variability of the input, the size of
   the predicted change, sigma and its neighbourhood, lead time, local velocity, the input's structure function at the
   element's lead, ...). u and v are stacked as channels of one image per lead frame; three 3x3 convolutions (dilation 2
   after the first, 64 channels) output a log-multiplier eta on sigma and a centre offset. h = h*(sigma exp(eta)),
   interval = [mu + d - h, mu + d + h]. 97 k parameters, 3 members.
4. The CNN is fitted on out-of-fold residuals: the same recipe trained on 5 folds that each hold out whole Reynolds rows
   (like the test cells without real data), 3 seeds each, predicts the windows of its held-out cases. Objective: a
   smoothed version of the score (sigmoid coverage x exp(-2h / sigma_g) x the window's accuracy factor).

Most of the development-score gain came from the interval construction (sigma head, Gaussian-optimal width, width CNN),
then from the non-dimensional inputs with simulation / frame-skip mixing, then from inference speed.

**Inference speed** (`runtime/predictor.py`): members in bf16 without autocast, fused QKV projection, residual add +
LayerNorm fused into one kernel (`runtime/fused_ln.cu`, shipped as an sm_80 cubin and PTX, loaded through the CUDA
driver API, plain torch ops otherwise), the width CNN in NHWC bf16 with a fused bias + SiLU, and one CUDA graph for the
whole N = 5 stage including output assembly. About 2.5 ms per sample on an A100.

## Repository layout

```
solution.py              train(data_dir, output_dir, device) and load_predictor(output_dir, device)
configs/model.json       transformer size
src/data/prep_real.py    real windows (train / validation) and full real traces at 32x64
src/data/prep_sim.py     simulation traces in physical units, masked like the PIV data
src/model.py             the point-token transformer
src/train.py             one training run (all cases, or one Reynolds-row fold left out)
src/oof_bundles.py       out-of-fold mu / sigma of the fold ensembles
src/train_width.py       the interval-width CNN
src/scoring.py           the official subscore implementation from the starting kit (unchanged)
runtime/                 inference runtime copied into the trained package
```

## Reproducing

The organizers' setup: the raw release mounted at `/data` (`train_real/*.h5`, `train_sim/*.h5`), output at `/output`.

```bash
docker build --platform linux/amd64 -t calibrated-sim2real-flow .
docker run --gpus all -v /path/to/release:/data:ro -v /path/to/output:/output calibrated-sim2real-flow \
    python -c "import solution; solution.train('/data', '/output', 'cuda')"
```

`train()` runs all stages (data, 3 + 15 training runs, out-of-fold bundles, width CNN, packaging, a smoke test of the
package) and writes `/output/package/` (the inference package) and `/output/logs/` (a timeline and the log of every
stage and run). The submitted version of this pipeline (same computation) took 6.8 h on one A100-SXM4-80GB in this image
(data 9 min, 18 training runs of 20-25 min, bundles and width model 11 min), peak GPU memory 3.0-3.4 GB per run. If `train()` is interrupted, calling it again resumes:
finished stages and runs are kept. `S2R_SMOKE=1` runs the same pipeline with tiny runs as an interface check.

Predicting with the trained package:

```python
import solution
p = solution.load_predictor('/output', 'cuda')
out = p.predict(x)   # x: (N, 20, 32, 64, 3) float32, physical units -> {"prediction", "lower", "upper"}
```

**Randomness.** Training runs use seeds 42 / 43 / 44 (a run that hits a non-finite step is restarted once with
seed + 1000), the width CNN members 0 / 1 / 2; data sampling uses NumPy generators seeded from the run seed. GPU kernels
(bf16 attention backward, `torch.compile`) are not bit-deterministic, so a retrain differs from ours at the level of
seed noise: between two seeds of the recipe we saw +-0.001 rel-L2 and 0.03-0.05 in the estimated final score.

This repository is a cleaned-up copy of the training package we submitted for the Decision Phase (the one the organizers
retrained), with unused options removed and files renamed. On the same inputs it computes the same numbers bit for bit:
the inference runtime on the submitted weights, the out-of-fold bundles and the width CNN, and the training steps when
both are run with deterministic GPU kernels.

## License

MIT, see `LICENSE`. `src/scoring.py` is the organizers' file from the competition starting kit.
