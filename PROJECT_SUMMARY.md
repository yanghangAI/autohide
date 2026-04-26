# Project Summary — autohide / imagehide

## Project overview

Target project: `imagehide` — research on a lightweight on-camera invisible-watermark scheme. The carrier embeds an 896-bit payload (`H | sig | pk`) into pixels using an INN-based encoder so that the payload survives social-media compression pipelines (resize, JPEG re-encoding, platform chains). The autohide framework will iterate on this carrier — searching for designs that improve the trade-off between image fidelity (PSNR) and bit-accuracy under the locked compression-attack suite — while keeping the eval protocol invariant for fair comparison.

## Baseline

Mobile-v2 architecture: 8 DKiS coupling blocks × `LightConvBlock` subnets, 0.167 M parameters, with a zero-parameter patch-tile adapter that maps the 896 bits onto a 32×32 grid in DWT space. Trained at 256×256 with batch_size=16, lr=4e-4, for 30 000 steps, using a Bernoulli α-ramp curriculum (clean 0–5k → ramp 5k–15k → full 15k–30k) against a compression-only attack mix (`resize → JPEG`, plus platform chains).

Key files (in `baseline/`):
- `exp0_inn_train.py` — training loop, optimizer, curriculum, checkpoint+resume.
- `inn_model.py` — `INNCodec` (DKiS-derived coupling INN + Haar DWT + patch-tile adapter).
- `losses.py` — loss formulations (currently L1 image + BCE bit).
- `train.py` — shim that delegates to `exp0_inn_train.py` (framework entrypoint).

## Metrics

**Primary metric:** `score = 2 × psnr_mean × (bit_acc_mean − 0.5)` (higher is better).

- `psnr_mean` = mean clean PSNR across the 9 eval resolutions.
- `bit_acc_mean` = mean bit_acc over 13 compression attacks × 9 resolutions = 117 cells (excluding the clean column).

**Hard condition:** `bit_acc_clean = 1.000` at every resolution. If `bit_acc_clean_min < 1.0` the run is flagged failed and `score` is not used for ranking.

**Tracked metrics** (per-row in `metrics.csv`): `step`, `score`, `psnr_mean`, `ssim_mean`, `bit_acc_mean`, `bit_acc_clean_mean`, `bit_acc_clean_min`, plus per-resolution `psnr_<size>` / `ssim_<size>` / `bit_acc_clean_<size>`, per-attack mean `bit_acc_<attack>_mean`, per-resolution mean `bit_acc_<size>_mean`, full per-cell `bit_acc_<attack>_<size>` and `score_<attack>_<size>`.

**Completion rule:** progress field `step`, done when `step >= 30000`.

## Runtime

SLURM cluster (UMass Unity HPC), `gpu-preempt` partition, single 2080Ti, 4 CPUs, 24 GB RAM, 12 h walltime, `--requeue`. Conda env `hang`.

Training is invoked via the framework:
- Full train: `python scripts/cli.py submit-train <design_dir>` (wraps `scripts/slurm/slurm_train.sh`).
- Smoke test: `python scripts/cli.py submit-test <design_dir>` (~2–5 min mini-train, 200 steps).

## Directory layout

```
autohide/
├── baseline/        Experimentable starter: exp0_inn_train.py, inn_model.py,
│                    losses.py, train.py (shim).
├── infra/           Locked, shared modules — must NOT change between designs.
│                    constants.py (paths + research invariants), dataset.py,
│                    attacks.py, exp0_inn_eval.py, exp0_eval.py, exp0_report.py,
│                    robust_hash.py.
├── runs/            Per-run output. baseline/ holds the canonical reference run.
├── agents/          Architect, Designer, Reviewer, Builder, Debugger,
│                    Orchestrator — one prompt.md each.
├── scripts/         cli.py + slurm/{slurm_train.sh, slurm_test.sh,
│                    submit_train.sh, submit_test.sh}.
└── .automation.json Framework config: metric_fields, primary_metric, runtime.
```

## How to start the research loop

In a **fresh** Claude Code session (so the Setup-Agent context doesn't leak):

```
Read agents/Orchestrator/prompt.md and act as the Orchestrator.
```

The Orchestrator drives Architect → Designer → Reviewer → Builder → Debugger and submits jobs through `scripts/cli.py`.

**Recommended first step:** run a full baseline training before starting the loop, so the Architect has ground-truth metrics to compare against:

```
python scripts/cli.py submit-train runs/baseline/code/train.py baseline
```

## Baseline test results

`runs/baseline/test_output/` (job 56263204, 2 min on 2080Ti):
- 200 training steps completed; `ckpt_latest.pt` written.
- Final: `bit_acc_ema=0.903`, `psnr_ema=14.71` (smoke-test only — the architecture is healthy at 30k steps per imagehide HANDOFF results, not at 200 steps).
- No `training_failed.txt` → pipeline confirmed working end-to-end.
