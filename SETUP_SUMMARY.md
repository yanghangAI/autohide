# Setup Summary — imagehide

Target project: `/work/pi_nwycoff_umass_edu/hang/imagehide`

## Project in one paragraph

`imagehide` is a research project on a lightweight on-camera invisible-watermark scheme. The carrier is an INN-based encoder that embeds an 896-bit payload (`H | sig | pk`) into pixels and recovers it under social-media compression pipelines. The current best ("mobile v2") is an 8-block DKiS coupling INN with `LightConvBlock` subnets (~0.167 M params), trained at 256×256 against a compression-only attack mix (resize → JPEG, plus platform chains), with a Bernoulli α-ramp curriculum. The research loop will iterate on this architecture and training recipe.

## 6-line summary

1. **Training script:** `src/exp0_inn_train.py` (Python entrypoint; canonical invocation = `scripts/exp0_inn_p896_light_v2.sbatch`)
2. **Primary metric:** `score = 2 × PSNR × (mean_bit_acc − 0.5)` (higher is better), computed from the end-of-training multi-size eval, where:
   - `PSNR` = mean clean PSNR across all 9 eval resolutions (cover vs. container).
   - `mean_bit_acc` = mean bit accuracy over the 13 compression attacks × 9 resolutions (117 cells), excluding the clean column.
   - **Hard condition:** clean bit_acc must equal 1.000 at every resolution. If violated → run is flagged failed and `score` is not used for ranking.
   Also tracked: `psnr_mean`, `bit_acc_mean`, `bit_acc_clean_min` (over resolutions), per-attack/per-resolution bit_acc, `ssim_mean`.
3. **Done when:** `step >= 30000`
4. **Runtime:** SLURM (gpu-preempt partition, 1× 2080Ti, 24 GB RAM, 4 CPUs, ~12 h walltime)
5. **Baseline files (experimentable):** `exp0_inn_train.py`, `inn_model.py`, `losses.py`
6. **Infra files (must never change):** `dataset.py`, `attacks.py`, `exp0_inn_eval.py`, `exp0_eval.py`, `exp0_report.py`, `robust_hash.py`
7. **Submit-test:** fast mini-train with `--max_steps 200 --stage_a_steps 50 --stage_b_steps 100 --val_every 0 --ckpt_every 200` (~2–5 min on a 2080Ti); validates pipeline end-to-end (model → embed → attack → extract → checkpoint), no inline val.

## Contract

**Experimentable (designs may modify):**
- `baseline/exp0_inn_train.py` — training loop, optimizer, loss weights, curriculum schedule.
- `baseline/inn_model.py` — INN architecture (block count, subnet type, adapter).
- `baseline/losses.py` — loss formulations.

**Must never change (infra; identical across all designs):**
- `infra/dataset.py` — `CoverDataset` (training data loader).
- `infra/attacks.py` — `AttackLayer` and `SEVERITY` defs. Locked because the eval attack suite must be invariant for fair cross-design comparison; designs may not redefine the attack distribution.
- `infra/exp0_inn_eval.py` — multi-resolution × 13-attack eval harness; `build_attacks()` and `load_to_size()` are also imported by the trainer for inline validation.
- `infra/exp0_eval.py` — CNN-baseline eval harness (legacy).
- `infra/exp0_report.py` — exp0 results aggregator.
- `infra/robust_hash.py` — pHash baseline for `H`.

**Locked hyperparameters (designs may not change):**
- `--batch_size 16`
- `--secret_len 896`
- `--resolution 256`
- `--max_steps 30000`, `--total_steps 30000`
- Eval attack suite (13 attacks × 9 sizes; defined in `infra/exp0_inn_eval.py`).

Designs may change: `--num_blocks`, `--subnet_type`, `--lr`, `--img_weight`, `--bit_weight`, `--stage_a_steps`, `--stage_b_steps`, model internals, loss formulations.

## Metrics flow

Training writes `metrics.jsonl` (per `--log_every`) and `val_metrics.jsonl` (per `--val_every`) into the run output directory. After training reaches `step >= 30000`, the submission script runs the multi-size eval (`infra/exp0_inn_eval.py`) on the final checkpoint, producing `eval.json` (9 sizes × 13 attacks).

The Infra_Baseline_Agent will add a post-eval adapter that aggregates `eval.json` into a single-row `metrics.csv` with columns:

```
step, score,
psnr_mean, ssim_mean,
bit_acc_mean, bit_acc_clean_mean, bit_acc_clean_min,
psnr_<size>, ssim_<size>,
bit_acc_clean_<size>,
bit_acc_<attack>_mean,             # per-attack mean across resolutions
bit_acc_<size>_mean,               # per-resolution mean across attacks
bit_acc_<attack>_<size>,           # full per-(attack, resolution) bit_acc
score_<attack>_<size>,             # full per-(attack, resolution) score
...
```

where:
- `score = 2 * psnr_mean * (bit_acc_mean - 0.5)` (primary metric, higher is better).
- `psnr_mean` = mean clean PSNR over the 9 eval resolutions.
- `bit_acc_mean` = mean bit_acc over 13 attacks × 9 resolutions (117 cells, excluding clean).
- `bit_acc_clean_min` = minimum clean bit_acc across the 9 resolutions; the **hard condition** requires this to equal 1.000.
- `score_<attack>_<size> = 2 * psnr_<size> * (bit_acc_<attack>_<size> - 0.5)` — per-attack, per-resolution score recorded for every (attack, resolution) cell.
- Per-resolution `psnr_<size>`, `bit_acc_clean_<size>`, and per-(attack, resolution) `bit_acc_<attack>_<size>` values are also written for transparency.

Progress field: `step`. Done value: 30000.

## Eval artifacts

During the final multi-size eval, save **1 (cover, container) pair per resolution** (9 pairs, 18 PNGs total) to `<run>/eval_samples/`:
- `cover_<size>.png` and `container_<size>.png`
- The cover image is picked **randomly and independently for each resolution** (different image per size). A fixed seed is used so the picks are reproducible within a run.

## Post-run cleanup

After training and eval both complete successfully, the submission script deletes all checkpoint files (`*.pt`, `*.pth`) in the run directory to save disk. `metrics.jsonl`, `val_metrics.jsonl`, `eval.json`, `metrics.csv`, and slurm logs are kept.

Note: the trainer's checkpoint file is `ckpt_latest.pt` (PyTorch convention). If a run fails (training did not reach `step >= 30000`, or eval errored), checkpoints are **kept** so the run can be resumed/debugged.

## Preferences

- **Architect model:** Opus (stronger model, per user preference).
- **Auto GitHub issue filing:** disabled.

## Notes on previous runs

Per user instruction: previous results under `results/` (collapsed v1/v2, baseline RDB, mobile v1, mobile v2, etc.) will **not** be carried over into the autohide framework. The research loop starts fresh — `runs/baseline/` will be a clean re-run of the mobile-v2 recipe, establishing ground-truth metrics inside the autohide pipeline.

## Project cleanliness check

The target repo is mid-experiment but the *code* is clean: no debug flags, no commented-out variants in `src/`. The `exp0_inn_p896_light_v2.sbatch` recipe is the canonical baseline starting point.

---

**Please review and approve before I run `scripts/setup.py`.** You can edit this file directly to make corrections; I'll re-read it before proceeding.
