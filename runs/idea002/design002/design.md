**Design Description:** Severity-annealed two-axis curriculum: in `NoiseCurriculum`, retain α₁ for firing-probability ramp (clean→full as today) and add α₂ that linearly anneals the *severity* parameters of the existing `AttackLayer` from "low" to "high" (jpeg_q_min 80→40, jpeg2 q range 60→20–40, resize_scale_min 0.9→0.5, chain_p 0.0→0.3) over the same ramp window, so the encoder first sees light compression then progressively the worst-case settings.

**Parent:** baseline/

**Starting Point:** baseline/

---

## 1. Motivation and approach

The baseline `NoiseCurriculum` has a single α that ramps Bernoulli firing probabilities of a fixed-severity (`high`) `AttackLayer`. The result is that even at α=0.1 (early in the ramp), the rare batch that *does* fire sees the full `jpeg_q_min=40, scale_min=0.5, chain_p=0.3` regime — a sharp cliff that the encoder must absorb while still being mostly-clean-trained. Per `idea.md`, this contributes to the wechat-tail underfit because the worst-case settings are encountered too rarely for stable convergence.

This design replaces "firing-probability-only" annealing with a two-axis schedule:
- **α₁** (firing): the existing α from `alpha(step)`. Scales `jpeg1.p`, `jpeg2.p`, `resize.p` exactly as today.
- **α₂** (severity): a parallel `severity_alpha(step)` that linearly interpolates the kornia attack parameters between SEVERITY["low"] (`jpeg_q_min=80, chain_q_min=60, resize_scale_min=0.9, chain_p=0.0`) at α₂=0 and SEVERITY["high"] (`jpeg_q_min=40, chain_q_min=20, resize_scale_min=0.5, chain_p=0.3`) at α₂=1. Both α₁ and α₂ are zero during the clean phase and ramp linearly across `stage_b_steps`, so the worst-case regime is reached only at `step >= stage_a_steps + stage_b_steps`.

Analogous to LR warmup but for attack difficulty: the encoder learns to embed under high-Q JPEG and mild resize first, and worst-case (q=40, scale=0.5, double-JPEG) only appears once the secret pathway is robust enough to absorb it. This closes the wechat-tail without any change to the loss or the model.

## 2. Files changed (Builder may only edit these)

- `code/baseline/exp0_inn_train.py` — extend `NoiseCurriculum` with `severity_alpha(step)` and per-step mutation of the kornia attack parameters; add logging.

Untouched: `code/baseline/inn_model.py`, `code/baseline/losses.py`, everything in `infra/**`.

## 3. CLI flag values for `baseline/exp0_inn_train.py` (config)

Locked: `--batch_size 16`, `--secret_len 896`, `--resolution 256`, `--max_steps 30000`, `--total_steps 30000`.

Tuned for this design (all baseline defaults retained):
- `--num_blocks 16`
- `--subnet_type rdb`
- `--lr 2e-4`
- `--img_weight 1.0`
- `--bit_weight 20.0`
- `--stage_a_steps 5000`
- `--stage_b_steps 10000`

Other flags: baseline defaults (`--ckpt_every 2000`, `--log_every 200`, `--val_every 1000`, `--val_n 50`, `--num_workers 4`, `--amp` on, `--seed 0`).

The severity-annealing endpoints (low → high) are hard-coded in `NoiseCurriculum.__init__` (see §4) so no new CLI flag is added.

## 4. Algorithmic changes (concrete)

### 4.1 Cache the severity endpoints in `NoiseCurriculum.__init__`

Add after the existing `self._p_max = {...}` block:

```python
# Severity endpoints for the secondary (α₂) annealing axis. Values mirror
# infra/attacks.py SEVERITY["low"] and SEVERITY["high"]. Re-declared here
# (not imported) because infra/attacks.py is locked.
self._sev_lo = dict(jpeg_q_min=80, chain_q_min=60,
                    resize_scale_min=0.9, chain_p=0.0)
self._sev_hi = dict(jpeg_q_min=40, chain_q_min=20,
                    resize_scale_min=0.5, chain_p=0.3)
self.last_sev_alpha = 0.0  # diagnostic for logging
```

### 4.2 Add `severity_alpha(step)`

Identical schedule to `alpha(step)` (so the firing and severity ramps move together):

```python
def severity_alpha(self, step: int) -> float:
    if step < self.ramp_start:
        return 0.0
    if step >= self.ramp_end:
        return 1.0
    return (step - self.ramp_start) / (self.ramp_end - self.ramp_start)
```

### 4.3 Replace `attack(step)` with severity-mutating version

```python
def attack(self, step: int):
    a1 = self.alpha(step)
    if a1 == 0.0:
        self.last_sev_alpha = 0.0
        return None
    a2 = self.severity_alpha(step)
    self.last_sev_alpha = a2
    # Linearly interpolate the four severity knobs.
    lo, hi = self._sev_lo, self._sev_hi
    jpeg_q_min   = lo["jpeg_q_min"]   + a2 * (hi["jpeg_q_min"]   - lo["jpeg_q_min"])
    chain_q_min  = lo["chain_q_min"]  + a2 * (hi["chain_q_min"]  - lo["chain_q_min"])
    resize_smin  = lo["resize_scale_min"] + a2 * (hi["resize_scale_min"] - lo["resize_scale_min"])
    chain_p_max  = lo["chain_p"]      + a2 * (hi["chain_p"]      - lo["chain_p"])

    # Mutate kornia parameters in-place. RandomJPEG.jpeg_quality is a
    # buffer/tensor (kornia stores quality bounds as a registered buffer);
    # write through .data to avoid breaking the registered Parameter/Buffer.
    q1_lo = float(min(94.0, max(10.0, jpeg_q_min)))
    self._attack.jpeg1.jpeg_quality = (q1_lo, 95.0)
    q2_lo = float(min(q1_lo - 1.0, max(10.0, chain_q_min)))
    self._attack.jpeg2.jpeg_quality = (q2_lo, q1_lo)
    self._attack.resize.scale_min = float(max(0.1, min(1.0, resize_smin)))
    # Bernoulli probabilities scale with α₁ (firing axis); chain_p target
    # itself is annealed by α₂.
    self._attack.jpeg1.p = self._p_max["jpeg1"] * a1
    self._attack.jpeg2.p = chain_p_max * a1
    self._attack.resize.p = self._p_max["resize"] * a1
    return self._attack
```

Notes:
- kornia's `RandomJPEG.jpeg_quality` accepts a Python tuple at construction; for runtime updates we re-assign the attribute. Builder must verify with a `print(self._attack.jpeg1.jpeg_quality)` after mutation in the `[init]` log that the new bounds are honored on the next forward pass; if kornia caches the bound internally, fall back to constructing a fresh `K.RandomJPEG(jpeg_quality=(q1_lo, 95.0), p=...)` and replacing `self._attack.jpeg1` (re-`.to(device)` after).
- `_p_max["jpeg2"]` is no longer used; jpeg2's per-step probability is `chain_p_max * a1`, which equals `_p_max["jpeg2"] * a1` only at α₂=1.

### 4.4 Update `stage(step)` for richer diagnostics

```python
def stage(self, step: int) -> str:
    a1 = self.alpha(step)
    a2 = self.severity_alpha(step)
    if a1 == 0.0:
        return "clean"
    if a1 >= 1.0 and a2 >= 1.0:
        return "full"
    return f"ramp(a1={a1:.2f},a2={a2:.2f})"
```

### 4.5 Sanity-check the mutation path at startup

In `main()`, after the existing sanity_check call:

```python
# Probe the curriculum at the easiest, midpoint, and worst severity to
# catch kornia parameter-update bugs at startup, not at step 5001.
for _probe_step in (curriculum.ramp_start + 1,
                    (curriculum.ramp_start + curriculum.ramp_end) // 2,
                    curriculum.ramp_end + 1):
    _atk = curriculum.attack(_probe_step)
    if _atk is not None:
        with torch.no_grad():
            _ = _atk(torch.zeros(2, 3, args.resolution, args.resolution,
                                  device=device, dtype=torch.float16))
print("[init] severity-annealing sanity check passed.")
```

After the probe, reset jpeg/resize parameters back to the worst-case endpoint values that `attack(ramp_end + 1)` set, so the in-memory state is consistent with what training will see at step `ramp_end + 1`.

### 4.6 Logging additions

Extend the per-`log_every` `rec` dict in `metrics.jsonl` to include:
- `severity_alpha` (float) — `curriculum.last_sev_alpha`.
- `jpeg1_q_min` (float) — current `self._attack.jpeg1.jpeg_quality[0]`.
- `resize_scale_min` (float) — current `self._attack.resize.scale_min`.

Update the printed line to append ` sev_α=<a2>`.

`val_metrics.jsonl`: unchanged.

## 5. Expected behavior after the change

- `metrics.jsonl` lines now include `severity_alpha`, `jpeg1_q_min`, `resize_scale_min`.
- During the ramp window (`stage_a_steps ≤ step < stage_a_steps + stage_b_steps`), `jpeg1_q_min` decays linearly from 80 to 40 and `resize_scale_min` from 0.9 to 0.5; after the ramp, both are pinned at the worst-case values for the remaining ~15k steps.
- `bit_acc_ema` should track higher than baseline through the early ramp (because attacks are easier early), then converge to within ±0.01 of baseline by step 15000 and exceed baseline at step 30000.
- `psnr_ema` at step 30000 expected within ±0.3 dB of baseline.
- Eval `bit_acc` on `chain_wechat`, `chain_whatsapp_std`, `chain_x` should rise (target: `chain_wechat` ≥ 0.995 vs baseline 0.988).
- `bit_acc_clean = 1.000` at every eval resolution must hold post-eval.
- Wallclock: parameter mutation per step is O(1); no measurable slowdown.
- `score = 2 * psnr_mean * (bit_acc_mean - 0.5)` is expected to improve over baseline 27.27.

## 6. Constraints and invariants Builder must preserve

- Do NOT modify any `infra/**` file. The severity annealing must mutate the *existing* `AttackLayer` instance in place.
- Do NOT change locked CLI flags (`--batch_size 16`, `--secret_len 896`, `--resolution 256`, `--max_steps 30000`, `--total_steps 30000`).
- Do NOT raise `--bit_weight` above 20.0; isolation from the bit-pressure axis is required.
- Do NOT swap in `ImageSecretLoss` — loss stays pure-L1 host + BCE bits (idea002 forbids combining with idea001's axis).
- Preserve the clean phase: `attack(step)` MUST return `None` while `α₁ == 0`.
- The α₂ schedule MUST be identical to α₁ in this design (same `ramp_start`/`ramp_end`). Builder must NOT introduce a separate `--severity_ramp_steps` flag; that is a different design.
- The `low` and `high` endpoints MUST equal the values in `infra/attacks.py::SEVERITY` at the time of writing (jpeg_q_min 80↔40, chain_q_min 60↔20, resize_scale_min 0.9↔0.5, chain_p 0.0↔0.3). If those constants change in `infra/`, this design must be re-evaluated.
- Preserve gradient clip, optimizer, cosine LR schedule, AMP setup, and resume/checkpoint logic verbatim.
- Preserve `container = container.clamp(-1.0, 1.0)` before the attack call.
- All edits must live inside `code/baseline/`; no edits anywhere else.

## 7. Edge cases

- kornia API: `RandomJPEG.jpeg_quality` is documented as a constructor arg. If runtime re-assignment is silently ignored by the version installed (verify in §4.5 startup probe), Builder must replace `self._attack.jpeg1` with a freshly-constructed module each call. Acceptable because cost is sub-millisecond and per-step rebuild is fine.
- The midpoint probe in §4.5 mutates the layer's state; the post-probe step-1 state must equal what `attack(0)` would produce (i.e., return `None`, since α₁=0). Just call `curriculum.attack(0)` once after the probe to reset Bernoulli probabilities to 0.
- Resume from `ckpt_latest.pt`: `NoiseCurriculum` carries no learnable parameters; `attack(step)` is reconstructed deterministically from `step` on every call, so resume is safe with no special handling.
- AMP: parameter mutation happens outside the autocast region; the kornia layer's internal LUT cast is unaffected.
- RNG: the only stochasticity in `attack(step)` is kornia's internal sampling, which is driven by the global torch RNG (already snapshotted in checkpoints).
