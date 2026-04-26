**Design Description:** Single-scale Sobel mask + complementary edge-aware TV penalty on the residual — keep the parent's single-scale Sobel mask and its masked-YUV + masked-L1 terms unchanged; add a new term `tv_w * mean((1 - w_norm) · |∇(stego - cover)|_luma)` with `tv_w = 0.3`, where `w_norm = w / max(w)` per image and `∇` is the same Sobel gradient (magnitude) on the luma residual. This concentrates the TV penalty in flat regions, explicitly forbidding residual gradient where the cover has none.

**Parent:** runs/idea003/design003

**Starting Point:** runs/idea003/design003

---

## 1. Goal and motivation

This is direction (2) of `runs/idea007/idea.md`. The parent (`runs/idea003/design003`, score=28.16) down-weights penalty in textured regions via the Sobel JND mask `w = 1/(1 + β·M)`. Equivalently, it concentrates penalty in flat regions — but only as a *soft* down-weighting. A complementary term that *explicitly forbids* residual gradient in flat regions is a classic steganographic prior (residuals must lie in the host's structural support).

This design adds:
- A new term `l_tv = mean((1 - w_norm) · g_res)` where `g_res` is the per-pixel Sobel-gradient magnitude of the **luma residual** `(stego_luma - cover_luma)`, and `w_norm = w / w.amax(per-image)` so that `(1 - w_norm) ∈ [0, 1]` is large in flat regions and small in textured regions.
- Weight `tv_w = 0.3` (mid-range of the idea's suggested [0.2, 0.5]); small enough not to dominate but large enough to be measurable. Designer chooses 0.3 as a midpoint; this value is the default and is hard-coded in the constructor call.

If `score > 28.16`, the explicit flat-region prohibition adds headroom on top of the parent's soft mask. If equal, the parent's soft down-weighting already captures the same signal. If lower, the TV term over-constrains the encoder (e.g. forces it to put all residual into already-textured regions, where bit-recovery is harder under JPEG).

## 2. Files changed (Builder may only edit these)

Builder may only touch the following files inside `code/` (mirroring `baseline/`):

- `code/losses.py` — add a `_sobel_mag_luma(x)` helper that returns the per-pixel Sobel-magnitude of the luma channel of an RGB tensor (used on the residual `stego - cover`); add `tv_w` and `tv_eps` constructor kwargs to `ImageSecretLoss`; add the TV term inside `forward` and include it in `l_quality`; emit `l_tv` and `tv_w` in the `logs` dict.
- `code/exp0_inn_train.py` — pass `tv_w=0.3` to the `ImageSecretLoss(...)` call; add `l_tv`, `tv_w` to the per-step `metrics.jsonl` record.

Untouched (locked or out of scope):
- All of `infra/**` (byte-locked).
- `code/inn_model.py` — no architecture change.
- `code/train.py` — unchanged.

## 3. CLI flag values for `baseline/exp0_inn_train.py`

Locked (must NOT change): `--batch_size 16`, `--secret_len 896`, `--resolution 256`, `--max_steps 30000`, `--total_steps 30000`.

Tuned (identical to parent `runs/idea003/design003`):
- `--num_blocks 16`
- `--subnet_type rdb`
- `--lr 2e-4`
- `--img_weight 1.0`
- `--bit_weight 20.0`
- `--stage_a_steps 5000`
- `--stage_b_steps 10000`

No new CLI flags. `tv_w=0.3` is hard-coded in the `ImageSecretLoss(...)` constructor call inside `code/exp0_inn_train.py` (mirroring how the parent hard-codes `pixel_w=0.5`). All loss-config values (`jnd_beta=4.0`, `pixel_w=0.5`, `jnd_mode="sobel"`, `tv_w=0.3`) live in this single config call.

## 4. Algorithmic spec (concrete)

### 4.1 `code/losses.py`

**Keep** all existing helpers and the `ImageSecretLoss` class structure as in the parent. Only add a new helper and a new term inside `forward`.

**Add** the residual-gradient helper (operates on RGB input, computes Sobel magnitude on the luma channel; gradient is in the autograd graph because it is applied to `(stego - cover)`):

```python
def _sobel_mag_luma(rgb: torch.Tensor) -> torch.Tensor:
    """Sobel gradient magnitude of the luma channel of an RGB tensor.
    Input shape (B, 3, H, W); output shape (B, 1, H, W). NOT detached —
    suitable for use inside a differentiable loss term on the residual.
    """
    yuv = rgb_to_yuv(rgb)
    luma = yuv[:, 0:1]  # (B, 1, H, W)
    sx = _SOBEL_X.to(luma.device, luma.dtype)
    sy = _SOBEL_Y.to(luma.device, luma.dtype)
    gx = F.conv2d(luma, sx, padding=1)
    gy = F.conv2d(luma, sy, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
    return mag
```

**Modify `ImageSecretLoss.__init__`** to accept `tv_w` and `tv_eps` (epsilon used to clamp `w.amax` in case of degenerate covers):

```python
def __init__(self,
             secret_weight: float = 20.0,
             yuv_w: float = 1.5,
             lpips_w: float = 1.0,
             ffl_w: float = 1.5,
             jnd_beta: float = 0.0,
             pixel_w: float = 0.0,
             jnd_mode: str = "sobel",
             tv_w: float = 0.0,
             tv_eps: float = 1e-6):
    super().__init__()
    assert jnd_mode in ("sobel", "variance"), f"unknown jnd_mode={jnd_mode}"
    self.secret_weight = secret_weight
    self.yuv_w = yuv_w
    self.lpips_w = lpips_w
    self.ffl_w = ffl_w
    self.jnd_beta = jnd_beta
    self.pixel_w = pixel_w
    self.jnd_mode = jnd_mode
    self.tv_w = tv_w
    self.tv_eps = tv_eps
    self.lpips = lpips_lib.LPIPS(net="alex", verbose=False)
    for p in self.lpips.parameters():
        p.requires_grad_(False)
    self.ffl = FocalFrequencyLoss()
```

**Modify `ImageSecretLoss.forward`**, inside the `if self.jnd_beta > 0.0:` branch (after `w` is computed, after `l_pixel` is computed), insert the TV term. The TV term uses the *same* `w` that already exists; it does NOT recompute the mask.

```python
if self.jnd_beta > 0.0:
    if self.jnd_mode == "sobel":
        M = sobel_jnd_mask(cover)
    else:
        M = variance_jnd_mask(cover, win=7)
    w = 1.0 / (1.0 + self.jnd_beta * M)  # (B, 1, H, W)
    l_yuv = (w * sq).mean()
    if self.pixel_w > 0.0:
        l_pixel = (w * (stego - cover).abs()).mean()
    else:
        l_pixel = torch.zeros((), device=stego.device, dtype=stego.dtype)

    if self.tv_w > 0.0:
        # Per-image normalize w to [0, 1] for the complementary mask.
        B = w.shape[0]
        w_max = w.view(B, -1).amax(dim=1).clamp_min(self.tv_eps)
        w_norm = w / w_max.view(B, 1, 1, 1)        # (B, 1, H, W) in [0, 1]
        comp = 1.0 - w_norm                         # 1 in flat regions, 0 in max-textured
        # Sobel magnitude of luma residual; differentiable wrt stego.
        g_res = _sobel_mag_luma(stego - cover)      # (B, 1, H, W)
        l_tv = (comp * g_res).mean()
    else:
        l_tv = torch.zeros((), device=stego.device, dtype=stego.dtype)
else:
    l_yuv = l_yuv_unweighted
    if self.pixel_w > 0.0:
        l_pixel = (stego - cover).abs().mean()
    else:
        l_pixel = torch.zeros((), device=stego.device, dtype=stego.dtype)
    if self.tv_w > 0.0:
        # No mask available; use unit comp (penalize gradient everywhere).
        g_res = _sobel_mag_luma(stego - cover)
        l_tv = g_res.mean()
    else:
        l_tv = torch.zeros((), device=stego.device, dtype=stego.dtype)
```

Update `l_quality`:

```python
l_quality = (self.yuv_w * l_yuv
             + self.lpips_w * l_lpips
             + self.ffl_w * l_ffl
             + self.pixel_w * l_pixel
             + self.tv_w * l_tv)
```

Update the `logs` dict to add `l_tv`:

```python
logs = dict(
    loss=total.detach(),
    l_bce=l_bce.detach(),
    l_yuv=l_yuv.detach(),
    l_yuv_unweighted=l_yuv_unweighted.detach(),
    l_lpips=l_lpips.detach(),
    l_ffl=l_ffl.detach(),
    l_pixel=l_pixel.detach(),
    l_tv=l_tv.detach(),
    l_quality=l_quality.detach(),
    alpha=alpha,
)
```

### 4.2 `code/exp0_inn_train.py`

Modify the `ImageSecretLoss(...)` call:

```python
img_loss_mod = ImageSecretLoss(
    secret_weight=args.bit_weight,   # 20.0
    yuv_w=1.5,
    lpips_w=0.0,
    ffl_w=1.5,
    jnd_beta=4.0,
    pixel_w=0.5,
    jnd_mode="sobel",
    tv_w=0.3,
).to(device)
```

In the per-step `metrics.jsonl` record, add:

```python
l_tv=float(loss_logs["l_tv"]),
tv_w=0.3,
```

(Keep all existing fields: `l_yuv`, `l_yuv_unweighted`, `l_pixel`, `jnd_beta=4.0`, `pixel_w=0.5`, `jnd_mode="sobel"`.)

No other change to `exp0_inn_train.py`.

## 5. Expected behavior after the change

- `metrics.jsonl` per-step records include `l_tv` (≥ 0) and `tv_w=0.3`, alongside the parent's existing fields.
- `l_tv` magnitude: residual luma values are in `[-1, 1]` and typical `|stego - cover|` ≈ 1e-2; Sobel magnitude on a smooth-ish residual is comparable in scale, so `g_res ≈ 1e-2`. With `comp ∈ [0, 1]` averaging ~0.5 (depends on cover statistics), `l_tv ≈ 5e-3`. With `tv_w=0.3`, the contribution to `l_quality` is ~1.5e-3 — same order as `pixel_w * l_pixel ≈ 2.5e-3` and `yuv_w * l_yuv ≈ 7.5e-4`. Competitive but not dominant; matches the design intent.
- Hypothesis on outcome: PSNR ≥ parent (28.48); `score` ≥ 28.16. The TV term should shift residual energy out of flat regions and into textured regions where it is perceptually invisible, raising PSNR (residual magnitude can stay the same or even grow but its visibility falls). Bit_acc cost is bounded — the encoder still has the textured regions for residual placement, which is the half of the image where bits should already be going.
- Wallclock: one extra Sobel pass on the luma residual per step (≈ 1 conv on (16, 1, 256, 256)). <0.5 ms/step on a 2080Ti. Negligible.
- A design is `Done` when training reaches `step >= 30000`. After training, the locked multi-resolution eval (`infra/exp0_inn_eval.py`, 13 attacks × 9 resolutions) runs and writes `eval.json` → aggregated into `metrics.csv`.

## 6. Constraints and invariants Builder must preserve

- No edits to `infra/**`. No edits to `code/inn_model.py` or `code/train.py`.
- Locked CLI flags untouched: `--batch_size 16`, `--secret_len 896`, `--resolution 256`, `--max_steps 30000`, `--total_steps 30000`.
- All non-locked CLI flags identical to the parent.
- All loss-construction kwargs identical to the parent except for the *added* `tv_w=0.3`. Specifically: `secret_weight=20.0`, `yuv_w=1.5`, `lpips_w=0.0`, `ffl_w=1.5`, `jnd_beta=4.0`, `pixel_w=0.5`, `jnd_mode="sobel"` are unchanged.
- LPIPS off (`lpips_w=0.0`); short-circuit preserved.
- Mask `M` (and therefore `w`) is computed exactly once per `forward` and reused for the YUV term, the L1 pixel term, and the TV term's `w_norm`. No second mask compute. `M` and `w` themselves are detached from autograd (the helper is `@torch.no_grad`).
- The TV term operates on `(stego - cover)`, the **residual** — never on `stego` directly. Penalizing `|∇stego|` would just blur the cover; penalizing `|∇(stego - cover)|` blurs only the *added* signal, which is the intended steganographic prior.
- The Sobel gradient inside `_sobel_mag_luma` MUST be differentiable wrt `stego` (no `torch.no_grad`, no `.detach()` on the residual). This is the gradient pathway through which the TV term shapes the encoder.
- The Sobel kernels MUST be reused (`_SOBEL_X`, `_SOBEL_Y`); no new dependencies.
- Per-image `w_norm = w / w.amax` ensures `comp = 1 - w_norm ∈ [0, 1]` regardless of cover statistics (β scaling cancels), so `tv_w` has consistent semantics across covers.
- `bit_acc_clean = 1.000` at every eval resolution remains the hard target. If a smoke run shows `bit_acc_clean_min < 0.99` at step 5000+, flag to Designer (would suggest `tv_w` is too high; do not silently retune). Idea spec allows reducing the new term's weight if the parent's clean tail degrades materially — but that's a Designer-side decision, not a Builder one.
- Optimizer, LR scheduler, AMP autocast scope, gradient clipping, curriculum α-ramp, validation cadence, checkpoint format unchanged.

## 7. Edge cases

- **Per-image `w.amax` normalization**: `w = 1/(1 + β·M)` with `M ≥ 0` and `M_max = 1` (parent's max-normalized mask) gives `w_min = 1/(1+β) = 0.2` at β=4 and `w_max = 1` at the flattest pixel. `w_norm` is therefore in `[0.2, 1]`, and `comp = 1 - w_norm ∈ [0, 0.8]`. The TV term penalizes flat-region gradient with weight up to 0.8 and textured-region gradient with weight 0. Intentional.
- **Constant-luma cover**: `M = 0` everywhere → `w = 1` everywhere → `w.amax = 1` → `w_norm = 1` → `comp = 0` everywhere → `l_tv = 0`. Correct degenerate behavior (no flat-vs-textured signal to exploit).
- **Saturated-edge cover**: `M ≈ 1` everywhere → `w ≈ 1/(1+β)` everywhere → `w_norm = 1` → `comp = 0` → `l_tv = 0`. Also correct (no flat regions to protect).
- **AMP / autocast**: the residual Sobel runs under autocast like the rest of the loss; `sqrt` with `+1e-12` avoids fp16 underflow in flat regions. Same precision regime as the parent's mask compute.
- **`tv_w = 0` fallback**: explicit `torch.zeros((), device=...)` for `l_tv`; the `_sobel_mag_luma` call is guarded so it does not run when `tv_w = 0`. This design always sets `tv_w = 0.3`, but the guard MUST remain for ablations.
- **Gradient through residual TV**: differentiating `mean(comp · sqrt(gx² + gy² + 1e-12))` wrt `stego` is well-defined; the `+1e-12` prevents zero-divide at flat pixels of the residual. The gradient pushes `stego` toward `cover` in flat regions, in a *gradient-magnitude* sense (not a pixel-magnitude sense — the latter is what `l_pixel` already does).
- **Interaction with `l_pixel`**: `l_pixel = mean(w · |stego - cover|)` penalizes residual *magnitude* in flat regions; `l_tv = mean(comp · |∇(stego - cover)|)` penalizes residual *structure* in flat regions. They are not redundant — a flat residual offset (e.g. constant +1e-3) has high `l_pixel` but zero `l_tv`; a high-frequency residual with mean 0 has low `l_pixel` but high `l_tv`. The TV term targets the structural component specifically.
- **JSON serialization**: `l_tv` is logged as `float(loss_logs["l_tv"])`. No tuple/list issues.
