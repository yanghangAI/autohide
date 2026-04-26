**Design Description:** Joint Sobel JND mask on YUV term **and** on a new per-pixel L1 residual term — adds a complementary spatial-domain pixel penalty `pixel_w * mean(w · |stego - cover|)` (with `pixel_w=0.5`) alongside the JND-weighted YUV term and the unchanged FFL term. Same Sobel mask + β=4.0 as design001. Tests whether an explicit spatial-domain term — currently absent from design003 — adds headroom when masked.

**Parent:** runs/idea001/design003

**Starting Point:** runs/idea001/design003

---

## 1. Motivation and approach

This is direction (3) of `runs/idea003/idea.md`. design001 of this idea adds a Sobel-JND mask to the YUV term only. The idea raises a complementary question: design003's quality loss has **no explicit spatial-domain pixel penalty** — it has YUV (channel-mixed but still pixel-domain MSE), FFL (purely spectral), and (off) LPIPS. An L1 pixel residual provides a **direct spatial sparsity prior** on the residual `(stego - cover)`, which combined with a JND mask becomes a content-adaptive sparsity penalty that pushes residual energy spatially toward textured regions while keeping its magnitude small everywhere.

This design adds:
1. The same Sobel JND mask weighting on the YUV term as design001 (`jnd_beta=4.0`, `jnd_mode="sobel"`).
2. A new term `pixel_w * mean(w · |stego - cover|)` with `pixel_w = 0.5`, where `w` is the *same* Sobel-derived mask reused (no second mask compute).
3. FFL term unchanged.

`pixel_w=0.5` is chosen small (1/3 of `yuv_w=1.5`) because L1 in pixel space is a coarser perceptual prior than YUV-MSE; it should *augment* not *dominate* the existing terms. Reusing the same mask `w` (computed once per step in `forward`) keeps the cost identical to design001.

If design003-of-idea003 (this design) > design001-of-idea003, the spatial-domain pixel term — masked — provides additional headroom. If it is worse, the YUV-MSE term already covers the pixel-domain signal and the extra L1 just over-penalizes the residual budget.

## 2. Files changed (Builder may only edit these)

- `code/baseline/losses.py` — same edits as design001 (add Sobel helper, add `jnd_beta`, weighted YUV) **plus** a new `pixel_w` constructor argument (default 0.0; this design sets 0.5) and a new `l_pixel = (w * (stego - cover).abs()).mean()` term added to `l_quality`.
- `code/baseline/exp0_inn_train.py` — pass `jnd_beta=4.0, pixel_w=0.5` to `ImageSecretLoss(...)`; add `l_pixel`, `l_yuv_unweighted`, `jnd_beta`, `pixel_w` to the per-step `metrics.jsonl` record.

Untouched (locked): everything in `infra/**`, `code/baseline/inn_model.py`, `code/baseline/train.py`.

## 3. CLI flag values for `baseline/exp0_inn_train.py` (config)

Locked (must NOT change): `--batch_size 16`, `--secret_len 896`, `--resolution 256`, `--max_steps 30000`, `--total_steps 30000`.

Tuned (identical to design003 parent):
- `--num_blocks 16`
- `--subnet_type rdb`
- `--lr 2e-4`
- `--img_weight 1.0`
- `--bit_weight 20.0`
- `--stage_a_steps 5000`
- `--stage_b_steps 10000`

No new CLI flags; `jnd_beta=4.0`, `pixel_w=0.5`, `jnd_mode="sobel"` are hard-coded inside `ImageSecretLoss(...)`.

## 4. Algorithmic changes (concrete)

### 4.1 `code/baseline/losses.py`

**Add** the `sobel_jnd_mask(cover)` helper exactly as specified in design001 §4.1 (Sobel-X, Sobel-Y, sqrt-of-sum-of-squares, 5×5 Gaussian blur, per-image max-normalize, `torch.no_grad`, `.detach()`).

**Modify `ImageSecretLoss.__init__`** to accept `jnd_beta`, `pixel_w` (and optionally `jnd_mode` for symmetry; this design uses "sobel"):

```python
def __init__(self,
             secret_weight: float = 20.0,
             yuv_w: float = 1.5,
             lpips_w: float = 1.0,
             ffl_w: float = 1.5,
             jnd_beta: float = 0.0,
             pixel_w: float = 0.0,
             jnd_mode: str = "sobel"):
    super().__init__()
    assert jnd_mode in ("sobel", "variance"), f"unknown jnd_mode={jnd_mode}"
    self.secret_weight = secret_weight
    self.yuv_w = yuv_w
    self.lpips_w = lpips_w
    self.ffl_w = ffl_w
    self.jnd_beta = jnd_beta
    self.pixel_w = pixel_w
    self.jnd_mode = jnd_mode
    self.lpips = lpips_lib.LPIPS(net="alex", verbose=False)
    for p in self.lpips.parameters():
        p.requires_grad_(False)
    self.ffl = FocalFrequencyLoss()
```

**Modify `ImageSecretLoss.forward`**, replacing the `l_yuv = ...` line with:

```python
yuv_stego = rgb_to_yuv(stego)
yuv_cover = rgb_to_yuv(cover)
sq = (yuv_stego - yuv_cover) ** 2  # (B, 3, H, W)
l_yuv_unweighted = sq.mean()

if self.jnd_beta > 0.0:
    if self.jnd_mode == "sobel":
        M = sobel_jnd_mask(cover)
    else:
        M = variance_jnd_mask(cover, win=7)  # only used if mode==variance
    w = 1.0 / (1.0 + self.jnd_beta * M)      # (B, 1, H, W), broadcast over chans
    l_yuv = (w * sq).mean()
    if self.pixel_w > 0.0:
        # Reuse the same mask w; broadcast over RGB channels.
        l_pixel = (w * (stego - cover).abs()).mean()
    else:
        l_pixel = torch.zeros((), device=stego.device, dtype=stego.dtype)
else:
    l_yuv = l_yuv_unweighted
    if self.pixel_w > 0.0:
        # No mask: unweighted L1 (still allowed but unused here).
        l_pixel = (stego - cover).abs().mean()
    else:
        l_pixel = torch.zeros((), device=stego.device, dtype=stego.dtype)
```

Update the `l_quality` accumulation:
```python
l_quality = (self.yuv_w * l_yuv
             + self.lpips_w * l_lpips
             + self.ffl_w * l_ffl
             + self.pixel_w * l_pixel)
```

Update the returned `logs` dict:
```python
logs = dict(
    loss=total.detach(),
    l_bce=l_bce.detach(),
    l_yuv=l_yuv.detach(),
    l_yuv_unweighted=l_yuv_unweighted.detach(),
    l_lpips=l_lpips.detach(),
    l_ffl=l_ffl.detach(),
    l_pixel=l_pixel.detach(),
    l_quality=l_quality.detach(),
    alpha=alpha,
)
```

### 4.2 `code/baseline/exp0_inn_train.py`

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
).to(device)
```

In the per-step `metrics.jsonl` record, add:
```python
l_yuv_unweighted=float(loss_logs["l_yuv_unweighted"]),
l_pixel=float(loss_logs["l_pixel"]),
jnd_beta=4.0,
pixel_w=0.5,
jnd_mode="sobel",
```

No other change to `exp0_inn_train.py`.

## 5. Expected behavior after the change

- `metrics.jsonl` per-step records include `l_pixel` (≥ 0), `l_yuv_unweighted`, `jnd_beta=4.0`, `pixel_w=0.5`, `jnd_mode="sobel"`.
- `l_pixel` magnitude is small (covers and stego are in [-1,1]; typical residuals ≈ 1e-2 → `l_pixel` ≈ 5e-3 before mask). After mask + `pixel_w=0.5`, the contribution to `l_quality` is small but non-negligible.
- `bit_acc_clean = 1.000` at every eval resolution (constraint).
- Hypothesis on outcome: PSNR (and `score`) ≥ design001-of-idea003. The added L1 pushes residuals toward sparsity, which on its own would lower PSNR (L1 is more permissive of small dense residuals than L2), but the JND mask localizes those residuals to perceptually-tolerant regions; net effect is uncertain — that uncertainty is the experimental purpose.
- Wallclock: same as design001-of-idea003 (mask is computed once and reused; the extra `(stego-cover).abs().mean()` is negligible).

## 6. Constraints and invariants Builder must preserve

- No edits to `infra/**`. No edits to `code/baseline/inn_model.py` or `code/baseline/train.py`.
- Locked CLI flags untouched.
- All non-locked CLI flags identical to design003 parent.
- LPIPS off (`lpips_w=0.0`); short-circuit preserved.
- `secret_weight=20.0` preserved.
- The mask `M` MUST be computed exactly once per `forward` call and reused for both the YUV-weighted MSE and the L1 pixel term. Computing it twice is a correctness no-op but doubles the helper cost.
- The L1 term uses `(stego - cover).abs()` in **RGB pixel space** (not YUV). Builder MUST NOT switch this to a YUV-space L1 — the design intent is to add a pixel-domain spatial signal complementary to YUV (which is already pixel-domain but channel-mixed) and FFL (purely spectral).
- Mask is detached; same constraints as design001 §6.
- Optimizer, scheduler, AMP autocast scope, gradient clipping, curriculum α-ramp, validation cadence, checkpoint format unchanged.
- `bit_acc_clean = 1.000` must remain achievable at convergence.

## 7. Edge cases

- **Mask reuse with broadcast**: `w` shape is `(B, 1, H, W)`; `(stego - cover).abs()` shape is `(B, 3, H, W)`; the same per-pixel weight is applied to all 3 RGB channels. Intentional and consistent with the YUV broadcast.
- **`pixel_w=0` fallback path**: when `jnd_beta=0` and `pixel_w>0`, the code path computes an unweighted L1; this design exercises the masked branch (`jnd_beta=4.0, pixel_w=0.5`). Builder MUST NOT remove the unweighted fallback; it is a defensive code path for future ablations.
- **AMP / autocast**: same fp32-wrapped scope as design003 parent; no fp16 issues on `abs` or `sqrt`.
- **Loss scale interaction**: with `pixel_w=0.5` and typical `l_pixel ≈ 5e-3`, the L1 contribution to `l_quality` is `≈ 2.5e-3`, comparable to `1.5 * l_yuv ≈ 1.5 * 5e-4 ≈ 7.5e-4` and to `1.5 * l_ffl` (similar order). The L1 term is therefore *competitive*, not negligible nor dominant. If during smoke tests the Builder observes `l_pixel * pixel_w` exceeding `yuv_w * l_yuv` by more than 5×, flag to Designer (would suggest `pixel_w` is too high for this dataset).
- **Constant-luma cover**: Sobel mask is 0 everywhere → `w = 1` → `l_pixel` reduces to plain mean-abs L1, scaled by `pixel_w=0.5`. Behavior is well-defined.
- **Normalization differences**: `l_pixel` is a mean of absolute values in `[-1,1]` units; `l_yuv` is a mean of squared values in the same units; do not be alarmed that `l_pixel` is ~1e-2 while `l_yuv` is ~1e-4 — that is the L1-vs-MSE scale difference, and the relative contribution to `l_quality` is set by `pixel_w` vs `yuv_w` chosen above.
