## design002
**Verdict:** APPROVED

**Feasibility evidence:** Parent `runs/idea001/design003/code/losses.py`:
- `rgb_to_yuv` at `losses.py:25` is reused inside the new `variance_jnd_mask`.
- `ImageSecretLoss.__init__` at `losses.py:38` with attrs `self.yuv_w/lpips_w/ffl_w/lpips/ffl` at `:45-51` — extending signature with `jnd_beta=0.0, jnd_mode="variance"` is backward compatible.
- Replacement target `l_yuv = F.mse_loss(rgb_to_yuv(stego), rgb_to_yuv(cover))` at `losses.py:60` exists; LPIPS short-circuit at `:61-65` and `l_quality` accumulation at `:68-70` preserved.
- The `E[X²] - E[X]²` recipe via two single-channel uniform-kernel `F.conv2d` calls with `padding=3` on `(B,1,H,W)` luma is mechanically straightforward; `clamp_min(0.0)` followed by `sqrt(... + 1e-12)` is the standard numerically-safe variant.
- `exp0_inn_train.py:258` `img_loss_mod = ImageSecretLoss(...)` accepts the new kwargs; `rec = dict(...)` at `:439` with `l_yuv=loss_logs["l_yuv"].item()` at `:447` accepts the additional logged fields.
- fp32 autocast wrapper at `exp0_inn_train.py:392-400` (`cover_fp32 = cover.float()` and `with torch.amp.autocast("cuda", enabled=False):`) matches the design's claim that variance/sqrt run in fp32.

**Idea contradiction check:** Idea direction (2) prescribes variance-pooled local std-dev JND mask over a 7×7 window, applied only to YUV term, FFL unchanged, β≤4.0, per-image normalization, mask detached, LPIPS off, `secret_weight=20.0`. Design matches exactly: `win=7`, `jnd_beta=4.0`, `jnd_mode="variance"`, `@torch.no_grad` + `.detach()`, per-image `flat_max` normalize, FFL untouched, `lpips_w=0.0`.

**Strongest objection:** Constant-luma cover degenerate case — design002's behavior diverges from design001's: where Sobel yields `M≈0 → w=1`, variance yields `std≈1e-6` everywhere, then `flat_max` normalization makes `M≈1` everywhere → `w=0.2` everywhere → uniform 5× down-weighting of the YUV term on that batch element. Could this destabilize training? Ruled out: (a) the design explicitly addresses this in §7, (b) constant-luma covers are vanishingly rare in COCO, (c) only the *relative* weighting drives gradients on textured covers in the same batch, and (d) `bit_acc_clean=1.000` is preserved regardless because the secret pathway is independent.
