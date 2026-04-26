## design003
**Verdict:** APPROVED

**Feasibility evidence:** Parent `runs/idea001/design003/code/losses.py`:
- `rgb_to_yuv` at `losses.py:25` and `class ImageSecretLoss` at `:30` with `__init__` at `:38`; `self.yuv_w/lpips_w/ffl_w` at `:45-47`, `self.lpips/self.ffl` at `:48-51` — extending the signature with `jnd_beta=0.0, pixel_w=0.0, jnd_mode="sobel"` is backward compatible.
- Target line `l_yuv = F.mse_loss(rgb_to_yuv(stego), rgb_to_yuv(cover))` at `losses.py:60` exists; LPIPS short-circuit at `:61-65` is preserved; the new `l_pixel = (w * (stego-cover).abs()).mean()` is differentiable wrt `stego` (the only model output here) — gradient flows through `(stego - cover).abs()` since `w` is detached.
- `l_quality` accumulation at `losses.py:68-70` is the exact site for adding `+ self.pixel_w * l_pixel`; `total = self.secret_weight * l_bce + alpha * l_quality` at `:71` keeps the new term inside the curriculum α-ramp.
- `logs = dict(...)` at `:76-79` accepts the new `l_pixel` and `l_yuv_unweighted` fields.
- `exp0_inn_train.py:258` `img_loss_mod = ImageSecretLoss(...)` call site accepts the new kwargs; `rec = dict(...)` at `:439` accepts `l_pixel`, `l_yuv_unweighted`, `jnd_beta`, `pixel_w`, `jnd_mode`.
- fp32 autocast wrapper at `:392-400` (`container_fp32 = container.float()`, `cover_fp32 = cover.float()`) ensures `(stego - cover).abs()` and the Sobel mask both run in fp32.
- Mask reuse: `M` is computed once per `forward` and the same `w` is broadcast over both `sq` (B,3,H,W) and `(stego-cover).abs()` (B,3,H,W) — broadcast over the leading channel-1 axis is well-defined.

**Idea contradiction check:** Idea direction (3) prescribes Sobel-mask JND on YUV term **and** a per-pixel L1 residual term `mean(w · |stego - cover|)` added with small weight (~0.5) — a "joint mask on YUV + per-pixel FFL surrogate". Design matches: `jnd_beta=4.0, jnd_mode="sobel"`, `pixel_w=0.5`, RGB-space L1 (idea text uses `|stego - cover|` without specifying YUV; design's §6 deliberately keeps it RGB to provide a complementary spatial-domain signal distinct from YUV-mixed MSE), FFL unchanged, LPIPS off, `secret_weight=20.0`.

**Strongest objection:** Adding L1 alongside YUV-MSE could double-penalize the same residual in pixel space and over-constrain the encoder, lowering PSNR rather than raising it. Ruled out as a *feasibility* concern (the design's own hypothesis acknowledges the outcome is uncertain — that uncertainty is the experimental purpose); the loss is well-formed, gradients flow, `bit_acc_clean=1.000` is preserved by the unchanged secret pathway, and `pixel_w=0.5` keeps the L1 contribution at roughly the same order as `yuv_w * l_yuv` (per design §7's loss-scale analysis). The design correctly instructs the Builder to flag if `l_pixel * pixel_w` exceeds `yuv_w * l_yuv` by >5×.
