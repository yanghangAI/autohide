## design001
**Verdict:** APPROVED

**Feasibility evidence:** Parent `runs/idea001/design003/code/losses.py` exposes everything the design references:
- `rgb_to_yuv` at `losses.py:25` — usable inside the new `sobel_jnd_mask` helper.
- `class ImageSecretLoss` at `losses.py:30`, `__init__` at `losses.py:38` (with `self.yuv_w`, `self.lpips_w`, `self.ffl_w` set at `:45-47`, `self.lpips` at `:48`, `self.ffl` at `:51`) — adding a `jnd_beta=0.0` kwarg is a backward-compatible signature extension.
- Target replacement line `l_yuv = F.mse_loss(rgb_to_yuv(stego), rgb_to_yuv(cover))` exists verbatim at `losses.py:60`; the LPIPS short-circuit at `:61-65` and `l_quality = self.yuv_w*l_yuv + ...` at `:68-70` are preserved by the design's surgical replacement.
- `logs = dict(...)` block with `l_yuv=l_yuv.detach()` at `losses.py:76-79` accepts the additional `l_yuv_unweighted` field.
- Parent `exp0_inn_train.py:258` `img_loss_mod = ImageSecretLoss(` matches the design's call-site target; the `cover_fp32` fp32 autocast block at `:392-400` confirms the JND mask runs in fp32 as the design claims; `rec = dict(...)` at `:439` with `l_yuv=loss_logs["l_yuv"].item()` at `:447` accepts the new `l_yuv_unweighted` and `jnd_beta` fields.
- Sobel/Gaussian convs: input `cover` shape `(B,3,H,W)` in `[-1,1]`; luma extracted from channel-0 of `rgb_to_yuv(cover)` then conv2d with `padding=1`/`padding=2` preserves `(B,1,H,W)` — broadcast against `sq` of shape `(B,3,H,W)` is well-defined.

**Idea contradiction check:** Idea direction (1) prescribes Sobel-gradient JND on YUV term, β≤4.0, mask from cover only under `no_grad`, per-image normalization, FFL unchanged, LPIPS off, `secret_weight=20.0`. Design matches exactly: β=4.0 (at the conservative ceiling), `@torch.no_grad()` decorator + `.detach()`, per-image `flat_max` normalization, FFL untouched, `lpips_w=0.0`, `secret_weight=args.bit_weight=20.0`.

**Strongest objection:** Could β=4.0 cause the encoder to dump enough residual into textured regions to break `bit_acc_clean=1.000` extraction in flat regions? Ruled out: the JND mask only reweights the *host loss*; the secret BCE loss is unweighted and `secret_weight=20.0` is preserved, so the bit pathway is unchanged. Worst case the encoder shifts residual energy spatially but the secret extractor still sees full-resolution stego and is trained against the same BCE — extraction quality is decoupled from the spatial weighting. Design also explicitly flags `bit_acc_clean` regression as a Designer concern, not a silent fix.
