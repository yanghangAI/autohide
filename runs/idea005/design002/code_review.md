## design002

**Verdict:** APPROVED

**Automated gate:** PASS (structural check passed; check-scope PASS against parent baseline/; verify-claims PASS on 3 snippets)

**Algorithm fidelity:**
- §4.1 Gaussian builder is implemented verbatim at `code/inn_model.py:103-115` (`_make_gaussian_kernel2d`), with the same odd-ksize check, normalization, and outer-product construction prescribed in design.md.
- §4.2 `PatchBitAdapter` matches the spec at `code/inn_model.py:118-182`:
  - Class constants `KERNEL_SIGMA = 2.0`, `KERNEL_SIZE = 9` (lines 131-132).
  - `smooth_kernel` registered as a persistent buffer with shape `(1, 1, 9, 9)` (lines 143-146).
  - `_smooth` does reflect-pad then depthwise `F.conv2d` with `groups=C` and a per-channel-expanded kernel (lines 148-155) — matches the depthwise-conv invariant in §7.6.
  - `encode` keeps the original tile path (lines 157-173), then applies `self._smooth(tiled12)`.
  - `decode` applies the matched filter first, then `adaptive_avg_pool2d` to (32,32), then channel mean, then sign-mask unscramble (lines 175-182). Order matches §4.2 exactly.
- §5 logging: `adapter_kind="smooth_splat"` present at `code/exp0_inn_train.py:423`; `metrics.jsonl` lines confirm the key is emitted.
- §4.4 step-0 sign assertion: deliberately omitted by Builder. `implementation_summary.md` documents this; under the relaxed project policy where `bit_acc_clean` is a soft signal, this is acceptable.

**Training-signal sanity:**
- Gradient flow: `_smooth` is a pure `F.conv2d` with reflect-pad — fully differentiable, no `.detach()` or `torch.no_grad()` wrap on the forward path. `smooth_kernel` is a buffer (no grad needed; deterministic).
- Encoder/decoder filter consistency: both call `self._smooth`, which uses the same registered buffer and the same `groups=C, expand(C,1,K,K)` per-channel weight construction — kernels are identical (matched filter property holds).
- Shape correctness: encode produces `(B, 12, h_dwt, w_dwt)`; decode consumes the same; reflect-pad adds 4 on each side and stride-1 conv removes it. Verified by mini-train running 200 steps without shape errors.
- Mode/mask: `sign_mask` correctly applied symmetrically (encode multiplies by `sign_mask` before tile; decode multiplies after pool). Sign mask is its own inverse (±1), so the symmetric application unscrambles cleanly.
- Optimizer / LR / α-ramp: `code/exp0_inn_train.py` is otherwise unchanged from baseline; the relevant only-training-touch is the new `adapter_kind` kwarg in the `rec` dict — no schedule, loss, or optimizer interaction.
- Mini-train evidence (`test_output/metrics.jsonl`, 200 steps): `bit_acc_ema` rises 0.585 → 0.681 → 0.760 → 0.806 across the four logged steps; `bit_loss` is non-trivial (0.36-0.41) and `secret_rec_err_rms` is well-defined (not NaN) — gradients are flowing through the smoothed adapter and through the INN.
- Per design §6, `bit_acc_clean = 1.000` is no longer guaranteed exactly because the matched-Gaussian self-convolution is not idempotent on the 4×4 cell average; per the relaxed policy this is allowed and is captured in the standard reported metric.

**Strongest objection (ruled out):** that the matched-Gaussian double-pass would not preserve sign on the clean path, breaking `bit_acc_clean` enough to corrupt the optimization. Ruled out because (a) project policy treats `bit_acc_clean` as a soft signal, (b) sign of each slot's pooled logit is dominated by the slot's own ±1 contribution (kernel sums to 1, mean-pool over 4×4 cell preserves DC of the smoothed constant up to small reflect-pad boundary error), and (c) the mini-train shows bit_acc trending upward — the training signal is alive and informative.

**Fixes required:** none.
