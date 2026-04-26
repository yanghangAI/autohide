**Idea Name:** Diffuse bit-carrier basis to eliminate visible 32×32 patch grid in container images

**Approach:** Replace the `PatchBitAdapter`'s nearest-neighbor patch tile (which tiles each bit as a piecewise-constant ±1 block on a 32×32 grid, producing visible block artifacts whose physical size scales with image resolution) with a spatially diffuse, zero-parameter, orthogonal bit↔tensor basis so each bit's energy is spread across many spatial locations rather than concentrated in one contiguous patch.

**Expected Designs:** 4

**Suggested Parent:** baseline/

**Baseline Source:** baseline/ (only `Done` entry in `results.csv`; score=27.27, PSNR=27.33, bit_acc=0.999, bit_acc_clean=1.0). The adapter axis is orthogonal to ideas 001–004, so building from baseline isolates the bit-carrier change rather than entangling it with their loss/curriculum modifications.

**Relationship to prior work:** New axis. Ideas 001–004 modify the loss (idea001 perceptual rebalance, idea003 spatial JND masking) or the optimizer/curriculum (idea002 attack curriculum, idea004 EMA + STE clipping); none touch the bit↔tensor basis. The blocky carrier is upstream of all of those — it sets a floor on visibility regardless of how the loss/optimizer is tuned.

**Motivation from the data and code:**
- `baseline/inn_model.py:103-166` (`PatchBitAdapter`): bits → bipolar ±1 → fixed sign-mask scramble → `view(B,1,32,32)` → `repeat_interleave` to (h_dwt, w_dwt) → broadcast across 12 DWT channels. The `repeat_interleave` is a piecewise-constant nearest-neighbor tile: at 256×256 RGB, DWT space is 128×128, each grid cell is 4×4 DWT samples = an 8×8 RGB block carrying one constant ±1.
- The baseline already saturates `bit_acc≈0.999` at 256×256 (only `chain_wechat` dips), so headroom is on the **PSNR / visibility** side. A piecewise-constant carrier is the worst case for visibility: all the bit energy lives at the patch-boundary discontinuities (high-frequency edges aligned to a periodic 8-px grid), exactly the kind of structured artifact human vision is most sensitive to.
- User confirmed the 32×32 grid is visible in current container images, with per-patch size scaling with image resolution — i.e., this is an observed defect, not a theoretical one.
- All four proposed designs keep the adapter zero-parameter and preserve exact invertibility on the clean path, so the `bit_acc_clean = 1.000` hard condition is structurally protected.

**Suggested design directions (Designer to refine):**
The four designs form a 2×2 cross of {minimal-edit vs. principled-basis} × {scramble in space vs. spread over a basis}.

1. **Random orthogonal pixel-permutation (minimal × scramble).** Replace `repeat_interleave` with a fixed deterministic random permutation of the 16,384 DWT-spatial positions across the 1024 slots, balanced so each slot owns exactly 16 positions (one slot ↔ one bit, but its 16 positions are scattered, not contiguous). Decode unchanged: `adaptive_avg_pool2d` is replaced by averaging the 16 positions belonging to each slot. Kills the 32×32 spatial periodicity with a one-line code change and is the cheapest sanity check that "blockiness, not amplitude" was the visibility cause.
2. **Smooth-kernel splat (minimal × spread).** Tile as today, then convolve the resulting 12-channel carrier with a fixed low-pass kernel (e.g. Gaussian σ ≈ 2 DWT-px, kernel ≈ patch size) before handing to the INN; the decode path applies the matched kernel before the 32×32 mean-pool so the per-bit projection stays exact on clean. Removes block edges by construction without changing the per-bit ownership of spatial regions.
3. **Walsh–Hadamard 1024-slot spreading (principled × scramble).** Replace the bit→grid step with `(B, 1024) → matmul(H_1024, sign_masked_bits) → reshape to (32, 32) → upsample to DWT plane`. Every output pixel is now a sum of all 1024 bits (orthogonal Hadamard rows), so no bit lives in a single block. Decode: pool to 32×32, then matmul by `H_1024^T / 1024` to invert. Exactly orthogonal, zero-parameter, exactly invertible on clean.
4. **DCT mid-band spreading (principled × spread).** Each bit ↔ one 2D-DCT basis function on a 32×32 frequency grid, restricted to the mid-band (exclude DC and the highest spatial frequencies). Carrier = sum of bit-weighted basis functions, upsampled to DWT plane. JPEG-friendly by construction (energy concentrated where JPEG's quantization tables preserve coefficients), naturally band-limited (no aliasing on resize attacks). Decode: 32×32 DCT of the pooled recovery, then read off the 896 selected mid-band coefficients.

**Constraints:**
- Parent is `baseline/` so designs do not inherit the loss/curriculum changes from ideas 001–004 (we want the adapter axis isolated). Designer may modify `baseline/inn_model.py` (the `PatchBitAdapter` class and its callers in `INNCodec.embed`/`extract_with_secret`) and may make small follow-on changes in `baseline/exp0_inn_train.py` if needed (e.g., logging the new diagnostic).
- Locked hyperparameters unchanged (`--batch_size 16`, `--secret_len 896`, `--resolution 256`, `--max_steps 30000`); no edits to any `infra/` file. The eval suite (13 attacks × 9 resolutions in `infra/exp0_inn_eval.py`) is untouched, so resolution-scaling behavior at inference can be read directly off the existing report.
- Hard condition: `bit_acc_clean = 1.000` at every eval resolution must hold. All four designs are structurally exactly-invertible on the clean path (orthogonal basis or balanced averaging), so the Designer should verify this analytically when picking the basis matrix and add an assertion at training step 0 that `extract(embed(host, bits)) == bits` on a noiseless batch.
- Walltime: designs 1, 2, 4 add negligible compute (≤1% of step time). Design 3 (Hadamard) adds one 1024×1024 matmul per forward and per reverse — also negligible vs. the INN. All four designs should comfortably fit the ~12 h SLURM budget for 30k steps on a 2080Ti.
- The new adapter must remain a `nn.Module` with persistent buffers (basis matrices, permutation indices, kernels) so checkpoints round-trip without depending on RNG seeds at load time.
