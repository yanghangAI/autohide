**Idea Name:** Worst-case-aware attack curriculum to lift the bit_acc tail

**Approach:** Restructure the training-time `NoiseCurriculum` / `AttackLayer` interaction in `baseline/exp0_inn_train.py` so the encoder is exposed to attack settings that match (and at times exceed) the eval suite's worst-case chains — chiefly the `chain_wechat` regime (resize scale=0.3, JPEG q=50) and aggressive resize + double-JPEG combinations — instead of the current uniform-Bernoulli sampling that rarely co-fires resize-low + JPEG-low together.

**Expected Designs:** 3

**Suggested Parent:** baseline/

**Baseline Source:** baseline/ (only `Done` entry in `results.csv`; score=27.27, PSNR=27.33, bit_acc=0.999, bit_acc_clean=1.0)

**Relationship to prior work:** New axis. idea001 attacks `score` from the PSNR side (perceptual loss reweighting on a saturated bit_acc baseline). This idea attacks `score` from the bit_acc side by closing the train/eval attack-distribution gap. The two axes are orthogonal and could compose later.

**Motivation from the data:**
- Baseline bit_acc per attack at 256² (from results / val): clean=1.000 and 11/13 attacks ≥0.999, but `chain_wechat=0.988` is a clear outlier and `chain_whatsapp_std`/`chain_x` sit slightly below 1.0 as well. The dominant residual loss is on the high-compression chained attacks.
- `infra/exp0_inn_eval.py` evaluates `chain_wechat = scale=0.3, q=50` and `chain_whatsapp_std = scale=0.5, q=70`. Both are *deterministic* worst-case operating points.
- The training-time `AttackLayer` uses `severity="high"`: `jpeg_q_min=40, jpeg_p=0.9`, `chain_p=0.3` (probability the second JPEG fires), `resize_scale_min=0.5, resize_p=0.5`. The expected fraction of training batches that simultaneously see (resize at scale ≤0.6) AND (jpeg1 at q ≤50) AND (jpeg2 firing at q ≤40) is roughly 0.5·0.2·0.3·0.3 ≈ 0.9% — i.e. the encoder sees the wechat-equivalent regime in ~1 of 100 batches. This explains the 0.988 floor: the network is undertrained on its hardest deployment attack.
- Score sensitivity: at PSNR≈27.3, lifting `bit_acc_mean` from 0.9988 to 0.9998 is worth ≈ +0.054 score; lifting the wechat tail from 0.988 to 0.999 (which would also pull up `bit_acc_min` and unblock further bit pressure trade-offs) is worth ~+0.6 score on its own. Closing the train/eval distribution gap is the highest-leverage non-PSNR lever.

**Suggested design directions (Designer to refine):**
1. **Eval-mirrored hard chains in training.** Add an explicit "hard chain" branch to `NoiseCurriculum.attack(step)` that, with some probability `p_hard(α)` ramped in alongside the existing α-ramp, replaces the random `AttackLayer` sample with a deterministic chain at one of the eval operating points (e.g. uniformly over `{scale=0.5, q=70}`, `{scale=0.5, q=85}`, `{scale=0.3, q=50}`). This gives the encoder direct gradient on the exact eval attacks without modifying any `infra/` file (the new chain is constructed from the same kornia primitives inside the training script).
2. **Severity-annealed curriculum.** Replace the single global α with a two-axis schedule: α₁ ramps the firing probability (clean → full) as today, and α₂ ramps the *severity* (jpeg_q_min from 80 → 40, resize_scale_min from 0.9 → 0.3). Encoder first learns to embed against light compression, then progressively against worst-case settings — analogous to learning-rate warmup but for attack difficulty.
3. **Per-attack hard-example mining.** During training, periodically (e.g. every K steps) score each chain in a small fixed bank by current `bit_acc` on a held-out batch, and bias the per-step attack sampler toward the worst-performing chain (softmax-temperature reweighting). Concentrates compute on whichever attack is currently the bottleneck — initially wechat, possibly something else later.

**Constraints:**
- Parent is `baseline/`; designs may modify `baseline/exp0_inn_train.py` (the curriculum/loop) and may construct new attack chains *in the training script* using `kornia` / `infra.attacks` primitives, but **must not modify any file under `infra/`** (locked).
- Locked hyperparameters unchanged: `--batch_size 16`, `--secret_len 896`, `--resolution 256`, `--max_steps 30000`, `--total_steps 30000`. The eval attack suite (13 attacks × 9 resolutions) is fixed.
- Hard condition `bit_acc_clean = 1.000` at every eval resolution must hold. The design must therefore preserve the early clean-α=0 phase (or equivalent) so the secret pathway is established before any attack pressure; otherwise extra-hard attacks early can collapse the secret pathway and tank clean bit_acc.
- Loss formulation stays at the baseline default (pure-L1 host + BCE bits). This isolates the attack-curriculum axis from idea001's loss-rebalancing axis so the two are independently measurable. Designer should not also swap in `ImageSecretLoss`.
- `--bit_weight=20.0` retained as the default; designs may not raise it as a workaround for a poorly-tuned curriculum (would confound this axis with bit-pressure tuning).
