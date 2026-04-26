## design003
**Verdict:** APPROVED

**Feasibility evidence:**
- `INNCodec.embed` (`baseline/inn_model.py:302`) and `INNCodec.extract` (line 313) exist with the exact signatures used in `update_weights` (§4.3). The probe call `model.embed(probe_cover, probe_secret).clamp(-1, 1)` mirrors the validation pattern at `baseline/exp0_inn_train.py:158`.
- `_val_load_to_size` is already imported at `baseline/exp0_inn_train.py:51` and used at line 155 — design003's probe-batch construction (`probe_cover = _val_load_to_size(val_paths[:n_probe], 256, 256).to(device)`) reuses that exact API.
- `val_paths` and the val-root scanning block already exist at `baseline/exp0_inn_train.py:313-330`, so the "after the val_paths block" insertion point in §4.4 is well-defined.
- Bank entry 0 (`"random"`, `self._attack`) is the existing `AttackLayer` at line 99; entries 1-4 are `_HardChain` instances built from `K.RandomJPEG` (used at `infra/attacks.py:113`). `torch.multinomial(self._bank_weights, num_samples=1)` operates on a 1D weight tensor, fully supported.
- Training-loop hook (§4.5, after `if global_step % args.ckpt_every == 0: save_ckpt()`) corresponds to line 457-458; `model.eval()` / `model.train()` toggling pattern matches `_run_validation` at lines 146-181.
- Checkpoint payload at lines 332-351 is mutable; adding `bank_weights` to the dict and restoring it (§7) is mechanically straightforward.

**Idea contradiction check:** Idea direction #3 prescribes "score each chain in a small fixed bank by current `bit_acc`...bias the per-step attack sampler toward the worst-performing chain (softmax-temperature reweighting)" with periodicity "every K steps." Design003 implements exactly this: K=500, τ=0.05, softmax over `(1 - bit_acc) / τ`, fixed bank with floor 0.05 to prevent starvation. Clean phase preserved (`a==0 → return None`; mining gated on `step >= curriculum.ramp_start`). `bit_weight=20.0` retained, no `ImageSecretLoss`, no infra/ edits, locked CLI unchanged. The bank includes a `(0.4, 75)` "chain_x" point — idea direction #3 does not list specific operating points (it specifies *adapting* sampling weights over a "small fixed bank"), so adding chain_x is elaboration, not contradiction. Note design001's constraint forbids adding chain_x to its three-point bank, but design003 is a different design with its own bank.

**Strongest objection (ruled out):** Could the mining loop concentrate weight on `hard_s0.3_q50` so heavily that the encoder forgets the easier eval attacks and `bit_acc_clean = 1.000` is lost? Two safeguards mitigate this: (a) the per-chain weight floor of 0.05 guarantees every chain (including `random`) gets ≥5% of post-α gradient, and the `random` entry contains the full Bernoulli mixture including clean-pass-through (when its child `p`s sample below threshold); (b) the clean phase (`step < ramp_start`) is preserved verbatim, establishing the secret pathway before any mining pressure. Risk to `bit_acc_clean = 1.000` is plausible but bounded by the 0.05 floor and the 5000-step clean head start.

**Fixes required:** None.
