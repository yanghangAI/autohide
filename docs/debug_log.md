# Debug Log

Use this file to record unexpected automation or execution issues fixed by the Debugger.

## 2026-04-26 — slurm_test.sh leaves checkpoint in test_output on success
- **Reported by:** user
- **Affects:** runs/idea001/design001, design002, design003 (all test_output/ dirs)
- **Problem:** After a successful smoke-test job, `<design>/test_output/ckpt_latest.pt` was left behind, wasting disk. The script header comment says checkpoints are kept "on failure", implying they should be cleaned on success.
- **Root cause:** `scripts/slurm/slurm_test.sh` had no cleanup branch for the success path; only the failure path wrote `training_failed.txt` and exited.
- **Files changed:**
  - `scripts/slurm/slurm_test.sh` — after the failure check, remove `$TEST_OUT/ckpt_latest.pt` (and any `ckpt_*.pt`) before printing success.
  - Removed stale `ckpt_latest.pt` from `runs/idea001/design00{1,2,3}/test_output/`.
- **Retry:** No re-run needed for already-tested designs; future `submit-test` invocations will clean up automatically.
