# Baseline — RTX PRO 5000 Blackwell 72 GB, 2026-10-03

One run of each width gate on build `91fb4da8`: the hot-path invariant fixes,
the BMMA (head-dim 128) and tile (head-dim 256) decode-kernel register-spill
fixes, and the one-fence-per-cohort seal sync. Single runs, so a gap under ~5%
against `docs/performance.md` §3.7 (the best of three sweeps) is noise.

‡ = the gate ran on this build before the seal-sync fence was restored, which
changes only the compressed (C*n*) rows.

| Model | ctx=1 prefill / decode | widest measured | prefill / decode |
|---|---|---|---|
| Qwen2-0.5B | 31,324.4 / 248.8 | ×60 (F16) | 93,581.1 / 4,549.9 |
| Qwen3.5-0.8B | 23,487.5 / 151.2 | ×256 (C8) | 37,947.8 / 3,122.7 |
| Llama-3.2-3B ‡ | 13,085.6 / 159.5 (C0) | ×10 (C8) | 15,451.3 / 978.4 |
| Qwen3-30B-A3B ‡ | 8,115.5 / 88.0 | ×20 (Q8_0) | 10,487.8 / 618.8 |
| Qwen3.5-35B-A3B | 7,175.4 / 116.1 | ×64 (C10) | 9,658.2 / 1,179.7 |
| Qwen3.6-35B-A3B | 6,978.1 / 112.3 | ×64 (C10) | 9,318.4 / 1,138.5 |
| Qwen3-8B ‡ | 6,234.6 / 82.9 | ×10 (C8) | 6,621.7 / 505.1 |
| Llama-2-7B ‡ | 6,159.9 / 136.9 | ×48 | 4,389.2 / 1,440.6 |
| Qwen3.5-9B | 5,556.9 / 123.7 | ×20 (C8) | 6,234.3 / 880.8 |
| Qwen3.8-27B | 1,796.9 / 59.9 | ×40 (C10) | 1,838.5 / 441.9 |
| Qwen3.8-Flash-Next | 2,347.7 / 111.4 (warm) | ×16 | 3,150.9 / 587.4 |
| DeepSeek-V4-Flash | not re-run on this build | | |

Every row validated, every session.

**Qwen3-30B-A3B engine probe** on the same build: story 20/20, weights fully
resident, worst sustained efficiency 85% — under its 90% gate; this probe's
efficiency has read 73–99% across builds (off-thread chunk frees).

## Against `docs/performance.md` §3.7

Above it: the head-dim-128 decode-kernel fix takes Llama-2-7B ×48 from 917.3 to
1,440.6, Llama-3.2-3B C8 ×10 from 745.1 to 978.4 and Qwen3-8B from 67.3 to 82.9
at one context; Flash-Next ×16 decodes 587.4 against 421.5; Qwen3.5-35B ×64 is
level (1,179.7 against 1,187.7).

Still under it:

- Qwen3.5-0.8B decode — 151.2 / 3,122.7 against 168.8 / 3,353.4.
- Qwen2-0.5B ×60 — 4,549.9 against 4,924.7; the day's runs on unchanged code
  spread 4,488–4,773.
- Qwen3.6-35B ×64 — 1,138.5 against 1,201.6, where its twin Qwen3.5-35B reaches
  1,179.7 in the same run.
- Qwen3.8-27B ×40 — 441.9 against 458.1.
