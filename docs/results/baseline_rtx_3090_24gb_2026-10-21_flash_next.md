# Baseline — Qwen3.8-Flash-Next, RTX 3090 24 GB, 2026-10-21

One run of the Flash-Next width gate on build `fa230fec3` (main, tracked tree
clean), RTX 3090 24 GB (driver 560.94), 64 GB RAM, motherd stopped. Experts are
**Q3_KO** (`Qwen3.8-Flash-Next-Q3_KOEXP-503c2f2d6ead.precision.pack.gguf`), the
floor for every card under 64 GiB since `6c3b34f24` (2026-10-11). Speculation is
on, at the VRAM-blended draft ceilings (24 GB row: 4 at widths 1–5, 3 at 6–15, 2
at 16–32). Every session in every row validated; the gate passed in 180 s.
The rows are in `performance_rtx_3090_24gb_rows_2026-10-21_flash_next.tsv`.

```
cargo test --release --features cuda -p candle-transformers --lib \
  quantized_qwen38_moe::tests::test_parallel_batched_forwarding \
  -- --ignored --nocapture --test-threads=1
```

| KV mode | Contexts | Prefill t/s | Decode t/s | Compress | Peak tokens | Accepted / step |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 1 (cold) | 276.4 | 71.9 | – | 905 | 4.90 |
| BF16 | 1 (warm) | 319.6 | 93.4 | – | 905 | 4.90 |
| BF16 | 4 | 958.0 | 238.3 | – | 3,662 | 19.62 |
| BF16 | 8 | 1,398.8 | 326.4 | – | 7,284 | 31.88 |
| BF16 | 16 | 1,855.0 | 342.5 | – | 14,554 | 48.00 |
| C0 | 2 | 593.1 | 163.6 | 2.20× | 1,850 | 9.81 |
| C5 | 2 | 595.2 | 157.4 | 3.81× | 1,850 | 9.81 |
| C5 | 8 | 1,511.6 | 309.3 | 3.81× | 7,284 | 31.88 |
| C8 | 2 | 576.1 | 132.3 | 4.91× | 1,850 | 9.81 |
| C10 | 2 | 577.6 | 106.8 | 5.94× | 1,466 | 9.69 |
| C10 | 8 | 1,419.9 | 196.5 | 5.90× | 5,748 | 31.50 |

"Accepted / step" is the cohort's tokens committed per speculative step (about
4.9 a session at a ceiling of 4).

## Against the previous Q3_KO run (2026-10-18, `50c65b683`)

| Row | 2026-10-18 prefill / decode | 2026-10-21 | Prefill | Decode |
|---|---|---|---:|---:|
| BF16 ×1 (warm) | 293.8 / 66.5 | 319.6 / 93.4 | +9% | +40% |
| BF16 ×4 | 882 / 144 | 958.0 / 238.3 | +9% | +65% |
| BF16 ×8 | 1,331 / 168 | 1,398.8 / 326.4 | +5% | +94% |
| BF16 ×16 | 1,792 / 250 | 1,855.0 / 342.5 | +4% | +37% |
| C5 ×8 | 1,341 / 210 | 1,511.6 / 309.3 | +13% | +47% |
| C10 ×8 | 1,343 / 144 | 1,419.9 / 196.5 | +6% | +36% |

The 2026-10-18 figures were recorded outside this directory (that run's notes);
whether it drafted is not recorded. The decode gain is consistent with this
run's speculation (about 4.9 accepted tokens a session a step at narrow
widths), while prefill moved only 4–13%; it is not isolated by experiment.

## Against the published Q2_KO figures

`docs/performance.md` I17/I18 and `sweep_rtx_3090_24gb_2026-10-10.md` were
measured with **Q2_KO** experts (about twice Q3_KO's decode, with about a third
of the signal lost), so they are not like for like:

| Row | Published (Q2_KO) | 2026-10-21 (Q3_KO) | Share |
|---|---:|---:|---:|
| decode ×1 | 138.1 | 93.4 | 68% |
| decode ×4 | 320.5 | 238.3 | 74% |
| decode ×16 | 543.8 | 342.5 | 63% |
| prefill ×16 | 2,301.9 | 1,855.0 | 81% |

## Notes

- **Scaling flattens past ×8:** ×16 decodes 5% more than ×8. The draft
  ceiling drops from 3 to 2 at that width, so each step proposes less.
- **KV compression costs in step with its ratio:** at ×8, C5 (3.81×) decodes
  5% under BF16 and C10 (5.90×) 40% under.
- **C10 ran short.** At ×2 it took 13 steps against 52 for every other mode, and
  at ×8 16 against 64; its peak tokens are 1,466 and 5,748 against 1,850 and
  7,284. Its sessions validated, so the rewrites matched, but why the run ended
  early is not yet explained, and its rates cover a shorter decode than the
  other rows.
- Single run: a gap under about 5% is noise.
