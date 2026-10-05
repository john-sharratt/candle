# Baseline for the decode-graph work — RTX 3090 24 GB, 2026-10-05

The performance bar `docs/decode_graphs.md` is measured against: the two gates that
implementation is brought up on, run on build `86c91146a` with the harness changes listed
below and **no graph code**. A graph build must pass both gates with every row validated
and be at least this fast row for row.

Machine: RTX 3090 24 GB (sm_86), i7-10700K, 64 GB RAM, PCIe 3.0 ×16. zend and npcd stopped,
card to itself.

## What changed in the gates to measure this

- **Decode length 256 tokens per session** (was 64 on Flash-Next, 10 on Qwen3.6). A wide
  prefill concedes the expert zone — to 11 GB at ×8 and 5 GB at ×16 on Flash-Next — and the
  decode behind it rebuilds its working set over its first steps. At 64 tokens (13
  speculative steps) that recovery was about a third of the decode, so the rate measured
  the pivot from prefill, not the decode.
- **C9 and C10 rows still decode 64 tokens** (`TestParams::with_top_rung_tokens`). They are
  the calibration probes, tuned just under their edge at that length; at 256 tokens C10×8
  failed on both models and C9×2 on Qwen3.6, on near-ties past the old window. The
  threshold rows were left as they are.
- **Expert pipeline stats are split at the prefill → decode boundary**, one table per
  phase, and each row prints its decode step times (`decode steps: first … | rest …`).
- **Pronoun validator:** `his`, `him` and `her` share one placeholder. A female session
  correctly wrote "pinned her to the seat" for "pinned him to the seat" and the old
  separate `[him/her]` failed it; the old 10/64-token windows never reached that word.

## Qwen3.6-35B-A3B — `quantized_qwen36_moe::tests::test_parallel_batched_forwarding_36_35b`

`Int8Mode::auto` (Precision). 70 s. Every row validated.

| mode | ctx | prefill t/s | decode t/s | compression | decode step (first / mean rest) |
|---|---:|---:|---:|---:|---|
| BF16 | 1 | 1001.8 | 119.7 | — | 41.4 / 24.5 ms |
| BF16 | 4 | 4187.3 | 513.3 | — | 15.2 / 23.2 ms |
| Q8_0 | 1 | 2692.6 | 153.8 | 1.88× | 18.1 / 19.3 ms |
| C0 | 1 | 2889.9 | 153.4 | 2.21× | 13.4 / 19.4 ms |
| C1 | 1 | 3006.8 | 150.4 | 2.66× | 37.7 / 19.5 ms |
| C2 | 1 | 3009.5 | 153.4 | 3.11× | 13.4 / 19.4 ms |
| C3 | 1 | 2983.6 | 153.1 | 3.34× | 13.5 / 19.4 ms |
| C4 | 1 | 3018.3 | 151.6 | 3.58× | 13.3 / 19.6 ms |
| C5 | 1 | 2992.8 | 152.4 | 3.91× | 13.4 / 19.5 ms |
| C5 | 8 | 4424.3 | 666.3 | 3.91× | 18.2 / 34.6 ms |
| C6 | 1 | 3035.5 | 148.4 | 4.40× | 13.5 / 20.1 ms |
| C7 | 1 | 3363.0 | 150.1 | 4.62× | 13.5 / 19.8 ms |
| C8 | 5 | 4858.7 | 570.8 | 5.24× | 15.9 / 26.1 ms |
| C9 | 2 | 4001.7 | 258.3 | 5.83× | 14.1 / 21.5 ms (64 tok) |
| C10 | 8 | 4595.0 | 627.0 | 6.43× | 19.9 / 34.1 ms (64 tok) |
| C10 | 16 | 4249.9 | 753.4 | 6.42× | 26.3 / 57.0 ms (64 tok) |

## Qwen3.8-Flash-Next, Q2_KO experts — `quantized_qwen38_moe::tests::test_parallel_batched_forwarding`

`Int8Mode::auto`. 148 s. Every row validated.

| mode | ctx | prefill t/s | decode t/s | compression | decode step (first / mean rest) |
|---|---:|---:|---:|---:|---|
| BF16 | 1 (cold) | 413.6 | 47.1 | — | 233.3 / 101.5 ms |
| BF16 | 4 | 1262.4 | 157.8 | — | 131.2 / 121.8 ms |
| BF16 | 8 | 1510.9 | 224.9 | — | 254.0 / 169.5 ms |
| BF16 | 16 | 1577.8 | 240.6 | — | 444.5 / 194.2 ms |
| BF16 | 1 (warm) | 491.7 | 71.1 | — | 103.7 / 68.3 ms |
| C0 | 2 | 894.1 | 138.2 | 2.19× | 83.8 / 70.7 ms |
| C5 | 2 | 898.1 | 137.2 | 3.80× | 78.2 / 71.3 ms |
| C5 | 8 | 1499.1 | 228.3 | 3.79× | 204.2 / 167.9 ms |
| C8 | 2 | 893.4 | 133.4 | 4.86× | 103.2 / 72.9 ms |
| C10 | 2 | 892.0 | 130.1 | 5.83× | 81.4 / 73.9 ms (64 tok) |
| C10 | 8 | 1521.4 | 209.2 | 5.79× | 186.4 / 171.0 ms (64 tok) |

Single runs; a gap under ~5% is noise. The decode rate is the generate phase's emitted
tokens over its wall time, speculative decode on (Flash-Next MTP, Qwen3.6 drafter).

## Reproduce

```bash
cargo test --release --features cuda -p candle-transformers --lib \
  models::quantized_qwen36_moe::tests::test_parallel_batched_forwarding_36_35b \
  -- --exact --ignored --nocapture --test-threads=1
cargo test --release --features cuda -p candle-transformers --lib \
  models::quantized_qwen38_moe::tests::test_parallel_batched_forwarding \
  -- --exact --ignored --nocapture --test-threads=1
```
