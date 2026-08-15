# perf-investigation/

Frozen historical measurement captures from a prefill-kernel investigation.
**Not live tooling** — nothing here is regenerated automatically, and the
kernel source snapshot no longer matches the live tree. Treat every file as a
point-in-time record, not a script or fixture to run against.

## What was measured

All six `.txt` files are raw PowerShell transcripts of the same test —
`candle_transformers::models::quantized_llama::tests::test_parallel_batched_forwarding_llama3`
(the `batch_test` story-rewrite harness, see
`candle-transformers/src/models/batch_test/README.md`) against
`VibeStudio/Nidum-Llama-3.2-3B-Uncensored-GGUF` — run three times each before
and after a prefill-kernel change. Each run prints a comparison table across
KV storage modes (`F32`, `F16`, `R16`, `C0`, `C1`) reporting `t/s` (bulk and
single-session), throughput delta vs. the `F32` baseline, `%Quantized`, and
compression ratio. The files are UTF-16-encoded PowerShell transcript output
(`Get-Content -Encoding Unicode` reads them correctly; a plain UTF-8 read
renders the double-byte padding as spaced-out characters).

## File naming

| Pattern | Meaning |
|---|---|
| `baseline_run{1,2,3}.txt` | Three repeat runs against the prefill kernel *before* the change under investigation. |
| `60353239_newprefill_run{1,2,3}.txt` | Three repeat runs *after* the change — recompiles `candle-kernels` (visible in the transcript) before running the same test. `60353239` identifies the specific build/job this snapshot was captured from; it is not a path or flag reproducible today. |
| `paged_prefill_kernel.current.cuh` | A snapshot of the paged-prefill attention kernel source (`.cuh`) as it stood for the "newprefill" runs — a Flash-Attention-style paged-KV kernel with a feature-comparison table against FA2/FA3/FlashInfer/vLLM/cuDNN in its header comment. |

The FP16 paged-prefill kernel this snapshot documents was later removed from
the live tree (per project history, 2026-07-12); today's live prefill
kernels live under `candle-kernels/src/paged-prefill/`
(`paged_prefill_int8_kernel.cuh` and friends — the INT8 path specified in
`docs/archived/prefill_optimization.md`). `paged_prefill_kernel.current.cuh`
does not correspond to any file currently in `candle-kernels/src/`; it is
kept only as the historical record of what was measured.

## Related docs

`docs/archived/prefill_optimization.md` (the INT8 prefill kernel design that
superseded the FP16 kernel captured here),
`candle-transformers/src/models/batch_test/README.md` (the test harness
these transcripts are output from).
