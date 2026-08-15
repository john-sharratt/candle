# candle-transformers/src/models/batch_test/

Shared integration-test harness for batched inference: story-rewrite
KV-quantization validation, the RULER long-context benchmark, and the
Markov-expert-prediction routing-trace fixture.

## What it does

This module is compiled into `candle-transformers` (not a standalone crate)
and consumed from `#[cfg(test)] mod tests` blocks inside the quantized model
implementations — `quantized_llama.rs`, `quantized_qwen3.rs`,
`quantized_qwen2.rs`, `quantized_qwen3_moe.rs` — plus
`candle-conversation/examples/ruler_stream.rs`. It gives every model's test
suite the same three things: HF-Hub download helpers that survive flaky
networks, a validated-output batched-session test loop across KV
quantization formats, and a pure-Rust RULER task generator.

## Key modules / layout

| File | Role |
|---|---|
| `mod.rs` | `pub mod` re-exports only: `ruler_gen` (always compiled), `test_helpers` and `utils` (`#[cfg(test)]` only). |
| `utils.rs` | `TestParams`/`TestConfig`/`TestMode` — drives N parallel batched sessions through a shared prompt with per-session name substitution, then validates the divergent output per session and reports a `t/s` / compression comparison table across `KvMode`s (F32/F16/R16/C0…C9). |
| `test_helpers.rs` | `#[cfg(test)]`-only HF download plumbing: `hf_get`/`api`/`ResilientApi` (IPv4-only resolver + HTTP-Range resume, for networks where IPv6 TLS to the HF CDN resets mid-handshake), `load_hf_tokenizer`, `download_hf_gguf`, `open_gguf`. |
| `ruler_gen.rs` | Pure-Rust generator + in-process evaluator for four RULER task types (`RulerTask`): `niah_single_1`, `niah_multikey_2`, `vt` (variable tracing), `cwe` (common-word extraction). `RulerDataSource::Generated` builds samples procedurally; `RulerDataSource::Jsonl` loads canonical RULER-format JSONL. |
| `story.md` | The user-turn story text ("The Backyard Astronaut") used by `TestParams::new` as the default `prompt_user` — the body the model must rewrite per session. |
| `system.md` | The default `prompt_system`: instructs the model to deterministically replace every occurrence of "Marcus" (and case variants) with a per-session name, changing nothing else — this is the `StoryRewrite` `TestMode`. |
| `names.md` | 99 newline-separated first names, one assigned per test session, used both to build each session's rewrite target and to check that adjacent sessions produce genuinely distinct output (catches KV cross-contamination). |
| `fixtures/routing_trace_qwen3_30b.bin.gz` | Captured Qwen3-30B-A3B MoE expert-routing trace (`candle-transformers/src/models/routing_capture.rs::FIXTURE_PATH`), written by a focused capture test and read by the offline Markov-expert-prediction evaluator — see `docs/markov_expert_prediction_eval.md`. |

## Key types & entry points

- `TestParams::new` / `new_with_defaults` — builds the shared prompt/name/tokenizer bundle from `system.md`/`story.md`/`names.md`.
- `TestMode` — `StoryRewrite` (default, requires instruction-following, 3B+ models), `NameGreeting` (small-model-friendly), `CoherenceCheck` (lossy-format sanity), `Skip` (throughput-only, no validation).
- `ruler_gen::generate_ruler_samples`, `run_ruler_eval`, `run_ruler_benchmark`, `run_ruler_continuous`, `sweep_parallelism` — RULER sample generation and batched-inference evaluation against a `ManagedBatchedModel`.

## How it is used

Tests live behind `#[ignore]` (they download a GGUF from HF Hub and need a
GPU) and are run explicitly, e.g.:

```bash
cargo test --release --features cuda --lib --package candle-transformers \
  quantized_llama::tests::test_parallel_batched_forwarding_llama3 -- --ignored --nocapture

cargo test --release --features cuda,verbose --lib --package candle-transformers \
  quantized_qwen3_moe::tests::test_ruler_eval -- --ignored --nocapture
```

The `quantized_llama` story-rewrite tests run against
**`VibeStudio/Nidum-Llama-3.2-3B-Uncensored-GGUF`** (per `CLAUDE.md`'s
`Llama-3.2-3B` / batch_test integration-testing entry) via `test_helpers::api()`.
The captured `perf-investigation/baseline_run*.txt` and
`60353239_newprefill_run*.txt` transcripts are prior runs of this same
`test_parallel_batched_forwarding_llama3` test.

## Related docs

`docs/markov_expert_prediction_eval.md` (the routing-trace fixture's
consumer), `docs/perplexity_results.md` (a different, standalone quality
harness — not this module).
