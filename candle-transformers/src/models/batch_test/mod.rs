//! RULER-style long-context integration harness for `batched_inference`.
//!
//! [`ruler_gen`] generates the RULER benchmark tasks (needle retrieval,
//! variable tracing, common-word extraction) and runs them against a loaded
//! `ManagedBatchedModel`. `test_helpers` and `utils` (test-only) provide
//! shared fixtures — `story.md`/`system.md` prompt bodies and a captured
//! Qwen3-30B-A3B expert routing trace (`fixtures/`) — for exercising the
//! batched decode/prefill/glue path end to end under `#[test]`.
#[cfg(test)]
pub mod host_ram_report;
/// The depth ladder: the batched forward at 32K–128K of KV, and the filler
/// that gets it there without the degenerate repetition a tiled corpus gives.
///
/// Test-only, like [`utils`] whose harness it drives.
#[cfg(test)]
pub mod long_context;
pub mod ruler_gen;
#[cfg(test)]
pub mod test_helpers;
#[cfg(test)]
pub mod utils;
#[cfg(test)]
pub mod vram_snapshot;
