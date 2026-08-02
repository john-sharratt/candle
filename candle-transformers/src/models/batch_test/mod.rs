//! RULER-style long-context integration harness for `batched_inference`.
//!
//! [`ruler_gen`] generates the RULER benchmark tasks (needle retrieval,
//! variable tracing, common-word extraction) and runs them against a loaded
//! `ManagedBatchedModel`. `test_helpers` and `utils` (test-only) provide
//! shared fixtures — `story.md`/`system.md` prompt bodies and a captured
//! Qwen3-30B-A3B expert routing trace (`fixtures/`) — for exercising the
//! batched decode/prefill/glue path end to end under `#[test]`.
pub mod ruler_gen;
#[cfg(test)]
pub mod test_helpers;
#[cfg(test)]
pub mod utils;
