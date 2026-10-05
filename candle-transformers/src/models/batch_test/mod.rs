//! RULER-style long-context integration harness for `batched_inference`.
//!
//! [`ruler_gen`] generates the RULER benchmark tasks (needle retrieval,
//! variable tracing, common-word extraction) and runs them against a loaded
//! `ManagedBatchedModel`. `test_helpers` and `utils` (test-only) provide
//! shared fixtures — `story.md`/`system.md` prompt bodies and a captured
//! Qwen3-30B-A3B expert routing trace (`fixtures/`) — for exercising the
//! batched decode/prefill/glue path end to end under `#[test]`.
/// The checked-in prompt bodies, readable outside `#[cfg(test)]` so a harness in
/// a crate above this one measures the SAME prompt the gates do.
pub mod fixtures;
/// The gate's summary of what the wave chains did across a decode.
#[cfg(feature = "cuda")]
pub(crate) mod graph_report;
/// Greedy token picks on the fused batched sampler, one launch per batch.
pub mod greedy;
/// This process's host RAM by allocation, printed after load, after each
/// prefill, and as a table of every config's decode end.
pub mod host_ram_report;
/// The depth ladder: the batched forward at 32K–128K of KV, and the filler
/// that gets it there without the degenerate repetition a tiled corpus gives.
///
/// Test-only, like [`utils`] whose harness it drives.
#[cfg(test)]
pub mod long_context;
pub mod ruler_gen;
/// Each quantized rung's K and V ratios and format mixes, beside the table's
/// combined `Compress`.
pub mod side_compression;
/// The whole-card VRAM decomposition off the span's accounting, printed at each
/// config's decode end and at the end of the run.
pub mod span_report;
/// The StoryRewrite comparison form: whitespace collapsed, gendered words
/// neutralised at any word boundary.
///
/// Outside `#[cfg(test)]` alongside [`fixtures`], and for the same reason: a
/// harness above this crate that validates a rewrite must apply the *same*
/// normalisation the gates do, or the two disagree about what a correct rewrite
/// is. Pure string work with no test-only dependencies.
pub mod story_normalize;
#[cfg(test)]
pub mod test_helpers;
/// The batched comparison harness: `TestParams`, the config ladder's row type, and the
/// performance table.
///
/// Readable outside `#[cfg(test)]` for the same reason [`fixtures`] and
/// [`story_normalize`] are, and it is the strongest instance of that reason. The rows
/// this harness produces drive `forward_wave` from a clean slate — the *ceiling*. What a
/// daemon delivers is shaped by admission, per-turn projection, the persistence thread
/// and accumulated fragmentation, none of which exist in this crate. Those rows are
/// measured one crate up and appended here through [`utils::TestParams::with_extra_rows`],
/// so ceiling and delivery land in one table; a harness only reachable under `#[test]`
/// could not be given them, and the two figures would have to be compared across two
/// outputs that do not measure the same machine.
pub mod utils;
/// The YaRN gates run on the models whose rungs they back.
#[cfg(test)]
mod yarn_gate_models;
/// The progressive-YaRN gates: a needle read back past the trained window,
/// against the model on rung-1 extrapolation.
#[cfg(test)]
pub mod yarn_gates;
