//! RULER-style long-context integration harness for `batched_inference`.
//!
//! [`ruler_gen`] generates the RULER benchmark tasks (needle retrieval,
//! variable tracing, common-word extraction) and runs them against a loaded
//! `ManagedBatchedModel`. `test_helpers` and `utils` (test-only) drive the
//! batched decode/prefill/glue path end to end under `#[test]`, on the
//! StoryRewrite prompts of the `story-rewrite` crate and a captured
//! Qwen3-30B-A3B expert routing trace (`fixtures/`).
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
/// The scheduler's between-forwards packing, run by the gate between decode
/// steps.
#[cfg(feature = "cuda")]
pub mod span_settle;
/// Strata's published single-session speed benchmark, rebuilt request for
/// request and run through the harness.
///
/// Outside `#[cfg(test)]` beside [`utils`], for the same reason: the engine one
/// crate up delivers the *same* requests (`fragmentation_probe::run_strata`), and
/// it can only send byte-identical ones if it builds them with these functions.
pub mod strata_bench;
#[cfg(test)]
pub mod test_helpers;
/// The batched comparison harness: `TestParams`, the config ladder's row type, and the
/// performance table.
///
/// Readable outside `#[cfg(test)]` for the same reason the StoryRewrite fixtures
/// live in their own crate (`story-rewrite`), and it is the strongest instance of
/// that reason. The rows
/// this harness produces drive `forward_wave` from a clean slate — the *ceiling*. What a
/// daemon delivers is shaped by admission, per-turn projection, the persistence thread
/// and accumulated fragmentation, none of which exist in this crate. Those rows are
/// measured one crate up and appended here through [`utils::TestParams::with_extra_rows`],
/// so ceiling and delivery land in one table; a harness only reachable under `#[test]`
/// could not be given them, and the two figures would have to be compared across two
/// outputs that do not measure the same machine.
pub mod utils;
/// The card wave by wave — the performance page's bands — written to a CSV
/// under `target/` as a batched gate runs.
#[cfg(feature = "cuda")]
pub mod vram_wave_log;
/// The YaRN gates run on the models whose rungs they back.
#[cfg(test)]
mod yarn_gate_models;
/// The progressive-YaRN gates: a needle read back past the trained window,
/// against the model on rung-1 extrapolation.
#[cfg(test)]
pub mod yarn_gates;
