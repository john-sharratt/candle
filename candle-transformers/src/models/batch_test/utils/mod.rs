//! The batch test harness: runs a model's configurations through the engine and
//! judges what they produce.
//!
//! The test mode decides how output is judged: story rewrite for models that
//! follow instructions, name greeting for small models, coherence for formats
//! so lossy that names are lost, and skip where only throughput can be measured.
//! A configuration may override the run's mode. The harness rows measure how
//! fast a forward can go from a clean slate. Callers' engine rows are measured a
//! crate up and joined to the same table, because the difference between the
//! two is the cost the daemon adds.
//!
//! - `accounting`: prefill pricing and the load's VRAM baseline.
//! - `format`: the printed prediction and regret figures.
//! - `params`: the parameters a gate runs with.
//! - `print_comparison`: the side-by-side configuration table.
//! - `print_expert`: the expert-cache tables.
//! - `print_profile`: the profile tables.
//! - `reproducibility`: whether decode and prefill repeat bitwise.
//! - `run`: loading a model and running its configurations.
//! - `run_batched`: one batched configuration, prefill through decode.
//! - `speculative`: the speculative generate phase.
//! - `validate`: judging output against its expected text.

// Test-harness code (the module is `cfg(any(test, feature = "ruler-bench"))`),
// blanket-allowed rather than restyled: the fixtures spell shapes and indices
// out to match the model configs they mirror.
#![allow(
    clippy::clone_on_copy,
    clippy::collapsible_if,
    clippy::manual_repeat_n,
    clippy::redundant_closure,
    clippy::type_complexity,
    clippy::useless_conversion
)]

use candle::forbidden_alloc;
use candle::quantized::Int8Mode;
use candle::{DType, Device, Result, Tensor};
use candle_nn::kv_cache::{aside, QuantFormat, REGION_BYTES};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokenizers::Tokenizer;

#[cfg(feature = "cuda")]
use crate::models::batch_test::gap_tally::{rate_ex_gap, GapTally, Window};
#[cfg(feature = "cuda")]
use crate::models::batch_test::graph_report::print_graph_line;
use crate::models::batch_test::greedy::{aside_bytes, greedy_token, live_vocab, token_ids};
use crate::models::batch_test::host_ram_report::{print_host_ram, print_host_ram_line};
use crate::models::batch_test::side_compression::{print_side_table, SideCompression};
use crate::models::batch_test::span_report::print_span;
#[cfg(feature = "cuda")]
use crate::models::batch_test::span_settle::SpanSettler;
#[cfg(feature = "cuda")]
use crate::models::batch_test::vram_wave_log::VramWaveLog;
use crate::models::batched_inference::{
    BatchedConfig, BatchedInferenceSession, InferenceMode, ManagedBatchedModel,
};
use crate::models::dialect::Dialect;
use crate::models::draft_depth::DraftDepth;
use crate::models::expert_lre::{cell_label, tag_label, PipelineStats, ReadLatency, CELLS, TAGS};
use crate::models::profile::{
    gpu_drain_blocking, pipeline_record, pipeline_snapshot_and_reset, profile_now, ProfileSnapshot,
};
use crate::models::speculative_choice::GreedyChooser;
use candle::vram::process_ram::ProcessRam;
#[cfg(feature = "cuda")]
use candle_nn::kv_cache::GapKind;
use std::cmp::Reverse;
use story_rewrite::fixtures;
use story_rewrite::normalize::normalize_story;

/// Determines the validation strategy for the test harness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestMode {
    /// Story rewrite: model must replace the character name "Marcus" with a
    /// per-session name.  Requires instruction-following ability (3 B+ models).
    StoryRewrite,
    /// Name greeting: model receives the user's name in the system prompt and
    /// must mention it in its reply.  Suitable for small models (0.5 B+).
    /// All adjacent different-name session pairs must produce distinct output.
    NameGreeting,
    /// Coherence check: verifies the model produces coherent text output
    /// (not garbage like repeated punctuation).  Does not compare sessions.
    /// Use for very lossy KV quantization formats (Q4_KS, Q3_0) where
    /// per-name differentiation is lost but correct decoding must still work.
    CoherenceCheck,
    /// Skip: run the config and collect timing/compression stats but perform
    /// no output validation.  Use for extremely lossy formats (Q2_0, Q3_0)
    /// where even coherence cannot be guaranteed but throughput matters.
    Skip,
}

#[derive(Debug, Clone)]
pub struct TestConfig {
    pub mode: InferenceMode,
    pub use_batched: bool,
    pub num_contexts: usize,
    pub num_repeats: usize,
    /// Override the global `TestParams::test_mode` for this specific config.
    /// `None` means use the global mode.
    pub test_mode: Option<TestMode>,
}

/// Model-agnostic test parameters for prompt and generate phases
pub struct TestParams {
    pub print_outputs: bool,
    pub skip_validation: bool,
    pub disable_session_isolation: bool,
    pub majority_pass_threshold: Option<usize>,
    pub test_mode: TestMode,
    pub suppress_thinking: bool,
    pub prompt_system: String, // System prompt text
    pub prompt_user: String,   // User prompt text
    /// Optional per-config user-prompt overrides, indexed by config position.
    /// A non-empty entry at index `n` replaces `prompt_user` for the n-th
    /// config in `run()`.  Empty string or missing index keeps `prompt_user`.
    /// Used by the routing-trace capture to drive diverse prompts through a
    /// single model load.
    pub per_config_prompts: Vec<String>,
    /// Optional per-config names, indexed by config position, heading that
    /// config's profile tables in place of its `mode×contexts` — for a run
    /// whose configs share a mode and width and differ only in their prompts.
    pub config_labels: Vec<String>,
    /// Token IDs that end generation early when sampled (e.g. EOS).  Empty =
    /// always generate the full `generate_token_count` (benchmark behaviour).
    /// When set, the batched generate loop stops once every session has emitted
    /// a stop token.  Used by capture to avoid post-EOS degenerate routing.
    pub stop_on_eos: Vec<u32>,
    pub names: Vec<String>,
    pub generate_token_count: usize, // Number of tokens to generate in generate phase
    /// Tokens the **C9 and C10** rows generate instead of
    /// [`Self::generate_token_count`].
    ///
    /// The top rungs are the calibration probes, tuned just under their edge at
    /// a fixed reproduction length; a longer reproduction is a stricter test, and
    /// holding the probes to it would cost compression at every level to buy
    /// margin at rungs nothing runs. C9 is included because it sits as close to
    /// the edge: on Qwen3.6-35B-A3B at 256 tokens C9×2 failed 0/2 on the same
    /// near-tie at character ~1,213 that C10 failed on (2026-10-05, RTX 3090),
    /// while C0–C8 passed. The other rows decode longer so their rate is
    /// measured after the expert working set has recovered from the prefill.
    /// `None` runs every row at [`Self::generate_token_count`].
    pub top_rung_token_count: Option<usize>,
    pub tokenizer: Tokenizer, // Tokenizer for text-to-token conversion
    /// The ids [`Self::tokenizer`] can name ([`token_ids`]), read once here: the
    /// decode phase clamps every pick to it inside its timed window, and walking
    /// the vocabulary there was charged to decode.
    pub token_ids: Option<usize>,
    pub dialect: Dialect, // Chat format dialect (ChatML, Llama3, etc.)
    pub device: Device,
    pub begin_document_token: Option<u32>,
    pub timeout_secs: u64, // Test timeout in seconds (default: 120)
    /// Inference `Int8Mode` for this run, shown in the comparison table's `int8` column so a saved
    /// table records which numeric mode produced it. Defaults to `Off`; set via
    /// [`Self::with_int8mode`].
    pub int8mode: Int8Mode,
    /// LoRA adapter every sequence decodes through, by name. `None` runs the
    /// base model, which is every gate but the adapter's own.
    pub lora: Option<String>,
    /// How the generate phase decides each step's draft budget.
    ///
    /// Defaults to [`DraftBudget::Adaptive`], so a gate measures what production
    /// would actually do at that width. **Asked for, not granted** either way: a
    /// model clamps the answer to its own drafter's ceiling, and a model with no
    /// drafter proposes nothing and decodes one token per step. Speculation
    /// accepts only the model's own greedy continuation, so the generated text —
    /// and therefore every gate's expected-output check — is identical at any
    /// budget. See [`Self::with_speculative`].
    pub speculative_max_draft: DraftBudget,
    /// Pin every sealed K block to one quantization format instead of letting
    /// the policy choose per block.
    ///
    /// For asking which *format* a rung's behaviour belongs to. A level's
    /// candidate list is a set, and the adaptive selection means a run tells
    /// you what the level did on average, never which member did it. Pinning
    /// one at a time turns that into an answerable question.
    pub override_k_quant: Option<QuantFormat>,
    /// Rows the caller measured itself, appended below the harness's own in the
    /// comparison table. See [`ExtraRow`] for why the table takes them.
    pub extra_rows: Vec<ExtraRow>,
    /// Where each config's expert routing is written after its decode — the
    /// decode interval with the prefill before it as warm-up
    /// (`ExpertCache::write_routing_trace`) — for replaying residency and
    /// prediction policies offline. `None` writes nothing.
    pub routing_trace_dir: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct PhaseResults {
    pub avg_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub runs_used: usize,
}

/// A row the caller measured itself, appended to the comparison table below the
/// harness's own.
///
/// **Why the table takes foreign rows at all.** The rows this harness produces drive
/// `forward_wave` on a session it constructs, from a clean slate, with nothing else
/// running — which is the right way to measure *how fast the forward can go*, and the
/// reason it is useless for measuring what a daemon sees. A daemon's throughput is
/// shaped by admission, by the projection it runs per turn, by the persistence thread
/// moving KV between tiers, and by whatever fragmentation the pool has accumulated —
/// none of which exists in this crate. So the engine-driven rows are measured one crate
/// up and handed here, and the two sit in one table because the comparison is the point:
/// the harness rows are the ceiling, the engine rows are the delivery.
///
/// The fragmentation columns are `None` for a harness row, and that is not a gap. Those
/// rows *cannot* fragment: every sequence is freed and every empty arena released
/// between configs, with a gate asserting nothing is live before the next one starts.
#[derive(Debug, Clone)]
pub struct ExtraRow {
    /// What produced this row, in the `KvMode` column's place — e.g. `Q8_0/eng`.
    pub label: String,
    /// Concurrent sequences.
    pub contexts: usize,
    pub prompt_tokens_per_sec: f64,
    pub generate_tokens_per_sec: f64,
    /// `(passed, total)` sessions, or `None` where the row does not validate content.
    pub valid: Option<(usize, usize)>,
    pub compression_ratio: Option<f64>,
    pub peak_tokens: usize,
    /// The arena frontier, in regions — one past the highest live region. **The figure
    /// that costs weights**: the wave transient tier stands above it and `weight_floor`
    /// is measured from there, so this one index sets expert residency and therefore
    /// decode.
    pub frontier_regions: Option<usize>,
    /// `packed_arenas / frontier` as a percentage: of the ground denied to the weight
    /// side, the share actually holding KV.
    pub efficiency_pct: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct TestRun {
    pub output: Vec<u32>,            // Generated output text
    pub expected: String,            // Expected output text
    pub logits: Tensor,              // Logits from the model
    pub prompt_phase: PhaseResults,  // Prompt processing results
    pub generate_avg_per_token: f64, // Average ms per generated token
    pub generate_total_ms: f64,      // Total time for all generate iterations
    pub total_time_ms: f64,
}

#[derive(Debug, Clone)]
pub struct TestResults {
    pub config: TestConfig,
    pub sessions: Vec<TestRun>,
    pub prompt_tokens_per_sec: f64, // Tokens/sec for bulk prompt processing
    pub generate_tokens_per_sec: f64, // Tokens/sec for single token generation
    /// The prompt rate with the gap's host time inside the prompt clock taken
    /// out (`gap_tally`): the forwards' own rate.
    pub prompt_ex_gap_tps: f64,
    /// The decode rate with the gap's host time inside the decode clock taken
    /// out.
    pub generate_ex_gap_tps: f64,
    pub all_valid: bool, // Whether all sessions passed validation
    pub quantized_token_percent: Option<f64>, // Percentage of tokens stored in quantized arenas
    pub compression_ratio: Option<f64>, // Float-equivalent bytes / actual quantized bytes
    /// K's side of `compression_ratio`: its ratio and format mix.
    pub k_side: SideCompression,
    /// V's side of `compression_ratio`: its ratio and format mix.
    pub v_side: SideCompression,
    pub peak_tokens: usize, // Total tokens across all sessions at peak (after generation)
    /// Expert cache telemetry for the decode phase alone (if the model has MoE):
    /// the counters are reset at the prefill → decode boundary, after
    /// [`Self::prefill_expert_stats`] is taken.
    pub expert_stats: Option<PipelineStats>,
    /// Expert cache telemetry for the prefill phase alone, and the gauges as
    /// the decode inherits them. Separate because one wide prefill slab misses
    /// on nearly every expert of every layer, and summed with the decode it
    /// swamps the figures a decode-side change is judged by.
    pub prefill_expert_stats: Option<PipelineStats>,
    /// This process's host RAM at the end of the decode. `None` where the
    /// platform has no address-space walk.
    pub host_after_decode: Option<ProcessRam>,
    /// `(hits, misses, evictions)` of a disk-resident embedding tier's row
    /// cache, if the model serves one. Cumulative across the run: the cache is
    /// process-wide and deliberately not reset per config, because its hit
    /// rate is a property of the traffic seen so far.
    pub row_cache_stats: Option<(u64, u64, u64)>,
    pub bulk_profile: ProfileSnapshot, // Profile data from prompt (bulk) phase
    pub single_profile: ProfileSnapshot, // Profile data from generate (single) phase
    pub pipeline_bulk_profile: ProfileSnapshot, // Pipeline spans, prefill (bulk) phase only
    pub pipeline_profile: ProfileSnapshot, // Pipeline spans, decode (single/generate) phase only
    /// Effective test mode used for this config (may override the global `TestParams::test_mode`).
    pub effective_test_mode: TestMode,
}

mod accounting;
mod format;
mod params;
mod print_comparison;
mod print_expert;
mod print_profile;
mod reproducibility;
mod run;
mod run_batched;
mod speculative;
mod validate;

pub use accounting::*;
use format::*;
pub use params::*;
pub use reproducibility::*;
