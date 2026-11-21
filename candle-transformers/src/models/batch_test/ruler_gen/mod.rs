//! The RULER long-context benchmark: generated or loaded task samples, scored
//! predictions from a batched run, and a streaming log and report.
//!
//! The four core RULER tasks are generated as pure data, so they can run inside
//! tests that already hold a loaded model. The samples are formatted as the Qwen3
//! chat prompt with thinking suppressed. A generic batched runner executes them,
//! and the scorer checks each prediction against its expected answers. A benchmark
//! run either generates its samples or loads them from the reference files, and
//! its results can be streamed to a log and reported from that log afterwards.
//!
//! - `continuous`: the open-ended streaming run and the report over its log.
//! - `generators`: procedural generation of each task's samples.
//! - `jsonl`: loading samples from the reference JSONL files.
//! - `prompt`: the Qwen3 prompt frame, the filler corpus and token counting.
//! - `scoring`: prediction scoring and the batched prefill and decode runner.
//! - `sweeps`: the batch-size sweep and the full benchmark over it.

// Benchmark-harness code (the module is `cfg(any(test, feature = "ruler-bench"))`).
// The scorers index task/length grids positionally, which is how the RULER
// results table is laid out.
#![allow(
    clippy::needless_range_loop,
    clippy::ptr_arg,
    clippy::redundant_closure,
    clippy::too_many_arguments,
    clippy::useless_conversion
)]

use candle::{Device, Result, Tensor};
use rand::{rngs::StdRng, Rng, SeedableRng};
use tokenizers::Tokenizer;

use super::greedy::greedy_rows;
use crate::models::batched_inference::{BatchedConfig, InferenceMode, ManagedBatchedModel};

/// Where to load samples from for [`run_ruler_benchmark`].
pub enum RulerDataSource<'a> {
    /// Generate samples procedurally (deterministic, no external files).
    Generated,
    /// Load from a directory of JSONL files produced by the canonical RULER
    /// data-generation script.  Expected filename pattern:
    /// `<task>_<ctx_len>.jsonl`  e.g. `niah_single_1_4096.jsonl`
    Jsonl(&'a std::path::Path),
}

// ── Public types ──────────────────────────────────────────────────────────────

/// A single RULER evaluation sample: the prompt and its expected answer(s).
#[derive(Clone, Debug)]
pub struct RulerSample {
    /// The full prompt string ready to be tokenized and fed to the model.
    pub input: String,
    /// Expected answer strings.  Scoring checks that all appear in the prediction.
    pub outputs: Vec<String>,
}

/// RULER task variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RulerTask {
    /// Single needle-in-a-haystack — one magic number.
    NiahSingle1,
    /// Multi-key needle — two independent magic numbers.
    NiahMultiKey2,
    /// Variable tracing — follow a 4-link assignment chain.
    Vt,
    /// Common word extraction — identify the 10 most frequent words.
    Cwe,
}

impl RulerTask {
    pub fn name(self) -> &'static str {
        match self {
            Self::NiahSingle1 => "niah_single_1",
            Self::NiahMultiKey2 => "niah_multikey_2",
            Self::Vt => "vt",
            Self::Cwe => "cwe",
        }
    }
}

impl std::fmt::Display for RulerTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

mod continuous;
mod generators;
mod jsonl;
mod prompt;
mod scoring;
mod sweeps;

pub use continuous::*;
pub use generators::*;
pub use jsonl::*;
pub use prompt::*;
pub use scoring::*;
pub use sweeps::*;
