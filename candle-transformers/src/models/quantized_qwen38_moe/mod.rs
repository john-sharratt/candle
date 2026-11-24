//! Qwen3.8-Flash-Next, the routed Qwen3.8 model: its checkpoint pins, engine
//! artifacts and batched-forward gates.
//!
//! Flash-Next is a 48-layer hybrid of 36 Gated DeltaNet layers and 12 QSA attention
//! layers, with a 512-expert top-10 MoE on every layer. The family's machinery
//! lives in the `qwen4exp` module. This file pins the checkpoints that module loads,
//! the engine artifacts built from them, and the gates that check the forward. Its
//! dense sibling is a different model in the `qwen35` lineage despite the shared
//! name, and its `_moe` suffix matches the 3.5 and 3.6 siblings.
//!
//! - `checkpoint_pins`: the pinned checkpoints and their revisions.
//! - `engine_artifact`: the engine artifact recipe for each card.
//! - `forward_gates`: the batched-forward gate's configuration ladder.
//! - `tests_gpu_engine`, `tests_oracle`, `tests_profiling`, `tests_recipes`,
//!   `tests_speculation`, `tests_support`: tests and profiling runs.

use std::path::PathBuf;

use candle::quantized::GgmlDType;
use candle::{Device, Result};

#[cfg(all(feature = "cuda", any(test, feature = "ruler-bench")))]
use crate::models::batch_test::utils::{TestConfig, TestMode};
#[cfg(all(feature = "cuda", any(test, feature = "ruler-bench")))]
use crate::models::batched_inference::InferenceMode;

#[cfg(feature = "cuda")]
use super::model_pack::model_pack;
use super::model_pack::{cache_root, PackFamily, PackRequest};
use super::quant_ladder;
#[cfg(feature = "cuda")]
use super::qwen4exp::prepare::EngineFetch;
use super::qwen4exp::prepare::{ExpertSource, Recipe, SourceFile, SourceRole, SourceStore};
use super::qwen4exp::{load_oracle_model, Qwen4ExpModel};
use candle::quantized::Int8Mode;

mod checkpoint_pins;
mod engine_artifact;
mod forward_gates;
#[cfg(any(test, feature = "hub"))]
mod vision_tower;
#[cfg(test)]
mod tests_gpu_engine;
#[cfg(test)]
mod tests_oracle;
#[cfg(test)]
mod tests_profiling;
#[cfg(test)]
mod tests_recipes;
#[cfg(test)]
mod tests_speculation;
#[cfg(test)]
mod tests_support;

pub use checkpoint_pins::*;
pub use engine_artifact::*;
pub use forward_gates::*;
#[cfg(any(test, feature = "hub"))]
pub use vision_tower::{cached_vision_tower_file, vision_tower_file};
