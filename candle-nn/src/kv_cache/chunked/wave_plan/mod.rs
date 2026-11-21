//! Prices what a wave costs in the transient tier, so admission sizes a wave from
//! its budget and the plan is an upper bound by construction.
//!
//! A layer opens two generations, one for attention and one for the FFN, which
//! never overlap, and a forward phase holds the residual and the head beside them.
//! Each buffer is named once as a `WaveBuffer` variant, and the chains within a
//! phase combine by maximum when they are alternative kinds of layer and by sum
//! when they are groups of one kind at their own widths. The variants are pinned
//! to the census, so a change to the attention or FFN chain is a change to that
//! list. The budget decides the width, and the plan answers how many rows fit.
//!
//! - `width`: how wide a wave is, in prefill, decode and scored rows.
//! - `wave_buffer`: every buffer a layer allocates, one variant per site.
//! - `buffer_impl`: the phase, chain and mode each buffer belongs to.
//! - `carve`: each buffer's shape at a given wave width.
//! - `chain`: the chains a phase can run and how they combine.
//! - `geometry`: the width-free model shapes the sizes derive from.
//! - `model_widths`: the DeltaNet, shared expert and Gated Residual widths.
//! - `phase`: the generations and how their spans are laid out.
//! - `plan`: prices a wave against the geometry for admission.
//! - `shape`: a buffer's byte count from its shape and encoding.
//! - `bounds`: width searches that stop at an unspendable budget.
//! - `tests_attention`: attention chain pinned to its census.
//! - `tests_bounds`: exactness and monotonicity of the width bounds.
//! - `tests_buffers`: phase and size of every buffer variant.
//! - `tests_chains`: DeltaNet chains pinned to their census.
//! - `tests_ffn`: FFN chains pinned to their measured generations.
//! - `tests_fixtures`: the model geometries the plan tests use.
//! - `tests_forward`: forward phase and setup tables.
//! - `tests_head`: draft head pass and its attention phase.
//! - `tests_hyper`: Gated Residual pre-mix and PLE block.
//! - `tests_phase`: how a phase's chains combine and share a region.
//! - `tests_plan`: each group priced for its own chain.
//! - `tests_tier`: the tier admission prices is the tier placed.
//! - `tests_width`: prefill and joint width bounds.

use super::prefill_kv_stage::PrefillKvStageLayout;
use super::types::TARGET_ARENA_BYTES;
use candle::DType;
use strum::IntoEnumIterator;
use strum_macros::EnumIter;

mod bounds;
mod buffer_impl;
mod carve;
mod chain;
mod geometry;
mod model_widths;
mod phase;
mod plan;
mod shape;
#[cfg(test)]
mod tests_attention;
#[cfg(test)]
mod tests_bounds;
#[cfg(test)]
mod tests_buffers;
#[cfg(test)]
mod tests_chains;
#[cfg(test)]
mod tests_ffn;
#[cfg(test)]
mod tests_fixtures;
#[cfg(test)]
mod tests_forward;
#[cfg(test)]
mod tests_head;
#[cfg(test)]
mod tests_hyper;
#[cfg(test)]
mod tests_phase;
#[cfg(test)]
mod tests_plan;
#[cfg(test)]
mod tests_tier;
#[cfg(test)]
mod tests_width;
mod wave_buffer;
mod width;

use bounds::*;
pub use chain::*;
pub use geometry::*;
pub use model_widths::*;
pub use phase::*;
pub use plan::*;
pub use shape::*;
pub use wave_buffer::*;
pub use width::*;
