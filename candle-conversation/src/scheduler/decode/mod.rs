//! The decode loop of the conversation scheduler: each step's wave, the commit of
//! every sampled token, the rollback of rejected speculation, and the
//! reprojections a step triggers.
//!
//! A step gathers the sequences that can run, plans their rows and drafts, runs
//! them as one forward, and then commits each accepted token through the same path
//! a single token takes. Stencil static runs ride the same forward. A step that
//! cannot run for want of room is retried rather than failing its turns. The
//! reprojections that a decode queues are prepared, batched into one gap-fill
//! forward, and then completed, so a step costs one shared forward rather than one
//! per view.
//!
//! - `commit`: commits each decoded token, and forwards healed prefixes.
//! - `decode_step`: runs the step's wave and rolls back rejected speculation.
//! - `failure`: retries refused steps and fails turns on real faults.
//! - `reprojection`: queues and runs the reprojections, batched into one forward.
//! - `stencil`: plays static runs into the wave and picks the next sampling action.
//!
//! - `tests_decode_step`, `tests_failure`: tests for the rollback targets and the
//!   refusal-versus-fault distinction.

use super::answered::{after_reasoning, answered};
use super::band::rank;
use super::block_guard::BlockGuard;

use super::named_tool::steer_to_named_tool;
use super::segment_rearm::drop_rearms_segment;
use super::spec_chooser::{in_head_order, locate_rows, SpecChooser};
use super::stencil_wave::{StencilRow, StencilWave};
use super::step_census::HostPart;
use super::wave_retry::{Next, WaveRetry};
use super::*;
use crate::batched_sampler::DEGENERATE_TOKEN_RUN;
use crate::recorded_reply::{departure, replayed_step};
use crate::sampler_args::is_nonfinite_refusal;
use crate::stats::streams_committed;
use candle_nn::kv_cache::{
    is_arena_exhausted, is_ground_refusal_message, is_span_full, is_tier_refusal,
};
use candle_transformers::models::expert_lre::{is_expert_stall, PipelineStats, ProfileSnapshot};
use candle_transformers::models::held_tensor::hold_rows;
use candle_transformers::models::speculative_choice::AcceptWalk;

mod commit;
mod decode_step;
#[cfg(test)]
use decode_step::*;
mod failure;
mod reprojection;
mod stencil;
#[cfg(test)]
mod tests_decode_step;
#[cfg(test)]
mod tests_failure;

pub(super) use commit::*;
use failure::*;
