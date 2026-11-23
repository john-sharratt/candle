//! Admits, runs and pressure-manages the prefills, section ingests and creeps
//! that share the wave with decode.
//!
//! Work enters a wave through admission, which prices each offer with the rate
//! model and holds the engine's deadlock-freedom rule. Admitted prefills and
//! sections are grouped by band and adapter into a creep that sweeps the layers
//! beside decode, and a finished prefill's first token is sampled from its row.
//! Under VRAM pressure the pools are packed and relieved before expert residency
//! is conceded, and ingest KV is shed first. A forward's failure is routed by its
//! kind, and a refused width narrows the next group.
//!
//! - `admitting`: admission of queued prefills, priced by the rate model.
//! - `decode_priority`: airtime shares between decodes and prompts by band.
//! - `first_samples`: first-token sampling and promotion into decode.
//! - `ingest_pressure`: shedding cold ingest KV to the warm tier.
//! - `pack_pools`: compaction of KV pools and span tenants.
//! - `prefill_runs`: forwarding a span cut at reasoning and turn boundaries.
//! - `rate_model`: learned rates and the cohorts they price.
//! - `relief`: the VRAM pressure ladder.
//! - `section_ingest`: ragged section-ingest chunks.
//! - `vram_phase`: pressure setpoints for loading and decoding.
//! - `wave_group`: forming, pausing and dropping the prefill group.
//! - `wave_run`: completing and routing a wave-group forward.
//! - `wave_step`: one wave step across its segments.
//!
//! - `tests_creep`: creep membership and throttle tests.
//! - `tests_prefill`: reasoning-boundary split and think-block tests.
//! - `tests_pressure`: setpoint, settle-loop and warm-budget tests.
//! - `tests_priority`: band and tenant priority tests.
//! - `tests_wave_admission`: wave admission and group formation tests.

use std::time::{Duration, Instant};

use super::admission::{
    admit_quantum, budget_notches, evidence_admit_grow, evidence_ticks_for, per_block_kv_bytes,
    ThrottleReason,
};
use super::admission_split::AdmitPart;
use super::admit;
use super::admit_ground::{joins_a_full_forward, AdmitPass};
use super::band::rank;
use super::compaction_split::TenantPart;
use super::compaction_stall::{FrontierPass, PassOutcome, PoolShape};
use super::creep_slice::{
    short_piece, slice_target, sweeps_whole, time_bounded_layers, window_end,
};
use super::first_sample::{
    contiguous_runs, row_is_nan_free, FirstSample, FirstSampled, PrefillEnd,
};
use super::gang;
use super::page_cut::{break_advance, chunk_closes_page};
use super::paused_creep::{drop_rows, PausedCreep};
use super::phase_split::{cobatched_decode_rows, full_sweep_us, window_tokens};
use super::section_pack;
use super::wave_logits::WaveLogits;
use super::MAX_PREFILL_WIDTH;
use super::*;
use crate::projection::DecodePriority;
use crate::recorded_reply::replayed_step;
use crate::recurrent_span::{snap_advance, span_at, ArmKind};
use crate::stats::streams_committed;
use crate::substrate::residence::{drain_reclaimed, rewrite_reclaimed};
use crate::substrate::EvictionReport;
use crate::token_buffer::TokenBuffer;
use candle_nn::kv_cache::{
    end_wave_transient, is_device_oom, is_span_full, quiesce_wait_ns, tier_refusal_shortfall,
    FreeRegionTarget,
};
use candle_transformers::models::batched_inference::PendingGlue;
use candle_transformers::models::claim_split::ClaimSplit;
use candle_transformers::models::expert_lre::is_expert_stall;
use candle_transformers::models::residual_rows::{compact_creeps, CreepResidual};
use candle_transformers::models::wave_driver::{StepCreep, StepRows, StepSpec};
use std::collections::HashSet;
use std::slice;

mod admitting;
mod decode_priority;
mod first_samples;
mod ingest_pressure;
mod pack_pools;
mod prefill_runs;
mod rate_model;
mod relief;
mod section_ingest;
#[cfg(test)]
mod tests_creep;
#[cfg(test)]
mod tests_prefill;
#[cfg(test)]
mod tests_pressure;
#[cfg(test)]
mod tests_priority;
#[cfg(test)]
mod tests_wave_admission;
mod vram_phase;
mod wave_group;
mod wave_run;
mod wave_step;

use decode_priority::*;
pub(super) use ingest_pressure::*;
pub(super) use pack_pools::*;
use prefill_runs::*;
pub(super) use rate_model::*;
pub(super) use relief::*;
#[cfg(test)]
use tests_wave_admission::*;
pub(crate) use vram_phase::*;
