//! Qwen4Exp's batched model on the shared wave loop, with this stack's architecture supplied through its `WaveSweep`.
//!
//! `forward_wave` is `drive_wave`, and the sweep carries the architecture: a 4-stream Gated Residual, a 3:1 GDN-to-attention hybrid, a 512-expert MoE on every layer, and PLE injection at layer 1. Each full-attention layer caches its index keys on every wave and narrows its read to the indexer's winners once a row passes the budget; below it the selection is the identity and is skipped. The GDN, PLE and QSA index state is snapshotted at wave entry, so a failed wave leaves all three unadvanced, while the KV is rolled back by the driver. The draft head's KV layer is capped at a ceiling below which a session keeps its own level, because a proposal is worth something only if it reproduces the trunk's argmax.
//!
//! - `wave_sweep`: the sweep's claims, pricing, folds and rollbacks.
//! - `layers`: the per-layer attention, MoE and draft-head work over packed rows.
//! - `delegates`: the scheduler's pricing, carried-state and rollback methods.
//! - `model_core`: the four carried classes and the model's setup.
//! - `recurrent`: the GDN state's claim, fork, park and snapshot.
//! - `positional`: the QSA index's cuts, seals and injected pages.
//! - `qsa_wave`: the selection's budget, strata, pins and hit counters.
//! - `experts`: the routed experts of each MoE layer.
//! - `pricing`: the pricing of the buffers a wave carves.
//! - `session`, `speculative`, `step`: the session, speculative and step-level operations.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use candle::engine_events::note;
use candle::quantized::cuda::{to_dynamic, DynamicActs, MOE_PREDICT_VOTES_MAX_TOKENS};
use candle::{DType, Device, LiveTensor, Result, Tensor};
use candle_kernels::simple::qsa_topk::MAX_KEEP;
use candle_nn::kv_cache::{
    arena_regions, begin_forward, begin_wave, end_wave_transient, ffn_work_dtype,
    launch_scratch_reserve, plan_slot_moves, plan_wave_transient, region_stats, relocate_tensor,
    ArenaSlot, DeltaNetWidths, HyperWidths, KvCache, LayerPhase, ModelGeometry,
    PrefillKvStageLayout, PrefillLaunchBounds, SharedExpertWidths, SlotTenant, SpanRegion,
    WavePlan, WaveWidth, REGION_BYTES, WAVE_SPAN_BYTES,
};

use super::kv_row::kv_factors_for;
use crate::models::claim_split::ClaimPart;

#[cfg(feature = "tensor-assert")]
use super::attn_capture::attention_checkpoint;
use super::batched_attention::Qwen4ExpAttentionLayer;
use super::capture_rows::CaptureRows;
use super::coverage::coverage_disagreements;
use super::draft::{HeadWave, SeedStore};
use super::engine::{GpuLayerMix, Qwen4ExpGpu};
use super::hits::{
    block_map, block_range, count_hits, fork_holder, move_holder, HitHolder, HitTable, KeepShape,
    PooledRounds, SeqHits, ShadowScan,
};
use super::hyper::{hc_combine, hc_combine_gated, hc_mix, hc_mix_with_operand};
use super::index_keys::key_page_growth_bytes;
use super::indexer::{
    compact_index_caches, select_layer, selection_stride, IndexCache, IndexSnapshot, SplicePiece,
};
use super::key_q8::KeyPlanes;
use super::paged_index;
use super::paged_index::SealedIndex;
use super::ple::{PleSpan, PleState};
use super::ple_fused::ple_apply_spans_fused;
use super::qsa::IndexerWeights;
use super::qsa_select::{budget_fits_kernel, coalesce_ranges, Strata, MAX_PINS};
use super::resident_page::{PageRegistry, ResidentPage};
use super::seal_memo::SealMemo;
use super::select_bytes::select_layer_bytes;
use super::sequence_pools::SequencePools;
use super::shadow::{ShadowGeometry, ShadowRows, ShadowSegment};
use super::spec::SpecCapture;
use super::state_slots::state_buffer;
use crate::models::batched_inference::{
    BatchedConfig, BatchedInferenceSession, ManagedBatchedModel, ModelCoreProperties, WaveResult,
    MAX_PREFILL_TOKENS,
};
use crate::models::batched_layer::{
    forward_attn_batched, BatchedAttentionParams, BatchedPrefillMeta, DecodeHeaders,
};
use crate::models::batched_model::{WaveGuard, WavePhase};
#[cfg(feature = "cuda")]
use crate::models::delta_net::cuda::build_wave_table;
use crate::models::delta_net::{
    commit_stores, compact_stores, end_steps, quantized_delta_net_layer_forward_spans, seq_spans,
    ArmKind, DeltaNetSeq, ExportedLayerState, LayerKind, RecurrentCompaction, RecurrentStateStore,
    ReplyRows, SeqSpan, SpanArms, StagedSnapshot, ZGate, TIMELINE_ROWS,
};
use crate::models::dense_span;
use crate::models::draft_blend::{blend, DraftTable};
use crate::models::expert_lre::{
    PipelineStats, WeightPlan, WeightPlanning, LOOK_AHEAD_HOPS, LOOK_AHEAD_MAX_TOKENS, STAGED_HOPS,
};
use crate::models::head_rows::select_head_rows;
use crate::models::hit_layers::{HitPercentiles, PieceHits, PooledRound};
use crate::models::lazy_rope::LazyRope;
use crate::models::operand_guard::expect_dense_view;
use crate::models::piece_key::PieceKey;
use crate::models::positional_piece::PositionalPiece;
use crate::models::prefill_utils::paged_decode_q8_head_dim;
use crate::models::prefill_utils::SharedPm;
use crate::models::profile::{pipeline_record, profile_now, span, ProfileSnapshot};
use crate::models::qsa_selection::QsaSelection;
use crate::models::qwen_vision::{rotation_tables, ImageRotation, ImageRun};
use crate::models::quantized_matmul::QMatMul;
use crate::models::qwen35::quantized_weights::SHARED_GATE_TILE;
use crate::models::qwen35::spec::split_block_rows;
use crate::models::qwen38_flash_next_sweep as flash_next_sweep;
use crate::models::residency_rows::residency_decode_rows;
use crate::models::residual_rows::{CreepResidual, ResidualArena, ResidualIn};
use crate::models::rope_schedule::{FactoredRope, RopeRungs, RopeSchedule, RungSelect};
use crate::models::selection_strata::{Recent, StrataTokens};
use crate::models::wave_buffers::{wave_empty_ticketed, wave_from_vec_ticketed};

use super::rope::flash_next_schedule;
use crate::models::tensor_cat::TensorCat;
use crate::models::verify_wave::{upload_plan_rows, VerifyPlan};
use crate::models::wave_admit::admit_wave_kv;
use crate::models::wave_driver::{
    assemble_wave_contexts, drive_step, drive_wave, StepOutcome, StepSpec, SweepMode, WaveGroups,
    WaveSweep,
};
use crate::models::wave_token_ids::host_token_ids;

/// Seal `c`'s live tail into a **position-free** record: its completed rows and
/// its open block, read back to the host.
///
/// The rows are un-rotated, so they carry no position already — the record is
/// injectable at any offset in any conversation, the index's half of "compute
/// once, inject anywhere". Read straight out of the cache's key pages; nothing is
/// gathered on the device first.
fn seal_live(c: &IndexCache, last_cells: usize) -> Result<SealedIndex> {
    Ok(SealedIndex {
        keys: c.live_rows_host()?,
        last_cells,
        open: c.open_rows_host()?,
    })
}

/// The deepest KV compression the draft head's own layer seals at, whatever
/// the trunk runs. A ceiling: a session below it is untouched.
///
/// **C3 is the deepest level whose K candidate list contains no sub-3-bit
/// format.** C0–C5 all floor at `Q3_0`; C6 is the first to admit `Q1_S`,
/// `Q2_A` and `Q2_S`, which measure 4.9%, 1.5% and 0% drafted-token acceptance
/// against `Q4_0`'s 90.9% and `Q8_1`'s 97.6%
/// (`test_which_k_format_breaks_drafting`). So the cap is not a precision
/// preference — it is the boundary of the formats that work for keys, with two
/// rungs of margin.
///
/// Changing it is a measurement: run `test_speculative_ladder` and read the
/// accepted/step column at C6 and above.
pub const DRAFT_HEAD_MAX_COMPRESSION: u8 = 3;

/// The batched model the scheduler drives.
pub struct Qwen4ExpBatched {
    /// `pub(super)` so the draft head's own module can reach the engine it runs
    /// against — the head is part of this model, not a consumer of it.
    pub(super) model: Qwen4ExpGpu,
    /// The draft ceiling at each wave width on this card: the swept rows
    /// blended by its VRAM, resolved once at load
    /// ([`crate::models::qwen38_flash_next_sweep`]).
    draft: DraftTable,
    /// Per-sequence GDN state (S + conv tails), wave-atomic.
    pub(super) recurrent: RwLock<HashMap<usize, RecurrentStateStore>>,
    /// Spans armed for capture or replay, committed and rolled back with the
    /// stores (`docs/deltanet_timeline.md` §14.2).
    arms: RwLock<SpanArms>,
    /// Per-sequence PLE state (conv history + hash window) — the third
    /// carried class. Advanced in place by the sweep; the failure bracket
    /// snapshots and restores it.
    pub(super) ple: RwLock<HashMap<usize, PleState>>,
    /// Per-sequence QSA index caches, one per full-attention layer — the
    /// fourth carried class (§6.3's "QSA index ring"). Bracketed exactly as
    /// the other two: a failed wave leaves none of them advanced.
    pub(super) index: RwLock<HashMap<usize, Vec<IndexCache>>>,
    /// The resident index pages of every injected piece some cache still holds,
    /// so a piece pushed into many slots is placed once and shared.
    pages: PageRegistry,
    /// The records tail seals made, so a seal of pages placed from them reads
    /// none back (`seal_memo`).
    seal_memo: SealMemo,
    /// Per sequence, the pages its last reset dropped, held until its next
    /// reset or its release.
    ///
    /// The registry holds pages weakly, so a page lives only while some cache
    /// holds it — and a reprojection resets the slot, the pieces' only holder,
    /// before re-injecting them. Without this every rebuild decodes and places
    /// every page it had a moment ago: measured at 12,372 placements against
    /// 12,503 lookups on one dialogue turn, 5.0 s of a 75 s turn.
    retired: RwLock<HashMap<usize, Vec<Arc<ResidentPage>>>>,
    /// Per sequence, the pieces its pool holds and their resident pages at
    /// the sampled layers (`docs/hit_driven_projection.md` §7.2): index rows
    /// with no K/V behind them in the slot, read by the shadow scan. Shared
    /// through [`Self::pages`] with every sequence that pools or places the
    /// same piece, and released with the last of them.
    pooled: SequencePools,
    /// Per-sequence carried residual for the draft head's next first row — the
    /// `h(t-1)` its input assembly needs across a wave boundary. Empty on a
    /// checkpoint with no head, and reset with the other carried state when a
    /// sequence starts over. See [`super::draft`].
    pub(super) seeds: RwLock<SeedStore>,
    /// Each sequence's second seed buffer, which the next carry writes before
    /// it becomes the seed. Dropped wherever a seed is put back or taken away,
    /// so it never names a buffer [`Self::seeds`] holds. See [`SeedStore`].
    pub(super) seed_spares: RwLock<SeedStore>,
    /// The draft walk's two carried-residual buffers, sized for the widest
    /// cohort drafted so far. See [`Self::draft_carry`].
    pub(super) draft_carry: Mutex<Option<(Tensor, Tensor)>>,
    /// The capture's kept-row buffers between speculative steps, sized for the
    /// widest cohort verified so far. See [`Self::capture_rows_for`].
    pub(super) parked_rows: Mutex<Option<CaptureRows>>,
    /// Sequences whose carried state was put there **deliberately** — by a view
    /// carve or by a resume — and which must therefore survive exactly one
    /// `offset == 0` reset in [`Self::ensure_seq_state`].
    ///
    /// Without it the reset rule and the restore contradict each other: a slot
    /// holding state it did not compute legitimately stands at offset 0 (a fork
    /// borrows the parent's K/V; a resume installs its state before the first
    /// wave), so "offset 0 means start over" throws away precisely the state
    /// that was just installed. Every shape still matches and nothing is raised
    /// — the sequence simply answers as though it remembers nothing.
    ///
    /// One flag for all four carried classes, because they are always seeded
    /// together: a fork copies all of them, and a resume installs the GDN rows
    /// and the auxiliary blob from one record. Consumed on the first wave either
    /// way — a flag that outlived that wave would suppress a later, genuine
    /// reset, which is the recycled-slot defect wearing the fix's clothes. The
    /// rule is `HybridBatched::ensure_recurrent`'s, which reaches it through
    /// [`RecurrentStateStore::take_seeded`]; this model needs it to cover the
    /// PLE and index caches as well, so the flag lives beside them.
    pub(super) seeded: RwLock<HashSet<usize>>,
    /// Sequences whose carried state was claimed at admission
    /// ([`Self::claim_carried_state`]) and has not yet ridden a wave.
    ///
    /// That state is already sequence-start — a zeroed store, zeroed PLE
    /// history, zeroed seeds — so the first wave keeps it rather than building
    /// a second set beside it. Consumed by that wave; cleared by anything that
    /// writes the state from elsewhere.
    pub(super) pristine: RwLock<HashSet<usize>>,
    /// GDN snapshots held for sequences that have no store yet
    /// ([`Self::stage_recurrent`]): `(schedule hash, layers)`, the layers shared
    /// by every slot opened on one prompt branch. Installed when the sequence's
    /// state is first wanted — a claim at admission or a wave on it builds its
    /// store, and a fork from it is held the same snapshot — so a slot that has
    /// not been admitted holds host bytes it shares, not a store of its own.
    pub(super) staged: RwLock<HashMap<usize, StagedSnapshot>>,
    /// Sequences whose store was dropped once their last seal was durable
    /// ([`Self::park_recurrent`]). Their state is the record's until it is
    /// staged back; a want of it before then is refused rather than met with a
    /// sequence-start store, which would read fluently and remember nothing.
    pub(super) parked: RwLock<HashSet<usize>>,
    /// Armed only while a speculative block is in flight: what the verify wave
    /// must capture so a partial accept can be rewound ([`super::spec`]).
    /// `None` on every plain decode, which is what keeps the capture sites a
    /// single `is_none` check rather than a cost.
    pub(super) verify: RwLock<Option<SpecCapture>>,
    /// The tables the indexer rotates its queries and — as the scorer loads
    /// them — its stored keys from: [`Self::rope`]'s rungs, shared, plus each
    /// rung's step tables. The model's own schedule at its rotary width, rung
    /// for rung, which is what the reference rotates the indexer with
    /// (`docs/progressive_yarn.md` §12). Built once at load: it covers every
    /// position below `ROPE_REACH`, so nothing ever rebuilds it.
    index_rope: FactoredRope,
    /// Query rows for which a selection was built — QSA's engagement, summed
    /// over layers and waves. Zero says every row was inside the budget and
    /// the stack ran the dense arithmetic, which is what a short context
    /// should report.
    qsa_rows: AtomicU64,
    /// The schedule's rungs over the rotary width (`super::rope`), which every
    /// paged attention kernel rotates from; the layout puts the head's
    /// non-rotary dims past them, where the kernels treat them as pass-through.
    pub(super) rope: RopeRungs,
    /// The step open across a segmented sweep, if any — what its sequences
    /// entered it with ([`ManagedBatchedModel::begin_step`]).
    step: Mutex<Option<StepSnapshot>>,
    /// The KV layers that count hits — the selecting layers at the engine's
    /// percentiles (`docs/hit_driven_projection.md` §6.1), ascending.
    hit_kv: Vec<usize>,
    /// Per sequence, the pieces its projection declared and their hit
    /// counters ([`super::hits`]). A sequence with none declared is not
    /// counted.
    hits: RwLock<HashMap<usize, HitHolder>>,
    /// The hit kernel's span table as last uploaded, reused by every sampled
    /// layer of a wave ([`HitTable`]).
    hit_table: Mutex<HitTable>,
}

/// What a step's sequences carried when it opened: everything outside the K/V
/// and the recurrent stores (which keep their own step record) that a wave
/// writes — the index, the PLE history and the head's seed.
struct StepSnapshot {
    seqs: Vec<usize>,
    index: Vec<(usize, Vec<IndexSnapshot>)>,
    ple: Vec<(usize, PleState)>,
    seeds: Vec<(usize, Option<Tensor>)>,
}

/// The GDN state a fork's child takes from its parent: a copy of the store the
/// parent holds, or the snapshot held for a parent that has none yet.
enum Inherited {
    Store(RecurrentStateStore),
    Staged(StagedSnapshot),
}

mod delegates;
#[cfg(test)]
mod experts;
mod images;
mod layers;
mod model_core;
mod positional;
#[cfg(test)]
mod pricing;
mod qsa_wave;
mod recurrent;
#[cfg(test)]
mod session;
#[cfg(test)]
mod speculative;
#[cfg(test)]
mod step;
mod wave_sweep;

use images::{count_image_rows, take_wave_images};
use layers::*;
