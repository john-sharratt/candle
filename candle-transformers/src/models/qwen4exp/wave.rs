//! Qwen3.8-Flash-Next on the shared wave loop: `forward_wave` is
//! [`drive_wave`] and this model's architecture lives in its [`WaveSweep`] —
//! the 4-stream Gated Residual, the 3:1 GDN/attention hybrid schedule, the
//! 512-expert MoE on every layer, and the PLE injection at layer 1.
//!
//! The sweep is the oracle's algebra (`model.rs::forward_batched`) with the
//! production mixers swapped in: attention runs the paged KvCache kernels
//! through [`forward_attn_batched`] under this layer's QSA selection, GDN runs
//! the quantized span driver with this generation's sigmoid z-gate, and the
//! MoE is the shared [`Qwen35MoeBlock`] over the streamed `ExpertCache`. The
//! Gated Residual and PLE run as eager F32 tensor ops for this bring-up —
//! their fusion is the §0.4 work the design doc records.
//!
//! **QSA** (§3.1): every full-attention layer caches its index keys on every
//! wave, and once a row has more visible cells than the budget the layer's
//! [`SelectionTable`] narrows its read to the indexer's winners. Below the
//! budget the selection is the identity and is skipped entirely (§12.5), so a
//! short context runs the arithmetic it always ran.
//!
//! Carried state (the §6.3 classes): the GDN `S` + conv tails in per-sequence
//! [`RecurrentStateStore`]s (wave-atomic begin/commit/rollback), the PLE conv
//! history + hash window, and the QSA index caches — all three snapshotted at
//! wave entry so a failed wave leaves none of them advanced. The KV fourth
//! lives in the session's paged caches, rolled back by the wave driver's
//! default hooks — this model keeps every [`WaveSweep`] default: its offsets
//! equal its backing lengths.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use candle::quantized::cuda::{to_dynamic, DynamicActs};
use candle::{DType, Device, LiveTensor, Result, Tensor};
use candle_kernels::simple::qsa_topk::MAX_KEEP;
use candle_nn::kv_cache::{
    arena_regions, begin_forward, begin_wave, end_wave_transient, ffn_work_dtype,
    plan_wave_transient, region_stats, DeltaNetWidths, HyperWidths, KvCache, LayerPhase,
    ModelGeometry, SharedExpertWidths, SlotTenant, SpanRegion, WavePlan, WaveWidth, REGION_BYTES,
    WAVE_SPAN_BYTES,
};

use super::kv_row::kv_factors_for;

use super::batched_attention::Qwen4ExpAttentionLayer;
use super::capture_rows::CaptureRows;
use super::coverage::coverage_disagreements;
use super::draft::{HeadWave, SeedStore};
use super::engine::{GpuLayerMix, Qwen4ExpGpu};
use super::hyper::{hc_combine, hc_combine_gated, hc_mix, hc_mix_with_operand};
use super::indexer::{
    compact_index_caches, select_layer, selection_stride, IndexCache, IndexSnapshot,
};
use super::paged_index;
use super::paged_index::SealedIndex;
use super::ple::{PleSpan, PleState};
use super::ple_fused::ple_apply_spans_fused;
use super::qsa::IndexerWeights;
use super::qsa_select::{budget_fits_kernel, Strata};
use super::resident_page::{PageRegistry, ResidentPage};
use super::select_bytes::select_layer_bytes;
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
use crate::models::delta_net::StashSlot;
use crate::models::delta_net::{
    compact_stores, quantized_delta_net_layer_forward_spans, seq_spans, DeltaNetSeq,
    ExportedLayerState, LayerKind, RecurrentCompaction, RecurrentStateStore, SeqSpan, ZGate,
};
use crate::models::draft_ladder::QWEN38_FLASH_NEXT_DRAFT;
use crate::models::expert_lre::{WeightPlan, WeightPlanning};
use crate::models::head_rows::select_head_rows;
use crate::models::lazy_rope::LazyRope;
use crate::models::operand_guard::expect_dense_view;
use crate::models::piece_key::PieceKey;
use crate::models::prefill_utils::paged_decode_q8_head_dim;
use crate::models::prefill_utils::SharedPm;
use crate::models::profile::{span, ProfileSnapshot};
use crate::models::qsa_selection::QsaSelection;
use crate::models::qwen35::quantized_weights::SHARED_GATE_TILE;
use crate::models::qwen35::spec::{compact_verify_stash, split_block_rows, VerifyStash};
use crate::models::residency_rows::residency_decode_rows;
use crate::models::rope_schedule::{FactoredRope, RopeRungs, RopeSchedule, RungSelect};
use crate::models::selection_strata::StrataTokens;
use crate::models::wave_buffers::{wave_empty_ticketed, wave_from_vec_ticketed};

use super::rope::flash_next_schedule;
use crate::models::tensor_cat::TensorCat;
use crate::models::verify_wave::{upload_plan_rows, VerifyPlan};
use crate::models::wave_admit::admit_wave_kv;
use crate::models::wave_driver::{assemble_wave_contexts, drive_wave, WaveGroups, WaveSweep};
use crate::models::wave_token_ids::host_token_ids;
use crate::models::window_residuals::WindowResiduals;

/// Seal `c`'s live tail into a **position-free** record: its completed rows and
/// its open block, read back to the host.
///
/// The rows are un-rotated, so they carry no position already — the record is
/// injectable at any offset in any conversation, the index's half of "compute
/// once, inject anywhere". Read straight out of the cache's key pages; nothing is
/// gathered on the device first.
fn seal_live(c: &IndexCache, last_cells: usize) -> Result<SealedIndex> {
    Ok(SealedIndex {
        rows: c.live_rows_host()?,
        dim: c.head_dim(),
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
    /// Per-sequence GDN state (S + conv tails), wave-atomic.
    pub(super) recurrent: RwLock<HashMap<usize, RecurrentStateStore>>,
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
    /// Per sequence, the pages its last reset dropped, held until its next
    /// reset or its release.
    ///
    /// The registry holds pages weakly, so a page lives only while some cache
    /// holds it — and a reprojection resets the slot, the pieces' only holder,
    /// before re-injecting them. Without this every rebuild decodes and places
    /// every page it had a moment ago: measured at 12,372 placements against
    /// 12,503 lookups on one dialogue turn, 5.0 s of a 75 s turn.
    retired: RwLock<HashMap<usize, Vec<Arc<ResidentPage>>>>,
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
    /// The rewind stash and the kept rows between speculative steps, sized for
    /// the widest cohort verified so far. See [`Self::verify_stash_for`].
    pub(super) verify_stash: Mutex<Option<(VerifyStash, CaptureRows)>>,
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
}

/// The carried per-sequence state, and what persisting it costs.
///
/// This model carries four classes outside the paged K/V, and they divide on
/// one question — **is it cheaper to store or to rebuild?**
///
/// * **GDN recurrence** — an accumulated sum with no per-token decomposition.
///   Not derivable from anything; must be stored. [`RecurrentStateStore`]
///   already exports and imports it, geometry-checked against a schedule hash,
///   and is shared with the hybrid lineage.
/// * **PLE** — a `(conv_kernel − 1) × ngram_size` row convolution history plus
///   the last `ngram_size − 1` token ids. Kilobytes: stored, in the snapshot's
///   opaque auxiliary blob.
/// * **QSA index** — `n_blocks × head_dim` keys per attention layer, which at
///   128K tokens and ratio 4 is ~16 MiB a layer and ~192 MiB a sequence. That
///   is not a per-turn snapshot; it is **rebuilt** from the restored K by
///   [`Self::reindex_from_kv`].
/// * **Draft seeds** — one residual row per sequence, and only meaningful
///   inside the wave that produced it. Neither stored nor rebuilt: a resumed
///   sequence simply drafts nothing until its first wave seeds it, which costs
///   one step of speculation and no correctness.
impl Qwen4ExpBatched {
    /// The QSA selection budget, in positions.
    ///
    /// The checkpoint's own value is what production runs; this exists so a
    /// caller can widen it past the prompt and get the **same engine reading
    /// densely**. `selection_engages` is a comparison against this number, so a
    /// budget above the depth makes the selection the identity — the same code
    /// path over every cell, rather than a second build with the feature
    /// removed.
    ///
    /// That control is what makes a retrieval claim falsifiable. A needle the
    /// selected read loses says nothing on its own: it could be lost to the
    /// selection, or the checkpoint could simply not answer that prompt. Run
    /// both and the difference names which.
    ///
    /// Refused when the selection kernel could not run it — see
    /// [`budget_fits_kernel`].
    pub fn set_selection_budget(&mut self, top_k: usize) -> Result<()> {
        budget_fits_kernel(
            top_k,
            &self.attention_ratios(),
            self.model.cfg.max_position_embeddings,
            MAX_KEEP,
        )
        .map_err(candle::Error::Msg)?;
        // A strata set before the budget was checked against the old one.
        let strata = self.model.cfg.indexer.strata;
        if strata != Strata::WHOLE {
            self.check_strata(top_k, self.selecting_ratio()?, &strata)?;
        }
        self.model.cfg.indexer.top_k = top_k;
        Ok(())
    }

    /// The one compression ratio every selecting layer shares — what a strata
    /// stated in positions is converted at.
    fn selecting_ratio(&self) -> Result<usize> {
        let mut ratios: Vec<usize> = self
            .attention_ratios()
            .into_iter()
            .filter(|&r| r > 0)
            .collect();
        ratios.sort_unstable();
        ratios.dedup();
        match ratios.as_slice() {
            &[ratio] => Ok(ratio),
            _ => candle::bail!(
                "qwen4exp: a stratified selection needs one compression ratio across the \
                 selecting layers, found {ratios:?}"
            ),
        }
    }

    /// Whether the selection kernel can run `strata` under a budget of `top_k`
    /// at every depth the RoPE schedule reaches.
    ///
    /// A budget at or past that reach never selects, so any strata runs. Below
    /// it, the deepest row is checked with one window more than `reach / ratio`
    /// candidates cut into: a sequence assembled from sealed pages has a short
    /// block at every page boundary, so its candidates can outnumber
    /// `reach / ratio`, and the headroom covers every count of boundaries
    /// narrower than a window.
    fn check_strata(&self, top_k: usize, ratio: usize, strata: &Strata) -> Result<()> {
        let reach = self.rope.select().reach();
        if top_k >= reach {
            return Ok(());
        }
        selection_stride(ratio, top_k, reach / ratio + strata.window_blocks, strata)?;
        Ok(())
    }

    /// The budget [`Self::set_selection_budget`] is currently at.
    pub fn selection_budget(&self) -> usize {
        self.model.cfg.indexer.top_k
    }

    /// How the QSA selection divides a query's candidates before ranking them
    /// (`docs/qsa_stratified_selection.md`), stated in positions. The model
    /// loads at [`StrataTokens::WHOLE`], the checkpoint's own selection; an
    /// engine sets its own on top ([`StrataTokens::DEFAULT`] unless told
    /// otherwise).
    ///
    /// The positions become blocks at the selecting layers' compression ratio,
    /// which must be one ratio: a window of positions is a different number of
    /// blocks at each, and the indexer carries one strata. Checked against the
    /// selection kernel at the deepest position the RoPE schedule reaches,
    /// under the current budget ([`Self::check_strata`]): a strata whose windows
    /// would gather more entries than the kernel holds is refused here, at load,
    /// not at the depth that first reaches it.
    pub fn set_selection_strata(&mut self, tokens: StrataTokens) -> Result<()> {
        let ratio = self.selecting_ratio()?;
        let strata = Strata::from_tokens(tokens, ratio);
        self.check_strata(self.model.cfg.indexer.top_k, ratio, &strata)?;
        self.model.cfg.indexer.strata = strata;
        Ok(())
    }

    /// The strata [`Self::set_selection_strata`] is currently at.
    pub fn selection_strata(&self) -> Strata {
        self.model.cfg.indexer.strata
    }

    /// Declare that `seq`'s first `tokens` positions hold the conversation's
    /// system prompt — the span a stratified selection ranks in every window.
    ///
    /// Recorded on every one of the sequence's index caches, creating them if
    /// the sequence has not run a wave yet, so the declaration travels with the
    /// index through a fork, a move or a truncate.
    pub fn set_selection_prompt(&self, seq: usize, tokens: usize) -> Result<()> {
        let cfg = &self.model.cfg;
        let mut idx = self
            .index
            .write()
            .map_err(|_| candle::Error::Msg("index lock poisoned".into()))?;
        let caches = match idx.entry(seq) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(
                (0..cfg.kv_layers().total())
                    .map(|_| IndexCache::new(cfg.indexer.head_dim, &self.model.device))
                    .collect::<Result<Vec<_>>>()?,
            ),
        };
        for cache in caches.iter_mut() {
            cache.set_prompt_end(tokens);
        }
        Ok(())
    }

    /// Whether `seq` carries any of the recurrent classes yet.
    ///
    /// The fork and move sites consult this rather than erroring, because a
    /// view can legitimately be carved before its parent has ever run a wave —
    /// a brand-new conversation's first turn does exactly that, and there the
    /// child correctly starts from the sequence-start value.
    pub fn has_recurrent(&self, seq: usize) -> Result<bool> {
        Ok(self
            .recurrent
            .read()
            .map_err(|_| candle::Error::Msg("qwen4exp: recurrent lock poisoned".into()))?
            .contains_key(&seq))
    }

    /// How many sequences currently carry recurrent state.
    pub fn recurrent_len(&self) -> Result<usize> {
        Ok(self
            .recurrent
            .read()
            .map_err(|_| candle::Error::Msg("qwen4exp: recurrent lock poisoned".into()))?
            .len())
    }

    /// A view carve: `child` begins as an independent copy of `parent`'s
    /// carried state.
    ///
    /// **All three copied classes move together or none does.** The K/V the
    /// child borrows is one history; a child holding the parent's GDN state but
    /// an empty index would attend over blocks its selector never scored, and
    /// one holding the index but a zero PLE would inject an n-gram embedding
    /// computed from a window it never saw. Both read as a plausible answer.
    pub fn fork_recurrent(&self, parent: usize, child: usize) -> Result<()> {
        // **Each carried class is forked on its own terms.** They are populated
        // by different things and a parent can hold one without the others:
        // the recurrent store comes from running a wave, the PLE state from the
        // same, but the QSA index also comes from INJECTION — a projection
        // installs pages into a slot that has never decoded a token.
        //
        // This used to `return Ok(())` when the parent had no recurrent store,
        // on the reasoning that a parent which has not run a wave has nothing to
        // copy. True of the recurrence, false of the index, and the early return
        // took all three out together. The base conversation is exactly that
        // shape — an Arc-injected prefix of sections, never decoded — so every
        // conversation forked from it inherited a slot holding the base's K/V
        // and none of the index describing it. Silent until the prefix passed
        // the QSA identity threshold, at which point every first turn failed
        // with `a query at position N needs M blocks but the index cache holds
        // 0 in injected pages`.
        let forked = {
            let map = self
                .recurrent
                .read()
                .map_err(|_| candle::Error::Msg("qwen4exp: recurrent lock poisoned".into()))?;
            map.get(&parent)
                .map(|store| store.fork_from())
                .transpose()?
        };
        let ple = {
            let map = self
                .ple
                .read()
                .map_err(|_| candle::Error::Msg("qwen4exp: ple lock poisoned".into()))?;
            map.get(&parent).map(PleState::snapshot)
        };
        let index = {
            let map = self
                .index
                .read()
                .map_err(|_| candle::Error::Msg("qwen4exp: index lock poisoned".into()))?;
            match map.get(&parent) {
                Some(caches) => Some(
                    caches
                        .iter()
                        .map(IndexCache::fork)
                        .collect::<Result<Vec<_>>>()?,
                ),
                None => None,
            }
        };
        // Whether the parent has ever run a wave, read before its store is moved
        // into the child: it decides what a missing index means below.
        let parent_ran = forked.is_some();
        if let Some(f) = forked {
            self.recurrent
                .write()
                .map_err(|_| candle::Error::Msg("qwen4exp: recurrent lock poisoned".into()))?
                .insert(child, f);
        }
        if let Some(p) = ple {
            self.ple
                .write()
                .map_err(|_| candle::Error::Msg("qwen4exp: ple lock poisoned".into()))?
                .insert(child, p);
        }
        match index {
            Some(i) => {
                // What the child actually inherited, per layer. A carve that
                // silently hands over nothing looks identical downstream to a
                // model that keeps no index at all, and the difference decides
                // whether to look at the fork or at the parent's prefix.
                let inherited: Vec<usize> = i
                    .iter()
                    .zip(self.attention_ratios())
                    .filter(|(_, r)| *r > 0)
                    .map(|(c, r)| c.indexed_tokens(r))
                    .collect();
                tracing::trace!(
                    target: "candle_conversation::scheduler::reproject",
                    parent,
                    child,
                    layers = i.len(),
                    inherited_min = inherited.iter().copied().min().unwrap_or(0),
                    inherited_max = inherited.iter().copied().max().unwrap_or(0),
                    "qwen4exp: forked index caches to the view",
                );
                self.index
                    .write()
                    .map_err(|_| candle::Error::Msg("qwen4exp: index lock poisoned".into()))?
                    .insert(child, i);
            }
            // The parent had no caches at all. After a parent has run a wave —
            // it holds recurrent state — that is a loss: the child starts empty
            // and every token the view borrows is unindexed from birth. A parent
            // that has never run one is a fresh slot with nothing to inherit;
            // the tree summariser's scratch turn is carved from exactly that,
            // on whatever id a finalized view just freed.
            None if parent_ran => tracing::warn!(
                parent,
                child,
                "qwen4exp: the view's parent holds no index caches, so the carve \
                 inherits none — the borrowed K/V is unindexed from the start",
            ),
            None => {}
        }
        self.mark_seeded(child)?;
        Ok(())
    }

    /// Materialise the norm weights this model meets activations with.
    ///
    /// **Two widths, because this model computes wider than it stores.** The
    /// residual stream is BF16 (the lineage publishes `"dtype": "bfloat16"`),
    /// while a live sequence's K/V sits in the arena at F16 — `R16` is raw F16
    /// with Q-capture space. `kv` is the second, and passing the first in its
    /// place is not a rounding difference: the per-head Q/K norms run INSIDE
    /// the projection, on operands that *become* the arena's bytes, so a
    /// BF16 weight there meets an F16 activation and `weight_for` refuses it —
    /// every wave, from the first ingest, with the model fully loaded and
    /// nothing else wrong.
    ///
    /// The Gated Residual runs F32 end to end and the projections quantize
    /// their own activations, so the Q/K norms are the only weights that need
    /// materialising; both are done here, at session creation, never inside the
    /// wave.
    ///
    /// The draft head's block is included deliberately. It is a full-attention
    /// layer with Q/K norms of its own and it was POPPED out of `layers` at
    /// load, so a loop over `layers` alone leaves exactly one attention layer
    /// holding unconverted norms, and the head's first pass fails inside the
    /// wave rather than here.
    pub fn maybe_change_dtype(&self, _act: DType, kv: DType) -> Result<()> {
        let head_block = self.model.mtp.as_ref().map(|h| &h.block);
        for layer in self.model.layers.iter().chain(head_block) {
            if let GpuLayerMix::Attention { w, .. } = &layer.mix {
                w.q_norm.maybe_change_dtype(kv)?;
                w.k_norm.maybe_change_dtype(kv)?;
            }
        }
        Ok(())
    }

    /// Drop `seq`'s index — the slot's K/V was truncated to nothing.
    ///
    /// The pages it held stay resident until `seq` resets again or is released
    /// (see the `retired` field), so the rebuild that follows a reset finds the
    /// pieces it re-injects already placed.
    pub fn reset_positional_state(&self, seq: usize) -> Result<()> {
        let mut dropped: Vec<Arc<ResidentPage>> = Vec::new();
        if let Ok(mut map) = self.index.write() {
            if let Some(caches) = map.get_mut(&seq) {
                for c in caches.iter_mut() {
                    dropped.extend(c.reset());
                }
            }
        }
        if let Ok(mut retired) = self.retired.write() {
            retired.insert(seq, dropped);
        }
        Ok(())
    }

    /// The largest position at or before `tokens` that every one of `seq`'s
    /// index caches can be cut back to (see [`IndexCache::cut_floor`]).
    pub fn positional_cut_floor(&self, seq: usize, tokens: usize) -> Result<usize> {
        let map = self
            .index
            .read()
            .map_err(|_| candle::Error::Msg("qwen4exp: index lock poisoned".into()))?;
        Ok(map.get(&seq).map_or(tokens, |caches| {
            caches
                .iter()
                .map(|c| c.cut_floor(tokens))
                .min()
                .unwrap_or(tokens)
        }))
    }

    /// Cut `seq`'s index back to `tokens` — the slot's K/V was truncated to a
    /// piece boundary there, and what follows is about to be re-injected. The
    /// pages above the cut are retired like a reset's (see the `retired` field).
    pub fn truncate_positional_state(&self, seq: usize, tokens: usize) -> Result<()> {
        let mut dropped: Vec<Arc<ResidentPage>> = Vec::new();
        {
            let mut map = self
                .index
                .write()
                .map_err(|_| candle::Error::Msg("qwen4exp: index lock poisoned".into()))?;
            if let Some(caches) = map.get_mut(&seq) {
                for c in caches.iter_mut() {
                    dropped.extend(c.truncate_to(tokens)?);
                }
            }
        }
        if let Ok(mut retired) = self.retired.write() {
            retired.insert(seq, dropped);
        }
        Ok(())
    }

    /// Close `seq`'s index on a block boundary and hand back the page.
    ///
    /// The flush is what makes the piece self-contained: a section's tokens do
    /// not end on a block boundary, so `T mod ratio` rows sit carried, and a
    /// page without them would leave those tokens indexed by a block the *next*
    /// section completes. The flushed block summarises fewer than `ratio`
    /// tokens and is deliberately not what a continuous run would have produced
    /// for that span — the scorer carries each page's width and so can express
    /// a short block, where a block pooled across two sections is not
    /// correctable at all.
    #[cfg(feature = "cuda")]
    pub fn seal_positional_state(&self, seq: usize) -> Result<Option<Vec<u8>>> {
        let cfg = &self.model.cfg;
        let ratios = self.attention_ratios();
        let indexers = self.attention_indexers();
        let mut map = self
            .index
            .write()
            .map_err(|_| candle::Error::Msg("qwen4exp: index lock poisoned".into()))?;
        let Some(caches) = map.get_mut(&seq) else {
            // Named, not silent. A slot with no index caches at seal time seals
            // a piece with no page, and the caller has no way to tell that from
            // "this model keeps no per-position state" — which is the confusion
            // that let an unindexed system prompt through.
            tracing::warn!(
                seq,
                live_slots = map.len(),
                "qsa seal: slot has no index caches, so this piece seals with no page — \
                 nothing ever appended to it, or its state was reset since",
            );
            return Ok(None);
        };
        let mut pages = Vec::with_capacity(caches.len());
        for ((c, &ratio), w) in caches.iter_mut().zip(ratios.iter()).zip(indexers.iter()) {
            if ratio == 0 {
                pages.push(seal_live(c, 1)?);
                continue;
            }
            // Sealing is not a forward: no phase is open to carve from. The
            // flush consumes the carried rows, so the page IS the whole piece
            // and there is no open block to carry with it.
            let cells = c.flush_open_block(w, cfg.rms_norm_eps)?;
            pages.push(seal_live(c, cells.unwrap_or(ratio))?);
        }
        Ok(Some(paged_index::encode_aux(&[], &pages)?))
    }

    /// Close `seq`'s index on a page boundary, on every attention layer.
    ///
    /// Called at a reasoning boundary during decode, so a turn's
    /// `<think>…</think>` occupies whole pages and can be dropped exactly when
    /// the turn is projected as history. Every layer closes together — they
    /// index one stream, so a cut on some of them would leave the rest
    /// addressing different blocks for the same position.
    ///
    /// Returns the tokens the closed page covers — `0` when the tail was already
    /// empty, which is the common case and a legitimate no-op.
    #[cfg(feature = "cuda")]
    pub fn close_positional_page(&self, seq: usize) -> Result<usize> {
        let cfg = &self.model.cfg;
        let ratios = self.attention_ratios();
        let indexers = self.attention_indexers();
        let mut map = self
            .index
            .write()
            .map_err(|_| candle::Error::Msg("qwen4exp: index lock poisoned".into()))?;
        let Some(caches) = map.get_mut(&seq) else {
            return Ok(0);
        };
        // Every layer indexes the same stream, so they close the same width;
        // taken from whichever is walked last.
        let mut closed = 0usize;
        for ((c, &ratio), w) in caches.iter_mut().zip(ratios.iter()).zip(indexers.iter()) {
            if ratio == 0 {
                continue;
            }
            closed = c.close_tail_into_page(w, ratio, cfg.rms_norm_eps)?;
        }
        Ok(closed)
    }

    /// The page covering `seq`'s tokens from `start_pos` onward.
    ///
    /// **Taken on a fork, because this slot keeps decoding.** A section is
    /// ingested on a throwaway slot, so closing its cache in place costs
    /// nothing; a conversation turn is sealed on the slot that produced it and
    /// the flush would leave the live cache with a short block mid-sequence,
    /// which every later position would then be addressed through. The fork is
    /// `n_blocks × head_dim` floats per attention layer — the same shape of cost
    /// a view carve already pays.
    ///
    /// `start_pos` is where the turn's own K/V begins. Rows below it belong to
    /// what the projection injected ahead of this turn and are already pages of
    /// their own; carrying them again would place the same block at two
    /// positions.
    #[cfg(feature = "cuda")]
    pub fn seal_positional_range(&self, seq: usize, start_pos: usize) -> Result<Option<Vec<u8>>> {
        let cfg = &self.model.cfg;
        let ratios = self.attention_ratios();
        let indexers = self.attention_indexers();
        let map = self
            .index
            .read()
            .map_err(|_| candle::Error::Msg("qwen4exp: index lock poisoned".into()))?;
        let Some(caches) = map.get(&seq) else {
            return Ok(None);
        };
        let mut pages = Vec::with_capacity(caches.len());
        for ((c, &ratio), w) in caches.iter().zip(ratios.iter()).zip(indexers.iter()) {
            if ratio == 0 {
                pages.push(seal_live(c, 1)?);
                continue;
            }
            // **Two row spaces, and they are not the same one.**
            // `candidates_at` counts blocks over the WHOLE sequence — the
            // injected pages ahead of this turn and the live tail together —
            // while `live_rows` is the tail alone. Subtracting the page span
            // converts the global block index into an offset into the tail,
            // which is where a turn's own rows live. Without it the offset
            // overshoots by exactly the number of rows the projection injected,
            // and the narrow is refused.
            //
            // Saturating below the pages; refused above the tail. A `start_pos`
            // that lands inside the injected pages has no offset in tail row
            // space, and silently clamping it would hand back a page describing
            // a different span than the caller asked for — rows for the wrong
            // positions, which selects fluently against the wrong context. Said
            // plainly here instead: `narrow` reports this as
            // `start > dim_len` naming neither the sequence nor the span, from
            // a call site several layers removed.
            let first = if start_pos == 0 {
                0
            } else {
                c.candidates_at(start_pos - 1, ratio)
                    .saturating_sub(c.page_row_span())
            };
            let mut fork = c.fork()?;
            let cells = fork.flush_open_block(w, cfg.rms_norm_eps)?;
            let d = fork.head_dim();
            let rows = fork.live_rows_host()?;
            let n = rows.len() / d;
            if first > n {
                candle::bail!(
                    "qsa seal: seq {seq} asked for the rows from position {start_pos}, which \
                     is row {first} of a {n}-row live tail — the span starts inside the \
                     injected pages, whose rows this cannot return. Seal the span from the \
                     sequence that forwarded it, where it is the tail."
                );
            }
            pages.push(SealedIndex {
                rows: rows[first * d..].to_vec(),
                dim: d,
                last_cells: cells.unwrap_or(ratio),
                open: fork.open_rows_host()?,
            });
        }
        Ok(Some(paged_index::encode_aux(&[], &pages)?))
    }

    /// Seal the rows for the last `tokens` tokens of `seq`, wherever they live.
    ///
    /// **Why this is not [`Self::seal_positional_range`].** That one returns
    /// live-tail rows only, which is right for a span the sequence forwarded
    /// and still holds. A turn's rows do not stay there: a reprojection
    /// re-injects the turn's user half from cache and pushes it as a *page*, so
    /// after one reprojection the turn is split — its opening tokens sit in a
    /// page and only its decoded tail sits in the live rows. Sealing the tail
    /// alone then drops exactly the user's message, and the model decodes a
    /// turn whose question it cannot attend to; measured, it invents a
    /// different question and answers that instead.
    ///
    /// Returns one blob per source page in ascending order, with the live tail
    /// last. They are pushed in that order, which preserves each piece's own
    /// ragged width — merging them into one page would re-pool rows across
    /// boundaries the pieces ended at.
    pub fn seal_positional_tail_span(
        &self,
        seq: usize,
        tokens: usize,
    ) -> Result<Vec<(usize, Vec<u8>)>> {
        if tokens == 0 {
            return Ok(Vec::new());
        }
        let cfg = &self.model.cfg;
        let ratios = self.attention_ratios();
        let indexers = self.attention_indexers();
        let map = self
            .index
            .read()
            .map_err(|_| candle::Error::Msg("qwen4exp: index lock poisoned".into()))?;
        let Some(caches) = map.get(&seq) else {
            return Ok(Vec::new());
        };
        // The page structure is the same on every layer — they index one stream
        // — so the walk is decided once, on the first live cache.
        let Some((probe, &probe_ratio)) = caches.iter().zip(ratios.iter()).find(|(_, r)| **r > 0)
        else {
            return Ok(Vec::new());
        };
        let (blocks, open) = probe.seal_shape();
        let tail_tokens = blocks * probe_ratio + open;
        // Whole trailing pages, until the span is covered — but never a page
        // that would reach back past the span's own start. The decision is
        // [`paged_index::tail_span_pages`], a free function so the boundary
        // arithmetic is pinned by unit tests rather than by a live daemon; see
        // its notes for the regression that shaped it.
        let widths: Vec<usize> = (0..probe.page_count())
            .map(|i| probe.page_at(i).map_or(0, |(_, w)| w))
            .collect();
        let (first_page, covered) = paged_index::tail_span_pages(&widths, tail_tokens, tokens);
        if covered != tokens {
            // **This is an alarm, not a mode.** The walk's refusal of an
            // over-wide page is a backstop: a unit's rows begin on a page
            // boundary because the scheduler closes one when the unit's K/V
            // anchor is taken (`Scheduler::begin_unit`), so reaching here means
            // some path opened a unit without passing that boundary. The
            // shortfall is the width of what this turn shares a page with.
            //
            // The AVAILABLE page widths, not just the taken ones, and the width
            // of the page the walk stopped at. Without those two the caller can
            // see only that the span does not line up, which is the question
            // rather than the answer: the refused page's width is what names the
            // piece this turn's rows share a page with.
            let refused = first_page.checked_sub(1).map(|i| widths[i]);
            tracing::warn!(
                seq,
                tokens,
                covered,
                tail_tokens,
                first_page,
                refused_page_width = ?refused,
                all_page_widths = ?widths,
                "qsa seal: a turn's pages cover {} token(s) {} the turn — its rows do not \
                 begin on a page boundary, so a unit began without one being closed",
                covered.abs_diff(tokens),
                if covered > tokens {
                    "more than"
                } else {
                    "less than"
                },
            );
        }

        // Each blob carries the tokens it covers. The caller cannot read a page
        // — the bytes are this model's — so the width is the only handle it has
        // on where a page sits in the span, and it needs one: the projection
        // drops the pages covering a turn's reasoning, and the page COUNT is not
        // stable (a mid-decode reprojection closes an extra one). Position is.
        let mut blobs: Vec<(usize, Vec<u8>)> = Vec::new();
        for pi in first_page..probe.page_count() {
            let mut layer_pages = Vec::with_capacity(caches.len());
            // Two descriptions of the same page travel together from here: the
            // width, which the caller uses to place the page in the span, and
            // the rows the blob carries, which only this model can read. They
            // are the one thing nothing downstream can cross-check — so they are
            // checked here, where both are in hand.
            let mut width: Option<usize> = None;
            for c in caches.iter() {
                let (p, w) = c
                    .page_at(pi)
                    .ok_or_else(|| candle::Error::Msg(format!("qsa seal: page {pi} vanished")))?;
                // Every layer indexes one stream, so a page spans the same
                // tokens on all of them. A layer that disagrees would ship rows
                // for a different span than the width the caller places them by,
                // and the projection would window out the wrong tokens on that
                // layer alone — visible only as degraded retrieval.
                match width {
                    None => width = Some(w),
                    Some(first) if first != w => candle::bail!(
                        "qsa seal: page {pi} is {first} token(s) wide on the first layer and {w} \
                         on another — the layers have diverged and the seal cannot say which \
                         span this page covers"
                    ),
                    Some(_) => {}
                }
                // A closed page's rows are un-rotated and immutable, so the
                // record is the page's rows as they stand. A page is already
                // closed; only the live tail carries an open block.
                layer_pages.push(SealedIndex {
                    rows: p.host_rows()?,
                    dim: c.head_dim(),
                    last_cells: p.last_cells(),
                    open: Vec::new(),
                });
            }
            blobs.push((
                width.unwrap_or(0),
                paged_index::encode_aux(&[], &layer_pages)?,
            ));
        }

        if tail_tokens > 0 {
            let mut layer_pages = Vec::with_capacity(caches.len());
            for ((c, &ratio), w) in caches.iter().zip(ratios.iter()).zip(indexers.iter()) {
                if ratio == 0 {
                    layer_pages.push(seal_live(c, 1)?);
                    continue;
                }
                let mut fork = c.fork()?;
                let cells = fork.flush_open_block(w, cfg.rms_norm_eps)?;
                layer_pages.push(seal_live(&fork, cells.unwrap_or(ratio))?);
            }
            blobs.push((tail_tokens, paged_index::encode_aux(&[], &layer_pages)?));
        }
        Ok(blobs)
    }

    /// The resident pages of an injected piece, one per KV layer (`None` where
    /// the layer indexes nothing): the ones already standing when any slot still
    /// holds this record's pages, otherwise decoded and placed now — every
    /// indexed layer in one launch, into slots of the index tenant.
    #[cfg(feature = "cuda")]
    fn resident_pages(
        &self,
        blob: &[u8],
        key: &PieceKey,
    ) -> Result<Vec<Option<Arc<ResidentPage>>>> {
        let cfg = &self.model.cfg;
        let want = cfg.kv_layers().total();
        if let Some(layers) = self.pages.get(key) {
            return Ok(layers);
        }
        let _place = span("qsa:page:decode_place");
        let (_, sealed) = paged_index::decode_aux(blob)?;
        if sealed.len() != want {
            candle::bail!(
                "qwen4exp: an injected piece carries {} index pages but this checkpoint has \
                 {want} KV layers — the layers it does not cover would select against an \
                 empty candidate set",
                sealed.len()
            );
        }
        let ratios = self.attention_ratios();
        let host: Vec<(&[f32], usize)> = sealed
            .iter()
            .zip(ratios.iter())
            .filter(|(_, &r)| r > 0)
            .map(|(s, _)| (s.rows.as_slice(), s.last_cells))
            .collect();
        let mut placed =
            ResidentPage::place_host(&host, cfg.indexer.head_dim, &self.model.device)?.into_iter();
        let layers: Vec<Option<Arc<ResidentPage>>> = ratios
            .iter()
            .map(|&r| if r > 0 { placed.next() } else { None })
            .collect();
        self.pages.insert(*key, &layers);
        Ok(layers)
    }

    /// Install a sealed page per attention layer ahead of `seq`'s live tail.
    /// `key` is the page's [`PieceKey`], taken by the caller.
    #[cfg(feature = "cuda")]
    pub fn push_positional_state(&self, seq: usize, blob: &[u8], key: &PieceKey) -> Result<()> {
        let cfg = &self.model.cfg;
        let want = cfg.kv_layers().total();
        let ratios = self.attention_ratios();
        // Resolved before the lock is taken: a placement is a launch.
        let layers = self.resident_pages(blob, key)?;
        let mut map = self
            .index
            .write()
            .map_err(|_| candle::Error::Msg("qwen4exp: index lock poisoned".into()))?;
        let caches = map.entry(seq).or_default();
        if caches.is_empty() {
            for _ in 0..want {
                caches.push(IndexCache::new(cfg.indexer.head_dim, &self.model.device)?);
            }
        }
        let mut pushed = 0usize;
        let mut had_open_tail = None;
        for ((c, &ratio), page) in caches.iter_mut().zip(ratios.iter()).zip(layers) {
            let Some(page) = page else { continue };
            if had_open_tail.is_none() {
                let (blocks, open) = c.seal_shape();
                had_open_tail = Some(blocks * ratio + open);
            }
            pushed = page.tokens(ratio);
            // Abutting the last placement. The base is where the scorer rotates
            // the page's rows to, so this is the one line that decides where the
            // piece's rows actually sit — and, since it is recorded rather than
            // accumulated, a preceding piece that carried no rows moves it and
            // nothing else.
            let base = c.next_base();
            c.push_page(page, base, ratio)?;
        }
        // **Only the pathological case is reported.** A page installed onto an
        // empty tail is the ordinary path and happens once per injected piece —
        // 1902 times in one startup sweep, which is noise, not evidence. A page
        // installed while the tail still holds live rows is the opposite: the
        // page lands *after* rows that chronologically precede it, so every
        // position past it resolves through the wrong block.
        if had_open_tail.is_some_and(|t| t > 0) {
            tracing::warn!(
                seq_id = seq,
                pushed,
                open_tail = had_open_tail.unwrap_or(0),
                "index: an injected page landed on a live tail — its rows sit before \
                 rows that precede them, so positions past it resolve through the wrong \
                 block"
            );
        }
        Ok(())
    }

    /// Account for injected K/V that carries no index rows, on every layer.
    ///
    /// Every layer indexes one stream, so they all move the same distance —
    /// moving some of them would leave the rest addressing different blocks for
    /// the same position, which is the divergence this exists to prevent.
    ///
    /// All this does is advance where the next page opens. It used to have to do
    /// more: positions were the running sum of the page widths, so a span that
    /// contributed none slid every later page's implied start earlier by its
    /// width, and a zero-row page had to stand in to keep the sum honest.
    /// Positions are recorded now, so an unindexed span is simply not indexed.
    pub fn push_positional_gap(&self, seq: usize, tokens: usize) -> Result<()> {
        if tokens == 0 {
            return Ok(());
        }
        let cfg = &self.model.cfg;
        let want = cfg.kv_layers().total();
        let mut map = self
            .index
            .write()
            .map_err(|_| candle::Error::Msg("qwen4exp: index lock poisoned".into()))?;
        let caches = map.entry(seq).or_default();
        if caches.is_empty() {
            for _ in 0..want {
                caches.push(IndexCache::new(cfg.indexer.head_dim, &self.model.device)?);
            }
        }
        for c in caches.iter_mut() {
            let to = c.next_base() + tokens;
            c.skip_to(to)?;
        }
        Ok(())
    }

    /// Declare `seq`'s carried state deliberately installed, so the next wave
    /// does not reset it out from under the caller. See the `seeded` field.
    pub(super) fn mark_seeded(&self, seq: usize) -> Result<()> {
        self.seeded
            .write()
            .map_err(|_| candle::Error::Msg("qwen4exp: seeded lock poisoned".into()))?
            .insert(seq);
        Ok(())
    }

    /// A view finalizes: `child`'s state becomes `parent`'s.
    ///
    /// A move, not a merge — a view is a linear continuation of its parent, so
    /// what it holds now is what the parent's state becomes. The child's
    /// entries are gone afterwards and the parent's previous ones are dropped.
    pub fn move_recurrent(&self, child: usize, parent: usize) -> Result<()> {
        let store = {
            let mut map = self
                .recurrent
                .write()
                .map_err(|_| candle::Error::Msg("qwen4exp: recurrent lock poisoned".into()))?;
            match map.remove(&child) {
                Some(s) => s,
                // The view never ran a wave; the parent keeps what it had.
                None => return Ok(()),
            }
        };
        self.recurrent
            .write()
            .map_err(|_| candle::Error::Msg("qwen4exp: recurrent lock poisoned".into()))?
            .insert(parent, store);
        {
            let mut map = self
                .ple
                .write()
                .map_err(|_| candle::Error::Msg("qwen4exp: ple lock poisoned".into()))?;
            if let Some(p) = map.remove(&child) {
                map.insert(parent, p);
            }
        }
        {
            let mut map = self
                .index
                .write()
                .map_err(|_| candle::Error::Msg("qwen4exp: index lock poisoned".into()))?;
            if let Some(i) = map.remove(&child) {
                map.insert(parent, i);
            }
        }
        {
            let mut map = self
                .seeds
                .write()
                .map_err(|_| candle::Error::Msg("qwen4exp: seeds lock poisoned".into()))?;
            if let Some(s) = map.remove(&child) {
                map.insert(parent, s);
            }
        }
        {
            // The spare follows its seed, so the parent's pair stays two
            // distinct buffers; a parent whose seed was not replaced keeps its
            // own spare.
            let mut map = self
                .seed_spares
                .write()
                .map_err(|_| candle::Error::Msg("qwen4exp: seeds lock poisoned".into()))?;
            if let Some(s) = map.remove(&child) {
                map.insert(parent, s);
            }
        }
        Ok(())
    }

    /// Read `seq`'s GDN state back as the snapshot record's layer rows.
    ///
    /// `None` when the sequence carries none — a slot that has never run a wave
    /// has nothing worth persisting, and writing a zero snapshot would be worse
    /// than writing none: resume would install it and report success.
    ///
    /// **This is the GDN state only.** The PLE window rides in the record's
    /// auxiliary blob ([`Self::export_aux_state`]) and the QSA index is rebuilt
    /// rather than stored — see this impl block's header for why the three
    /// classes are treated differently.
    pub fn export_recurrent(&self, seq: usize) -> Result<Option<(u64, Vec<ExportedLayerState>)>> {
        let map = self
            .recurrent
            .read()
            .map_err(|_| candle::Error::Msg("qwen4exp: recurrent lock poisoned".into()))?;
        let Some(store) = map.get(&seq) else {
            return Ok(None);
        };
        // `export` refuses mid-wave itself; the seal runs outside the wave, so
        // reaching that error means the ordering broke.
        let layers = store.export()?;
        Ok(Some((store.schedule_hash(), layers)))
    }

    /// Scatter a snapshot into `seq`'s GDN state — the resume path.
    ///
    /// Creates the store if the slot has none yet, which is the normal case:
    /// resume runs at `create_sequence`, before any wave. `import` validates
    /// the schedule hash and every layer's geometry before touching a tensor.
    pub fn restore_recurrent(
        &self,
        seq: usize,
        schedule_hash: u64,
        layers: &[ExportedLayerState],
    ) -> Result<()> {
        let cfg = &self.model.cfg;
        let mut map = self
            .recurrent
            .write()
            .map_err(|_| candle::Error::Msg("qwen4exp: recurrent lock poisoned".into()))?;
        let store = match map.entry(seq) {
            std::collections::hash_map::Entry::Occupied(slot) => slot.into_mut(),
            std::collections::hash_map::Entry::Vacant(slot) => slot.insert(
                RecurrentStateStore::new(&cfg.layer_kinds, &cfg.delta_net, &self.model.device)?,
            ),
        };
        store.import(schedule_hash, layers)?;
        drop(map);
        self.mark_seeded(seq)
    }

    /// The carried state that is not a DeltaNet layer stack — the PLE window
    /// and one QSA index page per attention layer — for the turn record's
    /// auxiliary slot.
    ///
    /// Each layer's record carries the cache's completed rows AND its carried
    /// open block, so a resumed sequence's index covers exactly the tokens its
    /// restored K/V does — including the `T mod ratio` that had not completed a
    /// row when the turn ended, which is the usual case rather than an edge one.
    /// Rows above `n_blocks` (and `n_open`) are dead until an append writes them
    /// and are not persisted.
    ///
    /// `last_cells` is `ratio`: every row this cache completed is a full one.
    /// A page whose last row is SHORT is what a per-turn seal emits after
    /// [`IndexCache::flush_open_block`], for a window reconstructed from several
    /// turns' pieces — the container's format is the same either way, which is
    /// what lets both forms share one record.
    pub fn export_aux_state(&self, seq: usize) -> Result<Option<Vec<u8>>> {
        let ple_blob = {
            let map = self
                .ple
                .read()
                .map_err(|_| candle::Error::Msg("qwen4exp: ple lock poisoned".into()))?;
            match map.get(&seq) {
                Some(state) => state.encode()?,
                None => return Ok(None),
            }
        };
        let layers = {
            let map = self
                .index
                .read()
                .map_err(|_| candle::Error::Msg("qwen4exp: index lock poisoned".into()))?;
            match map.get(&seq) {
                Some(caches) => {
                    let ratios = self.attention_ratios();
                    let mut v = Vec::with_capacity(caches.len());
                    for (c, ratio) in caches.iter().zip(ratios.iter()) {
                        // The live tail's rows as they are stored — un-rotated,
                        // so position-free — and `restore_aux_state` restores
                        // them into a cache whose tail opens at zero. (The
                        // injected pages ahead of the tail are not exported at
                        // all — a resume rebuilds them from the projection, and
                        // `indexed_tokens` is what reports the shortfall if one
                        // does not.)
                        v.push(seal_live(c, (*ratio).max(1))?);
                    }
                    v
                }
                None => Vec::new(),
            }
        };
        Ok(Some(paged_index::encode_aux(&ple_blob, &layers)?))
    }

    /// Install a PLE window and the QSA index pages from a snapshot's
    /// auxiliary blob.
    ///
    /// Refuses a page count that does not match this checkpoint's KV layers
    /// rather than installing a partial index: a sequence whose deepest
    /// attention layers had no index would select against an empty candidate
    /// set there and attend to nothing, which reads as a retrieval that simply
    /// found nothing relevant.
    pub fn restore_aux_state(&self, seq: usize, blob: &[u8]) -> Result<()> {
        let cfg = &self.model.cfg;
        let (ple_blob, layers) = paged_index::decode_aux(blob)?;
        let state = PleState::decode(
            &ple_blob,
            cfg.ple.conv_history(),
            cfg.hc.dim(cfg.hidden_size),
            &self.model.device,
        )?;
        let want = cfg.kv_layers().total();
        if !layers.is_empty() && layers.len() != want {
            candle::bail!(
                "qwen4exp: the snapshot carries {} index pages but this checkpoint has {want} \
                 KV layers — a partial index would select against an empty candidate set on \
                 the layers it does not cover",
                layers.len()
            );
        }
        self.ple
            .write()
            .map_err(|_| candle::Error::Msg("qwen4exp: ple lock poisoned".into()))?
            .insert(seq, state);
        if !layers.is_empty() {
            let mut caches = Vec::with_capacity(layers.len());
            for s in &layers {
                caches.push(IndexCache::from_rows(
                    &s.rows,
                    &s.open,
                    cfg.indexer.head_dim,
                    &self.model.device,
                )?);
            }
            self.index
                .write()
                .map_err(|_| candle::Error::Msg("qwen4exp: index lock poisoned".into()))?
                .insert(seq, caches);
        }
        self.mark_seeded(seq)
    }

    /// Bytes the carried state holds for every live sequence.
    ///
    /// Reported because it is large and it moves: this is a span-reservation
    /// tenant, and a total that omits it makes the partition look emptier than
    /// it is — the same blindness that let the dense weights hide.
    pub fn recurrent_reserved_bytes(&self) -> usize {
        // Every tenant arena's regions, not a sum over the holders: holders share
        // arenas, so their sum leaves out the free slots and unused tails.
        let gdn = RecurrentStateStore::arena_reserved_bytes(&self.model.device);
        let idx = arena_regions(&self.model.device, SlotTenant::QsaIndex) * SpanRegion::bytes();
        gdn + idx
    }

    /// Compact every arena the carried state lives in: each sequence's GDN state
    /// ([`compact_stores`]), each QSA index cache ([`compact_index_caches`]), and
    /// the speculative rewind stash ([`compact_verify_stash`]). The report sums all
    /// three. Between forwards; the locks are held for the pass, which is what
    /// keeps a wave from opening on a store, a cache or a stash mid-move.
    ///
    /// The stash's regions are counted into the recurrent-state figure rather than
    /// its own, matching `RecurrentStateStore::arena_reserved_bytes` — it is the
    /// buffer set that exists to rewind that state, and splitting the two across
    /// reports would leave neither total reconcilable. They are read here rather
    /// than inside `compact_stores`, because a `regions_released` that omitted the
    /// stash would leave `reclaim_spare_ground` unrun on exactly the passes that
    /// freed stash ground.
    pub fn compact_recurrent(&self, max_moves: usize) -> Result<RecurrentCompaction> {
        let mut map = self
            .recurrent
            .write()
            .map_err(|_| candle::Error::Msg("qwen4exp: recurrent lock poisoned".into()))?;
        let gdn = compact_stores(
            map.values_mut(),
            &self.model.cfg.delta_net,
            &self.model.device,
            max_moves,
        )?;
        let mut idx = self
            .index
            .write()
            .map_err(|_| candle::Error::Msg("index lock poisoned".into()))?;
        let mut caches: Vec<&mut IndexCache> = idx.values_mut().flatten().collect();
        let qsa = compact_index_caches(
            &mut caches,
            self.model.cfg.indexer.head_dim,
            &self.model.device,
            max_moves,
        )?;
        // The stash is armed inside a verify's capture or parked between steps —
        // the same buffers either way, and both are on this thread between
        // forwards, where a move cannot race a capture or a replay.
        let stash_before = arena_regions(&self.model.device, SlotTenant::RewindStash);
        let stash = {
            let mut g = self
                .verify
                .write()
                .map_err(|_| candle::Error::Msg("verify lock poisoned".into()))?;
            let mut parked = self
                .verify_stash
                .lock()
                .map_err(|_| candle::Error::Msg("verify stash lock poisoned".into()))?;
            let target = match g.as_mut() {
                Some(cap) => Some(&mut cap.delta),
                None => parked.as_mut().map(|(delta, _)| delta),
            };
            match target {
                Some(delta) => compact_verify_stash(
                    delta,
                    &self.model.cfg.delta_net,
                    &self.model.device,
                    max_moves,
                )?,
                None => (0, 0),
            }
        };
        let stash_after = arena_regions(&self.model.device, SlotTenant::RewindStash);
        Ok(RecurrentCompaction {
            planned: gdn.planned + qsa.planned + stash.0,
            moved: gdn.moved + qsa.moved + stash.1,
            regions_before: gdn.regions_before + qsa.regions_before + stash_before,
            regions_after: gdn.regions_after + qsa.regions_after + stash_after,
        })
    }

    /// What admitting **one** more sequence costs in carried state.
    ///
    /// The GDN store alone, priced from the geometry
    /// ([`RecurrentStateStore::reserved_bytes_for`]) — every store this model
    /// builds has the same shape, so the config answers for all of them, and it
    /// answers before the first one exists, which is when admission asks.
    ///
    /// **The index caches are deliberately not in it.** They are the other half
    /// of [`Self::recurrent_reserved_bytes`], but their key pages grow with the
    /// context the sequence has already decoded, so a *new* sequence brings only
    /// its open-block buffers — a few KiB per layer. Charging a fresh turn for the
    /// index a long conversation has accumulated would price arrivals by the
    /// depth of the sequences already resident, which is the mistake this
    /// function exists to end.
    pub fn recurrent_store_bytes(&self) -> usize {
        let cfg = &self.model.cfg;
        RecurrentStateStore::reserved_bytes_for(&cfg.layer_kinds, &cfg.delta_net)
    }

    pub fn new(model: Qwen4ExpGpu) -> Result<Self> {
        let schedule = flash_next_schedule(&model.cfg)?;
        let rope = RopeRungs::new(&schedule, &model.device)?;
        // The indexer rotates with the attention's own rungs at the same rotary
        // width, as the reference does.
        let index_rope = FactoredRope::over(&rope, &model.device)?;
        Ok(Self {
            model,
            recurrent: RwLock::new(HashMap::new()),
            seeds: RwLock::new(SeedStore::new()),
            seed_spares: RwLock::new(SeedStore::new()),
            draft_carry: Mutex::new(None),
            verify_stash: Mutex::new(None),
            seeded: RwLock::new(HashSet::new()),
            verify: RwLock::new(None),
            ple: RwLock::new(HashMap::new()),
            index: RwLock::new(HashMap::new()),
            pages: PageRegistry::default(),
            retired: RwLock::new(HashMap::new()),
            index_rope,
            qsa_rows: AtomicU64::new(0),
            rope,
        })
    }

    pub fn engine(&self) -> &Qwen4ExpGpu {
        &self.model
    }

    /// Query rows QSA has narrowed since load — see [`Self::qsa_rows`].
    pub fn qsa_rows_selected(&self) -> u64 {
        self.qsa_rows.load(Ordering::Relaxed)
    }

    /// Fresh-or-reset per-sequence state, keyed exactly as the oracle keys its
    /// sessions: a member arriving at offset 0 starts over.
    ///
    /// `tokens` is what the sequence will hold once this wave lands, which is
    /// what the index caches are grown against — growth reallocates, and an
    /// allocation inside the layer loop would move ground a placed tier is
    /// standing on (hot-path invariant 7). Admission is the place for it.
    ///
    /// **The reset needs `layer_start == 0` as well as `offset == 0`.** A sweep
    /// runs once per `forward_wave` layer window, and a creeping prefill does
    /// not advance the session's offset until it reaches the head — so every
    /// window of a fresh sequence's prompt arrives here at offset 0. Keyed on
    /// the offset alone, window 2 reset what window 1 had just built: the PLE
    /// layer's `conv_hist`/`prev`, the DeltaNet `S` those layers had
    /// accumulated over the prompt, and their filled QSA index caches. Only the
    /// last window's layers would then hold real state, the earlier ones
    /// standing at sequence-start as though the prompt had never run — with
    /// every shape still matching and nothing raised.
    fn ensure_seq_state(
        &self,
        seq: usize,
        offset: usize,
        tokens: usize,
        layer_start: usize,
    ) -> Result<()> {
        let cfg = &self.model.cfg;
        let hc_dim = cfg.hc.count * cfg.hidden_size;
        // Consumed here, once, whether or not it fires — see `seeded`. Taken
        // before the four class blocks so all of them see the same answer: a
        // flag read per class would be true for the first and false for the
        // rest, resetting three quarters of a restored sequence.
        let seeded = layer_start == 0
            && self
                .seeded
                .write()
                .map_err(|_| candle::Error::Msg("qwen4exp: seeded lock poisoned".into()))?
                .remove(&seq);
        let starting_over = offset == 0 && layer_start == 0 && !seeded;
        {
            let mut idx = self
                .index
                .write()
                .map_err(|_| candle::Error::Msg("index lock poisoned".into()))?;
            let caches = match idx.entry(seq) {
                std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                // One per KV layer, the draft head's included: the head selects
                // over its own cache exactly as a trunk attention layer does,
                // and its layer index — past every trunk layer — is also its
                // index here.
                std::collections::hash_map::Entry::Vacant(e) => e.insert(
                    (0..cfg.kv_layers().total())
                        .map(|_| IndexCache::new(cfg.indexer.head_dim, &self.model.device))
                        .collect::<Result<Vec<_>>>()?,
                ),
            };
            // **A sequence-start reset must not throw away installed pages.**
            //
            // `starting_over` means "this slot is at position 0 and nobody
            // declared its state deliberate", which for the index is only true
            // before anything was injected. A projection installs its pages and
            // THEN fills the K/V, so a wave that lands between those two steps
            // sees offset 0, resets, and silently discards the rows for a prefix
            // the slot is about to hold — leaving pages 0 against a K/V of
            // hundreds of tokens, invisible until the select refuses at depth.
            let injected_before_reset = starting_over
                && caches
                    .iter()
                    .zip(self.attention_ratios())
                    .any(|(c, ratio)| ratio > 0 && c.page_row_span() > 0);
            for (cache, ratio) in caches.iter_mut().zip(self.attention_ratios()) {
                if starting_over {
                    // The prompt end is a declaration about what this slot is
                    // about to hold (`set_selection_prompt`), made before its
                    // first wave, so the sequence-start reset keeps it.
                    let prompt_end = cache.prompt_end();
                    cache.reset();
                    cache.set_prompt_end(prompt_end);
                }
                if ratio > 0 {
                    cache.ensure_capacity(tokens, ratio)?;
                }
            }
            if injected_before_reset {
                tracing::warn!(
                    seq,
                    offset,
                    tokens,
                    "qwen4exp sequence-start reset discarded injected index pages — the \
                     slot keeps the K/V they described and loses every row, so the select \
                     refuses once it passes the identity threshold"
                );
            }
            // **The index covers exactly the K/V that is already there.**
            //
            // At wave entry the sequence holds `offset` tokens and every live
            // cache must already account for all of them — as injected pages,
            // as appended rows, or as a mix. Anything less and this slot carries
            // K/V nothing indexed, which is invisible while the sequence is
            // under the identity threshold and refuses the select the moment it
            // crosses. Checked here, on host counters already in hand (no
            // launch, no readback), because `score_rows` finds it at depth —
            // long after whichever step dropped the tokens, and only for the
            // sequences that get deep enough to look.
            //
            // Comparing the caches only against EACH OTHER is not this check:
            // a fork that rebuilds a child's caches from pages leaves all
            // thirteen short by the parent's un-sealed tail, in perfect
            // agreement.
            // **Both directions.** Short is the loud failure — the select
            // refuses. Long is the silent one: the rows are there, so nothing
            // errors, and the extra ones claim positions this slot does not
            // hold, which reads as a plausible answer about the wrong context.
            // A tail restored twice lands exactly here.
            // **Only on the window that enters the wave.** A sweep is split
            // into layer windows and `ensure_seq_state` runs per window, so by
            // the second one the earlier layers have already appended this
            // wave's rows — their coverage legitimately reads `tokens`, not
            // `offset`, and comparing there reports every wide wave as
            // over-covered. The first window is the only moment at which the
            // whole stack is still standing at `offset`.
            let off: Vec<(usize, usize)> = if layer_start != 0 {
                Vec::new()
            } else {
                // **Exact, to the token.** `indexed_tokens` counts the open
                // block's carried rows as well as the completed blocks and the
                // injected pages, so an index that is where its K/V is agrees
                // with `offset` exactly — and one that is off by less than a
                // block is misplaced just the same (see `coverage`).
                coverage_disagreements(
                    caches
                        .iter()
                        .zip(self.attention_ratios())
                        .enumerate()
                        .filter(|(_, (_, ratio))| *ratio > 0)
                        .map(|(layer, (c, ratio))| (layer, c.indexed_tokens(ratio))),
                    offset,
                )
            };
            if !off.is_empty() {
                tracing::warn!(
                    seq,
                    offset,
                    tokens,
                    starting_over,
                    layers = ?off,
                    "qwen4exp index disagrees with this sequence's K/V at wave entry — \
                     (kv layer, tokens indexed) against {offset} held: short refuses the \
                     select past the identity threshold, long selects over positions the \
                     slot does not hold"
                );
            }
        }
        {
            let mut rec = self
                .recurrent
                .write()
                .map_err(|_| candle::Error::Msg("recurrent lock poisoned".into()))?;
            if starting_over || !rec.contains_key(&seq) {
                rec.insert(
                    seq,
                    RecurrentStateStore::new(&cfg.layer_kinds, &cfg.delta_net, &self.model.device)?,
                );
            }
        }
        {
            let mut ple = self
                .ple
                .write()
                .map_err(|_| candle::Error::Msg("ple lock poisoned".into()))?;
            if starting_over || !ple.contains_key(&seq) {
                ple.insert(seq, PleState::zeros(&cfg.ple, hc_dim, &self.model.device)?);
            }
            // The fused block's second history buffer, made here rather than in
            // the forward: a new state has none, and a rewind drops it.
            ple.get_mut(&seq).expect("inserted above").ensure_spare()?;
        }
        // A sequence starting over has no previous row for the head to read,
        // and a stale seed here is the previous conversation's state feeding
        // this one's first proposal. Dropped rather than zeroed: the head pass
        // reads a missing seed as position 0 and takes zeros itself.
        if starting_over {
            self.seeds
                .write()
                .map_err(|_| candle::Error::Msg("seed lock poisoned".into()))?
                .remove(&seq);
            self.seed_spares
                .write()
                .map_err(|_| candle::Error::Msg("seed lock poisoned".into()))?
                .remove(&seq);
        }
        // The head's two seed buffers, made here rather than in the forward. A
        // sequence with no seed is one the head has never run behind, whose
        // first row reaches back to zeros (see `SeedStore`); its spare is the
        // buffer the first carry writes. A failed wave drops the spare, and
        // this replaces it before the next.
        if self.model.mtp.is_some() {
            let dims = (1, cfg.hc.count, cfg.hidden_size);
            let dev = &self.model.device;
            let mut seeds = self
                .seeds
                .write()
                .map_err(|_| candle::Error::Msg("seed lock poisoned".into()))?;
            if let Entry::Vacant(e) = seeds.entry(seq) {
                e.insert(state_buffer(dev, dims, true)?);
            }
            let mut spares = self
                .seed_spares
                .write()
                .map_err(|_| candle::Error::Msg("seed lock poisoned".into()))?;
            if let Entry::Vacant(e) = spares.entry(seq) {
                // Written by the carry before it is read (invariant 6).
                e.insert(state_buffer(dev, dims, false)?);
            }
        }
        Ok(())
    }

    /// Each KV layer's QSA compression ratio, in KV-layer order — the trunk's
    /// full-attention layers, then the draft head's if the artifact carries
    /// one. The head's cache is grown and reset with the rest, so the ratios
    /// have to reach it: zipped against the caches, a short list would silently
    /// leave the head's cache never sized and its first selection reading a
    /// zero-capacity buffer.
    pub(super) fn attention_ratios(&self) -> Vec<usize> {
        let mut ratios: Vec<usize> = self
            .model
            .layers
            .iter()
            .filter_map(|l| match &l.mix {
                GpuLayerMix::Attention { compress_ratio, .. } => Some(*compress_ratio),
                GpuLayerMix::DeltaNet(_) => None,
            })
            .collect();
        if let Some(head) = &self.model.mtp {
            if let GpuLayerMix::Attention { compress_ratio, .. } = &head.block.mix {
                ratios.push(*compress_ratio);
            }
        }
        ratios
    }

    /// The most any one KV layer a sweep over `[layer_start, layer_end)` runs
    /// carves for its sparse selection — the trunk's attention layers in the
    /// window, and the draft head's when the sweep reaches it. What the wave
    /// plan prices as `WaveBuffer::QsaSelection`; see
    /// [`super::select_bytes`] for why the model states it.
    ///
    /// The rows are the wave's, decode then prefill, `offsets[i]` the first
    /// position of `seq_ids[i]`'s `q_lens[i]` rows.
    #[cfg(feature = "cuda")]
    pub(super) fn wave_qsa_bytes(
        &self,
        seq_ids: &[usize],
        q_lens: &[usize],
        offsets: &[usize],
        layer_start: usize,
        layer_end: usize,
    ) -> Result<usize> {
        let spans = seq_spans(seq_ids, q_lens)?;
        let idx = self
            .index
            .read()
            .map_err(|_| candle::Error::Msg("index lock poisoned".into()))?;
        let cfg = &self.model.cfg;
        let mut worst = 0usize;
        let mut kv = 0usize;
        for (li, layer) in self.model.layers.iter().enumerate() {
            if let GpuLayerMix::Attention { compress_ratio, .. } = &layer.mix {
                if (layer_start..layer_end).contains(&li) {
                    worst = worst.max(select_layer_bytes(
                        kv,
                        *compress_ratio,
                        &spans,
                        offsets,
                        &idx,
                        &cfg.indexer,
                    )?);
                }
                kv += 1;
            }
        }
        if layer_end == cfg.num_layers {
            if let Some(head) = &self.model.mtp {
                if let GpuLayerMix::Attention { compress_ratio, .. } = &head.block.mix {
                    worst = worst.max(select_layer_bytes(
                        kv,
                        *compress_ratio,
                        &spans,
                        offsets,
                        &idx,
                        &cfg.indexer,
                    )?);
                }
            }
        }
        Ok(worst)
    }

    /// Each KV layer's indexer weights, in the same KV-layer order as
    /// [`Self::attention_ratios`] — the two are zipped against the caches, so
    /// they must walk the layers identically.
    pub(super) fn attention_indexers(&self) -> Vec<&IndexerWeights> {
        let mut out: Vec<&IndexerWeights> = self
            .model
            .layers
            .iter()
            .filter_map(|l| match &l.mix {
                GpuLayerMix::Attention { indexer, .. } => Some(indexer),
                GpuLayerMix::DeltaNet(_) => None,
            })
            .collect();
        if let Some(head) = &self.model.mtp {
            if let GpuLayerMix::Attention { indexer, .. } = &head.block.mix {
                out.push(indexer);
            }
        }
        out
    }

    /// The indexer's factored RoPE table, at the model's rotary width
    /// (`cfg.rope_dim`) and frequencies.
    pub(super) fn index_rope(&self) -> &FactoredRope {
        &self.index_rope
    }

    /// The schedule's rungs, which every paged attention kernel rotates from.
    pub fn rope(&self) -> &RopeRungs {
        &self.rope
    }

    /// Run `schedule` in place of Flash-Next's own, for the attention and the
    /// indexer alike — the control a long-context gate measures the model's
    /// rungs against (`docs/progressive_yarn.md` §10). Nothing stored depends
    /// on a rung (I1), so a session opened after this simply reads the new
    /// tables. The rotary width must be the model's.
    pub fn set_rope_schedule(&mut self, schedule: &RopeSchedule) -> Result<()> {
        if schedule.pairs() != self.rope.pairs() {
            candle::bail!(
                "qwen4exp: a {}-pair schedule on a {}-pair rotary width",
                schedule.pairs(),
                self.rope.pairs()
            );
        }
        let rope = RopeRungs::new(schedule, &self.model.device)?;
        self.index_rope = FactoredRope::over(&rope, &self.model.device)?;
        self.rope = rope;
        Ok(())
    }

    /// Run no sequence on a rung whose YaRN factor is below `min`, for the
    /// attention and the indexer alike ([`RopeRungs::with_min_factor`]). Set
    /// before any session opens.
    pub fn set_rope_min_factor(&mut self, min: f32) -> Result<()> {
        self.rope = self.rope.clone().with_min_factor(min)?;
        self.index_rope = self.index_rope.clone().with_min_factor(min)?;
        Ok(())
    }
}

impl ManagedBatchedModel for Qwen4ExpBatched {
    fn wave_geometry(&self, _act_dtype: DType) -> ModelGeometry {
        // The Gated Residual carries F32 throughout, so the wave's own dtype is
        // not what the spans are priced in.
        let act_dtype = DType::F32;
        let cfg = &self.model.cfg;
        let int8 = self.model.lm_head.int8mode().is_int8();
        ModelGeometry {
            hidden: cfg.hidden_size,
            vocab: cfg.vocab_size,
            intermediate: cfg.moe.expert_ffn_size,
            n_head: cfg.num_attention_heads,
            n_kv_head: cfg.num_kv_heads,
            head_dim: cfg.attn_head_dim,
            experts_per_tok: cfg.moe.n_experts_used.max(1),
            n_experts: cfg.moe.n_experts.max(1),
            // The 3:1 hybrid's DeltaNet mixer carves from the attention arena,
            // so its widths are priced beside the attention chain.
            delta_net: cfg
                .layer_kinds
                .iter()
                .any(|k| matches!(k, LayerKind::DeltaNet))
                .then_some(DeltaNetWidths {
                    conv_dim: cfg.delta_net.conv_dim(),
                    value_dim: cfg.delta_net.value_dim(),
                    n_v_heads: cfg.delta_net.n_v_heads,
                    layers: cfg
                        .layer_kinds
                        .iter()
                        .filter(|k| matches!(k, LayerKind::DeltaNet))
                        .count(),
                }),
            // Every MoE layer adds an always-active shared expert to the routed
            // block, its gate projection stored padded to one KO tile.
            shared_expert: Some(SharedExpertWidths {
                intermediate: cfg.moe.shared_expert_ffn_size,
                gate_cols: SHARED_GATE_TILE,
            }),
            act_dtype,
            accum_dtype: if int8 {
                DType::F32
            } else {
                ffn_work_dtype(act_dtype)
            },
            packed_norm: int8,
            packed_head: int8,
            // The Qwen3.5 lineage's attention: an interleaved `[q | gate]` Q
            // weight that does not pack with K and V, per-head Q/K norms, and a
            // wave flattened to `[rows, hidden]` before the sweep.
            gated_qkv: true,
            fused_qkv: false,
            // No Q/K/V biases in this lineage.
            qkv_bias: false,
            // The fused q8 decode combine serves this head dim on an int8
            // session, through the same predicate the dispatch asks.
            decode_q8_context: int8 && paged_decode_q8_head_dim(cfg.attn_head_dim),
            head_qk_norm: true,
            head_norm_reshapes: false,
            partial_rotary: cfg.rope_dim < cfg.attn_head_dim,
            // The hyper-connection streams — what made the Gated Residual's
            // transients unpriceable before this geometry carried them.
            hyper: Some(HyperWidths {
                streams: cfg.hc.count,
                low_rank: cfg.hc.low_rank,
                draft_head: self.model.mtp.is_some(),
                ple: true,
            }),
            // The multi-stream head is the one `HyperWidths` states.
            mtp_head: false,
        }
    }

    /// The widest prefill **this card** can place: the rows whose whole tier —
    /// every phase, priced in [`Self::wave_geometry`] — fits the ground a
    /// forward can reach ([`Self::placeable_tier_bytes`]), bounded by where
    /// compute saturates and by what the KV side can hold.
    ///
    /// Not the generic single-phase bound (rows that fit a fixed 512 MiB FFN
    /// span), which cuts a 72 GB card's prefill to a fraction of the width its
    /// tier can carry; and not unbounded either, which hands a 16 GB card slabs
    /// whose tier it cannot place. On a card with tens of GB the weight side
    /// could concede, this opens to the compute ceiling and the admission rate
    /// model picks the width within it; on a small card it sits near the tier
    /// the partition always has room for.
    fn prefill_width_cap(&self, act_dtype: DType) -> usize {
        let mut cap = MAX_PREFILL_TOKENS;
        if let Some(ground) = self.placeable_tier_bytes() {
            let fits = WavePlan::new(self.wave_geometry(act_dtype))
                .max_prefill_rows_for_tier(ground, WaveWidth::default());
            if fits > 0 {
                cap = cap.min(fits);
            }
        }
        if let Some(kv_fits) = self.kv_width_cap(act_dtype) {
            cap = cap.min(kv_fits);
        }
        cap
    }

    fn reclaimable_kv_bytes(&self) -> usize {
        self.model.experts.cedeable_span_bytes()
    }

    fn maybe_change_dtype(&self, dtype: DType) -> Result<()> {
        Qwen4ExpBatched::maybe_change_dtype(self, dtype, dtype)
    }

    fn num_layers(&self) -> usize {
        self.model.cfg.num_layers
    }

    fn n_kv_head(&self) -> usize {
        self.model.cfg.num_kv_heads
    }

    fn head_dim(&self) -> usize {
        self.model.cfg.attn_head_dim
    }

    fn device(&self) -> &Device {
        &self.model.device
    }

    fn carries_recurrent_state(&self) -> bool {
        // The GDN state is an accumulated sum with no per-token decomposition,
        // and the PLE conv history is likewise irrecoverable from the KV.
        true
    }

    fn recurrent_memory_count(&self) -> usize {
        Qwen4ExpBatched::recurrent_len(self).unwrap_or(0)
    }

    fn recurrent_reserved_bytes(&self) -> usize {
        Qwen4ExpBatched::recurrent_reserved_bytes(self)
    }

    fn recurrent_store_bytes(&self) -> usize {
        Qwen4ExpBatched::recurrent_store_bytes(self)
    }

    fn compact_recurrent(&self, max_moves: usize) -> Result<RecurrentCompaction> {
        Qwen4ExpBatched::compact_recurrent(self, max_moves)
    }

    /// A view carve. The child borrows the parent's K/V; its carried state has
    /// to be copied, because it is about to advance it.
    fn fork_recurrent(&self, parent: usize, child: usize) -> Result<()> {
        Qwen4ExpBatched::fork_recurrent(self, parent, child)
    }

    /// A view finalizes: its decoded blocks transfer to the parent, and its
    /// carried state goes with them.
    fn move_recurrent(&self, child: usize, parent: usize) -> Result<()> {
        Qwen4ExpBatched::move_recurrent(self, child, parent)
    }

    fn export_recurrent(&self, seq: usize) -> Result<Option<(u64, Vec<ExportedLayerState>)>> {
        Qwen4ExpBatched::export_recurrent(self, seq)
    }

    fn export_aux_state(&self, seq: usize) -> Result<Option<Vec<u8>>> {
        Qwen4ExpBatched::export_aux_state(self, seq)
    }

    /// **Attention layers only, and not the draft head's.**
    ///
    /// The default assumes a uniform transformer where every layer attends and
    /// so every layer has a Q in the KV cache to capture. Three quarters of
    /// this stack is gated DeltaNet and has no Q at all, and the layer past the
    /// trunk is the speculative head's — its Q is about a continuation the head
    /// proposed, not about the conversation.
    ///
    /// Reporting the trunk's 48 here is not a slow path, it is silence: the
    /// fold is derived from this number, so it grouped `[46, 1, 1]` over a
    /// stack that offers 12 KV layers, could not fill its three layer-groups,
    /// and refused every capture — turns sealed, K/V was written, and only the
    /// signature was quietly missing, leaving retrieval on recency.
    fn model_core_properties(&self) -> ModelCoreProperties {
        let mut props = ManagedBatchedModel::default_core_properties(self);
        props.provenance_capture_layers = self.model.cfg.n_attention_layers();
        props
    }

    fn carries_positional_state(&self) -> bool {
        // The QSA index: one pooled key per `ratio` tokens per attention layer,
        // derived from hidden states rather than from stored K, so borrowing a
        // prefix's K/V does not bring it along.
        true
    }

    fn reset_positional_state(&self, seq: usize) -> Result<()> {
        Qwen4ExpBatched::reset_positional_state(self, seq)
    }

    fn positional_cut_floor(&self, seq: usize, tokens: usize) -> Result<usize> {
        Qwen4ExpBatched::positional_cut_floor(self, seq, tokens)
    }

    fn truncate_positional_state(&self, seq: usize, tokens: usize) -> Result<()> {
        Qwen4ExpBatched::truncate_positional_state(self, seq, tokens)
    }

    fn seal_positional_state(&self, seq: usize) -> Result<Option<Vec<u8>>> {
        Qwen4ExpBatched::seal_positional_state(self, seq)
    }

    fn seal_positional_range(&self, seq: usize, start_pos: usize) -> Result<Option<Vec<u8>>> {
        Qwen4ExpBatched::seal_positional_range(self, seq, start_pos)
    }

    fn seal_positional_tail_span(
        &self,
        seq: usize,
        tokens: usize,
    ) -> Result<Vec<(usize, Vec<u8>)>> {
        Qwen4ExpBatched::seal_positional_tail_span(self, seq, tokens)
    }

    fn close_positional_page(&self, seq: usize) -> Result<usize> {
        Qwen4ExpBatched::close_positional_page(self, seq)
    }

    fn positional_coverage(&self, seq: usize) -> Option<usize> {
        let map = self.index.read().ok()?;
        // **A sequence with no entry covers nothing, and says so.** Returning
        // `None` here reads as "this model keeps no per-position state" to every
        // caller, which is how a slot holding borrowed K/V and no index at all
        // slipped past guards written as `if let Some(cov)` — the one case they
        // most needed to catch.
        let Some(caches) = map.get(&seq) else {
            return Some(0);
        };
        // The narrowest live layer: they index one stream, so the smallest is
        // what the sequence can actually select against.
        caches
            .iter()
            .zip(self.attention_ratios())
            .filter(|(_, ratio)| *ratio > 0)
            .map(|(c, ratio)| c.indexed_tokens(ratio))
            .min()
            .or(Some(0))
    }

    fn push_positional_state(&self, seq: usize, blob: &[u8], key: &PieceKey) -> Result<bool> {
        Qwen4ExpBatched::push_positional_state(self, seq, blob, key)?;
        Ok(true)
    }
    fn push_positional_gap(&self, seq: usize, tokens: usize) -> Result<()> {
        Qwen4ExpBatched::push_positional_gap(self, seq, tokens)
    }

    fn restore_aux_state(&self, seq: usize, blob: &[u8]) -> Result<bool> {
        Qwen4ExpBatched::restore_aux_state(self, seq, blob)?;
        Ok(true)
    }

    fn restore_recurrent(
        &self,
        seq: usize,
        schedule_hash: u64,
        layers: &[ExportedLayerState],
    ) -> Result<bool> {
        Qwen4ExpBatched::restore_recurrent(self, seq, schedule_hash, layers)?;
        Ok(true)
    }

    /// This checkpoint's ladder, gated on the head actually being loaded — a
    /// conversion without the NextN tensors would otherwise pay a drafting call
    /// on every wave that can only return nothing.
    fn draft_budget(&self, width: usize) -> usize {
        if self.model.mtp.is_none() {
            return 0;
        }
        // **Bounded by what its own rewind machinery costs**, not only by the
        // ladder. A block of `k` proposals makes each sequence stash `k + 1`
        // rows of post-projection operands for every DeltaNet layer, and this
        // stack has 36 of them — so the stash is `width × (k + 1) × 36` rows,
        // and the ladder prices only the throughput side of `k`.
        //
        // Clamping rather than refusing is what lets the ladder ask for depth
        // at any width: a narrow cohort still gets the full budget, a wide one
        // gets the deepest budget its stash can afford, and the wave does not
        // fail. The stash is allocated between forwards, where nothing is left
        // to concede to, so the alternative failure is a device OOM rather than
        // a refusal. Speculation is lossless, so a shallower budget costs
        // throughput and never a token.
        QWEN38_FLASH_NEXT_DRAFT
            .budget(width)
            .min(self.affordable_draft_budget(width))
    }

    /// The checkpoint's measured drafted-token cost — see its ladder.
    fn draft_token_cost(&self) -> f32 {
        QWEN38_FLASH_NEXT_DRAFT.token_cost()
    }

    /// Draft with the checkpoint's own NextN head, for the whole cohort in one
    /// batched walk ([`super::draft`]).
    fn speculative_draft(
        &self,
        session: &mut BatchedInferenceSession,
        seqs: &[usize],
        committed: &[u32],
        max_len: usize,
    ) -> Result<Vec<Vec<u32>>> {
        self.mtp_draft(session, seqs, committed, max_len)
    }

    /// A rewind is expressed, so the entry point admits this model.
    ///
    /// The claim is about [`Self::truncate_sequences`]: that it restores all
    /// three recurrences exactly for the targets it accepts. It is not a claim
    /// that every offset is rewindable — a replay covers the block it stashed
    /// operands for and nothing else, and `truncate_sequences` refuses an
    /// uncovered target itself, at the one place that knows.
    fn can_rewind_speculative_block(&self) -> bool {
        true
    }

    /// Plan a wave that verifies every drafted block alongside the plain
    /// cohort, and arm what that wave has to capture.
    ///
    /// Each block is a **prefill span**, not a run of decode rows: all three of
    /// this stack's recurrences are sequential within a sequence, so two rows
    /// of one sequence cannot decode in parallel against a single carried
    /// state. The prefill scan is the form that walks them in order. Plain rows
    /// lead as ordinary decode rows in the same wave, so the step pays one
    /// launch floor rather than two and the MoE's expert traffic amortises
    /// across both cohorts.
    fn begin_verify(
        &self,
        session: &mut BatchedInferenceSession,
        plain: &[(usize, u32)],
        seqs: &[usize],
        blocks: &[Vec<u32>],
        budget: usize,
    ) -> Result<Option<VerifyPlan>> {
        let _ = (session, budget);
        if plain.is_empty() && seqs.is_empty() {
            return Ok(Some(VerifyPlan {
                decode_seqs: Vec::new(),
                decode_inputs: Vec::new(),
                verify_seqs: Vec::new(),
                verify_inputs: Vec::new(),
                rows: 0,
            }));
        }
        if seqs.len() != blocks.len() {
            candle::bail!(
                "qwen4exp verify: {} sequences against {} blocks",
                seqs.len(),
                blocks.len()
            );
        }
        // A one-token block is a decode step; the driver routes those to
        // `plain` and never here. Refused rather than folded, because the
        // stash's `len` would then disagree with the block.
        if let Some(b) = blocks.iter().find(|b| b.len() < 2) {
            candle::bail!(
                "qwen4exp verify: a {}-token block is a plain decode step, not a verify",
                b.len()
            );
        }
        // The model's device, not the host: these rows are cat'd with whatever
        // else the caller has on the wave. One upload per group.
        let dseqs: Vec<usize> = plain.iter().map(|&(s, _)| s).collect();
        let tokens: Vec<u32> = plain.iter().map(|&(_, t)| t).collect();
        let (dinputs, pinputs) = upload_plan_rows(&tokens, blocks)?;

        // **Size the stash before the forward opens.** A wave's storage is
        // claimed by `admit_wave_kv` and the transient tier placed against that
        // claim, so the arena refuses a device allocation from inside the
        // forward — which is what a stash allocated as the sweep reached each
        // layer would be.
        let cohort: Vec<(usize, usize)> = seqs
            .iter()
            .enumerate()
            .map(|(i, &s)| (s, blocks[i].len()))
            .collect();
        // A capture still armed from a step that never reached its rewind is
        // released first, so its stash is the one this step reuses.
        let stale = self
            .verify
            .write()
            .map_err(|_| candle::Error::Msg("verify lock poisoned".into()))?
            .take();
        if let Some(old) = stale {
            self.park_verify_stash(old);
        }
        let rows: usize = cohort.iter().map(|&(_, n)| n).sum();
        let cap = SpecCapture::new(&cohort, self.verify_stash_for(rows)?)?;
        let mut tokens: Vec<(usize, Vec<u32>)> = Vec::with_capacity(seqs.len());
        for (i, &s) in seqs.iter().enumerate() {
            tokens.push((s, blocks[i].clone()));
        }
        {
            let mut g = self
                .verify
                .write()
                .map_err(|_| candle::Error::Msg("verify lock poisoned".into()))?;
            let mut cap = cap;
            for (s, t) in tokens {
                if let Some(slot) = cap.seqs.get_mut(&s) {
                    slot.tokens = t;
                }
            }
            *g = Some(cap);
        }
        Ok(Some(VerifyPlan {
            decode_seqs: dseqs,
            decode_inputs: dinputs,
            verify_seqs: seqs.to_vec(),
            verify_inputs: pinputs,
            rows: plain.len() + blocks.iter().map(|b| b.len()).sum::<usize>(),
        }))
    }

    /// Read the verify wave's rows back and advance what it wrote.
    ///
    /// The capture deliberately OUTLIVES this: the driver truncates back to the
    /// accepted prefix afterwards, and that rewind is what the capture exists
    /// for. [`Self::truncate_sequences`] is what releases it.
    fn end_verify(
        &self,
        session: &mut BatchedInferenceSession,
        plain: &[(usize, u32)],
        seqs: &[usize],
        blocks: &[Vec<u32>],
        logits: Vec<Tensor>,
    ) -> Result<(Vec<Tensor>, Vec<Vec<Tensor>>)> {
        for &(seq, _) in plain {
            session.advance_sequence(seq, 1)?;
        }
        for (i, &seq) in seqs.iter().enumerate() {
            session.advance_sequence(seq, blocks[i].len())?;
        }
        let lens: Vec<usize> = blocks.iter().map(|b| b.len()).collect();
        split_block_rows(&logits, plain.len(), &lens)
    }

    /// The wave rolled its own state back, so the capture names a rewind point
    /// that no longer exists — left in place, a later truncate would replay
    /// from it.
    fn abort_verify(&self, seqs: &[usize]) {
        let _ = seqs;
        let released = self.verify.write().ok().and_then(|mut g| g.take());
        if let Some(cap) = released {
            self.park_verify_stash(cap);
        }
    }

    /// Roll every target back to its accepted prefix — K/V and all three
    /// recurrences — in one pass over the cohort.
    fn truncate_sequences(
        &self,
        session: &mut BatchedInferenceSession,
        targets: &[(usize, usize)],
    ) -> Result<()> {
        self.rewind_cohort(session, targets)
    }

    fn rope_select(&self) -> RungSelect {
        self.rope.select().clone()
    }

    fn set_rope_min_factor(&mut self, min: f32) -> Result<()> {
        Qwen4ExpBatched::set_rope_min_factor(self, min)
    }

    fn set_selection_prompt(&self, seq: usize, tokens: usize) -> Result<()> {
        Qwen4ExpBatched::set_selection_prompt(self, seq, tokens)
    }

    fn set_selection_strata(&mut self, strata: StrataTokens) -> Result<()> {
        Qwen4ExpBatched::set_selection_strata(self, strata)
    }

    fn create_batched_session(&self, config: BatchedConfig) -> Result<BatchedInferenceSession> {
        let cfg = &self.model.cfg;
        // This model's KV threshold calibration, folded in here because this
        // override replaces the `ManagedBatchedModel` default that would
        // otherwise do it — the KV layer count differs from the transformer
        // depth on a hybrid, and dropping the fold would leave the per-model
        // row silently never reaching the compression policy. The row is the
        // one calibrated for this artifact's expert format (`kv_factors_for`).
        let mut config = config;
        let row = kv_factors_for(self.model.expert_format);
        config.k_hi_error_threshold_factor *= row.k_hi;
        config.k_low_error_threshold_factor *= row.k_low;
        config.v_hi_error_threshold_factor *= row.v_hi;
        config.v_low_error_threshold_factor *= row.v_low;
        // The trunk's KV layers plus the draft head's, so the head holds its
        // keys in this same paged cache and prefills, decodes and seals
        // alongside the trunk without any session-wide operation knowing it
        // exists. The head's layer is the last one and is stepped only by the
        // head's own pass — see `Qwen4ExpConfig::mtp_kv_layer`.
        let mut session = BatchedInferenceSession::new(
            cfg.kv_layers(),
            cfg.num_kv_heads,
            cfg.attn_head_dim,
            &self.model.device,
            config,
        )?;
        session.set_rope_select(self.rope.select().clone());
        // **The draft head's layer is capped; the trunk's twelve take the
        // session's level unchanged.**
        //
        // The trunk spreads every read over twelve KV layers, so a level's key
        // error is partly averaged over the depth. The head is one block with
        // one KV layer and absorbs none of it — and a proposal is worth
        // something only if it reproduces the trunk's argmax, which is a far
        // sharper test than "the text still reads correctly". Measured: capping
        // this one layer takes C6–C9 acceptance from 1.5–4.9% back to ~90%, and
        // throughput from ~25 to ~117 tok/s, with the trunk untouched.
        //
        // A ceiling rather than a setting, so a session running at C0–C3 keeps
        // its own level and only the deeper rungs are held back. One layer of
        // thirteen, so the ladder's calibrated KV footprint moves by well under
        // a tenth.
        if let Some(head_kv) = cfg.mtp_kv_layer() {
            session.cap_layer_seal_level(head_kv, DRAFT_HEAD_MAX_COMPRESSION)?;
        }
        // The Q/K norm weights meet activations at the KV arena's width, which
        // is NOT the residual stream's — see `maybe_change_dtype`. Both are
        // passed so the call site says which is which.
        Qwen4ExpBatched::maybe_change_dtype(
            self,
            session.activation_dtype(),
            session.kv_live_dtype(),
        )?;
        Ok(session)
    }

    fn prune(&self) -> Result<()> {
        if let Ok(mut m) = self.recurrent.write() {
            m.clear();
        }
        if let Ok(mut m) = self.ple.write() {
            m.clear();
        }
        if let Ok(mut m) = self.index.write() {
            m.clear();
        }
        if let Ok(mut m) = self.retired.write() {
            m.clear();
        }
        if let Ok(mut m) = self.seeds.write() {
            m.clear();
        }
        if let Ok(mut m) = self.seed_spares.write() {
            m.clear();
        }
        Ok(())
    }

    /// The model's half of freeing a sequence: the session owns the KV slots,
    /// these maps own the GDN stores (span-reservation tenants), the PLE
    /// state and the QSA index caches — leaving them keyed holds their
    /// regions for the life of the process while every KV-side diagnostic
    /// reads healthy.
    fn release_sequence(&self, seq: usize) -> Result<()> {
        if let Ok(mut m) = self.recurrent.write() {
            m.remove(&seq);
            // **The parked rewind stash goes with the last sequence.** It is
            // kept across steps for the next verify, and carved from the
            // reservation — so with nobody left to verify it would pin its
            // regions for the life of the process, where no arena sweep can
            // see them because they are not KV.
            if m.is_empty() {
                if let Ok(mut parked) = self.verify_stash.lock() {
                    *parked = None;
                }
            }
        }
        if let Ok(mut m) = self.ple.write() {
            m.remove(&seq);
        }
        if let Ok(mut m) = self.index.write() {
            m.remove(&seq);
        }
        if let Ok(mut m) = self.retired.write() {
            m.remove(&seq);
        }
        if let Ok(mut m) = self.seeds.write() {
            m.remove(&seq);
        }
        if let Ok(mut m) = self.seed_spares.write() {
            m.remove(&seq);
        }
        // Or the next sequence to be handed this slot id inherits a suppression
        // it never asked for, and its first wave keeps whatever the released
        // one left behind.
        if let Ok(mut m) = self.seeded.write() {
            m.remove(&seq);
        }
        Ok(())
    }

    fn expert_stats(&self) -> Option<crate::models::expert_lre::PipelineStats> {
        Some(self.model.experts.expert_stats())
    }

    fn weight_plan(&self) -> WeightPlanning {
        WeightPlan::from_stats(&self.model.experts.expert_stats())
    }

    /// The PLE row cache's hit/miss/eviction counters (§0.1). The table is
    /// disk-resident and never enters VRAM, so this is the number that says
    /// whether the 2 GB cache is sized for the traffic in front of it.
    fn row_cache_stats(&self) -> Option<(u64, u64, u64)> {
        self.model
            .ple_table
            .cache_stats()
            .map(|s| (s.hits, s.misses, s.evictions))
    }

    fn reset_expert_stats(&self) {
        self.model.experts.reset_expert_stats();
    }

    /// The expert pipeline thread's spans — where its time goes serving each
    /// routed layer, which decides how far behind the GPU it runs.
    fn snapshot_profiles(&self) -> ProfileSnapshot {
        self.model.experts.snapshot_profiles()
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_wave(
        &self,
        session: &mut BatchedInferenceSession,
        decode_seqs: &[usize],
        decode_inputs: &[Tensor],
        prefill_seqs: &[usize],
        prefill_inputs: &[Tensor],
        glue_seqs: &[usize],
        glue_inputs: &[Tensor],
        layer_start: usize,
        layer_end: usize,
        residual_in: Option<Tensor>,
    ) -> Result<WaveResult> {
        session.expect_rope_select(self.rope.select())?;
        drive_wave(
            self,
            session,
            decode_seqs,
            decode_inputs,
            prefill_seqs,
            prefill_inputs,
            glue_seqs,
            glue_inputs,
            layer_start,
            layer_end,
            residual_in,
        )
    }
}

impl WaveSweep for Qwen4ExpBatched {
    fn device(&self) -> &Device {
        &self.model.device
    }

    fn num_layers(&self) -> usize {
        self.model.cfg.num_layers
    }

    fn prefill_width_cap(&self, act_dtype: DType) -> usize {
        <Self as ManagedBatchedModel>::prefill_width_cap(self, act_dtype)
    }

    /// Caches are indexed by KV layer: three quarters of the trunk owns no
    /// paged cache, so the driver's rollback range must be translated.
    ///
    /// **Plus the draft head's layer when the window reaches the last trunk
    /// layer**, because that is when the head's pass runs and writes it. One
    /// answer, three callers — admission claims this range, the failure
    /// rollback restores it, and the sweep writes it — and they must name the
    /// same set. A claim short of the sweep is a chunk allocated from inside
    /// the forward that owns the partition (hot-path invariant 7); a rollback
    /// short of the sweep leaves the head's layer a token ahead of the trunk's
    /// after a failed wave, which the next wave "heals" by truncating a token
    /// the caller was already given.
    fn kv_layer_range(&self, layer_start: usize, layer_end: usize) -> (usize, usize) {
        let cfg = &self.model.cfg;
        let kinds = &cfg.layer_kinds;
        let count = |n: usize| {
            kinds[..n.min(kinds.len())]
                .iter()
                .filter(|k| matches!(k, LayerKind::Attention))
                .count()
        };
        let (start, mut end) = (count(layer_start), count(layer_end));
        if cfg.mtp_kv_layer().is_some() && layer_end == cfg.num_layers {
            end += 1;
        }
        (start, end)
    }

    fn sweep(
        &self,
        session: &mut BatchedInferenceSession,
        wave: WaveGroups<'_>,
    ) -> Result<(WavePhase, Option<WaveGuard>)> {
        let WaveGroups {
            n_decode,
            n_prefill,
            seq_ids,
            inputs,
            pending_glue,
            generation,
            layer_start,
            layer_end,
            x_in,
            window,
            act_dtype: _,
            adapter,
        } = wave;
        if seq_ids.is_empty() {
            candle::bail!("qwen4exp wave: empty batch");
        }
        let accept_in_place = session.accept_in_place();
        // Flash-Next runs its layers under the Gated Residual, which carries no
        // adapter plumbing, so `project_qkv_gated` is called with an empty
        // `LayerLora`. Refused rather than ignored: dropping the name here would
        // serve the BASE model under an adapter's name, which reads as a bad
        // fine-tune rather than as an unsupported architecture.
        if let Some(name) = adapter {
            candle::bail!(
                "qwen4exp wave: Flash-Next has no LoRA support, so adapter `{name}` \
                 cannot be applied"
            );
        }
        let m = &self.model;
        let cfg = &m.cfg;
        let eps = cfg.rms_norm_eps;
        let num_layers = cfg.num_layers;
        if layer_start > layer_end || layer_end > num_layers {
            candle::bail!("qwen4exp wave: bad layer range [{layer_start}, {layer_end})");
        }

        // Hand back the PREVIOUS wave's transient tier, here rather than at the
        // end of the wave that placed it. `end_wave_transient` gates on
        // `live_generations`, a host-side count: when a sweep returns its
        // generations are dropped but its kernels are still in flight, so
        // releasing there would hand ground back while the GPU is still reading
        // it. What makes this point safe is the work in between, which drains
        // the wave that placed it. It also has to precede `ensure_seq_state`
        // below, which claims span regions for any store it creates — and a
        // claim under a standing tier is refused outright.
        #[cfg(feature = "cuda")]
        if let Device::Cuda(d) = &m.device {
            end_wave_transient(&d.cuda_stream());
        }

        // The KV↔expert boundary's GROWING direction, in the one gap it is legal
        // in: between forwards, on the line after the transient tier goes back, so
        // no tier stands and the region ceiling is the pool's own size — which is
        // what `growth_policy`'s occupancy arithmetic is written against.
        //
        // Without this the boundary only ever moves toward KV (`request_kv_ground`
        // buys on the spot) and expert residency ratchets down across a long run:
        // spare KV regions above the KV side's recent high-water never come back as
        // resident-expert slots. The shrink direction needs no call here — a KV claim
        // that runs out buys its own ground. Mirrors `latent_moe`'s wave and the
        // blanket `BatchedModel` wave's phase 0; Flash-Next runs its own engine
        // rather than `BatchedModelCore`, so it does not inherit either.
        m.experts.reclaim_spare_ground();

        let n_glue = seq_ids.len() - n_decode - n_prefill;
        if n_glue > 0 || pending_glue.is_some() {
            candle::bail!(
                "qwen4exp wave: glue rows — reprojection glue is not implemented at \
                 head_dim {}; this stack must recompute rather than gap-fill",
                cfg.attn_head_dim
            );
        }

        // Offsets + query lengths from the session, BEFORE the contexts borrow.
        let offsets: Vec<usize> = seq_ids
            .iter()
            .map(|&s| session.sequence_offset(s).unwrap_or(0))
            .collect();
        let q_lens: Vec<usize> = inputs
            .iter()
            .map(|t| t.dims().get(1).copied().unwrap_or(1))
            .collect();
        let (dec_off, pre_off) = offsets.split_at(n_decode);
        let (dec_q, pre_q) = q_lens.split_at(n_decode);
        let pre_rows: usize = pre_q.iter().sum();
        let total_rows = n_decode + pre_rows;

        // Attention metadata from the session's shared borrow. The group is the
        // session's STREAM layers — the draft head's layer is stepped by the
        // head's own pass, and `build_decode_metadata` scopes itself to the
        // stream for exactly that reason.
        let decode_headers = if n_decode > 0 {
            let (buf, stride) = session.build_decode_metadata(&seq_ids[..n_decode], generation)?;
            DecodeHeaders::Decode { buf, stride }
        } else {
            DecodeHeaders::Decode {
                buf: None,
                stride: 0,
            }
        };
        // Read before the contexts borrow the session: the tier pricing below
        // needs it, and `assemble_wave_contexts` holds a mutable borrow across
        // everything that follows.
        let tier_act_dtype = session.activation_dtype();
        let mut contexts = assemble_wave_contexts(session, seq_ids, inputs)?;
        let contexts = contexts.as_mut_slice();

        // Admit: claim every KV chunk this wave writes, over the KV range.
        let (kv_start, kv_end) = self.kv_layer_range(layer_start, layer_end);
        admit_wave_kv(contexts, n_decode, n_prefill, kv_start, kv_end)?;

        // ── Carried state: ensure (reset at offset 0), open the GDN wave,
        // snapshot the PLE states for the failure bracket. ──
        //
        // **A claim, so it belongs with the other claims — before the tier is
        // placed and before the forward opens.** A sequence entering the wave
        // without a store builds one here, and a store takes state-arena slots —
        // which, when no arena of its geometry has room, claims a reservation region
        // through the same arena window `admit_wave_kv` does. Run after
        // `begin_forward` that claim asks
        // the partition for ground while holding the window that decides who
        // gets it, and the refusal is the unrecoverable one — the thread that
        // would have to end the wave is the thread asking.
        //
        // It sat below `begin_forward` and was invisible for as long as every
        // sequence in a wave already had its store: the lazy branch only fires
        // for a slot that has never run one, or one starting over. A deeper
        // `repo_map` walk (`--max-depth 3`) put the boot priming chain's own
        // ingest into exactly that shape, and the daemon failed to load with
        // "creating a KV arena from inside the forward that owns the partition".
        //
        // Above `plan_wave_transient` as well as `begin_forward`, and that order
        // is load-bearing in both directions: the tier is placed against the
        // arena frontier as it stands, so a region claimed after the placement
        // moves the frontier under a tier already standing on it (hot-path
        // invariant 7).
        for ((&seq, &off), &q) in seq_ids.iter().zip(&offsets).zip(&q_lens) {
            self.ensure_seq_state(seq, off, off + q, layer_start)?;
        }

        // **Price and reserve this wave's transient tier**, sized to this wave
        // rather than to the widest one the engine can run — after the KV claim
        // above, so the arena frontier is final when the tier is placed against
        // it, and before `begin_forward` below, because the placement may buy
        // ground from the weight side and `set_weight_floor` refuses while a
        // forward is open.
        #[cfg(feature = "cuda")]
        if total_rows > 0 {
            if let Device::Cuda(d) = &m.device {
                let plan = WavePlan::new(self.wave_geometry(tier_act_dtype));
                // The wave's composition, not just its row count: the chains
                // price a prefill row, a decode row and a scored row
                // differently, and the Gated Residual's streams are carried by
                // `ModelGeometry::hyper` so its wide intermediates are priced
                // rather than overflowing into the pool.
                //
                // **Scored rows are not all rows.** The head runs every decode
                // row and the LAST row of each prefill span — a verifying span
                // excepted, where every row is a prediction to compare a
                // proposal against (see the head section below). Pricing
                // `HeadLogits` at `total_rows × vocab` buys roughly a gigabyte
                // of tier for a logits block three orders of magnitude smaller,
                // and takes it from the weight side every wave.
                //
                // A wave that stops short of the last layer returns its residual
                // and runs no head, so it scores no rows at all.
                let verify_seqs: Vec<usize> = self
                    .verify
                    .read()
                    .ok()
                    .and_then(|g| g.as_ref().map(|c| c.seqs.keys().copied().collect()))
                    .unwrap_or_default();
                // **The rewind this wave may owe.** `replay_accepted_prefixes`
                // — the same one the hybrid runs — carves its conv, scan and
                // span table off THIS Attention span at accept time, and these
                // two units are what price that chain. Left at zero it prices to
                // nothing and the span is short by the whole chain. Rows are the
                // stash's CAPACITY, not this cohort's total: the buffers only
                // grow and the replay's kernels run over every stash row.
                let staged: Option<(usize, usize)> = self
                    .verify
                    .read()
                    .ok()
                    .and_then(|g| {
                        g.as_ref()
                            .map(|c| c.delta.capacity().map(|rows| (rows, c.delta.spans.len())))
                    })
                    .transpose()?;
                let scored_prefill: usize = pre_q
                    .iter()
                    .enumerate()
                    .map(|(k, &l)| {
                        if verify_seqs.contains(&seq_ids[n_decode + k]) {
                            l
                        } else {
                            1
                        }
                    })
                    .sum();
                let scored_rows = if layer_end == num_layers {
                    n_decode + scored_prefill
                } else {
                    0
                };
                let width = WaveWidth {
                    prefill_rows: pre_rows,
                    decode_rows: n_decode,
                    scored_rows,
                    // A one-row prefill group takes the decode kernels, so it
                    // carves no span-table entry and, alone, no scan transient.
                    prefill_spans: pre_q.iter().filter(|&&l| l > 1).count(),
                    staged_rows: staged.map_or(0, |(rows, _)| rows),
                    staged_spans: staged.map_or(0, |(_, spans)| spans),
                    // The caller's accept walk selects on the head's span only
                    // when it reads the logits there.
                    accept_rows: if accept_in_place { scored_rows } else { 0 },
                    qsa_bytes: self.wave_qsa_bytes(
                        seq_ids,
                        &q_lens,
                        &offsets,
                        layer_start,
                        layer_end,
                    )?,
                };
                let per_phase = [
                    plan.phase_bytes(LayerPhase::Attention, width),
                    plan.phase_bytes(LayerPhase::Ffn, width),
                    plan.phase_bytes(LayerPhase::Forward, width),
                ];
                // The refusal names the partition; the wave it was refused for
                // is what says which of the wave's parts to look at.
                plan_wave_transient(&d.cuda_stream(), per_phase).map_err(|e| {
                    candle::Error::Msg(format!("{e} — for a wave of {width:?}, {per_phase:?} B"))
                })?;
            }
        }

        // From here the forward owns the span partition: the tier is placed and
        // must not move under it.
        #[cfg(feature = "cuda")]
        let _forward_open = match &m.device {
            Device::Cuda(d) => Some(begin_forward(&d.cuda_stream())),
            _ => None,
        };

        // **The forward-scoped span**, held for the whole sweep rather than per
        // layer: it carries the metadata a forward builds once and every layer
        // reads — ragged prefill offsets, page and candidate tables, RoPE
        // tables, gathered position ids — which is exactly what
        // `WAVE_FORWARD_BYTES` describes. Its ticket travels instead of the
        // guard, because the builders are deep in the sweep and a `Copy`
        // coordinate reaches them without changing any signature's lifetime.
        #[cfg(feature = "cuda")]
        let fwd_span = match &m.device {
            Device::Cuda(d) => Some(begin_wave(&d.cuda_stream(), LayerPhase::Forward)?),
            _ => None,
        };
        #[cfg(feature = "cuda")]
        let fwd_ticket = fwd_span.as_ref().map(|g| g.ticket());
        #[cfg(not(feature = "cuda"))]
        let fwd_ticket: Option<candle::cuda_backend::wave_provenance::WaveTicket> = None;

        // Built AFTER the span opens, so its three tables land on it. They are
        // the wave's ragged prefill offsets — read by every layer, written once
        // — which is the forward span's stated purpose. Nothing above needs
        // them, so moving the construction down costs only this comment.
        let prefill_headers = DecodeHeaders::Prefill(BatchedPrefillMeta::new_ragged(
            pre_off, pre_q, &m.device, fwd_ticket,
        )?);

        let index_snapshot: Vec<(usize, Vec<IndexSnapshot>)> = {
            let idx = self
                .index
                .read()
                .map_err(|_| candle::Error::Msg("index lock poisoned".into()))?;
            seq_ids
                .iter()
                .map(|&s| {
                    let caches = idx.get(&s).expect("ensured above");
                    let snaps = caches
                        .iter()
                        .map(IndexCache::snapshot)
                        .collect::<Result<Vec<_>>>()?;
                    Ok((s, snaps))
                })
                .collect::<Result<Vec<_>>>()?
        };
        let ple_snapshot: Vec<(usize, PleState)> = {
            let ple = self
                .ple
                .read()
                .map_err(|_| candle::Error::Msg("ple lock poisoned".into()))?;
            seq_ids
                .iter()
                .map(|&s| {
                    let st = ple.get(&s).expect("ensured above");
                    Ok((s, st.clone()))
                })
                .collect::<Result<_>>()?
        };
        // The head's seed belongs in the same bracket as the state above.
        // `head_wave_pass` writes it at the end of a sweep that reaches the
        // head, so a wave that then failed — the LM-head GEMM, a relief-driven
        // failure, both routine here — left every other carried state rolled
        // back to `off` and the seed describing `off + q − 1`. The retried
        // wave's head pass would take that as row 0's `h(t−1)` and write the
        // head's committed K/V from it.
        // One row per sequence, and the storage is shared rather than copied:
        // a seed is already owned storage no generation reclaims, and the head
        // pass *replaces* the entry rather than writing through it.
        let seed_snapshot: Vec<(usize, Option<Tensor>)> = {
            let seeds = self
                .seeds
                .read()
                .map_err(|_| candle::Error::Msg("seed lock poisoned".into()))?;
            seq_ids
                .iter()
                .map(|&s| (s, seeds.get(&s).cloned()))
                .collect()
        };
        {
            let mut rec = self
                .recurrent
                .write()
                .map_err(|_| candle::Error::Msg("recurrent lock poisoned".into()))?;
            for &s in seq_ids {
                rec.get_mut(&s).expect("ensured above").begin_wave()?;
            }
        }

        let swept = self.sweep_layers(
            contexts,
            seq_ids,
            inputs,
            n_decode,
            (dec_off, dec_q, pre_off, pre_q),
            (total_rows, pre_rows),
            (layer_start, layer_end),
            x_in,
            window,
            (decode_headers, prefill_headers),
            generation,
            eps,
            fwd_ticket,
        );

        // Close the wave: commit on success, rewind everything on failure. A
        // rollback that itself fails leaves the sequence's state unaccounted
        // — report both, exactly as the driver's KV rollback does.
        {
            let mut rec = self
                .recurrent
                .write()
                .map_err(|_| candle::Error::Msg("recurrent lock poisoned".into()))?;
            for &s in seq_ids {
                if let Some(store) = rec.get_mut(&s) {
                    if swept.is_ok() {
                        store.commit_wave();
                    } else if let Err(rb) = store.rollback_wave() {
                        let e = swept
                            .as_ref()
                            .err()
                            .map(|e| e.to_string())
                            .unwrap_or_else(|| "unknown".into());
                        candle::bail!(
                            "wave failed ({e}) and sequence {s}'s recurrent rollback \
                             also failed ({rb})"
                        );
                    }
                }
            }
        }
        if swept.is_err() {
            let mut ple = self
                .ple
                .write()
                .map_err(|_| candle::Error::Msg("ple lock poisoned".into()))?;
            for (s, st) in ple_snapshot {
                ple.insert(s, st);
            }
            let mut idx = self
                .index
                .write()
                .map_err(|_| candle::Error::Msg("index lock poisoned".into()))?;
            for (s, snaps) in &index_snapshot {
                if let Some(caches) = idx.get_mut(s) {
                    for (cache, snap) in caches.iter_mut().zip(snaps) {
                        cache.restore(snap)?;
                    }
                }
            }
            let mut seeds = self
                .seeds
                .write()
                .map_err(|_| candle::Error::Msg("seed lock poisoned".into()))?;
            let mut spares = self
                .seed_spares
                .write()
                .map_err(|_| candle::Error::Msg("seed lock poisoned".into()))?;
            for (s, seed) in seed_snapshot {
                // The carry may already have flipped: the seed being put back
                // would then also be the spare, and the next carry would write
                // the seed it reads.
                spares.remove(&s);
                match seed {
                    // Restore what the sequence entered the wave with — which
                    // for a sequence that had none is nothing, so the entry is
                    // removed rather than left holding this wave's write.
                    Some(prev) => {
                        seeds.insert(s, prev);
                    }
                    None => {
                        seeds.remove(&s);
                    }
                }
            }
        } else {
            // **The successful path hands the same snapshots to the verify
            // capture instead of dropping them.**
            //
            // A rewind needs the state the block was ENTERED with, and that is
            // exactly what the failure bracket above already took — so a
            // speculative wave pays no snapshot of its own. Moved rather than
            // cloned: on a wave that committed, nothing else reads them.
            let mut g = self
                .verify
                .write()
                .map_err(|_| candle::Error::Msg("verify lock poisoned".into()))?;
            if let Some(cap) = g.as_mut() {
                let mut idx_snaps: HashMap<usize, Vec<IndexSnapshot>> =
                    index_snapshot.into_iter().collect();
                for (s, ple) in ple_snapshot {
                    // **Never an empty stand-in.** Both snapshots are built from
                    // this wave's `seq_ids`, so a sequence present in one and
                    // absent from the other is an inconsistency — and defaulting
                    // it to an empty vector does not paper over the gap, it
                    // silently disarms the rewind's QSA restore while leaving its
                    // re-append running, which puts the index a whole block past
                    // the K/V for the rest of the sequence.
                    let qsa = idx_snaps.remove(&s).ok_or_else(|| {
                        candle::Error::Msg(format!(
                            "qwen4exp: sequence {s} has a PLE entering snapshot but no index \
                             one — the rewind would re-append the accepted rows onto a cache \
                             it never rolled back"
                        ))
                    })?;
                    cap.take_entering(s, ple, qsa);
                }
            }
        }
        // The head's logits are carved from the forward span (`hc_mix` and the
        // head GEMM run on its ticket), so a wave that reached the head hands
        // the span's guard back with them: `WaveResult` keeps the span alive
        // while the caller reads, instead of the head copying the `[R, vocab]`
        // block off it first.
        #[cfg(feature = "cuda")]
        let swept = swept.map(|(phase, guard)| {
            if matches!(phase, WavePhase::Logits(_)) {
                (phase, guard.or(fwd_span))
            } else {
                (phase, guard)
            }
        });
        swept
    }
}

impl Qwen4ExpBatched {
    /// Bytes a forward's tier can be placed in: the KV side's free ground —
    /// counting what the standing tier blocks, which the next forward releases
    /// first — plus what the weight side could still concede above its floor.
    /// Never less than [`WAVE_SPAN_BYTES`], the tier the partition always
    /// leaves room for. `None` where there is no span to measure (a CPU
    /// device, a test).
    fn placeable_tier_bytes(&self) -> Option<usize> {
        let stats = region_stats(0)?;
        let free = (stats.free + stats.blocked).saturating_mul(REGION_BYTES);
        Some(
            free.saturating_add(self.model.experts.cedeable_span_bytes())
                .max(WAVE_SPAN_BYTES),
        )
    }

    /// One full-attention layer's QSA work: append every sequence's index
    /// keys, then build the wave's selection table if any row is past the
    /// budget.
    ///
    /// `kv` is the layer's index among the full-attention layers, which is
    /// also its index into a sequence's caches. Returns `None` when the layer
    /// attends densely — either the checkpoint declares it dense
    /// (`compress_ratio == 0`) or no row has more visible cells than the
    /// budget, where the selection is the identity and computing it would be
    /// arithmetic with no effect (§12.5).
    ///
    /// The keys are appended for EVERY wave regardless, because the cache is
    /// what a later, deeper wave scores against.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn layer_selection(
        &self,
        kv: usize,
        compress_ratio: usize,
        indexer: &IndexerWeights,
        rope: &FactoredRope,
        h: &Tensor,
        spans: &[SeqSpan],
        offsets: &[usize],
        idx_map: &mut HashMap<usize, Vec<IndexCache>>,
        capture: Option<&mut SpecCapture>,
        total_rows: usize,
        fwd_ticket: Option<candle::cuda_backend::wave_provenance::WaveTicket>,
    ) -> Result<Option<QsaSelection>> {
        select_layer(
            kv,
            compress_ratio,
            indexer,
            rope,
            h,
            spans,
            offsets,
            idx_map,
            capture,
            total_rows,
            &self.model.cfg.indexer,
            self.model.cfg.rms_norm_eps,
            &self.model.device,
            &self.qsa_rows,
            fwd_ticket,
        )
    }

    /// The layer sweep proper — the oracle's algebra over the wave's packed
    /// rows. See the module docs for the mixer substitutions.
    #[allow(clippy::too_many_arguments)]
    fn sweep_layers(
        &self,
        contexts: &mut [crate::models::kv_cache_utils::SequenceContext],
        seq_ids: &[usize],
        inputs: &[Tensor],
        n_decode: usize,
        offs: (&[usize], &[usize], &[usize], &[usize]),
        rows: (usize, usize),
        range: (usize, usize),
        x_in: Option<TensorCat>,
        window: &WindowResiduals,
        headers: (DecodeHeaders, DecodeHeaders),
        generation: &candle::quantized::pinned_staging::Generation,
        eps: f64,
        // The forward-scoped span's ticket — the home for every per-wave table
        // built below. `None` off CUDA, or when no tier was placed.
        fwd_ticket: Option<candle::cuda_backend::wave_provenance::WaveTicket>,
    ) -> Result<(WavePhase, Option<WaveGuard>)> {
        let (dec_off, dec_q, pre_off, pre_q) = offs;
        let (total_rows, pre_rows) = rows;
        let (layer_start, layer_end) = range;
        let (decode_headers, prefill_headers) = headers;
        let m = &self.model;
        let cfg = &m.cfg;
        let dev = &m.device;
        let hc = cfg.hc.count;
        let n_embd = cfg.hidden_size;
        let num_layers = cfg.num_layers;
        let q_lens: Vec<usize> = dec_q.iter().chain(pre_q.iter()).copied().collect();

        // ── Row embeddings: the trunk's entry into the wide stream, and the
        // draft head's `embed(t)` half. ──
        //
        // Gathered once and shared by both. A resumed wave takes its residual
        // from `x_in` and would never gather these, but if it is the window
        // that reaches the last trunk layer then the head runs behind it and
        // needs them — so the condition is "either consumer wants them", not
        // "the residual is fresh".
        //
        // One launch fills both: the fresh residual is the embedding repeated
        // across the `hc` streams, written by the gather itself, and the head
        // reads the bare rows. Both are fully written (invariant 6), and both
        // are on the forward span whenever the sweep reaches the head
        // (`WaveBuffer::HyperResidual`, `WaveBuffer::HeadRowEmbeds`) — see the
        // residual below for the window that does not.
        let fresh = x_in.is_none();
        let reaches_head = layer_end == num_layers;
        let head_embeds = reaches_head && m.mtp.is_some();
        let res_dims = (total_rows, hc, n_embd);
        // The residual's home: the forward span for a sweep that reaches the
        // head, a held window buffer for one that hands it on.
        let residual_buffer = || -> Result<Tensor> {
            if reaches_head {
                wave_empty_ticketed(res_dims, DType::F32, dev, fwd_ticket)
            } else {
                window.take(res_dims, DType::F32, dev)
            }
        };
        let entry = if fresh {
            Some(residual_buffer()?)
        } else {
            None
        };
        let row_embeds = if head_embeds {
            Some(wave_empty_ticketed(
                (total_rows, n_embd),
                DType::F32,
                dev,
                fwd_ticket,
            )?)
        } else {
            None
        };
        // Per-sequence host token ids, read once: the embedding gather below and
        // the PLE hash side both take them from here. The buffer they were read
        // through stays on the forward span as every row's id in wave order —
        // what the gather reads — so only a wave with a host-side input uploads
        // its ids again (`WaveBuffer::RowTokenIds` prices both).
        let ids = host_token_ids(inputs, fwd_ticket)?;
        let seq_tokens: Vec<Vec<u32>> = ids.per_input;
        if fresh || head_embeds {
            let ids = match ids.device {
                Some(t) => t,
                None => {
                    let flat_ids: Vec<u32> = seq_tokens.iter().flatten().copied().collect();
                    wave_from_vec_ticketed(flat_ids, (total_rows,), dev, fwd_ticket)?
                }
            };
            m.embed
                .gather_into(&ids, entry.as_ref(), row_embeds.as_ref())?;
        }

        // ── Residual: the fresh rows as gathered, or resume. ──
        //
        // **One buffer for the whole forward, updated in place.** `hc_combine`
        // adds each block's scatter into `res` where it stands, so the wide
        // `[rows, hc, hidden]` stream is this one buffer, not one per combine.
        // It crosses every layer phase's reset, so a sweep that reaches the
        // head carves it from the forward span, which outlives them all. A
        // window that stops short hands it back as `WavePhase::Residual` for
        // the next wave to resume from — past this span's reset — so there it
        // is one of the session's held window buffers.
        //
        // **A resumed residual is copied first.** `x_in` is the caller's tensor:
        // the residual a previous window handed back to be persisted, which the
        // driver passes through without a copy. Combining into it in place would
        // rewrite the persisted state under its owner — and a wave that fails
        // part-way would leave it advanced by the layers that did run. One copy
        // per resumed forward is the price; the fresh path writes its gathered
        // rows straight into a buffer it owns.
        let mut res = match x_in {
            Some(t) => {
                let src = t.to_tensor().reshape(res_dims)?;
                let dst = residual_buffer()?;
                dst.slice_set(&src, 0, 0)?;
                dst
            }
            None => entry.expect("allocated whenever the residual is fresh"),
        };

        // Row spans, beside the host token ids above (the PLE hash side).
        let spans = seq_spans(seq_ids, &q_lens)?;

        // ── Model-side RoPE for the non-paged paths; the paged kernels rotate
        // from `self.rope`, each sequence at its own rung. ──
        let theta = cfg.rope_theta;
        let dec_pos: Vec<u32> = dec_off.iter().map(|&o| o as u32).collect();
        let mut pre_pos: Vec<u32> = Vec::with_capacity(pre_rows);
        for (&o, &l) in pre_off.iter().zip(pre_q) {
            for i in 0..l {
                pre_pos.push((o + i) as u32);
            }
        }
        let dec_rope = LazyRope::new(|| {
            m.rotary
                .rope_cos_sin(&dec_pos, theta, DType::F32, dev, fwd_ticket)
        });
        let half = cfg.attn_head_dim / 2;
        let pre_rope = LazyRope::new(|| {
            let (cos, sin) = m
                .rotary
                .rope_cos_sin(&pre_pos, theta, DType::F32, dev, fwd_ticket)?;
            Ok((
                cos.reshape((1, pre_rows, half))?,
                sin.reshape((1, pre_rows, half))?,
            ))
        });
        let dec_pm: std::cell::RefCell<Option<SharedPm>> = std::cell::RefCell::new(None);
        let pre_pm: std::cell::RefCell<Option<SharedPm>> = std::cell::RefCell::new(None);
        let dec_params = BatchedAttentionParams::new(
            &dec_rope,
            false,
            &self.rope,
            decode_headers,
            dec_q,
            generation,
            &dec_pm,
        );
        let pre_params = BatchedAttentionParams::new(
            &pre_rope,
            false,
            &self.rope,
            prefill_headers,
            pre_q,
            generation,
            &pre_pm,
        );

        // QSA: the indexer's rotation table and this wave's index caches.
        // Absolute positions per row, in the wave's packed order — decode
        // rows first (one each), then each prefill sequence's span.
        let index_rope = self.index_rope().clone();
        let mut offsets_all: Vec<usize> = Vec::with_capacity(total_rows);
        offsets_all.extend_from_slice(dec_off);
        offsets_all.extend_from_slice(pre_off);
        let mut idx_map = self
            .index
            .write()
            .map_err(|_| candle::Error::Msg("index lock poisoned".into()))?;

        let mut rec = self
            .recurrent
            .write()
            .map_err(|_| candle::Error::Msg("recurrent lock poisoned".into()))?;
        let mut ple_map = self
            .ple
            .write()
            .map_err(|_| candle::Error::Msg("ple lock poisoned".into()))?;
        // Held for the sweep, like the three state maps: the capture sites are
        // inside the layer loop and a per-layer lock would be a lock per layer
        // per wave for a value that is `None` on every plain decode.
        let mut verify_guard = self
            .verify
            .write()
            .map_err(|_| candle::Error::Msg("verify lock poisoned".into()))?;
        let cap_map: &mut Option<SpecCapture> = &mut verify_guard;

        // The decode pointer table for the whole sweep — every DeltaNet layer's
        // state and tail addresses for every decode sequence, ONE host upload
        // here, before the first layer's launches, where the queue is still
        // empty. Each DeltaNet layer takes its slice, so no layer uploads a
        // table of its own behind the launches already queued. Valid for this
        // forward only: `commit_wave` exchanges the halves after the sweep.
        #[cfg(feature = "cuda")]
        let dn_table = {
            let at: HashMap<usize, usize> =
                spans.iter().enumerate().map(|(i, s)| (s.seq, i)).collect();
            let mut slot: Vec<Option<&mut RecurrentStateStore>> =
                spans.iter().map(|_| None).collect();
            for (seq, store) in rec.iter_mut() {
                if let Some(&i) = at.get(seq) {
                    slot[i] = Some(store);
                }
            }
            let stores: Vec<&mut RecurrentStateStore> = spans
                .iter()
                .zip(slot)
                .map(|(span, st)| {
                    st.ok_or_else(|| {
                        candle::Error::Msg(format!(
                            "qwen4exp: sequence {} has no recurrent store in this wave",
                            span.seq
                        ))
                    })
                })
                .collect::<Result<_>>()?;
            build_wave_table(&spans, &stores, fwd_ticket)?
        };

        // Sub-block finiteness probes for the layer bisect — sync readbacks,
        // so they exist only in `tensor-assert` diagnostic builds.
        #[cfg(feature = "tensor-assert")]
        /// Fold one layer value into its assert slot.
        ///
        /// **Asynchronous, and it reports through the drain like everything else.**
        /// This was an `eprintln!` of `max(abs(t))`, which cost a `to_scalar` — a
        /// synchronous readback, per value, per layer, per wave — and printed to
        /// stderr where nothing correlates it with the wave that produced it. Three
        /// separate problems: the readback is the fence these faults stop
        /// reproducing under, the print cannot say which site went bad *first*
        /// because stderr has no ordering against the device, and a human reading
        /// magnitudes is not a check.
        ///
        /// `assert_tensor` is one reduction kernel with no readback and no fence,
        /// and `wave_driver`'s per-wave drain then reports every bad site ordered by
        /// the kernel's own ticket — which is what makes "the first bad value in
        /// this wave" a question with an answer.
        fn probe(name: &'static str, t: &LiveTensor<'_>) {
            candle::tensor_assert::assert_tensor(t, name);
        }
        #[cfg(feature = "tensor-assert")]
        use candle::tensor_assert::site;

        // The rows the routed experts' residency scores as decode: the decode
        // rows, the verify segments and each prompt's last row
        // (`residency_rows`).
        // From the capture this sweep already holds — `self.verify` is
        // write-locked for the whole sweep.
        let verify_seqs: HashSet<usize> = cap_map
            .as_ref()
            .map(|c| c.seqs.keys().copied().collect())
            .unwrap_or_default();
        let decode_like = residency_decode_rows(
            n_decode,
            seq_ids[n_decode..]
                .iter()
                .copied()
                .zip(pre_q.iter().copied()),
            |s| verify_seqs.contains(&s),
        );

        dev.record_launches()?;

        for li in layer_start..layer_end {
            let layer = &m.layers[li];

            // ── PLE, before this layer's mixer (§12.4). ──
            let g_ple = if li == cfg.ple.layer {
                Some(crate::models::profile::gpu_span("q4e:ple", dev))
            } else {
                None
            };
            if li == cfg.ple.layer {
                // Every sequence's rows in one pass — one table gather, the
                // injection and the conv over the whole wave, the residual
                // updated in place of a per-sequence concatenation. A verifying
                // span stashes the rows it appends to the conv history; every
                // other span captures nothing.
                let at: HashMap<usize, usize> =
                    spans.iter().enumerate().map(|(i, s)| (s.seq, i)).collect();
                let mut slot: Vec<Option<&mut PleState>> = spans.iter().map(|_| None).collect();
                for (seq, st) in ple_map.iter_mut() {
                    if let Some(&i) = at.get(seq) {
                        slot[i] = Some(st);
                    }
                }
                let mut ple_spans = Vec::with_capacity(spans.len());
                for ((span, toks), st) in spans.iter().zip(&seq_tokens).zip(slot) {
                    let state = st.ok_or_else(|| {
                        candle::Error::Msg(format!(
                            "qwen4exp: sequence {} has no PLE state in this wave",
                            span.seq
                        ))
                    })?;
                    ple_spans.push(PleSpan {
                        start: span.start,
                        len: span.len,
                        tokens: toks,
                        state,
                        capture: cap_map
                            .as_ref()
                            .is_some_and(|c| c.seqs.contains_key(&span.seq)),
                    });
                }
                // Its own phase, opened and closed here before the mixer opens
                // the same generation: every PLE transient is consumed by the
                // two fused launches, and the residual they write is the
                // wave's own buffer.
                let Device::Cuda(cuda) = dev else {
                    candle::bail!("qwen4exp: the PLE block runs on CUDA");
                };
                let ple_wave = begin_wave(&cuda.cuda_stream(), LayerPhase::Attention)?;
                let captured = ple_apply_spans_fused(
                    &mut res,
                    &mut ple_spans,
                    m.ple_table.as_ref(),
                    &m.ple_w,
                    &cfg.ple,
                    eps,
                    Some(ple_wave.ticket()),
                )?;
                drop(ple_spans);
                // Kept before the phase closes: the captured rows are views on
                // it, copied into the cohort's kept-row buffer at each
                // sequence's stash row.
                if let Some(c) = cap_map.as_mut() {
                    let SpecCapture {
                        seqs, rows: kept, ..
                    } = c;
                    for (span, rows) in spans.iter().zip(captured) {
                        if let (Some(s), Some(rows)) = (seqs.get_mut(&span.seq), rows) {
                            s.ple_rows = Some(CaptureRows::keep(&kept.ple, s.row, &rows)?);
                        }
                    }
                }
                drop(ple_wave);
            }
            if let Some(g) = g_ple {
                g.end();
            }

            // ── Token mixer under the first HC module. ──
            //
            // The mixer's transients — the Gated Residual's wide intermediates,
            // the projections' q8a128 operands, the DeltaNet scan buffers —
            // belong to this phase's arena span. Opened here and dropped before
            // the FFN phase below, because `begin_wave` refuses a phase that is
            // already open: the two spans are reused in turn, not held at once.
            #[cfg(feature = "cuda")]
            let mix_wave = match dev {
                Device::Cuda(d) => Some(begin_wave(&d.cuda_stream(), LayerPhase::Attention)?),
                _ => None,
            };
            let g_pre = crate::models::profile::gpu_span("q4e:gr_pre", dev);
            // A DeltaNet layer whose four input projections run int8 reads the
            // mix's own q8a128 operand, written as the collapse stores `h` — the
            // bytes a quantize of `h` would produce, without the launch, provided
            // the mix quantizes under the projections' `Σx` convention (checked
            // where the operand is handed over). Any other layer's projections
            // take the float: an attention layer quantizes through its own hook.
            #[cfg(feature = "cuda")]
            let (h, inject, h_q8) = match &layer.mix {
                GpuLayerMix::DeltaNet(w) if w.input_mode().is_int8() => hc_mix_with_operand(
                    &res,
                    &layer.hc_attn,
                    eps,
                    mix_wave.as_ref().map(|g| g.ticket()),
                )?,
                _ => {
                    let (h, inject) = hc_mix(
                        &res,
                        &layer.hc_attn,
                        eps,
                        mix_wave.as_ref().map(|g| g.ticket()),
                    )?;
                    (h, inject, None)
                }
            };
            #[cfg(not(feature = "cuda"))]
            let (h, inject) = hc_mix(&res, &layer.hc_attn, eps, None)?;
            let inject = inject.expect("layer HC modules carry an inject");
            g_pre.end();
            #[cfg(feature = "tensor-assert")]
            {
                probe(site("q4e.hc_mix.h.L", li), &h);
                probe(site("q4e.hc_mix.inject.L", li), &inject);
            }

            // The mixer's block output for a combine over every row, or `None`
            // when the mixer has already combined each of its row groups into
            // its own rows of the residual.
            let y: Option<LiveTensor<'_>> = match &layer.mix {
                GpuLayerMix::DeltaNet(w) => {
                    // This layer's ordinal among the recurrent layers — the
                    // axis the stash is indexed on, and the same order
                    // `recurrent_layer_indices` yields, because both walk the
                    // trunk forwards.
                    let ord = cfg.layer_kinds[..li]
                        .iter()
                        .filter(|k| matches!(k, LayerKind::DeltaNet))
                        .count();
                    let mut seqs: Vec<DeltaNetSeq<'_>> = Vec::with_capacity(spans.len());
                    for span in &spans {
                        let store =
                            rec.get_mut(&span.seq).expect("ensured") as *mut RecurrentStateStore;
                        // SAFETY: each span names a distinct sequence (the
                        // driver refuses duplicates), so the mutable borrows
                        // are disjoint.
                        let store = unsafe { &mut *store };
                        let (state, out) = store.layer_state_pair_mut(li)?;
                        // A verifying span stashes this layer's post-projection
                        // operands so the accepted prefix can be replayed
                        // (`super::spec`); every other span stashes nothing.
                        let stash = cap_map.as_ref().and_then(|c| {
                            c.delta.span_of(span.seq).map(|s| StashSlot {
                                ops: &c.delta.layers[ord],
                                row: s.row,
                            })
                        });
                        seqs.push(DeltaNetSeq {
                            start: span.start,
                            len: span.len,
                            state,
                            out,
                            stash,
                        });
                    }
                    let acts = match h_q8 {
                        Some(op) if op.sum_scale == w.input_sum_scale() => {
                            DynamicActs::Int8(op.with_lead(vec![total_rows]))
                        }
                        Some(op) => candle::bail!(
                            "layer {li}: the hyper-connection mix quantized under {:?} but the \
                             DeltaNet projections read {:?} — the operand would be the wrong \
                             bytes for them",
                            op.sum_scale,
                            w.input_sum_scale()
                        ),
                        None => DynamicActs::Float(h.clone()),
                    };
                    #[cfg(feature = "cuda")]
                    let layer_table = match &dn_table {
                        Some(t) => Some(t.layer_slice(li)?),
                        None => None,
                    };
                    #[cfg(not(feature = "cuda"))]
                    let layer_table = None;
                    let mixed = quantized_delta_net_layer_forward_spans(
                        &acts,
                        h.dtype(),
                        w,
                        &cfg.delta_net,
                        &mut seqs,
                        eps,
                        layer_table.as_ref(),
                        ZGate::Sigmoid,
                        #[cfg(feature = "cuda")]
                        mix_wave.as_ref(),
                        #[cfg(not(feature = "cuda"))]
                        None,
                    )?;
                    drop(seqs);
                    // Allocated is not written: a sweep split into layer windows
                    // fills only its own ordinals, and a replay from a
                    // half-written stash advances some layers and not others.
                    // Recorded here, where the layer has actually run.
                    if let Some(c) = cap_map.as_mut() {
                        c.delta.filled[ord] = true;
                    }
                    Some(mixed)
                }
                GpuLayerMix::Attention {
                    w,
                    indexer,
                    compress_ratio,
                } => {
                    let kv = kv_index(&cfg.layer_kinds, li)?;
                    // ── QSA (§3.1): cache this segment's index keys, then
                    // select. The keys are cached at every depth — a wave that
                    // crosses the budget scores blocks the waves below it
                    // built — and the selection engages only once some row has
                    // more visible cells than the budget, which is where it
                    // stops being the identity (§12.5).
                    let g_sel = crate::models::profile::gpu_span("q4e:qsa_select", dev);
                    let qsa = self.layer_selection(
                        kv,
                        *compress_ratio,
                        indexer,
                        &index_rope,
                        &h,
                        &spans,
                        &offsets_all,
                        &mut idx_map,
                        cap_map.as_mut(),
                        total_rows,
                        // **The mixer's span, not the forward's.** These tables
                        // are rebuilt for every attention layer and dead by the
                        // end of it, so the forward-scoped span — sized for the
                        // handful of kilobytes a wave builds ONCE — filled up
                        // partway through the sweep and the rest fell back to
                        // the pool. A per-layer span resets under them, which is
                        // exactly their lifetime.
                        #[cfg(feature = "cuda")]
                        mix_wave.as_ref().map(|g| g.ticket()),
                        #[cfg(not(feature = "cuda"))]
                        None,
                    )?;
                    g_sel.end();
                    let dec_sel = match &qsa {
                        Some(s) if n_decode > 0 => Some(s.rows_slice(0, n_decode)?),
                        _ => None,
                    };
                    let pre_sel = match &qsa {
                        Some(s) if pre_rows > 0 => Some(s.rows_slice(n_decode, pre_rows)?),
                        _ => None,
                    };
                    let alayer = Qwen4ExpAttentionLayer {
                        w,
                        n_head: cfg.num_attention_heads,
                        n_kv_head: cfg.num_kv_heads,
                        head_dim: cfg.attn_head_dim,
                        rotary: &m.rotary,
                    };
                    // The two branches' outputs stay on the mixer's span. They
                    // are read once, by the combine below, inside the same
                    // phase — so owning them was a full `[rows, n_embd]` copy
                    // into a fresh pool allocation, per branch, per layer, per
                    // wave, to hand a kernel bytes it only reads. A pure-decode
                    // wave now copies nothing at all here; a mixed one pays only
                    // the `cat`.
                    //
                    // The generation is handed to the projection rather than
                    // left as `None`: the attention phase's plan already prices
                    // this layer's Q/K/V splits and out-projection output
                    // (`WaveBuffer::{QSplit, KSplit, VContiguous, OProjOutput}`,
                    // all of which name `LayerPhase::Attention`), so passing
                    // nothing spent that ground on the pool instead. It is also
                    // what binds `'w` here — without it these outputs type as
                    // `'static` while the provenance they inherit from `h` puts
                    // them on the span regardless, which is a lifetime the
                    // compiler cannot police.
                    //
                    // Each group's output is combined straight into its own row
                    // range of the residual (`hc_combine` takes a row view), so a
                    // mixed wave never concatenates the two (hot-path invariant
                    // 2). The groups read `h`, never `res`, so combining the
                    // decode rows before the prefill group runs changes nothing
                    // it sees.
                    let mut cache_refs: Vec<&mut KvCache> = contexts
                        .iter_mut()
                        .map(|c| &mut c.kv_caches.caches[kv])
                        .collect();
                    let (dec_c, pre_c) = cache_refs.split_at_mut(n_decode);
                    if n_decode > 0 {
                        // A row range of `h` is dense, so the group reads it in place.
                        let x_g = TensorCat::from_cat_tensor(
                            h.narrow(0, 0, n_decode)?.reshape((n_decode, 1, n_embd))?,
                            0,
                        )?;
                        expect_dense_view(x_g.as_cat_tensor(), "q4e decode attention rows")?;
                        // The whole attention block, not just its kernel.
                        // `decode:kernel` and `prefill:kernel` are reported by the
                        // kernel wrappers themselves, so the projections, rope, KV
                        // append and out-proj around them would otherwise go
                        // unattributed. The difference
                        // between this span and the kernel row inside it is that
                        // surrounding work.
                        let g_attn = crate::models::profile::gpu_span("q4e:attn_decode", dev);
                        let out = forward_attn_batched(
                            &alayer,
                            dec_c,
                            &x_g,
                            dec_off,
                            &dec_params,
                            kv,
                            dec_sel.as_ref(),
                            mix_wave.as_ref(),
                        )?;
                        g_attn.end();
                        combine_rows(
                            &mut res,
                            &out.reshape((n_decode, n_embd))?,
                            &inject,
                            0,
                            n_decode,
                            dev,
                        )?;
                        #[cfg(feature = "tensor-assert")]
                        probe(site("q4e.mix.y.L", li), &out);
                    }
                    if pre_rows > 0 {
                        let x_g = TensorCat::from_cat_tensor(
                            h.narrow(0, n_decode, pre_rows)?
                                .reshape((1, pre_rows, n_embd))?,
                            0,
                        )?;
                        // Starts at row `n_decode` of `h`; the projections
                        // address it from its own first element.
                        expect_dense_view(x_g.as_cat_tensor(), "q4e prefill attention rows")?;
                        let g_attn = crate::models::profile::gpu_span("q4e:attn_prefill", dev);
                        let out = forward_attn_batched(
                            &alayer,
                            pre_c,
                            &x_g,
                            pre_off,
                            &pre_params,
                            kv,
                            pre_sel.as_ref(),
                            mix_wave.as_ref(),
                        )?;
                        g_attn.end();
                        combine_rows(
                            &mut res,
                            &out.reshape((pre_rows, n_embd))?,
                            &inject,
                            n_decode,
                            pre_rows,
                            dev,
                        )?;
                        #[cfg(feature = "tensor-assert")]
                        probe(site("q4e.mix.y.L", li), &out);
                    }
                    None
                }
            };
            if let Some(y) = y {
                #[cfg(feature = "tensor-assert")]
                probe(site("q4e.mix.y.L", li), &y);
                combine_rows(&mut res, &y, &inject, 0, total_rows, dev)?;
            }
            #[cfg(feature = "tensor-assert")]
            probe(site("q4e.post_mix.res.L", li), &res);
            // The mixer's span is done with: `res` is the residual, which lives
            // outside the tier, and nothing below reads a mixer transient.
            //
            // `y` borrows this guard, and that borrow is what the compiler
            // polices — not by forcing a drop here (the borrow ends at `y`'s
            // last use, which is the combine above), but by refusing any *later*
            // read of it. Moving the combine below this line does not compile,
            // which is the property that matters: a phase output cannot be named
            // after the generation that reclaims its range.
            #[cfg(feature = "cuda")]
            drop(mix_wave);

            // ── MoE under the second HC module. The machinery consumes the
            // flat `[1, rows, hidden]` activation layout every FFN path feeds
            // it (the fused SwiGLU kernels are written for it). ──
            #[cfg(feature = "cuda")]
            let ffn_wave = match dev {
                Device::Cuda(d) => Some(begin_wave(&d.cuda_stream(), LayerPhase::Ffn)?),
                _ => None,
            };
            let g_pre2 = crate::models::profile::gpu_span("q4e:gr_pre_ffn", dev);
            // The FFN's input arrives already quantized when the module runs int8:
            // the collapse writes the operand as it stores `h2`.
            let (h2, inject2, h2_q8) = hc_mix_with_operand(
                &res,
                &layer.hc_ffn,
                eps,
                ffn_wave.as_ref().map(|g| g.ticket()),
            )?;
            let inject2 = inject2.expect("layer HC modules carry an inject");
            g_pre2.end();
            let candle::Device::Cuda(cuda) = dev else {
                candle::bail!("qwen4exp wave runs on CUDA");
            };
            // **The gap the first hunt could not see into.** `post_mix.res` was
            // clean and `moe.shared_gated` was the first bad site in the wave, and
            // everything between them — the pre-FFN HC module's output and its
            // inject — was uninstrumented. The MoE's two halves share exactly one
            // input, and they went non-finite with identical counts, which is what
            // a bad input looks like and not what bad expert weights look like.
            // So this is where the answer is.
            #[cfg(feature = "tensor-assert")]
            {
                probe(site("q4e.hc_ffn.h2.L", li), &h2);
                probe(site("q4e.hc_ffn.inject2.L", li), &inject2);
            }
            let h2_3d = h2.reshape((1, total_rows, n_embd))?;
            // Quantized ONCE, in the session's mode, into the one q8a128
            // operand every consumer of the FFN input reads — the shared
            // expert, its gate, the router, and the routed experts, whose tile
            // gather copies the 20 tiles of each 2560-wide row it routes. Handed
            // over as float, the experts gathered float rows and quantized the
            // stacked `rows × top_k` block themselves, once per layer.
            // Raw Σx — a language model's block sums stay far below f16's
            // ceiling.
            let g_acts = crate::models::profile::gpu_span("q4e:moe_acts", dev);
            let acts = match h2_q8 {
                // The mix's own operand, at the flat layout's leading dims.
                Some(op) => DynamicActs::Int8(op.with_lead(vec![1, total_rows])),
                None => to_dynamic(
                    &h2_3d,
                    m.lm_head.int8mode(),
                    cuda,
                    candle::quantized::SumScale::Raw,
                )?,
            };
            g_acts.end();
            #[cfg(feature = "tensor-assert")]
            {
                use crate::models::qwen35::quantized_moe::shared_expert_contribution;
                let sh = shared_expert_contribution(
                    &layer.moe.shared,
                    &layer.moe.shared_gate,
                    &acts,
                    DType::F32,
                )?;
                probe(site("q4e.moe.shared.L", li), &sh);
            }
            #[cfg(feature = "cuda")]
            let moe_wave = ffn_wave.as_ref();
            #[cfg(not(feature = "cuda"))]
            let moe_wave = None;
            // On the FFN span, like the mixer's output above: the combine below
            // reads it inside the same phase, so owning it was a full
            // `[rows, n_embd]` copy into a pool allocation on *every* layer.
            //
            // **The MoE block's device time, on its own span.** 512 experts per
            // layer — shared expert, router, bucketize, gather, the grouped expert
            // GEMMs (including any worker copies of non-VRAM experts) and the
            // scatter — is the largest block of device work in the model. Nothing
            // here waits on the host, so this event span is the one place that
            // device time is attributed.
            let g_moe = crate::models::profile::gpu_span("q4e:moe_routed", dev);
            // The layer's output in its three parts: the shared expert's gate is
            // applied by the combine below, which reads the block output anyway,
            // rather than by three launches of its own.
            let parts = layer
                .moe
                .forward_parts(acts, DType::F32, &decode_like, moe_wave)?;
            let routed = parts.routed.reshape((total_rows, n_embd))?;
            g_moe.end();
            #[cfg(feature = "tensor-assert")]
            probe(site("q4e.moe.routed.L", li), &routed);
            let g_comb2 = crate::models::profile::gpu_span("q4e:gr_combine_ffn", dev);
            hc_combine_gated(&mut res, &routed, &parts.shared, &inject2)?;
            g_comb2.end();
            #[cfg(feature = "tensor-assert")]
            probe(site("q4e.post_moe.res.L", li), &res);
            // Same reasoning as the mixer's drop above: `res` has left the span,
            // `y2`'s borrow ended at the combine, and the next layer's mixer
            // phase needs this one closed.
            #[cfg(feature = "cuda")]
            drop(ffn_wave);
        }
        // ── The draft head, in the same wave over the same rows. ──
        //
        // Only when the sweep reached the last trunk layer, because the head's
        // input is the trunk's *finished* residual — a windowed sweep that
        // stops early hands its residual on and the head runs behind the window
        // that completes it. Before the maps are dropped: the head selects over
        // its own index cache and needs the same borrow the trunk layers took.
        if layer_end == num_layers {
            if let Some(head) = &m.mtp {
                let mut seeds = self
                    .seeds
                    .write()
                    .map_err(|_| candle::Error::Msg("seed lock poisoned".into()))?;
                let mut spares = self
                    .seed_spares
                    .write()
                    .map_err(|_| candle::Error::Msg("seed lock poisoned".into()))?;
                let hw = HeadWave {
                    n_decode,
                    pre_rows,
                    dec_off,
                    pre_off,
                    dec_params: &dec_params,
                    pre_params: &pre_params,
                    spans: &spans,
                    offsets_all: &offsets_all,
                    index_rope: &index_rope,
                    kv_layer: cfg
                        .mtp_kv_layer()
                        .ok_or_else(|| candle::Error::msg("a head means a head KV layer"))?,
                    generation,
                };
                let embeds = row_embeds
                    .as_ref()
                    .expect("gathered whenever the head runs behind this window");
                self.head_wave_pass(
                    head,
                    contexts,
                    &mut idx_map,
                    cap_map.as_mut(),
                    &mut seeds,
                    &mut spares,
                    &res,
                    embeds,
                    &hw,
                    eps,
                )?;
            }
        }

        drop(rec);
        drop(ple_map);
        drop(idx_map);

        if layer_end < num_layers {
            let flat = res.reshape((1, total_rows, hc * n_embd))?;
            return Ok((
                WavePhase::Residual(TensorCat::from_cat_tensor(flat, 0)?),
                None,
            ));
        }

        // ── Head: the final mix IS the output norm; score decode rows + each
        // prefill's last row through the LM head in one GEMM. ──
        //
        // Decode rows, then each prefill span's LAST row — except a verifying
        // span, where EVERY row is scored.
        //
        // A prefill is normally scored once because only its final position
        // continues the sequence; the rows before it are context. A verify
        // block is the opposite: each of its positions is a proposal, and what
        // checks a proposal is the model's own prediction at the position
        // before it. Scoring only the last row would hand the accept walk one
        // row where it needs `len`, which is exactly what the driver's row
        // count catches.
        let mut sel: Vec<u32> = (0..n_decode as u32).collect();
        let mut acc = n_decode as u32;
        for (i, &l) in pre_q.iter().enumerate() {
            let verifying = cap_map.as_ref().is_some_and(|c| {
                seq_ids
                    .get(n_decode + i)
                    .is_some_and(|s| c.seqs.contains_key(s))
            });
            if verifying {
                sel.extend(acc..acc + l as u32);
            } else {
                sel.push(acc + l as u32 - 1);
            }
            acc += l as u32;
        }
        // **The rows are chosen BEFORE the mix, not after.** The mix is
        // row-wise, so mixing only the scored rows gives the same bits, and a
        // prefill wave scores a handful of its thousands of rows: mixing the
        // whole residual first ran the head's norm, its two low-rank GEMMs and
        // the collapse over every row to keep a few. A wave that scores every
        // row (all decode) takes the residual as it stands.
        //
        // The mix runs on the forward span, where the plan prices it at the
        // scored rows (`WaveBuffer::HyperHead*`). A scattered selection gathers
        // beside the residual, which is on that span too, with its index
        // (`WaveBuffer::HeadScoredResidual`, `WaveBuffer::HeadScoredRows`).
        let r_total = sel.len();
        let scored_res = select_head_rows(&res, sel, 0, fwd_ticket)?;
        let (scored, _) = hc_mix(&scored_res, &m.out_hc, eps, fwd_ticket)?;
        let acts = {
            let candle::Device::Cuda(cuda) = dev else {
                candle::bail!("qwen4exp wave runs on CUDA");
            };
            to_dynamic(
                &scored,
                m.lm_head.int8mode(),
                cuda,
                candle::quantized::SumScale::Raw,
            )?
        };
        let logits = m
            .lm_head
            .forward_dynamic(acts.as_dynamic(), DType::F32)?
            .reshape((r_total, cfg.vocab_size))?;
        // Keeping the session's per-layer lengths in step after the head is
        // the DRIVER's job (the default advance hook) — nothing more here.
        Ok((
            WavePhase::Logits(TensorCat::from_cat_tensor(logits, 0)?),
            None,
        ))
    }
}

/// `hc_combine` over rows `[start, start + len)` of the residual: the block
/// output covers exactly those rows, and the combine writes them in place
/// through a row view, reading the matching rows of `inject`.
fn combine_rows(
    res: &mut Tensor,
    block_out: &LiveTensor<'_>,
    inject: &Tensor,
    start: usize,
    len: usize,
    dev: &Device,
) -> Result<()> {
    let g_comb = crate::models::profile::gpu_span("q4e:gr_combine", dev);
    if start == 0 && len == res.dim(0)? {
        hc_combine(res, block_out, inject)?;
    } else {
        let mut rows = res.narrow(0, start, len)?;
        hc_combine(&mut rows, block_out, &inject.narrow(0, start, len)?)?;
    }
    g_comb.end();
    Ok(())
}

/// The KV-cache index of attention layer `li` (three quarters of the trunk
/// owns no cache).
fn kv_index(kinds: &[LayerKind], li: usize) -> Result<usize> {
    if !matches!(kinds[li], LayerKind::Attention) {
        candle::bail!("layer {li} is not an attention layer");
    }
    Ok(kinds[..li]
        .iter()
        .filter(|k| matches!(k, LayerKind::Attention))
        .count())
}
