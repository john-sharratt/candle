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

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, RwLock};

use candle::quantized::cuda::to_dynamic;
use candle::{DType, Device, Result, Tensor};
use candle_nn::kv_cache::{KvCache, ModelGeometry, QWEN4EXP_KV_FACTORS};

use super::batched_attention::Qwen4ExpAttentionLayer;
use super::draft::{HeadWave, SeedStore};
use super::engine::{GpuLayerMix, Qwen4ExpGpu};
use super::hyper::{hc_combine, hc_mix};
use super::indexer::{select_layer, IndexCache, IndexSnapshot};
use super::paged_index;
use super::paged_index::{IndexPage, SealedIndex};
use super::ple::{ple_apply, ple_row_ids, PleState};
use super::qsa::IndexerWeights;
use super::spec::SpecCapture;
use crate::models::batched_inference::{
    BatchedConfig, BatchedInferenceSession, ManagedBatchedModel, ModelCoreProperties, WaveResult,
    MAX_PREFILL_TOKENS,
};
use crate::models::batched_layer::{
    forward_attn_batched, BatchedAttentionParams, BatchedPrefillMeta, DecodeHeaders,
};
use crate::models::batched_model::{WaveGuard, WavePhase};
use crate::models::delta_net::StashSlot;
use crate::models::delta_net::{
    quantized_delta_net_layer_forward_spans, seq_spans, DeltaNetSeq, ExportedLayerState, LayerKind,
    RecurrentStateStore, SeqSpan, ZGate,
};
use crate::models::draft_ladder::QWEN38_FLASH_NEXT_DRAFT;
use crate::models::prefill_utils::SharedPm;
use crate::models::qsa_selection::QsaSelection;
use crate::models::qwen35::attention::RopeTables;
use crate::models::qwen35::spec::split_block_rows;
use crate::models::tensor_cat::TensorCat;
use crate::models::verify_wave::VerifyPlan;
use crate::models::wave_admit::admit_wave_kv;
use crate::models::wave_driver::{assemble_wave_contexts, drive_wave, WaveGroups, WaveSweep};

/// Seal `rows`, taken at absolute `frame`, into a **position-free** page.
///
/// A live cache ropes its blocks at their absolute positions, so rows lifted
/// straight out of one carry the place they came from and would score correctly
/// only if they were put back exactly there. Rotating by `-frame` takes them to
/// zero, which is what makes the record injectable at any offset in any
/// conversation — the index's half of "compute once, inject anywhere".
///
/// Rows already in the zero frame come back untouched.
fn seal_page(
    rows: &Tensor,
    frame: usize,
    last_cells: usize,
    rope: &RopeTables,
) -> Result<IndexPage> {
    let keys = super::place::rotate_rows(rows, -(frame as isize), rope)?;
    Ok(IndexPage::new(keys, last_cells))
}

/// Positions the indexer's rope tables must span for `caches`.
///
/// **The whole sequence, not the tail.** Blocks used to rope at their ordinal
/// within the live tail, so a table sized to the tail was exactly right; they
/// now rope at their absolute position, and a seal turns pages back through the
/// same magnitude in the other direction. The deepest of the two is one block
/// past whatever the deepest cache accounts for.
fn index_rope_depth(caches: &[IndexCache], ratios: &[usize]) -> usize {
    caches
        .iter()
        .zip(ratios.iter())
        .map(|(c, &r)| c.indexed_tokens(r).checked_div(r).map_or(0, |b| b + 1))
        .max()
        .unwrap_or(0)
        .max(1)
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
    /// Per-sequence carried residual for the draft head's next first row — the
    /// `h(t-1)` its input assembly needs across a wave boundary. Empty on a
    /// checkpoint with no head, and reset with the other carried state when a
    /// sequence starts over. See [`super::draft`].
    pub(super) seeds: RwLock<SeedStore>,
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
    /// The indexer's rotation tables, keyed by the arena block count they
    /// cover. Separate from `rope_cs` because the indexer ropes with the
    /// oracle's [`RopeTables`] — the same rotation the reference applies to
    /// its pooled block keys, at the same width.
    index_rope: Mutex<Option<(usize, RopeTables)>>,
    /// Query rows for which a selection was built — QSA's engagement, summed
    /// over layers and waves. Zero says every row was inside the budget and
    /// the stack ran the dense arithmetic, which is what a short context
    /// should report.
    qsa_rows: AtomicU64,
    /// Position-indexed interleaved (cos,sin) table for the paged kernels,
    /// keyed by the arena block count it covers.
    rope_cs: Mutex<Option<(usize, Tensor)>>,
    /// `[head_dim/2]` F32 — full-width table with the pass-through pairs at
    /// frequency zero, from the rotary layout (partial rotary 64/256).
    pub(super) inv_freq: Tensor,
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
    pub fn set_selection_budget(&mut self, top_k: usize) {
        self.model.cfg.indexer.top_k = top_k;
    }

    /// The budget [`Self::set_selection_budget`] is currently at.
    pub fn selection_budget(&self) -> usize {
        self.model.cfg.indexer.top_k
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
                tracing::debug!(
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
            // The parent had no caches at all — the child starts empty and every
            // token the view borrows is unindexed from birth.
            None => tracing::warn!(
                parent,
                child,
                "qwen4exp: the view's parent holds no index caches, so the carve \
                 inherits none — the borrowed K/V is unindexed from the start",
            ),
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
    pub fn reset_positional_state(&self, seq: usize) -> Result<()> {
        if let Ok(mut map) = self.index.write() {
            if let Some(caches) = map.get_mut(&seq) {
                for c in caches.iter_mut() {
                    c.reset();
                }
            }
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
        // The flush ropes its block at its ABSOLUTE position, so the tables have
        // to span the whole sequence, not just the tail. The normalisation below
        // turns through the same magnitude in the other direction, so one span
        // covers both.
        let depth = index_rope_depth(caches, &ratios);
        let rope = self.index_rope_for(depth)?;
        let mut pages = Vec::with_capacity(caches.len());
        for ((c, &ratio), w) in caches.iter_mut().zip(ratios.iter()).zip(indexers.iter()) {
            if ratio == 0 {
                pages.push(SealedIndex {
                    page: IndexPage::new(c.live_rows()?, 1),
                    open: c.open_rows()?,
                });
                continue;
            }
            let frame = c.page_token_span();
            let cells = c.flush_open_block(w, &rope, ratio, cfg.rms_norm_eps)?;
            pages.push(SealedIndex {
                page: seal_page(&c.live_rows()?, frame, cells.unwrap_or(ratio), &rope)?,
                // The flush consumed the carried rows, so the page IS the whole
                // piece and there is no open block to carry with it.
                open: c.open_rows()?,
            });
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
    /// The flush ropes its block at position `n_blocks`, so the RoPE table has
    /// to span one past the deepest cache here — the same reach
    /// [`Self::seal_positional_state`] needs, for the same reason.
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
        let depth = caches
            .iter()
            .map(|c| c.live_blocks() + 1)
            .max()
            .unwrap_or(0)
            .max(1);
        let rope = self.index_rope_for(depth)?;
        // Every layer indexes the same stream, so they close the same width;
        // taken from whichever is walked last.
        let mut closed = 0usize;
        for ((c, &ratio), w) in caches.iter_mut().zip(ratios.iter()).zip(indexers.iter()) {
            if ratio == 0 {
                continue;
            }
            closed = c.close_tail_into_page(w, &rope, ratio, cfg.rms_norm_eps)?;
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
        let depth = index_rope_depth(caches, &ratios);
        let rope = self.index_rope_for(depth)?;
        let mut pages = Vec::with_capacity(caches.len());
        for ((c, &ratio), w) in caches.iter().zip(ratios.iter()).zip(indexers.iter()) {
            if ratio == 0 {
                pages.push(SealedIndex {
                    page: IndexPage::new(c.live_rows()?, 1),
                    open: c.open_rows()?,
                });
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
            let cells = fork.flush_open_block(w, &rope, ratio, cfg.rms_norm_eps)?;
            let rows = fork.live_rows()?;
            let n = rows.dim(0)?;
            if first > n {
                candle::bail!(
                    "qsa seal: seq {seq} asked for the rows from position {start_pos}, which \
                     is row {first} of a {n}-row live tail — the span starts inside the \
                     injected pages, whose rows this cannot return. Seal the span from the \
                     sequence that forwarded it, where it is the tail."
                );
            }
            let take = n - first;
            // Row `first` of the tail sits at `tail_base + first · ratio`; that
            // is the frame these rows carry and the one they are normalised out
            // of.
            let frame = c.page_token_span() + first * ratio;
            pages.push(SealedIndex {
                page: seal_page(
                    &rows.narrow(0, first, take)?,
                    frame,
                    cells.unwrap_or(ratio),
                    &rope,
                )?,
                open: fork.open_rows()?,
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
        let seal_rope = self.index_rope_for(index_rope_depth(caches, &ratios))?;
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
                // A closed page carries the frame it was roped in; the record
                // must carry none, so it is normalised on the way out.
                layer_pages.push(SealedIndex {
                    page: seal_page(&p.keys, p.roped_base, p.last_cells, &seal_rope)?,
                    // A page is already closed; only the live tail carries an
                    // open block.
                    open: p.keys.narrow(0, 0, 0)?,
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
                    layer_pages.push(SealedIndex {
                        page: IndexPage::new(c.live_rows()?, 1),
                        open: c.open_rows()?,
                    });
                    continue;
                }
                let frame = c.page_token_span();
                let mut fork = c.fork()?;
                let cells = fork.flush_open_block(w, &seal_rope, ratio, cfg.rms_norm_eps)?;
                layer_pages.push(SealedIndex {
                    page: seal_page(
                        &fork.live_rows()?,
                        frame,
                        cells.unwrap_or(ratio),
                        &seal_rope,
                    )?,
                    open: fork.open_rows()?,
                });
            }
            blobs.push((tail_tokens, paged_index::encode_aux(&[], &layer_pages)?));
        }
        Ok(blobs)
    }

    /// Install a sealed page per attention layer ahead of `seq`'s live tail.
    #[cfg(feature = "cuda")]
    pub fn push_positional_state(&self, seq: usize, blob: &[u8]) -> Result<()> {
        let cfg = &self.model.cfg;
        let (_, sealed) = paged_index::decode_aux(blob, &self.model.device)?;
        let want = cfg.kv_layers().total();
        if sealed.len() != want {
            candle::bail!(
                "qwen4exp: an injected piece carries {} index pages but this checkpoint has \
                 {want} KV layers — the layers it does not cover would select against an \
                 empty candidate set",
                sealed.len()
            );
        }
        let ratios = self.attention_ratios();
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
        // The tables must span the placement, which is deeper than anything the
        // caches hold yet — the page is about to be put at `next_base`.
        let place_depth = caches
            .iter()
            .zip(ratios.iter())
            .map(|(c, &r)| c.next_base().checked_div(r).map_or(0, |b| b + 1))
            .max()
            .unwrap_or(0)
            .max(1);
        let rope = self.index_rope_for(place_depth)?;
        for ((c, s), &ratio) in caches.iter_mut().zip(sealed.iter()).zip(ratios.iter()) {
            if ratio == 0 {
                continue;
            }
            if had_open_tail.is_none() {
                let (blocks, open) = c.seal_shape();
                had_open_tail = Some(blocks * ratio + open);
            }
            pushed = s.page.tokens(ratio)?;
            // Abutting the last placement. The base is what the page is rotated
            // to, so this is the one line that decides where the piece's rows
            // actually sit — and, since it is recorded rather than accumulated,
            // a preceding piece that carried no rows moves it and nothing else.
            let base = c.next_base();
            c.push_page(s.page.clone(), base, ratio)?;
            c.place_pending(&rope)?;
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
                    let rope = self.index_rope_for(index_rope_depth(caches, &ratios))?;
                    let mut v = Vec::with_capacity(caches.len());
                    for (c, ratio) in caches.iter().zip(ratios.iter()) {
                        // Normalised like every other index artifact: the rows
                        // are roped at their absolute positions here, and
                        // `import_aux_state` restores them into a cache whose
                        // tail opens at zero. (The injected pages ahead of the
                        // tail are not exported at all — a resume rebuilds them
                        // from the projection, and `indexed_tokens` is what
                        // reports the shortfall if one does not.)
                        v.push(SealedIndex {
                            page: seal_page(
                                &c.live_rows()?,
                                c.page_token_span(),
                                (*ratio).max(1),
                                &rope,
                            )?,
                            open: c.open_rows()?,
                        });
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
        let (ple_blob, layers) = paged_index::decode_aux(blob, &self.model.device)?;
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
                    &s.page.keys,
                    &s.open,
                    cfg.indexer.head_dim,
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
        let gdn: usize = self
            .recurrent
            .read()
            .map(|m| m.values().map(|s| s.reserved_bytes()).sum())
            .unwrap_or(0);
        let idx: usize = self
            .index
            .read()
            .map(|m| {
                m.values()
                    .map(|caches| {
                        caches
                            .iter()
                            .map(|c| {
                                c.capacity_blocks()
                                    * self.model.cfg.indexer.head_dim
                                    * std::mem::size_of::<f32>()
                            })
                            .sum::<usize>()
                    })
                    .sum()
            })
            .unwrap_or(0);
        gdn + idx
    }

    pub fn new(model: Qwen4ExpGpu) -> Result<Self> {
        // `[rope_dim/2]` inverse frequencies for the paged kernels — the
        // rotated pairs only, exactly as the hybrid builds them; the layout's
        // pass-through pairs are handled by the permutation + the `rope_cs`
        // table's identity rows.
        let (theta, rope_dim) = (model.cfg.rope_theta, model.cfg.rope_dim);
        let inv: Vec<f32> = (0..rope_dim / 2)
            .map(|j| 1f32 / theta.powf(2.0 * j as f32 / rope_dim as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv, (rope_dim / 2,), &model.device)?;
        Ok(Self {
            model,
            recurrent: RwLock::new(HashMap::new()),
            seeds: RwLock::new(SeedStore::new()),
            seeded: RwLock::new(HashSet::new()),
            verify: RwLock::new(None),
            ple: RwLock::new(HashMap::new()),
            index: RwLock::new(HashMap::new()),
            index_rope: Mutex::new(None),
            qsa_rows: AtomicU64::new(0),
            rope_cs: Mutex::new(None),
            inv_freq,
        })
    }

    pub fn engine(&self) -> &Qwen4ExpGpu {
        &self.model
    }

    /// Query rows QSA has narrowed since load — see [`Self::qsa_rows`].
    pub fn qsa_rows_selected(&self) -> u64 {
        self.qsa_rows.load(Ordering::Relaxed)
    }

    fn token_ids(t: &Tensor) -> Result<Vec<u32>> {
        t.flatten_all()?.to_dtype(DType::U32)?.to_vec1::<u32>()
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
                    cache.reset();
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
                caches
                    .iter()
                    .zip(self.attention_ratios())
                    .enumerate()
                    .filter(|(_, (_, ratio))| *ratio > 0)
                    .filter_map(|(layer, (c, ratio))| {
                        let have = c.indexed_tokens(ratio);
                        // The open block carries up to `ratio - 1` tokens the
                        // wave is about to complete, so equality is not
                        // expected — only a whole block of disagreement is.
                        (have + ratio <= offset || have > offset + ratio).then_some((layer, have))
                    })
                    .collect()
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

    /// The indexer's rotation tables, covering every position the arena can
    /// address. Built at the same width the oracle builds them
    /// (`cfg.rope_dim`), because the block keys it prepares must be the
    /// reference's block keys.
    pub(super) fn index_rope_for(&self, max_blocks: usize) -> Result<RopeTables> {
        let mut slot = self
            .index_rope
            .lock()
            .map_err(|_| candle::Error::Msg("index_rope lock poisoned".into()))?;
        if let Some((blocks, table)) = slot.as_ref() {
            if *blocks >= max_blocks {
                return Ok(table.clone());
            }
        }
        let cfg = &self.model.cfg;
        let t = RopeTables::new(
            cfg.rope_dim,
            cfg.rope_theta,
            (max_blocks * candle_nn::CHUNK_SIZE).max(1),
            &self.model.device,
        )?;
        *slot = Some((max_blocks, t.clone()));
        Ok(t)
    }

    /// The interleaved `(cos, sin)` table the paged kernels index by position.
    /// Partial rotary, so it is the LAYOUT's own table — the generic
    /// `compute_rope_cs` would rotate all 256 dims where only 64 turn; the
    /// layout fills the pass-through pairs with exact `(cos 1, sin 0)` rows.
    pub(super) fn rope_cs_for(&self, max_blocks: usize) -> Result<Tensor> {
        let mut slot = self
            .rope_cs
            .lock()
            .map_err(|_| candle::Error::Msg("rope_cs lock poisoned".into()))?;
        if let Some((blocks, table)) = slot.as_ref() {
            if *blocks == max_blocks {
                return Ok(table.clone());
            }
        }
        let t = self.model.rotary.rope_table(
            max_blocks * candle_nn::CHUNK_SIZE,
            self.model.cfg.rope_theta,
            DType::F32,
            &self.model.device,
        )?;
        *slot = Some((max_blocks, t.clone()));
        Ok(t)
    }
}

impl ManagedBatchedModel for Qwen4ExpBatched {
    fn wave_geometry(&self, act_dtype: DType) -> ModelGeometry {
        let cfg = &self.model.cfg;
        ModelGeometry {
            hidden: cfg.hidden_size,
            intermediate: cfg.moe.expert_ffn_size,
            n_head: cfg.num_attention_heads,
            n_kv_head: cfg.num_kv_heads,
            head_dim: cfg.attn_head_dim,
            experts_per_tok: cfg.moe.n_experts_used.max(1),
            n_experts: cfg.moe.n_experts.max(1),
            act_dtype,
            accum_dtype: DType::F32,
            projection_accum_roundtrip: false,
            gated_qkv: true,
            partial_rotary: true,
        }
    }

    /// This forward takes its transients from the CUDA pool (the eager Gated
    /// Residual path has not adopted the span's wave arenas), so the default
    /// cap's FFN-span pricing bounds a tier this model never allocates from —
    /// the same posture DeepSeek-V4 holds.
    ///
    /// What DOES bind is the eager GR chain itself: at its peak it holds ~6
    /// wide `[rows, hc·n_embd]` F32 intermediates — ~250 KB a row — in the
    /// pool cushion the weight zone leaves (~2–3 GiB after the expert zone
    /// opens). Bounding one forward's rows keeps the peak inside it; the
    /// pure-prefill slab slicer turns a wider fleet into sequential slabs.
    /// Fusing the GR (the §0.4 work the design doc records) removes this term.
    fn prefill_width_cap(&self, act_dtype: DType) -> usize {
        const GR_EAGER_ROW_CAP: usize = 2048;
        let mut cap = MAX_PREFILL_TOKENS.min(GR_EAGER_ROW_CAP);
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

    fn push_positional_state(&self, seq: usize, blob: &[u8]) -> Result<bool> {
        Qwen4ExpBatched::push_positional_state(self, seq, blob)?;
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
        // else the caller has on the wave.
        let dseqs: Vec<usize> = plain.iter().map(|&(s, _)| s).collect();
        let dinputs: Vec<Tensor> = plain
            .iter()
            .map(|&(_, t)| Tensor::from_vec(vec![t], (1, 1), &self.model.device))
            .collect::<Result<_>>()?;
        let pinputs: Vec<Tensor> = blocks
            .iter()
            .map(|b| Tensor::from_vec(b.clone(), (1, b.len()), &self.model.device))
            .collect::<Result<_>>()?;

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
        let cap = SpecCapture::new(
            &cohort,
            &self.model.cfg.layer_kinds,
            &self.model.cfg.delta_net,
            &self.model.device,
        )?;
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
        if let Ok(mut g) = self.verify.write() {
            *g = None;
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

    fn create_batched_session(&self, config: BatchedConfig) -> Result<BatchedInferenceSession> {
        let cfg = &self.model.cfg;
        // This model's KV threshold calibration, folded in here because this
        // override replaces the `ManagedBatchedModel` default that would
        // otherwise do it — the KV layer count differs from the transformer
        // depth on a hybrid, and dropping the fold would leave the per-model
        // row silently never reaching the compression policy.
        let mut config = config;
        config.k_hi_error_threshold_factor *= QWEN4EXP_KV_FACTORS.k_hi;
        config.k_low_error_threshold_factor *= QWEN4EXP_KV_FACTORS.k_low;
        config.v_hi_error_threshold_factor *= QWEN4EXP_KV_FACTORS.v_hi;
        config.v_low_error_threshold_factor *= QWEN4EXP_KV_FACTORS.v_low;
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
        if let Ok(mut m) = self.seeds.write() {
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
        }
        if let Ok(mut m) = self.ple.write() {
            m.remove(&seq);
        }
        if let Ok(mut m) = self.index.write() {
            m.remove(&seq);
        }
        if let Ok(mut m) = self.seeds.write() {
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
            act_dtype: _,
            adapter,
        } = wave;
        if seq_ids.is_empty() {
            candle::bail!("qwen4exp wave: empty batch");
        }
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
        let prefill_headers =
            DecodeHeaders::Prefill(BatchedPrefillMeta::new_ragged(pre_off, pre_q, &m.device)?);

        let mut contexts = assemble_wave_contexts(session, seq_ids, inputs)?;
        let contexts = contexts.as_mut_slice();

        // Admit: claim every KV chunk this wave writes, over the KV range.
        let (kv_start, kv_end) = self.kv_layer_range(layer_start, layer_end);
        admit_wave_kv(contexts, n_decode, n_prefill, kv_start, kv_end)?;

        // ── Carried state: ensure (reset at offset 0), open the GDN wave,
        // snapshot the PLE states for the failure bracket. ──
        for ((&seq, &off), &q) in seq_ids.iter().zip(&offsets).zip(&q_lens) {
            self.ensure_seq_state(seq, off, off + q, layer_start)?;
        }
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
            (decode_headers, prefill_headers),
            generation,
            eps,
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
            for (s, seed) in seed_snapshot {
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
        swept
    }
}

impl Qwen4ExpBatched {
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
        rope: &RopeTables,
        h: &Tensor,
        spans: &[SeqSpan],
        offsets: &[usize],
        idx_map: &mut HashMap<usize, Vec<IndexCache>>,
        capture: Option<&mut SpecCapture>,
        total_rows: usize,
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
        headers: (DecodeHeaders, DecodeHeaders),
        generation: &candle::quantized::pinned_staging::Generation,
        eps: f64,
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
        let needs_embeds = x_in.is_none() || (layer_end == num_layers && m.mtp.is_some());
        let row_embeds = if needs_embeds {
            let mut flat_ids: Vec<u32> = Vec::with_capacity(total_rows);
            for t in inputs {
                flat_ids.extend(Self::token_ids(t)?);
            }
            let ids = Tensor::from_vec(flat_ids, (total_rows,), dev)?;
            // The table is stored BF16 (a load-time storage width); the gather
            // widens the wave's rows to the Gated Residual's F32.
            Some(m.embed.index_select(&ids, 0)?.to_dtype(DType::F32)?)
        } else {
            None
        };

        // ── Residual: lift fresh rows into the wide stream, or resume. ──
        let mut res = match x_in {
            Some(t) => t.to_tensor().reshape((total_rows, hc, n_embd))?,
            None => {
                let x = row_embeds
                    .as_ref()
                    .expect("gathered whenever the residual is fresh");
                x.reshape((total_rows, 1, n_embd))?
                    .broadcast_as((total_rows, hc, n_embd))?
                    .contiguous()?
            }
        };

        // Per-sequence host token ids (the PLE hash side) + row spans.
        let seq_tokens: Vec<Vec<u32>> = inputs
            .iter()
            .map(Self::token_ids)
            .collect::<Result<Vec<_>>>()?;
        let spans = seq_spans(seq_ids, &q_lens)?;

        // ── RoPE tables for the paged kernels. ──
        let max_blocks = contexts
            .first()
            .and_then(|c| {
                c.kv_caches
                    .caches
                    .first()
                    .map(|k| k.k_cache().chunked_max_blocks())
            })
            .unwrap_or(0);
        let rope_cs = self.rope_cs_for(max_blocks)?;
        let theta = cfg.rope_theta;
        let dec_pos: Vec<u32> = dec_off.iter().map(|&o| o as u32).collect();
        let mut pre_pos: Vec<u32> = Vec::with_capacity(pre_rows);
        for (&o, &l) in pre_off.iter().zip(pre_q) {
            for i in 0..l {
                pre_pos.push((o + i) as u32);
            }
        }
        let dec_rope = m.rotary.rope_cos_sin(&dec_pos, theta, DType::F32, dev)?;
        let (pre_cos, pre_sin) = m.rotary.rope_cos_sin(&pre_pos, theta, DType::F32, dev)?;
        let half = cfg.attn_head_dim / 2;
        let pre_rope = (
            pre_cos.reshape((1, pre_rows, half))?,
            pre_sin.reshape((1, pre_rows, half))?,
        );
        let dec_pm: std::cell::RefCell<Option<SharedPm>> = std::cell::RefCell::new(None);
        let pre_pm: std::cell::RefCell<Option<SharedPm>> = std::cell::RefCell::new(None);
        let dec_params = BatchedAttentionParams::new(
            &dec_rope.0,
            &dec_rope.1,
            false,
            &self.inv_freq,
            &rope_cs,
            decode_headers,
            dec_q,
            generation,
            &dec_pm,
        );
        let pre_params = BatchedAttentionParams::new(
            &pre_rope.0,
            &pre_rope.1,
            false,
            &self.inv_freq,
            &rope_cs,
            prefill_headers,
            pre_q,
            generation,
            &pre_pm,
        );

        // QSA: the indexer's rotation tables and this wave's index caches.
        // Absolute positions per row, in the wave's packed order — decode
        // rows first (one each), then each prefill sequence's span.
        let index_rope = self.index_rope_for(max_blocks)?;
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

        // Sub-block finiteness probes for the layer bisect — sync readbacks,
        // so they exist only in `tensor-assert` diagnostic builds.
        #[cfg(feature = "tensor-assert")]
        fn probe(li: usize, name: &str, t: &Tensor) {
            let m = t
                .abs()
                .and_then(|a| a.flatten_all())
                .and_then(|f| f.max(0))
                .and_then(|m| m.to_dtype(DType::F32))
                .and_then(|m| m.to_scalar::<f32>());
            eprintln!("[q4e probe] L{li} {name}: {m:?}");
        }

        for li in layer_start..layer_end {
            let layer = &m.layers[li];

            // ── PLE, before this layer's mixer (§12.4). ──
            let g_ple = if li == cfg.ple.layer {
                Some(crate::models::profile::gpu_span("q4e:ple", dev))
            } else {
                None
            };
            if li == cfg.ple.layer {
                let mut parts = Vec::with_capacity(spans.len());
                for span in &spans {
                    let st = ple_map.get_mut(&span.seq).expect("ensured");
                    let row_ids = ple_row_ids(
                        &cfg.ple,
                        &seq_tokens_of(&spans, &seq_tokens, span.seq)?,
                        &mut st.prev,
                    );
                    let flat: Vec<u32> = row_ids.into_iter().flatten().collect();
                    let emb = m
                        .ple_table
                        .rows(&flat)?
                        .reshape((span.len, cfg.hidden_size))?;
                    let rows_in = res.narrow(0, span.start, span.len)?;
                    // A verifying span stashes the rows it appends to the conv
                    // history; every other span captures nothing.
                    let mut rows_out = Tensor::zeros(0, DType::F32, dev)?;
                    let capture = cap_map
                        .as_ref()
                        .is_some_and(|c| c.seqs.contains_key(&span.seq))
                        .then_some(&mut rows_out);
                    parts.push(ple_apply(
                        &rows_in, &emb, &m.ple_w, &cfg.ple, st, eps, capture,
                    )?);
                    if let Some(c) = cap_map.as_mut() {
                        if let Some(s) = c.seqs.get_mut(&span.seq) {
                            s.ple_rows = Some(rows_out);
                        }
                    }
                }
                res = Tensor::cat(&parts, 0)?;
            }
            if let Some(g) = g_ple {
                g.end();
            }

            // ── Token mixer under the first HC module. ──
            let g_pre = crate::models::profile::gpu_span("q4e:gr_pre", dev);
            let (h, inject) = hc_mix(&res, &layer.hc_attn, eps)?;
            let inject = inject.expect("layer HC modules carry an inject");
            g_pre.end();
            #[cfg(feature = "tensor-assert")]
            {
                probe(li, "hc_mix.h", &h);
                probe(li, "hc_mix.inject", &inject);
            }

            let y = match &layer.mix {
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
                    let mixed = quantized_delta_net_layer_forward_spans(
                        &h.clone(),
                        w,
                        &cfg.delta_net,
                        &mut seqs,
                        eps,
                        None,
                        ZGate::Sigmoid,
                    )?;
                    drop(seqs);
                    // Allocated is not written: a sweep split into layer windows
                    // fills only its own ordinals, and a replay from a
                    // half-written stash advances some layers and not others.
                    // Recorded here, where the layer has actually run.
                    if let Some(c) = cap_map.as_mut() {
                        c.delta.filled[ord] = true;
                    }
                    mixed.to_owned_tensor()?
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
                    let mut parts: Vec<Tensor> = Vec::with_capacity(2);
                    let mut cache_refs: Vec<&mut KvCache> = contexts
                        .iter_mut()
                        .map(|c| &mut c.kv_caches.caches[kv])
                        .collect();
                    let (dec_c, pre_c) = cache_refs.split_at_mut(n_decode);
                    if n_decode > 0 {
                        let x_g = TensorCat::from_cat_tensor(
                            h.narrow(0, 0, n_decode)?
                                .reshape((n_decode, 1, n_embd))?
                                .contiguous()?,
                            0,
                        )?;
                        let out = forward_attn_batched(
                            &alayer,
                            dec_c,
                            &x_g,
                            dec_off,
                            &dec_params,
                            kv,
                            dec_sel.as_ref(),
                            None,
                        )?;
                        parts.push(out.to_owned_tensor()?.reshape((n_decode, n_embd))?);
                    }
                    if pre_rows > 0 {
                        let x_g = TensorCat::from_cat_tensor(
                            h.narrow(0, n_decode, pre_rows)?
                                .reshape((1, pre_rows, n_embd))?
                                .contiguous()?,
                            0,
                        )?;
                        let out = forward_attn_batched(
                            &alayer,
                            pre_c,
                            &x_g,
                            pre_off,
                            &pre_params,
                            kv,
                            pre_sel.as_ref(),
                            None,
                        )?;
                        parts.push(out.to_owned_tensor()?.reshape((pre_rows, n_embd))?);
                    }
                    if parts.len() == 1 {
                        parts.pop().unwrap()
                    } else {
                        Tensor::cat(&parts, 0)?
                    }
                }
            };
            #[cfg(feature = "tensor-assert")]
            probe(li, "mix.y", &y);
            let g_comb = crate::models::profile::gpu_span("q4e:gr_combine", dev);
            res = hc_combine(&res, &y, &inject)?;
            g_comb.end();
            #[cfg(feature = "tensor-assert")]
            probe(li, "post_mix.res", &res);

            // ── MoE under the second HC module. The machinery consumes the
            // flat `[1, rows, hidden]` activation layout every FFN path feeds
            // it (the fused SwiGLU kernels are written for it). ──
            let g_pre2 = crate::models::profile::gpu_span("q4e:gr_pre_ffn", dev);
            let (h2, inject2) = hc_mix(&res, &layer.hc_ffn, eps)?;
            let inject2 = inject2.expect("layer HC modules carry an inject");
            g_pre2.end();
            let candle::Device::Cuda(cuda) = dev else {
                candle::bail!("qwen4exp wave runs on CUDA");
            };
            let h2_3d = h2.reshape((1, total_rows, n_embd))?;
            // Float activations, deliberately: the int8 expert path gathers
            // token rows as q8a1024 (hidden must tile 1024) and 2560 does not.
            // The routed experts still run their quantized weights — only the
            // activation operand stays float. Teaching the gather the 2.5-tile
            // row is recorded §0.4 work.
            // Raw Σx — a language model's block sums stay far below f16's
            // ceiling. (`Off` produces no q8a128 here anyway.)
            let acts = to_dynamic(
                &h2_3d,
                candle::quantized::Int8Mode::Off,
                cuda,
                candle::quantized::SumScale::Raw,
            )?;
            #[cfg(feature = "tensor-assert")]
            {
                use crate::models::qwen35::quantized_moe::shared_expert_contribution;
                let sh = shared_expert_contribution(
                    &layer.moe.shared,
                    &layer.moe.shared_gate,
                    &acts,
                    DType::F32,
                )?;
                probe(li, "moe.shared", &sh);
            }
            let y2 = layer
                .moe
                .forward_dynamic(acts, DType::F32, None)?
                .to_owned_tensor()?
                .reshape((total_rows, n_embd))?;
            #[cfg(feature = "tensor-assert")]
            probe(li, "moe.y2", &y2);
            let g_comb2 = crate::models::profile::gpu_span("q4e:gr_combine_ffn", dev);
            res = hc_combine(&res, &y2, &inject2)?;
            g_comb2.end();
            #[cfg(feature = "tensor-assert")]
            probe(li, "post_moe.res", &res);
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
        let (mixed, _) = hc_mix(&res, &m.out_hc, eps)?;
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
        let r_total = sel.len();
        let idx = Tensor::from_vec(sel, r_total, dev)?;
        let scored = mixed.index_select(&idx, 0)?.contiguous()?;
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
            .to_owned_tensor()?
            .reshape((r_total, cfg.vocab_size))?;
        // Keeping the session's per-layer lengths in step after the head is
        // the DRIVER's job (the default advance hook) — nothing more here.
        Ok((
            WavePhase::Logits(TensorCat::from_cat_tensor(logits, 0)?),
            None,
        ))
    }
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

/// The token segment of `seq` in this wave.
fn seq_tokens_of(
    spans: &[crate::models::delta_net::SeqSpan],
    seq_tokens: &[Vec<u32>],
    seq: usize,
) -> Result<Vec<u32>> {
    for (span, toks) in spans.iter().zip(seq_tokens) {
        if span.seq == seq {
            return Ok(toks.clone());
        }
    }
    candle::bail!("sequence {seq} has no token segment in this wave")
}
