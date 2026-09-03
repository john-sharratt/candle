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

use std::collections::HashMap;
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
use super::ple::{ple_apply, ple_row_ids, PleState};
use super::qsa::IndexerWeights;
use super::spec::SpecCapture;
use crate::models::batched_inference::{
    BatchedConfig, BatchedInferenceSession, ManagedBatchedModel, WaveResult, MAX_PREFILL_TOKENS,
};
use crate::models::batched_layer::{
    forward_attn_batched, BatchedAttentionParams, BatchedPrefillMeta, DecodeHeaders,
};
use crate::models::batched_model::{WaveGuard, WavePhase};
use crate::models::delta_net::StashSlot;
use crate::models::delta_net::{
    quantized_delta_net_layer_forward_spans, seq_spans, DeltaNetSeq, LayerKind,
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

impl Qwen4ExpBatched {
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
        let starting_over = offset == 0 && layer_start == 0;
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
            for (cache, ratio) in caches.iter_mut().zip(self.attention_ratios()) {
                if starting_over {
                    cache.reset();
                }
                if ratio > 0 {
                    cache.ensure_capacity(tokens, ratio)?;
                }
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
        // The Gated Residual runs F32 end to end and the projections quantize
        // their own activations — but the per-head Q/K norms run INSIDE the
        // projection at the KV arena's width, so their weights are
        // materialised here, at session creation, never inside the wave.
        // The draft head's block too. It is a full-attention layer with Q/K
        // norms of its own, and it was POPPED out of `layers` at load — so a
        // loop over `layers` alone leaves exactly one attention layer holding
        // F32 norms against BF16 activations, and the head's first pass fails
        // inside the wave rather than here.
        let head_block = self.model.mtp.as_ref().map(|h| &h.block);
        for layer in self.model.layers.iter().chain(head_block) {
            if let GpuLayerMix::Attention { w, .. } = &layer.mix {
                w.q_norm.maybe_change_dtype(dtype)?;
                w.k_norm.maybe_change_dtype(dtype)?;
            }
        }
        Ok(())
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
        // The Q/K norm weights meet activations at the KV arena's width.
        self.maybe_change_dtype(session.activation_dtype())?;
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
        } = wave;
        if seq_ids.is_empty() {
            candle::bail!("qwen4exp wave: empty batch");
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
                    let qsa = idx_snaps.remove(&s).unwrap_or_default();
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
            let acts = to_dynamic(&h2_3d, candle::quantized::Int8Mode::Off, cuda)?;
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
            to_dynamic(&scored, m.lm_head.int8mode(), cuda)?
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
