//! DeepSeek-V4-Flash as a `ManagedBatchedModel`: the conversation engine's
//! wave-forward over the resident [`Engine`], with attention in the
//! `paged-latent` kernels and the mHC hyper-connection loop private to this
//! implementation (the scheduler sees only the trait surface — `WaveStep`
//! residuals are opaque, so the multi-stream mHC state rides them directly).
//!
//! Per-sequence corpus state (galleries + streaming compressors, one per
//! compression layer) lives behind interior mutability keyed by sequence
//! index: the trait has no sequence-free hook, so a prefill arriving at
//! offset 0 for a known index resets that sequence's state, and `prune`
//! clears everything.

use std::collections::{HashMap, HashSet};
use std::sync::RwLock;

use candle::quantized::pinned_staging::Generation;
use candle::{DType, Device, Result, Tensor};

use crate::models::batched_inference::{
    BatchedConfig, BatchedInferenceSession, ManagedBatchedModel, WaveResult, WaveStep,
    MAX_PREFILL_TOKENS,
};
use candle_nn::kv_cache::ModelGeometry;
use candle_nn::kv_cache::CHUNK_SIZE;

use super::attention::rms_norm;
use super::engine::Engine;
use super::gallery::{
    append_batch_all, gather_corpus_batched, two_stage_select_batched, AppendSlice, FloatGallery,
};
use super::kernel_attention::{
    kernel_attn_decode_assemble, kernel_attn_decode_capture, shortlist_m, DecodeAssemble,
    DecodeSel, KernelLayerSeqState, KernelLayerStatic, PrefillPrep, PrefillSel,
};
use super::linear::shared_int8_pair;
use super::paged::{GlueRun, HEAD_DIM, NOPE_BANDS, NOPE_DIM, ROPE_DIM};
use crate::models::expert_lre::PipelineStats;
use crate::models::profile::{span, span_if, ProfileSnapshot};

use super::compressor::{assemble_groups_batched, Compressor, GroupPool, SeqAssemble};
use super::desc;
use super::rope::RotaryCache;

/// A wave's pooled compressor entries: ONE `[ΣG, d]` block holding every slot's
/// completed groups in slot order, plus where each slot's run sits in it.
///
/// The block is kept WHOLE rather than split into per-slot tensors because the
/// gallery append consumes it whole — `append_batch_all` scatters the slots'
/// runs to their galleries from this one source (hot-path invariant 2b).
struct PooledGroups {
    /// `[ΣG, d]`, contiguous, slots in order.
    rows: Tensor,
    /// Per slot: `(first row, rows)`, or `None` if it completed no group.
    spans: Vec<Option<(usize, usize)>>,
}

impl PooledGroups {
    /// Total pooled groups across the wave.
    fn total(&self) -> usize {
        self.spans.iter().flatten().map(|(_, n)| n).sum()
    }
}

/// Pool EVERY slot's deferred compressor groups in ONE launch across the whole
/// wave — the cross-slot emit batch that replaces the per-slot pool. Prefill
/// batches its prompt fleet's sequences through here, decode its decode rows;
/// both were a per-slot pool (a softmax + weighted reduction + norm, ~15
/// launches) multiplied by the wave's width and the layer count.
///
/// The slots' [`GroupPool`]s go to [`Compressor::pool_and_norm`] AS THEY ARE —
/// each describes its own rows by address, so batching them widens the pool's
/// descriptor table instead of concatenating their pooling rows into one block.
/// Bit-identical per group to pooling each slot separately
/// (`pool_batched_across_seqs_matches_per_seq`). `rope` is `Some` for the roped
/// indexer keys, `None` for the pre-RoPE attention entries. `None` when no slot
/// completed a group.
fn pool_across_seqs(
    c: &Compressor,
    gps: &[Option<&GroupPool>],
    rope: Option<&RotaryCache>,
    generation: &Generation,
) -> Result<Option<PooledGroups>> {
    let some: Vec<(usize, &GroupPool)> = gps
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(i, g)| g.map(|g| (i, g)))
        .collect();
    if some.is_empty() {
        return Ok(None);
    }
    let pools: Vec<&GroupPool> = some.iter().map(|(_, g)| *g).collect();
    let rows = c.pool_and_norm(&pools, rope, generation)?; // [ΣG, d]
    let mut spans: Vec<Option<(usize, usize)>> = vec![None; gps.len()];
    let mut off = 0usize;
    for (i, g) in &some {
        let n = g.groups();
        spans[*i] = Some((off, n));
        off += n;
    }
    Ok(Some(PooledGroups { rows, spans }))
}

/// Append a wave's pooled groups to every slot's gallery in one pass —
/// [`append_batch_all`] over the whole `PooledGroups` block.
///
/// `keys` is the roped Indexer-key pool when the layer has an indexer. An HCA
/// layer has a compressor but no indexer, so its galleries carry a 1-wide
/// placeholder no selection ever reads (HCA attends every causal entry); that
/// placeholder is built once for the whole wave rather than per slot.
/// `gps` supplies each slot's group-start positions and `slot_seq` its sequence
/// id; `galleries` must hold one entry per sequence that pooled a group.
///
/// Slots are grouped BY SEQUENCE, not taken one-for-one: a speculative verify
/// wave carries a whole drafted block as consecutive decode rows of the same
/// sequence, so one gallery can receive several slots' groups in a single wave.
/// They append back to back, in slot order — the same order, and the same
/// resulting rows, as the per-slot loop this replaced.
fn append_pooled(
    entries: &PooledGroups,
    keys: Option<&PooledGroups>,
    gps: &[Option<&GroupPool>],
    slot_seq: &[usize],
    galleries: &mut HashMap<usize, &mut FloatGallery>,
    generation: &Generation,
) -> Result<()> {
    let total = entries.total();
    if total == 0 {
        return Ok(());
    }
    let key_rows = match keys {
        Some(k) => {
            if k.spans != entries.spans {
                candle::bail!(
                    "append_pooled: indexer keys and attention entries disagree on which \
                     slots closed a group — they share a ratio and a token stream, so this \
                     is a state divergence, not a shape difference"
                );
            }
            k.rows.clone()
        }
        None => Tensor::zeros((total, 1), DType::F32, entries.rows.device())?,
    };
    // `pos` is built in span order, so it stays parallel to the pooled block and
    // every run's slice of it is the same narrow as its rows'.
    let mut pos: Vec<u32> = Vec::with_capacity(total);
    let mut order: Vec<usize> = Vec::new();
    let mut runs: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for (i, span) in entries.spans.iter().enumerate() {
        let Some((off, n)) = *span else { continue };
        let gp = gps[i].expect("a pooled span implies the slot had a group");
        pos.extend_from_slice(&gp.positions);
        let seq = slot_seq[i];
        runs.entry(seq)
            .or_insert_with(|| {
                order.push(seq);
                Vec::new()
            })
            .push((off, n));
    }
    let mut slices: Vec<AppendSlice<'_>> = Vec::with_capacity(order.len());
    for seq in order {
        let gallery = galleries.remove(&seq).ok_or_else(|| {
            candle::Error::Msg(format!(
                "append_pooled: sequence {seq} pooled a group but has no gallery on this layer"
            ))
        })?;
        slices.push(AppendSlice {
            gallery,
            runs: runs.remove(&seq).expect("inserted with the order entry"),
        });
    }
    append_batch_all(&mut slices, &entries.rows, &key_rows, &pos, generation)
}

/// One layer's resident sliding-window ring, captured for turn-seal
/// persistence (Artifact A of docs/deepseek_turn_seal_persistence.md).
///
/// **No absolute position is stored.** The window is always the contiguous tail
/// of the turn, so on resume it is placed at `decode_pos − resident_len` — a
/// function of where decode resumes in the reconstructed context, exactly as
/// the chunked cache derives `rope_pos` from cumulative layout rather than a
/// persisted per-token position.
pub struct WindowRingLayer {
    /// Number of resident tokens (the sliding-window tail, ≤ `window_size`).
    pub resident_len: usize,
    /// Dense resident-window latent `[1, 1, resident_len, HEAD_DIM]` f32 (K ≡ V).
    pub kv: Tensor,
}

/// A sequence's full sliding-window ring across every layer (Artifact A).
pub struct WindowRingSnapshot {
    pub layers: Vec<WindowRingLayer>,
}

/// One sequence's per-layer corpus state plus the offset bookkeeping used to
/// detect slot reuse.
struct SeqEntry {
    /// Tokens this state has absorbed (compressor pushes). A prefill starting
    /// below this (or at 0) means the slot was freed and reused — reset.
    absorbed: usize,
    layers: Vec<KernelLayerSeqState>,
}

/// One layer's slice of a [`VerifySnapshot`]: the pre-verify compressor states +
/// gallery length, plus the per-ROW-boundary states stashed by `forward_wave`
/// during the verify pass. Rollback after a partial accept just INSTALLS the
/// state at the accepted boundary — the wave's own per-row streaming capture
/// already produced exactly the state a plain decode of the accepted prefix
/// would have (that is the decode-shaped verify's construction), so no replay
/// (and none of its per-layer kernel launches) is needed.
struct LayerVerifySnap {
    comp: Option<super::compressor::CompressorState>,
    icomp: Option<super::compressor::CompressorState>,
    gallery_len: usize,
    /// `(comp, icomp, gallery_len)` AFTER absorbing block row `k`, for each of
    /// the block's rows in order. `Arc`-clone cheap.
    row_states: Vec<(
        Option<super::compressor::CompressorState>,
        Option<super::compressor::CompressorState>,
        usize,
    )>,
}

/// Rolling speculative-acceptance state for one sequence — the driver-level
/// cost control. A speculative step only pays when the accepted tokens/step
/// exceed the verify-wave's cost ratio over a plain decode (≈2.5–3× on this
/// box: a block's diverse tokens hit the streaming expert cache far harder
/// than one decode token). The EMA tracks accepted/step over drafted steps;
/// below [`SPEC_MIN_ACCEPT`] the drafter is skipped (the step becomes a
/// 1-token verify — plain-decode cost, still lossless) and every
/// [`SPEC_PROBE_INTERVAL`] skipped steps one drafted probe re-measures, so a
/// workload shift (story → code) re-enables speculation.
struct SpecStats {
    /// EMA of accepted tokens per DRAFTED step. Seeded optimistically so fresh
    /// sequences speculate until measured otherwise.
    ema: f32,
    /// Verify-only steps since the last drafted step (probe scheduling).
    fallback_steps: u32,
}

/// EMA smoothing for [`SpecStats::ema`] (≈ last ~8 drafted steps dominate).
const SPEC_EMA_ALPHA: f32 = 0.25;
/// Draft only while the acceptance EMA clears this. Below it, a drafted step
/// loses to plain decode on this box's verify/decode cost ratio.
///
/// The break-even IS the cost ratio: a drafted step commits `ema` tokens for
/// one verify wave, a plain step commits 1 token for one decode wave, so
/// drafting wins iff `ema > verify_cost / decode_cost`. Measured in RELEASE
/// on the elastic partition with the writer-slice-patched verify metadata:
/// a drafted step is ~238 ms against a ~112 ms plain wave — ratio ≈ 2.1
/// (counting: 20.7 tok/s at 4.92 accepts, 2.32× greedy). 2.3 is that ratio
/// plus margin; StoryRewrite prose (~3.2-3.7 accepts) clears it and wins,
/// while genuinely unpredictable text still falls back to plain-cost steps
/// and re-probes. (An earlier 4.0 was derived from DEV-profile host costs —
/// the unoptimized serialization inflated the verify side of the ratio.)
const SPEC_MIN_ACCEPT: f32 = 2.3;
/// Plain-decode steps between drafted probes while below the threshold. A
/// probe costs one verify wave (≈ the cost ratio in decode waves), so the
/// steady fallback overhead is `ratio / interval` — 8 paid ~46% on this box's
/// ~3.7 ratio and dominated the fallback's throughput; 32 pays ~11% while a
/// workload shift (prose → code) is still re-detected within a few dozen
/// tokens.
const SPEC_PROBE_INTERVAL: u32 = 32;

/// Pre-verify snapshot of one sequence's streaming corpus state, taken by
/// `verify_blocks` before its decode-row forward. On a partial accept the
/// driver's `truncate_sequence` installs the stashed state at the accepted row
/// boundary, so the compressors/galleries never retain rejected draft tokens —
/// without this, a rejected tail stays absorbed (wrong rows in the partial-group
/// buffer, a group pooled over draft tokens in the gallery, `group_idx` advanced)
/// and the re-decoded positions get absorbed AGAIN as duplicate, shifted groups:
/// the model re-attends duplicated context and repeats itself.
struct VerifySnapshot {
    /// Absolute position of the block's first token (`q_start`).
    base: usize,
    /// The verify block's length (`[committed, drafts…]`).
    block_len: usize,
    layers: Vec<LayerVerifySnap>,
}

/// DeepSeek's batched wave model. See the module docs.
pub struct BatchedEngine {
    engine: Engine,
    layer_static: Vec<KernelLayerStatic>,
    seq_state: RwLock<HashMap<usize, SeqEntry>>,
    /// Optional DSpark speculative-decode drafter (loaded via [`Self::with_drafter`]). When
    /// present, `speculative_draft` proposes a block per decode step; absent → plain decode.
    /// Behind a `Mutex` because the streaming expert cache mutates its LRU residency each draft,
    /// while the `ManagedBatchedModel` hooks are all `&self` (the model is immutable during decode).
    drafter: Option<std::sync::Mutex<super::dspark::DsparkDrafter>>,
    /// Per-sequence target-feature stash keyed by ABSOLUTE token position: `seq → (pos → Hctx
    /// source)`. `forward_wave` records the concatenated target-layer hidden for every scored row at
    /// its absolute position; `speculative_draft` reads the feature at `q_start-1` — the position
    /// whose hidden predicted the token the drafter is about to extend. (Keying by seq alone and
    /// overwriting to the last scored row misaligns the feature by the block length after a
    /// partial-accept verify, which conditions the draft on a future/rejected position.)
    target_feat: RwLock<HashMap<usize, HashMap<usize, Tensor>>>,
    /// Set transiently by `verify_blocks` to the sequences whose prefill rows should ALL be
    /// scored (not just the last), so one forward over the `[committed, drafts…]` blocks yields
    /// the per-position next-token logits the speculative driver needs. Cleared immediately after
    /// the forward. Empty outside a batched verify (normal prefill scores only its last row —
    /// scoring every prompt row would materialise a ~1 GB discarded logits tensor).
    verify_all_rows: RwLock<Vec<usize>>,
    /// Per-sequence pre-verify corpus-state snapshot (see [`VerifySnapshot`]): inserted by
    /// `verify_block` before its forward (which also stashes the block's projected compressor
    /// rows into it, per layer), consumed by `truncate_sequence` — full accept discards it,
    /// partial accept restores + replays the accepted prefix.
    verify_snap: RwLock<HashMap<usize, VerifySnapshot>>,
    /// Per-sequence rolling speculative-acceptance stats (see [`SpecStats`]): updated by
    /// `rollback_verify_state` after every DRAFTED step, read by `speculative_draft` to fall
    /// back to verify-only (plain-decode-cost) steps while acceptance is below the economic
    /// threshold, with periodic probe steps so a workload shift re-enables drafting.
    spec_stats: RwLock<HashMap<usize, SpecStats>>,
    /// Shared expert-miss tracker for the graduated draft width: `(last global
    /// expert hits, last global expert misses, miss-rate EMA)`. Deltas of the
    /// expert pipeline's counters between drafted steps give the recent
    /// verify-wave miss rate; the EMA shaves the draft block down while the
    /// cache is missing hard (every drafted token widens the verify wave's
    /// routed-expert union, so at high miss rates draft width is a direct
    /// expert-DMA multiplier). Global across sequences — the cache pressure it
    /// measures is global.
    spec_miss: RwLock<(usize, usize, f32)>,
    /// The target-model layer indices (0-based) whose per-token hidden states condition the DSpark
    /// drafter (paper Eq. 2, `dflash.target_layers`). Empty when no drafter is attached. When set,
    /// `forward_wave` captures `head_reduce(h)` at each of these layers and stashes their
    /// concatenation as each scored sequence's `target_feat` — the faithful `Hctx` source (vs a
    /// single final hidden replicated, which starves the drafter of layer diversity → low acceptance).
    target_layers: Vec<usize>,
}

impl BatchedEngine {
    pub fn new(engine: Engine) -> Result<Self> {
        let cfg = engine.cfg();
        let mut layer_static = Vec::with_capacity(cfg.n_layers);
        let ws = std::sync::Arc::new(super::paged::LatentWorkspace::build(
            engine.engine_device(),
        )?);
        for l in 0..cfg.n_layers {
            let (theta, orig) = cfg.rope_params(l);
            layer_static.push(KernelLayerStatic::new(
                &engine.engine_layer(l).attn,
                theta,
                orig,
                cfg.rope_factor,
                cfg.beta_fast,
                cfg.beta_slow,
                ws.clone(),
                engine.engine_device(),
            )?);
        }
        Ok(Self {
            engine,
            layer_static,
            seq_state: RwLock::new(HashMap::new()),
            drafter: None,
            target_feat: RwLock::new(HashMap::new()),
            verify_all_rows: RwLock::new(Vec::new()),
            verify_snap: RwLock::new(HashMap::new()),
            spec_stats: RwLock::new(HashMap::new()),
            spec_miss: RwLock::new((0, 0, 0.0)),
            target_layers: Vec::new(),
        })
    }

    /// Attach a pre-loaded DSpark drafter, enabling speculative decode through the generic
    /// `ManagedBatchedModel` hook. The drafter shares the target's embedding + LM head (borrowed at
    /// draft time).
    ///
    /// **Load order matters.** The drafter's int8 backbone (attention + norms + router + shared) is
    /// GPU-resident (~1 GB); its 3×256 routed experts live in host RAM and stream into a small VRAM
    /// slot set on demand ([`super::dspark_experts::DsparkStreamingMoe`], VRAM-adaptive count). The
    /// target engine sizes its expert pool greedily to *all* free VRAM (spilling the remainder to
    /// the pinned pool), so the engine must be loaded **first**; the drafter then loads into the
    /// engine's activation headroom (shared with target KV/activations), leaving the target's expert
    /// pool — and its pinned remainder — at the baseline that fits the page-lock ceiling. On a
    /// device below the smallest slot tier (≤ 24 GiB) the drafter's `load` already errored, so this
    /// is never reached and speculative stays disabled.
    pub fn with_drafter(mut self, drafter: super::dspark::DsparkDrafter) -> Result<Self> {
        // `dflash.target_layers` (1-based, [41,42,43] on the 43-layer target) name the three
        // consecutive target layers whose RAW hidden states condition the drafter (llama.cpp
        // `dflash.cpp`: the encoder input is `target_layers.size()·n_embd` concatenated raw residual
        // streams — no norm, no per-layer reduction beyond the model's dim-wide readout). Shift to
        // 0-based → [40,41,42], the last three layer outputs. `forward_wave` captures `head_reduce(h)`
        // (our hc_mult→dim readout) after each and stashes the concatenation.
        let n = self.num_layers();
        self.target_layers = drafter
            .cfg
            .target_layers
            .iter()
            .map(|&l| if l >= 1 && l <= n { l - 1 } else { l })
            .collect();
        self.drafter = Some(std::sync::Mutex::new(drafter));
        Ok(self)
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// Turn seal — **Artifact B** (docs/deepseek_turn_seal_persistence.md):
    /// close every compression layer's trailing partial `comp`/`icomp` groups
    /// into the gallery for `seq`. After this the live sliding-window tail is
    /// fully represented in the compressed corpus, so the turn can be persisted
    /// and resumed from the corpus alone. No-op for an unknown sequence.
    pub fn seal_sequence(&self, seq: usize) -> Result<()> {
        let mut map = self
            .seq_state
            .write()
            .map_err(|_| candle::Error::Msg("seq_state lock poisoned".into()))?;
        if let Some(e) = map.get_mut(&seq) {
            for layer in e.layers.iter_mut() {
                layer.seal_close()?;
            }
        }
        Ok(())
    }

    /// **Artifact C** — snapshot `seq`'s sealed compressed corpus in native
    /// durable form: one [`CorpusSnapshot`] per layer, `None` on SWA layers
    /// (no gallery). Call AFTER [`Self::seal_sequence`] so the closed partials
    /// are included. Errors if the sequence is unknown.
    pub fn corpus_snapshot(
        &self,
        seq: usize,
    ) -> Result<Vec<Option<super::gallery::CorpusSnapshot>>> {
        let map = self
            .seq_state
            .read()
            .map_err(|_| candle::Error::Msg("seq_state lock poisoned".into()))?;
        let e = map.get(&seq).ok_or_else(|| {
            candle::Error::Msg(format!("corpus_snapshot: unknown sequence {seq}"))
        })?;
        e.layers.iter().map(|l| l.snapshot_gallery()).collect()
    }

    /// Resume — **Artifact C**: rebuild `seq`'s per-layer corpus state fresh and
    /// inject the persisted per-layer snapshots into the galleries at
    /// RECONSTRUCTED positions. The streaming compressors restart empty (their
    /// partials were closed at seal), so the first post-resume token opens a new
    /// group at `absorbed`. `snaps` must have exactly `num_layers()` entries (as
    /// produced by [`Self::corpus_snapshot`]).
    ///
    /// `corpus_base` is the absolute position the turn's tokens start at in the
    /// reconstructed context; each layer's entry `i` is injected at
    /// `corpus_base + i · ratio` (the layer's compression ratio from config) —
    /// the position follows the layout, never the (unstored) original. `absorbed`
    /// is the resume decode position (for slot-reuse detection).
    pub fn corpus_restore(
        &self,
        seq: usize,
        snaps: &[Option<super::gallery::CorpusSnapshot>],
        corpus_base: usize,
        absorbed: usize,
    ) -> Result<()> {
        if snaps.len() != self.num_layers() {
            candle::bail!(
                "corpus_restore: {} snapshots for {} layers",
                snaps.len(),
                self.num_layers()
            );
        }
        let cfg = self.engine.cfg();
        let dev = self.engine.engine_device();
        let mut entry = self.fresh_seq_entry()?;
        for (l, snap) in snaps.iter().enumerate() {
            if let Some(s) = snap {
                let ratio = cfg.compress_ratio(l) as u32;
                let base = corpus_base as u32;
                let positions: Vec<u32> = (0..s.len as u32).map(|i| base + i * ratio).collect();
                entry.layers[l].restore_gallery(dev, s, &positions)?;
            }
        }
        entry.absorbed = absorbed;
        let mut map = self
            .seq_state
            .write()
            .map_err(|_| candle::Error::Msg("seq_state lock poisoned".into()))?;
        map.insert(seq, entry);
        Ok(())
    }

    /// Resume a sequence's full decode state from its persisted artifacts at a
    /// (possibly new) absolute frame — **the resume path**. `corpus_snaps`
    /// (per-layer Artifact C) and `window` (Artifact A) are injected at
    /// RECONSTRUCTED positions determined by the reconstruction layout, not read
    /// from the artifacts:
    /// - the compressed corpus at `base + i · ratio` (per-layer ratio);
    /// - the raw sliding-window ring as the contiguous tail ending at the resume
    ///   position `base + total_tokens` (so its `base_pos` is
    ///   `decode_pos − resident_len`);
    /// - the session's logical offset set to the resume position so the first
    ///   decode continues there.
    ///
    /// `base` is where this conversation's frame starts in the reconstruction
    /// (0 for a standalone reopen; > 0 when injected after a prefix), and
    /// `total_tokens` is the conversation's token count. The compressors restart
    /// empty (partials were closed at seal), so decode opens a fresh group at the
    /// resume position. The `seq` slot must already exist (`create_sequence`).
    pub fn resume_sequence(
        &self,
        session: &mut BatchedInferenceSession,
        seq: usize,
        corpus_snaps: &[Option<super::gallery::CorpusSnapshot>],
        window: &WindowRingSnapshot,
        base: usize,
        total_tokens: usize,
    ) -> Result<()> {
        let decode_pos = base + total_tokens;
        self.corpus_restore(seq, corpus_snaps, base, decode_pos)?;
        self.window_ring_restore(session, seq, window, decode_pos)?;
        session.set_sequence_offset(seq, decode_pos)?;
        Ok(())
    }

    /// **Artifact A** — snapshot `seq`'s resident sliding-window ring across all
    /// layers: per layer the dense (dequantized) resident-window latent. This is
    /// the transient tail that warm-starts the ring on resume (the durable
    /// conversation state is the corpus, Artifact C). K≡V single latent, so one
    /// tensor per layer; FP8/BF16 ⇄ F32 is lossless for the writer formats, so a
    /// restore re-quantizes to byte-identical arena bytes. No absolute position
    /// is captured — it is reconstructed from the resume frame (see
    /// [`Self::window_ring_restore`]).
    pub fn window_ring_snapshot(
        &self,
        session: &BatchedInferenceSession,
        seq: usize,
    ) -> Result<WindowRingSnapshot> {
        let dev = self.engine.engine_device();
        let mut layers = Vec::with_capacity(self.num_layers());
        for backing in session.backings() {
            let resident_len = backing.resident_len(seq)?;
            let kv = if resident_len == 0 {
                Tensor::zeros((1, 1, 0, HEAD_DIM), DType::F32, dev)?
            } else {
                let (k, _v) = backing.read_contiguous(seq, 0, resident_len)?;
                k // K ≡ V single latent
            };
            layers.push(WindowRingLayer { resident_len, kv });
        }
        Ok(WindowRingSnapshot { layers })
    }

    /// Resume — **Artifact A**: rebuild `seq`'s sliding-window ring from a
    /// snapshot so the first post-resume decode continues the sliding window at
    /// the correct absolute frame. `decode_pos` is where decode resumes in the
    /// reconstructed context (the tail of the layout); the window is the
    /// contiguous tail ending there, so each layer is written at resident offset
    /// 0 with `base_pos = decode_pos − resident_len` — the position is
    /// RECONSTRUCTED from the resume frame, never read from the artifact. The
    /// sequence slots must already exist in `session` (e.g. via
    /// `create_sequence`); the caller sets the session offset to `decode_pos`
    /// (or use [`Self::resume_sequence`]).
    pub fn window_ring_restore(
        &self,
        session: &BatchedInferenceSession,
        seq: usize,
        snap: &WindowRingSnapshot,
        decode_pos: usize,
    ) -> Result<()> {
        if snap.layers.len() != self.num_layers() {
            candle::bail!(
                "window_ring_restore: {} layers for {} model layers",
                snap.layers.len(),
                self.num_layers()
            );
        }
        for (l, backing) in session.backings().iter().enumerate() {
            let layer = &snap.layers[l];
            backing.ensure_sequence_allocated(seq)?;
            let base = decode_pos.checked_sub(layer.resident_len).ok_or_else(|| {
                candle::Error::Msg(format!(
                    "window_ring_restore: decode_pos {decode_pos} < resident_len {}",
                    layer.resident_len
                ))
            })? as u32;
            if layer.resident_len > 0 {
                backing.ensure_for_offset(seq, 0, layer.resident_len)?;
                backing.write_contiguous(seq, 0, &layer.kv, &layer.kv)?;
                backing.set_len(seq, layer.resident_len);
            }
            backing.set_window_base_pos(seq, base)?;
        }
        Ok(())
    }

    fn fresh_seq_entry(&self) -> Result<SeqEntry> {
        let cfg = self.engine.cfg();
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            layers.push(KernelLayerSeqState::new(
                &self.engine.engine_layer(l).attn,
                cfg.index_head_dim,
                self.engine.engine_device(),
            )?);
        }
        Ok(SeqEntry {
            absorbed: 0,
            layers,
        })
    }

    /// Ensure per-seq state exists and matches the incoming positions; reset
    /// on slot reuse (prefill starting at/below a position we've already
    /// absorbed past, at offset 0).
    fn ensure_seq_state(&self, seq: usize, start_pos: usize) -> Result<()> {
        let mut map = self
            .seq_state
            .write()
            .map_err(|_| candle::Error::Msg("seq_state lock poisoned".into()))?;
        let reset = match map.get(&seq) {
            None => true,
            Some(e) => start_pos == 0 && e.absorbed > 0,
        };
        if reset {
            map.insert(seq, self.fresh_seq_entry()?);
        }
        Ok(())
    }

    /// Snapshot `seq`'s streaming corpus state ahead of a verify forward (see
    /// [`VerifySnapshot`]). Cheap: per layer a handful of `Arc` clones + the
    /// gallery length. No-op when the sequence has no state yet (nothing to
    /// roll back — `truncate_sequence` then degrades to the KV truncation).
    fn snapshot_verify_state(&self, seq: usize, base: usize, block_len: usize) -> Result<()> {
        let map = self
            .seq_state
            .read()
            .map_err(|_| candle::Error::Msg("seq_state lock poisoned".into()))?;
        let Some(entry) = map.get(&seq) else {
            return Ok(());
        };
        let layers = entry
            .layers
            .iter()
            .map(|ls| LayerVerifySnap {
                comp: ls.comp.as_ref().map(|c| c.state_snapshot()),
                icomp: ls.icomp.as_ref().map(|c| c.state_snapshot()),
                gallery_len: ls.gallery.as_ref().map_or(0, |g| g.len()),
                row_states: Vec::new(),
            })
            .collect();
        drop(map);
        self.verify_snap
            .write()
            .map_err(|_| candle::Error::Msg("verify_snap lock poisoned".into()))?
            .insert(
                seq,
                VerifySnapshot {
                    base,
                    block_len,
                    layers,
                },
            );
        Ok(())
    }

    /// Roll `seq`'s streaming corpus state back to `tokens` total absorbed
    /// tokens after a speculative verify. Consumes the pre-verify snapshot: a
    /// full accept (`tokens ≥ base + block_len`) discards it — the absorbed
    /// block IS the accepted text; a partial accept installs the stashed state
    /// at the accepted row boundary — bit-identical to having absorbed only
    /// those tokens, because the states ARE the wave's own per-row streaming
    /// capture (exact per-token decode semantics by construction).
    fn rollback_verify_state(&self, seq: usize, tokens: usize) -> Result<()> {
        let Some(snap) = self
            .verify_snap
            .write()
            .map_err(|_| candle::Error::Msg("verify_snap lock poisoned".into()))?
            .remove(&seq)
        else {
            return Ok(());
        };
        let accepted = tokens.saturating_sub(snap.base);
        // Rolling acceptance (DRAFTED steps only — a 1-token verify-only step
        // says nothing about the drafter): feeds `speculative_draft`'s
        // fallback decision.
        if snap.block_len > 1 {
            let kept = accepted.min(snap.block_len) as f32;
            let mut stats = self
                .spec_stats
                .write()
                .map_err(|_| candle::Error::Msg("spec_stats lock poisoned".into()))?;
            let s = stats.entry(seq).or_insert(SpecStats {
                // Seed ONE accept above the threshold: the session's FIRST
                // draft conditions on a 1-wide feature window (only the
                // prefill's last-row feature exists yet) and is the noisiest
                // step it will ever take — an at-threshold seed let that
                // single unlucky draft gate speculation for a whole
                // SPEC_PROBE_INTERVAL (measured: 33 plain steps = 25% of a
                // 128-token session at plain rate). One accept of margin
                // absorbs one bad opener; two consecutive misses still gate
                // within ~2 steps on genuinely unpredictable text.
                ema: SPEC_MIN_ACCEPT + 1.0,
                fallback_steps: 0,
            });
            s.ema = (1.0 - SPEC_EMA_ALPHA) * s.ema + SPEC_EMA_ALPHA * kept;
        }
        if accepted >= snap.block_len {
            return Ok(()); // full accept: the absorbed state is already exact
        }
        let mut map = self
            .seq_state
            .write()
            .map_err(|_| candle::Error::Msg("seq_state lock poisoned".into()))?;
        let Some(entry) = map.get_mut(&seq) else {
            return Ok(());
        };
        for (l, mut lsnap) in snap.layers.into_iter().enumerate() {
            let ls = &mut entry.layers[l];
            // Install the streaming state at the accepted boundary: the wave's
            // own per-row capture already advanced through the accepted prefix
            // with exact per-token decode semantics, so its row-`k` state IS
            // the state a plain decode of those tokens produces — no replay,
            // no per-layer launches. `accepted == 0` installs the pre-block
            // state; otherwise the state after row `accepted-1`. The gallery
            // truncates to the recorded length (its appends are append-only
            // within the wave, so entries from rejected rows sit past it).
            let (comp_s, icomp_s, glen) = if accepted == 0 {
                (lsnap.comp, lsnap.icomp, lsnap.gallery_len)
            } else {
                if accepted > lsnap.row_states.len() {
                    candle::bail!(
                        "verify rollback: {} accepted rows but only {} row-boundary states \
                         stashed (layer {l})",
                        accepted,
                        lsnap.row_states.len()
                    );
                }
                let (c, ic, g) = lsnap.row_states.swap_remove(accepted - 1);
                (c, ic, g)
            };
            if let (Some(c), Some(s)) = (ls.comp.as_mut(), comp_s) {
                c.state_restore(s);
            }
            if let (Some(c), Some(s)) = (ls.icomp.as_mut(), icomp_s) {
                c.state_restore(s);
            }
            if let Some(g) = ls.gallery.as_mut() {
                g.truncate(glen);
            }
        }
        entry.absorbed = tokens;
        Ok(())
    }

    /// Extract host token ids from an input tensor (`[1, s]` or `[s]`).
    /// Inputs are scheduler-built host tensors; when one arrives on the GPU
    /// this is a transfer — counted by the readback instrumentation.
    fn token_ids(t: &Tensor) -> Result<Vec<u32>> {
        if t.device().is_cuda() {
            super::readback::note_readback();
        }
        t.flatten_all()?.to_dtype(DType::U32)?.to_vec1::<u32>()
    }

    /// Embed a list of token ids: host-resident gather, one upload.
    fn embed_rows(&self, ids: &[u32]) -> Result<Tensor> {
        let dim = self.engine.cfg().dim;
        let idt = Tensor::from_vec(ids.to_vec(), ids.len(), &Device::Cpu)?;
        self.engine
            .embed()
            .index_select(&idt, 0)?
            .reshape((1, ids.len(), dim))?
            .to_dtype(DType::F32)?
            .to_device(self.engine.engine_device())
    }
}

impl ManagedBatchedModel for BatchedEngine {
    fn num_layers(&self) -> usize {
        self.engine.layer_count()
    }

    fn n_kv_head(&self) -> usize {
        1
    }

    fn head_dim(&self) -> usize {
        HEAD_DIM
    }

    fn wave_geometry(&self, act_dtype: DType) -> ModelGeometry {
        let cfg = self.engine.cfg();
        ModelGeometry {
            hidden: cfg.dim,
            // Per-expert intermediate — the MoE FFN phase is priced per routed
            // expert row, exactly like the qwen3 geometry.
            intermediate: cfg.moe_inter_dim,
            n_head: cfg.n_heads,
            // Single-latent MLA: one 576-wide latent per token stands for K≡V.
            n_kv_head: 1,
            head_dim: HEAD_DIM,
            experts_per_tok: cfg.n_activated_experts.max(1),
            n_experts: cfg.n_routed_experts.max(1),
            act_dtype,
            // The int8 tensor-core kernels emit F32 before the cast back to
            // `act_dtype`; both buffers are live at once, so both are planned.
            accum_dtype: DType::F32,
            // Moot here — this forward takes its transients from the CUDA pool
            // rather than the span (see `prefill_width_cap` below), so no
            // projection buffer is carved from a wave arena at all.
            projection_accum_roundtrip: false,
            gated_qkv: false,
            partial_rotary: false,
        }
    }

    fn prefill_width_cap(&self, act_dtype: DType) -> usize {
        // DeepSeek's forward takes its transients from the CUDA pool (it has
        // not adopted the span's wave arenas), so the default cap's FFN-span
        // pricing bounds a tier this model never allocates from — and at the
        // 8-way expert fan-out it sliced an 8-prompt fleet into three waves,
        // tripling the per-wave fixed costs (the per-layer routing readback +
        // expert-set assembly) that ARE the prefill wall. The engine's pool
        // cushion is reserved at load for exactly this activation peak, so the
        // real ceilings are compute saturation and what the KV side can admit
        // — which includes the expert ground the elastic boundary would cede
        // (`reclaimable_kv_bytes` below).
        let mut cap = MAX_PREFILL_TOKENS;
        if let Some(kv_fits) = self.kv_width_cap(act_dtype) {
            cap = cap.min(kv_fits);
        }
        cap
    }

    fn reclaimable_kv_bytes(&self) -> usize {
        // The weight zone's capacity above its floor: `request_kv_ground` cedes
        // it to stuck KV claims, so a prefill wave sized against it is
        // admissible even when few KV regions stand free. Without this the
        // pure-prefill slicer sized ingest sweeps against the ~332 load-time
        // regions (~6.5k tokens) while ~39 GB of cedeable expert span sat
        // uncounted; `MAX_PREFILL_TOKENS` remains the ceiling above.
        self.engine.experts().cedeable_span_bytes()
    }

    fn maybe_change_dtype(&self, _dtype: DType) -> Result<()> {
        // DeepSeek's dtypes are baked at load: norm/compressor constants are
        // widened to F32 once (`load_compressor`), the attention kernels take
        // bf16 in and emit F32, and the int8 projections quantize from F32.
        // There is nothing to re-materialise per session dtype — the forward
        // accepts the activation dtype the engine was built for.
        Ok(())
    }

    fn device(&self) -> &Device {
        self.engine.engine_device()
    }

    fn create_batched_session(&self, config: BatchedConfig) -> Result<BatchedInferenceSession> {
        use candle_nn::kv_cache::{ChunkedKvBacking, KvFormat};
        // DeepSeek pins the WRITER format to the reference two-region latent
        // window: the non-RoPE dims `[0,448)` are FP8 E4M3, the 64 RoPE dims
        // `[448,512)` stay BF16 (the rope tail must not lose precision). The
        // single-latent backing splits the two regions per head automatically
        // (`alloc_block_chunks`): the nope bands take this writer format tag,
        // the rope bands are pinned BF16. The fused scatter + fresh-diagonal
        // quant dispatch on the per-band format tag. The window stores FP8 at
        // unit scale (a per-64 ue8m0 scale would need per-token record scales —
        // the window ring is written a token at a time into a per-chunk scalar
        // record, so a non-unit scale is not representable here).
        let cfg = BatchedConfig {
            k_format: KvFormat::Float(DType::F8E4M3),
            v_format: KvFormat::Float(DType::F8E4M3),
            ..config
        };
        let first = ChunkedKvBacking::new_with_format_adaptive(
            1,
            1,
            HEAD_DIM,
            cfg.k_format,
            cfg.v_format,
            self.engine.engine_device(),
            cfg.initial_seq_len,
            None,
        )?;
        first.set_single_latent(true);
        let mut backings = Vec::with_capacity(self.num_layers());
        backings.push(first.clone());
        for layer_idx in 1..self.num_layers() {
            backings.push(first.new_layer(layer_idx, 1, cfg.initial_seq_len));
        }
        Ok(BatchedInferenceSession::new_with_backings(
            backings,
            cfg,
            self.engine.engine_device(),
        ))
    }

    fn prune(&self) -> Result<()> {
        if let Ok(mut map) = self.seq_state.write() {
            map.clear();
        }
        if let Ok(mut map) = self.verify_snap.write() {
            map.clear();
        }
        if let Ok(mut map) = self.target_feat.write() {
            map.clear();
        }
        if let Ok(mut map) = self.spec_stats.write() {
            map.clear();
        }
        Ok(())
    }

    /// Session KV truncation + streaming-corpus rollback: consume the pre-verify
    /// [`VerifySnapshot`], restore the compressor/gallery state, and replay
    /// exactly the accepted prefix — so rejected draft tokens never stay
    /// absorbed (the KV truncation alone cannot see that state).
    fn truncate_sequence(
        &self,
        session: &mut BatchedInferenceSession,
        seq: usize,
        tokens: usize,
    ) -> Result<()> {
        session.truncate_sequence_to_tokens(seq, tokens)?;
        self.rollback_verify_state(seq, tokens)
    }

    fn expert_stats(&self) -> Option<PipelineStats> {
        Some(self.engine.experts().expert_stats())
    }

    fn reset_expert_stats(&self) {
        self.engine.experts().reset_expert_stats();
    }

    fn snapshot_profiles(&self) -> ProfileSnapshot {
        // Drain the expert-pipeline worker profile (upload-wait vs GEMM vs
        // eviction) so the per-phase Bulk/Single profile tables surface the
        // MoE-internal breakdown — the forward thread's coarse `moe:submit`
        // span otherwise hides where the ~100ms/token of decode MoE goes.
        self.engine.experts().snapshot_profiles()
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
        if decode_inputs.len() != decode_seqs.len()
            || prefill_inputs.len() != prefill_seqs.len()
            || glue_inputs.len() != glue_seqs.len()
        {
            candle::bail!("forward_wave: input/seq length mismatch");
        }
        // The KV↔expert boundary's GROWING direction, in the one gap it is
        // legal in: between forwards, before this wave opens any state. Spare
        // KV regions above the KV side's recent high-water go back to the
        // weight zone as resident-expert slots — without this the boundary
        // only ever moves toward KV (`request_kv_ground` buys on the spot) and
        // expert residency ratchets down across a long run. The shrink
        // direction needs no call here: a KV claim that runs out buys its
        // ground itself. Mirrors the blanket `BatchedModel` wave's phase 0.
        self.engine.experts().reclaim_spare_ground();
        let e = &self.engine;
        let cfg = e.cfg();
        let n_layers = self.num_layers();
        let hc = e.hc();

        // Token ids (host) + per-seq geometry.
        let decode_ids: Vec<u32> = decode_inputs
            .iter()
            .map(|t| Ok(Self::token_ids(t)?[0]))
            .collect::<Result<Vec<_>>>()?;
        let prefill_ids: Vec<Vec<u32>> = prefill_inputs
            .iter()
            .map(Self::token_ids)
            .collect::<Result<Vec<_>>>()?;
        let prefill_lens: Vec<usize> = prefill_ids.iter().map(|v| v.len()).collect();
        let glue_ids: Vec<Vec<u32>> = glue_inputs
            .iter()
            .map(Self::token_ids)
            .collect::<Result<Vec<_>>>()?;
        let glue_lens: Vec<usize> = glue_ids.iter().map(|v| v.len()).collect();

        // Glue descriptors: staged on the session before this call, one per
        // glue sequence in order. Positions derive from the reserved gap
        // chunks: block logical start + (in-block offset − chunk offset).
        // DeepSeek glue is causal-only (locked §E) — fwd_ahead must be 0.
        let glue_desc = if glue_seqs.is_empty() {
            Vec::new()
        } else {
            let desc = session.take_pending_glue().ok_or_else(|| {
                candle::Error::Msg("glue rows without staged PendingGlue descriptors".into())
            })?;
            if desc.len() != glue_seqs.len() {
                candle::bail!(
                    "PendingGlue count {} != glue seq count {}",
                    desc.len(),
                    glue_seqs.len()
                );
            }
            desc
        };
        let mut glue_pos: Vec<Vec<u32>> = Vec::with_capacity(glue_seqs.len());
        for (gi, &seq) in glue_seqs.iter().enumerate() {
            let d = &glue_desc[gi];
            if d.fwd_ahead.iter().any(|&f| f != 0) {
                candle::bail!("deepseek glue is causal-only (fwd_ahead must be 0)");
            }
            let chunks = session.backings()[0]
                .live_chunks_as_sealed(seq)
                .unwrap_or_default();
            let mut block_start = Vec::with_capacity(chunks.len());
            let mut block_off = Vec::with_capacity(chunks.len());
            let mut cum = 0usize;
            for c in &chunks {
                block_start.push(cum);
                block_off.push(c.offset as usize);
                cum += c.token_count as usize;
            }
            let mut pos = Vec::with_capacity(d.write_slice.len());
            for (s_idx, &blk) in d.write_slice.iter().enumerate() {
                let b = blk as usize;
                if b >= block_start.len() {
                    candle::bail!("glue descriptor block {b} out of table range");
                }
                let within = d.write_in_blk[s_idx] as usize;
                pos.push((block_start[b] + within.saturating_sub(block_off[b])) as u32);
            }
            if pos.len() != glue_lens[gi] {
                candle::bail!(
                    "glue descriptor tokens {} != glue input tokens {}",
                    pos.len(),
                    glue_lens[gi]
                );
            }
            glue_pos.push(pos);
        }

        // Sequence offsets (position of the first new token per sequence).
        // A sequence may occupy SEVERAL decode rows (a speculative verify
        // block: one row per block position); the k-th occurrence sits at
        // `offset + k`, so consecutive rows of one seq are consecutive
        // positions — exactly the per-token decode stream, batched.
        let decode_pos: Vec<usize> = {
            let mut seen: HashMap<usize, usize> = HashMap::new();
            decode_seqs
                .iter()
                .map(|&s| {
                    let k = seen.entry(s).or_insert(0);
                    let p = session.sequence_offset(s).unwrap_or(0) + *k;
                    *k += 1;
                    p
                })
                .collect()
        };
        let prefill_base: Vec<usize> = prefill_seqs
            .iter()
            .map(|&s| session.sequence_offset(s).unwrap_or(0))
            .collect();

        // Slide each writer sequence's sliding-window RING before sizing the
        // arena for this wave: free the front chunks that have fully exited the
        // `window_size` window ending at the sequence's EARLIEST query position
        // in this wave. Decode has one query (`decode_pos`); a batched prefill
        // spans `[base, base+len)`, whose earliest row (`base`) has the
        // furthest-back window — evicting on `base` (the MIN, never the latest)
        // is what keeps a prefill row from losing a window key it still needs.
        // Glue rows scatter into reserved gap chunks and must not touch their
        // block tables, so they never evict. Positions stay ABSOLUTE across the
        // slide (the freed count folds into each backing's `base_pos`, seeding
        // the serialised rope), so attention is unchanged and stays consistent
        // with the absolute-positioned compressed corpus. All layers evict
        // identically (lockstep chunk layout), so the evicted count is uniform;
        // `resident = absolute − base_pos` is what the ARENA (set_len /
        // write_contiguous / decode metadata) is addressed by, while q_pos, the
        // fresh diagonal base, and the corpus stay absolute. `base_pos` is 0
        // until a sequence first exceeds `window_size`, so short sequences are
        // byte-identical to the never-evicted path.
        let window = cfg.window_size;
        let mut decode_base = vec![0u32; decode_seqs.len()];
        {
            // Evict once per SEQUENCE at its earliest row (the first
            // occurrence — occurrences are consecutive positions), exactly the
            // MIN-position rule the prefill span uses; later rows of the same
            // seq reuse the slid base.
            let mut base_of: HashMap<usize, u32> = HashMap::new();
            for (i, (&s, &pos)) in decode_seqs.iter().zip(&decode_pos).enumerate() {
                let bp = match base_of.get(&s) {
                    Some(&bp) => bp,
                    None => {
                        let mut bp = 0u32;
                        for backing in session.backings() {
                            bp = backing.evict_window_front(s, window, pos)?;
                        }
                        base_of.insert(s, bp);
                        bp
                    }
                };
                decode_base[i] = bp;
            }
        }
        let mut prefill_base_ev = vec![0u32; prefill_seqs.len()];
        for (pi, (&s, &base)) in prefill_seqs.iter().zip(&prefill_base).enumerate() {
            let mut bp = 0u32;
            for backing in session.backings() {
                bp = backing.evict_window_front(s, window, base)?;
            }
            prefill_base_ev[pi] = bp;
        }
        // Resident (arena) offsets = absolute − evicted-front. Equal to the
        // absolute values until the ring first slides.
        let decode_resident: Vec<usize> = decode_pos
            .iter()
            .zip(&decode_base)
            .map(|(&p, &b)| p - b as usize)
            .collect();
        let prefill_resident: Vec<usize> = prefill_base
            .iter()
            .zip(&prefill_base_ev)
            .map(|(&b, &e)| b - e as usize)
            .collect();

        // Per-seq corpus state (reset on slot reuse).
        for (&s, &base) in prefill_seqs.iter().zip(&prefill_base) {
            self.ensure_seq_state(s, base)?;
        }
        for &s in decode_seqs.iter().chain(glue_seqs) {
            self.ensure_seq_state(s, 1)?; // decode/glue never reset
        }

        // Residual: fresh waves embed all rows [decode… | prefill… | glue…]
        // flat and expand into the mHC stream; resumed waves reuse it.
        let total_rows: usize = decode_seqs.len()
            + prefill_lens.iter().sum::<usize>()
            + glue_lens.iter().sum::<usize>();
        let mut h = match residual_in {
            Some(r) => {
                // Persisted as [1, rows, hc·dim]; restore the mHC axes.
                r.reshape((1, total_rows, cfg.hc_mult, cfg.dim))?
            }
            None => {
                let mut flat_ids: Vec<u32> = decode_ids.clone();
                for ids in prefill_ids.iter().chain(&glue_ids) {
                    flat_ids.extend_from_slice(ids);
                }
                let rows = self.embed_rows(&flat_ids)?;
                hc.expand(&rows)?
            }
        };

        // Read once for the whole wave: the sequences (if any) whose DECODE
        // rows are speculative verify blocks (one row per block position, the
        // same seq on consecutive rows). Their attention runs the DECODE
        // kernel over per-position virtual slots with the rows pre-scattered,
        // the per-row capture stashes their projected compressor rows
        // (rollback replay source), and the head scores every decode row as
        // it always does.
        let verify_seqs: Vec<usize> = self
            .verify_all_rows
            .read()
            .map_err(|_| candle::Error::Msg("verify_all_rows lock poisoned".into()))?
            .clone();
        // A MIXED wave leads with plain decode rows (live slots, on-device
        // write-len commit, fused scatter) and ends with the verify blocks'
        // virtual rows (throwaway snapshot headers, pre-scattered) —
        // `verify_blocks` builds the rows in exactly that order, so a single
        // prefix bound (`n_plain_rows`) routes the kernel per row.
        let is_verify_wave = !verify_seqs.is_empty();
        let n_plain_rows = decode_seqs
            .iter()
            .position(|s| verify_seqs.contains(s))
            .unwrap_or(decode_seqs.len());
        if is_verify_wave
            && (!prefill_seqs.is_empty()
                || decode_seqs[n_plain_rows..]
                    .iter()
                    .any(|s| !verify_seqs.contains(s)))
        {
            candle::bail!(
                "verify wave: decode rows {:?} must be a plain prefix then a verify tail \
                 (verify set {:?})",
                decode_seqs,
                verify_seqs
            );
        }

        // Metadata: commit each layer's CPU chunk usage to the committed
        // prefix, then serialize slot headers for decode ++ prefill ++ glue
        // sequences (glue tables are already the scheduler-reserved state —
        // no usage commit). Prefill sequences get ONE HEADER SNAPSHOT PER
        // TOKEN: prompt token `t` absorbs through the SAME per-token decode
        // step the decode rows use, so its launch needs headers serialized at
        // offset `base + t`. This is the ONLY batched absorption that
        // bit-matches per-token decode: the decode kernel bf16-STAGES the
        // current token's PV (diagonal), while a settled-slot prefill would
        // read the diagonal as FP8 from the arena — a lossy difference that
        // heavy early-layer self-attention amplifies into CSA selection flips
        // (measured: garbage output). A single-launch batched prefill that
        // reproduces the decode diagonal needs a kernel-level bf16 diagonal
        // source (§L optimization note); until then, per-token is correct.
        // Snapshot `t` covers every layer; non-prefill slots serialize
        // identically in each snapshot, so snapshot 0 serves decode + glue.
        let all_seqs: Vec<usize> = decode_seqs
            .iter()
            .chain(prefill_seqs)
            .chain(glue_seqs)
            .copied()
            .collect();
        // Verify groups: (seq, resident-at-first-row, block_len), one per
        // sequence, from its CONSECUTIVE decode rows in the verify tail.
        // Empty on plain waves.
        let verify_groups: Vec<(usize, usize, usize)> = if is_verify_wave {
            let mut gs: Vec<(usize, usize, usize)> = Vec::new();
            for (i, &s) in decode_seqs.iter().enumerate().skip(n_plain_rows) {
                match gs.last_mut() {
                    Some(g) if g.0 == s => g.2 += 1,
                    _ => gs.push((s, decode_resident[i], 1)),
                }
            }
            gs
        } else {
            Vec::new()
        };
        // Per-group first-row offset into the wave's decode rows (GLOBAL row
        // index — the plain prefix precedes the verify tail).
        let verify_row_start: Vec<usize> = {
            let mut starts = Vec::with_capacity(verify_groups.len());
            let mut off = n_plain_rows;
            for &(_, _, s_len) in &verify_groups {
                starts.push(off);
                off += s_len;
            }
            starts
        };
        for backing in session.backings() {
            // Commit each sequence's prefix at its FIRST row's resident offset
            // — a verify block's later rows are the same seq at +k, and
            // committing those would advance the prefix over uncommitted
            // draft positions.
            let mut committed: HashSet<usize> = HashSet::new();
            for (&s, &resident) in decode_seqs.iter().zip(&decode_resident) {
                if committed.insert(s) {
                    backing.set_len(s, resident);
                }
            }
        }
        let s_meta = span("deepseek:wave_metadata");
        // The wave's pinned-arena guard. Everything staged from it stays valid
        // until this drops (line ~2426), and the arena resets then — freeing is
        // by lifetime, and nothing here syncs.
        //
        // ARENA BUDGET. Two draws, both bump-allocated:
        //   * the paged metadata below (slice + header snapshots), the original
        //     consumer;
        //   * the descriptor tables every batched kernel of a compression layer
        //     reads — the two compressor pools, the gallery-append scatter, the
        //     corpus gather, the BDP recall, the selector, the compressed-index
        //     expansion, the arena writeback and the prefill slot metadata —
        //     sized by `desc::wave_desc_bytes` for THIS wave's width.
        // The second is the addition this campaign makes, so it is computed and
        // checked against the arena rather than assumed: at 64 slots × 43 layers
        // it is a few hundred kilobytes against a 128 MB arena, but a future
        // width, ratio or prompt length should trip the assert rather than
        // silently start allocating overflow arenas mid-wave.
        let desc_bytes = desc::wave_desc_bytes(
            decode_seqs.len() + prefill_seqs.len(),
            self.num_layers(),
            self.engine
                .cfg()
                .compress_ratios
                .iter()
                .copied()
                .max()
                .unwrap_or(1),
            // The per-token draws are sized by the LONGEST slot in the wave —
            // a decode slot contributes one token, a prompt its whole length.
            prefill_lens
                .iter()
                .chain(glue_lens.iter())
                .copied()
                .max()
                .unwrap_or(1)
                .max(1),
        );
        debug_assert!(
            desc_bytes < desc::STAGER_ARENA_BYTES / 2,
            "wave descriptor tables draw {desc_bytes} B of the {} B stager arena — the paged \
             metadata shares it, so re-check the arena size",
            desc::STAGER_ARENA_BYTES
        );
        let generation = session.begin_stager_generation();
        // Glue-only sequences never decode-write (their latents scatter into
        // reserved gap chunks) — excluding them from the metadata build's
        // write-chunk ensure keeps their block tables untouched, so the
        // scheduler's turn-start block accounting stays exact.
        let non_writer: Vec<usize> = glue_seqs
            .iter()
            .copied()
            .filter(|s| !decode_seqs.contains(s) && !prefill_seqs.contains(s))
            .collect();
        // Prefill absorbs each prompt in ONE `paged_latent_prefill` launch per
        // layer (the per-token decode-step loop it replaces was the dominant
        // wall-clock cost — see the wave profile). So each prefill seq needs a
        // SINGLE header snapshot at its committed-prefix length `base` (the arena
        // walk covers `[0,base)`; the fresh bf16 latents cover `[base,base+s)`),
        // with its whole write-back range ensured up-front. Decode/glue seqs
        // serialize their live offsets in the same snapshot.
        // Metadata offsets are RESIDENT (arena-relative). Decode seqs must be
        // listed explicitly — `build_decode_metadata_at` otherwise falls back to
        // the session's ABSOLUTE offset, which overshoots the arena once the ring
        // has slid. Equal to absolute until the first eviction.
        let mut overrides: Vec<(usize, usize)> =
            Vec::with_capacity(prefill_seqs.len() + decode_seqs.len());
        // Verify sequences are skipped here: their override is the BLOCK
        // length (`resident + s_len`, pushed with the group commit below), and
        // `build_decode_metadata_at` takes the FIRST match per seq.
        for (&s, &resident) in decode_seqs.iter().zip(&decode_resident) {
            if !verify_seqs.contains(&s) {
                overrides.push((s, resident));
            }
        }
        for (pi, &s) in prefill_seqs.iter().enumerate() {
            let resident = prefill_resident[pi];
            // Allocate the WHOLE prompt's write chunks up front — this extends the
            // block table over `[resident, resident+s_len)` WITHOUT moving the
            // writer/committed prefix (`set_len` stays `resident`), so the header
            // snapshot built below covers every write chunk. The post-launch arena
            // writeback (`paged_latent_glue_scatter`) addresses those chunks
            // through that immutable snapshot, so they MUST be present before it is
            // taken (a chunk allocated after the snapshot is invisible to the
            // scatter → OOB). The header the attention reads is unchanged: extra
            // trailing chunks past `resident` are never read (window/pos-bounded).
            for backing in session.backings() {
                backing.ensure_for_offset(s, resident, prefill_lens[pi])?;
                backing.set_len(s, resident);
            }
            overrides.push((s, resident));
        }
        // DECODE rows always serialize their slot state through the LIVE
        // persistent gpu_chunks buffer (the cheap path Qwen/Llama use): a decode
        // row's write chunk is pre-ensured, so it never reallocs during the layer
        // loop, and the decode kernel commits its write-len on-device
        // (`commit_write_len=true`) so the buffer advances for the next step. This
        // skips the per-layer host snapshot COPY that dominated `wave_metadata`,
        // for EVERY wave — pure-decode AND mixed. Only rows that mutate the arena
        // mid-forward — PREFILL (absorbs across chunk boundaries in one launch)
        // and GLUE (gap-chunk scatter) — still snapshot, so their `slices_ptr`
        // survives the reallocation. Bit-exactness of the live vs snapshot decode
        // path across chunk boundaries is gated by
        // `decode_live_buffer_matches_snapshot_multistep`.
        // Speculative-verify: capacity-ensure every block's whole write range
        // BEFORE the standard header build — `ensure_for_offset` above uses
        // positional block math, which under-allocates on a slid ring (chunk
        // count exceeds positional blocks while the writer tail lacks free
        // slots), and a chunk allocated after the snapshot is invisible to the
        // writeback scatter (→ OOB).
        if is_verify_wave {
            for &(vseq, resident, s_len) in &verify_groups {
                for backing in session.backings() {
                    backing.ensure_for_batch_entries(&[(vseq, resident)], s_len)?;
                }
            }
        }
        // Speculative-verify rows run the DECODE kernel over one virtual slot
        // per block position — decode numerics by construction (fp PV,
        // read-time rope, decode softmax structure), vs the prefill kernel's
        // int8-PV envelope that flips narrow-margin argmaxes. All of a block's
        // virtual slots share IDENTICAL headers serialized with the write
        // length committed over the whole block (the rows are host-written
        // before each layer's launch); per-slot causal visibility comes from
        // the kernel's `key_pos <= q_pos` bound alone.
        //
        // Commit every block's write length BEFORE the (single) header build:
        // the virtual-slot headers must expose the block rows physically, and
        // each position map must cover [0, resident+s_len). `set_len`
        // deliberately never touches the serialized slot buffer (see its
        // DMA-race comment), so the cached slot state must be brought up to
        // date. Two arms, by whether the block FITS the current writer chunk:
        //
        // * Fits (the common case for a ≤block-size extension): the O(1)
        //   writer-slice PATCH — only that one slice's length changed.
        // * Crosses into a fresh chunk: full invalidation. The patch is
        //   NOT enough here even though `push_chunk` cleared the buffer at
        //   append time — an EARLIER wave's metadata build re-validated it
        //   at pre-`set_len` lengths, so the spanned block's earlier rows
        //   would read short through the stale predecessor slice (this
        //   exact failure was measured as an acceptance collapse to
        //   1.4 tok/step with a lossless-assert kill).
        //
        // (Each block's write range was capacity-ensured ABOVE, so the
        // writeback snapshot covers every chunk `set_len` fills here.)
        if is_verify_wave {
            for &(vseq, resident, s_len) in &verify_groups {
                for backing in session.backings() {
                    let room = backing.decode_writer_room(vseq).unwrap_or(0);
                    backing.set_len(vseq, resident + s_len);
                    if s_len <= room {
                        backing.refresh_decode_writer_slice(&[(vseq, 0)])?;
                    } else {
                        backing.invalidate_decode_slot(vseq);
                    }
                }
                overrides.push((vseq, resident + s_len));
            }
        }
        // ONE metadata build for the whole wave — plain decode rows serialize
        // through the LIVE slot buffer (cheap reuse pointer), verify rows get
        // immutable per-row snapshots of the freshly committed block state
        // (duplicate (seq, offset) rows share one snapshot copy), prefill/glue
        // as always. Header slot i pairs with decode row i in the single
        // launch below; prefill/glue headers follow the decode rows.
        let snapshot_seqs: Vec<usize> = prefill_seqs
            .iter()
            .chain(glue_seqs)
            .chain(verify_seqs.iter())
            .copied()
            .collect();
        let std_meta = {
            let (pm, headers, stride) = session.build_decode_metadata_at(
                0..session.num_layers(),
                &all_seqs,
                &generation,
                &overrides,
                &non_writer,
                &snapshot_seqs,
            )?;
            let headers = headers.ok_or_else(|| candle::Error::Msg("no decode metadata".into()))?;
            (pm, headers, stride)
        };
        let hdr_of = |layer: usize, seq_slot: usize| -> u64 {
            let (_, headers, stride) = &std_meta;
            headers.dev_ptr() + (layer as u64) * stride + (seq_slot as u64) * 24
        };
        // Per-group scatter header: a group's positions share IDENTICAL
        // headers (committed block write length), so its FIRST row's header
        // carries the chunk records the row scatter addresses through.
        let verify_scatter_hdr =
            |layer: usize, gi: usize| -> u64 { hdr_of(layer, verify_row_start[gi]) };
        s_meta.end();

        let mut state = self
            .seq_state
            .write()
            .map_err(|_| candle::Error::Msg("seq_state lock poisoned".into()))?;

        // Wave-invariant decode query positions: built ONCE here instead of rebuilding
        // the `[n_dec]` u32 upload inside every layer (the positions are fixed for the
        // whole wave). `None` when the wave carries no decode rows.
        let q_pos_dec_t = if decode_seqs.is_empty() {
            None
        } else {
            Some(Tensor::from_vec(
                decode_pos.iter().map(|&p| p as u32).collect::<Vec<u32>>(),
                decode_seqs.len(),
                h.device(),
            )?)
        };
        // Wave-invariant prefill query positions (`base..base+s_len` per prefill seq):
        // hoisted for the same reason — each is rebuilt in every layer otherwise.
        let prefill_q_pos: Vec<Tensor> = prefill_seqs
            .iter()
            .enumerate()
            .map(|(pi, _)| {
                let s_len = prefill_lens[pi];
                let base = prefill_base[pi];
                Tensor::from_vec(
                    (base as u32..(base + s_len) as u32).collect::<Vec<u32>>(),
                    s_len,
                    h.device(),
                )
            })
            .collect::<Result<_>>()?;
        // The MoE token-id list is identical across layers; assemble it once.
        let flat_ids: Vec<u32> = decode_ids
            .iter()
            .copied()
            .chain(prefill_ids.iter().chain(&glue_ids).flatten().copied())
            .collect();

        // DSpark target-feature capture (paper Eq. 2): when a drafter is attached and this wave runs
        // through the head (full stack), stash `head_reduce(h)` after each real target layer — the
        // per-layer hidden `fc`/`Wc` consumes. Keyed by layer so the head can order them by
        // `target_layers` and substitute the post-output-norm hidden for the `n_layers` sentinel.
        let capture_targets = layer_end == n_layers && !self.target_layers.is_empty();
        let mut target_reduced: Vec<(usize, Tensor)> = Vec::with_capacity(self.target_layers.len());

        for l in layer_start..layer_end {
            let layer = e.engine_layer(l);
            let a = &layer.attn;
            let st = &self.layer_static[l];
            let rope = e.rope_for(l);
            // Device handle for the sync-bracketed fine-span attribution below
            // (cheap Arc clone; only the sync arms compile in under `profile`).
            let dev = h.device().clone();

            // Attention sub-block.
            let s_attn_pre = span("deepseek:hc_pre_norm");
            let s_hcpre = span("attn:hc_pre");
            let (x, post, comb) = hc.pre(&h, &layer.hc_attn)?;
            s_hcpre.end();
            let s_anorm = span("attn:norm");
            let x = rms_norm(&x, &layer.attn_norm, cfg.norm_eps)?;
            s_anorm.end();
            s_attn_pre.end();

            // Phase A — glue scatter FIRST: every glue row's latent lands in
            // its reserved gap chunk before ANY attention pass of this layer
            // reads the arena (stream-ordered; glue keys then read like any
            // window key — no double-source, no garbage gaps).
            let glue_row_base = decode_seqs.len() + prefill_lens.iter().sum::<usize>();
            let mut glue_proj: Vec<(Tensor, Tensor)> = Vec::with_capacity(glue_seqs.len());
            let s_glue_scatter = span_if(!glue_seqs.is_empty(), "deepseek:glue_scatter");
            {
                let mut cursor = glue_row_base;
                // Every island's scatter collected first, then ONE launch: each
                // run names its own slot, so there is nothing to serialise.
                let mut runs: Vec<GlueRun> = Vec::with_capacity(glue_seqs.len());
                for (gi, _seq) in glue_seqs.iter().enumerate() {
                    let g_len = glue_lens[gi];
                    let xs = x.narrow(1, cursor, g_len)?.to_dtype(DType::F32)?;
                    let qr = rms_norm(&a.wq_a().forward(&xs)?, a.q_norm(), a.eps())?;
                    let kv = rms_norm(&a.wkv().forward(&xs)?, a.kv_norm(), a.eps())?;
                    let kv_bf = kv.reshape((g_len, a.head_dim()))?.to_dtype(DType::BF16)?;
                    let d = &glue_desc[gi];
                    runs.push(GlueRun {
                        kv: kv_bf,
                        headers_ptr: hdr_of(l, decode_seqs.len() + prefill_seqs.len() + gi),
                        slices: d.write_slice.clone(),
                        in_blk: d.write_in_blk.clone(),
                    });
                    glue_proj.push((xs, qr));
                    cursor += g_len;
                }
                super::paged::paged_latent_glue_scatter(&runs, &generation)?;
            }
            drop(s_glue_scatter);

            let mut attn_rows: Vec<Tensor> = Vec::with_capacity(total_rows);
            // Decode rows: one kernel step per sequence.
            // Decode rows: all sessions attend in ONE `paged_latent_decode`
            // launch over every decode slot (grid.x = n_decode), instead of a
            // per-session launch loop. Per session we run the same host
            // projections + on-device corpus select/gather, then concatenate the
            // gathered compressed blocks — each slot's selection is the dense
            // range `[offset, offset+k)` into the concat, so per-session galleries
            // stay isolated with no cross-bleed and no GID readback.
            if !decode_seqs.is_empty() {
                let s_decode = span("deepseek:decode_attn");
                let s_dprep = span("decode:prep");
                let n_dec = decode_seqs.len();
                let (h, hd) = (a.n_heads(), a.head_dim());

                // Batched attention projections: ONE GEMM each over ALL decode
                // rows (the decode rows are the first `n_dec` of `x`), replacing
                // the per-session projection GEMVs. Bit-identical per row (matmul
                // + last-dim norms are row-independent).
                let s_proj = span("dprep:proj");
                let xs_dec = x.narrow(1, 0, n_dec)?.to_dtype(DType::F32)?; // [1,n_dec,dim]
                                                                           // wq_a and wkv share `xs_dec`; quantize the activation once for both.
                let (qa_raw, kv_raw) = shared_int8_pair(&xs_dec, a.wq_a(), a.wkv())?;
                let qr_all = rms_norm(&qa_raw, a.q_norm(), a.eps())?; // [1,n_dec,qa]
                let q_all = a.wq_b().forward(&qr_all)?.reshape((1, n_dec, h, hd))?;
                let q_all = a.rms_scale(&q_all)?;
                let q_bf_all = q_all.reshape((n_dec, h, hd))?.to_dtype(DType::BF16)?; // [n_dec,h,hd]
                let kv_all = rms_norm(&kv_raw, a.kv_norm(), a.eps())?;
                let kv_bf_all = kv_all.reshape((n_dec, hd))?.to_dtype(DType::BF16)?; // [n_dec,hd]
                                                                                     // Batched compressor projections (shared layer weights) over all
                                                                                     // decode rows — the stateless part of the corpus push; each
                                                                                     // session's stateful pooling/emit then streams its pre-projected
                                                                                     // row (bit-identical to the per-session `push_raw`/`push`).
                let comp_proj = match a.compressor() {
                    Some(c) => Some(c.project_rows(&xs_dec)?), // (kv[n_dec,cd], score[n_dec,cd])
                    None => None,
                };
                let icomp_proj = match a.indexer() {
                    Some(ix) => Some(ix.compressor().project_rows(&xs_dec)?),
                    None => None,
                };
                // Batched indexer query projection AND rope (CSA layers have an
                // indexer): one GEMM over all decode rows for `wq_b` +
                // `weights_proj`, then ONE rope chain over all rows at their
                // per-row slot positions (`rope_query_at`) — the per-row
                // narrow/reshape/rope chain was ~5 launches × rows × CSA
                // layers of pure WDDM submission tax. Bit-identical per row to
                // `query_space`.
                let idx_query = match a.indexer() {
                    Some(ix) => {
                        let qr_2d = qr_all.reshape((n_dec, ()))?;
                        let xs_2d = xs_dec.reshape((n_dec, ()))?;
                        let (q_raw, weights) = ix.query_gemm_batched(&xs_2d, &qr_2d)?;
                        let pos32: Vec<u32> = decode_pos.iter().map(|&p| p as u32).collect();
                        let q_roped = ix.rope_query_at(&q_raw, rope, &pos32)?; // [n_dec,h,hd]
                        Some((ix, q_roped, weights))
                    }
                    None => None,
                };
                s_proj.end();

                // Pass 1a: per-session corpus ASSEMBLE — advance each session's
                // streaming compressor state with its pre-projected row and keep
                // the completed group's pool inputs, WITHOUT pooling. The pool
                // then runs once across the whole decode set in pass 1b.
                let mut asms: Vec<DecodeAssemble> = Vec::with_capacity(n_dec);
                // Verify waves: this layer's streaming state at every ROW
                // boundary (`Arc`-clone cheap), captured as the loop advances —
                // the rollback installs one of these directly on a partial
                // accept instead of replaying the accepted rows.
                let mut row_snaps: Vec<(
                    Option<super::compressor::CompressorState>,
                    Option<super::compressor::CompressorState>,
                    usize,
                )> = Vec::new();
                // Each row's corpus size AS OF ITSELF: this session's gallery
                // length entering the wave plus the groups its rows completed
                // THROUGH this one. It has to be counted here rather than read
                // back from the gallery, because pass 1b appends every row's
                // groups at once — so a `gallery.len()` taken afterwards would
                // hand an earlier row of a verify block the entries of a LATER
                // draft token. This keeps each row's view causal, exactly as the
                // per-row append it replaces did.
                let mut appended: HashMap<usize, usize> = HashMap::new();
                let mut row_entries: Vec<usize> = Vec::with_capacity(n_dec);
                let s_dasm = span("dprep:assemble_all");
                for (i, &seq) in decode_seqs.iter().enumerate() {
                    let comp_row = match &comp_proj {
                        Some((k, s)) => Some((k.narrow(0, i, 1)?, s.narrow(0, i, 1)?)),
                        None => None,
                    };
                    let icomp_row = match &icomp_proj {
                        Some((k, s)) => Some((k.narrow(0, i, 1)?, s.narrow(0, i, 1)?)),
                        None => None,
                    };
                    let entry = state.get_mut(&seq).expect("ensured above");
                    let asm = kernel_attn_decode_assemble(
                        &mut entry.layers[l],
                        comp_row.as_ref().map(|(k, s)| (k, s)),
                        icomp_row.as_ref().map(|(k, s)| (k, s)),
                    )?;
                    let n_new = asm.comp_gp.as_ref().map_or(0, |g| g.positions.len());
                    let done = appended.entry(seq).or_insert(0);
                    *done += n_new;
                    let ls = &entry.layers[l];
                    let n_entries = ls.gallery.as_ref().map_or(0, |g| g.len()) + *done;
                    row_entries.push(n_entries);
                    if is_verify_wave {
                        row_snaps.push((
                            ls.comp.as_ref().map(|c| c.state_snapshot()),
                            ls.icomp.as_ref().map(|c| c.state_snapshot()),
                            n_entries,
                        ));
                    }
                    asms.push(asm);
                    if l + 1 == n_layers {
                        entry.absorbed = decode_pos[i] + 1;
                    }
                }
                s_dasm.end();

                // Pass 1b: pool EVERY decode row's completed groups in ONE launch
                // per compressor family, then append. `comp` = attention entries
                // (pre-RoPE), `icomp` = indexer keys (roped); both share group
                // boundaries. The decode twin of the prefill pool batch, and
                // bit-identical per group to the per-row pool it replaces
                // (`pool_batched_across_seqs_matches_per_seq`).
                let s_dpool = span("dprep:pool");
                let comp_refs: Vec<Option<&GroupPool>> =
                    asms.iter().map(|m| m.comp_gp.as_ref()).collect();
                let icomp_refs: Vec<Option<&GroupPool>> =
                    asms.iter().map(|m| m.icomp_gp.as_ref()).collect();
                let comp_entries = match a.compressor() {
                    Some(c) => pool_across_seqs(c, &comp_refs, None, &generation)?,
                    None => None,
                };
                let icomp_keys = match a.indexer() {
                    Some(ix) => {
                        pool_across_seqs(ix.compressor(), &icomp_refs, Some(rope), &generation)?
                    }
                    None => None,
                };
                // Append every session's groups in ONE pass. The galleries are
                // pulled out of the state map together (they are distinct
                // sessions, so the borrows are disjoint) and handed over in slot
                // order.
                if let Some(entries) = comp_entries.as_ref() {
                    let mut by_seq: HashMap<usize, &mut FloatGallery> = state
                        .iter_mut()
                        .filter_map(|(s, e)| e.layers[l].gallery.as_mut().map(|g| (*s, g)))
                        .collect();
                    append_pooled(
                        entries,
                        icomp_keys.as_ref(),
                        &comp_refs,
                        decode_seqs,
                        &mut by_seq,
                        &generation,
                    )?;
                }
                s_dpool.end();

                // Pass 1c: capture each row's selection intent against its OWN
                // corpus size (the selection itself batches over all sessions
                // below).
                let mut sels: Vec<DecodeSel> = Vec::with_capacity(n_dec);
                for (i, &entries) in row_entries.iter().enumerate().take(n_dec) {
                    // This slot's row of the batched, already-roped indexer
                    // query — bare views into the batched tensors.
                    let (q_idx_i, w_i) = match &idx_query {
                        Some((ix, q_roped, weights)) => {
                            let qi = q_roped
                                .narrow(0, i, 1)?
                                .reshape((ix.n_heads(), ix.head_dim()))?;
                            let wi = weights.narrow(0, i, 1)?.reshape(ix.n_heads())?;
                            (Some(qi), Some(wi))
                        }
                        None => (None, None),
                    };
                    sels.push(kernel_attn_decode_capture(a, entries, q_idx_i, w_i)?);
                }
                // Distribute this layer's row-boundary states into each block's
                // pre-verify snapshot (rows are grouped consecutively, in the
                // same order as `verify_groups`).
                if is_verify_wave {
                    let mut vs = self
                        .verify_snap
                        .write()
                        .map_err(|_| candle::Error::Msg("verify_snap lock poisoned".into()))?;
                    for (gi, &(vseq, _, s_len)) in verify_groups.iter().enumerate() {
                        let row0 = verify_row_start[gi];
                        if let Some(snap) = vs.get_mut(&vseq) {
                            if let Some(lsnap) = snap.layers.get_mut(l) {
                                lsnap.row_states = row_snaps[row0..row0 + s_len].to_vec();
                            }
                        }
                    }
                }
                s_dprep.end();

                // Batched selection: ONE launch per Stage-1 kernel over EVERY CSA
                // decode session's gallery, replacing the per-session two-stage
                // selection loop (whose `topm_select` — a single-warp serial bin
                // scan × sessions — dominated the decode selection cost). HCA
                // sessions attend all causal entries; empty/SWA select nothing.
                let s_dsel = span("decode:select");
                let mut csa_idx: Vec<usize> = Vec::new();
                let mut csa_gals: Vec<&FloatGallery> = Vec::new();
                let mut csa_q: Vec<Tensor> = Vec::new();
                let mut csa_w: Vec<Tensor> = Vec::new();
                for (i, &seq) in decode_seqs.iter().enumerate() {
                    if let DecodeSel::TwoStage { q_idx, weights } = &sels[i] {
                        let g = state.get(&seq).expect("ensured").layers[l]
                            .gallery
                            .as_ref()
                            .expect("CSA session has a gallery");
                        csa_idx.push(i);
                        csa_gals.push(g);
                        csa_q.push(q_idx.clone());
                        csa_w.push(weights.clone());
                    }
                }
                let mut sel_gids: Vec<Option<Tensor>> = (0..n_dec).map(|_| None).collect();
                let mut cnts: Vec<u32> = vec![0u32; n_dec];
                if !csa_gals.is_empty() {
                    let ix = a.indexer().expect("CSA layer has an indexer");
                    let batched = two_stage_select_batched(
                        &csa_gals,
                        &csa_q,
                        &csa_w,
                        shortlist_m(ix.top_k()),
                        ix.top_k(),
                        &generation,
                    )?;
                    for (j, (gids, k)) in batched.into_iter().enumerate() {
                        let i = csa_idx[j];
                        cnts[i] = k as u32;
                        if k > 0 {
                            sel_gids[i] = Some(gids);
                        }
                    }
                }
                // ONE arange per layer, shared by every all-entries session as a
                // narrowed VIEW. Building it per session is an allocation plus a
                // launch each for what is the same ascending run every time —
                // with 16 sessions × 43 layers that is tens of thousands of
                // launches per run to materialise `0..n` repeatedly. The values
                // are identical, so the gather is bit-for-bit unchanged.
                let all_n: Vec<usize> = sels
                    .iter()
                    .filter_map(|s| match s {
                        DecodeSel::AllEntries(n) => Some(*n),
                        _ => None,
                    })
                    .collect();
                let all_ids = match all_n.iter().max() {
                    // `max(1)` keeps the allocation legal when every all-entries
                    // session is empty; the narrow below still yields 0 rows.
                    Some(&m) => Some(Tensor::arange(0u32, m.max(1) as u32, &dev)?),
                    None => None,
                };
                for (i, sel) in sels.iter().enumerate() {
                    if let DecodeSel::AllEntries(n) = *sel {
                        cnts[i] = n as u32;
                        sel_gids[i] = Some(
                            all_ids
                                .as_ref()
                                .expect("an all-entries session implies the shared arange")
                                .narrow(0, 0, n)?,
                        );
                    }
                }
                s_dsel.end();

                // Pass 2: gather EVERY session's selected HOT two-region rows into
                // one pre-allocated block in a SINGLE batched launch (each slot's
                // rows at its dense range `[offset, offset+k)`) — no per-region
                // `index_select`, no cross-session `cat`, no per-session launch.
                let s_dgather = span("decode:gather");
                let mut offsets: Vec<u32> = Vec::with_capacity(n_dec);
                let mut off = 0u32;
                for &c in cnts.iter().take(n_dec) {
                    offsets.push(off);
                    off += c;
                }
                let total_k = off as usize;
                let cache = if total_k == 0 {
                    st.empty_corpus_cache()?
                } else {
                    // Gather-kernel output block (total_k>0 here — the ==0 case
                    // took empty_corpus_cache above): every row written by the
                    // gather, so allocate uninit and skip the memset.
                    let out_nope = Tensor::empty((total_k, NOPE_DIM), DType::U8, &dev)?;
                    let out_scale = Tensor::empty((total_k, NOPE_BANDS), DType::F32, &dev)?;
                    let out_rope = Tensor::empty((total_k, ROPE_DIM), DType::BF16, &dev)?;
                    let out_pos = Tensor::empty(total_k, DType::U32, &dev)?;
                    let mut gg: Vec<&FloatGallery> = Vec::with_capacity(n_dec);
                    let mut ggids: Vec<Tensor> = Vec::with_capacity(n_dec);
                    let mut goff: Vec<u32> = Vec::with_capacity(n_dec);
                    for (i, &seq) in decode_seqs.iter().enumerate() {
                        if let Some(gids) = &sel_gids[i] {
                            let g = state.get(&seq).expect("ensured").layers[l]
                                .gallery
                                .as_ref()
                                .expect("selection implies a gallery");
                            gg.push(g);
                            ggids.push(gids.clone());
                            goff.push(offsets[i]);
                        }
                    }
                    gather_corpus_batched(
                        &gg,
                        &ggids,
                        &goff,
                        &out_nope,
                        &out_scale,
                        &out_rope,
                        &out_pos,
                        &generation,
                    )?;
                    super::paged::CorpusCache::from_gathered(
                        out_nope, out_scale, out_rope, out_pos, total_k,
                    )?
                };
                s_dgather.end();
                let s_dcache = span("decode:cache");
                // The batched projections already produced these contiguous
                // `[n_dec, …]` tensors; use them directly (the per-session
                // narrow-then-cat that rebuilt them was a redundant device copy).
                let q_all = q_bf_all; // [n_dec, H, hd]
                let kv_all = kv_bf_all; // [n_dec, hd]
                let max_sel = cnts.iter().map(|&c| c as usize).max().unwrap_or(0).max(1);
                // Each slot's selection is the dense range [offset, offset+k) —
                // strictly ascending, the compressed-index contract the kernel
                // documents — so the whole [n_dec, max_sel] matrix follows from
                // two numbers per slot. ONE launch expands them, reading the
                // table straight out of the wave's pinned arena; `comp_cnt`
                // falls out of the same pass because the decode kernel wants the
                // counts as a device array too.
                let (comp_idx, comp_cnt) =
                    super::comp_idx::build(&offsets, &cnts, max_sel, &dev, &generation)?;
                // Explicit per-slot query position (the decode kernel no longer
                // derives it from the writer slice, so the windowless slot works
                // and the compressed causal guard has a reference). Hoisted
                // above the layer loop — wave-fixed.
                let q_pos_dec = q_pos_dec_t
                    .as_ref()
                    .expect("decode branch runs only when the wave has decode rows");
                s_dcache.end();
                // `cache` is the gathered two-region hot cache (built above from
                // the gallery's pre-built int8 — no per-wave rebuild).
                let s_dkern = span("decode:kernel");
                // Speculative-verify tail rows are VIRTUAL decode slots: their
                // rows are written to the arena FIRST (same `store_band_elem`
                // encode the fused scatter uses → byte-identical read-back) so
                // later positions read earlier ones; causal visibility is the
                // kernel's `key_pos <= q_pos` bound; the shared per-group
                // headers carry the pre-committed block write length. Each
                // row's RESIDENT offset maps to its (slice, in_blk) by WALKING
                // the chunk table — positional `off/32` math shears once the
                // sliding ring has slid.
                if is_verify_wave {
                    let mut runs: Vec<GlueRun> = Vec::with_capacity(verify_groups.len());
                    for (gi, &(vseq, base_resident, s_len)) in verify_groups.iter().enumerate() {
                        let row0 = verify_row_start[gi];
                        let (wslice, wblk): (Vec<u32>, Vec<u32>) = {
                            let chunks = session.backings()[l]
                                .live_chunks_as_sealed(vseq)
                                .unwrap_or_default();
                            let mut map: Vec<(u32, u32)> = Vec::with_capacity(s_len);
                            let mut cum = 0usize;
                            for (si, c) in chunks.iter().enumerate() {
                                let cnt = c.token_count as usize;
                                for w in 0..cnt {
                                    let r = cum + w;
                                    if r >= base_resident && r < base_resident + s_len {
                                        map.push((si as u32, c.offset as u32 + w as u32));
                                    }
                                }
                                cum += cnt;
                            }
                            if map.len() != s_len {
                                candle::bail!(
                                    "verify writeback: chunk walk covered {} of {} block rows \
                                     (seq {vseq}, base_resident {base_resident}, {} chunks)",
                                    map.len(),
                                    s_len,
                                    chunks.len()
                                );
                            }
                            map.into_iter().unzip()
                        };
                        runs.push(GlueRun {
                            kv: kv_all.narrow(0, row0, s_len)?,
                            headers_ptr: verify_scatter_hdr(l, gi),
                            slices: wslice,
                            in_blk: wblk,
                        });
                    }
                    super::paged::paged_latent_glue_scatter(&runs, &generation)?;
                }
                // ONE launch over plain AND verify rows. The plain prefix
                // (`n_plain_rows`) uses live persistent slot buffers — the
                // kernel fused-scatters each of its tokens and commits its
                // write-len on-device for the next step; the verify tail's
                // throwaway snapshot headers do neither (rows pre-written
                // above, lengths patched host-side).
                let out = super::paged::paged_latent_decode_raw(
                    &q_all,
                    hdr_of(l, 0),
                    &kv_all,
                    &cache,
                    &comp_idx,
                    &comp_cnt,
                    q_pos_dec,
                    st.sinks(),
                    st.rope_tab(),
                    a.softmax_scale() as f32,
                    a.window_size(),
                    0,
                    n_plain_rows,
                    n_plain_rows,
                    st.ws(),
                    None,
                )?;
                s_dkern.end();
                // Batched output projection: ONE `output_proj` over all decode
                // rows (`b = n_dec`) instead of a per-session call. `output_proj`
                // is already batch-parametrized — its inner loop is over the 8
                // o_lora groups, not sessions — so this is bit-identical per row
                // (the group GEMMs are row-independent) and collapses `8·n_dec`
                // group-GEMM launches to 8.
                let s_doutp = span("decode:outproj");
                // Kernel output is token-major [n_dec, h, hd]; `output_proj` takes
                // [b, s, h, hd] (here b=n_dec, s=1).
                let o = out.reshape((n_dec, 1, h, hd))?; // kernel emits F32
                let proj = a.output_proj(&o, n_dec, 1)?; // [n_dec, 1, dim]
                                                         // One reshaped view instead of `n_dec` narrow slices — the rows are
                                                         // concatenated along axis 1 below exactly as the prefill/glue rows are,
                                                         // so `[1, n_dec, dim]` is bit-identical and skips the narrow round-trip.
                attn_rows.push(proj.reshape((1, n_dec, ()))?);
                s_doutp.end();
                s_decode.end();
            }
            // Prefill rows: each prompt is absorbed in ONE batched
            // `paged_latent_prefill` launch per layer (argmax-equal to per-token
            // decode absorption, validated by `wave_prefill_state_matches_decode_steps`),
            // which replaced the per-token launch loop that dominated the profile.
            let s_prefill = span_if(!prefill_seqs.is_empty(), "deepseek:prefill_attn");
            // ── Prefill projections: ONE batched projection over the WHOLE prompt
            // span (all sequences' rows), exactly as decode's `dprep` batches over
            // all decode rows — one `shared_int8_pair`/`wq_b`/`project_rows` instead
            // of one set per sequence. Row-independent ⇒ bit-identical.
            let s_pprep = span("prefill:prep");
            let prefill_total: usize = prefill_lens.iter().sum();
            let proj = if prefill_total == 0 {
                None
            } else {
                let xs_all = x.narrow(1, decode_seqs.len(), prefill_total)?;
                Some(super::kernel_attention::kernel_attn_prefill_project_batched(a, &xs_all)?)
            };
            // ── Prefill pass 1: the compressor ASSEMBLE (state advance +
            // deferred pool inputs), FLEET-BATCHED. Three phases: (1) per-seq
            // state snapshots (Arc clones, no device work); (2) ONE batched
            // assemble per compressor family — ~a dozen device ops for the
            // whole fleet where the per-seq loop issued ~15 PER SEQUENCE
            // (measured at width 20: 15.7s of a 29.8s prefill wall, every
            // enqueue stalling under WDDM submission pressure); (3) per-seq
            // state install + PrefillPrep assembly (host math + free views).
            let mut preps: Vec<PrefillPrep> = Vec::with_capacity(prefill_seqs.len());
            if !prefill_seqs.is_empty() {
                let p = proj.as_ref().expect("prefill rows imply a projection");
                let s_asm = span("pprep:assemble");
                // Phase 1 — collect. Compressor presence is LAYER-uniform
                // (every seq's layer state is built from the same layer kind),
                // so the fleet either all contribute or none do.
                let mut comp_ins: Vec<SeqAssemble> = Vec::with_capacity(prefill_seqs.len());
                let mut icomp_ins: Vec<SeqAssemble> = Vec::with_capacity(prefill_seqs.len());
                let mut l0s: Vec<usize> = Vec::with_capacity(prefill_seqs.len());
                let mut bases: Vec<usize> = Vec::with_capacity(prefill_seqs.len());
                let mut ratio_comp = 1usize;
                let mut off = 0usize;
                for (pi, &seq) in prefill_seqs.iter().enumerate() {
                    let s_len = prefill_lens[pi];
                    let ls = &state.get(&seq).expect("ensured above").layers[l];
                    if let (Some(c), Some((k, sc))) = (&ls.comp, &p.comp_proj) {
                        comp_ins.push(c.assemble_input(
                            &k.narrow(0, off, s_len)?,
                            &sc.narrow(0, off, s_len)?,
                        )?);
                        ratio_comp = c.ratio();
                    }
                    if let (Some(ic), Some((k, sc))) = (&ls.icomp, &p.icomp_proj) {
                        icomp_ins.push(ic.assemble_input(
                            &k.narrow(0, off, s_len)?,
                            &sc.narrow(0, off, s_len)?,
                        )?);
                    }
                    l0s.push(ls.comp.as_ref().map_or(0, |c| c.buffered_len()));
                    bases.push(ls.gallery.as_ref().map_or(0, |g| g.len()));
                    off += s_len;
                }
                // Phase 2 — fleet-wide assemble per compressor family.
                let comp_outs = if comp_ins.len() == prefill_seqs.len() {
                    let template = state.get(&prefill_seqs[0]).expect("ensured above").layers[l]
                        .comp
                        .as_ref()
                        .expect("phase 1 collected a comp input for every seq");
                    Some(assemble_groups_batched(template, &comp_ins)?)
                } else {
                    None
                };
                let icomp_outs = if icomp_ins.len() == prefill_seqs.len() {
                    let template = state.get(&prefill_seqs[0]).expect("ensured above").layers[l]
                        .icomp
                        .as_ref()
                        .expect("phase 1 collected an icomp input for every seq");
                    Some(assemble_groups_batched(template, &icomp_ins)?)
                } else {
                    None
                };
                // Phase 3 — install state + build each seq's PrefillPrep.
                let mut comp_outs = comp_outs.map(|v| v.into_iter());
                let mut icomp_outs = icomp_outs.map(|v| v.into_iter());
                let mut off = 0usize;
                for (pi, &seq) in prefill_seqs.iter().enumerate() {
                    let s_len = prefill_lens[pi];
                    let e = state.get_mut(&seq).expect("ensured above");
                    let ls = &mut e.layers[l];
                    let comp_gp = match comp_outs.as_mut().map(|it| it.next()) {
                        Some(Some((gp, upd))) => {
                            ls.comp
                                .as_mut()
                                .expect("comp outputs imply a compressor")
                                .assemble_apply(upd);
                            gp
                        }
                        _ => None,
                    };
                    let icomp_gp = match icomp_outs.as_mut().map(|it| it.next()) {
                        Some(Some((gp, upd))) => {
                            ls.icomp
                                .as_mut()
                                .expect("icomp outputs imply an indexer compressor")
                                .assemble_apply(upd);
                            gp
                        }
                        _ => None,
                    };
                    // The shared-boundary contract binds comp and icomp ONLY
                    // where both exist (CSA layers): an HCA layer has a
                    // compressor but no indexer, so its icomp side is always
                    // absent and only the comp pool exists.
                    debug_assert!(
                        ls.icomp.is_none()
                            || comp_gp.as_ref().map(|g| &g.positions)
                                == icomp_gp.as_ref().map(|g| &g.positions),
                        "comp/icomp group boundaries diverged (seq {seq}): {:?} vs {:?}",
                        comp_gp.as_ref().map(|g| &g.positions),
                        icomp_gp.as_ref().map(|g| &g.positions),
                    );
                    let g_total = comp_gp.as_ref().map_or(0, |g| g.positions.len());
                    // Each token sees the entries present before this prefill
                    // plus the groups completed through it; bounding the select
                    // to that prefix reproduces the per-token incremental
                    // gallery exactly.
                    let n_visible: Vec<usize> = (0..s_len)
                        .map(|t| bases[pi] + ((l0s[pi] + t + 1) / ratio_comp).min(g_total))
                        .collect();
                    preps.push(PrefillPrep {
                        kv_bf: p.kv_bf.narrow(0, off, s_len)?,
                        qr_all: p.qr_all.narrow(1, off, s_len)?,
                        xs: p.xs.narrow(1, off, s_len)?,
                        comp_gp,
                        icomp_gp,
                        n_visible,
                        base_entries: bases[pi],
                        g_total,
                    });
                    off += s_len;
                }
                s_asm.end();
            }
            s_pprep.end();

            // ── Prefill pool: pool every sequence's completed compressor groups
            // in ONE launch across the whole prompt fleet (`Compressor::pool_and_norm`)
            // instead of a per-seq pool — the prefill hot spot. `comp` = attention
            // entries (pre-RoPE), `icomp` = indexer keys (roped); both share group
            // boundaries. Bit-identical per group to the per-seq pool
            // (`pool_batched_across_seqs_matches_per_seq`).
            let s_ppool = span("prefill:pool");
            let comp_refs: Vec<Option<&super::compressor::GroupPool>> =
                preps.iter().map(|p| p.comp_gp.as_ref()).collect();
            let icomp_refs: Vec<Option<&super::compressor::GroupPool>> =
                preps.iter().map(|p| p.icomp_gp.as_ref()).collect();
            let comp_entries = match a.compressor() {
                Some(c) => pool_across_seqs(c, &comp_refs, None, &generation)?,
                None => None,
            };
            let icomp_keys = match a.indexer() {
                Some(ix) => {
                    pool_across_seqs(ix.compressor(), &icomp_refs, Some(rope), &generation)?
                }
                None => None,
            };
            s_ppool.end();

            // Append the whole fleet's groups BEFORE part A, in one pass. The
            // appends are independent across sequences and part A's selection
            // reads each gallery only after its own append, so hoisting them out
            // of the loop is the same order of effects with one launch set
            // instead of one per sequence.
            let s_pappend = span("ppush:append");
            if let Some(entries) = comp_entries.as_ref() {
                let mut by_seq: HashMap<usize, &mut FloatGallery> = state
                    .iter_mut()
                    .filter_map(|(s, e)| e.layers[l].gallery.as_mut().map(|g| (*s, g)))
                    .collect();
                append_pooled(
                    entries,
                    icomp_keys.as_ref(),
                    &comp_refs,
                    prefill_seqs,
                    &mut by_seq,
                    &generation,
                )?;
            }
            s_pappend.end();

            // ── Prefill pass 2 — FULLY BATCHED across sequences (invariant 5):
            // part A collects each seq's corpus gids + per-token (comp_idx,
            // comp_cnt); part B gathers the packed corpus, runs ONE multi-slot
            // attention kernel over the whole prompt fleet (each query indexes its
            // seq's arena slot + new-token diagonal slice via seq_of), and out-projects
            // in one GEMM; part C writes each seq's prompt latents back to its arena
            // (deferred — its set_len must not extend the committed prefix before
            // the batched kernel reads it). ──
            struct PfSeq {
                gids: Option<Tensor>, // packed-corpus entries to gather (None ⇒ empty)
                g: usize,             // this seq's corpus size
                comp_idx: Tensor,     // [s_len, max_sel_i] ids into the PACKED corpus
                comp_cnt: Tensor,     // [s_len]
            }
            let dev = x.device().clone();
            // Batched indexer query GEMM over the WHOLE prompt span — two
            // GEMMs per CSA layer-wave instead of two per SEQUENCE (the
            // per-seq form was 16 launches at width 8, each one a stall
            // point under submission pressure). Rows are seq-independent ⇒
            // bit-identical; each seq's slice ropes at its own base below.
            let idx_q_all = match (a.indexer(), proj.as_ref()) {
                (Some(ix), Some(p)) => Some(ix.query_gemm_batched(
                    &p.xs.reshape((prefill_total, ()))?,
                    &p.qr_all.reshape((prefill_total, ()))?,
                )?),
                _ => None,
            };
            let mut row_off = 0usize;
            let mut pf: Vec<PfSeq> = Vec::with_capacity(prefill_seqs.len());
            let mut seq_of_host: Vec<u32> = Vec::with_capacity(prefill_total);
            // Per-seq new-token diagonal metadata, flat {rows, base, start, -} × n_seq.
            let mut new_meta_host: Vec<u32> = Vec::with_capacity(prefill_seqs.len() * 4);
            let mut g_off = 0u32; // running packed-corpus row offset
            let mut new_off = 0u32; // running packed kv_new row offset

            // Part A: select + build (per seq); the appends ran above, and NO
            // gather/kernel yet.
            for (pi, &seq) in prefill_seqs.iter().enumerate() {
                let s_len = prefill_lens[pi];
                let base = prefill_base[pi];
                let prep = &preps[pi];
                let entry = state.get_mut(&seq).expect("ensured above");
                let q_rows = match &idx_q_all {
                    Some((q_raw, weights)) => Some((
                        q_raw.narrow(0, row_off, s_len)?,
                        weights.narrow(0, row_off, s_len)?,
                    )),
                    None => None,
                };
                row_off += s_len;
                let sel = super::kernel_attention::kernel_attn_prefill_select(
                    a,
                    entry.layers[l].gallery.as_ref(),
                    prep,
                    rope,
                    base,
                    q_rows,
                )?;
                let s_pgather = span("prefill:gather");
                // Build (gids, LOCAL comp_idx, comp_cnt, g) — same Device/Host
                // arms as before, but WITHOUT gathering (the gather is batched in
                // part B). Device: gather 0..n_corpus (absolute ids); Host: union
                // + remap. Both then get offset into the packed corpus below.
                let (gids, mut comp_idx, comp_cnt, g) = match sel {
                    PrefillSel::Device {
                        comp_idx,
                        comp_cnt,
                        n_corpus,
                    } => {
                        let gids = if n_corpus > 0 {
                            Some(Tensor::arange(0u32, n_corpus as u32, &dev)?)
                        } else {
                            None
                        };
                        (gids, comp_idx, comp_cnt, n_corpus)
                    }
                    PrefillSel::Host(idx_rows) => {
                        let mut union: Vec<u32> = idx_rows.iter().flatten().copied().collect();
                        union.sort_unstable();
                        union.dedup();
                        let remap: HashMap<u32, u32> = union
                            .iter()
                            .enumerate()
                            .map(|(i, &g)| (g, i as u32))
                            .collect();
                        let max_sel = idx_rows.iter().map(|v| v.len()).max().unwrap_or(0).max(1);
                        let mut idx_flat = vec![u32::MAX; s_len * max_sel];
                        let mut cnt_v = vec![0u32; s_len];
                        for (t, gids) in idx_rows.iter().enumerate() {
                            for (j, &g) in gids.iter().enumerate() {
                                idx_flat[t * max_sel + j] = remap[&g];
                            }
                            cnt_v[t] = gids.len() as u32;
                        }
                        let gids_t = if union.is_empty() {
                            None
                        } else {
                            Some(Tensor::from_vec(union.clone(), union.len(), &dev)?)
                        };
                        (
                            gids_t,
                            Tensor::from_vec(idx_flat, (s_len, max_sel), &dev)?,
                            Tensor::from_vec(cnt_v, s_len, &dev)?,
                            union.len(),
                        )
                    }
                };
                // Shift this seq's LOCAL ids into the PACKED corpus by g_off
                // (u32::MAX pads stay MAX — the kernel bounds by comp_cnt).
                if g_off > 0 && g > 0 {
                    let (r, c) = comp_idx.dims2()?;
                    let sentinel = Tensor::full(u32::MAX, (r, c), &dev)?;
                    let full_off = Tensor::full(g_off, (r, c), &dev)?;
                    let shifted = comp_idx.broadcast_add(&full_off)?;
                    comp_idx = comp_idx.lt(&sentinel)?.where_cond(&shifted, &comp_idx)?;
                }
                s_pgather.end();
                for _ in 0..s_len {
                    seq_of_host.push(pi as u32);
                }
                // {rows, base, start, -} for this seq.
                new_meta_host.extend_from_slice(&[s_len as u32, base as u32, new_off, 0]);
                new_off += s_len as u32;
                g_off += g as u32;
                pf.push(PfSeq {
                    gids,
                    g,
                    comp_idx,
                    comp_cnt,
                });
            }

            // Part B (only when the wave has prefill rows — decode-only waves skip
            // it): batched gather → packed corpus, ONE multi-slot kernel, one
            // out-proj; then part C writeback.
            if !prefill_seqs.is_empty() {
                let projref = proj.as_ref().expect("prefill rows imply a projection");
                let total_g = g_off as usize;
                let s_pkern = span("prefill:kernel");
                let cache = if total_g == 0 {
                    st.empty_corpus_cache()?
                } else {
                    let out_nope = Tensor::empty((total_g, NOPE_DIM), DType::U8, &dev)?;
                    let out_scale = Tensor::empty((total_g, NOPE_BANDS), DType::F32, &dev)?;
                    let out_rope = Tensor::empty((total_g, ROPE_DIM), DType::BF16, &dev)?;
                    let out_pos = Tensor::empty(total_g, DType::U32, &dev)?;
                    let mut gg: Vec<&FloatGallery> = Vec::new();
                    let mut ggids: Vec<Tensor> = Vec::new();
                    let mut goff: Vec<u32> = Vec::new();
                    let mut off = 0u32;
                    for (pi, p) in pf.iter().enumerate() {
                        if let Some(gids) = &p.gids {
                            gg.push(
                                state.get(&prefill_seqs[pi]).expect("ensured").layers[l]
                                    .gallery
                                    .as_ref()
                                    .expect("gids imply a gallery"),
                            );
                            ggids.push(gids.clone());
                            goff.push(off);
                            off += p.g as u32;
                        }
                    }
                    gather_corpus_batched(
                        &gg,
                        &ggids,
                        &goff,
                        &out_nope,
                        &out_scale,
                        &out_rope,
                        &out_pos,
                        &generation,
                    )?;
                    super::paged::CorpusCache::from_gathered(
                        out_nope, out_scale, out_rope, out_pos, total_g,
                    )?
                };
                // Pad each seq's comp_idx to the global max_sel and cat over tokens.
                let max_sel = pf
                    .iter()
                    .map(|p| p.comp_idx.dim(1))
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .max()
                    .unwrap_or(1);
                let mut idx_parts: Vec<Tensor> = Vec::with_capacity(pf.len());
                for p in &pf {
                    let (r, cur) = p.comp_idx.dims2()?;
                    if cur < max_sel {
                        let pad = Tensor::full(u32::MAX, (r, max_sel - cur), &dev)?;
                        idx_parts.push(Tensor::cat(&[&p.comp_idx, &pad], 1)?);
                    } else {
                        idx_parts.push(p.comp_idx.clone());
                    }
                }
                let comp_idx = Tensor::cat(&idx_parts.iter().collect::<Vec<_>>(), 0)?;
                let comp_cnt = Tensor::cat(&pf.iter().map(|p| &p.comp_cnt).collect::<Vec<_>>(), 0)?;
                // Per-query slot selector and per-seq diagonal metadata: read
                // once by the kernel and never by a tensor op, so they go
                // straight into the wave's pinned arena instead of costing a
                // device allocation plus a pageable upload each.
                let seq_of = desc::stage_slice(&seq_of_host, &generation)?;
                let new_meta = desc::stage_slice(&new_meta_host, &generation)?;
                let q_pos_all = Tensor::cat(&prefill_q_pos.iter().collect::<Vec<_>>(), 0)?;
                let out = super::paged::paged_latent_prefill_raw(
                    &projref.q_bf,
                    hdr_of(l, decode_seqs.len()),
                    &q_pos_all,
                    seq_of.ptr(),
                    &projref.kv_bf,
                    new_meta.ptr(),
                    &cache,
                    &comp_idx,
                    &comp_cnt,
                    st.sinks(),
                    st.rope_tab(),
                    a.softmax_scale() as f32,
                    a.window_size(),
                    0,
                    session.backings()[l].k_format().to_tag(),
                    st.ws(),
                )?;
                s_pkern.end();
                let s_poutp = span("prefill:outproj");
                let o_all = out.reshape((1, prefill_total, a.n_heads(), a.head_dim()))?;
                attn_rows.push(a.output_proj(&o_all, 1, prefill_total)?);
                s_poutp.end();

                // Part C: writeback each seq's prompt latents to its arena (deferred —
                // set_len must run AFTER the batched kernel read the committed prefix).
                let s_pwb = span("prefill:writeback");
                let mut runs: Vec<GlueRun> = Vec::with_capacity(prefill_seqs.len());
                for (pi, &seq) in prefill_seqs.iter().enumerate() {
                    let s_len = prefill_lens[pi];
                    let base = prefill_base[pi];
                    let base_resident = base - prefill_base_ev[pi] as usize;
                    let (wslice, wblk): (Vec<u32>, Vec<u32>) = (0..s_len)
                        .map(|t| {
                            let off = base_resident + t;
                            ((off / CHUNK_SIZE) as u32, (off % CHUNK_SIZE) as u32)
                        })
                        .unzip();
                    runs.push(GlueRun {
                        kv: preps[pi].kv_bf.clone(),
                        headers_ptr: hdr_of(l, decode_seqs.len() + pi),
                        slices: wslice,
                        in_blk: wblk,
                    });
                    session.backings()[l].set_len(seq, base_resident + s_len);
                    if l + 1 == n_layers {
                        state.get_mut(&seq).expect("ensured above").absorbed = base + s_len;
                    }
                }
                super::paged::paged_latent_glue_scatter(&runs, &generation)?;
                s_pwb.end();
            }
            drop(s_prefill);
            // Phase D — glue attention: each glue row attends its causal
            // window at its TRUE position, keys read from the arena (its own
            // island included — scattered in phase A). Compressed selection
            // and the compression-seam fold are step-7 scope; the corpus
            // state does not absorb glue tokens.
            let s_glue_attn = span_if(!glue_seqs.is_empty(), "deepseek:glue_attn");
            for (gi, _seq) in glue_seqs.iter().enumerate() {
                let g_len = glue_lens[gi];
                let (xs, qr) = &glue_proj[gi];
                let q = a
                    .wq_b()
                    .forward(qr)?
                    .reshape((1, g_len, a.n_heads(), a.head_dim()))?;
                let q = a.rms_scale(&q)?;
                let q_bf = q
                    .reshape((g_len, a.n_heads(), a.head_dim()))?
                    .to_dtype(DType::BF16)?;
                let dev = xs.device();
                let q_pos_t = Tensor::from_vec(glue_pos[gi].clone(), g_len, dev)?;
                let empty_idx = Tensor::full(u32::MAX, (g_len, 1), dev)?;
                let empty_cnt = Tensor::zeros(g_len, DType::U32, dev)?;
                // Glue rows read only their arena window — one slot, no new-token
                // diagonal (rows=0), no corpus.
                let seq_of = desc::stage_slice(&vec![0u32; g_len], &generation)?;
                let kv_dummy = Tensor::zeros((1, HEAD_DIM), DType::BF16, dev)?;
                let new_meta = desc::stage_slice(&[0u32, 0, 0, 0], &generation)?;
                let out = super::paged::paged_latent_prefill_raw(
                    &q_bf,
                    hdr_of(l, decode_seqs.len() + prefill_seqs.len() + gi),
                    &q_pos_t,
                    seq_of.ptr(),
                    &kv_dummy,
                    new_meta.ptr(),
                    &st.empty_corpus_cache()?,
                    &empty_idx,
                    &empty_cnt,
                    st.sinks(),
                    st.rope_tab(),
                    a.softmax_scale() as f32,
                    a.window_size(),
                    1,
                    session.backings()[l].k_format().to_tag(),
                    st.ws(),
                )?;
                // F32 token-major [g_len, h, hd] → [1, g_len, h, hd] = [b, s, h, hd].
                let o = out.reshape((1, g_len, a.n_heads(), a.head_dim()))?;
                attn_rows.push(a.output_proj(&o, 1, g_len)?);
            }
            drop(s_glue_attn);
            let s_hc_post = span("deepseek:hc_post_attn");
            let x = Tensor::cat(&attn_rows, 1)?; // [1, rows, dim]
            let h1 = hc.post(&x, &h, &post, &comb)?;
            s_hc_post.end();

            // MoE sub-block: one batched call — a single routing readback per
            // layer per wave, amortized over every row (the expert ids must be
            // host-visible to schedule the streaming cache's pinned→VRAM
            // uploads; it reaches zero only under full residency).
            let s_moe = span("deepseek:moe");
            let s_moe_hcpre = span("moe:hc_pre");
            let (x, post, comb) = hc.pre(&h1, &layer.hc_ffn)?;
            s_moe_hcpre.end();
            let moe = e.moe_forward_batch(layer, &x, &flat_ids)?;
            let s_moe_hcpost = span("moe:hc_post");
            h = hc.post(&moe, &h1, &post, &comb)?;
            s_moe_hcpost.end();
            s_moe.end();

            // Capture this layer's hidden for the drafter if it is a target layer. Per DeepSeek's
            // `inference/model.py` (`main_hiddens.append(h.mean(dim=2))`), the target feature is the
            // MEAN over the mHC copies of the RAW post-layer residual — not `head_reduce` (the final
            // output collapse) and not normed (the drafter's own `main_norm` does the only norm).
            if capture_targets && self.target_layers.contains(&l) {
                target_reduced.push((l, h.mean(2)?)); // [1, rows, hc, dim] → [1, rows, dim]
            }
        }
        drop(state);
        drop(generation);

        if layer_end < n_layers {
            // Pause: persist the mHC stream flattened to a plain 3-D hidden
            // shape (opaque to the scheduler).
            let flat = h.reshape((1, total_rows, cfg.hc_mult * cfg.dim))?;
            return Ok(WaveResult::owned(WaveStep {
                residual: Some(flat),
                logits: None,
            }));
        }

        // A batched prefill wrote its tokens via `write_contiguous` (not the
        // decode kernel's on-device write-len self-increment), so each prefill
        // seq's cached decode slot buffer — built at the pre-prefill base offset
        // during this wave's metadata build — carries a STALE writer-slice
        // length. The live-buffer decode path reuses that buffer, so the first
        // decode after a SHORT prefill (one that never crossed a chunk boundary —
        // which would itself have dropped the buffer) would read a stale window.
        // Re-serialize the writer slice to the prefilled length now (all layers
        // are absorbed in this final segment; O(1) per seq per layer, one-time
        // after each prefill — steady-state decode never pays it).
        for &pseq in prefill_seqs {
            session.refresh_decode_slot_state(pseq)?;
        }

        // Head: decode rows + each prefill sequence's LAST row.
        let s_head = span("deepseek:head_lm");
        let reduced = hc.head_reduce(&h, e.hc_head())?; // [1, rows, dim]
        let normed = rms_norm(&reduced, e.output_norm(), cfg.norm_eps)?;
        // Rows to score: every decode row (the [0,n_dec) prefix) + each prefill
        // sequence's LAST row. Gather them into one [1, R, dim] block and run
        // lm_head in a SINGLE batched GEMM (was R per-row GEMV launches);
        // bit-identical since each row's logits are independent.
        let hdev = normed.device().clone();
        // Every decode row gets a logits row — on a verify wave that is every
        // block position (the per-position next-token prediction the
        // speculative driver verifies), one row per position of each block.
        let mut sel_rows: Vec<u32> = (0..decode_seqs.len() as u32).collect();
        let mut scored_seqs: Vec<usize> = decode_seqs.to_vec();
        // Absolute token position of each scored row (for the position-keyed
        // target-feature stash). `decode_pos` is OCCURRENCE-indexed: a verify
        // block's k-th row sits at `offset + k`, so each block row's feature is
        // stashed under its own position — the accepted positions' features are
        // then present for the NEXT draft's window (a per-row `sequence_offset`
        // here collapsed every block row onto one key, starving the drafter
        // right after each multi-token accept).
        let mut scored_pos: Vec<usize> = decode_pos.clone();
        let mut cursor = decode_seqs.len();
        for (pi, &s_len) in prefill_lens.iter().enumerate() {
            let base = prefill_base[pi];
            if verify_seqs.contains(&prefill_seqs[pi]) {
                for r in 0..s_len {
                    sel_rows.push((cursor + r) as u32);
                    scored_seqs.push(prefill_seqs[pi]);
                    scored_pos.push(base + r);
                }
            } else {
                sel_rows.push((cursor + s_len - 1) as u32);
                scored_seqs.push(prefill_seqs[pi]);
                scored_pos.push(base + s_len - 1);
            }
            cursor += s_len;
        }
        let r_total = sel_rows.len();
        let idx = Tensor::from_vec(sel_rows, r_total, &hdev)?;
        let scored_hidden = normed.index_select(&idx, 1)?; // [1, R, dim]
        let logits_all = e.lm_head().forward(&scored_hidden)?; // [1,R,vocab]
        let mut logits_rows: Vec<Tensor> = Vec::with_capacity(r_total);
        for r in 0..r_total {
            logits_rows.push(logits_all.narrow(1, r, 1)?.reshape((1, cfg.vocab_size))?);
        }
        // Stash each scored row's target-layer feature — the drafter's conditioning source (`fc`
        // input, paper `Hctx = RMSNorm(Wc·[H^{l₁};…;H^{lₘ}])`), read by `speculative_draft` next
        // step. The per-layer captures (ordered by `target_layers`) are scored-row-selected and
        // concatenated along the feature axis → one `[m·dim]` vector per row, keyed by its absolute
        // position so the next draft picks the feature at `q_start-1`. Drafter only.
        if capture_targets {
            let per_layer: Vec<Tensor> = self
                .target_layers
                .iter()
                .map(|&tl| -> Result<Tensor> {
                    target_reduced
                        .iter()
                        .find(|(l, _)| *l == tl)
                        .ok_or_else(|| {
                            candle::Error::msg(format!("target layer {tl} not captured"))
                        })?
                        .1
                        .index_select(&idx, 1) // [1, R, dim]
                })
                .collect::<Result<_>>()?;
            let feats = Tensor::cat(&per_layer, 2)?; // [1, R, m·dim]
            let mdim = per_layer.len() * cfg.dim;
            let mut tf = self
                .target_feat
                .write()
                .map_err(|_| candle::Error::Msg("target_feat lock poisoned".into()))?;
            // ACCUMULATE a sliding window of the recent positions' features — the drafter attends to
            // the last `window_size` target hiddens (DSparkAttention's main_kv ring), not just the
            // current one. Rejected-draft positions are overwritten when re-decoded next step; we
            // keep `window + block_size` positions per seq to bound growth.
            let window = cfg.window_size;
            for (r, (&sq, &pos)) in scored_seqs.iter().zip(scored_pos.iter()).enumerate() {
                let feat = feats.narrow(1, r, 1)?.reshape((mdim,))?;
                tf.entry(sq).or_default().insert(pos, feat);
            }
            for &sq in scored_seqs.iter() {
                if let Some(m) = tf.get_mut(&sq) {
                    if let Some(&newest) = m.keys().max() {
                        let floor = newest.saturating_sub(window + 8);
                        m.retain(|&p, _| p >= floor);
                    }
                }
            }
        }
        s_head.end();
        Ok(WaveResult::owned(WaveStep {
            residual: None,
            logits: Some(logits_rows),
        }))
    }

    /// Batched speculative verify: run EVERY sequence's `[committed, drafts…]` block in ONE
    /// forward — each block's positions as virtual decode slots (decode numerics), all sequences
    /// in a single wave — and return each sequence's per-position next-token logits rows. The wave
    /// is launch-bound, so its fixed costs (per-layer MoE routing readbacks, expert DMA) amortize
    /// across every session instead of being paid once per session. Lossless is unchanged: the
    /// driver still accepts only the model's own argmaxes. `verify_all_rows` makes the head score
    /// every row of these sequences (see `forward_wave`); it is cleared before returning, even on
    /// error. Advances each sequence by its block length; the driver truncates back to the
    /// accepted lengths.
    fn verify_blocks(
        &self,
        session: &mut BatchedInferenceSession,
        plain: &[(usize, u32)],
        seqs: &[usize],
        blocks: &[Vec<u32>],
        layer_end: usize,
    ) -> Result<(Vec<Tensor>, Vec<Vec<Tensor>>)> {
        if plain.is_empty() && seqs.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        // Pre-verify corpus-state snapshots: the forward below absorbs the WHOLE
        // blocks (including drafts the driver may reject) into the streaming
        // compressors/galleries; the driver's `truncate_sequence` consumes each
        // snapshot to roll that state back to the accepted prefix — the KV
        // truncation alone cannot see it.
        let s_snap = span("verify:snapshot");
        // ONE MIXED WAVE: the plain cohort's committed tokens lead as ordinary
        // decode rows (live slots, on-device commit, fused scatter), the
        // verify blocks trail as virtual rows — each block position one decode
        // ROW of its sequence (the same seq repeated `block_len` times). The
        // wave's decode front-end does everything a plain step does for EVERY
        // row — batched projection, per-row streaming compressor capture
        // (exact per-token semantics), batched selection, one gather — and the
        // single kernel launch routes per row on the plain/verify boundary.
        // Splitting the cohorts into two waves paid a second launch floor
        // (WDDM's per-wave fixed cost) every step both were present.
        let n_rows: usize = plain.len() + blocks.iter().map(|b| b.len()).sum::<usize>();
        let mut row_seqs: Vec<usize> = Vec::with_capacity(n_rows);
        let mut row_inputs: Vec<Tensor> = Vec::with_capacity(n_rows);
        for &(seq, tok) in plain {
            row_seqs.push(seq);
            row_inputs.push(Tensor::from_vec(vec![tok], (1, 1), &Device::Cpu)?);
        }
        for (i, &seq) in seqs.iter().enumerate() {
            if blocks[i].is_empty() {
                candle::bail!("verify_blocks: empty block for seq {seq}");
            }
            let q_start = session.sequence_offset(seq).unwrap_or(0);
            self.snapshot_verify_state(seq, q_start, blocks[i].len())?;
            for &tok in &blocks[i] {
                row_seqs.push(seq);
                row_inputs.push(Tensor::from_vec(vec![tok], (1, 1), &Device::Cpu)?);
            }
        }
        s_snap.end();
        let s_fwd = span("verify:forward");
        *self
            .verify_all_rows
            .write()
            .map_err(|_| candle::Error::Msg("verify_all_rows lock poisoned".into()))? =
            seqs.to_vec();
        let step = self.forward_wave(
            session,
            &row_seqs,
            &row_inputs,
            &[],
            &[],
            &[],
            &[],
            0,
            layer_end,
            None,
        );
        self.verify_all_rows
            .write()
            .map_err(|_| candle::Error::Msg("verify_all_rows lock poisoned".into()))?
            .clear();
        let step = match step {
            Ok(s) => s,
            Err(e) => {
                // Failed verify: drop the snapshots so a later truncate cannot
                // replay from a half-populated one.
                if let Ok(mut m) = self.verify_snap.write() {
                    for &s in seqs {
                        m.remove(&s);
                    }
                }
                return Err(e);
            }
        };
        s_fwd.end();
        let s_post = span("verify:post");
        for &(seq, _) in plain {
            session.advance_sequence(seq, 1)?;
        }
        for (i, &seq) in seqs.iter().enumerate() {
            session.advance_sequence(seq, blocks[i].len())?;
        }
        // Copy the rows off the wave's span (`logits_owned`): the driver reads
        // them after this forward returns — argmax comparison, and on partial
        // accept a rollback + a NEXT forward — so span-lifetime views would
        // dangle by then.
        let logits = step.logits_owned()?;
        if logits.len() != n_rows {
            candle::bail!(
                "verify_blocks: expected {} scored rows, got {}",
                n_rows,
                logits.len()
            );
        }
        // Split the scored rows back per cohort/sequence (decode-row order:
        // plain prefix, then each block's rows).
        let plain_out = logits[..plain.len()].to_vec();
        let mut out = Vec::with_capacity(seqs.len());
        let mut off = plain.len();
        for b in blocks {
            out.push(logits[off..off + b.len()].to_vec());
            off += b.len();
        }
        s_post.end();
        Ok((plain_out, out))
    }

    /// DSpark speculative draft: propose a block of up to `max_len` tokens after `committed`,
    /// conditioned on `seq`'s stashed target feature (the concatenated `dflash.target_layers`
    /// hidden states — paper Eq. 2). Lossless — the caller verifies every proposal against the
    /// target — so the conditioning only affects acceptance, never output. Returns empty (⇒ plain
    /// decode) when no drafter is attached or the sequence has no stashed feature yet.
    fn speculative_draft(
        &self,
        session: &mut BatchedInferenceSession,
        seqs: &[usize],
        committed: &[u32],
        max_len: usize,
    ) -> Result<Vec<Vec<u32>>> {
        if committed.len() != seqs.len() {
            candle::bail!(
                "dspark draft: {} committed tokens for {} sequences",
                committed.len(),
                seqs.len()
            );
        }
        // Per sequence, because DSpark conditions each proposal on that
        // sequence's own stashed target feature and gates it on that sequence's
        // own acceptance EMA. The qwen35 NextN head batches its cohort instead
        // (`qwen35::mtp::MtpHead::draft_cohort`), which is what removes the
        // per-session weight read the width-aware break-even below is priced
        // against; doing the same here is a separate change to a drafter with
        // its own measured schedule, not a rename.
        let cohort = seqs.len();
        seqs.iter()
            .zip(committed)
            .map(|(&seq, &tok)| self.draft_one(session, seq, tok, max_len, cohort))
            .collect()
    }
}

impl BatchedEngine {
    /// One sequence's DSpark proposal — see
    /// [`ManagedBatchedModel::speculative_draft`].
    fn draft_one(
        &self,
        session: &mut BatchedInferenceSession,
        seq: usize,
        committed: u32,
        max_len: usize,
        cohort: usize,
    ) -> Result<Vec<u32>> {
        let drafter_lock = match &self.drafter {
            Some(d) => d,
            None => return Ok(Vec::new()),
        };
        // Rolling-acceptance fallback: while this sequence's accepted/step EMA
        // sits below the economic threshold, skip drafting — the step becomes a
        // plain-cost step, still lossless — and every `SPEC_PROBE_INTERVAL`
        // skipped steps run one drafted probe so a workload shift re-enables
        // speculation.
        //
        // The threshold is WIDTH-AWARE: a plain wave amortizes its launch
        // floor across every session in the cohort (a cfg-8 plain step costs
        // ~12 ms/token where a single-session step costs ~112), while a
        // drafted step pays one serial drafter forward per session plus a
        // wider verify. Measured break-even (release): ~2.1 accepts at width
        // 1, ~5.4 at width 8 — near-linear in width, so the gate adds the
        // measured slope per extra session. At width the cohort therefore
        // rides the batched plain wave unless the text is VERY predictable,
        // which is exactly the arithmetic that maximizes tokens/sec.
        let min_accept = SPEC_MIN_ACCEPT + 0.45 * cohort.saturating_sub(1) as f32;
        let accept_ema = {
            let mut stats = self
                .spec_stats
                .write()
                .map_err(|_| candle::Error::Msg("spec_stats lock poisoned".into()))?;
            match stats.get_mut(&seq) {
                Some(s) => {
                    if s.ema < min_accept {
                        if s.fallback_steps < SPEC_PROBE_INTERVAL {
                            s.fallback_steps += 1;
                            return Ok(Vec::new());
                        }
                        s.fallback_steps = 0; // probe this step
                    } else {
                        s.fallback_steps = 0;
                    }
                    s.ema
                }
                // Fresh sequence: optimistic seed (matches the SpecStats
                // insert) → full-width first draft.
                None => SPEC_MIN_ACCEPT + 1.0,
            }
        };

        // ── Graduated draft width K ──
        //
        // Between "full speculation" and the plain-decode fallback above sits
        // a continuum: every drafted token widens the verify wave's
        // routed-expert union, so at high expert-miss rates draft width is a
        // direct expert-DMA multiplier, and drafting far past the sequence's
        // measured acceptance only buys rejected rows that still pay that
        // cost. Two shaves, both measured:
        //
        // * Acceptance cap: the block never exceeds `ceil(EMA + 1)` — one
        //   token of headroom above the measured accepted/step, so a
        //   predictable stretch can still ratchet the EMA (and with it K)
        //   back up to full width, while a sagging sequence narrows its
        //   verify waves BEFORE the binary gate has to shut speculation off.
        // * Miss shave: the shared verify-wave expert-miss EMA takes off up
        //   to 2 more tokens as it climbs past 50% — speculating less under
        //   cache pressure trades speculative reach for expert-DMA certainty.
        //
        // Lossless either way — K shapes cost and acceptance ceiling, never
        // output. K ≥ 1 keeps probes meaningful; the τ confidence schedule
        // below can still shorten the block further on its own signal.
        let miss_ema = {
            let st = self.engine.experts().expert_stats();
            let mut g = self
                .spec_miss
                .write()
                .map_err(|_| candle::Error::Msg("spec_miss lock poisoned".into()))?;
            let (last_hits, last_misses, ema) = *g;
            let dh = st.expert_hits.saturating_sub(last_hits);
            let dm = st.expert_misses.saturating_sub(last_misses);
            let ema = if dh + dm > 0 {
                let rate = dm as f32 / (dh + dm) as f32;
                SPEC_EMA_ALPHA * rate + (1.0 - SPEC_EMA_ALPHA) * ema
            } else {
                ema
            };
            *g = (st.expert_hits, st.expert_misses, ema);
            ema
        };
        let k_accept = (accept_ema + 1.0).ceil() as usize;
        let miss_shave = ((miss_ema - 0.5).max(0.0) * 4.0).round() as usize;
        let draft_cap = k_accept.saturating_sub(miss_shave).clamp(1, max_len);
        // The block's first position (where `committed` will sit). The drafter conditions on a
        // sliding WINDOW of the last `window_size` target hiddens ending at `q_start-1` (the position
        // whose argmax produced `committed`) — matching DSparkAttention's main_kv ring. Gather the
        // longest CONSECUTIVE run of stashed positions ending at q_start-1 → `[W, m·dim]`, oldest
        // first, so the drafter ropes them at their true absolute positions (q_start-W .. q_start-1).
        let q_start = session.sequence_offset(seq).unwrap_or(0);
        if q_start == 0 {
            return Ok(Vec::new());
        }
        let window = self.engine.cfg().window_size;
        let feats: Vec<Tensor> = {
            let tf = self
                .target_feat
                .read()
                .map_err(|_| candle::Error::Msg("target_feat lock poisoned".into()))?;
            let map = match tf.get(&seq) {
                Some(m) => m,
                None => return Ok(Vec::new()),
            };
            let mut acc = Vec::new();
            let mut p = q_start - 1;
            while let Some(f) = map.get(&p) {
                acc.push(f.clone());
                if acc.len() >= window || p == 0 {
                    break;
                }
                p -= 1;
            }
            acc.reverse(); // oldest → newest (ascending absolute position)
            acc
        };
        if feats.is_empty() {
            return Ok(Vec::new());
        }
        let mut drafter = drafter_lock
            .lock()
            .map_err(|_| candle::Error::Msg("dspark drafter lock poisoned".into()))?;
        // `[W, m·dim]` window of per-target-layer hiddens → `Wc`'s input; the drafter starts the
        // context RoPE at `q_start - W` (the window is consecutive ending at q_start-1).
        let features = Tensor::stack(&feats, 0)?;
        // The block-diffusion mask token fills the not-yet-sampled block positions. τ is the
        // confidence-schedule threshold (paper Alg. 1): drafting stops at the longest prefix whose
        // cumulative survival probability ∏cᵢ clears it, so an unsure drafter proposes SHORT
        // blocks — fewer wasted verify rows (each rejected row still pays the block's expert-DMA
        // diversity in the verify wave). Confident runs (counting/structured code, cᵢ→1) still
        // draft full blocks. Both τ and the mask only affect acceptance/cost, never output
        // (lossless — the target's verify accepts only its own argmaxes).
        let mask_token = drafter.cfg.mask_token;
        const DSPARK_TAU: f32 = 0.3;
        let drafts = drafter.draft(
            &features,
            committed,
            mask_token,
            self.engine.embed(),
            self.engine.lm_head(),
            q_start,
            DSPARK_TAU,
        )?;
        Ok(drafts.into_iter().take(draft_cap).collect())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::needless_range_loop)]

    use super::*;
    // These are end-to-end gates against the real DeepSeek-V4-Flash checkpoint, so
    // they name the concrete architecture. The engine's model-independent tests
    // run against `arch::test_arch` instead.
    use crate::models::deepseek4::DEEPSEEK_V4;
    use crate::models::gpu_test_lock::gpu_serial;
    use candle::quantized::Int8Mode;
    use candle::IndexOp;

    // Full-model integration tests serialize on the CRATE-WIDE
    // [`gpu_test_lock::gpu_serial`], not a lock local to this module. Each loads
    // a multi-hundred-GB checkpoint with its multi-GB PINNED expert pool, so two
    // running at once exhaust host page-locked memory (`cuMemAllocHost` OOM) —
    // and the tests that must not overlap no longer all live in this file (the
    // batched-forwarding sweep is in `models::deepseek4`). A module-scoped lock
    // would serialize this file against itself and nothing else, which is the
    // failure mode `gpu_test_lock`'s own docs were written about.

    /// The wave path end to end: one batched prefill followed by decode waves
    /// through `ManagedBatchedModel::forward_wave`, greedy-decoded to EOS,
    /// must reproduce the continuation the per-token kernel engine produces for
    /// the same prompt. This is the semantic gate on the scheduler-facing trait
    /// surface — that prefill, the arena, selection, and the decode loop compose
    /// into a coherent generation rather than each merely passing in isolation.
    ///
    /// The prompt/expected pair is a fixture (a capital-city question), not the
    /// subject: what is under test is the wave loop. Model-quality gates live in
    /// `models::deepseek4`. Ignored (needs a checkpoint + CUDA).
    #[test]
    #[ignore]
    fn wave_prefill_then_decode_end_to_end() -> Result<()> {
        let _gpu = gpu_serial();
        let path = std::path::PathBuf::from(r"D:\models\deepseek-v4-flash-mxfp4")
            .join("DeepSeek-V4-Flash-0731-MXFP4_KO.gguf");
        if !path.exists() {
            eprintln!("[skip] merged file absent");
            return Ok(());
        }
        let device = Device::new_cuda(0)?;
        let engine = Engine::load(&path, &DEEPSEEK_V4, &device, Int8Mode::Performance)?;
        let model = BatchedEngine::new(engine)?;

        let tok_path = crate::models::batch_test::test_helpers::hf_get(
            "deepseek-ai/DeepSeek-V4-Flash-0731",
            hf_hub::RepoType::Model,
            "main",
            "tokenizer.json",
        )?;
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| candle::Error::msg(format!("tokenizer load: {e}")))?;
        let prompt = "<｜begin▁of▁sentence｜><｜User｜>What is the capital of France? \
             Reply with only the city name.<｜Assistant｜>";
        let ids: Vec<u32> = tokenizer
            .encode(prompt, false)
            .map_err(|e| candle::Error::msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();

        let mut session = model.create_batched_session(BatchedConfig::default())?;
        let seq = session.create_sequence()?;
        let n_layers = model.num_layers();
        super::super::readback::reset_readbacks();

        // Prefill the whole prompt in one wave.
        let prompt_t = Tensor::from_vec(ids.clone(), (1, ids.len()), &Device::Cpu)?;
        let t0 = std::time::Instant::now();
        let step = model.forward_wave(
            &mut session,
            &[],
            &[],
            &[seq],
            std::slice::from_ref(&prompt_t),
            &[],
            &[],
            0,
            n_layers,
            None,
        )?;
        eprintln!(
            "[wave] prefill {} tokens in {:.1}s",
            ids.len(),
            t0.elapsed().as_secs_f32()
        );
        session.advance_sequence(seq, ids.len())?;
        let logits = step.logits_owned()?;
        if logits.is_empty() {
            return Err(candle::Error::msg("prefill wave produced no logits"));
        }
        let mut next = logits[0].i(0)?.argmax(0)?.to_scalar::<u32>()?;

        // Greedy decode until EOS (bounded) through decode waves. STRICT gate:
        // the answer must be exactly "Paris" followed by EOS — the same stream
        // the per-token kernel engine produces; anything longer is wave-path
        // degradation, not a style choice.
        let eos = tokenizer
            .token_to_id("<｜end▁of▁sentence｜>")
            .expect("eos id");
        let mut gen = vec![next];
        let mut decode_waves = 0usize;
        let t1 = std::time::Instant::now();
        while gen.len() < 12 && next != eos {
            let tok = Tensor::from_vec(vec![next], (1, 1), &Device::Cpu)?;
            let step = model.forward_wave(
                &mut session,
                &[seq],
                std::slice::from_ref(&tok),
                &[],
                &[],
                &[],
                &[],
                0,
                n_layers,
                None,
            )?;
            session.advance_sequence(seq, 1)?;
            decode_waves += 1;
            let logits = step.logits_owned()?;
            if logits.is_empty() {
                return Err(candle::Error::msg("decode wave produced no logits"));
            }
            next = logits[0].i(0)?.argmax(0)?.to_scalar::<u32>()?;
            gen.push(next);
        }
        let dt = t1.elapsed().as_secs_f32();
        let text = tokenizer
            .decode(&gen, false)
            .map_err(|e| candle::Error::msg(format!("decode: {e}")))?;
        eprintln!("[wave] generated ids={gen:?}");
        eprintln!("[wave] continuation={text:?}");
        eprintln!(
            "[wave] {decode_waves} decode waves in {dt:.1}s = {:.2} tok/s",
            decode_waves as f32 / dt
        );
        assert_eq!(
            *gen.last().unwrap(),
            eos,
            "wave path did not stop on EOS within 12 tokens: {text:?}"
        );
        let answer = tokenizer
            .decode(&gen[..gen.len() - 1], false)
            .map_err(|e| candle::Error::msg(format!("decode: {e}")))?;
        assert_eq!(
            answer.trim(),
            "Paris",
            "wave path must answer exactly \"Paris\": {text:?}"
        );

        // Readback budget: the wave path's ONLY device→host transfers are the
        // per-layer MoE routing reads (one per wave per layer — intrinsic to
        // the streaming expert cache, amortized across the whole wave).
        // Sampling (`to_scalar` above) is the one-per-token the budget allows
        // and belongs to the caller.
        let expected = (1 + decode_waves) * n_layers;
        let got = super::super::readback::readback_count();
        assert_eq!(
            got, expected,
            "wave-path readbacks beyond the documented MoE-routing set: \
             {got} vs {expected}"
        );
        Ok(())
    }

    /// **Reproducibility gate.** Three independent greedy decodes of one prompt
    /// must produce identical token streams — greedy argmax from a fresh session
    /// is a pure function of the weights. DeepSeek-V4-Flash currently FAILS this
    /// (see `docs/deepseek_decode_reproducibility.md`); the assertion is here so
    /// the day it passes is recorded rather than guessed at.
    ///
    /// Uses the shared `decode_reproducibility` helper, the same one the Qwen3-MoE
    /// and Llama gates call, so the three models' verdicts are comparable: same
    /// harness, same pass count, same machine. That comparison is what localises
    /// the fault — Qwen3-MoE shares this model's `expert_lre` cache and Llama has
    /// no expert cache at all, and both reproduce.
    ///
    /// Token ids go in on the **host**: DeepSeek's embedding lookup is a CPU
    /// `index_select` (invariant 3's sanctioned transfer), unlike the other two
    /// models, which read ids on the device.
    #[test]
    #[ignore]
    fn wave_decode_is_reproducible() -> Result<()> {
        use crate::models::batch_test::utils::decode_reproducibility;

        let _gpu = gpu_serial();
        let path = std::path::PathBuf::from(r"D:\models\deepseek-v4-flash-mxfp4")
            .join("DeepSeek-V4-Flash-0731-MXFP4_KO.gguf");
        if !path.exists() {
            eprintln!("[skip] merged file absent");
            return Ok(());
        }
        let device = Device::new_cuda(0)?;
        let engine = Engine::load(&path, &DEEPSEEK_V4, &device, Int8Mode::Performance)?;
        let model = BatchedEngine::new(engine)?;

        let tok_path = crate::models::batch_test::test_helpers::hf_get(
            "deepseek-ai/DeepSeek-V4-Flash-0731",
            hf_hub::RepoType::Model,
            "main",
            "tokenizer.json",
        )?;
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| candle::Error::msg(format!("tokenizer load: {e}")))?;
        let prompt = "<｜begin▁of▁sentence｜><｜User｜>Write one sentence about the \
             sea.<｜Assistant｜>";
        let ids: Vec<u32> = tokenizer
            .encode(prompt, false)
            .map_err(|e| candle::Error::msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();

        let (agree, streams) =
            decode_reproducibility(&model, &Device::Cpu, &ids, 32, 3, "deepseek")?;
        for (i, s) in streams.iter().enumerate() {
            let text = tokenizer
                .decode(s, false)
                .map_err(|e| candle::Error::msg(format!("decode: {e}")))?;
            eprintln!("[repro:deepseek] pass{} = {text:?}", i + 1);
        }
        assert!(
            agree,
            "decode is not reproducible run-to-run; see docs/deepseek_decode_reproducibility.md"
        );
        Ok(())
    }

    /// **The localising probe for the reproducibility defect.**
    ///
    /// Comparing token streams (`wave_decode_is_reproducible`) proves the bug
    /// exists but says nothing about where it lives: argmax hides sub-ULP
    /// differences until one crosses a routing boundary, and each verdict costs
    /// a 50 s three-pass run. This replays **one decode step from bit-identical
    /// KV state** instead — prefill once, then repeatedly {decode one token,
    /// compare, roll the KV back} — so every repeat sees exactly the same
    /// inputs, the model is loaded once, and each sample costs one forward.
    ///
    /// Two axes:
    ///  * **repeat** — full-depth logits over `REPEATS` replays. Bitwise, so a
    ///    single differing ULP is caught where argmax would swallow it.
    ///  * **depth** — the same replay truncated to `layer_end = d` (which
    ///    returns the mHC residual rather than logits), swept over `d`. The
    ///    first `d` that differs names the layer that introduces the fault.
    ///
    /// Reports the number of differing elements and max|Δ| at each depth, which
    /// separates the two failure shapes: a *reassociated float sum* differs in
    /// the last few ULPs, whereas a *stale/racing read* differs grossly.
    #[test]
    #[ignore]
    fn wave_decode_step_bitwise_probe() -> Result<()> {
        const REPEATS: usize = 24;

        use crate::models::batch_test::utils::{decode_replay_probe, prefill_replay_probe};

        let _gpu = gpu_serial();
        let path = std::path::PathBuf::from(r"D:\models\deepseek-v4-flash-mxfp4")
            .join("DeepSeek-V4-Flash-0731-MXFP4_KO.gguf");
        if !path.exists() {
            eprintln!("[skip] merged file absent");
            return Ok(());
        }
        let device = Device::new_cuda(0)?;
        let engine = Engine::load(&path, &DEEPSEEK_V4, &device, Int8Mode::Performance)?;
        let model = BatchedEngine::new(engine)?;

        let tok_path = crate::models::batch_test::test_helpers::hf_get(
            "deepseek-ai/DeepSeek-V4-Flash-0731",
            hf_hub::RepoType::Model,
            "main",
            "tokenizer.json",
        )?;
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| candle::Error::msg(format!("tokenizer load: {e}")))?;
        let prompt = "<｜begin▁of▁sentence｜><｜User｜>Write one sentence about the \
             sea.<｜Assistant｜>";
        let ids: Vec<u32> = tokenizer
            .encode(prompt, false)
            .map_err(|e| candle::Error::msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();

        // Token ids go in on the HOST: DeepSeek's embedding lookup is a CPU
        // `index_select`, unlike the other models, which read ids on the device.
        // Prefill and decode are gated SEPARATELY. End-to-end token streams
        // cannot attribute a divergence to one or the other, so a fix to either
        // gets judged by a test the other also fails — which is exactly how the
        // decode fix came to look ineffective.
        let dirty_prefill = prefill_replay_probe(&model, &Device::Cpu, &ids, 6, "deepseek")?;
        let dirty_repeats = decode_replay_probe(&model, &Device::Cpu, &ids, REPEATS, "deepseek")?;
        assert_eq!(
            dirty_prefill, 0,
            "prefill is not bitwise repeatable: {dirty_prefill}/5 re-prefills of the same \
             prompt diverged; see docs/deepseek_decode_reproducibility.md"
        );

        assert_eq!(
            dirty_repeats,
            0,
            "a replayed decode step from identical KV state is not bitwise repeatable: \
             {dirty_repeats}/{} replays diverged; \
             see docs/deepseek_decode_reproducibility.md",
            REPEATS - 1
        );
        Ok(())
    }

    /// Speculative decode is **lossless**: the same prompt driven through
    /// `ManagedBatchedModel::speculative_decode_step` (draft → verify → accept
    /// the longest matching prefix → roll back the rest) must yield a token
    /// stream **bit-identical** to plain greedy
    /// [`wave_prefill_then_decode_end_to_end`]. That is true by construction —
    /// accepted tokens are the target's own argmaxes — so any difference is a
    /// bug in the driver, `verify_block`, or the `truncate_sequence_to_tokens`
    /// rollback, which is exactly what this pins down.
    ///
    /// Runs with whatever drafter the model supplies: none ⇒ the driver
    /// degenerates to plain decode (still must match); a real one ⇒ same
    /// stream, fewer waves. `--ignored` (needs a checkpoint).
    #[test]
    #[ignore]
    fn wave_speculative_step_is_lossless() -> Result<()> {
        let _gpu = gpu_serial();
        let path = std::path::PathBuf::from(r"D:\models\deepseek-v4-flash-mxfp4")
            .join("DeepSeek-V4-Flash-0731-MXFP4_KO.gguf");
        if !path.exists() {
            eprintln!("[skip] merged file absent");
            return Ok(());
        }
        let device = Device::new_cuda(0)?;
        let engine = Engine::load(&path, &DEEPSEEK_V4, &device, Int8Mode::Performance)?;
        let model = BatchedEngine::new(engine)?;

        let tok_path = crate::models::batch_test::test_helpers::hf_get(
            "deepseek-ai/DeepSeek-V4-Flash-0731",
            hf_hub::RepoType::Model,
            "main",
            "tokenizer.json",
        )?;
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| candle::Error::msg(format!("tokenizer load: {e}")))?;
        let prompt = "<｜begin▁of▁sentence｜><｜User｜>What is the capital of France? \
             Reply with only the city name.<｜Assistant｜>";
        let ids: Vec<u32> = tokenizer
            .encode(prompt, false)
            .map_err(|e| candle::Error::msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();
        let eos = tokenizer
            .token_to_id("<｜end▁of▁sentence｜>")
            .expect("eos id");

        let mut session = model.create_batched_session(BatchedConfig::default())?;
        let seq = session.create_sequence()?;
        let n_layers = model.num_layers();

        // Prefill the prompt; the first generated token is the argmax of the last
        // prefill row — held OUT of the KV as the driver's `committed` seed.
        let prompt_t = Tensor::from_vec(ids.clone(), (1, ids.len()), &Device::Cpu)?;
        let step = model.forward_wave(
            &mut session,
            &[],
            &[],
            &[seq],
            std::slice::from_ref(&prompt_t),
            &[],
            &[],
            0,
            n_layers,
            None,
        )?;
        session.advance_sequence(seq, ids.len())?;
        let logits = step.logits_owned()?;
        if logits.is_empty() {
            return Err(candle::Error::msg("prefill produced no logits"));
        }
        let mut committed = logits[0].i(0)?.argmax(0)?.to_scalar::<u32>()?;

        // Speculative decode loop: each step commits ≥1 token (the model's exact
        // greedy continuation), draft length up to 4.
        let mut gen = vec![committed];
        let mut steps = 0usize;
        let t0 = std::time::Instant::now();
        while gen.len() < 12 && committed != eos {
            steps += 1;
            let next = model.speculative_decode_step(
                &mut session,
                seq,
                committed,
                4,
                n_layers,
                &mut |t| {
                    gen.push(t);
                    gen.len() < 12 && t != eos
                },
            )?;
            match next {
                Some(c) => committed = c,
                None => break,
            }
        }
        let dt = t0.elapsed().as_secs_f32();
        let text = tokenizer
            .decode(&gen, false)
            .map_err(|e| candle::Error::msg(format!("decode: {e}")))?;
        eprintln!("[spec] ids={gen:?}");
        eprintln!("[spec] continuation={text:?}");
        eprintln!(
            "[spec] {steps} spec steps, {} tokens in {dt:.1}s = {:.2} tok/s",
            gen.len(),
            gen.len() as f32 / dt
        );
        assert_eq!(
            *gen.last().unwrap(),
            eos,
            "speculative path did not stop on EOS within 12 tokens: {text:?}"
        );
        let answer = tokenizer
            .decode(&gen[..gen.len() - 1], false)
            .map_err(|e| candle::Error::msg(format!("decode: {e}")))?;
        assert_eq!(
            answer.trim(),
            "Paris",
            "speculative path must answer exactly \"Paris\" (lossless vs greedy): {text:?}"
        );
        Ok(())
    }

    /// [`wave_speculative_step_is_lossless`], now with a REAL drafter attached
    /// rather than the degenerate no-drafter path — so `speculative_draft`
    /// proposes an actual block each step and the whole integration is live:
    /// the drafter's streaming MoE, its injected-context backbone, the Markov
    /// sampler, and accept/rollback. The stream must still be bit-identical to
    /// plain greedy decode; a drafter can only change how many waves it takes to
    /// get there, never what comes out.
    ///
    /// Also reports mean committed tokens/step. With the sequential
    /// `verify_block` that is not yet a wall-clock win (verify runs one wave per
    /// block token), but tokens/step > 1 is the ACCEPTANCE signal a batched
    /// verify converts into speedup on this launch-bound decode — measured by
    /// [`wave_speculative_speedup_with_drafter`]. Skips when the drafter
    /// checkpoint is absent (`zend --download-deepseek`). `--ignored`.
    #[test]
    #[ignore]
    fn wave_speculative_step_with_drafter_is_lossless() -> Result<()> {
        let _gpu = gpu_serial();
        let path = std::path::PathBuf::from(r"D:\models\deepseek-v4-flash-mxfp4")
            .join("DeepSeek-V4-Flash-0731-MXFP4_KO.gguf");
        let dspark = std::path::PathBuf::from(r"D:\models\deepseek-v4-flash-mxfp4")
            .join("dspark-DeepSeek-V4-Flash-0731-MXFP4.gguf");
        if !path.exists() || !dspark.exists() {
            eprintln!("[skip] target or DSpark drafter absent");
            return Ok(());
        }
        let device = Device::new_cuda(0)?;
        // Combined load: the drafter lands in the DENSE tier (before the span
        // reservation), so the elastic boundary balances target-experts vs KV
        // around it — see `Engine::load_with_drafter`.
        let (engine, drafter) = Engine::load_with_drafter(
            &path,
            Some(&dspark),
            &DEEPSEEK_V4,
            &device,
            Int8Mode::Performance,
        )?;
        let model =
            BatchedEngine::new(engine)?.with_drafter(drafter.expect("dspark path given"))?;

        let tok_path = crate::models::batch_test::test_helpers::hf_get(
            "deepseek-ai/DeepSeek-V4-Flash-0731",
            hf_hub::RepoType::Model,
            "main",
            "tokenizer.json",
        )?;
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| candle::Error::msg(format!("tokenizer load: {e}")))?;
        let prompt = "<｜begin▁of▁sentence｜><｜User｜>What is the capital of France? \
             Reply with only the city name.<｜Assistant｜>";
        let ids: Vec<u32> = tokenizer
            .encode(prompt, false)
            .map_err(|e| candle::Error::msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();
        let eos = tokenizer
            .token_to_id("<｜end▁of▁sentence｜>")
            .expect("eos id");

        let mut session = model.create_batched_session(BatchedConfig::default())?;
        let seq = session.create_sequence()?;
        let n_layers = model.num_layers();

        let prompt_t = Tensor::from_vec(ids.clone(), (1, ids.len()), &Device::Cpu)?;
        let step = model.forward_wave(
            &mut session,
            &[],
            &[],
            &[seq],
            std::slice::from_ref(&prompt_t),
            &[],
            &[],
            0,
            n_layers,
            None,
        )?;
        session.advance_sequence(seq, ids.len())?;
        let logits = step.logits_owned()?;
        if logits.is_empty() {
            return Err(candle::Error::msg("prefill produced no logits"));
        }
        let mut committed = logits[0].i(0)?.argmax(0)?.to_scalar::<u32>()?;

        let max_draft = 4usize;
        let mut gen = vec![committed];
        let mut steps = 0usize;
        let t0 = std::time::Instant::now();
        while gen.len() < 12 && committed != eos {
            steps += 1;
            // The driver emits each accepted token; we keep it and stop at the first EOS — the exact
            // per-token loop plain decode uses (the drafter drafts a whole block that can run PAST
            // EOS; those post-EOS argmaxes are simply never generated). No special block handling.
            let next = model.speculative_decode_step(
                &mut session,
                seq,
                committed,
                max_draft,
                n_layers,
                &mut |t| {
                    gen.push(t);
                    gen.len() < 12 && t != eos
                },
            )?;
            match next {
                Some(c) => committed = c,
                None => break,
            }
        }
        let dt = t0.elapsed().as_secs_f32();
        let text = tokenizer
            .decode(&gen, false)
            .map_err(|e| candle::Error::msg(format!("decode: {e}")))?;
        let per_step = gen.len() as f32 / steps.max(1) as f32;
        eprintln!("[spec-dspark] ids={gen:?}");
        eprintln!("[spec-dspark] continuation={text:?}");
        eprintln!(
            "[spec-dspark] {steps} spec steps for {} tokens ⇒ {per_step:.2} tokens/step \
             (max_draft={max_draft}); {dt:.1}s",
            gen.len(),
        );
        assert_eq!(
            *gen.last().unwrap(),
            eos,
            "speculative+DSpark did not stop on EOS within 12 tokens: {text:?}"
        );
        let answer = tokenizer
            .decode(&gen[..gen.len() - 1], false)
            .map_err(|e| candle::Error::msg(format!("decode: {e}")))?;
        assert_eq!(
            answer.trim(),
            "Paris",
            "speculative+DSpark must answer exactly \"Paris\" (lossless vs greedy): {text:?}"
        );
        Ok(())
    }

    /// The SPEEDUP measurement: a longer, drafter-friendly generation decoded two ways from the same
    /// prefill — (1) plain greedy (one wave per token), (2) speculative with the DSpark drafter +
    /// **batched** `verify_block` (draft a block, verify it in ONE wave, accept the matching prefix).
    /// Asserts the two token streams are **identical** (speculative is lossless) and reports the
    /// wall-clock speedup + mean accepted tokens/step. Because DeepSeek decode is launch-bound, a
    /// batched K-token verify costs ~one decode, so accepted drafts translate almost directly into
    /// throughput. `--ignored` (needs the 156 GB target + drafter; the KV VRAM reserve is
    /// installed from the engine's measured post-resident budget, so no tuning is needed).
    #[test]
    #[ignore]
    fn wave_speculative_speedup_with_drafter() -> Result<()> {
        let _gpu = gpu_serial();
        let path = std::path::PathBuf::from(r"D:\models\deepseek-v4-flash-mxfp4")
            .join("DeepSeek-V4-Flash-0731-MXFP4_KO.gguf");
        let dspark = std::path::PathBuf::from(r"D:\models\deepseek-v4-flash-mxfp4")
            .join("dspark-DeepSeek-V4-Flash-0731-MXFP4.gguf");
        if !path.exists() || !dspark.exists() {
            eprintln!("[skip] target or DSpark drafter absent");
            return Ok(());
        }
        let device = Device::new_cuda(0)?;
        let (engine, drafter) = Engine::load_with_drafter(
            &path,
            Some(&dspark),
            &DEEPSEEK_V4,
            &device,
            Int8Mode::Performance,
        )?;
        let model =
            BatchedEngine::new(engine)?.with_drafter(drafter.expect("dspark path given"))?;

        let tok_path = crate::models::batch_test::test_helpers::hf_get(
            "deepseek-ai/DeepSeek-V4-Flash-0731",
            hf_hub::RepoType::Model,
            "main",
            "tokenizer.json",
        )?;
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| candle::Error::msg(format!("tokenizer load: {e}")))?;
        // A deterministic, structured continuation the drafter predicts well (high acceptance).
        let prompt = "<｜begin▁of▁sentence｜><｜User｜>Count from 1 to 30, separated by \
             commas.<｜Assistant｜>";
        let ids: Vec<u32> = tokenizer
            .encode(prompt, false)
            .map_err(|e| candle::Error::msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();
        let eos = tokenizer
            .token_to_id("<｜end▁of▁sentence｜>")
            .expect("eos id");
        let n_layers = model.num_layers();
        const MAX_NEW: usize = 64;

        // Prefill helper → returns (session, first committed token).
        let prefill = |model: &BatchedEngine| -> Result<(BatchedInferenceSession, usize, u32)> {
            let mut session = model.create_batched_session(BatchedConfig::default())?;
            let seq = session.create_sequence()?;
            let prompt_t = Tensor::from_vec(ids.clone(), (1, ids.len()), &Device::Cpu)?;
            let step = model.forward_wave(
                &mut session,
                &[],
                &[],
                &[seq],
                std::slice::from_ref(&prompt_t),
                &[],
                &[],
                0,
                n_layers,
                None,
            )?;
            session.advance_sequence(seq, ids.len())?;
            let logits = step.logits_owned()?;
            if logits.is_empty() {
                return Err(candle::Error::msg("no prefill logits"));
            }
            let first = logits[0].i(0)?.argmax(0)?.to_scalar::<u32>()?;
            Ok((session, seq, first))
        };

        // ── Greedy baseline: one wave per token. ──
        let (mut gsession, gseq, gfirst) = prefill(&model)?;
        // Drop the prefill's spans so the snapshot below is pure plain-decode.
        #[cfg(feature = "profile")]
        let _ = crate::models::profile::pipeline_snapshot_and_reset();
        let mut greedy = vec![gfirst];
        let mut committed = gfirst;
        let t0 = std::time::Instant::now();
        while greedy.len() < MAX_NEW && committed != eos {
            let t = Tensor::from_vec(vec![committed], (1, 1), &Device::Cpu)?;
            let step = model.forward_wave(
                &mut gsession,
                &[gseq],
                std::slice::from_ref(&t),
                &[],
                &[],
                &[],
                &[],
                0,
                n_layers,
                None,
            )?;
            gsession.advance_sequence(gseq, 1)?;
            committed = step
                .logits
                .as_ref()
                .ok_or_else(|| candle::Error::msg("no decode logits"))?[0]
                .i(0)?
                .argmax(0)?
                .to_scalar::<u32>()?;
            greedy.push(committed);
        }
        let greedy_dt = t0.elapsed().as_secs_f32();
        // Per-phase anatomy at IDENTICAL corpus: the greedy snapshot is the
        // plain decode wave's per-layer cost, the spec snapshot (below) the
        // verify wave's — their diff is exactly what the decode-shaped-verify
        // work must collapse.
        #[cfg(feature = "profile")]
        {
            let snap = crate::models::profile::pipeline_snapshot_and_reset();
            let mut es = snap.entries.clone();
            es.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            eprintln!("[greedy-prof] plain decode waves (total ms, count):");
            for (name, ms, count) in es.iter().take(28) {
                eprintln!("  {name:34} {ms:9.1}ms  x{count}");
            }
        }

        // ── Speculative: drafter + batched verify_block, up to 4 drafts/step. ──
        let (mut ssession, sseq, sfirst) = prefill(&model)?;
        #[cfg(feature = "profile")]
        let _ = crate::models::profile::pipeline_snapshot_and_reset();
        let mut spec = vec![sfirst];
        let mut committed = sfirst;
        let mut steps = 0usize;
        let mut accepts: Vec<usize> = Vec::new(); // committed tokens per step (1 = 0 drafts accepted)
        let t0 = std::time::Instant::now();
        while spec.len() < MAX_NEW && committed != eos {
            steps += 1;
            let before = spec.len();
            let next = model.speculative_decode_step(
                &mut ssession,
                sseq,
                committed,
                4,
                n_layers,
                &mut |t| {
                    spec.push(t);
                    spec.len() < MAX_NEW && t != eos
                },
            )?;
            accepts.push(spec.len() - before); // committed tokens this step
            match next {
                Some(c) => committed = c,
                None => break,
            }
        }
        let spec_dt = t0.elapsed().as_secs_f32();
        #[cfg(feature = "profile")]
        {
            let snap = crate::models::profile::pipeline_snapshot_and_reset();
            let mut es = snap.entries.clone();
            es.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            eprintln!("[spec-prof] verify waves + drafts (total ms, count):");
            for (name, ms, count) in es.iter().take(28) {
                eprintln!("  {name:34} {ms:9.1}ms  x{count}");
            }
        }
        eprintln!("[spec-speedup] per-step committed (1=no draft accepted): {accepts:?}");

        let greedy_tps = greedy.len() as f32 / greedy_dt.max(1e-6);
        let spec_tps = spec.len() as f32 / spec_dt.max(1e-6);
        let text = tokenizer.decode(&spec, false).unwrap_or_default();
        eprintln!("[spec-speedup] continuation={text:?}");
        eprintln!(
            "[spec-speedup] greedy {greedy_tps:.2} tok/s ({} tok, {greedy_dt:.2}s) | speculative \
             {spec_tps:.2} tok/s ({} tok, {steps} steps ⇒ {:.2} tok/step, {spec_dt:.2}s) | \
             SPEEDUP {:.2}×",
            greedy.len(),
            spec.len(),
            spec.len() as f32 / steps.max(1) as f32,
            spec_tps / greedy_tps,
        );
        // Speculative commits whole blocks, so it may overshoot MAX_NEW by up to block_size-1; the
        // tokens must be identical on the common prefix (lossless — accepted tokens are the target's
        // own argmaxes).
        let n = spec.len().min(greedy.len());
        assert_eq!(
            &spec[..n],
            &greedy[..n],
            "speculative + batched verify must be BIT-IDENTICAL to greedy decode"
        );
        Ok(())
    }

    /// DSpark acceptance on REAL (non-counting) text — the honest metric. Counting is trivially
    /// predictable; this drafts an open-ended coding answer and reports mean accepted length
    /// (tokens/step) + the per-position histogram, the number the research target (~5+/6) is about.
    /// It does NOT assert bit-identical to greedy decode (the batched-verify prefill path is only
    /// tolerance-equal to the decode path on subtle tokens — a separate correctness item); it
    /// measures how well the drafter conditions on real context via the sliding target-hidden window.
    #[test]
    #[ignore]
    fn wave_speculative_realtext_acceptance() -> Result<()> {
        let _gpu = gpu_serial();
        let path = std::path::PathBuf::from(r"D:\models\deepseek-v4-flash-mxfp4")
            .join("DeepSeek-V4-Flash-0731-MXFP4_KO.gguf");
        let dspark = std::path::PathBuf::from(r"D:\models\deepseek-v4-flash-mxfp4")
            .join("dspark-DeepSeek-V4-Flash-0731-MXFP4.gguf");
        if !path.exists() || !dspark.exists() {
            eprintln!("[skip] target or DSpark drafter absent");
            return Ok(());
        }
        let device = Device::new_cuda(0)?;
        let (engine, drafter) = Engine::load_with_drafter(
            &path,
            Some(&dspark),
            &DEEPSEEK_V4,
            &device,
            Int8Mode::Performance,
        )?;
        let model =
            BatchedEngine::new(engine)?.with_drafter(drafter.expect("dspark path given"))?;

        let tok_path = crate::models::batch_test::test_helpers::hf_get(
            "deepseek-ai/DeepSeek-V4-Flash-0731",
            hf_hub::RepoType::Model,
            "main",
            "tokenizer.json",
        )?;
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| candle::Error::msg(format!("tokenizer load: {e}")))?;
        let eos = tokenizer.token_to_id("<｜end▁of▁sentence｜>").expect("eos");
        let n_layers = model.num_layers();
        let max_draft = 5usize;
        const MAX_NEW: usize = 128;

        let prompt = "<｜begin▁of▁sentence｜><｜User｜>Write a Python function that returns the \
             nth Fibonacci number, with a short docstring.<｜Assistant｜>";
        let ids: Vec<u32> = tokenizer
            .encode(prompt, false)
            .map_err(|e| candle::Error::msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();

        let mut session = model.create_batched_session(BatchedConfig::default())?;
        let seq = session.create_sequence()?;
        let prompt_t = Tensor::from_vec(ids.clone(), (1, ids.len()), &Device::Cpu)?;
        let step = model.forward_wave(
            &mut session,
            &[],
            &[],
            &[seq],
            std::slice::from_ref(&prompt_t),
            &[],
            &[],
            0,
            n_layers,
            None,
        )?;
        session.advance_sequence(seq, ids.len())?;
        let mut committed = step
            .logits
            .as_ref()
            .ok_or_else(|| candle::Error::msg("no prefill logits"))?[0]
            .i(0)?
            .argmax(0)?
            .to_scalar::<u32>()?;

        let mut gen = vec![committed];
        let mut steps = 0usize;
        let mut accepts: Vec<usize> = Vec::new();
        let t0 = std::time::Instant::now();
        while gen.len() < MAX_NEW && committed != eos {
            steps += 1;
            let before = gen.len();
            let next = model.speculative_decode_step(
                &mut session,
                seq,
                committed,
                max_draft,
                n_layers,
                &mut |t| {
                    gen.push(t);
                    gen.len() < MAX_NEW && t != eos
                },
            )?;
            accepts.push(gen.len() - before);
            match next {
                Some(c) => committed = c,
                None => break,
            }
        }
        let dt = t0.elapsed().as_secs_f32();
        let text = tokenizer.decode(&gen, false).unwrap_or_default();
        let mean = gen.len() as f32 / steps.max(1) as f32;
        // Histogram of committed-per-step (1 = no draft accepted … max_draft+1 = full block + bonus).
        let mut hist = vec![0usize; max_draft + 2];
        for &a in &accepts {
            hist[a.min(max_draft + 1)] += 1;
        }
        eprintln!("[spec-real] continuation={text:?}");
        eprintln!("[spec-real] per-step accepted: {accepts:?}");
        eprintln!(
            "[spec-real] {} tokens, {steps} steps ⇒ {mean:.2} tokens/step (max_draft={max_draft}); \
             {:.2} tok/s; committed-per-step histogram (idx=tokens): {hist:?}",
            gen.len(),
            gen.len() as f32 / dt.max(1e-6),
        );
        #[cfg(feature = "profile")]
        {
            // Aggregate forward_wave phase timings — dominated by the verify_block prefills, so this
            // shows WHERE the ~370 ms/step verify goes (attention vs MoE vs compressor/gallery/seal).
            let snap = crate::models::profile::pipeline_snapshot_and_reset();
            let mut es = snap.entries.clone();
            es.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            eprintln!("[verify-prof] forward_wave phases (total ms across run, count):");
            for (name, ms, count) in es.iter().take(24) {
                eprintln!("  {name:34} {ms:9.1}ms  x{count}");
            }
        }
        Ok(())
    }

    /// Bisect between the wave's two absorption paths. Same prompt as
    /// [`wave_chat_template_oracle`], but absorbed one token at a time through
    /// the DECODE path with no batched prefill — the per-token regime the
    /// single-session kernel engine already proves correct on the same weights.
    ///
    /// It is the discriminator, not a gate in its own right: a coherent
    /// generation here means batched prefill
    /// (`prefill_layer_rows`/`write_contiguous`/`set_len`) corrupts state, while
    /// an incoherent one means the wave decode state containers themselves
    /// diverge. Run it when the end-to-end gate fails.
    #[test]
    #[ignore]
    fn wave_decode_only_absorption_bisect() -> Result<()> {
        let _gpu = gpu_serial();
        let path = std::path::PathBuf::from(r"D:\models\deepseek-v4-flash-mxfp4")
            .join("DeepSeek-V4-Flash-0731-MXFP4_KO.gguf");
        if !path.exists() {
            eprintln!("[skip] merged file absent");
            return Ok(());
        }
        let device = Device::new_cuda(0)?;
        let engine = Engine::load(&path, &DEEPSEEK_V4, &device, Int8Mode::Performance)?;
        let model = BatchedEngine::new(engine)?;

        let tok_path = crate::models::batch_test::test_helpers::hf_get(
            "deepseek-ai/DeepSeek-V4-Flash-0731",
            hf_hub::RepoType::Model,
            "main",
            "tokenizer.json",
        )?;
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| candle::Error::msg(format!("tokenizer load: {e}")))?;
        let prompt = "<｜begin▁of▁sentence｜>You are a concise, factual assistant.\
             <｜User｜>What is the capital of France? \
             Reply with only the city name.<｜Assistant｜>";
        let ids: Vec<u32> = tokenizer
            .encode(prompt, false)
            .map_err(|e| candle::Error::msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();
        let eos = tokenizer
            .token_to_id("<｜end▁of▁sentence｜>")
            .expect("eos id");

        let mut session = model.create_batched_session(BatchedConfig::default())?;
        let seq = session.create_sequence()?;
        let n_layers = model.num_layers();

        let mut step_row = |tok: u32| -> Result<u32> {
            let t = Tensor::from_vec(vec![tok], (1, 1), &Device::Cpu)?;
            let step = model.forward_wave(
                &mut session,
                &[seq],
                std::slice::from_ref(&t),
                &[],
                &[],
                &[],
                &[],
                0,
                n_layers,
                None,
            )?;
            session.advance_sequence(seq, 1)?;
            let logits = step.logits_owned()?;
            if logits.is_empty() {
                return Err(candle::Error::msg("decode wave produced no logits"));
            }
            logits[0].i(0)?.argmax(0)?.to_scalar::<u32>()
        };

        // Absorb the prompt through decode steps (per-token regime).
        let mut next = 0u32;
        for &t in &ids {
            next = step_row(t)?;
        }
        let mut gen = vec![next];
        while gen.len() < 16 && next != eos {
            next = step_row(next)?;
            gen.push(next);
        }
        let text = tokenizer
            .decode(&gen, false)
            .map_err(|e| candle::Error::msg(format!("decode: {e}")))?;
        eprintln!("[wave-bisect] generated ids={gen:?}");
        eprintln!("[wave-bisect] continuation={text:?}");
        eprintln!("[wave-bisect] crisp={}", gen.len() >= 2 && gen[1] == eos);
        Ok(())
    }

    /// Layer-resolved divergence probe: run BOTH absorption paths segmented
    /// one layer at a time (residual pause/resume), capturing the residual at
    /// every layer boundary. The first layer whose residual diverges between
    /// batched prefill and per-token decode names the defective computation.
    /// Also prints each layer's kind + gallery size for the interpretation.
    #[test]
    #[ignore]
    fn wave_prefill_residual_divergence() -> Result<()> {
        let _gpu = gpu_serial();
        let path = std::path::PathBuf::from(r"D:\models\deepseek-v4-flash-mxfp4")
            .join("DeepSeek-V4-Flash-0731-MXFP4_KO.gguf");
        if !path.exists() {
            eprintln!("[skip] merged file absent");
            return Ok(());
        }
        let device = Device::new_cuda(0)?;
        let engine = Engine::load(&path, &DEEPSEEK_V4, &device, Int8Mode::Performance)?;
        let model = BatchedEngine::new(engine)?;

        let tok_path = crate::models::batch_test::test_helpers::hf_get(
            "deepseek-ai/DeepSeek-V4-Flash-0731",
            hf_hub::RepoType::Model,
            "main",
            "tokenizer.json",
        )?;
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| candle::Error::msg(format!("tokenizer load: {e}")))?;
        let prompt = "<｜begin▁of▁sentence｜>You are a concise, factual assistant.\
             <｜User｜>What is the capital of France? \
             Reply with only the city name.<｜Assistant｜>";
        let ids: Vec<u32> = tokenizer
            .encode(prompt, false)
            .map_err(|e| candle::Error::msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();
        let n = ids.len();
        let n_layers = model.num_layers();

        let mut session = model.create_batched_session(BatchedConfig::default())?;
        let seq_a = session.create_sequence()?; // per-token decode absorb
        let seq_b = session.create_sequence()?; // batched prefill

        // A: per-token, layer-segmented; keep residual after each layer for
        // every token: res_a[l] = [n] rows.
        let mut res_a: Vec<Vec<Tensor>> = vec![Vec::with_capacity(n); n_layers];
        for &t in &ids {
            let tt = Tensor::from_vec(vec![t], (1, 1), &Device::Cpu)?;
            let mut resid: Option<Tensor> = None;
            for l in 0..n_layers {
                let step = model.forward_wave(
                    &mut session,
                    &[seq_a],
                    std::slice::from_ref(&tt),
                    &[],
                    &[],
                    &[],
                    &[],
                    l,
                    l + 1,
                    resid.take(),
                )?;
                match step.into_residual() {
                    Some(r) => {
                        res_a[l].push(r.clone());
                        resid = Some(r);
                    }
                    None => {
                        // final layer returns logits; no residual to store
                    }
                }
            }
            session.advance_sequence(seq_a, 1)?;
        }

        // B: batched prefill, layer-segmented; res_b[l] = [1, n, hc*dim].
        let pt = Tensor::from_vec(ids.clone(), (1, n), &Device::Cpu)?;
        let mut res_b: Vec<Tensor> = Vec::with_capacity(n_layers);
        let mut resid: Option<Tensor> = None;
        for l in 0..n_layers {
            let step = model.forward_wave(
                &mut session,
                &[],
                &[],
                &[seq_b],
                std::slice::from_ref(&pt),
                &[],
                &[],
                l,
                l + 1,
                resid.take(),
            )?;
            if let Some(r) = step.into_residual() {
                res_b.push(r.clone());
                resid = Some(r);
            }
        }
        session.advance_sequence(seq_b, n)?;

        // Compare per layer: prefill row t vs decode step t residual.
        let state = model.seq_state.read().unwrap();
        let ea = state.get(&seq_a).expect("seq_a state");
        for l in 0..res_b.len() {
            let rb = &res_b[l]; // [1, n, hcdim]
            let hcdim = rb.dim(2)?;
            let mut worst = 0f32;
            let mut worst_t = 0usize;
            for t in 0..n {
                let a_row = res_a[l][t].reshape(hcdim)?.to_dtype(DType::F32)?;
                let b_row = rb.i((0, t))?.reshape(hcdim)?.to_dtype(DType::F32)?;
                let d = (a_row - b_row)?.abs()?.max_all()?.to_scalar::<f32>()?;
                if d > worst {
                    worst = d;
                    worst_t = t;
                }
            }
            let glen = ea.layers[l].gallery.as_ref().map_or(0, |g| g.len());
            let kind = {
                let e = model.engine.engine_layer(l);
                format!("{:?}", e.attn.kind())
            };
            eprintln!(
                "[resid-div] layer {l:2} ({kind:>13}, gallery {glen:2}): max|Δresidual| = {worst:.6} (worst token {worst_t})"
            );
        }
        Ok(())
    }

    /// Prefill-vs-decode state audit: absorb the SAME prompt into two
    /// sequences — one through per-token decode waves (proven crisp), one
    /// through the batched prefill — then diff every layer's arena window
    /// bytes, corpus state, and finally the next decode step's logits. The
    /// first divergent artifact is the batched-prefill bug.
    #[test]
    #[ignore]
    fn wave_prefill_state_matches_decode_steps() -> Result<()> {
        let _gpu = gpu_serial();
        let path = std::path::PathBuf::from(r"D:\models\deepseek-v4-flash-mxfp4")
            .join("DeepSeek-V4-Flash-0731-MXFP4_KO.gguf");
        if !path.exists() {
            eprintln!("[skip] merged file absent");
            return Ok(());
        }
        let device = Device::new_cuda(0)?;
        let engine = Engine::load(&path, &DEEPSEEK_V4, &device, Int8Mode::Performance)?;
        let model = BatchedEngine::new(engine)?;

        let tok_path = crate::models::batch_test::test_helpers::hf_get(
            "deepseek-ai/DeepSeek-V4-Flash-0731",
            hf_hub::RepoType::Model,
            "main",
            "tokenizer.json",
        )?;
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| candle::Error::msg(format!("tokenizer load: {e}")))?;
        let prompt = "<｜begin▁of▁sentence｜>You are a concise, factual assistant.\
             <｜User｜>What is the capital of France? \
             Reply with only the city name.<｜Assistant｜>";
        let ids: Vec<u32> = tokenizer
            .encode(prompt, false)
            .map_err(|e| candle::Error::msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();
        let n = ids.len();

        let mut session = model.create_batched_session(BatchedConfig::default())?;
        let seq_a = session.create_sequence()?; // decode-step absorb (reference)
        let seq_b = session.create_sequence()?; // batched prefill (suspect)
        let n_layers = model.num_layers();

        // A: per-token decode waves.
        for &t in &ids {
            let tt = Tensor::from_vec(vec![t], (1, 1), &Device::Cpu)?;
            model.forward_wave(
                &mut session,
                &[seq_a],
                std::slice::from_ref(&tt),
                &[],
                &[],
                &[],
                &[],
                0,
                n_layers,
                None,
            )?;
            session.advance_sequence(seq_a, 1)?;
        }
        // B: one batched prefill wave.
        let pt = Tensor::from_vec(ids.clone(), (1, n), &Device::Cpu)?;
        let step_b = model.forward_wave(
            &mut session,
            &[],
            &[],
            &[seq_b],
            std::slice::from_ref(&pt),
            &[],
            &[],
            0,
            n_layers,
            None,
        )?;
        session.advance_sequence(seq_b, n)?;
        let logits_b = step_b.logits_owned()?.swap_remove(0);
        let next_b = logits_b.i(0)?.argmax(0)?.to_scalar::<u32>()?;
        eprintln!("[state-diff] prefill argmax token={next_b}");

        // Window bytes per layer. The decode-absorbed slot's HOST population
        // count lags one token (the final token was written in-kernel with no
        // host set_len after it), so compare the fully-covered [0, n-1).
        let cmp_len = n - 1;
        for l in 0..n_layers {
            let b = &session.backings()[l];
            let (ka, _) = b.read_contiguous(seq_a, 0, cmp_len)?;
            let (kb, _) = b.read_contiguous(seq_b, 0, cmp_len)?;
            let d = (ka.to_dtype(DType::F32)? - kb.to_dtype(DType::F32)?)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            if d != 0.0 {
                eprintln!("[state-diff] layer {l}: window max|Δ| = {d}");
            }
        }
        // Corpus state per layer.
        {
            let state = model.seq_state.read().unwrap();
            let ea = state.get(&seq_a).expect("seq_a state");
            let eb = state.get(&seq_b).expect("seq_b state");
            for l in 0..n_layers {
                let ga = ea.layers[l].gallery.as_ref();
                let gb = eb.layers[l].gallery.as_ref();
                let (la, lb) = (ga.map_or(0, |g| g.len()), gb.map_or(0, |g| g.len()));
                if la != lb {
                    eprintln!("[state-diff] layer {l}: gallery len {la} vs {lb}");
                    continue;
                }
                if let (Some(ga), Some(gb)) = (ga, gb) {
                    if la > 0 {
                        let ea_t = ga.attn_entries()?.to_dtype(DType::F32)?;
                        let eb_t = gb.attn_entries()?.to_dtype(DType::F32)?;
                        let d = (&ea_t - &eb_t)?.abs()?.max_all()?.to_scalar::<f32>()?;
                        if d != 0.0 {
                            let na = ea_t.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
                            let nb = eb_t.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
                            let va: Vec<f32> = ea_t.i((0, 0..4))?.to_vec1::<f32>()?;
                            let vb: Vec<f32> = eb_t.i((0, 0..4))?.to_vec1::<f32>()?;
                            eprintln!(
                                "[state-diff] layer {l}: gallery entries max|Δ| = {d} \
                                 |A|={na:.4} |B|={nb:.4} A[0][..4]={va:?} B[0][..4]={vb:?}"
                            );
                        }
                        let pa = ga.positions()?.to_vec1::<u32>()?;
                        let pb = gb.positions()?.to_vec1::<u32>()?;
                        if pa != pb {
                            eprintln!("[state-diff] layer {l}: gallery positions {pa:?} vs {pb:?}");
                        }
                    }
                }
            }
        }

        // Next decode step on both with the same token.
        let tok = Tensor::from_vec(vec![next_b], (1, 1), &Device::Cpu)?;
        let toks = [tok.clone(), tok];
        let step = model.forward_wave(
            &mut session,
            &[seq_a, seq_b],
            &toks,
            &[],
            &[],
            &[],
            &[],
            0,
            n_layers,
            None,
        )?;
        let logits = step.logits_owned()?;
        let arg_a = logits[0].i(0)?.argmax(0)?.to_scalar::<u32>()?;
        let arg_b = logits[1].i(0)?.argmax(0)?.to_scalar::<u32>()?;
        let dl = (logits[0].to_dtype(DType::F32)? - logits[1].to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        eprintln!("[state-diff] next-step argmax: decode-absorbed={arg_a} prefill-absorbed={arg_b} max|Δlogits|={dl}");
        assert_eq!(
            arg_a, arg_b,
            "decode step after prefill diverges from decode step after per-token absorb"
        );
        Ok(())
    }

    /// Oracle for the conversation gate: the EXACT chat-templated token stream
    /// the conversation engine assembles (BOS + system text + the user/assistant
    /// role markers around the question), fed as ONE plain contiguous prefill
    /// through the reference wave path and greedy-decoded to EOS.
    ///
    /// What it prints is what the conversation engine MUST reproduce, which is
    /// what makes it a bisect: a coherent generation here with an incoherent one
    /// from the conversation engine puts the fault in that engine's glue and
    /// position wiring, not in the prompt or the wave path. The role markers are
    /// the model's, taken from its `Dialect`; the wave path never sees them as
    /// anything but token ids.
    #[test]
    #[ignore]
    fn wave_chat_template_oracle() -> Result<()> {
        let _gpu = gpu_serial();
        let path = std::path::PathBuf::from(r"D:\models\deepseek-v4-flash-mxfp4")
            .join("DeepSeek-V4-Flash-0731-MXFP4_KO.gguf");
        if !path.exists() {
            eprintln!("[skip] merged file absent");
            return Ok(());
        }
        let device = Device::new_cuda(0)?;
        let engine = Engine::load(&path, &DEEPSEEK_V4, &device, Int8Mode::Performance)?;
        let model = BatchedEngine::new(engine)?;

        let tok_path = crate::models::batch_test::test_helpers::hf_get(
            "deepseek-ai/DeepSeek-V4-Flash-0731",
            hf_hub::RepoType::Model,
            "main",
            "tokenizer.json",
        )?;
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| candle::Error::msg(format!("tokenizer load: {e}")))?;
        let prompt = "<｜begin▁of▁sentence｜>You are a concise, factual assistant.\
             <｜User｜>What is the capital of France? \
             Reply with only the city name.<｜Assistant｜>";
        let ids: Vec<u32> = tokenizer
            .encode(prompt, false)
            .map_err(|e| candle::Error::msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();
        eprintln!("[wave-conv] prompt ids={ids:?}");
        let eos = tokenizer
            .token_to_id("<｜end▁of▁sentence｜>")
            .expect("eos id");

        let mut session = model.create_batched_session(BatchedConfig::default())?;
        let seq = session.create_sequence()?;
        let n_layers = model.num_layers();

        let prompt_t = Tensor::from_vec(ids.clone(), (1, ids.len()), &Device::Cpu)?;
        let step = model.forward_wave(
            &mut session,
            &[],
            &[],
            &[seq],
            std::slice::from_ref(&prompt_t),
            &[],
            &[],
            0,
            n_layers,
            None,
        )?;
        session.advance_sequence(seq, ids.len())?;
        let logits = step.logits_owned()?;
        if logits.is_empty() {
            return Err(candle::Error::msg("prefill wave produced no logits"));
        }
        let mut next = logits[0].i(0)?.argmax(0)?.to_scalar::<u32>()?;

        let mut gen = vec![next];
        while gen.len() < 24 && next != eos {
            let tok = Tensor::from_vec(vec![next], (1, 1), &Device::Cpu)?;
            let step = model.forward_wave(
                &mut session,
                &[seq],
                std::slice::from_ref(&tok),
                &[],
                &[],
                &[],
                &[],
                0,
                n_layers,
                None,
            )?;
            session.advance_sequence(seq, 1)?;
            let logits = step.logits_owned()?;
            if logits.is_empty() {
                return Err(candle::Error::msg("decode wave produced no logits"));
            }
            next = logits[0].i(0)?.argmax(0)?.to_scalar::<u32>()?;
            gen.push(next);
        }
        let text = tokenizer
            .decode(&gen, false)
            .map_err(|e| candle::Error::msg(format!("decode: {e}")))?;
        eprintln!("[wave-conv] generated ids={gen:?}");
        eprintln!("[wave-conv] continuation={text:?}");
        eprintln!(
            "[wave-conv] stopped_on_eos={} after {} tokens",
            next == eos,
            gen.len()
        );
        Ok(())
    }
}
