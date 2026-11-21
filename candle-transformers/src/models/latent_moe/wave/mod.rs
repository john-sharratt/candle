//! The DeepSeek-V4-Flash model on the shared wave loop.
//!
//! `forward_wave` delegates to the shared driver. This model's architecture lives
//! in its `WaveSweep` implementation: attention in the paged latent kernels, the
//! mHC hyper-connection loop, and the per-layer corpus machinery. Each sequence's
//! corpus state sits behind interior mutability keyed by sequence index, because
//! the trait has no sequence-free hook, so a prefill that arrives at offset zero
//! for a known index resets that sequence's state. The optional DSpark drafter is
//! attached with `with_drafter`, and its target-feature stash is keyed by absolute
//! token position.
//!
//! - `corpus_state`: per-sequence galleries, compressor state and snapshots.
//! - `managed`: the scheduler-facing trait hooks and KV ground.
//! - `speculative`: draft gating and rollback to the accepted boundary.
//! - `sweep_wave`: the wave forward, its row groups, metadata and head.
//!
//! Test files:
//!
//! - `tests_prefill`: end-to-end prefill and decode gates.
//! - `tests_reproducibility`: greedy reproducibility and its bitwise probe.
//! - `tests_speculative`: lossless speculation and its acceptance measures.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};

use candle::quantized::pinned_staging::Generation;
use candle::{DType, Device, Result, Tensor};

use crate::models::batched_inference::{
    BatchedConfig, BatchedInferenceSession, ManagedBatchedModel, PendingGlue, WaveResult,
    MAX_PREFILL_TOKENS,
};
use crate::models::batched_model::{WaveGuard, WavePhase};
use crate::models::head_rows::select_head_rows;
use crate::models::kv_cache_utils::SequenceContext;
use crate::models::residual_rows::CreepResidual;
use crate::models::tensor_cat::TensorCat;
use crate::models::wave_driver::{
    drive_step, drive_wave, StepOutcome, StepSpec, SweepMode, WaveGroups, WaveSweep,
};
use candle_nn::kv_cache::ModelGeometry;
use candle_nn::kv_cache::CHUNK_SIZE;

use crate::models::verify_wave::{upload_plan_rows, VerifyPlan};
use crate::models::wave_token_ids::host_token_ids;

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
use super::rope_tables::LatentRopeTables;
use crate::models::expert_lre::PipelineStats;
use crate::models::profile::{span, span_if, ProfileSnapshot};

use super::compressor::{assemble_groups_batched, Compressor, GroupPool, SeqAssemble};
use super::desc;
use super::rope::RotaryCache;
use crate::models::slot_header::SLOT_HEADER_BYTES;

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
        let mut tables = LatentRopeTables::default();
        for l in 0..cfg.n_layers {
            let (theta, orig) = cfg.rope_params(l);
            let rope_tab = tables.for_layer(
                theta,
                orig,
                cfg.rope_factor,
                cfg.beta_fast,
                cfg.beta_slow,
                engine.engine_device(),
            )?;
            layer_static.push(KernelLayerStatic::new(
                &engine.engine_layer(l).attn,
                rope_tab,
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
    /// GPU-resident (~1 GB); its 3×256 routed experts are a permanently VRAM-resident Q2_KO set
    /// (~5 GiB, [`super::dspark_experts::DsparkStreamingMoe`]). The
    /// target engine sizes its expert pool greedily to *all* free VRAM (spilling the remainder to
    /// the pinned pool), so the engine must be loaded **first**; the drafter then loads into the
    /// engine's activation headroom (shared with target KV/activations), leaving the target's expert
    /// pool — and its pinned remainder — at the baseline that fits the page-lock ceiling. On a
    /// device of 64 GiB or less the drafter's `load` already errored, so this
    /// is never reached and speculative stays disabled.
    pub fn with_drafter(mut self, drafter: super::dspark::DsparkDrafter) -> Result<Self> {
        // `dflash.target_layers` (1-based, [41,42,43] on the 43-layer target) name the three
        // consecutive target layers whose RAW hidden states condition the drafter (llama.cpp
        // `dflash.cpp`: the encoder input is `target_layers.size()·n_embd` concatenated raw residual
        // streams — no norm, no per-layer reduction beyond the model's dim-wide readout). Shift to
        // 0-based → [40,41,42], the last three layer outputs. `forward_wave` captures `head_reduce(h)`
        // (our hc_mult→dim readout) after each and stashes the concatenation.
        let n = self.engine.layer_count();
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

    /// Turn seal — **Artifact B** (docs/deepseek/deepseek_turn_seal_persistence.md):
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
        if snaps.len() != self.engine.layer_count() {
            candle::bail!(
                "corpus_restore: {} snapshots for {} layers",
                snaps.len(),
                self.engine.layer_count()
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
        let mut layers = Vec::with_capacity(self.engine.layer_count());
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
        if snap.layers.len() != self.engine.layer_count() {
            candle::bail!(
                "window_ring_restore: {} layers for {} model layers",
                snap.layers.len(),
                self.engine.layer_count()
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

    /// Capture every wave member's streaming-corpus state ahead of a sweep —
    /// per layer a pair of `Arc` clones and the gallery length, host-only, no
    /// device work and no fences. See [`WaveCorpusSnap`].
    fn capture_corpus_state(&self, seqs: &[usize]) -> Result<Vec<WaveCorpusSnap>> {
        let map = self
            .seq_state
            .read()
            .map_err(|_| candle::Error::Msg("seq_state lock poisoned".into()))?;
        Ok(seqs
            .iter()
            .map(|&seq| WaveCorpusSnap {
                seq,
                state: map.get(&seq).map(|e| {
                    (
                        e.absorbed,
                        e.layers
                            .iter()
                            .map(|ls| {
                                (
                                    ls.comp.as_ref().map(|c| c.state_snapshot()),
                                    ls.icomp.as_ref().map(|c| c.state_snapshot()),
                                    ls.gallery.as_ref().map_or(0, |g| g.len()),
                                )
                            })
                            .collect(),
                    )
                }),
            })
            .collect())
    }

    /// Restore what [`Self::capture_corpus_state`] captured — the failed-wave
    /// path. Galleries truncate to the recorded lengths (their appends are
    /// append-only within a wave, and truncation is host metadata — the same
    /// operation `rollback_verify_state` performs in production, so in-flight
    /// kernels of the dead wave read rows that simply go unreferenced).
    fn restore_corpus_state(&self, snaps: Vec<WaveCorpusSnap>) -> Result<()> {
        let mut map = self
            .seq_state
            .write()
            .map_err(|_| candle::Error::Msg("seq_state lock poisoned".into()))?;
        for snap in snaps {
            let Some((absorbed, layers)) = snap.state else {
                // The sequence had no state entering the wave; whatever the
                // failed wave created for it is discarded outright, so the
                // retry's `ensure_seq_state` starts from nothing again.
                map.remove(&snap.seq);
                continue;
            };
            let Some(entry) = map.get_mut(&snap.seq) else {
                continue;
            };
            entry.absorbed = absorbed;
            for ((comp_s, icomp_s, glen), ls) in layers.into_iter().zip(entry.layers.iter_mut()) {
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
        }
        Ok(())
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

mod corpus_state;
mod managed;
mod speculative;
mod sweep_wave;
#[cfg(test)]
mod tests_prefill;
#[cfg(test)]
mod tests_reproducibility;
#[cfg(test)]
mod tests_speculative;

pub use corpus_state::*;
use speculative::*;
