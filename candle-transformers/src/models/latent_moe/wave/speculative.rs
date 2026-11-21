//! Speculative decoding: drafting, the batched verify, and the rollback of
//! rejected draft tokens.
//!
//! A partial accept installs the sequence's corpus state at the accepted row
//! boundary, so the compressors and galleries never keep a rejected draft. `SpecStats`
//! tracks accepted tokens per drafted step as an exponential average. Drafting is
//! skipped while that average is below a threshold, and a probe re-measures it
//! after a fixed number of skipped steps, so a change of workload re-enables
//! speculation. `draft_one` also narrows the block by the measured acceptance and
//! the verify wave's expert miss rate, and stops at the confidence threshold.
//! Each of these shapes cost and acceptance, and never the output.

use super::*;

/// One layer's slice of a [`VerifySnapshot`]: the pre-verify compressor states +
/// gallery length, plus the per-ROW-boundary states stashed by `forward_wave`
/// during the verify pass. Rollback after a partial accept just INSTALLS the
/// state at the accepted boundary — the wave's own per-row streaming capture
/// already produced exactly the state a plain decode of the accepted prefix
/// would have (that is the decode-shaped verify's construction), so no replay
/// (and none of its per-layer kernel launches) is needed.
pub(super) struct LayerVerifySnap {
    pub(super) comp: Option<super::super::compressor::CompressorState>,
    pub(super) icomp: Option<super::super::compressor::CompressorState>,
    pub(super) gallery_len: usize,
    /// `(comp, icomp, gallery_len)` AFTER absorbing block row `k`, for each of
    /// the block's rows in order. `Arc`-clone cheap.
    pub(super) row_states: Vec<(
        Option<super::super::compressor::CompressorState>,
        Option<super::super::compressor::CompressorState>,
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
pub(super) struct SpecStats {
    /// EMA of accepted tokens per DRAFTED step. Seeded optimistically so fresh
    /// sequences speculate until measured otherwise.
    pub(super) ema: f32,
    /// Verify-only steps since the last drafted step (probe scheduling).
    pub(super) fallback_steps: u32,
}

/// EMA smoothing for [`SpecStats::ema`] (≈ last ~8 drafted steps dominate).
pub(super) const SPEC_EMA_ALPHA: f32 = 0.25;

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
pub(super) const SPEC_MIN_ACCEPT: f32 = 2.3;

/// Plain-decode steps between drafted probes while below the threshold. A
/// probe costs one verify wave (≈ the cost ratio in decode waves), so the
/// steady fallback overhead is `ratio / interval` — 8 paid ~46% on this box's
/// ~3.7 ratio and dominated the fallback's throughput; 32 pays ~11% while a
/// workload shift (prose → code) is still re-detected within a few dozen
/// tokens.
pub(super) const SPEC_PROBE_INTERVAL: u32 = 32;

/// Pre-verify snapshot of one sequence's streaming corpus state, taken by
/// `verify_blocks` before its decode-row forward. On a partial accept the
/// driver's `truncate_sequence` installs the stashed state at the accepted row
/// boundary, so the compressors/galleries never retain rejected draft tokens —
/// without this, a rejected tail stays absorbed (wrong rows in the partial-group
/// buffer, a group pooled over draft tokens in the gallery, `group_idx` advanced)
/// and the re-decoded positions get absorbed AGAIN as duplicate, shifted groups:
/// the model re-attends duplicated context and repeats itself.
pub(super) struct VerifySnapshot {
    /// Absolute position of the block's first token (`q_start`).
    pub(super) base: usize,
    /// The verify block's length (`[committed, drafts…]`).
    pub(super) block_len: usize,
    pub(super) layers: Vec<LayerVerifySnap>,
}

impl BatchedEngine {
    /// One sequence's DSpark proposal — see
    /// [`ManagedBatchedModel::speculative_draft`].
    pub(super) fn draft_one(
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
        //
        // The tallies are read unsettled (`ExpertCache::hit_counts`): a settle
        // waits for the device to reach the last queued MoE layer and for the
        // pipeline thread to serve the backlog, once per draft, and an EMA
        // needs no exact interval boundary.
        let miss_ema = {
            let (hits, misses) = self.engine.experts().hit_counts();
            let mut g = self
                .spec_miss
                .write()
                .map_err(|_| candle::Error::Msg("spec_miss lock poisoned".into()))?;
            let (last_hits, last_misses, ema) = *g;
            let dh = hits.saturating_sub(last_hits);
            let dm = misses.saturating_sub(last_misses);
            let ema = if dh + dm > 0 {
                let rate = dm as f32 / (dh + dm) as f32;
                SPEC_EMA_ALPHA * rate + (1.0 - SPEC_EMA_ALPHA) * ema
            } else {
                ema
            };
            *g = (hits, misses, ema);
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
