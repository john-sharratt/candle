//! The trait a model implements for its layer sweep, which runs inside the shared
//! wave driver.
//!
//! The driver owns everything around a forward and calls the model's sweep over a
//! layer range. The sweep returns the residual when the range stops short of the
//! head, and the logits when it reaches it. A model states how its trunk layers
//! map onto its KV indices, which can differ on a hybrid, and whether its device
//! can give a cold expert up. A faulted segment is then rolled back and never
//! committed. Offsets are reconciled to the backing before each forward, and the
//! sweep builds its own attention metadata, since its arena may be moved first.

use super::*;

/// The per-model half of a wave: run one layer range over assembled contexts.
pub trait WaveSweep {
    fn device(&self) -> &Device;

    /// The sequences whose every prefill row the head scores this wave — a
    /// speculative verify's blocks — where every other prefill span is scored
    /// at its last row. What the driver maps the head's rows back to members
    /// by. Empty for a model that never verifies.
    fn verify_row_seqs(&self) -> Vec<usize> {
        Vec::new()
    }

    /// Transformer depth — what bounds a wave's layer range. On a hybrid this
    /// is the trunk depth, not the KV-layer count.
    fn num_layers(&self) -> usize;

    /// Widest prefill this model runs in one forward, in tokens.
    fn prefill_width_cap(&self, act_dtype: DType) -> usize;

    /// One row of this model's residual stream: its elements and their type —
    /// what a creep's arena holds per row (`docs/wave_step.md` §3). Asked
    /// before a step places its tier, since the arena is claimed then.
    fn residual_row(&self, session: &BatchedInferenceSession) -> Result<(usize, DType)>;

    /// The **KV-cache** index range a trunk-layer range writes to.
    ///
    /// A session allocates one paged cache per layer that attends, so on a
    /// uniform transformer the two coincide and this is the identity. On a
    /// hybrid three quarters of the layers own no KV, and a rollback driven by
    /// trunk indices would index past the end of the cache vector — so the
    /// translation lives here, where the driver can ask for it, rather than
    /// being assumed.
    fn kv_layer_range(&self, layer_start: usize, layer_end: usize) -> (usize, usize) {
        (layer_start, layer_end)
    }

    /// Forward-entry invariant: every member's logical offset must equal the
    /// token count its live block table covers. The varlen metadata is built
    /// from the offsets while the slot headers are built from the block tables
    /// — the attention kernels resolve every `[0, kv_len)` position through the
    /// table, so any divergence walks them past the slot's span in the packed
    /// staged uploads (garbage slice indices → wild record pointers →
    /// CUDA_ERROR_ILLEGAL_ADDRESS, or silent cross-slot reads). Offsets run
    /// AHEAD of the backing when a projection drops sections it could not lift
    /// under VRAM pressure; they run BEHIND after glue reserves gap chunks the
    /// wave didn't reflect. Positions are slot-relative (slice ropes), so the
    /// backing length is also the correct RoPE base either way.
    ///
    /// **A model hook because the invariant itself is a model property.** The
    /// default states the uniform/hybrid contract, `offset == backing`. A model
    /// whose arena is a sliding window ring keeps ABSOLUTE offsets against a
    /// RESIDENT backing (`offset == base_pos + backing`), and the default's
    /// clamp would silently re-base every slid sequence — so such a model
    /// overrides this with its own reconciliation (or a no-op, where its wave
    /// entry re-derives lengths from the session itself).
    fn reconcile_entry_offsets(
        &self,
        session: &mut BatchedInferenceSession,
        seq_groups: [&[usize]; 3],
    ) -> Result<()> {
        for ids in seq_groups {
            for &i in ids {
                let off = session.sequence_offset(i).unwrap_or(0);
                let backing = session.sequence_backing_tokens(i).unwrap_or(off);
                if backing != off {
                    if backing < off {
                        tracing::warn!(
                            seq = i,
                            offset = off,
                            backing,
                            "sequence offset AHEAD of backing at forward entry — \
                             clamped down (projection dropped un-liftable sections)"
                        );
                    } else {
                        tracing::debug!(
                            seq = i,
                            offset = off,
                            backing,
                            "sequence offset behind backing at forward entry — advanced"
                        );
                    }
                    session.set_sequence_offset(i, backing)?;
                }
            }
        }
        Ok(())
    }

    /// Advance the decode rows' per-layer KV lengths after a completed step —
    /// once per delivered token, every layer at once, called by the driver
    /// only after the head ran and the logits were copied off.
    ///
    /// The default is the uniform/hybrid contract: every cache to
    /// `offset + 1`, with a failure at row `i` unwound by truncating rows
    /// `0..=i` back to their entry lengths (leaving layers `0..k` at
    /// `offset+1` against the rest at `offset` is exactly the per-layer
    /// divergence this consolidation exists to prevent, and the wave rollback
    /// cannot reach it).
    ///
    /// A model whose decode write-length is committed **on-device** by its
    /// decode kernel — and re-committed from the session at every wave entry —
    /// overrides this as a no-op: the default's `set_current_seq_len` stamps
    /// the session's ABSOLUTE offset into the backing, which a sliding window
    /// ring addresses in RESIDENT terms.
    fn advance_decode_rows(&self, contexts: &mut [SequenceContext], n_decode: usize) -> Result<()> {
        for i in 0..n_decode {
            let offset = contexts[i].offset;
            let mut advance = || -> Result<()> {
                for cache in contexts[i].kv_caches.caches.iter_mut() {
                    cache.set_current_seq_len(offset + 1)?;
                }
                Ok(())
            };
            if let Err(e) = advance() {
                // Every layer of every touched row, even past a failure. The
                // `try_for_each` this replaced stopped at the first error and
                // discarded it, so an unwind that hit a bad layer left the rows
                // *before* it truncated and the rest advanced — reproducing, from
                // the recovery path, the exact per-layer skew the unwind exists to
                // erase, and saying nothing about it.
                for c in contexts[..=i].iter_mut() {
                    let off = c.offset;
                    for (li, cache) in c.kv_caches.caches.iter_mut().enumerate() {
                        if let Err(te) = cache.rollback_to_offset(off) {
                            tracing::warn!(
                                layer = li,
                                offset = off,
                                "decode advance unwind: layer truncate failed; continuing \
                                 so the remaining layers are not left advanced: {te}"
                            );
                        }
                    }
                }
                return Err(e);
            }
        }
        Ok(())
    }

    /// Undo a failed wave's per-layer KV bookkeeping, so the scheduler's retry
    /// is a retry rather than a decode against per-layer token windows.
    ///
    /// The default truncates every row on every KV layer of the range back to
    /// its entry offset ([`super::wave_admit::rollback_wave_kv`]). A model
    /// that re-derives every layer's writer length from the session at wave
    /// entry overrides this to match — the default's absolute-offset truncate
    /// would mis-address a sliding window ring's resident arena.
    fn rollback_wave(
        &self,
        contexts: &mut [SequenceContext],
        kv_start: usize,
        kv_end: usize,
    ) -> Result<()> {
        super::super::wave_admit::rollback_wave_kv(contexts, kv_start, kv_end)
    }

    /// Fail the segment just swept if the device gave up part of it — an expert
    /// cache whose workers lost a cold expert (`ExpertCache::take_fault`). The
    /// driver calls it after every segment, residual or logits, and on an error
    /// rolls the segment back exactly as a failed sweep, so a faulted segment
    /// never commits its KV or hands its residual on.
    ///
    /// The fault word is complete only once every launch that could claim it
    /// is, so a model whose device can give work up waits for the segment's
    /// launches first (`ExpertCache::take_segment_fault`). Only a cache that can
    /// leave a worker waiting on a cold expert can: one whose every expert is
    /// in VRAM or pinned host memory never waits, and its check is free. The
    /// wait is not: nothing before it has synchronised the segment — the logits
    /// stay on the device — so it stalls the host until the device drains, and
    /// taken on every wave it cost Flash-Next's single-session decode ~6 ms a
    /// step on a card that could never fault.
    ///
    /// Required, not defaulted: every sweep states whether its device can give
    /// work up, so a model gaining an expert cache cannot silently skip it.
    fn take_device_fault(&self) -> Result<()>;

    /// Run `[layer_start, layer_end)` over the wave, returning the residual
    /// (range stopped short of the head) or the logits (range reached it) —
    /// or, under [`SweepMode::Prepare`], the segment's price.
    ///
    /// **A residual comes back on the forward span, with the span's guard.**
    /// The driver moves it where it goes next — a step's next segment, a
    /// creep's arena — while the guard still holds it.
    ///
    /// The sweep owns the session for its duration. It builds its own attention
    /// metadata (see the module docs for why the driver cannot), assembles its
    /// [`SequenceContext`]s via [`assemble_wave_contexts`] when its layer body
    /// works through per-sequence `KvCache`s, and is free to skip that entirely
    /// when its KV lives behind the session's per-layer backings. Order matters
    /// inside: metadata builds borrow the session shared, the contexts borrow
    /// it mutably, so headers are built before contexts are taken.
    fn sweep(
        &self,
        session: &mut BatchedInferenceSession,
        wave: WaveGroups<'_>,
    ) -> Result<(WavePhase, Option<WaveGuard>)>;
}

/// Assemble the per-sequence contexts for a wave, in `seq_ids` order.
///
/// `inputs` is one tensor per sequence in the same order. Fails loudly when a
/// named sequence has no live slot — `caches_for_sequences_mut` silently skips
/// those, which used to surface ~100 lines later as a `checked_sub` underflow
/// naming neither the fault nor the sequence.
///
/// Used by the sweeps on the way in, and by [`drive_wave`] after the sweep
/// returns — the KV rollback of a failed wave and the decode advance of a
/// completed one need the same borrows, and re-assembling is legal because the
/// sweep's mutable borrow of the session ends when it returns (nothing inside a
/// wave moves the session's sequence offsets; the scheduler advances them after
/// the forward).
pub(crate) fn assemble_wave_contexts<'s>(
    session: &'s mut BatchedInferenceSession,
    seq_ids: &[usize],
    inputs: &'s [Tensor],
) -> Result<Vec<SequenceContext<'s>>> {
    if inputs.len() != seq_ids.len() {
        candle::bail!(
            "assemble_wave_contexts: {} inputs against {} sequences",
            inputs.len(),
            seq_ids.len()
        );
    }
    let caches_data = session.caches_for_sequences_mut(seq_ids);
    if caches_data.len() != seq_ids.len() {
        let live: std::collections::HashSet<usize> =
            caches_data.iter().map(|(i, _, _)| *i).collect();
        let missing: Vec<usize> = seq_ids
            .iter()
            .copied()
            .filter(|s| !live.contains(s))
            .collect();
        candle::bail!(
            "forward_wave: {} sequences requested but only {} have live slots \
             (missing/duplicated: {missing:?}) — the wave group named a sequence \
             the scheduler has since released",
            seq_ids.len(),
            caches_data.len(),
        );
    }
    let mut contexts: Vec<SequenceContext<'s>> = Vec::with_capacity(seq_ids.len());
    for ((_seq_idx, offset, caches), input) in caches_data.into_iter().zip(inputs.iter()) {
        contexts.push(SequenceContext {
            offset,
            kv_caches: caches,
            input_ids: input,
            input_len: input.dims().get(1).copied().unwrap_or(1),
        });
    }
    Ok(contexts)
}

/// **Between forwards:** fit the held ground's arenas to what recent forwards
/// took, so the buffers this forward takes inside itself — a slot-state slab
/// promoted at a 32-token boundary, a sampler pack growing — find room already
/// claimed (`docs/held_ground_elimination.md` §4). Before admission, so the
/// claims stand below the frontier the forward's tier is placed against.
pub(super) fn provision_held<S: WaveSweep + ?Sized>(model: &S) {
    if let Device::Cuda(cuda) = model.device() {
        provision_held_ground(cuda);
    }
}
