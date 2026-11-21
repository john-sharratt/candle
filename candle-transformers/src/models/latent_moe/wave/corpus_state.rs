//! Holds each sequence's streaming compressor and gallery state, and appends a
//! wave's pooled groups into it.
//!
//! A wave pools every slot's completed compressor groups in one launch, and
//! `PooledGroups` keeps that block whole so the gallery append can scatter each
//! slot's run from it. `append_pooled` appends a sequence's groups back to back
//! in slot order, which keeps a speculative verify block that lands several
//! slots on one gallery in order. `WindowRingLayer` records one layer's resident
//! sliding window for persistence without an absolute position, because the
//! window is always the tail of the turn. `WaveCorpusSnap` captures every member's
//! corpus state before a sweep, so a failed sweep can be undone.

use super::*;

/// A wave's pooled compressor entries: ONE `[ΣG, d]` block holding every slot's
/// completed groups in slot order, plus where each slot's run sits in it.
///
/// The block is kept WHOLE rather than split into per-slot tensors because the
/// gallery append consumes it whole — `append_batch_all` scatters the slots'
/// runs to their galleries from this one source (hot-path invariant 2b).
pub(super) struct PooledGroups {
    /// `[ΣG, d]`, contiguous, slots in order.
    pub(super) rows: Tensor,
    /// Per slot: `(first row, rows)`, or `None` if it completed no group.
    pub(super) spans: Vec<Option<(usize, usize)>>,
}

impl PooledGroups {
    /// Total pooled groups across the wave.
    pub(super) fn total(&self) -> usize {
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
pub(super) fn pool_across_seqs(
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
pub(super) fn append_pooled(
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
/// persistence (Artifact A of docs/deepseek/deepseek_turn_seal_persistence.md).
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
pub(super) struct SeqEntry {
    /// Tokens this state has absorbed (compressor pushes). A prefill starting
    /// below this (or at 0) means the slot was freed and reused — reset.
    pub(super) absorbed: usize,
    pub(super) layers: Vec<KernelLayerSeqState>,
}

/// One wave member's pre-wave streaming-corpus state — the wave-failure twin
/// of [`VerifySnapshot`]. Captured by `sweep` for every sequence before the
/// layer loop can advance anything, consumed only when the sweep fails: the
/// galleries and streaming compressors advance IN PLACE per layer, so an
/// error mid-sweep otherwise leaves early layers holding entries later layers
/// never produced. `state == None` records that the sequence had no entry at
/// all (a fresh prefill member), so the restore removes what the failed wave
/// created rather than zeroing it.
pub(super) struct WaveCorpusSnap {
    pub(super) seq: usize,
    /// `(absorbed, per-layer (comp, icomp, gallery_len))`; `None` ⇒ no entry.
    #[allow(clippy::type_complexity)]
    pub(super) state: Option<(
        usize,
        Vec<(
            Option<super::super::compressor::CompressorState>,
            Option<super::super::compressor::CompressorState>,
            usize,
        )>,
    )>,
}
