//! Rewrites a sealed payload's chunk identities through a compaction map and
//! returns the replacements, leaving the holder itself untouched.
//!
//! The caller keeps what it handed in and installs what comes back, so no holder
//! has to be found from ground it might not be on. Copy on write means nothing is
//! cloned until a chunk actually changes. A changed chunk that had a record is given
//! a freshly minted record rather than a rewritten one, because a record is shared
//! by every holder of its chunk. A host copy names no GPU arena that the pass moved,
//! so it is not walked at all.

use super::*;

/// Band allocations and records a sweep replaced — see [`Sweep::replaced`].
///
/// What decides whether a cached decode buffer is stale: a buffer pinning any of these
/// was serialised from ground the pass has since moved its holders off.
#[cfg(feature = "cuda")]
pub(in super::super) struct Replaced {
    pub(super) allocs: AHashSet<usize>,
    pub(super) records: AHashSet<i64>,
}

#[cfg(feature = "cuda")]
impl Replaced {
    /// Whether the sweep replaced nothing, so no buffer can name replaced ground.
    pub(in super::super) fn is_empty(&self) -> bool {
        self.allocs.is_empty() && self.records.is_empty()
    }

    /// Whether a cached buffer holding `pins` names anything the sweep replaced.
    ///
    /// Sound only while `self`'s sweep is alive, and while the buffer's pins are held —
    /// which they are, since they are read from the buffer that holds them.
    pub(in super::super) fn names_any(&self, pins: &[ChunkPin]) -> bool {
        pins.iter().any(|p| {
            self.allocs.contains(&p.bands_alloc_id())
                || p.record_raw().is_some_and(|r| self.records.contains(&r))
        })
    }
}

/// Rewrite the gids of `seqs` through the map, returning replacements.
///
/// `None` when nothing in `seqs` moved — the caller then keeps what it has rather
/// than installing an identical copy, which would invalidate device block tables
/// for no reason.
///
/// **Returns replacements; never edits a holder.** That is the shape
/// `quantize_sealed_in_place` and `migrate_sealed_to_cpu_batch_async` already use
/// to move KV thousands of times per run without once corrupting it: the caller
/// owns what it handed in and installs what comes back, so no holder has to be
/// *found*. The earlier relocation pass inverted this — it started from an arena
/// and tried to discover everyone pointing into it — and there is no index from a
/// gid back to its holders, which is why it could not be made correct.
///
/// A rewritten chunk is given a **freshly minted** `MetaGid` when the sweep has a minter
/// and it had a record to replace; no record is ever rewritten in place. See
/// [`compact_mint`](super::compact_mint).
pub fn rewrite_sealed(
    seqs: &[SealedSequence],
    sweep: &mut Sweep<'_>,
) -> candle::Result<Option<Vec<SealedSequence>>> {
    if sweep.map.is_empty() {
        return Ok(None);
    }
    // **Copy on write.** A pass moves a handful of chunks and the sweep visits every
    // chunk in the process, so cloning each one into a replacement that is then thrown
    // away was the whole cost of a sweep — a dozen refcount round trips per chunk across
    // every resident turn, ~120 ms a pass on a long-lived substrate. Nothing is cloned
    // until the first chunk that actually changes, and only its sequence and the ones
    // after it are rebuilt.
    //
    // **A host sequence is not walked.** A pass relocates GPU arenas only, and every
    // chunk of a sequence resolves to an arena at its `location`, so a warm copy names
    // nothing a pass can have moved — and the warm tier is most of what a long-lived
    // substrate holds, so it was most of every sweep. Skipping it is sound for the
    // same reason an unreached holder is: a holder the sweep does not rewrite keeps
    // its own ground alive and reads it correctly.
    let mut out: Option<Vec<SealedSequence>> = None;
    for (i, seq) in seqs.iter().enumerate() {
        let chunks = if seq.location == ArenaLocation::Gpu {
            rewrite_chunks(&seq.chunks, sweep)?
        } else {
            None
        };
        match (chunks, out.as_mut()) {
            (Some(chunks), Some(out)) => out.push(with_chunks(seq, chunks)),
            (Some(chunks), None) => {
                let mut fresh = Vec::with_capacity(seqs.len());
                fresh.extend_from_slice(&seqs[..i]);
                fresh.push(with_chunks(seq, chunks));
                out = Some(fresh);
            }
            (None, Some(out)) => out.push(seq.clone()),
            (None, None) => {}
        }
    }
    Ok(out)
}

/// `seq` holding `chunks` in place of its own.
pub(super) fn with_chunks(seq: &SealedSequence, chunks: Vec<SealedChunk>) -> SealedSequence {
    SealedSequence {
        chunks,
        token_count: seq.token_count,
        chunk_size: seq.chunk_size,
        location: seq.location,
    }
}

/// One sequence's chunks rewritten through the sweep, or `None` when none of them
/// changed. Clones only from the first chunk that changed — see [`rewrite_sealed`].
pub(super) fn rewrite_chunks(
    chunks: &[SealedChunk],
    sweep: &mut Sweep<'_>,
) -> candle::Result<Option<Vec<SealedChunk>>> {
    let mut out: Option<Vec<SealedChunk>> = None;
    for (i, chunk) in chunks.iter().enumerate() {
        match (rewrite_chunk(chunk, sweep)?, out.as_mut()) {
            (Some(c), Some(out)) => out.push(c),
            (Some(c), None) => {
                let mut fresh = Vec::with_capacity(chunks.len());
                fresh.extend_from_slice(&chunks[..i]);
                fresh.push(c);
                out = Some(fresh);
            }
            (None, Some(out)) => out.push(chunk.clone()),
            (None, None) => {}
        }
    }
    Ok(out)
}

/// One chunk's replacement, or `None` when the sweep leaves it as it is.
pub(super) fn rewrite_chunk(
    chunk: &SealedChunk,
    sweep: &mut Sweep<'_>,
) -> candle::Result<Option<SealedChunk>> {
    let mut changed: Option<SealedChunk> = None;
    if let Some(next) = sweep.rewrite_gids(&chunk.gids)? {
        let mut c = chunk.clone();
        // **A fresh record for the new bands, and only if this chunk had one.**
        // Installing it retires the old record — and with it, once the sweep ends, the
        // clone of the old gids the old record was holding — which is what lets the
        // source be reclaimed. A chunk with no record is addressed from its gids and
        // must not be given one; a chunk with one that cannot have a fresh one stays
        // whole on its source.
        #[cfg(feature = "cuda")]
        let movable = sweep.remint(
            &next,
            &mut c.meta,
            RecordInputs {
                k_pal: &c.k_pal,
                v_pal: &c.v_pal,
                k_scale: &c.k_scale,
                v_scale: &c.v_scale,
                k_fmt: &c.k_fmt,
                v_fmt: &c.v_fmt,
            },
        )?;
        #[cfg(not(feature = "cuda"))]
        let movable = true;
        if movable {
            sweep.witness(&next);
            c.gids = next;
            changed = Some(c);
        }
    }
    // The record itself may have been copied lower. After the band rewrite, so a record
    // just minted for new bands is never swapped for a copy of the old one.
    #[cfg(feature = "cuda")]
    {
        let meta = changed.as_ref().map_or(&chunk.meta, |c| &c.meta);
        if sweep.follows(meta) {
            let c = changed.get_or_insert_with(|| chunk.clone());
            sweep.follow_record(&mut c.meta);
        }
    }
    Ok(changed)
}
