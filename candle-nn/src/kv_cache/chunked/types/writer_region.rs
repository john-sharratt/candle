//! The writer boundary of a sequence, and the compaction rewrite that relocates
//! its blocks without leaving the cached decode buffer stale.
//!
//! Chunks below the boundary are shared prefix history and are never extended.
//! Chunks at or above it are written by the decode kernel. Compaction rewrites
//! the gids and invalidates the cached buffer inside the same method, because a
//! count-preserving move would otherwise hand the next forward addresses that
//! now hold other data. A writer-owned chunk that carries a record is skipped
//! rather than rewritten, and it is reported so that a path breaking the
//! invariant announces itself. A sealed partial that the paged stores cannot
//! write moves the boundary past it.

use super::*;

impl SequenceState {
    #[inline]
    pub(crate) fn writer_start_idx(&self) -> usize {
        self.writer_start_idx
    }

    #[inline]
    pub(crate) fn set_writer_start_idx(&mut self, idx: usize) {
        self.writer_start_idx = idx;
    }

    /// The first chunk not yet handed to a gap. See `gap_cursor`.
    #[inline]
    pub(crate) fn gap_cursor(&self) -> usize {
        self.gap_cursor
    }

    /// Record that every chunk before `cursor` has been handed to a gap.
    #[inline]
    pub(crate) fn set_gap_cursor(&mut self, cursor: usize) {
        self.gap_cursor = cursor;
    }

    /// Move the writer boundary past every chunk in the writer region the
    /// paged stores cannot write, answering whether it moved.
    ///
    /// A chunk sealed into a quantized format is read-only whatever room it
    /// has left — `store_kv_chunk_arena` writes nothing into it — so above the
    /// boundary the first such chunk with room is taken for the writer and
    /// swallows the next write without a fault. Below the boundary it is what
    /// every sealed partial tail is: a gap the next write steps over.
    pub(crate) fn seal_unwritable_writer_region(&mut self) -> bool {
        let last = (self.writer_start_idx..self.chunks.len())
            .rev()
            .find(|&i| !self.chunks[i].takes_writes());
        match last {
            Some(i) => {
                self.set_writer_start_idx(i + 1);
                true
            }
            None => false,
        }
    }

    /// Rewrite every block's gids through a compaction sweep, and throw away the
    /// cached decode buffer if anything moved.
    ///
    /// Returns whether this slot held anything the pass relocated.
    ///
    /// **The invalidation is inside this method on purpose.** The cached buffer is
    /// reused whenever the chunk COUNT agrees (`sync_decode_gpu_chunks`), and a
    /// compaction changes gids without ever changing a count — so a caller that
    /// rewrote the block table and forgot to invalidate would hand the next forward
    /// pre-compaction band addresses. That does not fault, because every address in
    /// the reservation is mapped; it reads whatever now occupies the vacated slot.
    /// Making the rewrite carry its own invalidation is what stops that from being
    /// a step anyone can omit.
    ///
    /// **A record is replaced, never rewritten, and a WRITER-owned chunk that has one is
    /// skipped entirely.**
    ///
    /// The pass rewrites gids and never rewrites a record's bytes. A chunk whose gids moved
    /// is given a freshly minted record; a chunk whose record was copied lower is moved
    /// onto the copy (`Sweep::follow_record`). A record holds a clone of the gids it was
    /// serialized from, so whichever record a chunk ends up holding keeps the bands it
    /// names alive, and reading through it stays correct. See `compact_backings`.
    ///
    /// That argument needs the chunk to be *immutable*, which is what
    /// [`Self::writer_start_idx`] decides: chunks below it are Arc-shared prefix history
    /// and must not be extended, chunks at or above it are the writer's and are written
    /// by the decode kernel. A writer-owned chunk writes through its **gids** (the
    /// destination) while anything resolving its record would read the **source**, so the
    /// two would diverge on the next token. Every path that gives a `ChunkWindow` a
    /// record gives it to sealed prefix history, and `set_block_gids` clears `meta` on any
    /// gid mutation — so this combination should not arise. It is skipped rather than
    /// asserted because skipping is safe under every circumstance: the chunk keeps naming
    /// its source, which its record also names, and the pass simply reclaims nothing for
    /// it. The chunks it skipped come back in [`CompactionRewrite::stranded`], which the
    /// caller reports once per slot — that report is how a new path that breaks the
    /// invariant announces itself.
    ///
    /// Only the compaction pass sweeps, and there is no host compaction.
    #[cfg(feature = "cuda")]
    pub(in super::super) fn rewrite_for_compaction(
        &mut self,
        sweep: &mut super::super::compact_map::Sweep<'_>,
    ) -> candle::Result<CompactionRewrite> {
        let mut moved = false;
        let mut stranded: Option<StrandedRecords> = None;
        let writer_start = self.writer_start_idx;
        for (idx, cw) in self.chunks.iter_mut().enumerate() {
            if idx >= writer_start && cw.meta.is_some() {
                let s = stranded.get_or_insert(StrandedRecords {
                    first: idx,
                    last: idx,
                    count: 0,
                    writer_start,
                });
                s.last = idx;
                s.count += 1;
                continue;
            }
            if let Some(next) = sweep.rewrite_gids(&cw.gids)? {
                // A fresh record for the new bands, and only for a chunk that had one —
                // prefix history, by the check above. Installing it retires the old
                // record, which is what releases the source once the sweep ends. A
                // chunk that cannot have a fresh one stays whole on its source.
                let movable = sweep.remint(
                    &next,
                    &mut cw.meta,
                    RecordInputs {
                        k_pal: &cw.k_pal,
                        v_pal: &cw.v_pal,
                        k_scale: &cw.k_scale,
                        v_scale: &cw.v_scale,
                        k_fmt: &cw.k_fmt,
                        v_fmt: &cw.v_fmt,
                    },
                )?;
                if movable {
                    sweep.witness(&next);
                    cw.gids = next;
                    moved = true;
                }
            }
            // A relocated record moves the address this slot's cached decode buffer
            // holds as `kvheads_ptr`, so it counts as a move for the invalidation below.
            if sweep.follow_record(&mut cw.meta) {
                moved = true;
            }
        }
        if moved {
            self.invalidate_gpu_chunks();
        }
        Ok(CompactionRewrite { moved, stranded })
    }
}
