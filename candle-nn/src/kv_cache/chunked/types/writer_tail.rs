//! A snapshot of a slot's writer-owned chunks, taken before a rebuild and put
//! back after it.
//!
//! The stateless-slot rebuild truncates the slot, re-injects the projected
//! prefix, and restores the in-flight tail in one call. The snapshot holds its
//! chunks by RAII, so their bytes stay alive across the truncate and never move.
//! A snapshot with no tokens counts as empty, because restoring a bare empty
//! writer chunk after a rebuild would leave a write slice that no commit
//! described. The compaction result reports whether a pass moved anything in a
//! slot, and how many writer-owned records it left in place.

use super::*;

/// Opaque snapshot of a slot's writer-owned chunks, taken by
/// [`super::ChunkedKvBacking::split_off_writer_tail`] and restored by
/// [`super::ChunkedKvBacking::extend_writer_tail`].
///
/// Used by the stateless-slot rebuild path: the scheduler snapshots
/// the in-flight decode chunks before truncating the slot, re-injects
/// the projected prefix from substrate, and restores the tail in the
/// same call. RAII on the contained `ChunkGid`s keeps the underlying
/// arena chunks alive across the truncate — the bytes never move.
pub struct WriterTail {
    pub(in super::super) chunks: Vec<ChunkWindow>,
}

impl WriterTail {
    /// `true` when the snapshot holds no tokens — the common turn-boundary
    /// case where decode hasn't started yet.
    ///
    /// Tokens, not chunks: a slot can carry an empty writer chunk with nothing
    /// in it — the one a view is carved with so its first write has somewhere
    /// unshared to land. That chunk is no part of what the slot forwarded, so
    /// there is nothing of it to restore, and restoring it after a rebuild that
    /// placed its own writer leaves a stale empty chunk past the rebuilt
    /// writer region, which the decode then reads a write slice for that no
    /// commit ever described.
    pub fn is_empty(&self) -> bool {
        self.chunks.iter().all(|c| c.usage == 0)
    }

    /// Number of chunks in the snapshot.
    pub fn len(&self) -> usize {
        self.chunks.len()
    }
}

/// What one slot's compaction rewrite did.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CompactionRewrite {
    /// A chunk's gids or record moved, so the slot's decode buffer was thrown
    /// away.
    pub moved: bool,
    /// Writer-owned chunks that carry a record, left where they are — see
    /// [`SequenceState::rewrite_for_compaction`].
    pub stranded: Option<StrandedRecords>,
}

/// The writer-owned chunks of one slot that carry a record: the first and last
/// such block, how many, and the boundary they sit at or above.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StrandedRecords {
    pub first: usize,
    pub last: usize,
    pub count: usize,
    pub writer_start: usize,
}
