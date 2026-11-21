//! Carries the new locations of relocated records, to apply to the index.
//!
//! Once the copies are durable, their locations are applied to the substrate
//! index under a brief write lock. A record whose index entry has moved since
//! the plan was made has been rewritten, so its copy is dead and is skipped.
//! A planned record the index still names in a target is reported, and its
//! segment is kept, because unlinking it would leave an indexed record
//! unreadable.

use super::*;

/// The relocated records' new locations, produced by
/// [`SubstratePersistence::execute_maintenance`] and applied to the substrate
/// index under a **brief** write lock ([`Self::apply_to_substrate`]).
pub struct MaintenanceResult {
    /// `(stream, chunk_index, old_loc, new_loc)`.
    pub(super) chunk_updates: Vec<(StreamKey, u64, ChunkLoc, ChunkLoc)>,
    /// `(stream, old_loc, new_loc)`.
    pub(super) token_updates: Vec<(StreamKey, RecordLoc, RecordLoc)>,
    pub(super) index_page_updates: Vec<(StreamKey, RecordLoc, RecordLoc)>,
    /// `(snapshot stream, old_loc, new_loc)`.
    pub(super) snapshot_updates: Vec<(StreamId, RecordLoc, RecordLoc)>,
    /// `(branch stream, old_loc, new_loc)`.
    pub(super) branch_checkpoint_updates: Vec<(StreamId, RecordLoc, RecordLoc)>,
}

impl MaintenanceResult {
    /// Repoint the substrate index at the relocated records, then refresh cold
    /// residence refs. **Supersession guard:** a record whose index entry no
    /// longer points at `old_loc` was rewritten since the plan was made, so its
    /// relocated copy is dead (reclaimed by a later pass) and skipped — the
    /// newer copy wins. One that still does is the record the copy was taken
    /// from, so the copy is applied under that record's identity, which it
    /// kept. Held under a brief substrate write lock only: the cold references
    /// refreshed are those of the streams whose chunks moved, not every
    /// residence's.
    ///
    /// Answers every record of `plan` the index still names where the plan
    /// found it, inside a target ([`Stranded`]): one the relocation filtered
    /// out and the index was never moved off. Unlinking its segment would
    /// leave it indexed but unreadable, so
    /// [`SubstratePersistence::finish_maintenance`] keeps that segment.
    pub fn apply_to_substrate(
        &self,
        plan: &MaintenancePlan,
        substrate: &mut Substrate,
    ) -> Stranded {
        let mut moved: HashSet<StreamId> = HashSet::new();
        for (key, idx, old, new) in &self.chunk_updates {
            let held = substrate.stream_at(*key).and_then(|s| {
                let id = s.ids.chunks.get(idx).copied().unwrap_or_default();
                s.chunks.get(idx).map(|loc| (*loc, id))
            });
            if let Some((loc, id)) = held {
                if loc == *old {
                    substrate.apply_chunk_loc(key.stream_id, *idx, *new, id);
                    moved.insert(key.stream_id);
                }
            }
        }
        for (key, old, new) in &self.token_updates {
            let held = substrate
                .stream_at(*key)
                .and_then(|s| s.tokens.map(|loc| (loc, s.ids.tokens)));
            if let Some((loc, id)) = held {
                if loc == *old {
                    substrate.apply_tokens_loc(key.stream_id, *new, id);
                }
            }
        }
        for (key, old, new) in &self.index_page_updates {
            // Same guard: a page rewritten since the plan keeps its newer copy.
            let held = substrate.stream_at(*key).and_then(|s| {
                s.index_page
                    .as_ref()
                    .and_then(IndexPage::stored)
                    .map(|loc| (loc, s.ids.index_page))
            });
            if let Some((loc, id)) = held {
                if loc == *old {
                    substrate.apply_index_page_loc(key.stream_id, *new, id);
                }
            }
        }
        for (sid, old, new) in &self.snapshot_updates {
            // Same supersession guard: a snapshot rewritten (a newer seal)
            // since the plan was made keeps its newer location; the relocated
            // copy of the old one is dead and reclaimed by a later pass.
            if substrate.recurrent_snapshot_loc(*sid) == Some(*old) {
                let id = substrate.snapshot_id(*sid);
                substrate.apply_snapshot_loc(*sid, *new, id);
            }
        }
        for (sid, old, new) in &self.branch_checkpoint_updates {
            if substrate.branch_checkpoint_loc(*sid) == Some(*old) {
                let id = substrate.branch_checkpoint_id(*sid);
                substrate.apply_branch_checkpoint_loc(*sid, *new, id);
            }
        }
        substrate.refresh_cold_refs_of(&moved);
        plan.stranded_in(substrate)
    }
}
