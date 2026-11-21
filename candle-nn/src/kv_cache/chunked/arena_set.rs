//! The arenas a sealed payload's chunks and records live in — what lets a
//! compaction's sweep pass by a payload it can have moved nothing of.
//!
//! A sweep asks of every chunk of every holder whether any of its gids moved
//! ([`super::compact_map::Sweep::rewrite_gids`]) and whether its record was
//! copied lower (`Sweep::follows`). Both answers are `no` unless an arena the
//! pass touched holds one of them: a moved gid is in an arena that gave up a
//! chunk, a moved record in one that gave up a record. A payload whose arenas
//! the pass did not touch is therefore passed by without a chunk read — the
//! same answer [`super::compact_map::rewrite_sealed`] gives after reading every
//! one. On BlackWorld (2026-11-21) that read was 50–80 ms of each pass's sweep,
//! the substrate's hot residences walked chunk by chunk whatever the pass
//! moved.
//!
//! A payload never changes once installed, so its set is taken once
//! ([`ArenaSet::of`]) and kept with it.

use super::types::{SealedSequence, GID_STRIDE};
use crate::kv_cache::ArenaLocation;

/// See the module documentation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ArenaSet {
    /// Arenas the chunks' gids are in, ascending and distinct.
    chunks: Box<[u32]>,
    /// Arenas the chunks' records are in, ascending and distinct.
    records: Box<[u32]>,
}

impl ArenaSet {
    /// The arenas of `seqs`' device-located chunks and records. A host
    /// sequence is left out: no compaction moves anything of it.
    pub fn of(seqs: &[SealedSequence]) -> Self {
        let mut chunks: Vec<u32> = Vec::new();
        let mut records: Vec<u32> = Vec::new();
        for seq in seqs.iter().filter(|s| s.location == ArenaLocation::Gpu) {
            for chunk in &seq.chunks {
                let mut last: Option<u32> = None;
                for g in chunk.gids.as_slice() {
                    if let Some(a) = arena_of(g.raw()) {
                        if last != Some(a) {
                            chunks.push(a);
                            last = Some(a);
                        }
                    }
                }
                if let Some(a) = chunk.meta.as_ref().and_then(|m| arena_of(m.raw())) {
                    records.push(a);
                }
            }
        }
        chunks.sort_unstable();
        chunks.dedup();
        records.sort_unstable();
        records.dedup();
        Self {
            chunks: chunks.into(),
            records: records.into(),
        }
    }

    /// Arenas the chunks' gids are in.
    pub fn chunk_arenas(&self) -> &[u32] {
        &self.chunks
    }

    /// Arenas the chunks' records are in.
    pub fn record_arenas(&self) -> &[u32] {
        &self.records
    }
}

/// The arena a raw id names, or `None` for a sentinel.
fn arena_of(raw: i64) -> Option<u32> {
    (raw >= 0).then_some((raw.max(0) as usize / GID_STRIDE) as u32)
}
