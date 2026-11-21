//! Defines what a stream sheds, so liveness counting matches what is carried.
//!
//! The planner and the compactor must classify each stream the same way, or a
//! segment reads as live weight that is never carried, or the reverse. A
//! whole-timeline tombstone sheds the stream unless it is distilled. A turn
//! tombstone sheds that turn's chunks, tokens and signature, while its
//! declaration stays as a placeholder. A section generation needs no flag,
//! because the tombstone that ended it already removed its records from the
//! index.

use super::*;

/// What a stream sheds, matching [`super::compaction::collect_live_records`].
///
/// # Turn tombstones were missing from this, and that is why nothing compacted
///
/// A stream can be dead two ways and this only knew one of them. A whole
/// timeline tombstone (a day rolling over, a character deleted) was classified;
/// a **turn-scoped** tombstone — what `retention` writes when a conversation's
/// tail falls past `keep_turns` — was not.
///
/// The compactor always knew: `collect_live_records` reads
/// `substrate.tombstoned_turns()` and sheds the turn's chunks, tokens and sig
/// while keeping its `StreamDecl` as a placeholder. But the two things that
/// decide *whether the compactor ever runs* — `segment_liveness`, and the
/// relocation planner — went through here, so every retired turn was still
/// counted as live weight.
///
/// The consequence was total. A live daemon retired ~950 of every 1000 turns
/// exactly as configured, and the dead-byte ratio it was measured against
/// stayed at **0.0%**, so the trigger never fired once. The log reached 132 GB
/// and stopped only when the disk did.
#[derive(Clone, Copy)]
pub(super) struct Shed {
    /// The whole timeline is gone — the stream goes wholesale, unless it is
    /// also distilled (the provenance corpus). A section generation a
    /// tombstone ended needs no flag: the tombstone took its records out of
    /// the index (`Substrate::end_section`), so nothing here meets them.
    pub(super) timeline_dead: bool,
    /// This one turn was retired from a timeline that is still live.
    pub(super) turn_dead: bool,
    pub(super) distill: Option<DistillMode>,
}

impl Shed {
    /// The three gates, spelled exactly as `collect_live_records` spells them
    /// so the planner and the compactor cannot drift apart.
    pub(super) fn keep_chunks(&self) -> bool {
        !self.turn_dead && self.distill.is_none()
    }
    pub(super) fn keep_tokens(&self) -> bool {
        !self.turn_dead && self.distill != Some(DistillMode::ProvenanceOnly)
    }
    pub(super) fn keep_sig(&self) -> bool {
        !self.turn_dead && self.distill != Some(DistillMode::TextOnly)
    }
    /// Whether the stream is skipped outright rather than shed piecemeal.
    pub(super) fn dropped_wholesale(&self) -> bool {
        self.timeline_dead && self.distill.is_none()
    }
}

/// What `key`'s stream `entry` sheds.
pub(super) fn classify(
    substrate: &Substrate,
    key: StreamKey,
    entry: &StreamRuntime,
    tombstoned: &HashSet<u64>,
    distilled: &HashMap<u64, DistillMode>,
) -> Shed {
    match &entry.decl {
        Some(StreamDecl::Turn(t)) => Shed {
            timeline_dead: tombstoned.contains(&t.timeline_id),
            turn_dead: substrate.is_turn_stream_retired(key, t),
            distill: distilled.get(&t.timeline_id).copied(),
        },
        Some(StreamDecl::PromptSection(_)) | None => Shed {
            timeline_dead: false,
            turn_dead: false,
            distill: None,
        },
    }
}
