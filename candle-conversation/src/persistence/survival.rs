//! How each record type survives a log rewrite.
//!
//! A rewrite — full compaction ([`super::compaction`]) or incremental segment
//! maintenance ([`super::maintenance`]) — retires whole segments. Neither path
//! copies "whatever is left" in a retired segment: compaction *constructs* the
//! live set and maintenance drives a *worklist*, so a record class survives only
//! if some site explicitly carries it across. A class nobody carries is deleted,
//! permanently and silently — no test fails, no warning is logged, and
//! [`super::accounting`] does not even count the bytes as dead.
//!
//! That is how three classes were lost. `TurnCoupling` was carried by neither
//! path, so every tool round-trip's exchange grouping evaporated on the first
//! rewrite that touched its segment. `Npc` was carried by compaction and not by
//! maintenance, so incremental maintenance deleted characters that a full
//! compaction would have kept. Turn-scoped `Tombstone`s were likewise
//! compaction-only, and losing one *resurrects* the dead turn — maintenance
//! relocates the turn's content regardless, so the marker is lost while the
//! condemned content is carried forward.
//!
//! [`Survival`] closes that hole by making the question total. The match in
//! [`survival`] is exhaustive over [`RecordType`], so a new record type does not
//! compile until its author states how it survives; `compaction`'s
//! `every_record_type_survives_compaction` then builds a store holding one record
//! of every written type and asserts the rewrite honours what this file says.

use super::record::RecordType;

/// The mechanism by which one record class outlives a segment retirement.
///
/// Every [`RecordType`] maps to exactly one of these. The variant is a
/// *contract*: it names the site that must carry the class across, and the test
/// module below checks that site exists.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Survival {
    /// Carried across **verbatim** from wherever the record physically lives,
    /// via a location map (`npc_locs`, `snapshot_locs`, the substrate stream
    /// index, the manifest's singleton hints).
    ///
    /// Forced, not chosen, whenever the payload is not in this process's RAM to
    /// re-encode — an `Npc` belongs to the daemon's registry, a `Snapshot` to
    /// the log — and the natural choice when the payload is bulk (`Chunk`,
    /// `Tokens`) and re-encoding would cost more than a byte copy.
    Relocated,

    /// Re-encoded fresh from the substrate's in-RAM state on every rewrite. The
    /// substrate is authoritative, so the record on disk is a projection of it
    /// and the copy in a retired segment is not the only one.
    Resident,

    /// Derived data with no supersession key. Every copy is dropped and the
    /// writer rebuilds the chain in the new file.
    Regenerated,

    /// Never written, so never carried. [`RecordType::Unknown`] is the
    /// forward-compatibility sentinel for tags this build does not recognise;
    /// `encode_record` panics on one. A record that decodes to `Unknown` came
    /// from a newer build and is skipped by the walker — dropping it on rewrite
    /// is the format's stated behaviour, not a loss this crate can prevent.
    NeverWritten,
}

/// How `rt` survives a log rewrite.
///
/// **Exhaustive by construction.** Adding a variant to [`RecordType`] breaks
/// this match, which is the entire point: the compiler asks the one question
/// that was never asked of `TurnCoupling`, `Npc`, or the turn-scoped
/// `Tombstone`. Answer it here *and* at the site the answer names — the tests
/// below check the pair agrees.
pub fn survival(rt: RecordType) -> Survival {
    match rt {
        // ── Relocated verbatim ──────────────────────────────────────────────
        // Bulk payloads, tracked in the substrate stream index.
        RecordType::Chunk | RecordType::Tokens => Survival::Relocated,
        // Single-tail-per-key records whose payload never enters RAM, keyed in
        // the header so supersession is mechanical.
        RecordType::Snapshot | RecordType::BranchCheckpoint => Survival::Relocated,
        // Owned by the daemon's registry, not the substrate — nothing here can
        // re-encode a character, so the winner is carried byte-for-byte.
        RecordType::Npc => Survival::Relocated,
        // Workspace singletons, located through the manifest.
        RecordType::ModelSpec | RecordType::Template | RecordType::Tokenizer => Survival::Relocated,

        // ── Re-encoded from substrate RAM ───────────────────────────────────
        // Per-stream metadata.
        RecordType::StreamDecl
        | RecordType::Commit
        | RecordType::ProjectionEvents
        | RecordType::WideQSig
        | RecordType::TurnIndexPage => Survival::Resident,
        // Per-timeline metadata.
        RecordType::Label
        | RecordType::ConvState
        | RecordType::TreeMetadata
        | RecordType::DebugId
        | RecordType::Distilled => Survival::Resident,
        // Both scopes: `turn_index: None` kills a timeline, `Some` kills one
        // turn. BOTH must be re-emitted. Losing a turn-scoped marker does not
        // merely forget a deletion — the rewrite carries the turn's content
        // forward regardless, so the condemned turn returns on the next reload.
        RecordType::Tombstone => Survival::Resident,
        // The exchange grouping for a tool round-trip, held in
        // `Timeline::couplings` and re-emitted from `live_couplings`.
        RecordType::TurnCoupling => Survival::Resident,

        // ── Rebuilt by the writer ───────────────────────────────────────────
        RecordType::HeaderIndex => Survival::Regenerated,

        // ── Not ours to carry ───────────────────────────────────────────────
        RecordType::Unknown => Survival::NeverWritten,
    }
}

/// Every record type this build writes, in tag order — the domain the survival
/// contract has to cover. Excludes [`RecordType::Unknown`], which is never
/// written.
///
/// Kept as an explicit list rather than derived from a range so that a tag gap
/// (tag 8 is retired) does not silently become a hole in the tests below.
pub const WRITTEN_RECORD_TYPES: &[RecordType] = &[
    RecordType::ModelSpec,
    RecordType::Template,
    RecordType::StreamDecl,
    RecordType::Chunk,
    RecordType::Tokens,
    RecordType::Commit,
    RecordType::Tokenizer,
    RecordType::Label,
    RecordType::ConvState,
    RecordType::TreeMetadata,
    RecordType::DebugId,
    RecordType::Tombstone,
    RecordType::ProjectionEvents,
    RecordType::Distilled,
    RecordType::WideQSig,
    RecordType::HeaderIndex,
    RecordType::TurnCoupling,
    RecordType::Snapshot,
    RecordType::BranchCheckpoint,
    RecordType::Npc,
    RecordType::TurnIndexPage,
];

/// A per-record-type tally, for reporting what a store holds and what a rewrite
/// carried forward.
///
/// **Cheap enough to sit in the walk.** One array index and increment per
/// record — no allocation, no hashing, no formatting until something asks for
/// the summary. The load walk already visits every record for the dead-weight
/// accounting, so the census rides along at no measurable cost, and each rewrite
/// tallies a `Vec` it has just built in RAM.
///
/// It exists because the failure mode this module documents is *invisible by
/// construction*: a dropped class leaves no error, no warning, and no dead-byte
/// accounting, so the only way to see it is to count the records before and
/// after and notice a number that went to zero. Restarts print a census, every
/// rewrite prints what it carried, and the pair makes a drop legible instead of
/// something to be inferred later from a missing feature.
#[derive(Clone, Copy, Default)]
pub struct RecordCensus {
    /// Indexed by [`RecordType::tag`]. Sized past the largest live tag so a new
    /// type lands in the array rather than off the end; `Unknown`'s tag is its
    /// enum position and shares the last slot with nothing that matters.
    counts: [u64; 32],
}

impl RecordCensus {
    pub fn new() -> Self {
        Self::default()
    }

    /// Tally one record. Inlined and branchless past the bounds check.
    #[inline]
    pub fn record(&mut self, rt: RecordType) {
        let tag = rt.tag() as usize;
        if let Some(slot) = self.counts.get_mut(tag) {
            *slot += 1;
        }
    }

    /// Count for one type.
    pub fn get(&self, rt: RecordType) -> u64 {
        self.counts.get(rt.tag() as usize).copied().unwrap_or(0)
    }

    pub fn total(&self) -> u64 {
        self.counts.iter().sum()
    }

    /// `"chunk=1024 tokens=64 turn_coupling=12"` — non-zero types only, in tag
    /// order, so a line stays short and two lines diff by eye. Built on demand,
    /// never in the walk.
    pub fn summary(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        for &rt in WRITTEN_RECORD_TYPES {
            let n = self.get(rt);
            if n > 0 {
                parts.push(format!("{}={n}", type_label(rt)));
            }
        }
        if parts.is_empty() {
            "empty".to_string()
        } else {
            parts.join(" ")
        }
    }

    /// The types this census has none of that `other` has — what a rewrite
    /// dropped, named. Empty is the healthy answer.
    pub fn types_lost_against(&self, other: &RecordCensus) -> Vec<&'static str> {
        WRITTEN_RECORD_TYPES
            .iter()
            .filter(|&&rt| other.get(rt) > 0 && self.get(rt) == 0)
            .map(|&rt| type_label(rt))
            .collect()
    }
}

/// Short snake_case name for a record type — the census's column heading.
///
/// Spelled out rather than `{rt:?}`, so the log line is stable if the enum's
/// `Debug` ever changes and greppable without matching Rust identifiers.
pub fn type_label(rt: RecordType) -> &'static str {
    match rt {
        RecordType::ModelSpec => "model_spec",
        RecordType::Template => "template",
        RecordType::StreamDecl => "stream_decl",
        RecordType::Chunk => "chunk",
        RecordType::Tokens => "tokens",
        RecordType::Commit => "commit",
        RecordType::Tokenizer => "tokenizer",
        RecordType::Label => "label",
        RecordType::ConvState => "conv_state",
        RecordType::TreeMetadata => "tree_metadata",
        RecordType::DebugId => "debug_id",
        RecordType::Tombstone => "tombstone",
        RecordType::ProjectionEvents => "projection_events",
        RecordType::Distilled => "distilled",
        RecordType::WideQSig => "wide_qsig",
        RecordType::HeaderIndex => "header_index",
        RecordType::TurnCoupling => "turn_coupling",
        RecordType::Snapshot => "snapshot",
        RecordType::BranchCheckpoint => "branch_checkpoint",
        RecordType::Npc => "npc",
        RecordType::TurnIndexPage => "turn_index_page",
        RecordType::Unknown => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every written record type has a survival contract that is not
    /// `NeverWritten` — i.e. some rewrite site is on the hook for it.
    #[test]
    fn every_written_record_type_has_a_carrier() {
        for &rt in WRITTEN_RECORD_TYPES {
            assert_ne!(
                survival(rt),
                Survival::NeverWritten,
                "{rt:?} is written to the log but names no mechanism that carries it \
                 across a rewrite — it would be deleted silently on the first \
                 compaction or segment maintenance pass that touched its segment",
            );
        }
    }

    /// `WRITTEN_RECORD_TYPES` really is every writable tag. Guards against a new
    /// record type being added to the enum and to `survival` while the list the
    /// cross-checks iterate stays stale — which would leave the new type
    /// untested by every test that walks this list.
    #[test]
    fn written_record_types_covers_every_writable_tag() {
        for tag in 0u8..=u8::MAX {
            let rt = RecordType::from_tag(tag);
            if rt == RecordType::Unknown {
                continue;
            }
            assert!(
                WRITTEN_RECORD_TYPES.contains(&rt),
                "tag {tag} decodes to {rt:?}, which is missing from \
                 WRITTEN_RECORD_TYPES — add it there and give it a survival arm",
            );
        }
    }

    /// The list has no duplicates and its tags are unique, so a copy-paste slip
    /// cannot make one type stand in for another in the cross-checks.
    #[test]
    fn written_record_types_are_distinct() {
        let mut tags: Vec<u8> = WRITTEN_RECORD_TYPES.iter().map(|rt| rt.tag()).collect();
        let before = tags.len();
        tags.sort_unstable();
        tags.dedup();
        assert_eq!(
            before,
            tags.len(),
            "duplicate entry in WRITTEN_RECORD_TYPES"
        );
    }

    /// `Unknown` is the one type with no carrier, and it must stay that way: it
    /// is the sentinel for a tag from a newer build, and `encode_record` panics
    /// rather than write one.
    #[test]
    fn unknown_is_never_written() {
        assert_eq!(survival(RecordType::Unknown), Survival::NeverWritten);
    }

    /// Every written type has a slot in the census array. A tag past the end
    /// would be counted as nothing and read as "this store holds none", which
    /// is precisely the false negative the census exists to rule out.
    #[test]
    fn every_written_type_fits_the_census() {
        for &rt in WRITTEN_RECORD_TYPES {
            let mut c = RecordCensus::new();
            c.record(rt);
            assert_eq!(
                c.get(rt),
                1,
                "{} (tag {}) fell outside the census array",
                type_label(rt),
                rt.tag(),
            );
            assert_eq!(c.total(), 1, "{} leaked into another slot", type_label(rt));
        }
    }

    /// Types share no slot — otherwise one class going to zero could be masked
    /// by another's count sitting in the same cell.
    #[test]
    fn census_slots_are_distinct_per_type() {
        let mut c = RecordCensus::new();
        for &rt in WRITTEN_RECORD_TYPES {
            c.record(rt);
        }
        for &rt in WRITTEN_RECORD_TYPES {
            assert_eq!(c.get(rt), 1, "{} shares a slot", type_label(rt));
        }
        assert_eq!(c.total(), WRITTEN_RECORD_TYPES.len() as u64);
    }

    /// The drop report names exactly the classes that went to zero — not one a
    /// rewrite legitimately holds fewer of, and not one neither side has.
    #[test]
    fn types_lost_names_only_what_went_to_zero() {
        let mut before = RecordCensus::new();
        before.record(RecordType::Chunk);
        before.record(RecordType::Chunk);
        before.record(RecordType::TurnCoupling);
        before.record(RecordType::Npc);

        let mut after = RecordCensus::new();
        after.record(RecordType::Chunk); // fewer, but still carried
        after.record(RecordType::Npc);

        let lost = after.types_lost_against(&before);
        assert_eq!(lost, vec!["turn_coupling"]);
        // Nothing is "lost" against an empty baseline.
        assert!(after.types_lost_against(&RecordCensus::new()).is_empty());
    }

    /// The summary lists only non-zero types, so a line stays short enough to
    /// read and to diff against the previous restart's by eye.
    #[test]
    fn summary_lists_only_present_types() {
        let mut c = RecordCensus::new();
        assert_eq!(c.summary(), "empty");
        c.record(RecordType::Chunk);
        c.record(RecordType::Chunk);
        c.record(RecordType::TurnCoupling);
        assert_eq!(c.summary(), "chunk=2 turn_coupling=1");
    }

    /// Labels are unique — a duplicate would make two classes indistinguishable
    /// in the one place a human reads this.
    #[test]
    fn type_labels_are_distinct() {
        let mut seen: Vec<&str> = WRITTEN_RECORD_TYPES
            .iter()
            .map(|&rt| type_label(rt))
            .collect();
        let before = seen.len();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(before, seen.len(), "duplicate label in `type_label`");
    }
}
