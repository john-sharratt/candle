//! Tests that a compaction keeps tool-round-trip couplings and the turn-scoped
//! and section tombstones, and drops the couplings of a dead timeline.
//!
//! A coupling is keyed by payload and appears in no location map, so
//! re-emitting it from memory is the only thing that carries it forward. Without
//! that, every exchange would come apart into separate turns, and nothing would
//! report the loss. Turn-scoped and section tombstones are carried so that a
//! reload restores a dead turn as an empty hole and keeps a dropped section
//! dropped. A tombstoned timeline's couplings go with it.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::persistence::log_file::{MemLog, SUPERBLOCK_SIZE};

    use crate::persistence::record::SectionGeneration;

    use super::super::tests_support::tests::{has_synth, has_type, record, NO_EXPIRY};

    /// A tool round-trip's coupling survives compaction.
    ///
    /// This is the gate whose absence let every coupling in the store be
    /// deleted. `TurnCoupling` is payload-keyed, so `accounting` never credits a
    /// copy as dead and no location map relocates it: re-emitting from the
    /// substrate's live set is the *only* thing carrying it. And the loss is
    /// invisible — a timeline with no couplings is a legitimate state (nothing
    /// there ever called a tool), so the store reads as intact while every
    /// exchange has quietly come apart into independent turns.
    #[test]
    fn collect_carries_turn_couplings_forward() {
        use crate::persistence::streams::{StreamDecl, TurnDecl};
        let tl = 77u64;
        let decl = |turn_index: u32, seq: u32| {
            StreamDecl::Turn(TurnDecl {
                timeline_id: tl,
                turn_index,
                turn_id_day: 0,
                turn_id_seq: seq,
                role: 2,
                block_start: 0,
                block_end: 2,
                layer_id: 1,
                group_id: 1,
                anchored_prefix: Vec::new(),
                view: Vec::new(),
                segments: Vec::new(),
                tags: Vec::new(),
            })
        };
        // A three-turn repo_map chain (list → read → summarise): the first two
        // couple, joining all three into one exchange.
        let mut blob = Vec::new();
        for (i, sid) in [(0u32, 600u64), (1, 601), (2, 602)] {
            blob.extend_from_slice(&record(
                RecordType::StreamDecl,
                sid,
                0,
                &decl(i, sid as u32).encode(),
            ));
        }
        for from_turn in [0u32, 1] {
            blob.extend_from_slice(&record(
                RecordType::TurnCoupling,
                0,
                0,
                &TurnCouplingPayload {
                    timeline_id: tl,
                    from_turn,
                }
                .encode(),
            ));
        }

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        // Precondition: the walk really installed them, so a failure below is
        // compaction dropping them rather than replay never loading them.
        assert_eq!(
            substrate
                .live_couplings()
                .into_iter()
                .map(|(t, from, _)| (t, from))
                .collect::<Vec<_>>(),
            vec![(tl, 0), (tl, 1)],
            "replay must install both couplings",
        );

        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );
        for from_turn in [0u32, 1] {
            assert!(
                has_synth(
                    &live,
                    RecordType::TurnCoupling,
                    &TurnCouplingPayload {
                        timeline_id: tl,
                        from_turn,
                    }
                    .encode(),
                ),
                "coupling {from_turn} must survive compaction — without it the \
                 call turn and its response become independent exchanges",
            );
        }
    }

    /// A tombstoned timeline's couplings go with it. They describe turns that no
    /// longer exist, so carrying them would make the marker set immortal.
    #[test]
    fn collect_drops_couplings_of_a_tombstoned_timeline() {
        let tl = 88u64;
        let mut blob = Vec::new();
        blob.extend_from_slice(&record(
            RecordType::TurnCoupling,
            0,
            0,
            &TurnCouplingPayload {
                timeline_id: tl,
                from_turn: 0,
            }
            .encode(),
        ));
        blob.extend_from_slice(&record(
            RecordType::Tombstone,
            0,
            0,
            &TombstonePayload {
                timeline_id: tl,
                turn_index: None,
                turn_writer: 0,
                reason: None,
            }
            .encode(),
        ));

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );
        assert!(!has_type(&live, RecordType::TurnCoupling));
    }

    /// Turn-scoped tombstones survive. The reload's placeholder logic keys off
    /// the marker to restore the dead turn as an empty hole; without it the turn
    /// reads as content-declaring with no content.
    #[test]
    fn collect_carries_turn_scoped_tombstones_forward() {
        let tl = 99u64;
        let mut blob = Vec::new();
        blob.extend_from_slice(&record(
            RecordType::Tombstone,
            0,
            0,
            &TombstonePayload {
                timeline_id: tl,
                turn_index: Some(3),
                turn_writer: 0,
                reason: None,
            }
            .encode(),
        ));

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );
        assert!(
            has_synth(
                &live,
                RecordType::Tombstone,
                &TombstonePayload {
                    timeline_id: tl,
                    turn_index: Some(3),
                    turn_writer: 0,
                    reason: None,
                }
                .encode(),
            ),
            "a turn-scoped tombstone must survive, or the dead turn returns",
        );
    }

    /// A section tombstone survives compaction — the section counterpart of
    /// [`collect_carries_turn_scoped_tombstones_forward`]. Losing it would
    /// resurrect the corrupted chunks the reactive repair tombstoned.
    #[test]
    fn collect_carries_section_tombstones_forward() {
        let stream_id = 55_001u64;
        let generation = SectionGeneration { writer: 4, oseq: 9 };
        let mut blob = Vec::new();
        blob.extend_from_slice(&record(
            RecordType::SectionTombstone,
            stream_id,
            0,
            &SectionTombstonePayload {
                generation: Some(generation),
                reason: Some(
                    "chunk 4: (offset 0, 32 tokens) on layer(s) [0]; (offset 0, 0 \
                              tokens) on layer(s) [1..47]"
                        .to_string(),
                ),
            }
            .encode(),
        ));

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        assert!(
            substrate.is_section_tombstoned(StreamId(stream_id)),
            "replay must mark the stream tombstoned",
        );

        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );
        // The reason is diagnostic only and is not re-derived at compaction
        // time — the same rule the turn-scoped tombstone's re-emit follows —
        // so only the marker's presence (not its original reason text) is
        // pinned here.
        assert!(
            has_synth(
                &live,
                RecordType::SectionTombstone,
                &SectionTombstonePayload {
                    generation: Some(generation),
                    reason: None,
                }
                .encode(),
            ),
            "a section tombstone must survive, or the corrupted chunks return",
        );
    }
}
