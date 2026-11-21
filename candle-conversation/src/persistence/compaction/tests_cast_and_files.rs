//! Tests that a compaction carries the character store and each conversation's
//! file events forward.
//!
//! Live characters are carried in id order, so the same inputs always produce
//! the same file. A tombstoned character keeps its record, because the acts it
//! committed still name it. A conversation's file events are carried only while
//! the conversation is live. A file store's events are carried even when no
//! conversation is registered on it.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::persistence::identity::RecordIdentity;

    use crate::persistence::log_file::{MemLog, SUPERBLOCK_SIZE};

    use crate::persistence::record::CustomObjectPayload;

    use crate::persistence::vfs::VfsEventId;

    use crate::substrate::file_store_key;

    use super::super::tests_support::tests::{empty_store, has_type, loc_at, record, NO_EXPIRY};

    #[test]
    fn collect_carries_the_live_cast_forward() {
        let (manifest, substrate) = empty_store();
        // What the persistence layer tracks after a walk: one location per
        // character, already resolved to its greatest stamp.
        let mut npc_locs = StampedLocs::new();
        npc_locs.offer(701, Some(loc_at(4_096, 256)), RecordIdentity::NONE);
        npc_locs.offer(700, Some(loc_at(8_192, 320)), RecordIdentity::NONE);

        let live = collect_live_records(
            &manifest,
            &substrate,
            &npc_locs,
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );
        let npcs: Vec<&CompactItem> = live
            .iter()
            .filter(|it| it.header().record_type == RecordType::Npc)
            .collect();

        assert_eq!(npcs.len(), 2, "both characters survive, neither duplicated");
        // Sorted by id despite the reverse insertion order, so the same inputs
        // always produce the same file.
        assert_eq!(npcs[0].header().stream_id, 700);
        assert_eq!(npcs[1].header().stream_id, 701);
        // Each stages the exact bytes the map named.
        assert_eq!(raw_loc(npcs[0]), (8_192, 320));
        assert_eq!(raw_loc(npcs[1]), (4_096, 256));
    }

    /// With no characters, nothing is emitted — the loop must not invent an
    /// empty record or disturb an existing store.
    #[test]
    fn collect_emits_nothing_for_an_empty_cast() {
        let (manifest, substrate) = empty_store();
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
        assert!(!has_type(&live, RecordType::Npc));
    }

    /// A deleted character keeps its record. The id must stay taken — acts it
    /// already committed still name it — so tombstoning is a superseding
    /// record, never a removal from the map.
    #[test]
    fn collect_carries_a_tombstoned_character_forward_too() {
        let (manifest, substrate) = empty_store();
        let mut npc_locs = StampedLocs::new();
        npc_locs.offer(900, Some(loc_at(4_096, 192)), RecordIdentity::NONE);

        let live = collect_live_records(
            &manifest,
            &substrate,
            &npc_locs,
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );
        assert!(
            live.iter().any(
                |it| it.header().record_type == RecordType::Npc && it.header().stream_id == 900
            ),
            "a tombstoned character is still a record",
        );
    }

    /// **A conversation's file events are carried while it lives, and only
    /// then**: a tombstoned, a distilled and an unregistered conversation's
    /// events are orphans and stay behind, and no tombstone is carried — what
    /// it killed is not carried either.
    #[test]
    fn collect_carries_only_live_conversations_file_events() {
        use crate::persistence::record::{DistillMode, DistillPayload};
        use crate::persistence::streams::{StreamDecl, TurnDecl};

        let decl = |timeline_id: u64| {
            StreamDecl::Turn(TurnDecl {
                timeline_id,
                turn_index: 0,
                turn_id_day: 0,
                turn_id_seq: 1,
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
            .encode()
        };
        let (live_tl, dead_tl, distilled_tl, never_tl) = (11u64, 12u64, 13u64, 14u64);
        let mut blob = Vec::new();
        for (sid, tl) in [(801u64, live_tl), (802, dead_tl), (803, distilled_tl)] {
            blob.extend_from_slice(&record(RecordType::StreamDecl, sid, 0, &decl(tl)));
        }
        blob.extend_from_slice(&record(
            RecordType::Tombstone,
            0,
            0,
            &TombstonePayload {
                timeline_id: dead_tl,
                turn_index: None,
                turn_writer: 0,
                reason: None,
            }
            .encode(),
        ));
        blob.extend_from_slice(&record(
            RecordType::Distilled,
            0,
            0,
            &DistillPayload {
                timeline_id: distilled_tl,
                mode: DistillMode::ProvenanceOnly,
            }
            .encode(),
        ));
        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();

        let ev = |seq| VfsEventId { origin: 0, seq };
        let mut vfs = VfsIndex::new();
        for tl in [live_tl, dead_tl, distilled_tl, never_tl] {
            vfs.record_event(tl, ev(0), loc_at(4_096 * tl, 128));
            vfs.record_event(tl, ev(1), loc_at(4_096 * tl + 1_024, 128));
        }
        vfs.record_tombstone(live_tl, ev(2), loc_at(90_000, 64), &[ev(0)]);

        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &vfs,
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );
        let events: Vec<(u64, u64, (u64, u64))> = live
            .iter()
            .filter(|it| it.header().record_type == RecordType::VfsEvent)
            .map(|it| (it.header().stream_id, it.header().chunk_index, raw_loc(it)))
            .collect();
        assert_eq!(events, vec![(live_tl, 1, (4_096 * live_tl + 1_024, 128))]);
        assert!(!has_type(&live, RecordType::VfsTombstone));
    }

    /// **A file store's events are carried though no conversation is
    /// registered on it** — declared by a record read back with the log, as
    /// every member reads it before its maintenance starts — and a declared
    /// timeline that is tombstoned, or an undeclared one, is left behind.
    #[test]
    fn collect_carries_a_declared_file_stores_events() {
        let (store_tl, dead_tl, undeclared_tl) = (21u64, 22u64, 23u64);
        let mut blob = Vec::new();
        for tl in [store_tl, dead_tl] {
            let declared = CustomObjectPayload::new(
                file_store_key(TimelineId::from_raw(tl).unwrap()),
                serde_json::Value::Null,
            );
            blob.extend_from_slice(&record(
                RecordType::CustomObject,
                declared.stream_id(),
                0,
                &declared.encode(),
            ));
        }
        blob.extend_from_slice(&record(
            RecordType::Tombstone,
            0,
            0,
            &TombstonePayload {
                timeline_id: dead_tl,
                turn_index: None,
                turn_writer: 0,
                reason: None,
            }
            .encode(),
        ));
        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();

        let mut vfs = VfsIndex::new();
        for tl in [store_tl, dead_tl, undeclared_tl] {
            vfs.record_event(
                tl,
                VfsEventId { origin: 0, seq: 0 },
                loc_at(4_096 * tl, 128),
            );
        }
        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &vfs,
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );
        let carried: Vec<u64> = live
            .iter()
            .filter(|it| it.header().record_type == RecordType::VfsEvent)
            .map(|it| it.header().stream_id)
            .collect();
        assert_eq!(carried, vec![store_tl]);
    }
}
