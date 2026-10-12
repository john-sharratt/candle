//! **Tombstone expiry** (`docs/replicated_substrate.md` §6.8, §14.1).
//!
//! A timeline tombstone is kept by every compaction until a complete
//! compaction has finished after it **and** it is older than the retention;
//! then it shrinks to its entry in the `Deleted` record, and the timeline stays
//! dead for good. A keyed value's tombstone does the same, its entry keeping
//! the stamp a later write must exceed.

use std::time::Duration;

use super::blob::BlobIndex;
use super::compaction::{collect_live_records, CompactItem, Housekeeping};
use super::deleted::{DeletedKey, DeletedPayload, DeletedTimeline};
use super::identity::{MarkerIds, RecordIdentity, COUNTER_BITS};
use super::log_file::{MemLog, SUPERBLOCK_SIZE};
use super::maintenance::gather_resident_set;
use super::manifest::{encode_label_payload, Manifest};
use super::record::{
    encode_record, CustomObjectPayload, RecordHeader, RecordType, TombstonePayload,
};
use super::stamped_locs::StampedLocs;
use super::streams::{StreamDecl, TurnDecl};
use super::tombstone_expiry::TombstoneExpiry;
use super::vfs::VfsIndex;
use super::SubstratePersistence;
use crate::projection::{GroupId, LayerId, TimelineId};
use crate::substrate::Substrate;

const DAY_MS: u64 = 24 * 60 * 60 * 1000;
const A: u64 = 0xA;
const DEAD: u64 = 0x2000_0010;
const LIVE: u64 = 0x2000_0011;

fn at(oseq: u64, ms: u64) -> RecordIdentity {
    RecordIdentity {
        origin: A,
        oseq,
        gen: 0,
        epoch: 0,
        hlc: ms << COUNTER_BITS,
    }
}

fn record(rt: RecordType, stream_id: u64, payload: &[u8], id: RecordIdentity) -> Vec<u8> {
    encode_record(
        &RecordHeader {
            record_type: rt,
            payload_len: payload.len() as u64,
            stream_id,
            ..RecordHeader::NONE
        }
        .with_identity(id),
        payload,
    )
}

fn turn(timeline_id: u64) -> StreamDecl {
    StreamDecl::Turn(TurnDecl {
        timeline_id,
        turn_index: 0,
        turn_id_day: 0,
        turn_id_seq: 1,
        role: 1,
        block_start: 0,
        block_end: 0,
        layer_id: 1,
        group_id: 1,
        anchored_prefix: Vec::new(),
        view: Vec::new(),
        segments: Vec::new(),
        tags: Vec::new(),
    })
}

fn tombstone(timeline_id: u64) -> Vec<u8> {
    TombstonePayload {
        timeline_id,
        turn_index: None,
        turn_writer: 0,
        reason: None,
    }
    .encode()
}

/// A store as its walk rebuilds it: the manifest, the substrate, and the
/// marker identities a rewrite carries.
struct Store {
    manifest: Manifest,
    substrate: Substrate,
    markers: MarkerIds,
}

fn open(records: &[(RecordType, u64, Vec<u8>, RecordIdentity)]) -> Store {
    let mut blob = Vec::new();
    let mut markers = MarkerIds::new();
    for (rt, sid, payload, id) in records {
        blob.extend_from_slice(&record(*rt, *sid, payload, *id));
        markers.record(*rt, *sid, payload, *id);
    }
    let mut mem = MemLog::with_records(&blob);
    let (manifest, substrate, _) =
        Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
    Store {
        manifest,
        substrate,
        markers,
    }
}

/// Compact `store` at `now_ms` with a 30-day retention, returning what it
/// writes.
fn compact(store: &Store, now_ms: u64) -> Vec<(RecordType, u64, Vec<u8>, RecordIdentity)> {
    collect_live_records(
        &store.manifest,
        &store.substrate,
        &StampedLocs::new(),
        &VfsIndex::new(),
        &BlobIndex::new(),
        &store.markers,
        0,
        TombstoneExpiry {
            now_ms,
            retention: Duration::from_secs(30 * 24 * 60 * 60),
        },
        &Housekeeping::default(),
    )
    .into_iter()
    .map(|item| match item {
        CompactItem::Synth { header, payload } => (
            header.record_type,
            header.stream_id,
            payload.to_vec(),
            header.identity(),
        ),
        CompactItem::Raw { .. } => panic!("this store holds no read-back records"),
    })
    .collect()
}

fn of_type(written: &[(RecordType, u64, Vec<u8>, RecordIdentity)], rt: RecordType) -> Vec<Vec<u8>> {
    written
        .iter()
        .filter(|(t, ..)| *t == rt)
        .map(|(_, _, p, _)| p.clone())
        .collect()
}

/// The store's first state: a dead timeline's turn, label and tombstone
/// (written on day 10), and a live timeline beside it.
fn initial() -> Vec<(RecordType, u64, Vec<u8>, RecordIdentity)> {
    let dead = turn(DEAD);
    let live = turn(LIVE);
    vec![
        (
            RecordType::StreamDecl,
            dead.stream_id().0,
            dead.encode(),
            at(1, 9 * DAY_MS),
        ),
        (
            RecordType::Label,
            0,
            encode_label_payload(DEAD, "c-dead", "Dead", &Default::default()),
            at(2, 9 * DAY_MS),
        ),
        (
            RecordType::StreamDecl,
            live.stream_id().0,
            live.encode(),
            at(3, 9 * DAY_MS),
        ),
        (
            RecordType::Tombstone,
            0,
            tombstone(DEAD),
            at(7, 10 * DAY_MS),
        ),
    ]
}

/// **Neither condition alone shrinks a tombstone, and both do.** The first
/// compaction finds it old but not yet swept, and keeps it, recording that it
/// has now run past it; a later one finds it swept but young, and keeps it;
/// one that finds it both shrinks it to its entry — the identity of the
/// tombstone it replaced — and keeps the timeline dead.
#[test]
fn a_tombstone_shrinks_only_once_swept_and_old() {
    let store = open(&initial());
    let first = compact(&store, 100 * DAY_MS);
    assert_eq!(
        of_type(&first, RecordType::Tombstone),
        vec![tombstone(DEAD)]
    );
    assert_eq!(
        first
            .iter()
            .find(|(t, ..)| *t == RecordType::Tombstone)
            .unwrap()
            .3,
        at(7, 10 * DAY_MS),
        "the tombstone keeps its identity"
    );
    assert_eq!(
        of_type(&first, RecordType::Deleted),
        vec![DeletedPayload {
            swept: vec![DEAD],
            ..DeletedPayload::default()
        }
        .encode()]
    );
    assert!(
        of_type(&first, RecordType::Label).is_empty(),
        "the dead timeline's records are gone"
    );

    let swept = open(&first);
    assert!(swept
        .substrate
        .is_tombstone_swept(TimelineId::from_raw(DEAD).unwrap()));
    let young = compact(&swept, 11 * DAY_MS);
    assert_eq!(
        of_type(&young, RecordType::Tombstone),
        vec![tombstone(DEAD)],
        "swept but young: kept"
    );

    let expired = compact(&swept, 100 * DAY_MS);
    assert!(of_type(&expired, RecordType::Tombstone).is_empty());
    let entry = DeletedPayload {
        timelines: vec![DeletedTimeline {
            timeline: DEAD,
            origin: A,
            oseq: 7,
        }],
        ..DeletedPayload::default()
    }
    .encode();
    assert_eq!(of_type(&expired, RecordType::Deleted), vec![entry.clone()]);

    // The entry survives every later compaction byte for byte.
    let after = open(&expired);
    let again = compact(&after, 400 * DAY_MS);
    assert_eq!(of_type(&again, RecordType::Deleted), vec![entry]);
    assert!(of_type(&again, RecordType::Tombstone).is_empty());
}

/// **A deleted timeline stays dead, however late its records arrive.** A year
/// after expiry, a record for it — in every order with the `Deleted` record —
/// leaves the observable state as it was, registering it leaves it dead, and
/// maintenance carries the entry rather than a tombstone.
#[test]
fn a_deleted_timeline_stays_dead_whatever_arrives() {
    let swept = open(&compact(&open(&initial()), 100 * DAY_MS));
    let expired = compact(&swept, 100 * DAY_MS);
    let late_decl = turn(DEAD);
    let late: Vec<(RecordType, u64, Vec<u8>, RecordIdentity)> = vec![
        (
            RecordType::StreamDecl,
            late_decl.stream_id().0,
            late_decl.encode(),
            at(40, 465 * DAY_MS),
        ),
        (
            RecordType::Label,
            0,
            encode_label_payload(DEAD, "c-dead", "Back", &Default::default()),
            at(41, 465 * DAY_MS),
        ),
    ];
    let reference = open(&expired).substrate.replicated_state();
    let after: Vec<_> = expired
        .iter()
        .cloned()
        .chain(late.iter().cloned())
        .collect();
    let before: Vec<_> = late
        .iter()
        .cloned()
        .chain(expired.iter().cloned())
        .collect();
    for (order, records) in [("after", after), ("before", before)] {
        let mut store = open(&records);
        let dead = TimelineId::from_raw(DEAD).unwrap();
        assert!(store.substrate.is_tombstoned(dead), "{order}");
        assert_eq!(
            store.substrate.replicated_state(),
            reference,
            "a late record {order} the entry changed the state"
        );
        let group = GroupId::from_raw(1).unwrap();
        store
            .substrate
            .register_timeline(dead, LayerId::from_raw(1).unwrap(), group);
        assert!(store.substrate.is_tombstoned(dead));
        assert!(store
            .substrate
            .active_timelines_for_group(group)
            .all(|t| t != dead));
        let resident = gather_resident_set(&store.substrate, &store.markers, 0);
        assert!(resident.iter().any(|r| r.rt == RecordType::Deleted));
        assert!(
            !resident
                .iter()
                .any(|r| r.rt == RecordType::Tombstone && *r.payload == *tombstone(DEAD)),
            "an expired tombstone is not carried as itself"
        );
    }
}

const ENDED: &str = "world/w1/mission/m1";
const KEPT: &str = "world/w1/body/a";

fn object(
    obj: CustomObjectPayload,
    id: RecordIdentity,
) -> (RecordType, u64, Vec<u8>, RecordIdentity) {
    (RecordType::CustomObject, obj.stream_id(), obj.encode(), id)
}

/// A world's entity that ceased: its value (day 9) and its tombstone (day 10),
/// beside one that lives on.
fn keyed() -> Vec<(RecordType, u64, Vec<u8>, RecordIdentity)> {
    vec![
        object(
            CustomObjectPayload::new(ENDED, serde_json::json!({"step": 2})),
            at(1, 9 * DAY_MS),
        ),
        object(
            CustomObjectPayload::new(KEPT, serde_json::json!({"at": "hall/door"})),
            at(2, 9 * DAY_MS),
        ),
        object(CustomObjectPayload::ceased(ENDED), at(3, 10 * DAY_MS)),
    ]
}

/// **A keyed tombstone hides its key and expires like a timeline's.** It is
/// carried until a complete compaction has run past it and it is old, then
/// shrinks to an entry keeping its stamp; the live entity beside it is
/// carried throughout.
#[test]
fn a_keyed_tombstone_shrinks_once_swept_and_old() {
    let hash = CustomObjectPayload::stream_id_for(ENDED);
    let store = open(&keyed());
    assert!(
        store.substrate.custom_object(ENDED).is_none(),
        "hidden by its tombstone"
    );
    assert!(store.substrate.custom_object(KEPT).is_some());

    let first = compact(&store, 100 * DAY_MS);
    assert_eq!(
        of_type(&first, RecordType::CustomObject).len(),
        2,
        "the tombstone and the live entity"
    );
    assert_eq!(
        of_type(&first, RecordType::Deleted),
        vec![DeletedPayload {
            swept_keys: vec![hash],
            ..DeletedPayload::default()
        }
        .encode()]
    );

    let expired = compact(&open(&first), 100 * DAY_MS);
    let objects = of_type(&expired, RecordType::CustomObject);
    assert_eq!(objects.len(), 1, "only the live entity");
    assert_eq!(CustomObjectPayload::decode(&objects[0]).unwrap().key, KEPT);
    let entry = DeletedKey::of(hash, at(3, 10 * DAY_MS));
    assert_eq!(
        of_type(&expired, RecordType::Deleted),
        vec![DeletedPayload {
            keys: vec![entry],
            ..DeletedPayload::default()
        }
        .encode()]
    );
    let after = open(&expired);
    assert_eq!(after.substrate.deleted_key(hash), Some(entry));
    assert!(after.substrate.custom_object(ENDED).is_none());
}

/// **The entry outranks writes below its tombstone, and not above.** A write
/// replicated late from before the tombstone stays dead in either order with
/// the entry; one stamped after it — the entity coming back — stands.
#[test]
fn a_keyed_entry_holds_against_earlier_writes_only() {
    let expired = compact(&open(&compact(&open(&keyed()), 100 * DAY_MS)), 100 * DAY_MS);
    let late = object(
        CustomObjectPayload::new(ENDED, serde_json::json!({"step": 1})),
        at(4, 8 * DAY_MS),
    );
    let back = object(
        CustomObjectPayload::new(ENDED, serde_json::json!({"step": 9})),
        at(5, 200 * DAY_MS),
    );
    for records in [
        expired
            .iter()
            .cloned()
            .chain([late.clone()])
            .collect::<Vec<_>>(),
        [late.clone()]
            .into_iter()
            .chain(expired.iter().cloned())
            .collect(),
    ] {
        assert!(open(&records).substrate.custom_object(ENDED).is_none());
    }
    let returned = open(&expired.iter().cloned().chain([back]).collect::<Vec<_>>());
    assert_eq!(
        returned.substrate.custom_object(ENDED).unwrap().blob,
        serde_json::json!({"step": 9})
    );
}

/// **The retention reaches compaction.** Through the real store: a tombstone
/// survives the first compaction after it whatever the retention, and with
/// no retention the second one shrinks it.
#[test]
fn compaction_applies_the_members_retention() {
    let dir = tempfile::TempDir::new().unwrap();
    let dead = TimelineId::from_raw(DEAD).unwrap();
    {
        let mut substrate = Substrate::new();
        let mut sp =
            SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
        sp.write_tombstone(DEAD, None).unwrap();
        sp.commit().unwrap();
    }
    let mut substrate = Substrate::new();
    let mut sp = SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
    sp.set_tombstone_retention(Duration::ZERO);
    sp.compact(&mut substrate, None).unwrap();
    assert!(substrate.is_tombstone_swept(dead));
    assert!(
        substrate.deleted_entry(dead).is_none(),
        "not swept before it"
    );
    // The clock reads in milliseconds: step past the tombstone's.
    std::thread::sleep(Duration::from_millis(2));
    sp.compact(&mut substrate, None).unwrap();
    let entry = substrate.deleted_entry(dead).expect("shrunk to its entry");
    assert_eq!(entry.origin, sp.node());
    assert!(substrate.is_tombstoned(dead));
    drop(sp);
    let mut reopened = Substrate::new();
    let _sp = SubstratePersistence::open_in_with_substrate(dir.path(), &mut reopened).unwrap();
    assert!(reopened.is_tombstoned(dead));
    assert_eq!(reopened.deleted_entry(dead), Some(entry));
}
