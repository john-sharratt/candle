use super::*;
use crate::odb::memory_store::MemoryStore;
use crate::odb::pack::fixture::{full_pack, history};
use crate::odb::pack::indexer::index_pack;
use crate::odb::store::KeptPack;

fn nothing(_: &RawId) -> Result<Option<Arc<Object>>, GitError> {
    Ok(None)
}

/// A store holding git's pack of a ten-commit history, and the pack's ids.
fn stored() -> (Arc<MemoryStore>, Vec<RawId>, usize) {
    let (t, _) = history(10);
    let (pack, _) = full_pack(&t);
    let indexed = index_pack(pack, &nothing).unwrap();
    let ids: Vec<RawId> = PackIndex::parse(indexed.index.clone())
        .unwrap()
        .ids()
        .collect();
    let size = indexed.pack.len();
    let store = Arc::new(MemoryStore::default());
    store
        .put(KeptPack {
            pack: indexed.pack,
            index: indexed.index,
            checksum: indexed.checksum.to_string(),
            objects: indexed.objects,
        })
        .unwrap();
    (store, ids, size)
}

/// **Every object the store's packs hold reads back to its id**, its
/// header agreeing.
#[test]
fn every_object_reads_from_the_stores_packs() {
    let (store, ids, _) = stored();
    let mut packs = Packs::open(store, 1 << 30).unwrap();
    for id in &ids {
        let object = packs.read(id).unwrap().unwrap();
        assert_eq!(object.id(), *id);
        assert_eq!(
            packs.header(id).unwrap(),
            Some((object.kind, object.data.len() as u64))
        );
        assert!(packs.contains(id));
    }
    let absent = RawId([0xee; 20]);
    assert_eq!(packs.read(&absent).unwrap(), None);
    assert!(!packs.contains(&absent));
}

/// **A pack's bytes are let go past the budget and read again on demand**:
/// with a budget smaller than the pack, nothing stays resident between reads
/// of another pack, and every read still answers.
#[test]
fn a_pack_past_the_budget_is_let_go_and_read_again() {
    let (store, ids, size) = stored();
    let (t2, _) = history(3);
    let (pack2, _) = full_pack(&t2);
    let second = index_pack(pack2, &nothing).unwrap();
    let second_ids: Vec<RawId> = PackIndex::parse(second.index.clone())
        .unwrap()
        .ids()
        .collect();
    store
        .put(KeptPack {
            pack: second.pack,
            index: second.index,
            checksum: second.checksum.to_string(),
            objects: second.objects,
        })
        .unwrap();
    let mut packs = Packs::open(store, size).unwrap();
    for round in 0..3 {
        for id in ids.iter().chain(&second_ids) {
            assert_eq!(packs.read(id).unwrap().unwrap().id(), *id, "round {round}");
            assert!(packs.resident() <= size, "never more than the budget");
        }
    }
}

/// **A pack not yet arrived is taken in by a refresh once it has.**
#[test]
fn a_refresh_takes_in_a_pack_that_arrived() {
    let (store, ids, _) = stored();
    let pack_id = store.packs().unwrap()[0].clone();
    store.hidden.lock().unwrap().push(pack_id);
    let mut packs = Packs::open(Arc::clone(&store) as Arc<dyn PackStore>, 1 << 30).unwrap();
    assert_eq!(packs.read(&ids[0]).unwrap(), None);
    store.hidden.lock().unwrap().clear();
    assert_eq!(packs.refresh().unwrap().added, 1);
    assert!(packs.read(&ids[0]).unwrap().is_some());
    assert_eq!(
        packs.refresh().unwrap().added,
        0,
        "a second refresh takes nothing"
    );
}

/// **A pack whose index cannot be read is reported, not fatal**: the refresh
/// takes every other pack and names the bad one.
#[test]
fn a_pack_with_an_unreadable_index_is_reported_and_the_rest_taken() {
    let (store, ids, _) = stored();
    store
        .put(KeptPack {
            pack: b"PACK".to_vec(),
            index: b"not an index".to_vec(),
            checksum: "0".repeat(40),
            objects: 1,
        })
        .unwrap();
    let mut packs = Packs::open(Arc::clone(&store) as Arc<dyn PackStore>, 1 << 30).unwrap();
    assert!(
        packs.read(&ids[0]).unwrap().is_some(),
        "the good pack is taken"
    );
    let again = packs.refresh().unwrap();
    assert_eq!(
        (again.added, again.failed.len()),
        (0, 1),
        "the bad one named again"
    );
}

/// **A prefix finds every id it begins and no other**, across packs.
#[test]
fn a_prefix_finds_its_ids() {
    let (store, ids, _) = stored();
    let packs = Packs::open(store, 1 << 30).unwrap();
    let target = ids[3];
    let found = packs.prefixed(&target, 7);
    assert!(found.contains(&target));
    assert!(found.iter().all(|id| starts_with(id, &target, 7)));
    assert_eq!(packs.prefixed(&target, 40), vec![target]);
}
