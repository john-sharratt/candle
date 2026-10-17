use std::sync::Arc;

use git2::{Oid as LibOid, Signature, Time};

use super::*;
use crate::odb::db::DEFAULT_PACK_BUDGET;
use crate::odb::memory_store::MemoryStore;
use crate::odb::store::PackStore;

/// `len` bytes no compressor shrinks, different for each `seed`.
fn noise(seed: u32, len: usize) -> Vec<u8> {
    let mut x = 0x9e37_79b9_u32 ^ seed.wrapping_mul(0x85eb_ca6b);
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x as u8
        })
        .collect()
}

/// A pack of `blobs` incompressible blobs of `size` bytes and a commit of
/// them, as another database writes it; the pack, and every id in it.
fn big_pack(blobs: u32, size: usize) -> (Vec<u8>, Vec<LibOid>) {
    let store = Arc::new(MemoryStore::default());
    let source = ObjectDb::open("r", store.clone(), DEFAULT_PACK_BUDGET).unwrap();
    let repo = repository(&source).unwrap();
    let mut ids = Vec::new();
    let mut tree = repo.treebuilder(None).unwrap();
    for i in 0..blobs {
        let blob = repo.blob(&noise(i, size)).unwrap();
        tree.insert(format!("f{i}"), blob, 0o100644).unwrap();
        ids.push(blob);
    }
    let tree = repo.find_tree(tree.write().unwrap()).unwrap();
    let sig = Signature::new("Ada", "ada@example.com", &Time::new(1_700_000_000, 0)).unwrap();
    let commit = repo.commit(None, &sig, &sig, "big\n", &tree, &[]).unwrap();
    ids.extend([tree.id(), commit]);
    let kept = source.flush().unwrap();
    assert_eq!(kept.len(), 1);
    (store.pack(&kept[0]).unwrap(), ids)
}

/// **A pack larger than the database keeps is kept as several that fit**:
/// every part within the limit, every object readable from them by a member
/// that never saw the large pack, every part announced — and the large pack
/// itself neither written nor read from afterwards.
#[test]
fn a_pack_over_the_limit_is_kept_as_parts_that_fit() {
    let limit = 64 * 1024;
    let (pack, ids) = big_pack(20, 16 * 1024);
    assert!(pack.len() > limit, "the fixture crosses the limit");

    let store = Arc::new(MemoryStore::default());
    let db = ObjectDb::open_limited("r", store.clone(), DEFAULT_PACK_BUDGET, limit).unwrap();
    let kept = db.take_pack(pack).unwrap();
    assert!(kept.len() > 1, "{} parts", kept.len());
    for id in &kept {
        assert!(store.pack(id).unwrap().len() <= limit, "part {id}");
    }
    assert_eq!(
        store.packs().unwrap().len(),
        kept.len(),
        "nothing else written"
    );
    assert_eq!(db.held_packs().len(), kept.len(), "the large pack let go");
    assert_eq!(db.take_unannounced(), kept);

    let other = ObjectDb::open("r", store, DEFAULT_PACK_BUDGET).unwrap();
    let read = repository(&other).unwrap();
    for id in ids {
        assert!(read.odb().unwrap().exists(id), "{id}");
    }
}

/// **A pack within the limit is kept whole**, as one pack.
#[test]
fn a_pack_within_the_limit_is_kept_whole() {
    let (pack, _) = big_pack(2, 1024);
    let store = Arc::new(MemoryStore::default());
    let db = ObjectDb::open_limited("r", store.clone(), DEFAULT_PACK_BUDGET, 64 * 1024).unwrap();
    let kept = db.take_pack(pack.clone()).unwrap();
    assert_eq!(kept.len(), 1);
    assert_eq!(store.pack(&kept[0]).unwrap(), pack);
}
