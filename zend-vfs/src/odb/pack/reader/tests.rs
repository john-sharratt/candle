use super::*;
use crate::odb::pack::fixture::{full_pack, history};

fn nothing(_: &RawId) -> Result<Option<Arc<Object>>, GitError> {
    Ok(None)
}

fn no_header(_: &RawId) -> Result<Option<(ObjectType, u64)>, GitError> {
    Ok(None)
}

/// **Every object in git's pack reads back to itself**: decoded through its
/// delta chain, its bytes hash to the id the index names — which a single
/// wrong byte anywhere in the chain would break — and its header-only read
/// agrees on kind and size.
#[test]
fn every_object_in_a_git_pack_reads_back_to_its_own_id() {
    let (t, _) = history(10);
    let (pack, index) = full_pack(&t);
    let index = PackIndex::parse(index).unwrap();
    let find = |id: &RawId| offset_in(&index, id);
    let reader = Pack {
        bytes: &pack,
        find: &find,
    };
    let mut cache = BaseCache::new(1 << 20);
    let mut deltas = 0;
    for i in 0..index.len() {
        let id = index.id(i);
        let offset = index.offset(i);
        let object = reader.read_at(offset, &mut cache, &nothing).unwrap();
        assert_eq!(object.id(), id, "object {i}");
        assert_eq!(
            reader.header_at(offset, &no_header).unwrap(),
            (object.kind, object.data.len() as u64)
        );
        let header = read_header(&pack, offset as usize).unwrap();
        if !matches!(header.kind, EntryKind::Whole(_)) {
            deltas += 1;
        }
    }
    assert!(
        deltas > 5,
        "the fixture's pack holds delta chains: {deltas}"
    );
}

/// **No cache at all reads the same**: every chain walked to its base each
/// time.
#[test]
fn a_cache_of_nothing_reads_the_same() {
    let (t, _) = history(5);
    let (pack, index) = full_pack(&t);
    let index = PackIndex::parse(index).unwrap();
    let find = |id: &RawId| offset_in(&index, id);
    let reader = Pack {
        bytes: &pack,
        find: &find,
    };
    let mut none = BaseCache::new(0);
    let mut some = BaseCache::new(1 << 20);
    for i in 0..index.len() {
        let offset = index.offset(i);
        assert_eq!(
            reader.read_at(offset, &mut none, &nothing).unwrap(),
            reader.read_at(offset, &mut some, &nothing).unwrap()
        );
    }
}
