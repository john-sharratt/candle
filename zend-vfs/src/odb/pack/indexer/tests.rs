use std::cell::RefCell;
use std::sync::Arc;

use super::*;
use crate::odb::pack::fixture::{full_pack, git_bytes, history, thin_pack};
use crate::odb::pack::idx::PackIndex;
use crate::odb::pack::reader::offset_in;
use crate::testing::TestRepo;

fn nothing(_: &RawId) -> Result<Option<Arc<Object>>, GitError> {
    Ok(None)
}

/// **Our index of git's pack is git's index, byte for byte**: every id, CRC
/// and offset, the fan-out, both checksums. The pack is left as it was.
#[test]
fn our_index_of_a_git_pack_is_gits_index_byte_for_byte() {
    let (t, _) = history(8);
    let (pack, git_index) = full_pack(&t);
    let ours = index_pack(pack.clone(), &nothing).unwrap();
    assert_eq!(ours.index, git_index);
    assert_eq!(ours.pack, pack);
    assert_eq!(ours.appended, 0);
    assert_eq!(ours.checksum.0[..], pack[pack.len() - HASH..]);
    let parsed = PackIndex::parse(ours.index).unwrap();
    assert_eq!(parsed.len() as u32, ours.objects);
}

/// **A thin pack is made whole**: the bases its ref deltas name, which only
/// `external` holds, are appended, and `git index-pack` — with no
/// `--fix-thin` — takes the result and indexes it exactly as we did.
#[test]
fn a_thin_pack_is_made_whole() {
    let (t, commits) = history(8);
    let thin = thin_pack(&t, &commits[7], &commits[3]);
    let (full, full_index) = full_pack(&t);
    let full_index = PackIndex::parse(full_index).unwrap();
    let cache = RefCell::new(BaseCache::new(1 << 20));
    let external = |id: &RawId| -> Result<Option<Arc<Object>>, GitError> {
        let find = |id: &RawId| offset_in(&full_index, id);
        let Some(at) = find(id) else {
            return Ok(None);
        };
        let reader = Pack {
            bytes: &full,
            find: &find,
        };
        reader
            .read_at(at, &mut cache.borrow_mut(), &nothing)
            .map(Some)
    };
    let whole = index_pack(thin.clone(), &external).unwrap();
    assert!(whole.appended > 0, "the thin pack named bases outside it");
    assert!(whole.pack.len() > thin.len());

    let other = TestRepo::init();
    let said = git_bytes(&other.path, &["index-pack", "--stdin"], &whole.pack);
    let name = String::from_utf8(said).unwrap();
    assert_eq!(
        name.trim().trim_start_matches("pack\t"),
        whole.checksum.to_string()
    );
    let idx_path = other
        .path
        .join(".git/objects/pack")
        .join(format!("pack-{}.idx", whole.checksum));
    assert_eq!(std::fs::read(idx_path).unwrap(), whole.index);
}

/// **A thin pack whose bases are held nowhere is refused**, naming one.
#[test]
fn a_thin_pack_with_no_bases_held_is_refused() {
    let (t, commits) = history(6);
    let thin = thin_pack(&t, &commits[5], &commits[2]);
    let err = index_pack(thin, &nothing).unwrap_err().to_string();
    assert!(err.contains("held nowhere"), "{err}");
}

/// **A damaged pack is refused**: a flipped byte breaks the trailer, a cut
/// loses it, and a header that is not a pack's says so.
#[test]
fn a_damaged_pack_is_refused() {
    let (t, _) = history(3);
    let (pack, _) = full_pack(&t);
    let mut flipped = pack.clone();
    let middle = flipped.len() / 2;
    flipped[middle] ^= 0x40;
    assert!(index_pack(flipped, &nothing).is_err());
    assert!(index_pack(pack[..pack.len() - 1].to_vec(), &nothing).is_err());
    let mut not_a_pack = pack;
    not_a_pack[0] = b'X';
    assert!(index_pack(not_a_pack, &nothing).is_err());
}
