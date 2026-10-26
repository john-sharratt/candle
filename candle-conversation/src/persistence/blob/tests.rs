//! Blobs through a real store: named, written once, reopened, compacted and
//! maintained.

use std::path::Path;

use tempfile::TempDir;

use super::{forget_if_retired, hex, parse_hex, BlobIndex, BlobKey, BlobPayload};
use crate::persistence::accounting::RecordAccounting;
use crate::persistence::identity::RecordIdentity;
use crate::persistence::maintenance::MaintenanceOp;
use crate::persistence::manifest::RecordLoc;
use crate::persistence::record::CustomObjectPayload;
use crate::persistence::segment::SegmentId;
use crate::persistence::SubstratePersistence;
use crate::substrate::Substrate;

fn open(dir: &Path) -> (SubstratePersistence, Substrate) {
    let mut substrate = Substrate::new();
    let sp = SubstratePersistence::open_in_with_substrate(dir, &mut substrate).unwrap();
    (sp, substrate)
}

/// SHA-256 of `abc`.
const ABC: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

/// **A blob is its hash, then its bytes**, named by the hash's first 128 bits
/// in the header, and refused when the bytes no longer hash to the name.
#[test]
fn a_blob_is_named_by_its_hash() {
    let blob = BlobPayload::of(b"abc".to_vec());
    assert_eq!(hex(&blob.sha256), ABC);
    assert_eq!(parse_hex(ABC), Some(blob.sha256));
    assert_eq!(
        blob.key(),
        BlobKey {
            hi: 0xba78_16bf_8f01_cfea,
            lo: 0x4141_40de_5dae_2223,
        }
    );
    let mut encoded = blob.encode();
    assert_eq!(encoded.len(), 35);
    assert_eq!(&encoded[..4], &[0xba, 0x78, 0x16, 0xbf]);
    assert_eq!(&encoded[32..], b"abc");
    assert_eq!(BlobPayload::decode(&encoded).unwrap(), blob);

    encoded[33] = b'X';
    assert!(BlobPayload::decode(&encoded).is_err(), "the bytes changed");
    assert!(
        BlobPayload::decode(&[0u8; 31]).is_err(),
        "shorter than a hash"
    );
    assert_eq!(parse_hex("ba78"), None);
    assert_eq!(parse_hex(&ABC.replace('b', "g")), None);
}

/// **One hash is written once**, reads back, and outlives a reopen.
#[test]
fn a_blob_is_written_once_and_reads_back_after_a_reopen() {
    let dir = TempDir::new().unwrap();
    let sha = {
        let (mut sp, _) = open(dir.path());
        let sha = sp.write_blob(b"a portrait".to_vec()).unwrap();
        assert_eq!(sp.write_blob(b"a portrait".to_vec()).unwrap(), sha);
        assert_eq!(sp.blob_index().len(), 1, "the same bytes again");
        assert_eq!(sp.blob(&sha).unwrap().as_deref(), Some(&b"a portrait"[..]));
        sha
    };
    let (mut sp, _) = open(dir.path());
    assert_eq!(sp.blob(&sha).unwrap().as_deref(), Some(&b"a portrait"[..]));
    assert_eq!(sp.blob(&[7u8; 32]).unwrap(), None, "nothing of that hash");
    assert_eq!(sp.blob_index().len(), 1);
}

/// **A blob outlives a compaction and a maintenance pass**, read back from
/// wherever it was carried.
#[test]
fn a_blob_outlives_compaction_and_maintenance() {
    let dir = TempDir::new().unwrap();
    let (sha, other) = {
        let (mut sp, _) = open(dir.path());
        let sha = sp.write_blob(b"first".to_vec()).unwrap();
        sp.seal_active().unwrap();
        let other = sp.write_blob(b"second".to_vec()).unwrap();
        sp.seal_active().unwrap();
        (sha, other)
    };
    {
        let (mut sp, mut substrate) = open(dir.path());
        let at = sp.blob_index().get(BlobKey::of(&sha)).unwrap().segment;
        sp.apply_maintenance_op(&mut substrate, &MaintenanceOp::Compact(at))
            .unwrap();
        assert_ne!(
            sp.blob_index().get(BlobKey::of(&sha)).unwrap().segment,
            at,
            "carried off"
        );
        assert_eq!(sp.blob(&sha).unwrap().as_deref(), Some(&b"first"[..]));
        sp.compact(&mut substrate, None).unwrap();
        assert_eq!(sp.blob(&sha).unwrap().as_deref(), Some(&b"first"[..]));
        assert_eq!(sp.blob(&other).unwrap().as_deref(), Some(&b"second"[..]));
    }
    let (mut sp, _) = open(dir.path());
    assert_eq!(sp.blob(&sha).unwrap().as_deref(), Some(&b"first"[..]));
    assert_eq!(sp.blob(&other).unwrap().as_deref(), Some(&b"second"[..]));
}

/// **A blob tombstone's key is `blob/` and the key's 32 lowercase hex
/// digits**, and only that spelling names a blob.
#[test]
fn a_tombstone_is_keyed_by_the_blobs_key() {
    let key = BlobPayload::of(b"abc".to_vec()).key();
    assert_eq!(key.tombstone_key(), "blob/ba7816bf8f01cfea414140de5dae2223");
    assert_eq!(
        BlobKey::of_tombstone_key("blob/ba7816bf8f01cfea414140de5dae2223"),
        Some(key)
    );
    for other in [
        "blob/BA7816BF8F01CFEA414140DE5DAE2223",
        "blob/ba7816bf8f01cfea414140de5dae222",
        "blob/ba7816bf8f01cfea414140de5dae2223a",
        "blob/ba7816bf8f01cfea414140de5dae222g",
        "account/ba7816bf8f01cfea414140de5dae2223",
    ] {
        assert_eq!(BlobKey::of_tombstone_key(other), None, "{other}");
    }
}

/// The total bytes of the store's segment files.
fn store_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir.join("substrate"))
        .unwrap()
        .flatten()
        .filter_map(|e| e.metadata().ok())
        .filter(|m| m.is_file())
        .map(|m| m.len())
        .sum()
}

/// **A retired blob reads as absent at once and after a reopen, and its
/// bytes leave the log at the next compaction** — while the blob beside it
/// stays, and the tombstone is carried.
#[test]
fn a_retired_blob_is_gone_and_its_bytes_leave_at_compaction() {
    let dir = TempDir::new().unwrap();
    let big = vec![0x5a; 1 << 20];
    let (sha, kept) = {
        let (mut sp, mut substrate) = open(dir.path());
        let sha = sp.write_blob(big.clone()).unwrap();
        let kept = sp.write_blob(b"a portrait".to_vec()).unwrap();
        sp.commit().unwrap();
        sp.retire_blob(&mut substrate, &sha).unwrap();
        assert_eq!(sp.blob(&sha).unwrap(), None, "retired");
        assert!(!sp.has_blob(&sha));
        assert_eq!(sp.blob_index().len(), 1);
        (sha, kept)
    };
    let (mut sp, mut substrate) = open(dir.path());
    assert_eq!(sp.blob(&sha).unwrap(), None, "retired across a reopen");
    assert!(substrate.blob_tombstone(BlobKey::of(&sha)).is_some());
    let before = store_bytes(dir.path());
    assert!(before > 1 << 20, "the bytes are still on disk: {before}");
    sp.compact(&mut substrate, None).unwrap();
    let after = store_bytes(dir.path());
    assert!(
        after < before - (1 << 20) + (64 << 10),
        "compaction let the retired megabyte go: {before} -> {after}"
    );
    assert_eq!(sp.blob(&kept).unwrap().as_deref(), Some(&b"a portrait"[..]));
    drop(sp);
    let (mut sp, substrate) = open(dir.path());
    assert_eq!(sp.blob(&sha).unwrap(), None);
    assert_eq!(sp.blob(&kept).unwrap().as_deref(), Some(&b"a portrait"[..]));
    assert!(
        substrate.blob_tombstone(BlobKey::of(&sha)).is_some(),
        "the tombstone is carried until it expires"
    );
}

/// **The same bytes written after their blob was retired live again** — the
/// new record outranks the tombstone — through a reopen and a compaction.
#[test]
fn bytes_written_again_after_retirement_live() {
    let dir = TempDir::new().unwrap();
    let sha = {
        let (mut sp, mut substrate) = open(dir.path());
        let sha = sp.write_blob(b"drawn".to_vec()).unwrap();
        sp.retire_blob(&mut substrate, &sha).unwrap();
        assert_eq!(sp.write_blob(b"drawn".to_vec()).unwrap(), sha);
        sp.commit().unwrap();
        assert_eq!(sp.blob(&sha).unwrap().as_deref(), Some(&b"drawn"[..]));
        sha
    };
    let (mut sp, mut substrate) = open(dir.path());
    assert_eq!(sp.blob(&sha).unwrap().as_deref(), Some(&b"drawn"[..]));
    sp.compact(&mut substrate, None).unwrap();
    assert_eq!(sp.blob(&sha).unwrap().as_deref(), Some(&b"drawn"[..]));
    drop(sp);
    let (mut sp, _) = open(dir.path());
    assert_eq!(sp.blob(&sha).unwrap().as_deref(), Some(&b"drawn"[..]));
}

fn stamp(hlc: u64) -> RecordIdentity {
    RecordIdentity {
        origin: 7,
        oseq: hlc,
        gen: 0,
        epoch: 0,
        hlc,
    }
}

/// **A tombstone kills the records of its blob it outranks, in either order
/// of arrival, and no other**: a copy stamped after it lives.
#[test]
fn a_tombstone_kills_what_it_outranks_whichever_arrives_first() {
    let key = BlobKey { hi: 1, lo: 2 };
    let loc = RecordLoc {
        segment: SegmentId(1),
        offset: 0,
        payload_len: 40,
        record_size: 4096,
    };
    for (blob_hlc, tombstone_hlc, lives) in [(1, 2, false), (3, 2, true)] {
        for blob_first in [true, false] {
            let mut index = BlobIndex::new();
            let mut accounting = RecordAccounting::new();
            let mut substrate = Substrate::new();
            let tombstone = || CustomObjectPayload::ceased(key.tombstone_key());
            if blob_first {
                index.record(key, loc, stamp(blob_hlc));
                substrate.apply_custom_object(tombstone(), stamp(tombstone_hlc));
            } else {
                substrate.apply_custom_object(tombstone(), stamp(tombstone_hlc));
                index.record(key, loc, stamp(blob_hlc));
            }
            forget_if_retired(&mut index, &mut accounting, &substrate, key);
            assert_eq!(
                index.get(key).is_some(),
                lives,
                "blob at {blob_hlc}, tombstone at {tombstone_hlc}, blob first {blob_first}"
            );
        }
    }
}
