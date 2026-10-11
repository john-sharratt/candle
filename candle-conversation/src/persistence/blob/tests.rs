//! Blobs through a real store: named, written once, reopened, compacted and
//! maintained.

use std::path::Path;

use tempfile::TempDir;

use super::{hex, parse_hex, BlobKey, BlobPayload};
use crate::persistence::maintenance::MaintenanceOp;
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
        assert_eq!(sp.blob_index().blobs().len(), 1, "the same bytes again");
        assert_eq!(sp.blob(&sha).unwrap().as_deref(), Some(&b"a portrait"[..]));
        sha
    };
    let (mut sp, _) = open(dir.path());
    assert_eq!(sp.blob(&sha).unwrap().as_deref(), Some(&b"a portrait"[..]));
    assert_eq!(sp.blob(&[7u8; 32]).unwrap(), None, "nothing of that hash");
    assert_eq!(sp.blob_index().blobs().len(), 1);
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
