//! Which kept images name a blob, so a blob goes only with the last of them.
//!
//! Blobs are content-addressed: two images whose files differ but whose
//! pixels agree share their rows and thumbnail, and a file someone else keeps
//! as a blob (npcd's pictures) is the same blob as the image sent from it. So
//! an image never retires a blob by itself. It names each of its blobs with a
//! keyed record, `vision/blob/<sha256>/<id>`, and releasing one retires the
//! blob only when no record names it.
//!
//! A blob that is already held when the first image names it was kept by
//! something outside the vision store, which keeps it as long as it likes: it
//! is named by [`OUTSIDE`] as well, and that name is never released.

use serde_json::json;
use sha2::{Digest, Sha256};

use crate::persistence::record::CustomObjectPayload;
use crate::persistence::SharedSubstrate;

/// The name a blob kept outside the vision store is held under.
pub const OUTSIDE: &str = "outside";

/// A blob's digest as the hex its records are named by.
pub fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn prefix(sha: &[u8; 32]) -> String {
    format!("vision/blob/{}/", hex(sha))
}

fn name(substrate: &SharedSubstrate, key: String) -> Result<(), String> {
    if substrate.custom_object(&key).is_some() {
        return Ok(());
    }
    substrate
        .put_custom_object(CustomObjectPayload::new(key, json!({})))
        .map_err(|e| e.to_string())
}

/// Keep `bytes` as a blob image `id` names, and its digest.
pub fn put_named(
    substrate: &SharedSubstrate,
    bytes: Vec<u8>,
    id: &str,
) -> Result<[u8; 32], String> {
    let sha: [u8; 32] = Sha256::digest(&bytes).into();
    let at = prefix(&sha);
    if substrate.has_blob(&sha) && substrate.custom_objects_count_under(&at) == 0 {
        name(substrate, format!("{at}{OUTSIDE}"))?;
    }
    let kept = substrate.put_blob(bytes).map_err(|e| e.to_string())?;
    name(substrate, format!("{at}{id}"))?;
    Ok(kept)
}

/// Image `id` no longer names blob `sha`; the blob is retired when nothing
/// else does.
pub fn release(substrate: &SharedSubstrate, sha: &[u8; 32], id: &str) -> Result<(), String> {
    let at = prefix(sha);
    substrate
        .put_custom_object(CustomObjectPayload::ceased(format!("{at}{id}")))
        .map_err(|e| e.to_string())?;
    if substrate.custom_objects_count_under(&at) == 0 {
        substrate.retire_blob(sha).map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::Admit;

    /// **A blob two images name goes with the second**, and one kept outside
    /// the vision store never goes.
    #[test]
    fn a_blob_goes_with_the_last_image_that_names_it() {
        let dir = tempfile::tempdir().unwrap();
        let s = SharedSubstrate::open_in(dir.path(), Admit::Unbound).unwrap();
        let shared = put_named(&s, b"rows".to_vec(), "aa").unwrap();
        assert_eq!(put_named(&s, b"rows".to_vec(), "bb").unwrap(), shared);
        release(&s, &shared, "aa").unwrap();
        assert!(s.has_blob(&shared));
        release(&s, &shared, "bb").unwrap();
        assert!(!s.has_blob(&shared));

        let outside = s.put_blob(b"npc picture".to_vec()).unwrap();
        assert_eq!(
            put_named(&s, b"npc picture".to_vec(), "cc").unwrap(),
            outside
        );
        release(&s, &outside, "cc").unwrap();
        assert!(s.has_blob(&outside), "a blob kept outside was retired");
    }

    /// A blob's records are named by its digest's hex: `abc`'s starts
    /// `ba7816bf`.
    #[test]
    fn a_blobs_records_are_named_by_its_digest() {
        let sha: [u8; 32] = Sha256::digest(b"abc").into();
        assert!(prefix(&sha).starts_with("vision/blob/ba7816bf"));
    }
}
