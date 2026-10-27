//! The key a vote is held under (§5.9).
//!
//! A journal slot holds a key of at most [`MAX_KEY`] bytes, and a key is a
//! name the app chooses: an ingest lease is `ingest/v3:` and a file's path and
//! commit, and a gateway's tie-break holds it as `app`, a zero byte, then the
//! key. A key past the slot was refused as a write error, so the round for it
//! could never be decided: the gateway's tie-break refused every vote for a
//! 128-byte ingest key it held as 133, and BlackWorld re-prepared that one
//! lease ~28,000 times while its file went unread (2026-10-27).
//!
//! A key that fits is held as itself. A longer one is held under a digest of
//! it, the same on every acceptor and across a restart, so it is voted on like
//! any other.

use std::borrow::Cow;

use sha2::{Digest, Sha256};

use super::MAX_KEY;

/// What a digested key starts with. No key an app names starts this way, so
/// a digested key never collides with one held as itself.
const DIGESTED: &[u8] = b"#sha256:";

/// `key` as a vote for it is held: itself while it fits [`MAX_KEY`], and
/// otherwise [`DIGESTED`] and its SHA-256 — 40 bytes.
pub fn held_key(key: &[u8]) -> Cow<'_, [u8]> {
    if key.len() <= MAX_KEY {
        return Cow::Borrowed(key);
    }
    let mut out = Vec::with_capacity(DIGESTED.len() + 32);
    out.extend_from_slice(DIGESTED);
    out.extend_from_slice(&Sha256::digest(key));
    Cow::Owned(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A key that fits is held as itself, up to the slot's last byte.
    #[test]
    fn a_key_that_fits_is_held_as_itself() {
        assert_eq!(&*held_key(b"world/w1"), b"world/w1");
        let full = [b'k'; MAX_KEY];
        assert_eq!(&*held_key(&full), &full[..]);
    }

    /// One byte past the slot is held under its digest: the prefix, then the
    /// SHA-256 of the whole key.
    #[test]
    fn a_longer_key_is_held_under_its_digest() {
        let long = [b'k'; MAX_KEY + 1];
        let held = held_key(&long);
        assert_eq!(held.len(), 40);
        assert_eq!(&held[..8], b"#sha256:");
        assert_eq!(&held[8..], &Sha256::digest(long)[..]);
        assert!(held.len() <= MAX_KEY);
    }

    /// Two long keys that differ only past the slot are held apart.
    #[test]
    fn long_keys_differing_past_the_slot_are_held_apart() {
        let mut a = vec![b'k'; MAX_KEY + 8];
        let b = a.clone();
        a[MAX_KEY + 4] = b'z';
        assert_ne!(held_key(&a), held_key(&b));
    }
}
