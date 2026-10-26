//! Links to a stored image that carry their own permission.
//!
//! `GET /v1/image/:iid` serves an image to a signed-in caller: an id is a
//! content hash and unguessable, and unguessable is not a permission. A link
//! made here *is* one. It names the image, the moment it stops working, and an
//! HMAC over both under a key every member holds, so whoever is handed it — a
//! client's sandbox fetching a drawn picture to give it to its user — opens it
//! at `GET /v1/image/link/:iid?exp=…&sig=…` without signing in, on any member,
//! until it expires. Changing the id or the expiry breaks the signature.
//!
//! The key is derived from the cluster's group key
//! (`host::cluster::Join::derive_key`). A daemon with no cluster is its own
//! only member and mints one at start, so its links stop working when it
//! restarts.

use std::fmt::{self, Debug, Formatter};
use std::time::Duration;

use candle_conversation::persistence::blob::{hex, parse_hex};
use hmac::{Hmac, Mac};
use rand::rngs::OsRng;
use rand::TryRngCore;
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

/// How long a link works: a week, long enough to be fetched, passed on and
/// opened again by the person it was passed to.
pub const LIFETIME: Duration = Duration::from_secs(7 * 24 * 60 * 60);

/// What a link's key is derived for, from the group key.
pub const PURPOSE: &str = "npcd image links";

/// Signs links to stored images, and checks them.
#[derive(Clone)]
pub struct ImageLinks {
    key: [u8; 32],
}

impl Debug for ImageLinks {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("ImageLinks(…)")
    }
}

/// Why a link was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refused {
    /// Its signature is not this key's over its id and expiry.
    Invalid,
    /// It was signed, and its time is up.
    Expired,
}

impl ImageLinks {
    /// Links signed with `key`.
    pub fn new(key: [u8; 32]) -> Self {
        Self { key }
    }

    /// Links signed with a key of this process's own, from the OS — a daemon
    /// with no cluster to share one with.
    pub fn minted() -> Self {
        let mut key = [0u8; 32];
        OsRng
            .try_fill_bytes(&mut key)
            .expect("the OS must provide entropy");
        Self { key }
    }

    /// The path, from the origin, that opens image `id` for [`LIFETIME`] from
    /// `now_s` (Unix seconds).
    pub fn path(&self, id: &str, now_s: u64) -> String {
        let exp = now_s + LIFETIME.as_secs();
        let sig: [u8; 32] = self.mac(id, exp).finalize().into_bytes().into();
        format!("/v1/image/link/{id}?exp={exp}&sig={}", hex(&sig))
    }

    /// Whether `sig` opens image `id` until `exp`, at `now_s`. The signature is
    /// checked first and in constant time, so a forged link learns nothing
    /// from which refusal it gets.
    pub fn check(&self, id: &str, exp: u64, sig: &str, now_s: u64) -> Result<(), Refused> {
        let sig: [u8; 32] = parse_hex(sig).ok_or(Refused::Invalid)?;
        self.mac(id, exp)
            .verify_slice(&sig)
            .map_err(|_| Refused::Invalid)?;
        if now_s >= exp {
            return Err(Refused::Expired);
        }
        Ok(())
    }

    /// The MAC over `id` and `exp`, a newline between them: neither holds one,
    /// so no other pair signs the same bytes.
    fn mac(&self, id: &str, exp: u64) -> HmacSha256 {
        let mut m = HmacSha256::new_from_slice(&self.key).expect("HMAC takes a key of any length");
        m.update(format!("{id}\n{exp}").as_bytes());
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ID: &str = "0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f.jpg";
    const NOW: u64 = 1_800_000_000;

    /// The parts of a link's path: its expiry and signature.
    fn parts(path: &str) -> (u64, String) {
        let (_, query) = path.split_once('?').unwrap();
        let (exp, sig) = query.split_once('&').unwrap();
        (
            exp.strip_prefix("exp=").unwrap().parse().unwrap(),
            sig.strip_prefix("sig=").unwrap().to_string(),
        )
    }

    /// **A link opens its own image for a week, on any member holding the key.**
    #[test]
    fn a_link_opens_its_image_for_a_week_on_any_member() {
        let links = ImageLinks::new([7; 32]);
        let path = links.path(ID, NOW);
        assert!(path.starts_with(&format!("/v1/image/link/{ID}?exp=1800604800&sig=")));
        let (exp, sig) = parts(&path);
        assert_eq!(exp, NOW + 7 * 24 * 60 * 60);
        assert_eq!(sig.len(), 64);
        let member = ImageLinks::new([7; 32]);
        assert_eq!(member.check(ID, exp, &sig, NOW), Ok(()));
        assert_eq!(member.check(ID, exp, &sig, exp - 1), Ok(()));
        assert_eq!(member.check(ID, exp, &sig, exp), Err(Refused::Expired));
    }

    /// **Nothing in a link can be changed**: another image, a later expiry, an
    /// edited or malformed signature, or another key's, is refused.
    #[test]
    fn a_link_altered_in_any_part_is_refused() {
        let links = ImageLinks::new([7; 32]);
        let (exp, sig) = parts(&links.path(ID, NOW));
        let other = ID.replace(".jpg", ".png");
        assert_eq!(links.check(&other, exp, &sig, NOW), Err(Refused::Invalid));
        assert_eq!(links.check(ID, exp + 1, &sig, NOW), Err(Refused::Invalid));
        let flipped = format!("{}{}", if &sig[..1] == "0" { "1" } else { "0" }, &sig[1..]);
        assert_eq!(links.check(ID, exp, &flipped, NOW), Err(Refused::Invalid));
        assert_eq!(links.check(ID, exp, "zz", NOW), Err(Refused::Invalid));
        assert_eq!(links.check(ID, exp, "", NOW), Err(Refused::Invalid));
        assert_eq!(
            ImageLinks::new([8; 32]).check(ID, exp, &sig, NOW),
            Err(Refused::Invalid)
        );
        // An expired forgery is refused as a forgery.
        assert_eq!(
            links.check(ID, exp, &flipped, exp + 1),
            Err(Refused::Invalid)
        );
    }

    #[test]
    fn a_minted_key_is_the_processes_own() {
        let a = ImageLinks::minted();
        let (exp, sig) = parts(&a.path(ID, NOW));
        assert_eq!(a.check(ID, exp, &sig, NOW), Ok(()));
        assert_eq!(
            ImageLinks::minted().check(ID, exp, &sig, NOW),
            Err(Refused::Invalid)
        );
    }
}
