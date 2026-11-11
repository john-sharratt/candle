//! Conversation refs (`docs/private_mode.md` §3): the handle every Zen call
//! names its conversation by, issued at `get_user_context`.
//!
//! A ref carries what is needed to trust it, so nothing is looked up to open
//! one: its id, whether its conversation is private, when it was issued, and
//! a MAC over those and the account it was issued to, under a key every
//! member derives from the group key ([`PURPOSE`]). Another account's ref,
//! or one changed by a byte, does not open.
//!
//! ```text
//! id (16) || flags (1: bit 0 private) || issued (4, Unix seconds, BE) || mac (16)
//! mac = HMAC-SHA256(key, owner || 0 || id || flags || issued)[..16]
//! ```
//!
//! written base64url without padding: 50 characters.

use std::fmt::{self, Debug, Formatter};

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use hmac::{Hmac, Mac};
use rand::rngs::OsRng;
use rand::TryRngCore;
use sha2::{Digest, Sha256};

type HmacSha256 = Hmac<Sha256>;

/// What the refs' key is derived for, from the group key.
pub const PURPOSE: &str = "mcp conversation refs";

/// How long a ref may go unused before it is expired.
pub const REF_IDLE_MS: u64 = 4 * 60 * 60 * 1000;

/// How often, at most, a ref's use is told to the record both aggregators
/// read (`docs/private_mode.md` §3.2): the record is never further behind.
pub const REF_TOUCH_EVERY_MS: u64 = 10 * 60 * 1000;

/// How `get_user_context`'s answer opens: the ref follows, in backticks, on
/// the answer's first line.
pub const OPENING: &str = "This conversation's ref is `";

/// The ref an answer opening with [`OPENING`] names.
pub fn named_in(answer: &str) -> Option<&str> {
    let first = answer.lines().next()?;
    first.strip_prefix(OPENING)?.split('`').next()
}

const ID: usize = 16;
const MAC: usize = 16;
const LEN: usize = ID + 1 + 4 + MAC;
const PRIVATE: u8 = 1;

/// A conversation, as its ref names it.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Ref {
    pub id: [u8; ID],
    /// The conversation is private (`docs/private_mode.md` §4).
    pub private: bool,
    /// When it was issued, in Unix seconds.
    pub issued_s: u32,
}

impl Debug for Ref {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Ref({}, private: {}, issued: {})",
            self.conversation(),
            self.private,
            self.issued_s
        )
    }
}

impl Ref {
    /// The key the conversation is kept under — records, its journal, its
    /// noted calls: a digest of the id, so the ref itself is never written.
    pub fn conversation(&self) -> String {
        let digest = Sha256::digest(self.id);
        digest[..8].iter().map(|b| format!("{b:02x}")).collect()
    }

    /// When it was issued, in Unix milliseconds.
    pub fn issued_ms(&self) -> u64 {
        u64::from(self.issued_s) * 1000
    }

    /// Whether the conversation is live at `now_ms`, last used at `used_ms`
    /// when anything has said: issued, or used, within [`REF_IDLE_MS`].
    pub fn live(&self, used_ms: Option<u64>, now_ms: u64) -> bool {
        let last = used_ms.unwrap_or(0).max(self.issued_ms());
        now_ms.saturating_sub(last) < REF_IDLE_MS
    }
}

/// Why a ref did not open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refused {
    /// It is not one: the wrong length, or not base64url.
    Malformed,
    /// It was not issued under this key to this account, or was changed since.
    Forged,
}

/// Issues refs, and opens them.
#[derive(Clone)]
pub struct Refs {
    key: [u8; 32],
}

impl Debug for Refs {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("Refs(…)")
    }
}

impl Refs {
    /// Refs under `key`, derived for [`PURPOSE`].
    pub fn new(key: [u8; 32]) -> Self {
        Self { key }
    }

    /// Refs under a key of this process's own, from the OS — a server with no
    /// cluster to share one with.
    pub fn minted() -> Self {
        let mut key = [0u8; 32];
        OsRng
            .try_fill_bytes(&mut key)
            .expect("the OS must provide entropy");
        Self { key }
    }

    /// A new conversation for `owner`, private or not, at `now_ms`: its ref as
    /// written, and what it names.
    pub fn issue(&self, owner: &str, private: bool, now_ms: u64) -> (String, Ref) {
        let mut id = [0u8; ID];
        OsRng
            .try_fill_bytes(&mut id)
            .expect("the OS must provide entropy");
        let r = Ref {
            id,
            private,
            issued_s: u32::try_from(now_ms / 1000).unwrap_or(u32::MAX),
        };
        (self.seal(owner, &r), r)
    }

    /// `r`, issued to `owner`, as written.
    pub fn seal(&self, owner: &str, r: &Ref) -> String {
        let mut bytes = [0u8; LEN];
        bytes[..ID].copy_from_slice(&r.id);
        bytes[ID] = if r.private { PRIVATE } else { 0 };
        bytes[ID + 1..ID + 5].copy_from_slice(&r.issued_s.to_be_bytes());
        let mac = self.mac(owner, &bytes[..ID + 5]);
        bytes[ID + 5..].copy_from_slice(&mac);
        URL_SAFE_NO_PAD.encode(bytes)
    }

    /// The conversation `text` names, when it is a ref issued to `owner`
    /// under this key.
    pub fn open(&self, owner: &str, text: &str) -> Result<Ref, Refused> {
        let bytes = URL_SAFE_NO_PAD
            .decode(text.trim())
            .map_err(|_| Refused::Malformed)?;
        if bytes.len() != LEN {
            return Err(Refused::Malformed);
        }
        let mut m = HmacSha256::new_from_slice(&self.key).expect("HMAC takes a key of any length");
        m.update(owner.as_bytes());
        m.update(&[0]);
        m.update(&bytes[..ID + 5]);
        m.verify_truncated_left(&bytes[ID + 5..])
            .map_err(|_| Refused::Forged)?;
        // A flag no ref is issued with is no ref of ours.
        if bytes[ID] & !PRIVATE != 0 {
            return Err(Refused::Forged);
        }
        let mut id = [0u8; ID];
        id.copy_from_slice(&bytes[..ID]);
        Ok(Ref {
            id,
            private: bytes[ID] & PRIVATE != 0,
            issued_s: u32::from_be_bytes(bytes[ID + 1..ID + 5].try_into().expect("four bytes")),
        })
    }

    fn mac(&self, owner: &str, body: &[u8]) -> [u8; MAC] {
        let mut m = HmacSha256::new_from_slice(&self.key).expect("HMAC takes a key of any length");
        m.update(owner.as_bytes());
        m.update(&[0]);
        m.update(body);
        let full: [u8; 32] = m.finalize().into_bytes().into();
        full[..MAC].try_into().expect("sixteen bytes")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refs() -> Refs {
        Refs::new([7; 32])
    }

    fn fixed(private: bool) -> Ref {
        Ref {
            id: [0x11; ID],
            private,
            issued_s: 1_794_000_000,
        }
    }

    /// **A ref is written exactly as its layout says**: the id, the flag, the
    /// issue time big-endian, and the MAC's first sixteen bytes, base64url.
    #[test]
    fn a_ref_is_its_bytes_in_base64url() {
        let text = refs().seal("o1", &fixed(true));
        assert_eq!(text.len(), 50);
        let bytes = URL_SAFE_NO_PAD.decode(&text).unwrap();
        assert_eq!(&bytes[..16], &[0x11; 16]);
        assert_eq!(bytes[16], 1);
        assert_eq!(&bytes[17..21], &1_794_000_000u32.to_be_bytes());
        let mut m = HmacSha256::new_from_slice(&[7; 32]).unwrap();
        m.update(b"o1\0");
        m.update(&bytes[..21]);
        let mac: [u8; 32] = m.finalize().into_bytes().into();
        assert_eq!(&bytes[21..], &mac[..16]);
        assert_eq!(
            URL_SAFE_NO_PAD
                .decode(refs().seal("o1", &fixed(false)))
                .unwrap()[16],
            0
        );
    }

    #[test]
    fn a_ref_opens_as_what_it_was_issued() {
        for private in [false, true] {
            let text = refs().seal("o1", &fixed(private));
            assert_eq!(refs().open("o1", &text), Ok(fixed(private)));
        }
        let (text, issued) = refs().issue("o1", true, 1_794_000_000_123);
        assert_eq!(issued.issued_s, 1_794_000_000);
        assert_eq!(issued.issued_ms(), 1_794_000_000_000);
        assert_eq!(refs().open("o1", &text), Ok(issued));
        assert_ne!(
            refs().issue("o1", true, 0).1.id,
            issued.id,
            "each id is new"
        );
    }

    /// **Another account's ref, another key's, and one changed by a bit are
    /// forgeries**; text that is no ref at all is malformed.
    #[test]
    fn a_ref_opens_only_for_its_account_under_its_key() {
        let text = refs().seal("o1", &fixed(false));
        assert_eq!(refs().open("o2", &text), Err(Refused::Forged));
        assert_eq!(Refs::new([8; 32]).open("o1", &text), Err(Refused::Forged));
        let mut bytes = URL_SAFE_NO_PAD.decode(&text).unwrap();
        bytes[16] = 1;
        assert_eq!(
            refs().open("o1", &URL_SAFE_NO_PAD.encode(&bytes)),
            Err(Refused::Forged),
            "a normal ref made private is no ref"
        );
        assert_eq!(refs().open("o1", "abc"), Err(Refused::Malformed));
        assert_eq!(refs().open("o1", "!!!"), Err(Refused::Malformed));
        assert_eq!(refs().open("o1", ""), Err(Refused::Malformed));
    }

    /// **A ref idle for four hours is expired**: counted from its issue, or
    /// from its last use when one is known.
    #[test]
    fn a_ref_idle_for_four_hours_is_expired() {
        let r = fixed(false);
        let issued = r.issued_ms();
        assert!(r.live(None, issued));
        assert!(r.live(None, issued + REF_IDLE_MS - 1));
        assert!(!r.live(None, issued + REF_IDLE_MS), "idle from its issue");
        let used = issued + 3 * REF_IDLE_MS;
        assert!(r.live(Some(used), used + REF_IDLE_MS - 1));
        assert!(
            !r.live(Some(used), used + REF_IDLE_MS),
            "idle from its last use"
        );
        assert!(
            r.live(Some(issued - 10), issued + 1),
            "a use older than the issue counts from the issue"
        );
    }

    #[test]
    fn an_answers_ref_is_read_off_its_first_line() {
        assert_eq!(
            named_in("This conversation's ref is `abc-_1`. Pass it on.\n\nMemory."),
            Some("abc-_1")
        );
        assert_eq!(
            named_in("Memory first.\nThis conversation's ref is `x`."),
            None
        );
        assert_eq!(named_in(""), None);
    }

    /// **The conversation key is a digest of the id**, never the ref.
    #[test]
    fn the_conversation_is_kept_under_a_digest_of_the_id() {
        let r = fixed(false);
        let key = r.conversation();
        assert_eq!(key.len(), 16);
        let digest = Sha256::digest([0x11u8; 16]);
        let want: String = digest[..8].iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(key, want);
        assert_eq!(
            fixed(true).conversation(),
            key,
            "the flag is not the conversation"
        );
    }
}
