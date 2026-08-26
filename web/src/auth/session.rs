//! The session token: who you are, signed so it cannot be edited.
//!
//! Deliberately stateless. A server-side session table would have to be shared
//! by every process that validates a session, and the estate is three machines
//! — so the token carries the claims and an HMAC over them, and validating is
//! arithmetic rather than a lookup. Signing out everyone is one operation:
//! change the key.
//!
//! Format: `base64url(json) . base64url(hmac-sha256)`. No JWT header, because
//! there is one algorithm and a field naming it is the part of JWT that has
//! caused the most trouble — `alg: none` is not expressible here.

use base64::engine::general_purpose::URL_SAFE_NO_PAD as B64;
use base64::Engine;
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

/// Who the browser is. The claims a signed session carries.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Identity {
    /// The provider's stable subject id. The only field safe to key on: an
    /// email address can be reassigned, a name is not unique.
    pub sub: String,
    #[serde(default)]
    pub email: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub picture: String,
    /// Unix seconds. Absolute rather than a duration so a clock that moves
    /// cannot extend a session.
    pub exp: u64,
}

impl Identity {
    pub fn expired(&self, now_secs: u64) -> bool {
        now_secs >= self.exp
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Invalid {
    Malformed,
    BadSignature,
    Expired,
}

impl std::fmt::Display for Invalid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Invalid::Malformed => "malformed session",
            Invalid::BadSignature => "bad session signature",
            Invalid::Expired => "session expired",
        })
    }
}

/// The HMAC key. Wrapped so it cannot be printed into a log by accident.
#[derive(Clone)]
pub struct Key(Vec<u8>);

impl std::fmt::Debug for Key {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Key({} bytes)", self.0.len())
    }
}

impl Key {
    /// Reject a key short enough to be guessable rather than quietly signing
    /// with it — this is the one input where a weak value looks like it works.
    pub fn new(bytes: Vec<u8>) -> anyhow::Result<Self> {
        if bytes.len() < 32 {
            anyhow::bail!(
                "session key is {} bytes; at least 32 are required (try `head -c 48 /dev/urandom | base64`)",
                bytes.len()
            );
        }
        Ok(Key(bytes))
    }

    fn mac(&self, msg: &[u8]) -> Vec<u8> {
        let mut m = HmacSha256::new_from_slice(&self.0).expect("hmac takes a key of any length");
        m.update(msg);
        m.finalize().into_bytes().to_vec()
    }
}

pub fn sign<T: Serialize>(key: &Key, claims: &T) -> String {
    let json = serde_json::to_vec(claims).expect("claims serialise");
    let payload = B64.encode(&json);
    let sig = B64.encode(key.mac(payload.as_bytes()));
    format!("{payload}.{sig}")
}

/// Verify and decode. Does **not** check expiry — [`open`] does, because not
/// every signed value this is used for has an `exp`.
pub fn verify<T: for<'de> Deserialize<'de>>(key: &Key, token: &str) -> Result<T, Invalid> {
    let (payload, sig) = token.split_once('.').ok_or(Invalid::Malformed)?;
    let want = key.mac(payload.as_bytes());
    let got = B64.decode(sig).map_err(|_| Invalid::Malformed)?;
    // Constant-time: a byte-by-byte early return leaks how much of a forged
    // signature was right, which is enough to construct one.
    if !constant_time_eq(&want, &got) {
        return Err(Invalid::BadSignature);
    }
    let json = B64.decode(payload).map_err(|_| Invalid::Malformed)?;
    serde_json::from_slice(&json).map_err(|_| Invalid::Malformed)
}

/// Verify, decode, and reject an expired session.
pub fn open(key: &Key, token: &str, now_secs: u64) -> Result<Identity, Invalid> {
    let id: Identity = verify(key, token)?;
    if id.expired(now_secs) {
        return Err(Invalid::Expired);
    }
    Ok(id)
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b) {
        diff |= x ^ y;
    }
    diff == 0
}

pub fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> Key {
        Key::new(b"0123456789abcdef0123456789abcdef".to_vec()).unwrap()
    }

    fn ident(exp: u64) -> Identity {
        Identity {
            sub: "1234".into(),
            email: "a@b.c".into(),
            name: "A B".into(),
            picture: String::new(),
            exp,
        }
    }

    #[test]
    fn a_signed_session_round_trips() {
        let k = key();
        let token = sign(&k, &ident(9_000));
        assert_eq!(open(&k, &token, 100).unwrap(), ident(9_000));
    }

    #[test]
    fn editing_the_claims_invalidates_it() {
        // The whole point: the browser holds this and must not be able to
        // promote itself by editing the payload.
        let k = key();
        let token = sign(&k, &ident(9_000));
        let (payload, sig) = token.split_once('.').unwrap();
        let mut claims: Identity = serde_json::from_slice(&B64.decode(payload).unwrap()).unwrap();
        claims.email = "admin@tokera.com".into();
        let forged = format!("{}.{sig}", B64.encode(serde_json::to_vec(&claims).unwrap()));
        assert_eq!(open(&k, &forged, 100), Err(Invalid::BadSignature));
    }

    #[test]
    fn another_key_cannot_open_it() {
        let token = sign(&key(), &ident(9_000));
        let other = Key::new(b"ffffffffffffffffffffffffffffffff".to_vec()).unwrap();
        assert_eq!(open(&other, &token, 100), Err(Invalid::BadSignature));
    }

    #[test]
    fn expiry_is_enforced_on_open() {
        let k = key();
        let token = sign(&k, &ident(1_000));
        assert!(open(&k, &token, 999).is_ok());
        assert_eq!(open(&k, &token, 1_000), Err(Invalid::Expired));
    }

    #[test]
    fn rubbish_is_malformed_not_a_panic() {
        let k = key();
        for t in ["", ".", "no-dot", "a.b", "!!!.???"] {
            assert!(matches!(
                open(&k, t, 0),
                Err(Invalid::Malformed) | Err(Invalid::BadSignature)
            ));
        }
    }

    #[test]
    fn a_short_key_is_refused_rather_than_accepted_quietly() {
        assert!(Key::new(b"short".to_vec()).is_err());
        assert!(Key::new(vec![0u8; 32]).is_ok());
    }

    #[test]
    fn the_key_does_not_print_itself() {
        let dbg = format!("{:?}", key());
        assert!(!dbg.contains("0123"), "{dbg}");
        assert_eq!(dbg, "Key(32 bytes)");
    }
}
