//! Who is asking — established by verifying a signature, not by reading a header.
//!
//! The gateway strips every inbound `X-Tokera-*` and sets its own from the
//! session cookie, so on the intended path those headers are already
//! trustworthy. This daemon does not rely on that, and the reason is the shape
//! of the deployment rather than distrust of the gateway: `npcd` binds a LAN
//! address so the DMZ box can reach it, which means **anything else on that LAN
//! can reach it too**. A daemon that believed `X-Tokera-User` would hand any
//! identity to anyone who could open a socket to it.
//!
//! So the only header that counts is `X-Tokera-Assertion` — the signed session
//! token itself. It is verified against the same key the gateway signed it
//! with, and every field of the identity comes out of the verified payload.
//! The other `X-Tokera-*` headers are read by nothing here; they exist for
//! logging and for services that have chosen to trust the hop.
//!
//! # Without a key, nobody is signed in
//!
//! If no key is configured the verifier authenticates *nothing*. That direction
//! matters: the alternative — treating an unconfigured daemon as "trust
//! whatever arrives" — is the failure that turns a missing config file into an
//! open door. Fail closed, and say so at startup.

use std::path::Path;

use axum::http::HeaderMap;
use web::auth::session::{self, Identity, Invalid, Key};

/// The name of the one header that is evidence rather than assertion.
const ASSERTION: &str = "x-tokera-assertion";

/// Verifies session assertions against the estate's shared signing key.
#[derive(Debug)]
pub struct Verifier {
    /// `None` means sign-in is not configured here, and therefore that no
    /// request can ever be authenticated.
    key: Option<Key>,
}

/// Why a request is not authenticated. Distinguished because a signed-out
/// browser and a daemon with no key are different problems with different
/// fixes, and a console that cannot tell them apart shows the wrong thing.
#[derive(Debug, PartialEq, Eq)]
pub enum NotSignedIn {
    /// No key configured — nothing can be verified here at all.
    Unconfigured,
    /// No assertion presented: an ordinary signed-out visitor.
    Absent,
    /// An assertion was presented and did not check out.
    Rejected(Invalid),
}

impl Verifier {
    /// Load the shared signing key.
    ///
    /// The same file the gateway signs with, copied to each machine that has to
    /// verify. It is never in the repository — one key across tokera.com,
    /// code. and bot. is exactly what makes a single sign-in carry to all three,
    /// and is exactly why it cannot be committed.
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let raw = std::fs::read(path)
            .map_err(|e| anyhow::anyhow!("reading session key {}: {e}", path.display()))?;
        // Tolerate a trailing newline — the documented way to make one of these
        // is `head -c 48 /dev/urandom | base64 > session.key`, and a shell
        // redirect adds one.
        let trimmed = raw
            .iter()
            .rev()
            .position(|b| !b.is_ascii_whitespace())
            .map(|n| &raw[..raw.len() - n])
            .unwrap_or(&raw);
        Ok(Self {
            key: Some(Key::new(trimmed.to_vec())?),
        })
    }

    /// A verifier that authenticates nothing.
    pub fn unconfigured() -> Self {
        Self { key: None }
    }

    /// A verifier over a key held in memory. Test-only: in a running daemon the
    /// key comes from the file the gateway shares, and offering a way to
    /// construct one from bytes would be an invitation to hardcode it.
    #[cfg(test)]
    pub fn with_key(key: Key) -> Self {
        Self { key: Some(key) }
    }

    pub fn is_configured(&self) -> bool {
        self.key.is_some()
    }

    /// Establish identity from request headers, or explain why not.
    pub fn identify(&self, headers: &HeaderMap, now_secs: u64) -> Result<Identity, NotSignedIn> {
        let Some(key) = &self.key else {
            return Err(NotSignedIn::Unconfigured);
        };
        let token = headers
            .get(ASSERTION)
            .and_then(|v| v.to_str().ok())
            .ok_or(NotSignedIn::Absent)?;
        session::open(key, token, now_secs).map_err(NotSignedIn::Rejected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> Key {
        Key::new(vec![7u8; 32]).unwrap()
    }

    fn verifier() -> Verifier {
        Verifier::with_key(key())
    }

    fn id(sub: &str, exp: u64) -> Identity {
        Identity {
            sub: sub.into(),
            email: "a@b.c".into(),
            name: "Name".into(),
            picture: String::new(),
            exp,
        }
    }

    fn headers(assertion: Option<&str>) -> HeaderMap {
        let mut h = HeaderMap::new();
        if let Some(a) = assertion {
            h.insert(ASSERTION, a.parse().unwrap());
        }
        h
    }

    #[test]
    fn a_validly_signed_assertion_identifies_the_caller() {
        let token = session::sign(&key(), &id("sub-1", 2_000));
        let got = verifier().identify(&headers(Some(&token)), 1_000).unwrap();
        assert_eq!(got.sub, "sub-1");
    }

    /// The whole point. `npcd` is reachable from the LAN, so a caller that can
    /// set headers must not be able to set an identity.
    #[test]
    fn plain_identity_headers_are_not_evidence_of_anything() {
        let mut h = HeaderMap::new();
        h.insert("x-tokera-user", "someone-else".parse().unwrap());
        h.insert("x-tokera-email", "victim@example.com".parse().unwrap());
        h.insert("x-tokera-name", "Victim".parse().unwrap());
        assert_eq!(
            verifier().identify(&h, 1_000),
            Err(NotSignedIn::Absent),
            "an unsigned header set was accepted as identity"
        );
    }

    #[test]
    fn a_forged_or_edited_assertion_is_refused() {
        let token = session::sign(&key(), &id("sub-1", 2_000));

        // Signed with a different key.
        let other = session::sign(&Key::new(vec![9u8; 32]).unwrap(), &id("admin", 2_000));
        assert_eq!(
            verifier().identify(&headers(Some(&other)), 1_000),
            Err(NotSignedIn::Rejected(Invalid::BadSignature))
        );

        // Payload edited, signature left alone.
        let (_, sig) = token.split_once('.').unwrap();
        let forged = format!(
            "{}.{sig}",
            base64::Engine::encode(
                &base64::engine::general_purpose::URL_SAFE_NO_PAD,
                br#"{"sub":"admin","exp":99999999999}"#
            )
        );
        assert_eq!(
            verifier().identify(&headers(Some(&forged)), 1_000),
            Err(NotSignedIn::Rejected(Invalid::BadSignature))
        );

        for junk in ["", ".", "not-a-token", "a.b.c"] {
            assert!(verifier().identify(&headers(Some(junk)), 1_000).is_err());
        }
    }

    #[test]
    fn an_expired_assertion_is_refused() {
        let token = session::sign(&key(), &id("sub-1", 1_000));
        assert_eq!(
            verifier().identify(&headers(Some(&token)), 1_001),
            Err(NotSignedIn::Rejected(Invalid::Expired))
        );
    }

    /// Fail closed. An unconfigured daemon authenticates nobody — including a
    /// caller presenting a token that would otherwise be perfectly good.
    #[test]
    fn without_a_key_nothing_is_authenticated() {
        let v = Verifier::unconfigured();
        assert!(!v.is_configured());
        let token = session::sign(&key(), &id("sub-1", 2_000));
        assert_eq!(
            v.identify(&headers(Some(&token)), 1_000),
            Err(NotSignedIn::Unconfigured)
        );
    }
}
