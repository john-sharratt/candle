//! What the authorization server signs: the consent a page posts back, the
//! code it issues, and the access and refresh tokens a client holds
//! (`docs/claude_zend_gateway.md` §12).
//!
//! **Stateless, like the session.** Each is a value signed with the session
//! key ([`session::sign`]), so peered gateways — which share that key — honour
//! what any of them issued, and nothing is stored. Each carries a `kind` that
//! the reader requires, so a code is never accepted as a token, a token never
//! as a code, and a session cookie — which has no `kind` — never as either.
//!
//! What being stateless costs, and why it is acceptable here: a code works for
//! its five minutes, not once, and a refresh token works until it expires,
//! not once. Neither is useful without what the client alone holds — the code
//! without the PKCE verifier, the token without the client — and signing every
//! grant out at once is rotating the session key, as it is for sessions.

use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::auth::session::{self, Identity, Key};

/// How long a consent page may wait for an answer.
pub const CONSENT_TTL_SECS: u64 = 600;
/// How long a code may wait for its exchange.
pub const CODE_TTL_SECS: u64 = 300;
/// How long an access token lasts.
pub const ACCESS_TTL_SECS: u64 = 3600;
/// How long a refresh token lasts.
pub const REFRESH_TTL_SECS: u64 = 30 * 24 * 3600;

/// What a client was granted, once the request was checked and the user said
/// yes: who, for which client and redirect, to which resource.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Grant {
    pub client_id: String,
    pub redirect_uri: String,
    pub resource: String,
    pub provider: String,
    pub sub: String,
    #[serde(default)]
    pub email: String,
    #[serde(default)]
    pub name: String,
}

impl Grant {
    /// Whether `identity` is the person this grant is for: a person is their
    /// provider and subject.
    pub fn is_for(&self, identity: &Identity) -> bool {
        self.provider == identity.provider && self.sub == identity.sub
    }

    /// The identity a token on this grant states downstream, valid until
    /// `exp`.
    pub fn identity(&self, exp: u64) -> Identity {
        Identity {
            provider: self.provider.clone(),
            sub: self.sub.clone(),
            email: self.email.clone(),
            name: self.name.clone(),
            picture: String::new(),
            exp,
        }
    }
}

/// A consent the user has been shown.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Consent {
    pub grant: Grant,
    pub challenge: String,
    pub state: Option<String>,
    pub client_name: String,
}

/// A code waiting for its exchange.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Code {
    pub grant: Grant,
    pub challenge: String,
}

/// The kinds a signed grant may be.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    McpConsent,
    McpCode,
    McpAccess,
    McpRefresh,
}

#[derive(Serialize, Deserialize)]
struct Signed<T> {
    kind: Kind,
    exp: u64,
    body: T,
}

/// Sign `body` as `kind`, valid until `exp`.
pub fn seal<T: Serialize>(key: &Key, kind: Kind, exp: u64, body: &T) -> String {
    session::sign(key, &Signed { kind, exp, body })
}

/// The body of `token`, if it is a `kind` this key signed and it has not
/// expired at `now`.
pub fn open<T: DeserializeOwned>(key: &Key, kind: Kind, token: &str, now: u64) -> Option<T> {
    let signed: Signed<T> = session::verify(key, token).ok()?;
    (signed.kind == kind && now < signed.exp).then_some(signed.body)
}

#[cfg(test)]
pub mod testing {
    use super::*;

    pub fn key() -> Key {
        Key::new(b"0123456789abcdef0123456789abcdef".to_vec()).unwrap()
    }

    pub fn grant() -> Grant {
        Grant {
            client_id: "https://claude.ai/oauth/mcp-oauth-client-metadata".into(),
            redirect_uri: "https://claude.ai/api/mcp/auth_callback".into(),
            resource: "https://code.tokera.com/mcp".into(),
            provider: "google".into(),
            sub: "boss".into(),
            email: "boss@example.com".into(),
            name: "Boss".into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{grant, key};
    use super::*;

    #[test]
    fn a_grant_opens_as_its_own_kind_until_it_expires() {
        let k = key();
        let t = seal(&k, Kind::McpAccess, 100, &grant());
        assert_eq!(open::<Grant>(&k, Kind::McpAccess, &t, 99), Some(grant()));
        assert_eq!(open::<Grant>(&k, Kind::McpAccess, &t, 100), None);
        assert_eq!(open::<Grant>(&k, Kind::McpRefresh, &t, 99), None);
        assert_eq!(open::<Grant>(&k, Kind::McpCode, &t, 99), None);
    }

    #[test]
    fn another_key_and_a_session_cookie_are_never_a_grant() {
        let t = seal(&key(), Kind::McpAccess, 100, &grant());
        let other = Key::new(b"fedcba9876543210fedcba9876543210".to_vec()).unwrap();
        assert_eq!(open::<Grant>(&other, Kind::McpAccess, &t, 1), None);
        let cookie = session::sign(&key(), &grant().identity(100));
        assert_eq!(open::<Grant>(&key(), Kind::McpAccess, &cookie, 1), None);
    }

    #[test]
    fn a_grant_is_for_its_provider_and_subject_only() {
        let g = grant();
        let mut id = g.identity(9);
        assert!(g.is_for(&id));
        id.email = "changed@example.com".into();
        assert!(g.is_for(&id));
        id.sub = "someone".into();
        assert!(!g.is_for(&id));
    }
}
