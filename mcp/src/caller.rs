//! Who is calling the aggregator (`docs/mcp_app.md` §2).
//!
//! The identity is the gateway's: it signed the person in, and on `/mcp` it
//! turns the client's token back into that identity and states it on the
//! `X-Tokera-*` headers (`web::auth::mcp`), passing the token on beside them.
//! The app believes those headers only from a peer it lists as a gateway; from
//! anywhere else nobody is calling, and the client is sent to sign in.
//!
//! Every backend is called through the gateway with that same token, never
//! with headers of its own: the gateway reads the person from the token it
//! issued, so the app can act for nobody it was not handed.

use std::net::IpAddr;

use axum::http::{header, HeaderMap, HeaderValue};
use sha2::{Digest, Sha256};
use web::auth::forwarded::identify;
use web::auth::mcp::bearer;

/// The person a request speaks for, and the token that names them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Caller {
    pub provider: String,
    pub sub: String,
    /// The person's `/mcp` access token, as the client sent it.
    pub token: String,
}

impl Caller {
    /// Whoever `headers` name, when they came from `peer`, `peer` is one of
    /// `gateways`, and the client's token came with them; nobody otherwise.
    pub fn of(headers: &HeaderMap, peer: Option<IpAddr>, gateways: &[IpAddr]) -> Option<Self> {
        if !peer.is_some_and(|p| gateways.contains(&p)) {
            return None;
        }
        let identity = identify(headers).ok()?;
        Some(Self {
            provider: identity.provider,
            sub: identity.sub,
            token: bearer(headers)?.to_string(),
        })
    }

    /// The key this person's sessions are kept under: the first 16 hex digits
    /// of `sha256(provider \0 sub)`, as zend and npcd key an account.
    pub fn key(&self) -> String {
        let digest = Sha256::digest(format!("{}\0{}", self.provider, self.sub).as_bytes());
        digest[..8].iter().map(|b| format!("{b:02x}")).collect()
    }

    /// What names this caller to the gateway: their token.
    pub fn authorization(&self) -> Option<(header::HeaderName, HeaderValue)> {
        let value = HeaderValue::from_str(&format!("Bearer {}", self.token)).ok()?;
        Some((header::AUTHORIZATION, value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use web::auth::forwarded::{PROVIDER, USER};

    fn named() -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(USER, HeaderValue::from_static("boss"));
        h.insert(PROVIDER, HeaderValue::from_static("google"));
        h.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer t0k"),
        );
        h
    }

    /// **The headers are believed from a gateway only, and with the token**:
    /// the same request from any other peer, with no peer known, or without
    /// the token the gateway passed on, names nobody.
    #[test]
    fn a_caller_is_named_only_by_a_listed_gateway_with_its_token() {
        let gw: IpAddr = "192.168.0.5".parse().unwrap();
        let stranger: IpAddr = "192.168.0.99".parse().unwrap();
        let c = Caller::of(&named(), Some(gw), &[gw]).unwrap();
        assert_eq!(
            (c.provider.as_str(), c.sub.as_str(), c.token.as_str()),
            ("google", "boss", "t0k")
        );
        assert!(Caller::of(&named(), Some(stranger), &[gw]).is_none());
        assert!(Caller::of(&named(), None, &[gw]).is_none());
        let mut tokenless = named();
        tokenless.remove(header::AUTHORIZATION);
        assert!(Caller::of(&tokenless, Some(gw), &[gw]).is_none());
    }

    /// The key is zend's and npcd's: sixteen hex digits of provider and
    /// subject, the same for the same person.
    #[test]
    fn the_key_is_sixteen_hex_digits_of_provider_and_subject() {
        let gw: IpAddr = "192.168.0.5".parse().unwrap();
        let c = Caller::of(&named(), Some(gw), &[gw]).unwrap();
        let k = c.key();
        assert_eq!(k.len(), 16);
        assert!(k.bytes().all(|b| b.is_ascii_hexdigit()));
        let other = Caller {
            provider: "github".into(),
            ..c.clone()
        };
        assert_ne!(other.key(), k);
    }

    #[test]
    fn the_caller_is_named_onward_by_its_token() {
        let gw: IpAddr = "192.168.0.5".parse().unwrap();
        let c = Caller::of(&named(), Some(gw), &[gw]).unwrap();
        let (name, value) = c.authorization().unwrap();
        assert_eq!(name, header::AUTHORIZATION);
        assert_eq!(value, "Bearer t0k");
    }
}
