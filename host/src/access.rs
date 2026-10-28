//! Who a caller is — one rule for every tenant (`docs/shared_model_host.md`
//! §3.5).
//!
//! The gateway in front of the host signs people in and forwards who they are
//! on the `x-tokera-*` headers ([`web::auth::forwarded`]). The server clears
//! those headers from every peer that is not a trusted gateway
//! ([`web::auth::Gateways`], installed with [`web::Builder::trust`]) before any
//! handler sees the request, so what a tenant reads was sent by the gateway or
//! is absent. Every tenant reads it the same way — [`identify`] — and resolves
//! the role against the estate's one table ([`ROLES`]).
//!
//! # `--local-signin` — standing in for a gateway that isn't there
//!
//! A developer running a daemon directly, with no gateway in front of it, has
//! no way to arrive with `x-tokera-*` headers and so is always
//! [`Role::Unauthenticated`] — there is deliberately no API that grants a role.
//! `--local-signin <email>` is the one boot-time way out: [`LocalSignin`] writes
//! that identity onto a request whose peer is genuine loopback (this machine,
//! not a configured gateway) and that carries no forwarded identity of its own,
//! exactly as the gateway would have. Any real forwarded identity always takes
//! precedence. Because it is written as the gateway's headers, every tenant
//! sees it with no code of its own. It is still a widening of trust on this box
//! — any other local process or account can now reach the port and be
//! recognized as that email, and so can every remote caller of anything that
//! relays onto loopback (a tunnel, a proxy on this box pointed at
//! `127.0.0.1`) — which is why it is off by default, refused beside a
//! `--gateway`, and logged loudly at startup when set.

use std::net::SocketAddr;

use axum::extract::{ConnectInfo, Request};
use axum::http::{HeaderMap, HeaderValue};
use axum::middleware::Next;
use axum::response::Response;
use web::auth::forwarded::{identify, EMAIL, PROVIDER, USER};
use web::auth::gateways::canonical;
use web::auth::{Role, Roles};

/// The estate's role table: the people who may use the estate, each with a
/// role, and anyone else signed in a guest. Compiled in — there is no API that
/// grants a role, and whoever can edit this file and restart the process
/// already holds what the roles grant.
pub const ROLES: &str = include_str!("../roles.yaml");

/// [`ROLES`], parsed.
pub fn roles() -> Roles {
    serde_yaml::from_str(ROLES).expect("host/roles.yaml is valid")
}

/// The caller's role: the identity the request carries, against `roles`.
pub fn role(headers: &HeaderMap, roles: &Roles) -> Role {
    roles.of(identify(headers).ok().as_ref())
}

/// The issuer and subject `--local-signin` presents. A fixed, non-empty
/// subject: [`Roles::of`] treats an empty one as no identity at all, and this
/// identity is deliberately real.
const LOCAL_SIGNIN: &str = "local-signin";

/// `--local-signin`, as a request rewrite.
#[derive(Debug, Clone, Default)]
pub struct LocalSignin(Option<HeaderValue>);

impl LocalSignin {
    /// Stand in for the gateway as `email`, or not at all. An email that cannot
    /// be a header value is refused rather than dropped.
    pub fn new(email: Option<&str>) -> anyhow::Result<Self> {
        match email {
            None => Ok(Self(None)),
            Some(e) => Ok(Self(Some(HeaderValue::from_str(e).map_err(|_| {
                anyhow::anyhow!("--local-signin {e:?} is not a usable email address")
            })?))),
        }
    }

    /// Write the stand-in identity onto `headers` when `peer` is loopback and
    /// no identity was forwarded.
    pub fn apply(&self, headers: &mut HeaderMap, peer: Option<SocketAddr>) {
        let Some(email) = &self.0 else {
            return;
        };
        let loopback = peer.is_some_and(|p| canonical(p.ip()).is_loopback());
        if !loopback || identify(headers).is_ok() {
            return;
        }
        headers.insert(USER, HeaderValue::from_static(LOCAL_SIGNIN));
        headers.insert(PROVIDER, HeaderValue::from_static(LOCAL_SIGNIN));
        headers.insert(EMAIL, email.clone());
    }

    /// The middleware: [`Self::apply`] on every request.
    pub async fn layer(self, mut req: Request, next: Next) -> Response {
        let peer = req
            .extensions()
            .get::<ConnectInfo<SocketAddr>>()
            .map(|ConnectInfo(a)| *a);
        self.apply(req.headers_mut(), peer);
        next.run(req).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peer(s: &str) -> Option<SocketAddr> {
        Some(s.parse().unwrap())
    }

    fn table(yaml: &str) -> Roles {
        serde_yaml::from_str(yaml).unwrap()
    }

    fn signed_in(email: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(USER, "g-1".parse().unwrap());
        h.insert(PROVIDER, "google".parse().unwrap());
        h.insert(EMAIL, email.parse().unwrap());
        h
    }

    fn with_signin(email: &str, headers: HeaderMap, from: Option<SocketAddr>) -> HeaderMap {
        let mut h = headers;
        LocalSignin::new(Some(email)).unwrap().apply(&mut h, from);
        h
    }

    /// The compiled-in table parses and names the estate's owner at the top.
    #[test]
    fn the_estate_table_parses_and_names_its_creator() {
        let h = with_signin(
            "johnathan.sharratt@gmail.com",
            HeaderMap::new(),
            peer("127.0.0.1:1"),
        );
        assert_eq!(role(&h, &roles()), Role::Creator);
    }

    /// The role comes from the gateway's headers; a request without them is
    /// anonymous.
    #[test]
    fn the_role_is_read_from_the_gateways_headers() {
        let t = table(
            "people:\n  - email: admin@example.com\n    role: admin\n  - email: friend@example.com\n    role: user\n",
        );
        assert_eq!(role(&HeaderMap::new(), &t), Role::Unauthenticated);
        assert_eq!(role(&signed_in("someone@example.com"), &t), Role::Guest);
        assert_eq!(role(&signed_in("friend@example.com"), &t), Role::User);
        assert_eq!(role(&signed_in("admin@example.com"), &t), Role::Admin);
    }

    /// A loopback caller with no forwarded identity is `--local-signin`'s email,
    /// written as the gateway would have written it.
    #[test]
    fn local_signin_names_a_loopback_caller_with_no_headers() {
        let t = table("people:\n  - email: me@example.com\n    role: creator\n");
        for from in ["127.0.0.1:9", "[::1]:9", "[::ffff:127.0.0.1]:9"] {
            let h = with_signin("me@example.com", HeaderMap::new(), peer(from));
            assert_eq!(role(&h, &t), Role::Creator, "{from}");
            assert_eq!(h[USER], "local-signin");
            assert_eq!(h[PROVIDER], "local-signin");
            assert_eq!(h[EMAIL], "me@example.com");
        }
        let h = with_signin("nobody@example.com", HeaderMap::new(), peer("127.0.0.1:9"));
        assert_eq!(role(&h, &t), Role::Guest);
    }

    /// A real forwarded identity always wins over the flag.
    #[test]
    fn local_signin_never_overrides_a_forwarded_identity() {
        let h = with_signin(
            "me@example.com",
            signed_in("other@example.com"),
            peer("127.0.0.1:9"),
        );
        assert_eq!(h[EMAIL], "other@example.com");
        assert_eq!(h[PROVIDER], "google");
    }

    /// **The flag must not reach past loopback** — not a gateway, not a LAN
    /// peer, not a request with no known peer.
    #[test]
    fn local_signin_applies_to_loopback_only() {
        for from in [peer("192.168.0.5:9"), peer("10.0.0.9:9"), None] {
            let h = with_signin("me@example.com", HeaderMap::new(), from);
            assert!(h.get(EMAIL).is_none(), "{from:?}");
        }
    }

    /// Off, nothing is written.
    #[test]
    fn without_the_flag_nothing_is_written() {
        let mut h = HeaderMap::new();
        LocalSignin::default().apply(&mut h, peer("127.0.0.1:9"));
        assert!(h.is_empty());
        assert!(LocalSignin::new(Some("bad\nvalue")).is_err());
    }
}
