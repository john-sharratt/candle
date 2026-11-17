//! Sending a visitor on the plain entrance to the TLS one
//! (`server.redirect_http`).
//!
//! A request is answered `308` to `https://<its host><its path>` when all of
//! these hold, and is served as it came otherwise:
//!
//! - **It arrived on the plain entrance**, and no proxy in front says the
//!   visitor is already on HTTPS (`X-Forwarded-Proto: https`): a proxy that
//!   ends TLS itself and reaches this gateway over plain HTTP would otherwise
//!   be sent round in a circle.
//! - **Its host is a public name of one of this gateway's sites**: the names
//!   it holds a certificate for. A request to an address, a `.localhost`
//!   name or the gateway's LAN name has no certificate to be sent to.
//! - **It is not the CA's challenge.** Let's Encrypt asks for
//!   `/.well-known/acme-challenge/<token>` over plain HTTP on every name it
//!   validates, and this gateway asks itself the same before it orders; that
//!   path is always answered where it arrived.
//! - **It came from outside.** A request from loopback or a backend this
//!   gateway trusts (`server.trusted_backends`) is a service on the estate
//!   calling the gateway by its plain port, and one a member forwarded up its
//!   socket never crossed the plain entrance at all.
//!
//! `308`, not `301`: a client that posted keeps its method and its body.

use std::collections::HashSet;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;

use axum::extract::{ConnectInfo, Request, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use crate::acme::challenges::PREFIX;
use crate::acme::names::groups;
use crate::cluster::{ClusterPeer, MemberOrigin};
use crate::config::{host_name, Config};
use crate::tls::SecureEntrance;

/// The header a proxy in front states the visitor's scheme on.
const FORWARDED_PROTO: &str = "x-forwarded-proto";

/// The port HTTPS is reached on when a URL names none.
const HTTPS_PORT: u16 = 443;

/// What decides a redirect: the names with a certificate, the TLS entrance's
/// port, and the peers that are the estate's own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    names: HashSet<String>,
    port: u16,
    inside: Vec<IpAddr>,
}

/// One request, as the plan reads it.
#[derive(Debug, Clone, Copy)]
pub struct Arrival<'a> {
    /// It came over the TLS entrance, or a proxy in front says the visitor
    /// is on HTTPS.
    pub secure: bool,
    /// A member forwarded it, or it came over a cluster socket.
    pub from_member: bool,
    pub peer: Option<IpAddr>,
    pub host: Option<&'a str>,
    pub path_and_query: &'a str,
}

impl Plan {
    /// The plan `cfg` asks for: `None` unless it sets `server.redirect_http`
    /// and has a TLS entrance to send a visitor to.
    pub fn of(cfg: &Config) -> Option<Plan> {
        let tls = cfg.server.tls.as_ref()?;
        if !cfg.server.redirect_http {
            return None;
        }
        Some(Plan {
            names: groups(cfg).into_iter().flat_map(|g| g.names).collect(),
            port: tls.bind.port(),
            inside: cfg.server.trusted_backends.clone(),
        })
    }

    /// Where `arrival` is sent, when it is to be redirected.
    pub fn location(&self, arrival: Arrival<'_>) -> Option<String> {
        let path = arrival.path_and_query;
        let inside = arrival
            .peer
            .is_some_and(|ip| ip.is_loopback() || self.inside.contains(&ip));
        if arrival.secure || arrival.from_member || inside || path.starts_with(PREFIX) {
            return None;
        }
        let host = host_name(arrival.host?);
        if !self.names.contains(&host) {
            return None;
        }
        Some(match self.port {
            HTTPS_PORT => format!("https://{host}{path}"),
            port => format!("https://{host}:{port}{path}"),
        })
    }
}

fn said(headers: &HeaderMap, name: impl header::AsHeaderName) -> Option<&str> {
    headers.get(name).and_then(|v| v.to_str().ok())
}

/// The middleware: redirect what the plan says to, and serve the rest.
pub async fn layer(State(plan): State<Arc<Plan>>, req: Request, next: Next) -> Response {
    let forwarded_https = said(req.headers(), FORWARDED_PROTO)
        .is_some_and(|proto| proto.trim().eq_ignore_ascii_case("https"));
    let extensions = req.extensions();
    let arrival = Arrival {
        secure: extensions.get::<SecureEntrance>().is_some() || forwarded_https,
        from_member: extensions.get::<MemberOrigin>().is_some()
            || extensions.get::<ClusterPeer>().is_some(),
        peer: extensions
            .get::<ConnectInfo<SocketAddr>>()
            .map(|ConnectInfo(addr)| addr.ip()),
        host: said(req.headers(), header::HOST),
        path_and_query: req.uri().path_and_query().map_or("/", |p| p.as_str()),
    };
    match plan.location(arrival) {
        Some(to) => (StatusCode::PERMANENT_REDIRECT, [(header::LOCATION, to)]).into_response(),
        None => next.run(req).await,
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    const YAML: &str = r#"
server:
  redirect_http: true
  trusted_backends: ["192.168.1.3"]
  tls: {bind: "0.0.0.0:443", acme: {store: acme}}
sites:
  - name: tokera
    default: true
    hosts: ["tokera.com", "tokera.localhost"]
    roots: ["."]
  - name: game
    hosts: ["battlecities.net"]
    roots: ["."]
"#;

    fn plan() -> Plan {
        Plan::of(&Config::from_yaml(YAML, Path::new(".")).unwrap()).unwrap()
    }

    fn outside<'a>(host: &'a str, path_and_query: &'a str) -> Arrival<'a> {
        Arrival {
            secure: false,
            from_member: false,
            peer: Some("203.0.113.9".parse().unwrap()),
            host: Some(host),
            path_and_query,
        }
    }

    /// **A visitor on the plain entrance is sent to the same address over
    /// HTTPS**, path and query kept, the host as sites are matched by it.
    #[test]
    fn a_plain_visitor_is_sent_to_https_at_the_same_address() {
        let p = plan();
        assert_eq!(
            p.location(outside("battlecities.net", "/api/v1/world?x=1")),
            Some("https://battlecities.net/api/v1/world?x=1".to_string())
        );
        assert_eq!(
            p.location(outside("Tokera.com:80", "/")),
            Some("https://tokera.com/".to_string())
        );
    }

    /// **The CA's challenge is never redirected**, on any name: it is asked
    /// for over plain HTTP, and a certificate that cannot renew takes its
    /// site down.
    #[test]
    fn the_cas_challenge_is_served_where_it_arrived() {
        let p = plan();
        for host in ["tokera.com", "battlecities.net"] {
            let challenge = outside(host, "/.well-known/acme-challenge/tok");
            assert_eq!(p.location(challenge), None, "{host}");
        }
        // Beside it, the rest of `/.well-known` goes to HTTPS like any path.
        assert!(p
            .location(outside(
                "tokera.com",
                "/.well-known/oauth-authorization-server"
            ))
            .is_some());
    }

    /// A request already on HTTPS, one from the estate's own machines, one a
    /// member forwarded, and one to a name with no certificate are served as
    /// they came.
    #[test]
    fn what_is_secure_inside_or_unnamed_is_not_redirected() {
        let p = plan();
        let plain = outside("tokera.com", "/x");
        assert!(p.location(plain).is_some());
        assert_eq!(
            p.location(Arrival {
                secure: true,
                ..plain
            }),
            None
        );
        assert_eq!(
            p.location(Arrival {
                from_member: true,
                ..plain
            }),
            None
        );
        for inside in ["127.0.0.1", "::1", "192.168.1.3"] {
            let peer = Some(inside.parse().unwrap());
            assert_eq!(p.location(Arrival { peer, ..plain }), None, "{inside}");
        }
        for unnamed in [
            "tokera.localhost",
            "192.168.1.2",
            "gateway.tokera.local",
            "evil.example",
        ] {
            assert_eq!(p.location(outside(unnamed, "/x")), None, "{unnamed}");
        }
        assert_eq!(
            p.location(Arrival {
                host: None,
                ..plain
            }),
            None
        );
    }

    /// The redirect names the TLS entrance's port when it is not HTTPS's own,
    /// and there is no plan without the setting or without a TLS entrance.
    #[test]
    fn the_plan_follows_the_configuration() {
        let other = YAML.replace("0.0.0.0:443", "0.0.0.0:9443");
        let p = Plan::of(&Config::from_yaml(&other, Path::new(".")).unwrap()).unwrap();
        assert_eq!(
            p.location(outside("tokera.com", "/x")),
            Some("https://tokera.com:9443/x".to_string())
        );
        let off = YAML.replace("redirect_http: true", "redirect_http: false");
        assert_eq!(
            Plan::of(&Config::from_yaml(&off, Path::new(".")).unwrap()),
            None
        );
        let no_tls = YAML.replace("  tls: {bind: \"0.0.0.0:443\", acme: {store: acme}}\n", "");
        assert_eq!(
            Plan::of(&Config::from_yaml(&no_tls, Path::new(".")).unwrap()),
            None
        );
    }
}
