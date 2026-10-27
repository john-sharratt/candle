//! Whose `X-Tokera-*` headers are believed (`docs/mcp_app.md` §2): a
//! request's identity headers are kept only when it came off the cluster
//! socket this app dialled — the gateway at the other end proved the group
//! key, and set them itself — or from a peer listed as a gateway on its
//! listener. From anywhere else they are dropped before anything reads them,
//! so nobody is calling and the client is sent to sign in.

use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;

use axum::extract::{ConnectInfo, Request, State};
use axum::http::HeaderMap;
use axum::middleware::Next;
use axum::response::Response;
use web::cluster::ClusterPeer;
use web::proxy::IDENTITY_HEADERS;

/// Whether a request that came off the socket (`on_socket`), or from `peer`,
/// carries identity headers this app may believe.
pub fn believed(on_socket: bool, peer: Option<IpAddr>, gateways: &[IpAddr]) -> bool {
    on_socket || peer.is_some_and(|p| gateways.contains(&p))
}

/// Drop `headers`' identity unless it is [`believed`].
pub fn strip_unbelieved(
    headers: &mut HeaderMap,
    on_socket: bool,
    peer: Option<IpAddr>,
    gateways: &[IpAddr],
) {
    if !believed(on_socket, peer, gateways) {
        for name in IDENTITY_HEADERS {
            headers.remove(name);
        }
    }
}

/// The layer every request passes before the router: [`strip_unbelieved`],
/// with the socket marked by [`ClusterPeer`].
pub async fn layer(
    State(gateways): State<Arc<[IpAddr]>>,
    mut req: Request,
    next: Next,
) -> Response {
    let on_socket = req.extensions().get::<ClusterPeer>().is_some();
    let peer = req
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ConnectInfo(a)| a.ip());
    strip_unbelieved(req.headers_mut(), on_socket, peer, &gateways);
    next.run(req).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;
    use web::auth::forwarded::{PROVIDER, USER};

    fn named() -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(USER, HeaderValue::from_static("boss"));
        h.insert(PROVIDER, HeaderValue::from_static("google"));
        h.insert("authorization", HeaderValue::from_static("Bearer t0k"));
        h
    }

    /// **The socket and a listed gateway are believed; nobody else is.** What
    /// is dropped is the identity alone: the token rides on, naming nobody
    /// here without the headers beside it.
    #[test]
    fn identity_is_kept_from_the_socket_or_a_listed_gateway_only() {
        let gw: IpAddr = "127.0.0.1".parse().unwrap();
        let stranger: IpAddr = "192.168.0.99".parse().unwrap();
        for (on_socket, peer, kept) in [
            (true, None, true),
            (true, Some(stranger), true),
            (false, Some(gw), true),
            (false, Some(stranger), false),
            (false, None, false),
        ] {
            let mut h = named();
            strip_unbelieved(&mut h, on_socket, peer, &[gw]);
            assert_eq!(
                (h.contains_key(USER), h.contains_key(PROVIDER)),
                (kept, kept),
                "socket {on_socket}, peer {peer:?}"
            );
            assert_eq!(h["authorization"], "Bearer t0k");
        }
    }
}
