//! The cluster socket's WebSocket upgrade (§3.1), made here rather than by
//! axum's `WebSocket`, which keeps its transport to itself: the upgraded
//! connection is wrapped in a [`Stamped`] transport, so the hub's session
//! hears a member by the bytes of a frame still arriving (§3.2, keep-alive). A
//! mebibyte fragment over a link of a few kilobytes a second takes minutes to
//! arrive whole, from a member alive throughout.

use std::future::Future;

use axum::extract::Request;
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use cluster::pipe::socket::{split, SocketRx, SocketTx};
use cluster::pipe::tungstenite::Tungstenite;
use cluster::pipe::{Heard, Stamped};
use hyper::upgrade::Upgraded;
use hyper_util::rt::TokioIo;
use tokio_tungstenite::tungstenite::handshake::derive_accept_key;
use tokio_tungstenite::tungstenite::protocol::{Role, WebSocketConfig};
use tokio_tungstenite::WebSocketStream;

/// The receiving half of an accepted cluster socket.
pub type ClusterRx = SocketRx<WebSocketStream<Stamped<TokioIo<Upgraded>>>, Tungstenite>;

/// Whether a comma-separated header names `token`, ignoring case.
fn lists(headers: &HeaderMap, name: header::HeaderName, token: &str) -> bool {
    headers
        .get_all(name)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(','))
        .any(|t| t.trim().eq_ignore_ascii_case(token))
}

/// The `Sec-WebSocket-Accept` for a valid upgrade request, or why it is not
/// one.
fn accept_key(headers: &HeaderMap) -> Result<String, &'static str> {
    if !lists(headers, header::CONNECTION, "upgrade") {
        return Err("no `Connection: upgrade`");
    }
    if !lists(headers, header::UPGRADE, "websocket") {
        return Err("no `Upgrade: websocket`");
    }
    if headers
        .get(header::SEC_WEBSOCKET_VERSION)
        .is_none_or(|v| v.as_bytes() != b"13")
    {
        return Err("not WebSocket version 13");
    }
    let key = headers
        .get(header::SEC_WEBSOCKET_KEY)
        .ok_or("no `Sec-WebSocket-Key`")?;
    Ok(derive_accept_key(key.as_bytes()))
}

/// Answer `req`'s upgrade, and once the connection is upgraded run `serve` on
/// the socket's two halves, messages up to `max_message` bytes.
pub fn upgrade<F, Fut>(mut req: Request, max_message: usize, serve: F) -> Response
where
    F: FnOnce(SocketTx, ClusterRx) -> Fut + Send + 'static,
    Fut: Future<Output = ()> + Send + 'static,
{
    let accept = match accept_key(req.headers()) {
        Ok(a) => a,
        Err(why) => {
            return (
                StatusCode::BAD_REQUEST,
                format!("the cluster socket is a WebSocket upgrade: {why}"),
            )
                .into_response()
        }
    };
    let on_upgrade = hyper::upgrade::on(&mut req);
    tokio::spawn(async move {
        let upgraded = match on_upgrade.await {
            Ok(u) => u,
            Err(e) => {
                tracing::debug!(error = %e, "the cluster socket's upgrade failed");
                return;
            }
        };
        let heard = Heard::new();
        let transport = Stamped::new(TokioIo::new(upgraded), heard.clone());
        let config = WebSocketConfig {
            max_message_size: Some(max_message),
            max_frame_size: Some(max_message),
            ..WebSocketConfig::default()
        };
        let ws = WebSocketStream::from_raw_socket(transport, Role::Server, Some(config)).await;
        let (tx, rx) = split::<Tungstenite, _, _>(ws, heard);
        serve(tx, rx).await;
    });
    let mut res = StatusCode::SWITCHING_PROTOCOLS.into_response();
    let h = res.headers_mut();
    h.insert(header::CONNECTION, HeaderValue::from_static("upgrade"));
    h.insert(header::UPGRADE, HeaderValue::from_static("websocket"));
    match HeaderValue::from_str(&accept) {
        Ok(v) => {
            h.insert(header::SEC_WEBSOCKET_ACCEPT, v);
        }
        Err(_) => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(pairs: &[(header::HeaderName, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.append(k.clone(), HeaderValue::from_str(v).unwrap());
        }
        h
    }

    /// RFC 6455 §1.3's example key and accept value.
    #[test]
    fn a_valid_upgrade_is_accepted_with_the_rfcs_key() {
        let h = headers(&[
            (header::CONNECTION, "keep-alive, Upgrade"),
            (header::UPGRADE, "WebSocket"),
            (header::SEC_WEBSOCKET_VERSION, "13"),
            (header::SEC_WEBSOCKET_KEY, "dGhlIHNhbXBsZSBub25jZQ=="),
        ]);
        assert_eq!(
            accept_key(&h).as_deref(),
            Ok("s3pPLMBiTxaQ9kYGzzhZRbK+xOo=")
        );
    }

    #[test]
    fn a_request_that_is_not_an_upgrade_is_refused() {
        let full = [
            (header::CONNECTION, "Upgrade"),
            (header::UPGRADE, "websocket"),
            (header::SEC_WEBSOCKET_VERSION, "13"),
            (header::SEC_WEBSOCKET_KEY, "dGhlIHNhbXBsZSBub25jZQ=="),
        ];
        for missing in 0..full.len() {
            let some: Vec<_> = full
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != missing)
                .map(|(_, p)| p.clone())
                .collect();
            assert!(accept_key(&headers(&some)).is_err(), "without {missing}");
        }
        let old = headers(&[
            (header::CONNECTION, "Upgrade"),
            (header::UPGRADE, "websocket"),
            (header::SEC_WEBSOCKET_VERSION, "8"),
            (header::SEC_WEBSOCKET_KEY, "dGhlIHNhbXBsZSBub25jZQ=="),
        ]);
        assert!(accept_key(&old).is_err());
    }
}
