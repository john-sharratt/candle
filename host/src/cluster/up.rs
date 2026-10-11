//! Requests between members (`docs/replicated_substrate.md` §9.3, §9.6): a
//! member sends one up a stream to a gateway, which routes it by the route's
//! dispatch — to a world's authority, a character's holder — and stamps it
//! with the member it came from. The member it reaches believes that stamp
//! only on a request that arrived over a cluster socket.

use axum::body::{Body, Bytes};
use axum::extract::Request;
use axum::http::{header, Method};
use cluster::member::{AppMember, FROM_MEMBER};
use http_body_util::BodyExt;
use hyper::client::conn::http1;
use hyper_util::rt::TokioIo;
use web::cluster::ClusterPeer;

/// Send `method path` with a JSON `body` up to a gateway, for it to route to
/// whichever member the route names; its status and body.
pub async fn request_up(
    member: &AppMember,
    method: Method,
    path: &str,
    body: Vec<u8>,
) -> anyhow::Result<(u16, Bytes)> {
    let conn = member
        .open_up()
        .ok_or_else(|| anyhow::anyhow!("no socket to a gateway is open"))?;
    let (mut sender, connection) = http1::handshake(TokioIo::new(conn)).await?;
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            tracing::debug!(error = %e, "a member's request stream ended");
        }
    });
    let req = Request::builder()
        .method(method)
        .uri(path)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body))?;
    let res = sender.send_request(req).await?;
    let status = res.status().as_u16();
    let bytes = res.into_body().collect().await?.to_bytes();
    Ok((status, bytes))
}

/// The member a request came from: named by the gateway that carried it, on
/// a request that arrived over a cluster socket — `None` for anything else,
/// whatever its headers say.
pub fn from_member(req: &Request) -> Option<u64> {
    req.extensions().get::<ClusterPeer>()?;
    let hex = req.headers().get(FROM_MEMBER)?.to_str().ok()?;
    u64::from_str_radix(hex, 16).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(peer: bool, header: Option<&str>) -> Request {
        let mut b = Request::builder().uri("/v1/world/w1/act");
        if let Some(h) = header {
            b = b.header(FROM_MEMBER, h);
        }
        let mut req = b.body(Body::empty()).unwrap();
        if peer {
            req.extensions_mut().insert(ClusterPeer { cluster: 0 });
        }
        req
    }

    /// Believed only over a socket, and only as a node.
    #[test]
    fn the_stamp_is_believed_over_a_socket_only() {
        assert_eq!(
            from_member(&request(true, Some("00000000000000ab"))),
            Some(0xab)
        );
        assert_eq!(from_member(&request(false, Some("00000000000000ab"))), None);
        assert_eq!(from_member(&request(true, None)), None);
        assert_eq!(from_member(&request(true, Some("not a node"))), None);
    }
}
