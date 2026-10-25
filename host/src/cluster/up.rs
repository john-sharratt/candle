//! Requests between members (`docs/replicated_substrate.md` §9.3, §9.6): a
//! member sends one up a stream to a gateway, which routes it by the route's
//! dispatch — to a world's authority, a character's holder — and stamps it
//! with the member it came from. The member it reaches believes that stamp
//! only on a request that arrived over a cluster socket.

use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::extract::Request;
use axum::http::{header, HeaderName, HeaderValue, Method};
use bytes::BytesMut;
use cluster::member::{AppMember, FROM_MEMBER};
use cluster::session::VirtualConn;
use http_body_util::BodyExt;
use hyper::client::conn::http1;
use hyper_util::rt::TokioIo;
use tokio::time::{timeout_at, Instant};
use web::cluster::ClusterPeer;

/// Send `method path` with a JSON `body` up to a gateway, for it to route to
/// whichever member the route names; its status and body.
pub async fn request_up(
    member: &AppMember,
    method: Method,
    path: &str,
    body: Vec<u8>,
) -> anyhow::Result<(u16, Bytes)> {
    let reply = send_up(member, method, path, &[], body, None).await?;
    Ok((reply.status, reply.body))
}

/// What a gateway answered a request sent up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reply {
    pub status: u16,
    pub body: Bytes,
    /// Whether the body ended; `false` when reading stopped at its deadline —
    /// a stream still running, read as far as it had come.
    pub complete: bool,
}

/// [`request_up`] carrying `headers` — the caller's identity, the member it
/// names — and, given `read_for`, reading the body no longer than that: a
/// streamed answer is read as far as it has come, and said so.
pub async fn send_up(
    member: &AppMember,
    method: Method,
    path: &str,
    headers: &[(HeaderName, HeaderValue)],
    body: Vec<u8>,
    read_for: Option<Duration>,
) -> anyhow::Result<Reply> {
    let conn = member
        .open_up()
        .ok_or_else(|| anyhow::anyhow!("no socket to a gateway is open"))?;
    send_over(conn, method, path, headers, body, read_for).await
}

/// [`send_up`] towards `group:<app>` rather than this member's own app: the
/// gateway sends it to the member of `app` it names — an agent, a machine's
/// keep.
pub async fn send_to_group(
    member: &AppMember,
    app: &str,
    method: Method,
    path: &str,
    headers: &[(HeaderName, HeaderValue)],
    body: Vec<u8>,
    read_for: Option<Duration>,
) -> anyhow::Result<Reply> {
    let conn = member
        .open_group(app)
        .ok_or_else(|| anyhow::anyhow!("no socket to a gateway is open"))?;
    send_over(conn, method, path, headers, body, read_for).await
}

/// One request on a stream to a gateway, its body read no longer than
/// `read_for`.
async fn send_over(
    conn: VirtualConn,
    method: Method,
    path: &str,
    headers: &[(HeaderName, HeaderValue)],
    body: Vec<u8>,
    read_for: Option<Duration>,
) -> anyhow::Result<Reply> {
    let (mut sender, connection) = http1::handshake(TokioIo::new(conn)).await?;
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            tracing::debug!(error = %e, "a member's request stream ended");
        }
    });
    let mut builder = Request::builder()
        .method(method)
        .uri(path)
        .header(header::CONTENT_TYPE, "application/json");
    for (name, value) in headers {
        builder = builder.header(name, value);
    }
    let req = builder.body(Body::from(body))?;
    let deadline = read_for.map(|d| Instant::now() + d);
    let res = match deadline {
        Some(at) => timeout_at(at, sender.send_request(req))
            .await
            .map_err(|_| anyhow::anyhow!("the gateway did not answer in time"))??,
        None => sender.send_request(req).await?,
    };
    let status = res.status().as_u16();
    let mut body = res.into_body();
    let mut read = BytesMut::new();
    loop {
        let frame = match deadline {
            Some(at) => match timeout_at(at, body.frame()).await {
                Ok(frame) => frame,
                Err(_) => {
                    return Ok(Reply {
                        status,
                        body: read.freeze(),
                        complete: false,
                    })
                }
            },
            None => body.frame().await,
        };
        match frame {
            Some(frame) => {
                if let Ok(data) = frame?.into_data() {
                    read.extend_from_slice(&data);
                }
            }
            None => {
                return Ok(Reply {
                    status,
                    body: read.freeze(),
                    complete: true,
                })
            }
        }
    }
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
