//! A request to one member, over a stream on its socket (§5.1): one request
//! per stream, HTTP/1.1 with upgrades, so a WebSocket rides it like any other
//! response.

use std::fmt;

use axum::body::Body;
use cluster::gateway::Hub;
use hyper::body::Incoming;
use hyper::client::conn::http1;
use hyper::{Request, Response};
use hyper_util::rt::TokioIo;

/// Why no response came back.
#[derive(Debug)]
pub enum Unsent {
    /// The member has no socket on this gateway any more.
    Gone,
    /// The stream failed before the response head arrived.
    Failed(hyper::Error),
}

impl fmt::Display for Unsent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Unsent::Gone => f.write_str("the member's socket is gone"),
            Unsent::Failed(e) => write!(f, "the member's stream failed: {e}"),
        }
    }
}

/// Send `req` — its URI in origin form — to `node`, a member of `app`.
pub async fn send(
    hub: &Hub,
    app: &str,
    node: u64,
    req: Request<Body>,
) -> Result<Response<Incoming>, Unsent> {
    let conn = hub.open(app, node, app).ok_or(Unsent::Gone)?;
    let (mut sender, connection) = http1::handshake(TokioIo::new(conn))
        .await
        .map_err(Unsent::Failed)?;
    tokio::spawn(async move {
        // A stream ending is routine; a reset is the member's socket going,
        // which the response body reports to whoever reads it.
        if let Err(e) = connection.with_upgrades().await {
            tracing::debug!(error = %e, "member stream ended");
        }
    });
    sender.send_request(req).await.map_err(Unsent::Failed)
}
