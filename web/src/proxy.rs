//! Reverse proxy, with health gating.
//!
//! Three things it must get right:
//!
//!   * **Streaming** — SSE bodies are piped, never collected, so a stream that
//!     stays open for an hour costs nothing here.
//!   * **Upgrades** — on a 101 it stops speaking HTTP and becomes a byte tunnel
//!     between the two upgraded connections. That is how `/ws` works.
//!   * **Backoff** — a down upstream is not retried on every request. See
//!     [`crate::health`]; requests inside the window fail fast, and one probe
//!     is released when it expires.

use std::net::IpAddr;
use std::time::Duration;

use axum::body::Body;
use axum::extract::Request;
use axum::http::{header, HeaderValue, StatusCode, Uri};
use axum::response::Response;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;

use crate::errors::{self, Problem};
use crate::health::{Gate, Health};

pub type HttpClient = Client<HttpConnector, Body>;

pub fn client(connect_timeout: Duration) -> HttpClient {
    let mut connector = HttpConnector::new();
    connector.set_nodelay(true); // SSE frames must not wait on Nagle
    connector.set_connect_timeout(Some(connect_timeout));
    connector.enforce_http(false);
    Client::builder(TokioExecutor::new())
        .pool_idle_timeout(Duration::from_secs(30))
        .build(connector)
}

/// Hop-by-hop headers belong to one connection and are never forwarded
/// (RFC 9110 §7.6.1). `Upgrade` is handled explicitly rather than stripped —
/// tunnelling is the whole point for `/ws`.
const HOP_BY_HOP: [header::HeaderName; 7] = [
    header::CONNECTION,
    header::PROXY_AUTHENTICATE,
    header::PROXY_AUTHORIZATION,
    header::TE,
    header::TRAILER,
    header::TRANSFER_ENCODING,
    header::HeaderName::from_static("keep-alive"),
];

/// Identity headers the gateway sets on every forwarded request.
///
/// **These are the daemons' only source of identity**, which makes stripping
/// whatever the client sent non-negotiable: an inbound `X-Tokera-Email` that
/// survived would let anyone be anyone. They are cleared unconditionally and
/// re-set only from a session this gateway itself verified.
///
/// The daemons are behind the gateway on private addresses, so trusting the
/// headers is sound. A daemon that would rather verify than trust can check
/// `X-Tokera-Assertion` — the signed session token itself — against the same
/// key the gateway signs with.
const IDENTITY_HEADERS: [&str; 5] = [
    "x-tokera-user",
    "x-tokera-email",
    "x-tokera-name",
    "x-tokera-picture",
    "x-tokera-assertion",
];

pub struct Forward<'a> {
    pub client: &'a HttpClient,
    pub health: &'a Health,
    pub upstream: &'a str,
    pub rewrite_host: bool,
    pub peer: Option<IpAddr>,
    pub want_html: bool,
    /// Who the gateway decided this request is, if anyone.
    pub identity: Option<&'a crate::auth::Identity>,
    /// The signed session token, for a daemon that verifies rather than trusts.
    pub assertion: Option<&'a str>,
}

pub async fn forward(f: Forward<'_>, req: Request) -> Response {
    // Fail fast while the backoff window is open — never hold a socket against
    // a machine we already know is not answering.
    if let Gate::Blocked {
        retry_after,
        last_error,
    } = f.health.gate(f.upstream)
    {
        let detail = match last_error {
            Some(e) => format!("{} is not answering ({e})", f.upstream),
            None => format!("{} is not answering", f.upstream),
        };
        return errors::respond(Problem::backing_off(detail, retry_after), f.want_html);
    }

    let (mut parts, body) = req.into_parts();
    let pq = parts
        .uri
        .path_and_query()
        .map(|p| p.as_str())
        .unwrap_or("/");
    let target = format!("{}{pq}", f.upstream.trim_end_matches('/'));
    let uri: Uri = match target.parse() {
        Ok(u) => u,
        Err(e) => {
            return errors::respond(
                Problem::upstream_down(format!("bad upstream URI `{target}`: {e}"), None),
                f.want_html,
            )
        }
    };

    let is_upgrade = parts.headers.contains_key(header::UPGRADE);
    // Take the client's upgrade handle BEFORE the parts are consumed, or the
    // 101 arrives with nothing to tunnel to.
    let client_upgrade = parts.extensions.remove::<hyper::upgrade::OnUpgrade>();
    let orig_host = parts
        .headers
        .get(header::HOST)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);

    let mut headers = parts.headers.clone();
    for h in HOP_BY_HOP {
        headers.remove(&h);
    }
    // Strip first, always — before deciding whether we have anyone to name.
    // Anything the client sent under these names is a forgery attempt by
    // construction, since only this process may set them.
    for name in IDENTITY_HEADERS {
        headers.remove(name);
    }
    if let Some(id) = f.identity {
        for (name, value) in [
            ("x-tokera-user", id.sub.as_str()),
            ("x-tokera-email", id.email.as_str()),
            ("x-tokera-name", id.name.as_str()),
            ("x-tokera-picture", id.picture.as_str()),
        ] {
            // A name can hold anything a provider allows, including bytes no
            // header may carry. Skipping one field is the right failure: it
            // must never be able to inject a second header.
            if let Ok(v) = HeaderValue::from_str(value) {
                headers.insert(header::HeaderName::from_static(name), v);
            }
        }
        if let Some(v) = f.assertion.and_then(|t| HeaderValue::from_str(t).ok()) {
            headers.insert(header::HeaderName::from_static("x-tokera-assertion"), v);
        }
    }
    if is_upgrade {
        // Hop-by-hop, but exactly what the upstream must see for a websocket.
        headers.insert(header::CONNECTION, HeaderValue::from_static("upgrade"));
    }
    if f.rewrite_host {
        if let Some(v) = uri
            .authority()
            .and_then(|a| HeaderValue::from_str(a.as_str()).ok())
        {
            headers.insert(header::HOST, v);
        }
    }
    if let Some(v) = orig_host
        .as_deref()
        .and_then(|h| HeaderValue::from_str(h).ok())
    {
        headers.insert(header::HeaderName::from_static("x-forwarded-host"), v);
    }
    headers.insert(
        header::HeaderName::from_static("x-forwarded-proto"),
        HeaderValue::from_static("http"),
    );
    if let Some(v) = f
        .peer
        .and_then(|ip| HeaderValue::from_str(&ip.to_string()).ok())
    {
        headers.insert(header::HeaderName::from_static("x-forwarded-for"), v);
    }

    let mut up_req = hyper::Request::builder()
        .method(parts.method.clone())
        .uri(uri)
        .body(body)
        .expect("parts came from a valid request");
    *up_req.headers_mut() = headers;

    let mut up_res = match f.client.request(up_req).await {
        Ok(r) => r,
        Err(e) => {
            let retry = f.health.on_failure(f.upstream, &e.to_string());
            return errors::respond(
                Problem::upstream_down(
                    format!("{} is not answering: {e}", f.upstream),
                    Some(retry),
                ),
                f.want_html,
            );
        }
    };
    // Reaching the upstream at all counts — a 500 from a live daemon is the
    // daemon's answer to pass through, not a reason to take the route out of
    // service. Only transport failures open the window.
    f.health.on_success(f.upstream);

    if up_res.status() == StatusCode::SWITCHING_PROTOCOLS {
        let upstream_upgrade = up_res
            .extensions_mut()
            .remove::<hyper::upgrade::OnUpgrade>();
        match (client_upgrade, upstream_upgrade) {
            (Some(cu), Some(uu)) => {
                tokio::spawn(async move {
                    match tokio::try_join!(cu, uu) {
                        Ok((c, u)) => {
                            let mut c = hyper_util::rt::TokioIo::new(c);
                            let mut u = hyper_util::rt::TokioIo::new(u);
                            // A websocket ending is routine, so this is debug.
                            if let Err(e) = tokio::io::copy_bidirectional(&mut c, &mut u).await {
                                tracing::debug!(error = %e, "tunnel closed");
                            }
                        }
                        Err(e) => tracing::warn!(error = %e, "upgrade handshake failed"),
                    }
                });
            }
            _ => tracing::warn!("101 from upstream with no upgrade handle on one side"),
        }
    }

    let upgraded = up_res.status() == StatusCode::SWITCHING_PROTOCOLS;
    let (rp, rb) = up_res.into_parts();
    let mut out = Response::from_parts(rp, Body::new(rb)); // piped, not collected
    for h in HOP_BY_HOP {
        out.headers_mut().remove(&h);
    }
    if upgraded {
        // `Connection` is hop-by-hop and was just stripped with the rest — but
        // on a 101 it is the header that makes the handshake valid, and a
        // client that does not see it rejects the response outright rather than
        // switching protocols. It describes this hop, so we state it for this
        // hop instead of forwarding the upstream's.
        out.headers_mut()
            .insert(header::CONNECTION, HeaderValue::from_static("upgrade"));
    }
    out
}
