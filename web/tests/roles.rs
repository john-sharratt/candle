//! End-to-end tests for both roles of this crate, over real sockets.
//!
//! The unit tests cover the pieces in isolation. What matters here is that the
//! assembled server behaves the same whether the API is a router merged into
//! this process or a daemon on another box — because that equivalence is the
//! whole reason the two deployments share one crate. Each test binds port 0 and
//! speaks HTTP to itself, so nothing is mocked below the socket.

use std::net::SocketAddr;
use std::path::Path;
use std::time::{Duration, Instant};

use axum::routing::{get, post};
use axum::Router;
use web::{Builder, Config};

/// Serve a router on an ephemeral port and return its address. The task is
/// detached: it dies with the test process, which is what we want.
async fn spawn(router: Router) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(
            listener,
            router.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await
        .unwrap();
    });
    addr
}

struct Res {
    status: u16,
    content_type: String,
    body: String,
}

/// A minimal HTTP/1.1 client. Using `hyper` directly rather than the crate's
/// own proxy client keeps the test from passing because both sides share a bug.
async fn get_with(addr: SocketAddr, path: &str, host: &str, accept: &str) -> Res {
    use http_body_util::BodyExt;
    use hyper::Request;

    let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let io = hyper_util::rt::TokioIo::new(stream);
    let (mut sender, conn) = hyper::client::conn::http1::handshake(io).await.unwrap();
    tokio::spawn(async move {
        let _ = conn.await;
    });

    let req = Request::builder()
        .uri(path)
        .header("host", host)
        .header("accept", accept)
        .body(axum::body::Body::empty())
        .unwrap();
    let res = sender.send_request(req).await.unwrap();
    let status = res.status().as_u16();
    let content_type = res
        .headers()
        .get(hyper::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let body = String::from_utf8_lossy(&res.into_body().collect().await.unwrap().to_bytes()).into();
    Res {
        status,
        content_type,
        body,
    }
}

async fn fetch(addr: SocketAddr, path: &str) -> Res {
    get_with(addr, path, "127.0.0.1", "*/*").await
}

fn content_dir() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn cfg(yaml: &str) -> Config {
    Config::from_yaml(yaml, content_dir()).expect("test config is valid")
}

/// The API a daemon merges in. Three shapes worth exercising: a plain JSON GET,
/// a POST so the method and body actually reach the router, and a route that
/// reports the identity headers it was handed.
fn fake_api() -> Router {
    Router::new()
        .route(
            "/v1/status",
            get(|| async { axum::Json(serde_json::json!({"state": "ready"})) }),
        )
        .route("/v1/echo", post(|body: String| async move { body }))
        .route(
            "/v1/saw-identity",
            get(|headers: axum::http::HeaderMap| async move {
                // Exactly what a daemon does with the documented contract: read
                // the header and believe it.
                let seen = |n: &str| {
                    headers
                        .get(n)
                        .and_then(|v| v.to_str().ok())
                        .unwrap_or("")
                        .to_string()
                };
                axum::Json(serde_json::json!({
                    "user": seen("x-tokera-user"),
                    "email": seen("x-tokera-email"),
                    "assertion": seen("x-tokera-assertion"),
                }))
            }),
        )
}

/// A `GET` carrying caller-chosen headers, for the forgery test below.
async fn get_with_headers(addr: SocketAddr, path: &str, extra: &[(&str, &str)]) -> Res {
    use http_body_util::BodyExt;
    use hyper::Request;

    let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let io = hyper_util::rt::TokioIo::new(stream);
    let (mut sender, conn) = hyper::client::conn::http1::handshake(io).await.unwrap();
    tokio::spawn(async move {
        let _ = conn.await;
    });

    let mut req = Request::builder()
        .uri(path)
        .header("host", "127.0.0.1")
        .header("accept", "*/*");
    for (k, v) in extra {
        req = req.header(*k, *v);
    }
    let res = sender
        .send_request(req.body(axum::body::Body::empty()).unwrap())
        .await
        .unwrap();
    let status = res.status().as_u16();
    let content_type = res
        .headers()
        .get(hyper::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let body = String::from_utf8_lossy(&res.into_body().collect().await.unwrap().to_bytes()).into();
    Res {
        status,
        content_type,
        body,
    }
}

const AUTHORITATIVE: &str = r#"
sites:
  - name: npcd
    default: true
    roots: ["content/npcd", "content/common"]
    fallback: "index.html"
    api:
      - {prefix: /v1, upstream: local}
"#;

// ── authoritative: content + in-process API ────────────────────────────────

/// A client cannot assert an identity to an API merged into this process.
///
/// The equivalence this file exists to defend cuts both ways: an `upstream:
/// local` route must be exactly as safe as a proxied one. It was not. The
/// gateway stripped inbound `x-tokera-*` in `proxy::forward`, which a local
/// route never reaches — so the daemon's router was called with the client's
/// own headers, and a handler written against the documented contract believed
/// `x-tokera-email: admin@tokera.com` from anyone. The identical code behind the
/// proxy was safe, which is what made it invisible.
#[tokio::test]
async fn a_client_cannot_forge_identity_headers_into_a_local_api() {
    let addr = spawn(
        Builder::new(cfg(AUTHORITATIVE))
            .local_api("npcd", fake_api())
            .router(),
    )
    .await;

    let r = get_with_headers(
        addr,
        "/v1/saw-identity",
        &[
            ("x-tokera-user", "root"),
            ("x-tokera-email", "admin@tokera.com"),
            ("x-tokera-assertion", "made-up"),
        ],
    )
    .await;

    assert_eq!(r.status, 200, "{}", r.body);
    assert!(r.body.contains(r#""user":"""#), "{}", r.body);
    assert!(!r.body.contains("admin@tokera.com"), "{}", r.body);
    assert!(!r.body.contains("made-up"), "{}", r.body);
}

/// A daemon behind the gateway does receive them — that is how it learns who
/// is calling.
///
/// The mirror of the test above, and the pair is the point. Identity crosses an
/// in-process boundary as a request extension and a network boundary as these
/// headers, so a daemon on another box has nothing else to read. Clearing them
/// for everyone makes the documented contract unimplementable; clearing them
/// for everyone who has not declared `behind_gateway` makes it safe by default
/// and possible on purpose.
#[tokio::test]
async fn a_daemon_behind_the_gateway_reads_the_identity_it_is_sent() {
    let addr = spawn(
        Builder::new(cfg(AUTHORITATIVE))
            .behind_gateway()
            .local_api("npcd", fake_api())
            .router(),
    )
    .await;

    let r = get_with_headers(
        addr,
        "/v1/saw-identity",
        &[
            ("x-tokera-user", "google-1"),
            ("x-tokera-email", "wren@example.com"),
        ],
    )
    .await;

    assert_eq!(r.status, 200, "{}", r.body);
    assert!(r.body.contains(r#""user":"google-1""#), "{}", r.body);
    assert!(r.body.contains("wren@example.com"), "{}", r.body);
}

/// The declaration is opt-in, so the safe behaviour is what you get by
/// forgetting it — not what you get by remembering.
///
/// Worth asserting rather than assuming: this is the direction a refactor
/// breaks silently, since flipping the default turns every existing caller into
/// an open door and no test of the *new* behaviour would notice.
#[tokio::test]
async fn stripping_is_the_default_and_must_stay_that_way() {
    let addr = spawn(
        Builder::new(cfg(AUTHORITATIVE))
            .local_api("npcd", fake_api())
            .router(),
    )
    .await;

    let r = get_with_headers(addr, "/v1/saw-identity", &[("x-tokera-user", "root")]).await;
    assert!(r.body.contains(r#""user":"""#), "{}", r.body);
}

#[tokio::test]
async fn authoritative_serves_files_and_answers_its_own_api() {
    let addr = spawn(
        Builder::new(cfg(AUTHORITATIVE))
            .local_api("npcd", fake_api())
            .router(),
    )
    .await;

    let r = fetch(addr, "/v1/status").await;
    assert_eq!(r.status, 200);
    assert!(r.body.contains("ready"), "{}", r.body);

    let r = fetch(addr, "/").await;
    assert_eq!(r.status, 200);
    assert!(r.content_type.starts_with("text/html"));
    assert!(
        r.body.contains("<!doctype html>"),
        "index came from the site root"
    );
}

#[tokio::test]
async fn a_second_root_completes_one_url_tree() {
    let addr = spawn(Builder::new(cfg(AUTHORITATIVE)).router()).await;

    // Site root only.
    assert_eq!(fetch(addr, "/app.css").await.status, 200);
    // Common root only — the fall-through that lets sites share the framework.
    assert_eq!(fetch(addr, "/lib/dom.js").await.status, 200);
    assert_eq!(fetch(addr, "/base.css").await.status, 200);
}

#[tokio::test]
async fn a_deep_link_falls_back_but_a_missing_module_does_not() {
    let addr = spawn(Builder::new(cfg(AUTHORITATIVE)).router()).await;

    // Navigation: the hash router owns this path, so a refresh must work.
    let r = get_with(addr, "/npc/42", "127.0.0.1", "text/html").await;
    assert_eq!(r.status, 200);
    assert!(r.content_type.starts_with("text/html"));

    // An asset: falling back here would hand the browser HTML where it expects
    // JavaScript, which surfaces as `Unexpected token '<'` a long way from the
    // actual cause.
    let r = fetch(addr, "/lib/definitely-not-here.js").await;
    assert_eq!(r.status, 404);
    assert!(
        !r.content_type.starts_with("text/html"),
        "got {}",
        r.content_type
    );
}

#[tokio::test]
async fn traversal_is_refused_before_any_filesystem_access() {
    let addr = spawn(Builder::new(cfg(AUTHORITATIVE)).router()).await;
    for path in ["/../Cargo.toml", "/..%2fCargo.toml", "/%2e%2e%2fCargo.toml"] {
        let r = fetch(addr, path).await;
        assert_eq!(r.status, 400, "{path} should be refused");
        assert!(!r.body.contains("[package]"), "{path} escaped the root");
    }
}

#[tokio::test]
async fn a_local_route_without_an_api_says_so_instead_of_hanging() {
    // What `--authoritative` does to a site that ships no mock, and what a
    // stray `upstream: local` does in a proxy config. Either way it must be an
    // immediate, legible error that names the site.
    let addr = spawn(Builder::new(cfg(AUTHORITATIVE)).router()).await;
    let r = fetch(addr, "/v1/status").await;
    assert_eq!(r.status, 502);
    assert!(r.body.contains("site `npcd` has no API"), "{}", r.body);
}

// ── --authoritative ────────────────────────────────────────────────────────

#[tokio::test]
async fn authoritative_answers_locally_without_touching_the_upstream() {
    // Point the config at a port nothing is listening on, then take the
    // upstreams away with `.authoritative()`. A reply proves nothing was
    // forwarded: the configured address could not have answered.
    let dead: SocketAddr = {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        l.local_addr().unwrap()
    };
    let addr = spawn(
        Builder::new(proxy_cfg(dead))
            .authoritative()
            .local_api("npcd", fake_api())
            .router(),
    )
    .await;

    let r = get_with(addr, "/v1/status", "npcd.test", "application/json").await;
    assert_eq!(r.status, 200);
    assert!(r.body.contains("ready"), "{}", r.body);

    // Content is unaffected — it was never coming from an upstream.
    assert_eq!(
        get_with(addr, "/base.css", "npcd.test", "*/*").await.status,
        200
    );
}

#[tokio::test]
async fn the_real_mock_serves_the_console_site_end_to_end() {
    // The `web --authoritative` arrangement exactly: the shipped mock behind
    // the shipped files, so the console runs with no daemon anywhere.
    let addr = spawn(
        Builder::new(cfg(AUTHORITATIVE))
            .local_api(
                "npcd",
                web::mock::for_site("npcd").expect("npcd ships a mock"),
            )
            .router(),
    )
    .await;

    let r = fetch(addr, "/v1/status").await;
    assert_eq!(r.status, 200);
    assert!(r.body.contains("mock"), "{}", r.body);

    let r = fetch(addr, "/v1/npc").await;
    assert_eq!(r.status, 200);
    assert!(
        r.body.contains("npc_id"),
        "the roster has something to render"
    );

    assert_eq!(fetch(addr, "/").await.status, 200);
}

#[tokio::test]
async fn only_sites_with_a_mock_report_one() {
    // `--check` reads this to describe the run, so a site that would 502 must
    // not be listed as mocked.
    assert!(web::mock::for_site("npcd").is_some());
    assert!(web::mock::for_site("zend").is_none());
    assert!(web::mock::for_site("landing").is_none());
}

// ── proxy: same crate, upstream URLs ───────────────────────────────────────

fn proxy_cfg(upstream: SocketAddr) -> Config {
    cfg(&format!(
        r#"
server:
  backoff: {{initial_ms: 200, max_ms: 400}}
  connect_timeout_ms: 500
sites:
  - name: npcd
    hosts: ["npcd.test"]
    roots: ["content/npcd", "content/common"]
    api:
      - {{prefix: /v1, upstream: "http://{upstream}"}}
  - name: other
    default: true
    roots: ["content/common"]
    api: []
"#
    ))
}

#[tokio::test]
async fn proxy_forwards_to_a_live_upstream() {
    let up = spawn(fake_api()).await;
    let addr = spawn(Builder::new(proxy_cfg(up)).router()).await;

    let r = get_with(addr, "/v1/status", "npcd.test", "application/json").await;
    assert_eq!(r.status, 200);
    assert!(r.body.contains("ready"), "{}", r.body);
}

#[tokio::test]
async fn the_host_header_picks_the_site() {
    let up = spawn(fake_api()).await;
    let addr = spawn(Builder::new(proxy_cfg(up)).router()).await;

    // `other` is the default and has no /v1 route, so this is a file lookup
    // that misses — proof the route table is per-site and not global.
    let r = get_with(addr, "/v1/status", "somewhere.else", "application/json").await;
    assert_eq!(r.status, 404);

    // And its own roots still work.
    assert_eq!(
        get_with(addr, "/base.css", "somewhere.else", "*/*")
            .await
            .status,
        200
    );
}

#[tokio::test]
async fn a_dead_upstream_gives_a_readable_page_then_backs_off() {
    // Bind and immediately drop, so the port is almost certainly closed: a
    // connection refused, which is the failure an operator actually sees when a
    // daemon is not running.
    let dead = {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        l.local_addr().unwrap()
    };
    let addr = spawn(Builder::new(proxy_cfg(dead)).router()).await;

    // First request pays the connect attempt and reports it.
    let r = get_with(addr, "/v1/status", "npcd.test", "text/html").await;
    assert_eq!(r.status, 502);
    assert!(
        r.content_type.starts_with("text/html"),
        "a browser gets a page"
    );
    assert!(r.body.contains("not answering"), "{}", r.body);
    assert!(
        r.body.contains("http-equiv=\"refresh\""),
        "the page retries itself"
    );

    // Subsequent requests inside the window fail fast rather than repeating the
    // connect — that is the point of the backoff, so assert on the clock.
    let t = Instant::now();
    let r = get_with(addr, "/v1/status", "npcd.test", "application/json").await;
    assert_eq!(r.status, 503);
    assert!(r.body.contains("upstream_backoff"), "{}", r.body);
    assert!(
        t.elapsed() < Duration::from_millis(150),
        "took {:?}",
        t.elapsed()
    );

    // An API caller gets the same failure as JSON, never a page.
    assert!(!r.content_type.starts_with("text/html"));
}

#[tokio::test]
async fn recovery_needs_no_operator() {
    // Reserve a port, close it, point the proxy at it, then start the real
    // upstream there once the window has opened.
    let port = {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        l.local_addr().unwrap().port()
    };
    let target: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
    let addr = spawn(Builder::new(proxy_cfg(target)).router()).await;

    assert_eq!(
        get_with(addr, "/v1/status", "npcd.test", "application/json")
            .await
            .status,
        502
    );

    let listener = tokio::net::TcpListener::bind(target).await.unwrap();
    tokio::spawn(async move {
        axum::serve(
            listener,
            fake_api().into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await
        .unwrap();
    });

    // max_ms is 400 in this config, so the probe is released well inside this.
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let r = get_with(addr, "/v1/status", "npcd.test", "application/json").await;
        if r.status == 200 {
            assert!(r.body.contains("ready"), "{}", r.body);
            break;
        }
        assert!(
            Instant::now() < deadline,
            "never recovered (last status {})",
            r.status
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

// ── upgrades ───────────────────────────────────────────────────────────────

/// An upstream that answers any request with `101 Switching Protocols` and then
/// echoes bytes forever.
///
/// Deliberately raw TCP rather than a websocket library: what the proxy has to
/// get right is stopping speaking HTTP after a 101 and joining the two upgraded
/// connections, and a handshake helper would hide exactly that. Nothing above
/// the byte tunnel is this crate's business.
async fn spawn_upgrading_echo() -> SocketAddr {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            tokio::spawn(async move {
                // Read to the end of the request head; the body is irrelevant.
                let mut head = Vec::new();
                let mut byte = [0u8; 1];
                while !head.ends_with(b"\r\n\r\n") {
                    match sock.read(&mut byte).await {
                        Ok(1) => head.push(byte[0]),
                        _ => return,
                    }
                }
                let res = "HTTP/1.1 101 Switching Protocols\r\n\
                           upgrade: websocket\r\n\
                           connection: Upgrade\r\n\
                           sec-websocket-accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n\r\n";
                if sock.write_all(res.as_bytes()).await.is_err() {
                    return;
                }
                let mut buf = [0u8; 1024];
                loop {
                    match sock.read(&mut buf).await {
                        Ok(0) | Err(_) => return,
                        Ok(n) => {
                            if sock.write_all(&buf[..n]).await.is_err() {
                                return;
                            }
                        }
                    }
                }
            });
        }
    });
    addr
}

#[tokio::test]
async fn a_websocket_upgrade_becomes_a_byte_tunnel() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let up = spawn_upgrading_echo().await;
    let addr = spawn(Builder::new(proxy_cfg(up)).router()).await;

    let mut sock = tokio::net::TcpStream::connect(addr).await.unwrap();
    sock.write_all(
        b"GET /v1/socket HTTP/1.1\r\n\
          host: npcd.test\r\n\
          connection: Upgrade\r\n\
          upgrade: websocket\r\n\
          sec-websocket-version: 13\r\n\
          sec-websocket-key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n",
    )
    .await
    .unwrap();

    // The 101 comes back through, headers and all.
    let mut head = Vec::new();
    let mut byte = [0u8; 1];
    while !head.ends_with(b"\r\n\r\n") {
        assert_eq!(
            sock.read(&mut byte).await.unwrap(),
            1,
            "connection closed mid-handshake"
        );
        head.push(byte[0]);
    }
    let head = String::from_utf8_lossy(&head).to_ascii_lowercase();
    assert!(head.starts_with("http/1.1 101"), "{head}");
    assert!(head.contains("upgrade: websocket"), "{head}");
    // `Connection` is hop-by-hop and stripped from every other response, but a
    // 101 without it is rejected by the client as a malformed handshake — which
    // is precisely the bug a test that only checked `Upgrade` would miss.
    assert!(head.contains("connection: upgrade"), "{head}");

    // And after it, the proxy is out of the way: bytes in either direction are
    // whatever the two ends said, unframed and unmodified.
    for probe in [&b"the first frame"[..], &b"\x00\x01\x02 binary too"[..]] {
        sock.write_all(probe).await.unwrap();
        let mut back = vec![0u8; probe.len()];
        sock.read_exact(&mut back).await.unwrap();
        assert_eq!(back, probe, "the tunnel altered the payload");
    }
}

// ── the equivalence the split is for ───────────────────────────────────────

#[tokio::test]
async fn local_and_proxied_answers_are_indistinguishable() {
    let up = spawn(fake_api()).await;
    let proxied = spawn(Builder::new(proxy_cfg(up)).router()).await;
    let local = spawn(
        Builder::new(cfg(AUTHORITATIVE))
            .local_api("npcd", fake_api())
            .router(),
    )
    .await;

    let a = get_with(proxied, "/v1/status", "npcd.test", "application/json").await;
    let b = get_with(local, "/v1/status", "127.0.0.1", "application/json").await;
    assert_eq!(a.status, b.status);
    assert_eq!(a.body, b.body);
    assert_eq!(a.content_type, b.content_type);
}

// ── the shipped table ──────────────────────────────────────────────────────

/// The real `web.yaml`, resolved against the real content tree.
fn shipped() -> Config {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    Config::from_yaml(include_str!("../web.yaml"), manifest).expect("web.yaml parses")
}

/// A product's identity is per-site, so two products cannot share a site entry.
///
/// `battlecities.net` was a `hosts:` entry on the tokera site, which meant it
/// served tokera's pages *and* tokera's brand mark — the game's own domain wore
/// the Tokera triskelion, and nothing failed to say so. Splitting it out is what
/// gives it back its own icon, and this is the assertion that keeps it split.
#[tokio::test]
async fn each_brand_has_its_own_site_and_its_own_mark() {
    let cfg = shipped();

    let bc = cfg.site_for(Some("battlecities.net"));
    assert_eq!(
        bc.name, "battlecities",
        "battlecities.net lost its own site"
    );
    assert_eq!(
        cfg.site_for(Some("www.battlecities.net")).name,
        "battlecities"
    );

    let tk = cfg.site_for(Some("tokera.com"));
    assert_eq!(tk.name, "tokera");

    // Separate roots is the mechanism: an icon is a file, and a shared root is
    // a shared icon however different the two brands are meant to look.
    assert!(
        bc.roots.iter().all(|r| !tk.roots.contains(r)),
        "the two brands share a content root, so they share a favicon"
    );

    // And each one's mark is actually present where its site will look for it.
    // `content_dir` is the config's base, not the content tree, so this walks
    // the site's own declared root rather than assuming where it points.
    for site in [bc, tk] {
        let root = content_dir().join(&site.roots[0]);
        let found = ["favicon.ico", "favicon.png", "favicon.svg"]
            .iter()
            .any(|f| root.join(f).is_file());
        assert!(found, "{} has no icon of its own", site.name);
    }
}

/// An unknown Host lands on the default site, not on whichever was declared
/// first — the difference decides what a bare IP or a stray CNAME serves.
#[tokio::test]
async fn an_unknown_host_still_lands_on_tokera() {
    assert_eq!(shipped().site_for(Some("nowhere.example")).name, "tokera");
    assert_eq!(shipped().site_for(None).name, "tokera");
}
