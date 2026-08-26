//! The sign-in flow, end to end, against a stand-in provider.
//!
//! The provider's endpoints are configuration — any OIDC client takes an issuer
//! — so a local one is not a test seam bolted onto the code: it is the same
//! code path Google takes, with a different URL. Nothing is stubbed below the
//! socket, and the assertions are on what the browser is actually told.
//!
//! What is checked here is the part that carries risk: the cookie's `Domain`,
//! because that single attribute is what makes one sign-in reach the subsites;
//! the `state` and `next` checks, because getting either wrong is a real
//! vulnerability rather than a bug; and the header stripping, because the
//! daemons believe those headers absolutely.

use std::net::SocketAddr;
use std::path::PathBuf;

use axum::extract::Query;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::Router;
use base64::engine::general_purpose::URL_SAFE_NO_PAD as B64;
use base64::Engine;
use serde_json::json;
use web::{Builder, Config};

// ── harness ────────────────────────────────────────────────────────────────

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
    headers: axum::http::HeaderMap,
    body: String,
}

impl Res {
    fn header(&self, name: &str) -> String {
        self.headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string()
    }
    fn set_cookies(&self) -> Vec<String> {
        self.headers
            .get_all(axum::http::header::SET_COOKIE)
            .iter()
            .filter_map(|v| v.to_str().ok())
            .map(str::to_owned)
            .collect()
    }
}

async fn request(addr: SocketAddr, path: &str, headers: &[(&str, &str)]) -> Res {
    use http_body_util::BodyExt;
    use hyper::Request;

    let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let io = hyper_util::rt::TokioIo::new(stream);
    let (mut sender, conn) = hyper::client::conn::http1::handshake(io).await.unwrap();
    tokio::spawn(async move {
        let _ = conn.await;
    });

    // The caller's own host wins; two Host headers would silently resolve to
    // whichever came first, which is exactly the bug that would make a
    // subsite test pass against the wrong site.
    let host = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("host"))
        .map(|(_, v)| *v)
        .unwrap_or("tokera.localhost");
    let mut b = Request::builder().uri(path).header("host", host);
    for (k, v) in headers
        .iter()
        .filter(|(k, _)| !k.eq_ignore_ascii_case("host"))
    {
        b = b.header(*k, *v);
    }
    let res = sender
        .send_request(b.body(axum::body::Body::empty()).unwrap())
        .await
        .unwrap();
    Res {
        status: res.status().as_u16(),
        headers: res.headers().clone(),
        body: String::from_utf8_lossy(&res.into_body().collect().await.unwrap().to_bytes())
            .into_owned(),
    }
}

/// Pull one `Set-Cookie` by name, as `name=value` ready to send back.
fn cookie_pair(res: &Res, name: &str) -> Option<String> {
    res.set_cookies().into_iter().find_map(|c| {
        let head = c.split(';').next()?.trim().to_string();
        head.starts_with(&format!("{name}=")).then_some(head)
    })
}

/// A provider that authorises everyone as one person. It never redirects
/// anywhere itself — the test plays the browser's part — it only has to answer
/// the token exchange the way Google does.
async fn spawn_provider(sub: &'static str, email: &'static str) -> SocketAddr {
    async fn authorize(Query(q): Query<serde_json::Value>) -> impl IntoResponse {
        // Never actually called: the test drives the redirect itself. Present
        // so a mistake shows up as an assertion rather than a hang.
        axum::Json(q)
    }

    let token = move |body: String| async move {
        // `nonce` has to come back in the id_token or the replay check is
        // meaningless, and the provider only knows it from the authorize step —
        // so the test threads it through the form as a stand-in for that.
        let nonce = form_value(&body, "nonce").unwrap_or_default();
        let claims = json!({
            "sub": sub, "email": email, "name": "Test Person",
            "picture": "", "nonce": nonce,
        });
        let id_token = format!(
            "header.{}.signature",
            B64.encode(serde_json::to_vec(&claims).unwrap())
        );
        axum::Json(json!({ "id_token": id_token, "token_type": "Bearer" }))
    };

    spawn(
        Router::new()
            .route("/authorize", get(authorize))
            .route("/token", post(token)),
    )
    .await
}

fn form_value(body: &str, key: &str) -> Option<String> {
    body.split('&')
        .filter_map(|p| p.split_once('='))
        .find(|(k, _)| *k == key)
        .map(|(_, v)| v.replace("%20", " ").replace('+', " "))
}

fn write_secret(dir: &std::path::Path, name: &str, value: &str) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, value).unwrap();
    p
}

struct Fixture {
    gateway: SocketAddr,
    _dir: tempfile::TempDir,
}

async fn fixture(provider: SocketAddr, cookie_domain: &str) -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let session = write_secret(
        dir.path(),
        "session.key",
        "0123456789abcdef0123456789abcdef",
    );
    let secret = write_secret(dir.path(), "google.secret", "shh");

    let yaml = format!(
        r#"
auth:
  cookie_domain: "{cookie_domain}"
  session_ttl_hours: 24
  session_secret_file: "{session}"
  google:
    client_id: "cid"
    client_secret_file: "{secret}"
    redirect_uri: "http://tokera.localhost:9/auth/callback"
    auth_endpoint: "http://{provider}/authorize"
    token_endpoint: "http://{provider}/token"
sites:
  - name: tokera
    default: true
    hosts: ["tokera.localhost"]
    roots: ["content/tokera", "content/common"]
    papers: "../docs"
    api:
      - {{prefix: /, exact: true, upstream: local}}
      - {{prefix: /blog, upstream: local}}
      - {{prefix: /papers, upstream: local}}
"#,
        session = session.display().to_string().replace('\\', "/"),
        secret = secret.display().to_string().replace('\\', "/"),
    );

    let cfg = Config::from_yaml(&yaml, &PathBuf::from(env!("CARGO_MANIFEST_DIR"))).unwrap();
    let gateway = spawn(Builder::new(cfg).with_auth().unwrap().router()).await;
    Fixture { gateway, _dir: dir }
}

// ── the flow ───────────────────────────────────────────────────────────────

#[tokio::test]
async fn signing_in_issues_a_cookie_on_the_parent_domain() {
    let provider = spawn_provider("u-1", "someone@example.com").await;
    let fx = fixture(provider, ".tokera.com").await;

    // 1. The browser asks to sign in and is sent to the provider.
    let start = request(fx.gateway, "/auth/login?next=%2Fpapers", &[]).await;
    assert_eq!(start.status, 303, "{}", start.body);
    let location = start.header("location");
    assert!(location.contains("/authorize?"), "{location}");
    assert!(
        location.contains("code_challenge_method=S256"),
        "{location}"
    );

    let pending = cookie_pair(&start, "tokera_oauth").expect("a pending cookie");
    let state = location
        .split("state=")
        .nth(1)
        .and_then(|s| s.split('&').next())
        .unwrap()
        .to_string();
    let nonce = location
        .split("nonce=")
        .nth(1)
        .and_then(|s| s.split('&').next())
        .unwrap()
        .to_string();

    // 2. The provider sends the browser back with a code. The nonce rides
    //    along so the stand-in can echo it, as a real provider does.
    let back = request(
        fx.gateway,
        &format!("/auth/callback?code=abc&state={state}&nonce={nonce}"),
        &[("cookie", &pending)],
    )
    .await;
    assert_eq!(back.status, 303, "{}", back.body);
    assert_eq!(
        back.header("location"),
        "/papers",
        "returned to the wrong page"
    );

    // 3. The cookie is what carries sign-in to code. and bot.
    let jar = back
        .set_cookies()
        .into_iter()
        .find(|c| c.starts_with("tokera_session="))
        .unwrap_or_else(|| panic!("no session cookie among {:?}", back.set_cookies()));
    assert!(
        jar.contains("Domain=.tokera.com"),
        "the subsites will not see it: {jar}"
    );
    assert!(jar.contains("HttpOnly"), "{jar}");
    assert!(jar.contains("SameSite=Lax"), "{jar}");
    // The redirect URI here is http, so a Secure cookie would be dropped.
    assert!(!jar.contains("Secure"), "{jar}");

    // 4. And it identifies the person on the next request.
    let session = cookie_pair(&back, "tokera_session").unwrap();
    let me = request(fx.gateway, "/auth/me", &[("cookie", &session)]).await;
    assert_eq!(me.status, 200);
    assert!(me.body.contains("\"authenticated\":true"), "{}", me.body);
    assert!(me.body.contains("someone@example.com"), "{}", me.body);
}

#[tokio::test]
async fn an_https_deployment_gets_secure_cookies_without_asking() {
    let provider = spawn_provider("u-2", "b@example.com").await;
    let dir = tempfile::tempdir().unwrap();
    let session = write_secret(dir.path(), "s.key", "0123456789abcdef0123456789abcdef");
    let secret = write_secret(dir.path(), "g.secret", "shh");
    let yaml = format!(
        r#"
auth:
  cookie_domain: ".tokera.com"
  session_secret_file: "{s}"
  google:
    client_id: "cid"
    client_secret_file: "{g}"
    redirect_uri: "https://tokera.com/auth/callback"
    auth_endpoint: "http://{provider}/authorize"
    token_endpoint: "http://{provider}/token"
sites:
  - name: tokera
    default: true
    roots: ["content/tokera", "content/common"]
    api: [{{prefix: /, exact: true, upstream: local}}]
"#,
        s = session.display().to_string().replace('\\', "/"),
        g = secret.display().to_string().replace('\\', "/"),
    );
    let cfg = Config::from_yaml(&yaml, &PathBuf::from(env!("CARGO_MANIFEST_DIR"))).unwrap();
    let gw = spawn(Builder::new(cfg).with_auth().unwrap().router()).await;

    let start = request(gw, "/auth/login", &[]).await;
    let jar = cookie_pair(&start, "tokera_oauth").unwrap();
    let full = start.set_cookies().into_iter().next().unwrap();
    assert!(
        full.contains("Secure"),
        "https deployment, insecure cookie: {full}"
    );
    assert!(jar.starts_with("tokera_oauth="));
}

// ── the checks that are security rather than correctness ───────────────────

#[tokio::test]
async fn a_callback_with_the_wrong_state_is_refused() {
    // Without this check an attacker's authorization code can be redeemed into
    // someone else's browser, signing them into the attacker's account.
    let provider = spawn_provider("u-3", "c@example.com").await;
    let fx = fixture(provider, ".tokera.com").await;

    let start = request(fx.gateway, "/auth/login", &[]).await;
    let pending = cookie_pair(&start, "tokera_oauth").unwrap();

    let back = request(
        fx.gateway,
        "/auth/callback?code=abc&state=not-the-one",
        &[("cookie", &pending)],
    )
    .await;
    assert_eq!(back.status, 400, "{}", back.body);
    assert!(back.body.contains("state_mismatch"), "{}", back.body);
    assert!(
        cookie_pair(&back, "tokera_session").is_none(),
        "a session was issued anyway"
    );
}

#[tokio::test]
async fn a_callback_with_no_pending_signin_is_refused() {
    let provider = spawn_provider("u-4", "d@example.com").await;
    let fx = fixture(provider, ".tokera.com").await;
    let back = request(fx.gateway, "/auth/callback?code=abc&state=x", &[]).await;
    assert_eq!(back.status, 400, "{}", back.body);
    assert!(back.body.contains("no_pending_signin"), "{}", back.body);
}

#[tokio::test]
async fn login_will_not_be_used_as_an_open_redirect() {
    let provider = spawn_provider("u-5", "e@example.com").await;
    let fx = fixture(provider, ".tokera.com").await;
    for next in [
        "https://evil.example/",
        "//evil.example/",
        "https://tokera.com.evil.example/",
    ] {
        let res = request(
            fx.gateway,
            &format!("/auth/login?next={}", urlencode(next)),
            &[],
        )
        .await;
        assert_eq!(res.status, 400, "accepted {next}");
        assert!(res.body.contains("bad_redirect"), "{}", res.body);
    }
}

#[tokio::test]
async fn a_forged_session_cookie_is_nobody() {
    let provider = spawn_provider("u-6", "f@example.com").await;
    let fx = fixture(provider, ".tokera.com").await;

    let claims = json!({ "sub": "admin", "email": "admin@tokera.com", "exp": 99_999_999_999u64 });
    let forged = format!(
        "tokera_session={}.{}",
        B64.encode(serde_json::to_vec(&claims).unwrap()),
        B64.encode("not-a-real-signature")
    );
    let me = request(fx.gateway, "/auth/me", &[("cookie", &forged)]).await;
    assert_eq!(me.status, 200);
    assert!(me.body.contains("\"authenticated\":false"), "{}", me.body);
}

#[tokio::test]
async fn signing_out_clears_the_cookie_on_the_same_domain() {
    // A cleared cookie that omits Domain leaves the original in place, which
    // looks exactly like sign-out not working.
    let provider = spawn_provider("u-7", "g@example.com").await;
    let fx = fixture(provider, ".tokera.com").await;
    let out = request(fx.gateway, "/auth/logout", &[]).await;
    let jar = out
        .set_cookies()
        .into_iter()
        .find(|c| c.starts_with("tokera_session="))
        .expect("a clearing cookie");
    assert!(jar.contains("Domain=.tokera.com"), "{jar}");
    assert!(jar.contains("Max-Age=0"), "{jar}");
}

#[tokio::test]
async fn without_an_auth_block_the_ui_is_told_there_is_no_sign_in() {
    // Distinct from "signed out": the button should not be offered at all.
    let cfg = Config::from_yaml(
        r#"
sites:
  - name: tokera
    default: true
    roots: ["content/tokera", "content/common"]
    api: [{prefix: /, exact: true, upstream: local}]
"#,
        &PathBuf::from(env!("CARGO_MANIFEST_DIR")),
    )
    .unwrap();
    let gw = spawn(Builder::new(cfg).with_auth().unwrap().router()).await;

    let me = request(gw, "/auth/me", &[]).await;
    assert_eq!(me.status, 200);
    assert!(me.body.contains("\"configured\":false"), "{}", me.body);

    // And the site still serves — sign-in is optional, not load-bearing.
    assert_eq!(request(gw, "/", &[]).await.status, 200);
}

// ── the part that makes it single sign-on ──────────────────────────────────

/// A stand-in daemon that reports back exactly what the gateway told it about
/// the caller — the same thing `zend` and `npcd` will read.
async fn spawn_echo_daemon() -> SocketAddr {
    spawn(Router::new().route(
        "/v1/whoami",
        get(|headers: axum::http::HeaderMap| async move {
            let h = |n: &str| {
                headers
                    .get(n)
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("")
                    .to_string()
            };
            axum::Json(json!({
                "user": h("x-tokera-user"),
                "email": h("x-tokera-email"),
                "assertion": h("x-tokera-assertion"),
            }))
        }),
    ))
    .await
}

async fn gateway_with_subsite(provider: SocketAddr, daemon: SocketAddr) -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let session = write_secret(
        dir.path(),
        "session.key",
        "0123456789abcdef0123456789abcdef",
    );
    let secret = write_secret(dir.path(), "google.secret", "shh");
    let yaml = format!(
        r#"
auth:
  cookie_domain: ".tokera.com"
  session_secret_file: "{s}"
  google:
    client_id: "cid"
    client_secret_file: "{g}"
    redirect_uri: "http://tokera.localhost:9/auth/callback"
    auth_endpoint: "http://{provider}/authorize"
    token_endpoint: "http://{provider}/token"
sites:
  - name: tokera
    default: true
    hosts: ["tokera.localhost"]
    roots: ["content/tokera", "content/common"]
    papers: "../docs"
    api: [{{prefix: /, exact: true, upstream: local}}]
  - name: npcd
    hosts: ["bot.tokera.localhost"]
    api: [{{prefix: /v1, upstream: "http://{daemon}"}}]
"#,
        s = session.display().to_string().replace('\\', "/"),
        g = secret.display().to_string().replace('\\', "/"),
    );
    let cfg = Config::from_yaml(&yaml, &PathBuf::from(env!("CARGO_MANIFEST_DIR"))).unwrap();
    Fixture {
        gateway: spawn(Builder::new(cfg).with_auth().unwrap().router()).await,
        _dir: dir,
    }
}

/// Complete a sign-in and return the session cookie, ready to send.
async fn sign_in(gateway: SocketAddr) -> String {
    let start = request(gateway, "/auth/login", &[]).await;
    let pending = cookie_pair(&start, "tokera_oauth").unwrap();
    let loc = start.header("location");
    let pick = |k: &str| {
        loc.split(&format!("{k}="))
            .nth(1)
            .and_then(|s| s.split('&').next())
            .unwrap()
            .to_string()
    };
    let back = request(
        gateway,
        &format!(
            "/auth/callback?code=abc&state={}&nonce={}",
            pick("state"),
            pick("nonce")
        ),
        &[("cookie", &pending)],
    )
    .await;
    cookie_pair(&back, "tokera_session").expect("a session")
}

#[tokio::test]
async fn the_session_carries_to_a_proxied_subsite() {
    // The whole point of the cookie living on the parent domain: sign in at
    // tokera.com and bot.tokera.com already knows who you are.
    let provider = spawn_provider("u-100", "carry@example.com").await;
    let daemon = spawn_echo_daemon().await;
    let fx = gateway_with_subsite(provider, daemon).await;

    let session = sign_in(fx.gateway).await;

    let anon = request(
        fx.gateway,
        "/v1/whoami",
        &[("host", "bot.tokera.localhost")],
    )
    .await;
    assert!(
        anon.body.contains("\"user\":\"\""),
        "anonymous was named: {}",
        anon.body
    );

    let known = request(
        fx.gateway,
        "/v1/whoami",
        &[("host", "bot.tokera.localhost"), ("cookie", &session)],
    )
    .await;
    assert!(known.body.contains("\"user\":\"u-100\""), "{}", known.body);
    assert!(known.body.contains("carry@example.com"), "{}", known.body);
    // And the signed token, for a daemon that would rather verify than trust.
    assert!(
        !known.body.contains("\"assertion\":\"\""),
        "no assertion forwarded: {}",
        known.body
    );
}

#[tokio::test]
async fn a_client_cannot_forge_identity_headers_through_the_gateway() {
    // The daemons treat these headers as authoritative, so the gateway
    // stripping them on the way in is the whole of that trust.
    let provider = spawn_provider("u-101", "real@example.com").await;
    let daemon = spawn_echo_daemon().await;
    let fx = gateway_with_subsite(provider, daemon).await;

    let forged = request(
        fx.gateway,
        "/v1/whoami",
        &[
            ("host", "bot.tokera.localhost"),
            ("x-tokera-user", "root"),
            ("x-tokera-email", "admin@tokera.com"),
            ("x-tokera-assertion", "made-up"),
        ],
    )
    .await;
    assert!(forged.body.contains("\"user\":\"\""), "{}", forged.body);
    assert!(!forged.body.contains("admin@tokera.com"), "{}", forged.body);
    assert!(!forged.body.contains("made-up"), "{}", forged.body);

    // And a real session is not upgraded by headers sent alongside it.
    let session = sign_in(fx.gateway).await;
    let mixed = request(
        fx.gateway,
        "/v1/whoami",
        &[
            ("host", "bot.tokera.localhost"),
            ("cookie", &session),
            ("x-tokera-user", "root"),
        ],
    )
    .await;
    assert!(mixed.body.contains("\"user\":\"u-101\""), "{}", mixed.body);
}

fn urlencode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            _ => format!("%{b:02X}"),
        })
        .collect()
}
