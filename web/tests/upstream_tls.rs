//! The gateway's hop to a daemon over TLS, as `code.tokera.com` reaches zend:
//! the daemon serves the TLS entrance with a certificate it made at startup,
//! and the gateway's route says `self_signed: true`.
//!
//! The upstream is addressed by IP, so the gateway's handshake carries no SNI —
//! the case a resolver keyed by name would refuse.

use std::future::pending;
use std::path::Path;

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use axum::routing::get;
use axum::Router;
use http_body_util::BodyExt;
use tower::Service;
use web::tls::{self_signed, Entrance};
use web::{Builder, Config};

/// A daemon on the TLS entrance with a fresh self-signed certificate; returns
/// its port.
async fn daemon() -> u16 {
    let names = vec!["localhost".to_string(), "127.0.0.1".to_string()];
    let entrance = Entrance::bind("127.0.0.1:0".parse().unwrap(), self_signed(&names).unwrap())
        .await
        .unwrap();
    let port = entrance.local_addr().unwrap().port();
    let api = Router::new().route("/hello", get(|| async { "from the daemon" }));
    tokio::spawn(entrance.serve(api, pending()));
    port
}

fn gateway(port: u16, self_signed: bool) -> Router {
    let yaml = format!(
        "sites:\n  - name: zend\n    default: true\n    api:\n      - {{prefix: /, upstream: \"https://127.0.0.1:{port}\", self_signed: {self_signed}}}\n"
    );
    Builder::new(Config::from_yaml(&yaml, Path::new(".")).unwrap()).router()
}

async fn get_hello(router: Router) -> (StatusCode, String, Option<String>) {
    let req = Request::get("/hello")
        .header(header::HOST, "code.example.net")
        .header(header::ACCEPT, "text/plain")
        .body(Body::empty())
        .unwrap();
    let res = router.clone().call(req).await.unwrap();
    let status = res.status();
    let alt_svc = res
        .headers()
        .get(header::ALT_SVC)
        .map(|v| v.to_str().unwrap().to_owned());
    let body = res.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8_lossy(&body).into_owned(), alt_svc)
}

/// The route reaches the daemon — and the daemon's own `Alt-Svc`, which
/// names its private port, does not come back out of the gateway.
#[tokio::test]
async fn a_self_signed_route_reaches_the_daemon_over_tls() {
    let port = daemon().await;
    assert_eq!(
        get_hello(gateway(port, true)).await,
        (StatusCode::OK, "from the daemon".to_string(), None)
    );
}

/// The same daemon, on a route that does not say `self_signed`, is verified
/// like any https upstream — and its certificate, which no CA signed, fails.
#[tokio::test]
async fn without_the_flag_the_certificate_is_verified_and_refused() {
    let port = daemon().await;
    let (status, _, _) = get_hello(gateway(port, false)).await;
    assert!(
        status.is_server_error(),
        "an unverifiable certificate was accepted: {status}"
    );
}
