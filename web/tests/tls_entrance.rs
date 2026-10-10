//! The TLS entrance over real connections: HTTP/3, HTTP/2 and HTTP/1.1 each
//! reach the same router, and each is routed by the name the client asked for.
//!
//! The certificate is generated per run and put in the ACME store exactly where
//! an issued one lands, so the entrance is tested through the same resolver it
//! serves with. Two sites are served — a default one and one named
//! `localhost` — so a request that lost its host on the way in would read the
//! default site's file and fail, which is how an HTTP/2 or HTTP/3 request reads
//! when `:authority` is not carried across.

use std::future::{pending, poll_fn};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::routing::{get, post};
use axum::Router;
use bytes::{Buf, Bytes};
use h3::client::RequestStream;
use h3::quic::RecvStream;
use h3_quinn::Connection as QuinnConnection;
use http_body_util::{BodyExt, Empty, Full};
use hyper::Request;
use hyper_util::rt::{TokioExecutor, TokioIo};
use quinn::crypto::rustls::QuicClientConfig;
use quinn::{ClientConfig as QuinnClientConfig, Endpoint};
use rustls::crypto::ring::default_provider;
use rustls::pki_types::{CertificateDer, ServerName};
use rustls::version::TLS13;
use rustls::{ClientConfig, RootCertStore};
use tempfile::TempDir;
use tokio::net::TcpStream;
use tokio::sync::oneshot;
use tokio::time::{sleep, timeout};
use tokio_rustls::client::TlsStream;
use tokio_rustls::TlsConnector;
use web::acme::store::Store;
use web::tls::{Entrance, Resolver};
use web::{Builder, Config};

const NAMED: &[u8] = b"the named site";

struct Served {
    addr: SocketAddr,
    root: CertificateDer<'static>,
    _dir: TempDir,
}

/// An entrance holding a fresh certificate for `localhost`, put in the store
/// in `dir` exactly where an issued one lands; returns it with its root.
async fn bind_localhost(dir: &TempDir) -> (Entrance, CertificateDer<'static>) {
    let ck = rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).unwrap();
    let store = Store::new(&dir.path().join("acme"));
    store
        .save_cert("named", &ck.cert.pem(), &ck.signing_key.serialize_pem())
        .unwrap();
    let resolver = Resolver::new();
    resolver.install(&store.load_certs().unwrap());
    let entrance = Entrance::bind("127.0.0.1:0".parse().unwrap(), resolver)
        .await
        .unwrap();
    (entrance, ck.cert.der().clone())
}

async fn serve() -> Served {
    let dir = tempfile::tempdir().unwrap();
    let (entrance, root) = bind_localhost(&dir).await;
    for (site, body) in [("default", &b"the default site"[..]), ("named", NAMED)] {
        std::fs::create_dir_all(dir.path().join(site)).unwrap();
        std::fs::write(dir.path().join(site).join("hello.txt"), body).unwrap();
    }
    let yaml = r#"
server:
  bind: "127.0.0.1:0"
sites:
  - name: default
    default: true
    roots: ["default"]
  - name: named
    hosts: ["localhost"]
    roots: ["named"]
    api:
      - {prefix: /echo, upstream: local}
"#;
    let cfg = Config::from_yaml(yaml, dir.path()).unwrap();
    let addr = entrance.local_addr().unwrap();
    let echo = Router::new().route("/echo", post(|body: Bytes| async move { body }));
    let router = Builder::new(cfg).local_api("named", echo).router();
    tokio::spawn(entrance.serve(router, pending()));
    Served {
        addr,
        root,
        _dir: dir,
    }
}

fn client_tls(s: &Served, quic: bool, alpn: &[&[u8]]) -> ClientConfig {
    let mut roots = RootCertStore::empty();
    roots.add(s.root.clone()).unwrap();
    let builder = ClientConfig::builder_with_provider(Arc::new(default_provider()));
    let builder = if quic {
        builder.with_protocol_versions(&[&TLS13])
    } else {
        builder.with_safe_default_protocol_versions()
    };
    let mut cfg = builder
        .unwrap()
        .with_root_certificates(roots)
        .with_no_client_auth();
    cfg.alpn_protocols = alpn.iter().map(|p| p.to_vec()).collect();
    cfg
}

async fn tls_connect(s: &Served, alpn: &[&[u8]]) -> TlsStream<TcpStream> {
    let tcp = TcpStream::connect(s.addr).await.unwrap();
    TlsConnector::from(Arc::new(client_tls(s, false, alpn)))
        .connect(ServerName::try_from("localhost").unwrap(), tcp)
        .await
        .unwrap()
}

fn alt_svc(s: &Served) -> String {
    format!("h3=\":{}\"; ma=86400", s.addr.port())
}

/// A browser's first connection: HTTP/2 by ALPN, answered by the site its
/// `:authority` names, with HTTP/3 advertised on the same port.
#[tokio::test]
async fn http2_is_negotiated_routed_by_authority_and_advertises_http3() {
    let s = serve().await;
    let tls = tls_connect(&s, &[b"h2", b"http/1.1"]).await;
    assert_eq!(tls.get_ref().1.alpn_protocol(), Some(&b"h2"[..]));
    let (mut send, conn) =
        hyper::client::conn::http2::handshake(TokioExecutor::new(), TokioIo::new(tls))
            .await
            .unwrap();
    tokio::spawn(conn);
    let req = Request::get(format!("https://localhost:{}/hello.txt", s.addr.port()))
        .body(Empty::<Bytes>::new())
        .unwrap();
    let res = send.send_request(req).await.unwrap();
    assert_eq!(res.status(), 200);
    assert_eq!(res.headers()["alt-svc"], alt_svc(&s));
    let body = res.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(&body[..], NAMED);
}

/// A name the store holds no certificate for is refused at the handshake — no
/// other site's certificate is presented in its place.
#[tokio::test]
async fn a_name_without_a_certificate_fails_the_handshake() {
    let s = serve().await;
    let tcp = TcpStream::connect(s.addr).await.unwrap();
    let res = TlsConnector::from(Arc::new(client_tls(&s, false, &[b"h2"])))
        .connect(ServerName::try_from("elsewhere.example.net").unwrap(), tcp)
        .await;
    assert!(res.is_err());
}

/// A client that offers nothing newer still gets HTTP/1.1 over TLS.
#[tokio::test]
async fn http11_is_still_served_to_a_client_without_http2() {
    let s = serve().await;
    let tls = tls_connect(&s, &[b"http/1.1"]).await;
    assert_eq!(tls.get_ref().1.alpn_protocol(), Some(&b"http/1.1"[..]));
    let (mut send, conn) = hyper::client::conn::http1::handshake(TokioIo::new(tls))
        .await
        .unwrap();
    tokio::spawn(conn);
    let req = Request::get("/hello.txt")
        .header("host", "localhost")
        .body(Empty::<Bytes>::new())
        .unwrap();
    let res = send.send_request(req).await.unwrap();
    assert_eq!(res.status(), 200);
    assert_eq!(res.headers()["alt-svc"], alt_svc(&s));
    let body = res.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(&body[..], NAMED);
}

/// HTTP/3 over QUIC on the advertised port: the same router, the same site by
/// `:authority`, and a request body streamed in and back out.
#[tokio::test]
async fn http3_reaches_the_same_router_with_bodies_both_ways() {
    let s = serve().await;
    let crypto = QuicClientConfig::try_from(client_tls(&s, true, &[b"h3"])).unwrap();
    let mut endpoint = Endpoint::client("127.0.0.1:0".parse().unwrap()).unwrap();
    endpoint.set_default_client_config(QuinnClientConfig::new(Arc::new(crypto)));
    let conn = endpoint
        .connect(s.addr, "localhost")
        .unwrap()
        .await
        .unwrap();
    let (mut driver, mut send) = h3::client::new(QuinnConnection::new(conn)).await.unwrap();
    tokio::spawn(async move {
        let _ = poll_fn(|cx| driver.poll_close(cx)).await;
    });

    let mut fetch = send
        .send_request(
            Request::get(format!("https://localhost:{}/hello.txt", s.addr.port()))
                .body(())
                .unwrap(),
        )
        .await
        .unwrap();
    fetch.finish().await.unwrap();
    let res = fetch.recv_response().await.unwrap();
    assert_eq!(res.status(), 200);
    assert_eq!(read_h3_body(&mut fetch).await, NAMED);

    let payload = Bytes::from_static(b"a request body, carried over QUIC");
    let mut post = send
        .send_request(
            Request::post(format!("https://localhost:{}/echo", s.addr.port()))
                .body(())
                .unwrap(),
        )
        .await
        .unwrap();
    post.send_data(payload.clone()).await.unwrap();
    post.finish().await.unwrap();
    let res = post.recv_response().await.unwrap();
    assert_eq!(res.status(), 200);
    assert_eq!(read_h3_body(&mut post).await, &payload[..]);
}

async fn read_h3_body<S>(stream: &mut RequestStream<S, Bytes>) -> Vec<u8>
where
    S: RecvStream,
{
    let mut out = Vec::new();
    while let Some(mut chunk) = stream.recv_data().await.unwrap() {
        out.extend_from_slice(&chunk.copy_to_bytes(chunk.remaining()));
    }
    out
}

/// An HTTP/2 request body arrives whole at the router too.
#[tokio::test]
async fn http2_carries_a_request_body() {
    let s = serve().await;
    let tls = tls_connect(&s, &[b"h2"]).await;
    let (mut send, conn) =
        hyper::client::conn::http2::handshake(TokioExecutor::new(), TokioIo::new(tls))
            .await
            .unwrap();
    tokio::spawn(conn);
    let payload = Bytes::from_static(b"a request body, carried over HTTP/2");
    let req = Request::post(format!("https://localhost:{}/echo", s.addr.port()))
        .body(Full::new(payload.clone()))
        .unwrap();
    let res = send.send_request(req).await.unwrap();
    assert_eq!(res.status(), 200);
    let body = res.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(body, payload);
}

/// **Shutdown drains.** A request in flight when the signal comes is still
/// answered in full, `serve` returns once it has been, and no new connection
/// is taken afterwards — what lets a daemon flush its state knowing no request
/// is still writing to it.
#[tokio::test]
async fn shutdown_answers_the_request_in_flight_then_returns() {
    let dir = tempfile::tempdir().unwrap();
    let (entrance, root) = bind_localhost(&dir).await;
    let addr = entrance.local_addr().unwrap();
    let slow = Router::new().route(
        "/slow",
        get(|| async {
            sleep(Duration::from_millis(300)).await;
            "answered"
        }),
    );
    let (stop, stopped) = oneshot::channel::<()>();
    let served = tokio::spawn(entrance.serve(slow, async move {
        let _ = stopped.await;
    }));
    let s = Served {
        addr,
        root,
        _dir: dir,
    };

    let tls = tls_connect(&s, &[b"h2"]).await;
    let (mut send, conn) =
        hyper::client::conn::http2::handshake(TokioExecutor::new(), TokioIo::new(tls))
            .await
            .unwrap();
    tokio::spawn(conn);
    let req = Request::get(format!("https://localhost:{}/slow", addr.port()))
        .body(Empty::<Bytes>::new())
        .unwrap();
    let response = send.send_request(req);
    let in_flight = tokio::spawn(async move {
        let res = response.await.unwrap();
        res.into_body().collect().await.unwrap().to_bytes()
    });
    sleep(Duration::from_millis(50)).await;
    stop.send(()).unwrap();

    assert_eq!(&in_flight.await.unwrap()[..], b"answered");
    timeout(Duration::from_secs(5), served)
        .await
        .expect("serve returned once the request in flight was answered")
        .unwrap()
        .unwrap();
    assert!(
        TcpStream::connect(addr).await.is_err(),
        "a connection was taken after shutdown"
    );
}
