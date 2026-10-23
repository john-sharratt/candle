//! The connector one app's socket dials its cluster through
//! (`docs/replicated_substrate.md` §3.2, paths): every address of every
//! endpoint, best first, skipping the paths cooling; a session judged by how it
//! ended; and, on a lesser path, a probe of a better one that closes the
//! session to return to it.

use std::net::SocketAddr;
use std::sync::{Arc, Mutex, MutexGuard};

use cluster::close::{Close, CloseCode};
use cluster::dial::{Attempt, Connect};
use cluster::handshake::HandshakeError;
use cluster::offline;
use cluster::pipe::socket::{split, SocketRx, SocketTx};
use cluster::pipe::tungstenite::Tungstenite;
use cluster::pipe::{Message, PipeRx};
use cluster::session::Ended;
use rustls_pki_types::ServerName;
use tokio::net::{lookup_host, TcpStream};
use tokio::sync::Notify;
use tokio::task::JoinHandle;
use tokio::time::{interval, timeout, Instant, MissedTickBehavior};
use tokio_rustls::TlsConnector;
use tokio_tungstenite::client_async_with_config;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;

use crate::paths::{PathId, Paths, Timing};
use crate::target::{Endpoint, Target};
use crate::{Io, Socket, MAX_MESSAGE};

/// The reason a socket gives the gateway when it leaves for a better path.
pub const RETURNING: &str = "returning to a better path";

/// The session on a path.
struct On {
    path: PathId,
    since: Instant,
    /// The watch for a better path, while one exists.
    probe: Option<JoinHandle<()>>,
}

/// The connector [`cluster::dial::dial`] dials one app's socket through.
pub struct Dial {
    target: Arc<Target>,
    app: &'static str,
    tls: TlsConnector,
    timing: Timing,
    paths: Arc<Mutex<Paths>>,
    on: Option<On>,
    /// The path the last session ran on, to log a change of path.
    last: Option<PathId>,
}

fn lock(paths: &Mutex<Paths>) -> MutexGuard<'_, Paths> {
    paths.lock().unwrap_or_else(|e| e.into_inner())
}

impl Dial {
    pub fn new(target: Arc<Target>, app: &'static str, tls: TlsConnector) -> Self {
        Self::with_timing(target, app, tls, Timing::default())
    }

    fn with_timing(
        target: Arc<Target>,
        app: &'static str,
        tls: TlsConnector,
        timing: Timing,
    ) -> Self {
        Self {
            target,
            app,
            tls,
            timing,
            paths: Arc::new(Mutex::new(Paths::new(timing))),
            on: None,
            last: None,
        }
    }

    fn label(&self, p: PathId) -> String {
        label(&self.target, p)
    }

    /// Every endpoint's addresses, best first, and why any endpoint gave
    /// none.
    async fn resolve(&self) -> (Vec<PathId>, Vec<String>) {
        let mut ranked = Vec::new();
        let mut failed = Vec::new();
        for (i, e) in self.target.endpoints.iter().enumerate() {
            if !offline::reaches(&e.host) {
                failed.push(offline::refused(&format!("{}:{}", e.host, e.port)));
                continue;
            }
            match timeout(self.timing.connect, lookup_host((e.host.as_str(), e.port))).await {
                Ok(Ok(addrs)) => {
                    let mut seen: Vec<SocketAddr> = Vec::new();
                    for addr in addrs {
                        if !seen.contains(&addr) {
                            seen.push(addr);
                            ranked.push(PathId { endpoint: i, addr });
                        }
                    }
                    if seen.is_empty() {
                        failed.push(format!("{}: resolved to no address", e.url));
                    }
                }
                Ok(Err(err)) => failed.push(format!("{}: {err}", e.url)),
                Err(_) => failed.push(format!("{}: no answer resolving it", e.url)),
            }
        }
        (ranked, failed)
    }

    async fn open(&self, p: PathId) -> Result<Socket, String> {
        let e = &self.target.endpoints[p.endpoint];
        let stream = transport(&self.tls, e, p.addr).await?;
        let request = e
            .socket_url(self.app)
            .into_client_request()
            .map_err(|e| e.to_string())?;
        let config = WebSocketConfig {
            max_message_size: Some(MAX_MESSAGE),
            max_frame_size: Some(MAX_MESSAGE),
            ..WebSocketConfig::default()
        };
        let (ws, _) = client_async_with_config(request, stream, Some(config))
            .await
            .map_err(|err| format!("the WebSocket handshake: {err}"))?;
        Ok(ws)
    }

    /// Start a session's watch for a better path, when there is one to have.
    fn watch(
        &self,
        ranked: Vec<PathId>,
        on: PathId,
        leaving: Arc<Notify>,
    ) -> Option<JoinHandle<()>> {
        if !ranked.iter().any(|p| p.endpoint < on.endpoint) {
            return None;
        }
        let target = self.target.clone();
        let tls = self.tls.clone();
        let paths = self.paths.clone();
        let timing = self.timing;
        let app = self.app;
        Some(tokio::spawn(async move {
            let mut tick = interval(timing.return_every);
            tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
            tick.tick().await;
            loop {
                tick.tick().await;
                let Some(better) = lock(&paths).better(&ranked, on, Instant::now()) else {
                    continue;
                };
                let e = &target.endpoints[better.endpoint];
                match timeout(timing.connect, transport(&tls, e, better.addr)).await {
                    Ok(Ok(_)) => {
                        tracing::info!(
                            app,
                            from = %label(&target, on),
                            to = %label(&target, better),
                            "cluster path: a better path answers; returning to it"
                        );
                        leaving.notify_one();
                        return;
                    }
                    Ok(Err(why)) => {
                        let cool = lock(&paths).strike(better, Instant::now());
                        tracing::debug!(
                            app,
                            path = %label(&target, better),
                            cool_s = cool.as_secs(),
                            "cluster path: the probe of a better path failed: {why}"
                        );
                    }
                    Err(_) => {
                        lock(&paths).strike(better, Instant::now());
                    }
                }
            }
        }))
    }
}

fn label(target: &Target, p: PathId) -> String {
    format!("{} ({})", target.endpoints[p.endpoint].url, p.addr)
}

/// TCP to `addr`, and TLS over it verified by `e`'s name.
async fn transport(
    tls: &TlsConnector,
    e: &Endpoint,
    addr: SocketAddr,
) -> Result<Box<dyn Io>, String> {
    let tcp = TcpStream::connect(addr)
        .await
        .map_err(|err| format!("{addr}: {err}"))?;
    let _ = tcp.set_nodelay(true);
    Ok(match &e.tls_name {
        Some(name) => {
            let name = ServerName::try_from(name.clone())
                .map_err(|err| format!("`{name}` cannot name a certificate: {err}"))?;
            Box::new(
                tls.connect(name, tcp)
                    .await
                    .map_err(|err| format!("TLS with {}: {err}", e.url))?,
            )
        }
        None => Box::new(tcp),
    })
}

/// A session's receiving half: the socket's, until the session leaves for a
/// better path, when it reads as the gateway's close.
pub struct PathRx {
    inner: SocketRx<Socket, Tungstenite>,
    leaving: Arc<Notify>,
}

impl PipeRx for PathRx {
    async fn recv(&mut self) -> Option<Message> {
        tokio::select! {
            m = self.inner.recv() => m,
            _ = self.leaving.notified() => {
                Some(Message::Close(Close::new(CloseCode::GoingAway, RETURNING)))
            }
        }
    }
}

impl Connect for Dial {
    type Tx = SocketTx;
    type Rx = PathRx;

    async fn connect(&mut self) -> Result<(SocketTx, PathRx), String> {
        offline::refresh();
        let (ranked, mut failed) = self.resolve().await;
        let order = lock(&self.paths).order(&ranked, Instant::now());
        for p in order {
            let why = match timeout(self.timing.connect, self.open(p)).await {
                Ok(Ok(ws)) => {
                    if self.last != Some(p) {
                        tracing::info!(app = self.app, path = %self.label(p), "cluster path: dialled");
                    }
                    self.last = Some(p);
                    let leaving = Arc::new(Notify::new());
                    let probe = self.watch(ranked.clone(), p, leaving.clone());
                    self.on = Some(On {
                        path: p,
                        since: Instant::now(),
                        probe,
                    });
                    let (tx, rx) = split::<Tungstenite, _, _>(ws);
                    return Ok((tx, PathRx { inner: rx, leaving }));
                }
                Ok(Err(why)) => why,
                Err(_) => format!("no answer within {:?}", self.timing.connect),
            };
            let cool = lock(&self.paths).strike(p, Instant::now());
            tracing::debug!(
                app = self.app,
                path = %self.label(p),
                cool_s = cool.as_secs(),
                "cluster path: could not dial: {why}"
            );
            failed.push(format!("{}: {why}", self.label(p)));
        }
        Err(failed.join("; "))
    }

    fn ended(&mut self, attempt: &Attempt) {
        let Some(on) = self.on.take() else {
            return;
        };
        if let Some(probe) = on.probe {
            probe.abort();
        }
        let now = Instant::now();
        let mut paths = lock(&self.paths);
        if now.duration_since(on.since) >= self.timing.healthy_after {
            paths.lasted(on.path);
        }
        let fault = match attempt {
            Attempt::Handshake(HandshakeError::Dropped) => Some("dropped mid-handshake"),
            Attempt::Session(Ended::Dead) => Some("went silent for 45 s"),
            _ => None,
        };
        if let Some(fault) = fault {
            let cool = paths.strike(on.path, now);
            tracing::warn!(
                app = self.app,
                path = %label(&self.target, on.path),
                cool_s = cool.as_secs(),
                "cluster path: {fault}; cooling it"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tokio::net::TcpListener;

    use super::*;
    use crate::tls_connector;

    fn timing() -> Timing {
        Timing {
            connect: Duration::from_millis(1500),
            cool_first: Duration::from_millis(50),
            cool_max: Duration::from_secs(1),
            healthy_after: Duration::from_secs(600),
            return_every: Duration::from_millis(100),
        }
    }

    /// A gateway stand-in on `listener`: completes each WebSocket handshake and
    /// holds the socket open.
    fn serve(listener: TcpListener) -> JoinHandle<()> {
        tokio::spawn(async move {
            let mut held = Vec::new();
            while let Ok((tcp, _)) = listener.accept().await {
                if let Ok(ws) = tokio_tungstenite::accept_async(tcp).await {
                    held.push(ws);
                }
            }
        })
    }

    /// A loopback port nothing listens on.
    async fn closed_port() -> u16 {
        let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
        l.local_addr().unwrap().port()
    }

    async fn gateway() -> (u16, JoinHandle<()>) {
        let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = l.local_addr().unwrap().port();
        (port, serve(l))
    }

    fn dial(ports: &[u16]) -> Dial {
        let spec = ports
            .iter()
            .map(|p| format!("http://127.0.0.1:{p}"))
            .collect::<Vec<_>>()
            .join(" ");
        Dial::with_timing(
            Arc::new(Target::parse(&spec, None).unwrap()),
            "zend",
            tls_connector().unwrap(),
            timing(),
        )
    }

    fn on(d: &Dial) -> usize {
        d.on.as_ref().expect("a session").path.endpoint
    }

    #[tokio::test]
    async fn a_dead_endpoint_is_passed_over_and_cools() {
        let dead = closed_port().await;
        let (up, _g) = gateway().await;
        let mut d = dial(&[dead, up]);
        let _session = d.connect().await.expect("the second endpoint answers");
        assert_eq!(on(&d), 1);
        let first = d.target.endpoints[0].port;
        let cooling = lock(&d.paths).cooling(
            PathId {
                endpoint: 0,
                addr: SocketAddr::from(([127, 0, 0, 1], first)),
            },
            Instant::now(),
        );
        assert!(cooling, "the endpoint that refused is cooling");
    }

    #[tokio::test]
    async fn every_path_failing_fails_the_dial_and_names_each() {
        let (a, b) = (closed_port().await, closed_port().await);
        let mut d = dial(&[a, b]);
        let err = d.connect().await.err().expect("nothing answers");
        assert!(
            err.contains(&format!(":{a}")) && err.contains(&format!(":{b}")),
            "{err}"
        );
        assert!(d.on.is_none());
        // Every path cooling, the next dial still tries them all.
        let err = d.connect().await.err().expect("nothing answers");
        assert!(
            err.contains(&format!(":{a}")) && err.contains(&format!(":{b}")),
            "{err}"
        );
    }

    #[tokio::test]
    async fn a_session_that_goes_silent_cools_its_path_and_a_close_does_not() {
        let (a, _ga) = gateway().await;
        let (b, _gb) = gateway().await;
        let mut d = dial(&[a, b]);

        let _s = d.connect().await.unwrap();
        assert_eq!(on(&d), 0);
        // The gateway closed it, or the transport dropped: no fault of the path.
        d.ended(&Attempt::Session(Ended::Closed(Close::new(
            CloseCode::GoingAway,
            "restarting",
        ))));
        let _s = d.connect().await.unwrap();
        assert_eq!(on(&d), 0);
        d.ended(&Attempt::Session(Ended::Dropped));
        let _s = d.connect().await.unwrap();
        assert_eq!(on(&d), 0);

        // Silent for 45 s: the path cools and the next dial takes the other.
        d.ended(&Attempt::Session(Ended::Dead));
        let _s = d.connect().await.unwrap();
        assert_eq!(on(&d), 1);
    }

    #[tokio::test]
    async fn a_socket_on_a_lesser_path_returns_once_a_better_one_answers() {
        let first = closed_port().await;
        let (second, _g) = gateway().await;
        let mut d = dial(&[first, second]);
        let (_tx, mut rx) = d.connect().await.unwrap();
        assert_eq!(on(&d), 1);

        // The better endpoint comes up: the session reads its own leaving as a
        // close, 1001.
        let l = TcpListener::bind(("127.0.0.1", first)).await.unwrap();
        let _g1 = serve(l);
        let left = timeout(Duration::from_secs(10), rx.recv())
            .await
            .expect("the session leaves for the better path");
        assert_eq!(
            left,
            Some(Message::Close(Close::new(CloseCode::GoingAway, RETURNING)))
        );
        d.ended(&Attempt::Session(Ended::Closed(Close::new(
            CloseCode::GoingAway,
            RETURNING,
        ))));
        let _s = d.connect().await.unwrap();
        assert_eq!(on(&d), 0, "the redial takes the better path");
        assert!(
            d.on.as_ref().unwrap().probe.is_none(),
            "on the best endpoint, nothing better to watch for"
        );
    }
}
