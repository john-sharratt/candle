//! The connector one app's socket dials its cluster through
//! (`docs/replicated_substrate.md` §3.2, paths): every address of every
//! endpoint from each address of this machine's that may reach it, best first,
//! skipping the paths cooling; a session judged by how it ended; and, on a
//! lesser path, a probe of a better one that closes the session to return to
//! it.

use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use cluster::close::{Close, CloseCode};
use cluster::dial::{Attempt, Connect};
use cluster::handshake::HandshakeError;
use cluster::offline;
use cluster::pipe::socket::{split, SocketRx, SocketTx};
use cluster::pipe::tungstenite::Tungstenite;
use cluster::pipe::{Heard, Message, PipeRx, Stamped};
use cluster::session::Ended;
use rustls_pki_types::ServerName;
use tokio::net::{lookup_host, TcpSocket, TcpStream};
use tokio::sync::Notify;
use tokio::task::JoinHandle;
use tokio::time::{interval, timeout, Instant, MissedTickBehavior};
use tokio_rustls::TlsConnector;
use tokio_tungstenite::client_async_with_config;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;

use crate::paths::{Fault, PathId, Paths, Timing};
use crate::sources::sources_for;
use crate::target::{Endpoint, Target};
use crate::{Io, Socket, MAX_MESSAGE};

/// The reason a socket gives the gateway when it leaves for a better path.
pub const RETURNING: &str = "returning to a better path";

/// The sources to dial a destination from: [`sources_for`], or a test's.
type Sources = Arc<dyn Fn(IpAddr) -> Vec<Option<IpAddr>> + Send + Sync>;

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
    sources: Sources,
    paths: Arc<Mutex<Paths>>,
    on: Option<On>,
}

fn lock(paths: &Mutex<Paths>) -> MutexGuard<'_, Paths> {
    paths.lock().unwrap_or_else(|e| e.into_inner())
}

impl Dial {
    pub fn new(target: Arc<Target>, app: &'static str, tls: TlsConnector) -> Self {
        Self::with(target, app, tls, Timing::default(), Arc::new(sources_for))
    }

    fn with(
        target: Arc<Target>,
        app: &'static str,
        tls: TlsConnector,
        timing: Timing,
        sources: Sources,
    ) -> Self {
        Self {
            target,
            app,
            tls,
            timing,
            sources,
            paths: Arc::new(Mutex::new(Paths::new(timing))),
            on: None,
        }
    }

    fn label(&self, p: PathId) -> String {
        label(&self.target, p)
    }

    /// Every endpoint's addresses, each from every source that may reach it,
    /// best first; and why any endpoint gave none.
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
                            for source in (self.sources)(addr.ip()) {
                                ranked.push(PathId {
                                    endpoint: i,
                                    addr,
                                    source,
                                });
                            }
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

    /// The socket on `p`, and the stamp its reads leave.
    async fn open(&self, p: PathId) -> Result<(Socket, Heard), Failed> {
        let e = &self.target.endpoints[p.endpoint];
        let heard = Heard::new();
        let stream = transport(&self.tls, e, p, heard.clone(), self.timing.connect).await?;
        let request = e
            .socket_url(self.app)
            .into_client_request()
            .map_err(|err| Failed::after(err.to_string()))?;
        let config = WebSocketConfig {
            max_message_size: Some(MAX_MESSAGE),
            max_frame_size: Some(MAX_MESSAGE),
            ..WebSocketConfig::default()
        };
        let handshake = client_async_with_config(request, stream, Some(config));
        let (ws, _) = match timeout(self.timing.connect, handshake).await {
            Ok(Ok(ws)) => ws,
            Ok(Err(err)) => return Err(Failed::after(format!("the WebSocket handshake: {err}"))),
            Err(_) => return Err(Failed::after(no_answer("the WebSocket handshake"))),
        };
        Ok((ws, heard))
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
                match transport(&tls, e, better, Heard::new(), timing.connect).await {
                    Ok(_) => {
                        tracing::info!(
                            app,
                            from = %label(&target, on),
                            to = %label(&target, better),
                            "cluster path: a better path answers; returning to it"
                        );
                        leaving.notify_one();
                        return;
                    }
                    Err(failed) => {
                        let struck = lock(&paths).strike(better, Instant::now(), failed.fault);
                        tracing::debug!(
                            app,
                            path = %label(&target, better),
                            cool_s = struck.cool.as_secs(),
                            "cluster path: the probe of a better path failed: {}",
                            failed.why
                        );
                        suspected(app, better, struck.suspect);
                    }
                }
            }
        }))
    }
}

fn label(target: &Target, p: PathId) -> String {
    let url = &target.endpoints[p.endpoint].url;
    match p.source {
        Some(s) => format!("{url} ({} from {s})", p.addr),
        None => format!("{url} ({})", p.addr),
    }
}

/// A fault just made `p`'s source suspect: say so, once.
fn suspected(app: &str, p: PathId, turned: bool) {
    if let (true, Some(s)) = (turned, p.source) {
        tracing::warn!(
            app,
            source = %s,
            "cluster path: {s} could not reach {}; its paths are dialled last until it succeeds",
            p.addr
        );
    }
}

/// Why a path failed, and which part of it.
struct Failed {
    fault: Fault,
    why: String,
}

impl Failed {
    fn reach(why: String) -> Self {
        Self {
            fault: Fault::Reach,
            why,
        }
    }

    fn after(why: String) -> Self {
        Self {
            fault: Fault::After,
            why,
        }
    }
}

fn no_answer(what: &str) -> String {
    format!("{what}: no answer in time")
}

/// TCP to `p`'s address — bound to its source, when it has one — within
/// `within`: a failure here is the source's failure to reach.
async fn reach(p: PathId, within: Duration) -> Result<TcpStream, Failed> {
    let addr = p.addr;
    let connecting = async {
        match p.source {
            None => TcpStream::connect(addr).await,
            Some(source) => {
                let socket = match source {
                    IpAddr::V4(_) => TcpSocket::new_v4(),
                    IpAddr::V6(_) => TcpSocket::new_v6(),
                }?;
                socket.bind(SocketAddr::new(source, 0))?;
                socket.connect(addr).await
            }
        }
    };
    match timeout(within, connecting).await {
        Ok(Ok(tcp)) => Ok(tcp),
        Ok(Err(err)) => Err(Failed::reach(format!("{addr}: {err}"))),
        Err(_) => Err(Failed::reach(no_answer(&addr.to_string()))),
    }
}

/// The transport on `p`: TCP, its reads stamping `heard`, and TLS over it
/// verified by `e`'s name, each within `within`.
async fn transport(
    tls: &TlsConnector,
    e: &Endpoint,
    p: PathId,
    heard: Heard,
    within: Duration,
) -> Result<Box<dyn Io>, Failed> {
    let tcp = reach(p, within).await?;
    let _ = tcp.set_nodelay(true);
    let stamped = Stamped::new(tcp, heard);
    Ok(match &e.tls_name {
        Some(name) => {
            let name = ServerName::try_from(name.clone()).map_err(|err| {
                Failed::after(format!("`{name}` cannot name a certificate: {err}"))
            })?;
            match timeout(within, tls.connect(name, stamped)).await {
                Ok(Ok(s)) => Box::new(s),
                Ok(Err(err)) => return Err(Failed::after(format!("TLS with {}: {err}", e.url))),
                Err(_) => return Err(Failed::after(no_answer(&format!("TLS with {}", e.url)))),
            }
        }
        None => Box::new(stamped),
    })
}

/// A session's receiving half: the socket's, until the session leaves for a
/// better path, when it reads as the gateway's close. It reports the
/// transport's reads, so a message still arriving keeps the session alive.
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

    fn heard(&self) -> Option<Instant> {
        self.inner.heard()
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
            let failure = match self.open(p).await {
                Ok((ws, heard)) => {
                    tracing::info!(app = self.app, path = %self.label(p), "cluster path: dialled");
                    if let (true, Some(s)) = (lock(&self.paths).connected(p), p.source) {
                        tracing::warn!(app = self.app, source = %s, "cluster path: {s} works again");
                    }
                    let leaving = Arc::new(Notify::new());
                    let probe = self.watch(ranked.clone(), p, leaving.clone());
                    self.on = Some(On {
                        path: p,
                        since: Instant::now(),
                        probe,
                    });
                    let (tx, rx) = split::<Tungstenite, _, _>(ws, heard);
                    return Ok((tx, PathRx { inner: rx, leaving }));
                }
                Err(failure) => failure,
            };
            let struck = lock(&self.paths).strike(p, Instant::now(), failure.fault);
            // A path that was well failing is news; one already failing is not.
            if struck.first {
                tracing::warn!(
                    app = self.app,
                    path = %self.label(p),
                    cool_s = struck.cool.as_secs(),
                    "cluster path: could not dial: {}; cooling it",
                    failure.why
                );
            } else {
                tracing::debug!(
                    app = self.app,
                    path = %self.label(p),
                    cool_s = struck.cool.as_secs(),
                    "cluster path: could not dial: {}",
                    failure.why
                );
            }
            suspected(self.app, p, struck.suspect);
            failed.push(format!("{}: {}", self.label(p), failure.why));
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
            // The session connected: whatever stalled it, its source reached.
            let struck = paths.strike(on.path, now, Fault::After);
            tracing::warn!(
                app = self.app,
                path = %label(&self.target, on.path),
                cool_s = struck.cool.as_secs(),
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
        dial_from(ports, vec![None])
    }

    /// Dialling every destination from `sources`.
    fn dial_from(ports: &[u16], sources: Vec<Option<IpAddr>>) -> Dial {
        let spec = ports
            .iter()
            .map(|p| format!("http://127.0.0.1:{p}"))
            .collect::<Vec<_>>()
            .join(" ");
        Dial::with(
            Arc::new(Target::parse(&spec, None).unwrap()),
            "zend",
            tls_connector().unwrap(),
            timing(),
            Arc::new(move |_| sources.clone()),
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
        let (_tx, rx) = d.connect().await.expect("the second endpoint answers");
        assert_eq!(on(&d), 1);
        assert!(
            rx.heard().is_some(),
            "the handshake's reply came through the stamped transport"
        );
        let first = d.target.endpoints[0].port;
        let cooling = lock(&d.paths).cooling(
            PathId {
                endpoint: 0,
                addr: SocketAddr::from(([127, 0, 0, 1], first)),
                source: None,
            },
            Instant::now(),
        );
        assert!(cooling, "the endpoint that refused is cooling");
    }

    /// A source that cannot reach — here an address not on this machine, so
    /// binding it fails — is passed over for the next, and turns suspect; the
    /// next dial starts from the source that worked.
    #[tokio::test]
    async fn a_source_that_cannot_reach_is_passed_over_and_dialled_last() {
        let (up, _g) = gateway().await;
        let dead = IpAddr::from([192, 0, 2, 1]);
        let live = IpAddr::from([127, 0, 0, 1]);
        let mut d = dial_from(&[up], vec![Some(dead), Some(live)]);

        let _s = d.connect().await.expect("the second source answers");
        let path = d.on.as_ref().unwrap().path;
        assert_eq!(path.source, Some(live));
        let dead_path = PathId {
            source: Some(dead),
            ..path
        };
        assert!(lock(&d.paths).suspect(dead_path));

        d.ended(&Attempt::Session(Ended::Dropped));
        let _s = d.connect().await.unwrap();
        assert_eq!(d.on.as_ref().unwrap().path.source, Some(live));
        assert!(
            lock(&d.paths).suspect(dead_path),
            "a success from another source does not clear it"
        );
        assert!(label(&d.target, path).ends_with(&format!("from {live})")));
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
