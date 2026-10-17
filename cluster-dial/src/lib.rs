//! A member's or an agent's connection to one gateway
//! (`docs/replicated_substrate.md` §3.2, §13.3): `wss` to an `https` cluster,
//! its certificate verified against the Mozilla roots built into the binary —
//! by the URL's host, or by `--cluster-name` when the URL names an address —
//! and never self-signed. An `http` cluster is refused unless its host is
//! loopback. A process told the network is down ([`offline`]) dials only a
//! gateway on this machine.

pub use cluster::offline;

use std::net::IpAddr;
use std::sync::Arc;

use cluster::dial::Connect;
use cluster::pipe::socket::{split, SocketRx, SocketTx};
use cluster::pipe::tungstenite::Tungstenite;
use rustls::crypto::ring::default_provider;
use rustls::{ClientConfig, RootCertStore};
use rustls_pki_types::ServerName;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::TcpStream;
use tokio_rustls::TlsConnector;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;
use tokio_tungstenite::{client_async_with_config, WebSocketStream};

/// The largest message a gateway sends: an `EVENT` fragment of 1 MiB and its
/// header, with room to spare.
const MAX_MESSAGE: usize = 2 << 20;

/// One `--cluster`, parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Target {
    /// As given, for display.
    pub url: String,
    pub host: String,
    pub port: u16,
    /// The name its certificate is verified against; `None` for a plain
    /// loopback cluster.
    pub tls_name: Option<String>,
}

fn is_loopback(host: &str) -> bool {
    host.eq_ignore_ascii_case("localhost")
        || host.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

impl Target {
    /// `url` is `https://<host>[:port]` or, to loopback, `http://…`; a path is
    /// ignored. `name` is `--cluster-name`.
    pub fn parse(url: &str, name: Option<&str>) -> anyhow::Result<Target> {
        let (tls, rest) = if let Some(r) = url.strip_prefix("https://") {
            (true, r)
        } else if let Some(r) = url.strip_prefix("http://") {
            (false, r)
        } else {
            anyhow::bail!("--cluster {url}: expected an https:// URL");
        };
        let authority = rest.split('/').next().unwrap_or_default();
        let (host, port) = if let Some(v6) = authority.strip_prefix('[') {
            let (h, after) = v6
                .split_once(']')
                .ok_or_else(|| anyhow::anyhow!("--cluster {url}: an unclosed IPv6 address"))?;
            (h.to_string(), after.strip_prefix(':'))
        } else {
            match authority.rsplit_once(':') {
                Some((h, p)) => (h.to_string(), Some(p)),
                None => (authority.to_string(), None),
            }
        };
        if host.is_empty() {
            anyhow::bail!("--cluster {url}: no host");
        }
        let port = match port {
            Some(p) => p
                .parse()
                .map_err(|_| anyhow::anyhow!("--cluster {url}: port `{p}` is not a port"))?,
            None if tls => 443,
            None => 80,
        };
        if !tls && !is_loopback(&host) {
            anyhow::bail!(
                "--cluster {url}: a cluster is dialled over https, verified; plain http is \
                 for loopback only"
            );
        }
        Ok(Target {
            url: url.to_string(),
            tls_name: tls.then(|| name.map_or_else(|| host.clone(), str::to_string)),
            host,
            port,
        })
    }

    /// The socket for `app` on this cluster.
    pub fn socket_url(&self, app: &str) -> String {
        let scheme = if self.tls_name.is_some() { "wss" } else { "ws" };
        let host = if self.host.contains(':') {
            format!("[{}]", self.host)
        } else {
            self.host.clone()
        };
        format!("{scheme}://{host}:{}/v1/cluster/{app}", self.port)
    }
}

/// A TLS client trusting the Mozilla roots built into the binary, on `ring`.
pub fn tls_connector() -> anyhow::Result<TlsConnector> {
    let roots = RootCertStore {
        roots: webpki_roots::TLS_SERVER_ROOTS.to_vec(),
    };
    let config = ClientConfig::builder_with_provider(Arc::new(default_provider()))
        .with_safe_default_protocol_versions()?
        .with_root_certificates(roots)
        .with_no_client_auth();
    Ok(TlsConnector::from(Arc::new(config)))
}

/// A byte stream, plain or TLS.
pub trait Io: AsyncRead + AsyncWrite + Send + Unpin {}

impl<T: AsyncRead + AsyncWrite + Send + Unpin> Io for T {}

pub type Socket = WebSocketStream<Box<dyn Io>>;

/// The connector [`cluster::dial::dial`] dials one app's socket through.
pub struct Dial {
    pub target: Arc<Target>,
    pub app: &'static str,
    pub tls: TlsConnector,
}

impl Dial {
    async fn open(&self) -> Result<Socket, String> {
        let t = &self.target;
        offline::refresh();
        if !offline::reaches(&t.host) {
            return Err(offline::refused(&format!("{}:{}", t.host, t.port)));
        }
        let tcp = TcpStream::connect((t.host.as_str(), t.port))
            .await
            .map_err(|e| format!("{}:{}: {e}", t.host, t.port))?;
        let _ = tcp.set_nodelay(true);
        let stream: Box<dyn Io> = match &t.tls_name {
            Some(name) => {
                let name = ServerName::try_from(name.clone())
                    .map_err(|e| format!("`{name}` cannot name a certificate: {e}"))?;
                Box::new(
                    self.tls
                        .connect(name, tcp)
                        .await
                        .map_err(|e| format!("TLS with {}: {e}", t.url))?,
                )
            }
            None => Box::new(tcp),
        };
        let request = t
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
            .map_err(|e| format!("the WebSocket handshake with {}: {e}", t.url))?;
        Ok(ws)
    }
}

impl Connect for Dial {
    type Tx = SocketTx;
    type Rx = SocketRx<Socket, Tungstenite>;

    async fn connect(&mut self) -> Result<(SocketTx, SocketRx<Socket, Tungstenite>), String> {
        Ok(split::<Tungstenite, _, _>(self.open().await?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn https_clusters_verify_their_name_and_http_is_loopback_only() {
        let t = Target::parse("https://code.tokera.com", None).unwrap();
        assert_eq!(
            (t.host.as_str(), t.port, t.tls_name.as_deref()),
            ("code.tokera.com", 443, Some("code.tokera.com"))
        );
        assert_eq!(
            t.socket_url("zend"),
            "wss://code.tokera.com:443/v1/cluster/zend"
        );

        // An address is verified by `--cluster-name`.
        let lan = Target::parse("https://192.168.0.5:9443/", Some("code.tokera.com")).unwrap();
        assert_eq!(
            (lan.host.as_str(), lan.port, lan.tls_name.as_deref()),
            ("192.168.0.5", 9443, Some("code.tokera.com"))
        );
        assert_eq!(
            lan.socket_url("npcd"),
            "wss://192.168.0.5:9443/v1/cluster/npcd"
        );

        let local = Target::parse("http://127.0.0.1:8080", None).unwrap();
        assert_eq!((local.port, local.tls_name.clone()), (8080, None));
        assert_eq!(
            local.socket_url("zend"),
            "ws://127.0.0.1:8080/v1/cluster/zend"
        );
        let v6 = Target::parse("http://[::1]:8080", None).unwrap();
        assert_eq!(v6.socket_url("agent"), "ws://[::1]:8080/v1/cluster/agent");

        for bad in [
            "http://192.168.0.5:8080",
            "ws://127.0.0.1",
            "code.tokera.com",
            "https://:443",
            "https://host:port",
        ] {
            assert!(Target::parse(bad, None).is_err(), "{bad}");
        }
    }

    #[test]
    fn the_tls_client_builds_on_ring() {
        assert!(tls_connector().is_ok());
    }
}
