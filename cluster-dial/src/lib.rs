//! A member's or an agent's connection to one gateway
//! (`docs/replicated_substrate.md` §3.2, §13.3): `wss` to an `https` cluster,
//! its certificate verified against the Mozilla roots built into the binary —
//! by the URL's host, or by `--cluster-name` when the URL names an address —
//! and never self-signed. An `http` cluster is refused unless its host is
//! loopback. A process told the network is down ([`offline`]) dials only a
//! gateway on this machine.
//!
//! A cluster may be reached several ways — a LAN name with two addresses, then
//! the internet name — and a socket runs on the best of them that works
//! ([`paths`]).

pub use cluster::offline;

mod dial;
pub mod paths;
mod target;

pub use dial::{Dial, PathRx, RETURNING};
pub use target::{Endpoint, Target};

use std::sync::Arc;

use rustls::crypto::ring::default_provider;
use rustls::{ClientConfig, RootCertStore};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio_rustls::TlsConnector;
use tokio_tungstenite::WebSocketStream;

/// The largest message a gateway sends: an `EVENT` fragment of 1 MiB and its
/// header, with room to spare.
const MAX_MESSAGE: usize = 2 << 20;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tls_client_builds_on_ring() {
        assert!(tls_connector().is_ok());
    }
}
