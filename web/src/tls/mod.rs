//! The TLS entrance: HTTP/3 over QUIC, with HTTP/2 and HTTP/1.1 over TCP behind
//! it, on one port.
//!
//! A browser cannot open HTTP/3 cold — it has no way to know a UDP port is
//! listening — so its first request always arrives over TCP, negotiating HTTP/2
//! by ALPN (or HTTP/1.1, for a client older than that). Every TCP response
//! carries `Alt-Svc: h3=":<port>"`, and the browser's next connection tries QUIC
//! on that port. When UDP is blocked between them it keeps using TCP, which is
//! why both halves are always served together.
//!
//! The plain entrance at `server.bind` is untouched: it is where a tunnel
//! terminating TLS itself delivers requests, where the CA's HTTP-01 challenges
//! arrive, and it serves the same router.
//!
//! At the gateway the certificates come from [`crate::acme`], through a
//! [`Resolver`] that picks one by the name the client asked for and is updated
//! in place when one is renewed. A daemon behind the gateway serves the same
//! entrance with a certificate it makes for itself at startup ([`self_signed`]),
//! for the hop from the gateway to it.

mod certs;
mod origin;
mod quic;
pub mod resolver;
mod self_signed;
mod tcp;

use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::http::HeaderValue;
use axum::Router;
use quinn::Endpoint;
use rustls::server::ResolvesServerCert;
use tokio::net::TcpListener;
use tokio::sync::watch;
use tokio_rustls::TlsAcceptor;

pub use origin::origin_form;
pub use resolver::Resolver;
pub use self_signed::self_signed;

/// Marks a request that arrived over the TLS entrance — its client is on
/// https whatever the deployment's other settings say, which is what
/// `X-Forwarded-Proto` reports to a daemon behind the proxy.
#[derive(Clone, Copy, Debug)]
pub struct SecureEntrance;

/// The `Alt-Svc` value advertising HTTP/3 on `port`, held for a day.
pub fn alt_svc(port: u16) -> HeaderValue {
    HeaderValue::from_str(&format!("h3=\":{port}\"; ma=86400")).expect("ASCII")
}

/// Both halves of the entrance, bound and ready to serve.
pub struct Entrance {
    tcp: TcpListener,
    acceptor: TlsAcceptor,
    quic: Endpoint,
}

impl Entrance {
    /// Bind TCP at `bind`, then UDP on the port TCP was given — so `bind` may
    /// name port 0 and both halves still share one. Both present whatever
    /// `resolver` chooses for the name asked: the ACME [`Resolver`] at the
    /// gateway, a [`self_signed`] certificate at a daemon behind it.
    ///
    /// A port that is taken fails here, before anything is served.
    pub async fn bind(bind: SocketAddr, resolver: Arc<dyn ResolvesServerCert>) -> Result<Self> {
        // Port 0 means "a port both halves can share", and the one the OS
        // gives TCP is not always free for UDP — Windows reserves ranges of
        // UDP ports the TCP allocator still hands out. So an OS-chosen port is
        // tried again until one serves both; a named port is the deployment's
        // and fails as it is.
        const ATTEMPTS: usize = 32;
        let attempts = if bind.port() == 0 { ATTEMPTS } else { 1 };
        let mut last = None;
        for _ in 0..attempts {
            let tcp = TcpListener::bind(bind)
                .await
                .with_context(|| format!("tls: binding tcp {bind}"))?;
            let addr = tcp.local_addr()?;
            match Endpoint::server(certs::quic_config(resolver.clone())?, addr) {
                Ok(quic) => {
                    return Ok(Self {
                        tcp,
                        acceptor: TlsAcceptor::from(certs::tcp_config(resolver)?),
                        quic,
                    })
                }
                Err(e) => last = Some((addr, e)),
            }
        }
        let (addr, e) = last.expect("at least one attempt was made");
        Err(e).with_context(|| format!("tls: binding udp {addr}"))
    }

    /// The address both halves listen on.
    pub fn local_addr(&self) -> Result<SocketAddr> {
        Ok(self.tcp.local_addr()?)
    }

    /// Serve `router` on both halves until `shutdown` resolves, then stop
    /// taking connections and return once every request already in flight on
    /// either half has been answered.
    pub async fn serve(
        self,
        router: Router,
        shutdown: impl Future<Output = ()> + Send + 'static,
    ) -> Result<()> {
        let alt = alt_svc(self.local_addr()?.port());
        let (stop_tx, stop) = watch::channel(false);
        tokio::spawn(async move {
            shutdown.await;
            let _ = stop_tx.send(true);
        });
        tokio::join!(
            tcp::serve(self.tcp, self.acceptor, router.clone(), alt, stop.clone()),
            quic::serve(self.quic, router, stop),
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alt_svc_names_the_port_and_a_day() {
        assert_eq!(alt_svc(443), "h3=\":443\"; ma=86400");
        assert_eq!(alt_svc(8443), "h3=\":8443\"; ma=86400");
    }
}
