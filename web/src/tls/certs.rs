//! The two rustls configurations the entrance serves with — one per transport,
//! because they negotiate different things — both presenting whatever the
//! resolver they are given chooses for the name asked.

use std::sync::Arc;

use anyhow::{Context, Result};
use quinn::crypto::rustls::QuicServerConfig;
use quinn::ServerConfig as QuinnServerConfig;
use rustls::crypto::ring::default_provider;
use rustls::server::ResolvesServerCert;
use rustls::version::TLS13;
use rustls::ServerConfig;

/// ALPN over TCP, in preference order: HTTP/2, then HTTP/1.1 for a client that
/// speaks nothing newer.
pub const TCP_ALPN: [&[u8]; 2] = [b"h2", b"http/1.1"];

/// ALPN over QUIC. HTTP/3 is the only protocol QUIC carries here.
pub const QUIC_ALPN: &[u8] = b"h3";

/// TLS 1.2 and 1.3 over TCP, offering HTTP/2 and HTTP/1.1.
pub fn tcp_config(resolver: Arc<dyn ResolvesServerCert>) -> Result<Arc<ServerConfig>> {
    let mut cfg = ServerConfig::builder_with_provider(Arc::new(default_provider()))
        .with_safe_default_protocol_versions()?
        .with_no_client_auth()
        .with_cert_resolver(resolver);
    cfg.alpn_protocols = TCP_ALPN.iter().map(|p| p.to_vec()).collect();
    Ok(Arc::new(cfg))
}

/// TLS 1.3 only — QUIC is defined over nothing older — offering HTTP/3.
pub fn quic_config(resolver: Arc<dyn ResolvesServerCert>) -> Result<QuinnServerConfig> {
    let mut cfg = ServerConfig::builder_with_provider(Arc::new(default_provider()))
        .with_protocol_versions(&[&TLS13])?
        .with_no_client_auth()
        .with_cert_resolver(resolver);
    cfg.alpn_protocols = vec![QUIC_ALPN.to_vec()];
    let crypto =
        QuicServerConfig::try_from(cfg).context("the TLS configuration cannot carry QUIC")?;
    Ok(QuinnServerConfig::with_crypto(Arc::new(crypto)))
}

#[cfg(test)]
mod tests {
    use super::super::resolver::Resolver;
    use super::*;

    #[test]
    fn tcp_offers_http2_before_http11_and_quic_offers_http3() {
        let tcp = tcp_config(Resolver::new()).unwrap();
        assert_eq!(
            tcp.alpn_protocols,
            vec![b"h2".to_vec(), b"http/1.1".to_vec()]
        );
        quic_config(Resolver::new()).expect("a TLS 1.3 configuration carries QUIC");
    }
}
