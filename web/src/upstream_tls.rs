//! The proxy's TLS to an `https://` upstream.
//!
//! Two clients, chosen per route. The **verified** one checks the upstream's
//! certificate against the Mozilla roots built into the binary, as any client
//! would. The
//! **self-signed** one is for a daemon behind the gateway that made its own
//! certificate at startup ([`crate::tls::self_signed`]): there is no CA to
//! check it against and no fixed key to pin, so the chain is accepted as
//! presented — but the handshake signatures are still verified, so the peer has
//! to hold the key of the certificate it sent. That makes the hop private from
//! anyone watching the network; it does not prove *which* machine answered,
//! which is why it is for the private network the gateway already trusts with
//! its identity headers, and only on a route that says `self_signed: true`.
//!
//! HTTP/1.1 only, either way: a websocket is an HTTP/1.1 upgrade, and the proxy
//! tunnels it after the upstream's `101`.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use hyper_rustls::{ConfigBuilderExt, HttpsConnector, HttpsConnectorBuilder};
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::ring::default_provider;
use rustls::crypto::{verify_tls12_signature, verify_tls13_signature, CryptoProvider};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{ClientConfig, DigitallySignedStruct, Error as TlsError, SignatureScheme};

pub type HttpClient = Client<HttpsConnector<HttpConnector>, Body>;

/// The clients a forward goes out through.
#[derive(Clone)]
pub struct Clients {
    pub verified: HttpClient,
    pub self_signed: HttpClient,
}

impl Clients {
    pub fn new(connect_timeout: Duration) -> Self {
        let provider = Arc::new(default_provider());
        let verified = ClientConfig::builder_with_provider(provider.clone())
            .with_safe_default_protocol_versions()
            .expect("ring supports the default protocol versions")
            .with_webpki_roots()
            .with_no_client_auth();
        let self_signed = ClientConfig::builder_with_provider(provider.clone())
            .with_safe_default_protocol_versions()
            .expect("ring supports the default protocol versions")
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(AcceptSelfSigned(provider)))
            .with_no_client_auth();
        Self {
            verified: client(verified, connect_timeout),
            self_signed: client(self_signed, connect_timeout),
        }
    }

    /// The client for a route.
    pub fn for_route(&self, self_signed: bool) -> &HttpClient {
        if self_signed {
            &self.self_signed
        } else {
            &self.verified
        }
    }
}

fn client(tls: ClientConfig, connect_timeout: Duration) -> HttpClient {
    let mut connector = HttpConnector::new();
    connector.set_nodelay(true); // SSE frames must not wait on Nagle
    connector.set_connect_timeout(Some(connect_timeout));
    connector.enforce_http(false);
    let https = HttpsConnectorBuilder::new()
        .with_tls_config(tls)
        .https_or_http()
        .enable_http1()
        .wrap_connector(connector);
    Client::builder(TokioExecutor::new())
        .pool_idle_timeout(Duration::from_secs(30))
        .build(https)
}

/// Accepts the certificate an upstream made for itself, and verifies the
/// handshake was signed with its key.
#[derive(Debug)]
struct AcceptSelfSigned(Arc<CryptoProvider>);

impl ServerCertVerifier for AcceptSelfSigned {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, TlsError> {
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        verify_tls12_signature(
            message,
            cert,
            dss,
            &self.0.signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        verify_tls13_signature(
            message,
            cert,
            dss,
            &self.0.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.0.signature_verification_algorithms.supported_schemes()
    }
}
