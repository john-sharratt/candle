//! A certificate a daemon makes for itself when it starts.
//!
//! For the hop from the gateway to a daemon behind it — not for the public: a
//! browser rejects a certificate no CA signed. A fresh key every start means
//! nothing is kept on disk to leak or rotate, and the gateway, which cannot pin
//! a key that changes each start, is told so per route (`self_signed: true`) and
//! accepts it there only.
//!
//! The one certificate is presented for every handshake, whatever name was
//! asked — including none, which is what a client connecting by IP address
//! sends.

use std::sync::Arc;

use anyhow::{Context, Result};
use rustls::crypto::ring::sign::any_supported_type;
use rustls::server::ResolvesServerCert;
use rustls::sign::{CertifiedKey, SingleCertAndKey};
use rustls_pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};

/// A certificate for `names` — host names, or IP addresses as text — with a key
/// generated now, as a resolver the entrance can serve.
pub fn self_signed(names: &[String]) -> Result<Arc<dyn ResolvesServerCert>> {
    let ck = rcgen::generate_simple_self_signed(names.to_vec())
        .context("generating a self-signed certificate")?;
    let chain: Vec<CertificateDer<'static>> = vec![ck.cert.der().clone()];
    let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(ck.signing_key.serialize_der()));
    let signer = any_supported_type(&key).context("the generated key is not usable")?;
    Ok(Arc::new(SingleCertAndKey::from(CertifiedKey::new(
        chain, signer,
    ))))
}
