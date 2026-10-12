//! What this gateway serves: the ACME store on disk, the resolver the TLS
//! entrance picks certificates from, and the HTTP-01 answers it gives. One
//! place both the renewal loop and the peers install a certificate through, so
//! the store and the resolver never disagree.

use std::sync::Arc;

use anyhow::{anyhow, Result};

use super::challenges::Challenges;
use super::store::{Store, Stored};
use crate::tls::Resolver;

pub struct Certs {
    pub store: Store,
    pub resolver: Arc<Resolver>,
    pub challenges: Challenges,
}

impl Certs {
    /// Every certificate the store holds.
    pub fn held(&self) -> Result<Vec<Stored>> {
        self.store.load_certs()
    }

    /// Write `site`'s chain and key, serve them, and return the leaf's
    /// `not_after`.
    pub fn install(&self, site: &str, chain_pem: &str, key_pem: &str) -> Result<u64> {
        self.store.save_cert(site, chain_pem, key_pem)?;
        let held = self.held()?;
        self.resolver.install(&held);
        held.iter()
            .find(|s| s.site == site)
            .map(|s| s.not_after)
            .ok_or_else(|| anyhow!("{site}'s certificate was written and does not load"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_installed_certificate_is_stored_and_served() {
        let dir = tempfile::tempdir().unwrap();
        let certs = Certs {
            store: Store::new(dir.path()),
            resolver: Resolver::new(),
            challenges: Challenges::default(),
        };
        let ck = rcgen::generate_simple_self_signed(vec!["tokera.com".to_string()]).unwrap();
        let not_after = certs
            .install("tokera", &ck.cert.pem(), &ck.signing_key.serialize_pem())
            .unwrap();
        assert_eq!(certs.held().unwrap()[0].not_after, not_after);
        assert_eq!(certs.resolver.names(), ["tokera.com"]);
    }
}
