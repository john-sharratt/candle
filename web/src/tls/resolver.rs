//! The certificate each handshake presents, chosen by the name the client asked
//! for (SNI), from whatever the ACME store holds right now.
//!
//! Renewal replaces a site's certificate while the entrance is serving, so the
//! table is swapped in place: a handshake already under way keeps the key it
//! picked, and the next one sees the new certificate. Nothing restarts.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use rustls::server::{ClientHello, ResolvesServerCert};
use rustls::sign::CertifiedKey;

use crate::acme::store::Stored;

/// Name → certificate, for every name a stored certificate covers.
#[derive(Debug, Default)]
pub struct Resolver {
    by_name: RwLock<HashMap<String, Arc<CertifiedKey>>>,
}

impl Resolver {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Replace the whole table with these certificates.
    pub fn install(&self, certs: &[Stored]) {
        let mut table = HashMap::new();
        for c in certs {
            for name in &c.names {
                table.insert(name.clone(), c.certified.clone());
            }
        }
        *self.by_name.write().expect("resolver poisoned") = table;
    }

    /// The names a certificate is held for.
    pub fn names(&self) -> Vec<String> {
        let mut n: Vec<String> = self
            .by_name
            .read()
            .expect("resolver poisoned")
            .keys()
            .cloned()
            .collect();
        n.sort();
        n
    }

    fn lookup(&self, name: &str) -> Option<Arc<CertifiedKey>> {
        self.by_name
            .read()
            .expect("resolver poisoned")
            .get(&name.trim_end_matches('.').to_ascii_lowercase())
            .cloned()
    }
}

impl ResolvesServerCert for Resolver {
    /// A name with no certificate fails the handshake. There is no default to
    /// present instead: any other certificate would be for the wrong name, and
    /// a browser shows that as an attack rather than as an absent site.
    fn resolve(&self, hello: ClientHello<'_>) -> Option<Arc<CertifiedKey>> {
        self.lookup(hello.server_name()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acme::store::Store;
    use std::path::Path;

    fn stored(dir: &Path, site: &str, names: &[&str]) -> Vec<Stored> {
        let store = Store::new(dir);
        let ck = rcgen::generate_simple_self_signed(
            names.iter().map(|n| n.to_string()).collect::<Vec<_>>(),
        )
        .unwrap();
        store
            .save_cert(site, &ck.cert.pem(), &ck.signing_key.serialize_pem())
            .unwrap();
        store.load_certs().unwrap()
    }

    #[test]
    fn every_name_a_certificate_covers_resolves_to_it() {
        let dir = tempfile::tempdir().unwrap();
        let r = Resolver::new();
        r.install(&stored(
            dir.path(),
            "tokera",
            &["tokera.com", "www.tokera.com"],
        ));
        assert_eq!(r.names(), vec!["tokera.com", "www.tokera.com"]);
        let a = r.lookup("tokera.com").expect("covered");
        let b = r
            .lookup("WWW.Tokera.COM.")
            .expect("matched case-blind, root dot dropped");
        assert!(Arc::ptr_eq(&a, &b), "one certificate for both names");
        assert!(r.lookup("code.tokera.com").is_none());
    }

    /// A renewal swaps the table: the old certificate is gone for the next
    /// handshake.
    #[test]
    fn installing_again_replaces_the_table() {
        let dir = tempfile::tempdir().unwrap();
        let r = Resolver::new();
        r.install(&stored(dir.path(), "a", &["a.net"]));
        let first = r.lookup("a.net").unwrap();
        r.install(&stored(dir.path(), "a", &["a.net"]));
        assert!(!Arc::ptr_eq(&first, &r.lookup("a.net").unwrap()));
    }
}
