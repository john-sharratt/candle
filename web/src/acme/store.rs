//! The ACME store on disk: the account, one certificate per site, and the
//! ledger.
//!
//! ```text
//! <store>/account.json
//! <store>/ledger.json
//! <store>/certs/<site>.key.pem
//! <store>/certs/<site>.crt.pem   chain, leaf first
//! ```
//!
//! Every write goes to a temporary file that is then renamed over the target,
//! so an interrupted write leaves the previous file whole rather than a
//! truncated one. A site's key is written before its certificate, so a
//! certificate on disk always has a key beside it — and the pair is checked on
//! load, because a crash between the two writes leaves a renewal's new key
//! beside the old certificate ([`Store::load_certs`]).

use std::fs::File;
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use instant_acme::AccountCredentials;
use rustls::crypto::ring::default_provider;
use rustls::sign::CertifiedKey;
use rustls::Error as TlsError;
use rustls_pki_types::pem::PemObject;
use rustls_pki_types::{CertificateDer, PrivateKeyDer};
use x509_parser::extensions::GeneralName;
use x509_parser::parse_x509_certificate;

/// Write `bytes` to `path` through a temporary sibling and a rename.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let tmp = path.with_extension("tmp");
    {
        let mut f = File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path).with_context(|| format!("replacing {}", path.display()))
}

/// One site's certificate as the store holds it.
pub struct Stored {
    pub site: String,
    /// The DNS names the leaf covers.
    pub names: Vec<String>,
    /// Unix seconds.
    pub not_before: u64,
    pub not_after: u64,
    /// The chain (leaf first) with its signing key, checked to belong to it.
    pub certified: Arc<CertifiedKey>,
}

impl Stored {
    /// Whether the leaf is past two thirds of its life — see
    /// [`past_renewal_floor`].
    pub fn past_renewal_floor(&self, now: u64) -> bool {
        past_renewal_floor(self.not_before, self.not_after, now)
    }
}

/// Whether a certificate valid from `not_before` to `not_after` is past two
/// thirds of its life at `now` — the floor below which it is renewed whatever
/// the CA's renewal window says. A third of a 90-day certificate is 30 days,
/// and it scales with shorter lifetimes.
pub fn past_renewal_floor(not_before: u64, not_after: u64, now: u64) -> bool {
    let life = not_after.saturating_sub(not_before);
    not_after.saturating_sub(now) < life / 3
}

pub struct Store {
    root: PathBuf,
}

impl Store {
    pub fn new(store: &Path) -> Self {
        Self {
            root: store.to_path_buf(),
        }
    }

    pub fn ledger_path(&self) -> PathBuf {
        self.root.join("ledger.json")
    }

    fn account_path(&self) -> PathBuf {
        self.root.join("account.json")
    }

    fn certs(&self) -> PathBuf {
        self.root.join("certs")
    }

    pub fn load_account(&self) -> Result<Option<AccountCredentials>> {
        let path = self.account_path();
        match std::fs::read(&path) {
            Ok(b) => Ok(Some(
                serde_json::from_slice(&b)
                    .with_context(|| format!("parsing {}", path.display()))?,
            )),
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
        }
    }

    pub fn save_account(&self, credentials: &AccountCredentials) -> Result<()> {
        write_atomic(
            &self.account_path(),
            &serde_json::to_vec_pretty(credentials)?,
        )
    }

    /// Store a site's issued chain and its key — the key first.
    pub fn save_cert(&self, site: &str, chain_pem: &str, key_pem: &str) -> Result<()> {
        let dir = self.certs();
        write_atomic(&dir.join(format!("{site}.key.pem")), key_pem.as_bytes())?;
        write_atomic(&dir.join(format!("{site}.crt.pem")), chain_pem.as_bytes())
    }

    /// Every certificate in the store whose key belongs to it.
    ///
    /// A file that does not parse is an error: each file is written atomically
    /// by this process alone, so a broken one is something to look at, not
    /// something to quietly re-order over. A pair whose key is not its
    /// certificate's is the one inconsistency the two writes can leave — a
    /// crash between a renewal's new key and its new certificate — and that
    /// pair is set aside, loudly: the site then reads as having no certificate
    /// and is ordered again, rather than failing every handshake until its old
    /// certificate's renewal comes round.
    pub fn load_certs(&self) -> Result<Vec<Stored>> {
        let dir = self.certs();
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(e) if e.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e).with_context(|| format!("listing {}", dir.display())),
        };
        let mut out = Vec::new();
        for entry in entries {
            let path = entry?.path();
            let Some(site) = path
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(|n| n.strip_suffix(".crt.pem"))
            else {
                continue;
            };
            if let Some(stored) = load_pair(site, &path, &dir.join(format!("{site}.key.pem")))? {
                out.push(stored);
            }
        }
        out.sort_by(|a, b| a.site.cmp(&b.site));
        Ok(out)
    }
}

/// One site's pair, or `None` when its key is not its certificate's.
fn load_pair(site: &str, cert: &Path, key_path: &Path) -> Result<Option<Stored>> {
    let chain = CertificateDer::pem_file_iter(cert)
        .with_context(|| format!("reading {}", cert.display()))?
        .collect::<Result<Vec<_>, _>>()
        .with_context(|| format!("parsing {}", cert.display()))?;
    let Some(leaf) = chain.first().cloned() else {
        bail!("{} holds no certificate", cert.display());
    };
    let key = PrivateKeyDer::from_pem_file(key_path)
        .with_context(|| format!("reading {}", key_path.display()))?;
    let certified = match CertifiedKey::from_der(chain, key, &default_provider()) {
        Ok(c) => Arc::new(c),
        Err(TlsError::InconsistentKeys(why)) => {
            tracing::error!(
                %site, ?why,
                "acme: {} is not the key of {} — the pair is set aside and the site ordered again",
                key_path.display(),
                cert.display()
            );
            return Ok(None);
        }
        Err(e) => return Err(e).with_context(|| format!("loading the key {}", key_path.display())),
    };
    let (_, parsed) = parse_x509_certificate(leaf.as_ref())
        .map_err(|e| anyhow!("parsing the leaf of {}: {e}", cert.display()))?;
    let mut names = Vec::new();
    if let Ok(Some(san)) = parsed.subject_alternative_name() {
        for n in &san.value.general_names {
            if let GeneralName::DNSName(d) = n {
                names.push(d.to_ascii_lowercase());
            }
        }
    }
    let validity = parsed.validity();
    Ok(Some(Stored {
        site: site.to_owned(),
        names,
        not_before: validity.not_before.timestamp().max(0) as u64,
        not_after: validity.not_after.timestamp().max(0) as u64,
        certified,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn self_signed(names: &[&str]) -> (String, String) {
        let ck = rcgen::generate_simple_self_signed(
            names.iter().map(|n| n.to_string()).collect::<Vec<_>>(),
        )
        .unwrap();
        (ck.cert.pem(), ck.signing_key.serialize_pem())
    }

    #[test]
    fn a_saved_certificate_loads_with_its_names_and_dates() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::new(dir.path());
        let (cert, key) = self_signed(&["tokera.com", "www.tokera.com"]);
        store.save_cert("tokera", &cert, &key).unwrap();

        let certs = store.load_certs().unwrap();
        assert_eq!(certs.len(), 1);
        assert_eq!(certs[0].site, "tokera");
        assert_eq!(certs[0].names, vec!["tokera.com", "www.tokera.com"]);
        assert!(certs[0].not_after > certs[0].not_before);
        assert!(dir.path().join("certs/tokera.crt.pem").is_file());
        assert!(dir.path().join("certs/tokera.key.pem").is_file());
        assert!(
            !dir.path().join("certs/tokera.crt.tmp").exists(),
            "no temporary left"
        );
    }

    #[test]
    fn an_empty_store_holds_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::new(dir.path());
        assert!(store.load_certs().unwrap().is_empty());
        assert!(store.load_account().unwrap().is_none());
    }

    #[test]
    fn a_certificate_without_its_key_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::new(dir.path());
        let (cert, _) = self_signed(&["tokera.com"]);
        write_atomic(&dir.path().join("certs/tokera.crt.pem"), cert.as_bytes()).unwrap();
        assert!(store.load_certs().is_err());
    }

    /// **A renewal's new key beside the old certificate is set aside, not
    /// served.** That is what a crash between the two writes leaves, and served
    /// it fails every handshake for the site until the old certificate's
    /// renewal comes round. Set aside, the site reads as having none and is
    /// ordered again; its neighbours load as before.
    #[test]
    fn a_key_that_is_not_its_certificates_is_set_aside() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::new(dir.path());
        let (old_cert, _) = self_signed(&["tokera.com"]);
        let (_, new_key) = self_signed(&["tokera.com"]);
        let (other_cert, other_key) = self_signed(&["bot.tokera.com"]);
        store.save_cert("tokera", &old_cert, &new_key).unwrap();
        store.save_cert("npcd", &other_cert, &other_key).unwrap();

        let sites: Vec<String> = store
            .load_certs()
            .unwrap()
            .into_iter()
            .map(|s| s.site)
            .collect();
        assert_eq!(sites, vec!["npcd"]);
    }

    /// Renewed with a third of its life left: 30 days of a 90-day certificate.
    #[test]
    fn the_renewal_floor_is_a_third_of_the_lifetime() {
        let day = 86_400;
        assert!(!past_renewal_floor(0, 90 * day, 60 * day));
        assert!(past_renewal_floor(0, 90 * day, 60 * day + 1));
    }
}
