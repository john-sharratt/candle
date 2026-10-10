//! Which names get a certificate, grouped one certificate per site.
//!
//! One per site rather than one for the whole estate, so a name that stops
//! reaching this box — a domain whose DNS moved, say — costs its own site's
//! renewal and nobody else's. One per site rather than one per name, so the
//! eleven alternate names that only redirect share a single order instead of
//! spending eleven.

use std::net::IpAddr;

use crate::config::Config;

/// Top-level labels that are never public: a CA cannot validate them, and
/// asking would only spend a failed authorization.
const PRIVATE_TLDS: [&str; 8] = [
    "localhost",
    "test",
    "example",
    "invalid",
    "local",
    "internal",
    "lan",
    "home",
];

/// The names one site's certificate covers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Group {
    /// The site's name — the certificate's name in the store and the ledger.
    pub site: String,
    /// Lowercased, deduplicated, sorted: the same hosts always make the same
    /// list, which is what the ledger compares.
    pub names: Vec<String>,
}

/// Whether a configured host is one a CA can issue for over HTTP-01: a real
/// public name, not a wildcard (HTTP-01 cannot prove one) and not an IP.
pub fn is_public(host: &str) -> bool {
    let h = host.trim_end_matches('.');
    if h.contains('*') || h.parse::<IpAddr>().is_ok() {
        return false;
    }
    let Some((_, tld)) = h.rsplit_once('.') else {
        return false;
    };
    !PRIVATE_TLDS.contains(&tld)
}

/// Every site with at least one public name, in table order.
pub fn groups(cfg: &Config) -> Vec<Group> {
    cfg.sites
        .iter()
        .filter_map(|s| {
            let mut names: Vec<String> = s
                .hosts
                .iter()
                .map(|h| h.trim_end_matches('.').to_ascii_lowercase())
                .filter(|h| is_public(h))
                .collect();
            names.sort();
            names.dedup();
            (!names.is_empty()).then(|| Group {
                site: s.name.clone(),
                names,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn only_public_names_are_issued_for() {
        for public in [
            "tokera.com",
            "code.tokera.com",
            "tokera.com.au",
            "npcd.dev",
            "tokera.sh",
        ] {
            assert!(is_public(public), "{public}");
        }
        for private in [
            "*.npcd.dev",
            "zend.localhost",
            "npcd.test",
            "box.lan",
            "localhost",
            "127.0.0.1",
            "::1",
            "nodot",
        ] {
            assert!(!is_public(private), "{private}");
        }
    }

    #[test]
    fn each_site_is_one_group_of_its_public_names() {
        let y = r#"
sites:
  - name: tokera
    default: true
    hosts: ["tokera.com", "tokera.localhost"]
    roots: ["."]
  - name: npcd
    hosts: ["bot.tokera.com", "npcd.localhost", "npcd.test", "*.npcd.dev"]
    roots: ["."]
  - name: local-only
    hosts: ["x.localhost"]
    roots: ["."]
  - name: alt
    hosts: ["WWW.Tokera.com", "tokera.net", "www.tokera.com"]
    redirect: "https://tokera.com"
"#;
        let cfg = Config::from_yaml(y, Path::new(".")).unwrap();
        assert_eq!(
            groups(&cfg),
            vec![
                Group {
                    site: "tokera".into(),
                    names: vec!["tokera.com".into()],
                },
                Group {
                    site: "npcd".into(),
                    names: vec!["bot.tokera.com".into()],
                },
                Group {
                    site: "alt".into(),
                    names: vec!["tokera.net".into(), "www.tokera.com".into()],
                },
            ]
        );
    }

    /// The shipped table: every public name it serves is covered, once.
    #[test]
    fn the_shipped_table_covers_every_public_name() {
        let cfg = Config::from_yaml(
            include_str!("../../web.yaml"),
            Path::new(env!("CARGO_MANIFEST_DIR")),
        )
        .unwrap();
        let all: Vec<String> = groups(&cfg).into_iter().flat_map(|g| g.names).collect();
        for name in [
            "tokera.com",
            "www.tokera.com",
            "code.tokera.com",
            "bot.tokera.com",
            "battlecities.net",
            "www.battlecities.net",
            "tokera.co.uk",
        ] {
            assert_eq!(all.iter().filter(|n| *n == name).count(), 1, "{name}");
        }
        assert!(!all
            .iter()
            .any(|n| n.contains('*') || n.ends_with(".localhost")));
    }
}
