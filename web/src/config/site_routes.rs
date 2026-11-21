//! The sites the gateway serves, each mapping its hostnames to content roots,
//! redirects and routes.
//!
//! A site's hostnames are matched lowercased, with the port removed. A request is
//! answered by its longest matching prefix, and an exact match beats any prefix,
//! so a generated home page can stand beside a catch-all for files. A route is
//! served by the app whose members run it once one of them is ready, and
//! otherwise by its configured upstream, which is either handled in this process
//! or forwarded to another machine. A site with only a redirect answers every
//! path with a `301` to its target, so alternate domains never serve copies of
//! the same pages.

use super::*;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Site {
    pub name: String,
    /// Hostnames reaching this site. Matched lowercased with the port stripped;
    /// a leading `*.` matches subdomains but not the bare apex.
    #[serde(default)]
    pub hosts: Vec<String>,
    /// Content roots, searched in order. This is how a site gets its own files
    /// plus the shared ones: `[content/npcd, content/common]` means a request
    /// for `/lib/dom.js` falls through to common while `/pages/roster.js` does
    /// not. One URL tree, several directories.
    #[serde(default)]
    pub roots: Vec<PathBuf>,
    #[serde(default)]
    pub default: bool,
    #[serde(default = "d_fallback")]
    pub fallback: String,
    #[serde(default)]
    pub api: Vec<Route>,
    /// Send every request for this site's hosts somewhere else, permanently.
    ///
    /// A site with this set serves nothing of its own: the path and query are
    /// appended to the target and answered `301`, before routing, before
    /// content, before sign-in. `roots` and `api` on such a site are dead, and
    /// the startup table says so.
    ///
    /// This is how the alternate domains for one brand are consolidated —
    /// `tokera.net` and the rest onto `tokera.com`. **Redirected, not cloned**:
    /// serving the same pages on six hostnames splits a brand against itself
    /// (search engines pick one and the copies count for nothing) and turns one
    /// deploy into six chances to serve a stale one. A `301` puts every link
    /// and every typed address onto the one name.
    #[serde(default)]
    pub redirect: Option<String>,
    /// Directory holding the source documents this site publishes as papers.
    ///
    /// Deliberately **not** a content root: only documents named in the site's
    /// `papers.yaml` are reachable, so pointing this at `docs/` publishes the
    /// two papers rather than the whole design directory. The papers stay live
    /// against the working documents, which is the point of not copying them.
    #[serde(default)]
    pub papers: Option<PathBuf>,
    /// The site is a log source (`docs/mcp_file_tools.md` §6.1): its upstream
    /// answers `/v1/logs` and `/v1/logs/sources`, which the connector's log
    /// tools read through `/gateway/logs/<site>/…`.
    #[serde(default)]
    pub logs: bool,
    /// The scopes a bearer token for this site's `/api` may be granted
    /// (`docs/battlecities_app.md` §6). With any named, the gateway's
    /// authorization server issues tokens for `https://<host>/api` to
    /// whoever signs in, and states a token's scopes upstream as
    /// `X-Tokera-Scope`. Empty, the site's API takes no token.
    #[serde(default)]
    pub api_scopes: Vec<String>,
    #[serde(skip)]
    pub roots_abs: Vec<PathBuf>,
    #[serde(skip)]
    pub papers_abs: Option<PathBuf>,
}

pub(super) fn d_fallback() -> String {
    "index.html".into()
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct Route {
    /// Segment-aware prefix: `/v1` covers `/v1` and `/v1/x`, never `/v1x`.
    pub prefix: String,
    /// Match this path and nothing beneath it. `{prefix: /, exact: true}` is
    /// the reason it exists: a site whose home page is generated still wants
    /// `/base.css` to come from a file, and a bare prefix of `/` would swallow
    /// every asset on the site.
    #[serde(default)]
    pub exact: bool,
    pub upstream: Upstream,
    #[serde(default)]
    pub rewrite_host: bool,
    /// The `https://` upstream presents a certificate it made for itself at
    /// startup ([`crate::tls::self_signed`]), so the proxy accepts it without a
    /// CA. The handshake still has to prove the upstream holds the key; what is
    /// given up is knowing *which* key, so this is for a daemon on the private
    /// network the gateway already trusts with its identity headers, never for
    /// a host across the internet. An `https://` upstream without it is
    /// verified against the Mozilla roots built into the binary.
    #[serde(default)]
    pub self_signed: bool,
    /// The app whose members serve this route once one of them is ready on
    /// this gateway — `zend`, `npcd` or `agent` (§5.7). Until then, and again
    /// when the last leaves, the configured `upstream` serves it.
    #[serde(default)]
    pub cluster: Option<String>,
}

/// Where a prefix is answered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Upstream {
    /// Handled in this process by the API router merged at build time. This is
    /// what makes a daemon standalone.
    Local,
    /// Forwarded to another machine.
    Url(String),
}

impl<'de> Deserialize<'de> for Upstream {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Ok(if s == "local" {
            Upstream::Local
        } else {
            Upstream::Url(s)
        })
    }
}

impl Upstream {
    pub fn url(&self) -> Option<&str> {
        match self {
            Upstream::Url(u) => Some(u),
            Upstream::Local => None,
        }
    }
}

/// A `Host` header's name as sites are matched by it: without its port or an
/// IPv6 literal's brackets, lowercased.
pub fn host_name(raw: &str) -> String {
    let h = raw.trim();
    let h = h
        .rsplit_once(':')
        .filter(|(head, _)| !head.contains(':') || head.starts_with('['))
        .map(|(head, _)| head)
        .unwrap_or(h);
    h.trim_matches(|c| c == '[' || c == ']')
        .to_ascii_lowercase()
}

/// Whether `host` — a [`host_name`] — is one of `hosts`, a site's patterns: a
/// leading `*.` matches subdomains but not the bare apex.
pub fn hosts_match(hosts: &[String], host: &str) -> bool {
    hosts.iter().any(|pattern| {
        let p = pattern.to_ascii_lowercase();
        match p.strip_prefix("*.") {
            Some(suffix) => {
                host.len() > suffix.len() + 1
                    && host.ends_with(suffix)
                    && host.as_bytes()[host.len() - suffix.len() - 1] == b'.'
            }
            None => host == p,
        }
    })
}

impl Site {
    pub fn matches_host(&self, host: &str) -> bool {
        hosts_match(&self.hosts, host)
    }

    /// Longest matching prefix wins, so `/v1/stream` can go somewhere other
    /// than `/v1` without disturbing the general rule.
    pub fn route_for(&self, path: &str) -> Option<&Route> {
        self.api
            .iter()
            .filter(|r| {
                if r.exact {
                    path.trim_end_matches('/') == r.prefix.trim_end_matches('/')
                } else {
                    segment_prefix(path, &r.prefix)
                }
            })
            // An exact match beats any prefix regardless of length, so
            // `{/, exact}` still wins over a `/` catch-all for the home page.
            .max_by_key(|r| (r.exact, r.prefix.len()))
    }
}

pub fn segment_prefix(path: &str, prefix: &str) -> bool {
    let prefix = prefix.trim_end_matches('/');
    if prefix.is_empty() {
        return true;
    }
    if !path.starts_with(prefix) {
        return false;
    }
    matches!(path.as_bytes().get(prefix.len()), None | Some(b'/'))
}
