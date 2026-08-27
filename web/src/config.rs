//! The site table, in YAML.
//!
//! One file describes everything this server does, in both of its roles:
//!
//!   * **Authoritative** — it owns the files and answers `/v1` from an API
//!     merged into the same process (`upstream: local`). This is what `zend`
//!     and `npcd` run: one binary, testable on its own.
//!   * **Proxy** — it owns nothing and forwards by hostname to whichever
//!     machine is running that daemon. This is what sits in the DMZ.
//!
//! The same config shape covers both, so moving a site from one to the other is
//! an edit to one line rather than a different deployment.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub server: Server,
    #[serde(default)]
    pub sites: Vec<Site>,
    /// Sign-in, for the whole estate rather than per site — one account across
    /// every hostname is the point. Absent means no sign-in anywhere.
    ///
    /// Normally supplied by [`auth_file`](Self::auth_file) rather than written
    /// here, so this table stays identical on every machine.
    #[serde(default)]
    pub auth: Option<Auth>,
    /// A file to read [`auth`](Self::auth) from, resolved against this config.
    ///
    /// This is what lets the site table be pulled onto a deployment without
    /// touching its credentials. The block used to live inline, which meant the
    /// one machine that had a real `client_id` carried a locally-modified
    /// `web.yaml` forever — and every attempt to update the sites there failed
    /// with *your local changes would be overwritten*, on exactly the file that
    /// had nothing deployment-specific about it except those four lines.
    ///
    /// **A missing file means sign-in is off**, which is the same thing the
    /// commented-out block used to mean: a public site must not stop serving
    /// because a key it does not need is absent. A file that *is* present and
    /// does not parse is fatal, because at that point the deployment has said
    /// it wants sign-in and a gateway that quietly has none is worse.
    #[serde(default)]
    pub auth_file: Option<PathBuf>,
    /// Directory the config was loaded from; relative roots resolve against it.
    #[serde(skip)]
    pub base: PathBuf,
}

/// OpenID Connect sign-in, owned by the gateway.
///
/// The gateway is the only ingress, so it is the only place that can hold one
/// session for every hostname: the cookie is issued on the parent domain and
/// the browser presents it to each subdomain by itself. Daemons behind the
/// gateway never run their own OAuth dance; they read the identity the gateway
/// asserts (see [`crate::auth`]).
#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct Auth {
    /// Cookie `Domain`. A leading dot is what makes one sign-in reach
    /// `code.` and `bot.` — without it the cookie is confined to one host.
    pub cookie_domain: String,
    /// How long a session lasts. Long by default: this is a personal estate,
    /// and being signed out weekly is friction with no security payoff.
    #[serde(default = "d_session_hours")]
    pub session_ttl_hours: u64,
    /// File holding the HMAC key that signs sessions. A file, never an inline
    /// value, so the site table can be read and committed without carrying a
    /// secret. Rotating it signs everyone out, which is the intended lever.
    pub session_secret_file: PathBuf,
    pub google: Provider,
}

fn d_session_hours() -> u64 {
    24 * 30
}

/// An OIDC provider. Endpoints are settable because a test needs to point the
/// flow at a local one — the same reason any OIDC client takes an issuer.
#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct Provider {
    pub client_id: String,
    pub client_secret_file: PathBuf,
    /// Must exactly match a redirect URI registered with the provider. Its
    /// scheme also decides whether cookies are marked `Secure`, so an https
    /// deployment cannot forget to set that.
    pub redirect_uri: String,
    #[serde(default = "d_google_auth")]
    pub auth_endpoint: String,
    #[serde(default = "d_google_token")]
    pub token_endpoint: String,
}

fn d_google_auth() -> String {
    "https://accounts.google.com/o/oauth2/v2/auth".into()
}
fn d_google_token() -> String {
    "https://oauth2.googleapis.com/token".into()
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Server {
    #[serde(default = "d_bind")]
    pub bind: SocketAddr,
    #[serde(default = "d_cache")]
    pub cache_control: String,
    #[serde(default)]
    pub backoff: Backoff,
    /// Upstream response headers are streamed, never collected — but a request
    /// that never gets its first byte has to give up eventually.
    #[serde(default = "d_connect_timeout_ms")]
    pub connect_timeout_ms: u64,
}

fn d_bind() -> SocketAddr {
    "127.0.0.1:8080".parse().unwrap()
}
fn d_cache() -> String {
    "no-store".into()
}
fn d_connect_timeout_ms() -> u64 {
    5_000
}

impl Default for Server {
    fn default() -> Self {
        Self {
            bind: d_bind(),
            cache_control: d_cache(),
            backoff: Backoff::default(),
            connect_timeout_ms: d_connect_timeout_ms(),
        }
    }
}

/// Exponential backoff per upstream.
///
/// A daemon that is down stays down for a while, and hammering it neither helps
/// it nor helps the caller. After a failure the upstream is held open for
/// `initial_ms`, doubling to `max_ms`, and requests during that window fail
/// fast rather than waiting for another connect timeout. One probe is allowed
/// through when the window expires, so recovery is automatic and needs no
/// operator action.
#[derive(Debug, Deserialize, Clone, Copy)]
#[serde(deny_unknown_fields)]
pub struct Backoff {
    #[serde(default = "d_backoff_initial")]
    pub initial_ms: u64,
    #[serde(default = "d_backoff_max")]
    pub max_ms: u64,
}

fn d_backoff_initial() -> u64 {
    250
}
fn d_backoff_max() -> u64 {
    10_000
}

impl Default for Backoff {
    fn default() -> Self {
        Self {
            initial_ms: d_backoff_initial(),
            max_ms: d_backoff_max(),
        }
    }
}

impl Backoff {
    /// Delay after `failures` consecutive failures, capped at `max_ms`.
    pub fn delay(&self, failures: u32) -> Duration {
        let shift = failures.saturating_sub(1).min(31);
        let ms = self
            .initial_ms
            .saturating_mul(1u64 << shift)
            .min(self.max_ms);
        Duration::from_millis(ms)
    }
}

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
    /// Directory holding the source documents this site publishes as papers.
    ///
    /// Deliberately **not** a content root: only documents named in the site's
    /// `papers.yaml` are reachable, so pointing this at `docs/` publishes the
    /// two papers rather than the whole design directory. The papers stay live
    /// against the working documents, which is the point of not copying them.
    #[serde(default)]
    pub papers: Option<PathBuf>,
    #[serde(skip)]
    pub roots_abs: Vec<PathBuf>,
    #[serde(skip)]
    pub papers_abs: Option<PathBuf>,
}

fn d_fallback() -> String {
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

impl Site {
    pub fn matches_host(&self, host: &str) -> bool {
        self.hosts.iter().any(|pattern| {
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

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Self::from_yaml(&text, path.parent().unwrap_or(Path::new(".")))
            .with_context(|| format!("in {}", path.display()))
    }

    /// Parse from a string — the path an embedded daemon takes, since its
    /// config is `include_str!`d rather than read from disk.
    pub fn from_yaml(text: &str, base: &Path) -> Result<Self> {
        let mut cfg: Config = serde_yaml::from_str(text).context("parsing YAML")?;
        cfg.base = base.to_path_buf();
        cfg.finish()?;
        Ok(cfg)
    }

    /// Fold `auth_file` into [`auth`](Self::auth).
    ///
    /// Absent file, sign-in off. Present and broken, hard failure — the two
    /// halves of the rule that keeps a public site serving without credentials
    /// while refusing to run a half-configured one.
    fn load_auth_file(&mut self, base: &Path) -> Result<()> {
        let Some(rel) = self.auth_file.clone() else {
            return Ok(());
        };
        if self.auth.is_some() {
            // Silently preferring one would make the other look ignored, and
            // which one won would depend on knowing this function exists.
            bail!("both `auth:` and `auth_file:` are set; use one");
        }

        let path = if rel.is_absolute() {
            rel
        } else {
            base.join(rel)
        };
        // Kept resolved, so `--check` can name the exact path it looked at
        // rather than the relative one it was given.
        self.auth_file = Some(path.clone());

        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            // Distinguished on purpose: a file that is not there is a
            // deployment without sign-in, and any other error — unreadable,
            // a directory, a bad mount — is a deployment that meant to have it.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                tracing::info!("sign-in: no {} — running without it", path.display());
                return Ok(());
            }
            Err(e) => bail!("reading {}: {e}", path.display()),
        };

        self.auth = Some(
            serde_yaml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?,
        );
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        if self.sites.is_empty() {
            bail!("no sites declared — nothing to serve");
        }
        match self.sites.iter().filter(|s| s.default).count() {
            0 => bail!("no site has `default: true` — an unmatched Host would have nowhere to go"),
            1 => {}
            n => bail!("{n} sites have `default: true`; exactly one may"),
        }

        let base = self.base.clone();

        self.load_auth_file(&base)?;

        // Secret paths resolve against the config file, exactly like `roots`
        // and `papers`. Read as given they would follow the working directory
        // instead, so the same config would work under systemd and fail from a
        // shell — and a relative path is what makes the table portable between
        // a laptop and the DMZ box in the first place.
        if let Some(auth) = &mut self.auth {
            for p in [
                &mut auth.session_secret_file,
                &mut auth.google.client_secret_file,
            ] {
                if p.is_relative() {
                    *p = base.join(&*p);
                }
            }
        }

        for site in &mut self.sites {
            for root in &site.roots {
                let abs = if root.is_absolute() {
                    root.clone()
                } else {
                    base.join(root)
                };
                let real = abs.canonicalize().with_context(|| {
                    format!(
                        "site `{}`: root {} does not exist",
                        site.name,
                        abs.display()
                    )
                })?;
                site.roots_abs.push(real);
            }
            if let Some(dir) = &site.papers {
                let abs = if dir.is_absolute() {
                    dir.clone()
                } else {
                    base.join(dir)
                };
                site.papers_abs = Some(abs.canonicalize().with_context(|| {
                    format!(
                        "site `{}`: papers directory {} does not exist",
                        site.name,
                        abs.display()
                    )
                })?);
            }
            for r in &site.api {
                if !r.prefix.starts_with('/') {
                    bail!(
                        "site `{}`: prefix `{}` must start with /",
                        site.name,
                        r.prefix
                    );
                }
                if let Upstream::Url(u) = &r.upstream {
                    if !u.starts_with("http://") && !u.starts_with("https://") {
                        bail!(
                            "site `{}`: upstream `{u}` must be `local` or an http(s) URL",
                            site.name
                        );
                    }
                }
            }
            // A site with no roots is a pure gateway — legal, and how a proxy
            // that owns no content is expressed.
            if site.roots.is_empty() && site.api.is_empty() {
                bail!(
                    "site `{}` has neither roots nor api routes — it can answer nothing",
                    site.name
                );
            }
        }
        Ok(())
    }

    pub fn site_for(&self, host: Option<&str>) -> &Site {
        if let Some(raw) = host {
            let h = raw.trim();
            let h = h
                .rsplit_once(':')
                .filter(|(head, _)| !head.contains(':') || head.starts_with('['))
                .map(|(head, _)| head)
                .unwrap_or(h);
            let h = h
                .trim_matches(|c| c == '[' || c == ']')
                .to_ascii_lowercase();
            if let Some(s) = self.sites.iter().find(|s| s.matches_host(&h)) {
                return s;
            }
        }
        self.sites
            .iter()
            .find(|s| s.default)
            .expect("validated: exactly one default")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const YAML: &str = r#"
server:
  bind: "0.0.0.0:80"
sites:
  - name: npcd
    hosts: ["npcd.test", "*.npcd.dev"]
    roots: ["."]
    default: true
    api:
      - prefix: /v1
        upstream: local
      - prefix: /v1/stream
        upstream: "http://10.0.0.9:8081"
"#;

    fn cfg() -> Config {
        Config::from_yaml(YAML, Path::new(".")).expect("valid")
    }

    #[test]
    fn yaml_parses_and_resolves_hosts() {
        let c = cfg();
        assert_eq!(c.server.bind.port(), 80);
        assert_eq!(c.site_for(Some("NPCD.test:8080")).name, "npcd");
        assert_eq!(c.site_for(Some("a.npcd.dev")).name, "npcd");
        assert_eq!(c.site_for(Some("nothing.example")).name, "npcd"); // default
    }

    #[test]
    fn local_and_url_upstreams_both_parse() {
        let c = cfg();
        let s = &c.sites[0];
        assert_eq!(s.route_for("/v1/status").unwrap().upstream, Upstream::Local);
        assert_eq!(
            s.route_for("/v1/stream/x").unwrap().upstream,
            Upstream::Url("http://10.0.0.9:8081".into())
        );
    }

    #[test]
    fn wildcard_excludes_the_apex() {
        let c = cfg();
        assert!(c.sites[0].matches_host("x.npcd.dev"));
        assert!(!c.sites[0].matches_host("npcd.dev"));
    }

    #[test]
    fn prefix_respects_segment_boundaries() {
        assert!(segment_prefix("/v1", "/v1"));
        assert!(segment_prefix("/v1/x", "/v1"));
        assert!(!segment_prefix("/v1x", "/v1"));
    }

    #[test]
    fn backoff_doubles_then_caps_at_ten_seconds() {
        let b = Backoff {
            initial_ms: 250,
            max_ms: 10_000,
        };
        assert_eq!(b.delay(1).as_millis(), 250);
        assert_eq!(b.delay(2).as_millis(), 500);
        assert_eq!(b.delay(3).as_millis(), 1_000);
        assert_eq!(b.delay(6).as_millis(), 8_000);
        assert_eq!(b.delay(7).as_millis(), 10_000); // capped
        assert_eq!(b.delay(99).as_millis(), 10_000); // and stays capped
    }

    #[test]
    fn two_defaults_is_an_error() {
        let y = "sites:\n  - {name: a, roots: ['.'], default: true}\n  - {name: b, roots: ['.'], default: true}\n";
        assert!(Config::from_yaml(y, Path::new(".")).is_err());
    }

    #[test]
    fn no_default_is_an_error() {
        let y = "sites:\n  - {name: a, roots: ['.']}\n";
        assert!(Config::from_yaml(y, Path::new(".")).is_err());
    }

    #[test]
    fn unknown_key_is_rejected_rather_than_ignored() {
        // A typo in a config that silently does nothing is worse than a refusal.
        let y = "sites:\n  - {name: a, roots: ['.'], default: true, hsots: []}\n";
        assert!(Config::from_yaml(y, Path::new(".")).is_err());
    }

    // ── auth_file ───────────────────────────────────────────────────────────
    //
    // The site table has to be identical on every machine, so the one piece
    // that is not — a real `client_id` — lives in a file beside it. These pin
    // the three states that split creates.

    /// A directory of this test's own. Counted rather than timestamped: the
    /// platform clock is coarse enough that tests starting together get the
    /// same nanosecond, and then the same directory.
    fn tmpdir() -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "web-authfile-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn with_auth_file(dir: &Path, body: Option<&str>) -> Result<Config> {
        if let Some(b) = body {
            std::fs::create_dir_all(dir.join("secrets")).unwrap();
            std::fs::write(dir.join("secrets").join("auth.yaml"), b).unwrap();
        }
        let y = format!(
            "auth_file: \"secrets/auth.yaml\"\n{}",
            YAML.trim_start_matches('\n')
        );
        Config::from_yaml(&y, dir)
    }

    const AUTH: &str = r#"
cookie_domain: ".tokera.com"
session_ttl_hours: 720
session_secret_file: "secrets/session.key"
google:
  client_id: "abc.apps.googleusercontent.com"
  client_secret_file: "secrets/google.secret"
  redirect_uri: "https://tokera.com/auth/callback"
"#;

    /// Absent file, sign-in off — the same meaning the commented-out block had.
    /// A public site must not stop serving because a key it does not need is
    /// missing.
    #[test]
    fn a_missing_auth_file_means_no_sign_in_rather_than_a_failure() {
        let dir = tmpdir();
        let c = with_auth_file(&dir, None).expect("a missing auth file is not an error");
        assert!(c.auth.is_none());
        // And the resolved path is kept, so `--check` can say what it looked for.
        assert_eq!(c.auth_file.unwrap(), dir.join("secrets").join("auth.yaml"));
    }

    #[test]
    fn a_present_auth_file_configures_sign_in() {
        let dir = tmpdir();
        let c = with_auth_file(&dir, Some(AUTH)).expect("valid auth file");
        let a = c.auth.expect("sign-in is configured");
        assert_eq!(a.cookie_domain, ".tokera.com");
        assert_eq!(a.google.client_id, "abc.apps.googleusercontent.com");
        // Its paths resolve against the config, exactly like `roots`.
        assert_eq!(
            a.session_secret_file,
            dir.join("secrets").join("session.key")
        );
    }

    /// Present but broken is fatal. By then the deployment has said it wants
    /// sign-in, and a gateway that quietly has none is the worse outcome —
    /// which is the whole reason a *missing* file is treated differently.
    #[test]
    fn a_broken_auth_file_is_a_hard_failure() {
        for bad in [
            "cookie_domain: [not a string]\n",
            "google: {}\n",
            "cookie_domain: \".x\"\nsession_secret_file: \"k\"\ngoogle:\n  client_id: a\n  client_secret_file: s\n  redirect_uri: r\n  typo: 1\n",
            ": : :\n",
        ] {
            let dir = tmpdir();
            assert!(
                with_auth_file(&dir, Some(bad)).is_err(),
                "{bad:?} was accepted"
            );
        }
    }

    /// Two sources for one block would make whichever lost look ignored.
    #[test]
    fn inline_auth_and_an_auth_file_together_are_refused() {
        let dir = tmpdir();
        std::fs::create_dir_all(dir.join("secrets")).unwrap();
        std::fs::write(dir.join("secrets").join("auth.yaml"), AUTH).unwrap();
        let y = format!(
            "auth_file: \"secrets/auth.yaml\"\nauth:\n{}\n{}",
            AUTH.trim_start_matches('\n')
                .lines()
                .map(|l| format!("  {l}"))
                .collect::<Vec<_>>()
                .join("\n"),
            YAML.trim_start_matches('\n')
        );
        assert!(Config::from_yaml(&y, &dir).is_err());
    }

    /// The shipped table names an auth file and does not carry the block, so
    /// it is the same bytes on every machine and can always be pulled.
    #[test]
    fn the_shipped_table_keeps_its_credentials_out_of_line() {
        let text = include_str!("../web.yaml");
        assert!(
            text.contains("auth_file:"),
            "web.yaml does not name an auth file"
        );
        // Only as a comment. An uncommented `auth:` here is the thing that made
        // the file locally-modified on one box and unpullable ever after.
        for line in text.lines() {
            assert!(
                !line.starts_with("auth:"),
                "web.yaml carries an inline `auth:` block"
            );
        }
    }
}
