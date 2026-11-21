//! The server's listening addresses, its TLS entrance, its static-file cache
//! lifetimes and its backoff for upstreams that are down.
//!
//! The TLS entrance listens on TCP and UDP on one port, with certificates from
//! Let's Encrypt. It is opened only where a TLS file names one, so the same table
//! runs on a machine that should not bind port 443. Static files have two
//! lifetimes: documents and code are short, so a refresh shows an edit, and images
//! and fonts are long. Both keep their ETag, so an unchanged file is answered
//! `304`. An upstream that is down is held open for a window that doubles on each
//! failure, so requests fail fast, and one probe is let through when the window
//! ends.

use super::*;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Server {
    #[serde(default = "d_bind")]
    pub bind: SocketAddr,
    #[serde(default)]
    pub cache: Cache,
    #[serde(default)]
    pub backoff: Backoff,
    /// Upstream response headers are streamed, never collected — but a request
    /// that never gets its first byte has to give up eventually.
    #[serde(default = "d_connect_timeout_ms")]
    pub connect_timeout_ms: u64,
    /// The TLS entrance, beside the plain one at [`bind`](Self::bind): HTTP/3
    /// over QUIC, and HTTP/2 or HTTP/1.1 over TCP for a client that cannot reach
    /// UDP, with certificates from Let's Encrypt. See [`crate::tls`] and
    /// [`crate::acme`].
    ///
    /// Normally supplied by [`tls_file`](Self::tls_file), under the same rule
    /// as [`Config::auth_file`]: the file absent means no TLS entrance, so this
    /// table stays identical on a machine that should not bind 443 or ask a CA
    /// for anything; present but unparseable is a hard failure.
    #[serde(default)]
    pub tls: Option<Tls>,
    /// A file to read [`tls`](Self::tls) from, resolved against this config.
    #[serde(default)]
    pub tls_file: Option<PathBuf>,
    /// Backends that call back through this gateway for the person they serve
    /// (`docs/mcp_app.md` §2) — the mcp app, by address. A request from one
    /// may present the person's `/mcp` access token on any path and site, and
    /// is that person; the token names them, so the address alone names
    /// nobody and a listed backend's `X-Tokera-*` headers are stripped like
    /// anyone's. The list keeps a token from working that way from anywhere
    /// else.
    #[serde(default)]
    pub trusted_backends: Vec<IpAddr>,
    /// Send a visitor who arrives on the plain entrance to the TLS one
    /// ([`crate::https_redirect`]). It takes effect only where this process
    /// has a TLS entrance, and never for the CA's challenge, which is asked
    /// for over plain HTTP.
    #[serde(default)]
    pub redirect_http: bool,
}

/// Where the TLS entrance listens, and where its certificates come from.
#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct Tls {
    /// TCP and UDP both, on the same port. `Alt-Svc` advertises HTTP/3 on the
    /// port the TCP answer came from, so the two cannot differ.
    pub bind: SocketAddr,
    pub acme: Acme,
}

/// Certificates from Let's Encrypt, for every public name the site table
/// serves.
#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct Acme {
    /// Where the account, the certificates and the issuance ledger live,
    /// resolved against the config.
    pub store: PathBuf,
}

pub(super) fn d_bind() -> SocketAddr {
    "127.0.0.1:8080".parse().unwrap()
}

pub(super) fn d_connect_timeout_ms() -> u64 {
    5_000
}

/// How long a static file may be reused before the browser asks again.
///
/// # Why two numbers and not one
///
/// This was a single `cache_control` string, and it was `no-cache` — keep the
/// copy, but revalidate every time. That is always *correct*, and it makes every
/// asset cost a round trip on every load to be told nothing changed. With the
/// bodies already down to nothing, those round trips became the whole cost: a
/// console boot is fifteen of them.
///
/// The two numbers split the files by how much a stale one matters.
///
/// - **`short`** — the documents and the code: HTML, JavaScript, CSS. These are
///   edited during a session, and they are served from disk precisely so a
///   refresh shows the change. A minute is short enough that no edit is lost
///   for long and long enough that clicking around costs nothing.
/// - **`long`** — images, icons and fonts. They change rarely, and a stale one
///   is a slightly old picture rather than a page running code that disagrees
///   with the API it is talking to.
///
/// **The ETag still applies to both.** These say when to *ask*, not what the
/// answer is: once the age is up the request carries `If-None-Match` and an
/// unchanged file is still `304` with no body. See [`crate::asset`].
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cache {
    /// Seconds for HTML, JS and CSS.
    #[serde(default = "d_short")]
    pub short_secs: u32,
    /// Seconds for images and fonts.
    #[serde(default = "d_long")]
    pub long_secs: u32,
}

pub(super) fn d_short() -> u32 {
    60
}

pub(super) fn d_long() -> u32 {
    900
}

impl Default for Cache {
    fn default() -> Self {
        Self {
            short_secs: d_short(),
            long_secs: d_long(),
        }
    }
}

impl Default for Server {
    fn default() -> Self {
        Self {
            bind: d_bind(),
            cache: Cache::default(),
            backoff: Backoff::default(),
            connect_timeout_ms: d_connect_timeout_ms(),
            tls: None,
            tls_file: None,
            trusted_backends: Vec::new(),
            redirect_http: false,
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

pub(super) fn d_backoff_initial() -> u64 {
    250
}

pub(super) fn d_backoff_max() -> u64 {
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
