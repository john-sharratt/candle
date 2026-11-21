//! The gateway's OpenID Connect sign-in, shared by every hostname it serves.
//!
//! The gateway is the only ingress, so it holds one session for every hostname.
//! The cookie is issued on the parent domain and the browser presents it to each
//! subdomain itself, and daemons behind the gateway read the identity it asserts
//! instead of running their own sign-in. Each estate signed into through the same
//! client has its own cookie domain and callback, and a request signs in on the
//! domain it arrived on, falling back to the first. The session key is read from
//! a file so the site table can be committed without a secret, and rotating the
//! key signs everyone out.

use super::*;

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
    /// Further estates signed in through the same client and the same session
    /// key — `yenbot.com.au` beside `tokera.com` — each with its own cookie
    /// domain and its own callback registered on the client. A request signs
    /// in on the domain it arrived on ([`Auth::domain_for`]).
    #[serde(default)]
    pub domains: Vec<CookieDomain>,
}

/// One estate a sign-in can belong to: the cookie's `Domain` and the callback
/// the provider returns to on it.
#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CookieDomain {
    pub cookie_domain: String,
    pub redirect_uri: String,
}

impl Auth {
    /// The estate a request to `host` signs in on: the one of
    /// [`domains`](Self::domains) whose cookie domain covers it, else the
    /// first — `cookie_domain` and the provider's `redirect_uri`.
    pub fn domain_for(&self, host: &str) -> CookieDomain {
        let host = host
            .rsplit_once(':')
            .map_or(host, |(h, _)| h)
            .to_ascii_lowercase();
        self.domains
            .iter()
            .find(|d| covers(&d.cookie_domain, &host))
            .cloned()
            .unwrap_or_else(|| CookieDomain {
                cookie_domain: self.cookie_domain.clone(),
                redirect_uri: self.google.redirect_uri.clone(),
            })
    }
}

/// Whether cookie domain `domain` reaches `host`.
pub fn covers(domain: &str, host: &str) -> bool {
    let apex = domain.trim_start_matches('.').to_ascii_lowercase();
    !apex.is_empty() && (host == apex || host.ends_with(&format!(".{apex}")))
}

pub(super) fn d_session_hours() -> u64 {
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

pub(super) fn d_google_auth() -> String {
    "https://accounts.google.com/o/oauth2/v2/auth".into()
}

pub(super) fn d_google_token() -> String {
    "https://oauth2.googleapis.com/token".into()
}
