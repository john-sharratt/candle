//! Sign-in for the whole estate, owned by the gateway.
//!
//! One account reaches `tokera.com`, `code.tokera.com` and `bot.tokera.com`,
//! and the mechanism is deliberately boring: the gateway runs the OIDC dance
//! and issues a session cookie on the **parent domain**, so the browser
//! presents it to each subsite without any of them being involved. No
//! cross-origin handshake, no token in a URL fragment, no second login page.
//!
//! The daemons therefore never authenticate anyone. They read the identity the
//! gateway states, on headers the gateway sets and strips
//! ([`crate::proxy`]) — which is safe exactly as far as the daemons are
//! unreachable except through the gateway. That is the trust boundary, it is
//! the reason they listen on private addresses, and a daemon that wants proof
//! rather than assertion can verify `X-Tokera-Assertion` with the same session
//! key.
//!
//! Routes, mounted ahead of site routing so a site that proxies `/` cannot
//! swallow them:
//!
//! ```text
//! GET  /auth/login?next=…   → the provider
//! GET  /auth/callback       → sets the session cookie, returns to `next`
//! POST /auth/logout         → clears it
//! GET  /auth/me             → who the browser is, or that nobody is
//! ```

use std::sync::Arc;

use axum::extract::{Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;

pub mod cookie;
pub mod oidc;
pub mod session;

pub use session::Identity;

use crate::config::Auth as AuthConfig;
use session::Key;

/// How long a browser has to complete the round trip to the provider.
const PENDING_TTL_SECS: u64 = 15 * 60;

/// Everything the auth routes need, resolved once at startup.
pub struct Auth {
    cfg: AuthConfig,
    key: Key,
    client_secret: String,
    /// Derived from the redirect URI's scheme — see [`cookie::set`].
    secure: bool,
    http: reqwest::Client,
}

impl Auth {
    /// Read the secrets off disk and assemble.
    ///
    /// Fails loudly, but only ever reached when a config *has* an `auth:`
    /// block — sign-in is opt-in precisely so that a public site is never held
    /// hostage to a key it does not need. Once opted in, a gateway that starts
    /// with silently broken sign-in is the worse outcome, so this is fatal.
    pub fn new(cfg: AuthConfig) -> anyhow::Result<Self> {
        let raw = std::fs::read(&cfg.session_secret_file).map_err(|e| {
            anyhow::anyhow!(
                "reading {}: {e}\n\
                 \n\
                 This file signs sign-in sessions. Either create it —\n\
                     32+ random bytes, e.g. `head -c 48 /dev/urandom | base64 > {0}`\n\
                 — or comment out the `auth:` block to run without sign-in; the\n\
                 public pages do not need it.",
                cfg.session_secret_file.display()
            )
        })?;
        let key = Key::new(trim(raw))?;

        let secret = std::fs::read(&cfg.google.client_secret_file).map_err(|e| {
            anyhow::anyhow!(
                "reading {}: {e}\n\
                 \n\
                 This is the OAuth client secret from the Google Cloud console\n\
                 (APIs & Services → Credentials → OAuth 2.0 Client ID). Comment\n\
                 out the `auth:` block to run without sign-in.",
                cfg.google.client_secret_file.display()
            )
        })?;
        let client_secret = String::from_utf8(trim(secret))?;
        if client_secret.is_empty() {
            anyhow::bail!("{} is empty", cfg.google.client_secret_file.display());
        }

        let secure = cfg.google.redirect_uri.starts_with("https://");
        Ok(Self {
            cfg,
            key,
            client_secret,
            secure,
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(15))
                .build()?,
        })
    }

    /// The identity this request carries, if any. Silent on a bad or expired
    /// cookie: to a caller "not signed in" and "signed in with a stale
    /// session" are the same state, and both mean show the sign-in button.
    pub fn identity(&self, headers: &HeaderMap) -> Option<Identity> {
        let raw = cookie::get(headers, cookie::SESSION)?;
        match session::open(&self.key, raw, session::now_secs()) {
            Ok(id) => Some(id),
            Err(e) => {
                tracing::debug!(error = %e, "ignoring session cookie");
                None
            }
        }
    }

    /// The signed session token as presented, for forwarding to a daemon that
    /// wants to verify rather than trust.
    pub fn raw_token<'a>(&self, headers: &'a HeaderMap) -> Option<&'a str> {
        cookie::get(headers, cookie::SESSION)
    }

    pub fn router(self: Arc<Self>) -> Router {
        Router::new()
            .route("/auth/login", get(login))
            .route("/auth/callback", get(callback))
            .route("/auth/logout", post(logout).get(logout))
            .route("/auth/me", get(me))
            .with_state(self)
    }
}

/// A 303 that sets cookies, appending each one.
///
/// Built by hand because `[(SET_COOKIE, a), (SET_COOKIE, b)]` **replaces**
/// rather than appends — the second cookie silently discards the first, and the
/// callback sets two. The symptom is a sign-in that redirects correctly and
/// leaves the browser with no session.
fn see_other(location: &str, cookies: &[String]) -> Response {
    let mut b = Response::builder()
        .status(StatusCode::SEE_OTHER)
        .header(header::LOCATION, location);
    for c in cookies {
        b = b.header(header::SET_COOKIE, c);
    }
    b.body(axum::body::Body::empty())
        .expect("a redirect with valid header values")
}

fn trim(mut bytes: Vec<u8>) -> Vec<u8> {
    // A secret file almost always ends with the newline an editor added, and a
    // key that differs by one byte between two machines is a miserable bug.
    while matches!(bytes.last(), Some(b'\n' | b'\r' | b' ' | b'\t')) {
        bytes.pop();
    }
    bytes
}

#[derive(Deserialize)]
struct LoginQuery {
    #[serde(default)]
    next: Option<String>,
}

async fn login(State(auth): State<Arc<Auth>>, Query(q): Query<LoginQuery>) -> Response {
    let next = q.next.unwrap_or_else(|| "/".into());
    if !oidc::safe_next(&next, &auth.cfg.cookie_domain) {
        return problem(
            StatusCode::BAD_REQUEST,
            "bad_redirect",
            "that `next` is not somewhere this can send you after sign-in",
        );
    }

    let start = oidc::start(&auth.cfg.google, &next, PENDING_TTL_SECS);
    let sealed = oidc::seal(&auth.key, &start.pending);
    let jar = cookie::set(
        cookie::PENDING,
        &sealed,
        &auth.cfg.cookie_domain,
        PENDING_TTL_SECS as i64,
        auth.secure,
    );

    see_other(&start.redirect_to, &[jar])
}

#[derive(Deserialize)]
struct CallbackQuery {
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    state: Option<String>,
    /// The provider says why it refused — usually the user pressed cancel.
    #[serde(default)]
    error: Option<String>,
}

async fn callback(
    State(auth): State<Arc<Auth>>,
    headers: HeaderMap,
    Query(q): Query<CallbackQuery>,
) -> Response {
    let drop_pending = cookie::clear(cookie::PENDING, &auth.cfg.cookie_domain, auth.secure);

    if let Some(err) = q.error {
        // Cancelling is not a failure; put them back where they started.
        tracing::info!(error = %err, "sign-in was declined at the provider");
        return see_other("/", &[drop_pending]);
    }

    let Some(raw) = cookie::get(&headers, cookie::PENDING) else {
        return problem(
            StatusCode::BAD_REQUEST,
            "no_pending_signin",
            "no sign-in is in progress — it may have taken too long. Start again.",
        );
    };
    let pending = match oidc::unseal(&auth.key, raw) {
        Ok(p) => p,
        Err(e) => {
            return problem(
                StatusCode::BAD_REQUEST,
                "bad_pending_signin",
                &e.to_string(),
            )
        }
    };

    // The CSRF check. Without it, an attacker's code can be redeemed into the
    // victim's browser, signing them into the attacker's account.
    if q.state.as_deref() != Some(pending.state.as_str()) {
        return problem(
            StatusCode::BAD_REQUEST,
            "state_mismatch",
            "this sign-in did not start here",
        );
    }
    let Some(code) = q.code else {
        return problem(
            StatusCode::BAD_REQUEST,
            "no_code",
            "the provider returned no authorization code",
        );
    };

    let claims = match oidc::exchange(
        &auth.http,
        &auth.cfg.google,
        &auth.client_secret,
        &code,
        &pending.verifier,
    )
    .await
    {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(error = %e, "token exchange failed");
            return problem(
                StatusCode::BAD_GATEWAY,
                "exchange_failed",
                "the sign-in provider would not complete the exchange",
            );
        }
    };

    // Replay protection: the id_token must be the one minted for this attempt.
    if !claims.nonce.is_empty() && claims.nonce != pending.nonce {
        return problem(
            StatusCode::BAD_REQUEST,
            "nonce_mismatch",
            "this sign-in response belongs to a different attempt",
        );
    }

    let ttl = auth.cfg.session_ttl_hours * 3600;
    let identity = Identity {
        sub: claims.sub,
        email: claims.email,
        name: claims.name,
        picture: claims.picture,
        exp: session::now_secs() + ttl,
    };
    let token = session::sign(&auth.key, &identity);
    let jar = cookie::set(
        cookie::SESSION,
        &token,
        &auth.cfg.cookie_domain,
        ttl as i64,
        auth.secure,
    );

    tracing::info!(sub = %identity.sub, email = %identity.email, "signed in");
    see_other(&pending.next, &[jar, drop_pending])
}

async fn logout(State(auth): State<Arc<Auth>>, Query(q): Query<LoginQuery>) -> Response {
    let jar = cookie::clear(cookie::SESSION, &auth.cfg.cookie_domain, auth.secure);
    let next = q
        .next
        .filter(|n| oidc::safe_next(n, &auth.cfg.cookie_domain))
        .unwrap_or_else(|| "/".into());
    see_other(&next, &[jar])
}

async fn me(State(auth): State<Arc<Auth>>, headers: HeaderMap) -> Response {
    match auth.identity(&headers) {
        Some(id) => Json(json!({
            "authenticated": true,
            "sub": id.sub,
            "email": id.email,
            "name": id.name,
            "picture": id.picture,
            "expires_at": id.exp,
        }))
        .into_response(),
        None => Json(json!({ "authenticated": false })).into_response(),
    }
}

/// Sign-in errors answer in the same shape as every other API error, so a
/// client needs no special case for them.
fn problem(status: StatusCode, code: &str, detail: &str) -> Response {
    (
        status,
        Json(json!({ "error": code, "detail": detail, "field": null })),
    )
        .into_response()
}

/// What `/auth/me` says when no `auth:` block is configured — a distinct state
/// from "not signed in", because the UI should hide the button rather than
/// offer one that cannot work.
pub fn unconfigured_router() -> Router {
    Router::new().route(
        "/auth/me",
        get(|| async { Json(json!({ "authenticated": false, "configured": false })) }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_trailing_newline_in_a_secret_file_is_not_part_of_the_secret() {
        assert_eq!(trim(b"abc\n".to_vec()), b"abc");
        assert_eq!(trim(b"abc\r\n".to_vec()), b"abc");
        assert_eq!(trim(b"abc".to_vec()), b"abc");
    }
}
