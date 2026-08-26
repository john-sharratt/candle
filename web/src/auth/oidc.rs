//! The OpenID Connect authorization-code exchange, with PKCE.
//!
//! Two notes on what is *not* here.
//!
//! **No JWKS fetch, no RSA verification.** The `id_token` is read from the
//! token endpoint's response over a TLS connection this process opened to the
//! configured host, using a client secret only this process holds. OIDC Core
//! §3.1.3.7 says a token obtained that way may be used without validating its
//! signature, and every alternative means shipping a JWKS cache, a key-rotation
//! story and an RSA implementation to re-derive a fact TLS already established.
//!
//! **No server-side state between the two requests.** The `state`, `nonce` and
//! PKCE verifier live in a short-lived signed cookie, so a redirect that comes
//! back to a different process in a restarted gateway still completes.

use serde::{Deserialize, Serialize};

use super::session::{self, Key};
use crate::config::Provider;

/// What the login redirect stashes for the callback to check.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Pending {
    pub state: String,
    pub nonce: String,
    pub verifier: String,
    /// Where to send the browser afterwards. Validated against the cookie
    /// domain before use — an unchecked `next` is an open redirect.
    pub next: String,
    pub exp: u64,
}

/// Claims we read out of the `id_token`. Everything else Google sends is
/// ignored rather than modelled: fields we do not use cannot go stale.
#[derive(Debug, Clone, Deserialize)]
pub struct Claims {
    pub sub: String,
    #[serde(default)]
    pub email: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub picture: String,
    #[serde(default)]
    pub nonce: String,
}

pub struct Start {
    pub redirect_to: String,
    pub pending: Pending,
}

/// Build the provider redirect and the state that has to survive until the
/// callback.
pub fn start(p: &Provider, next: &str, ttl_secs: u64) -> Start {
    let state = random_token();
    let nonce = random_token();
    let verifier = random_token();
    let challenge = pkce_challenge(&verifier);

    let redirect_to = format!(
        "{}?{}",
        p.auth_endpoint,
        query(&[
            ("client_id", &p.client_id),
            ("redirect_uri", &p.redirect_uri),
            ("response_type", "code"),
            ("scope", "openid email profile"),
            ("state", &state),
            ("nonce", &nonce),
            ("code_challenge", &challenge),
            ("code_challenge_method", "S256"),
            // Without this a returning user is bounced straight back with no
            // way to pick a different account, which reads as a broken button.
            ("prompt", "select_account"),
        ])
    );

    Start {
        redirect_to,
        pending: Pending {
            state,
            nonce,
            verifier,
            next: next.to_string(),
            exp: session::now_secs() + ttl_secs,
        },
    }
}

#[derive(Debug, Deserialize)]
struct TokenResponse {
    id_token: String,
}

/// Exchange the code for an `id_token` and read its claims.
pub async fn exchange(
    http: &reqwest::Client,
    p: &Provider,
    secret: &str,
    code: &str,
    verifier: &str,
) -> anyhow::Result<Claims> {
    let res = http
        .post(&p.token_endpoint)
        .form(&[
            ("code", code),
            ("client_id", p.client_id.as_str()),
            ("client_secret", secret),
            ("redirect_uri", p.redirect_uri.as_str()),
            ("grant_type", "authorization_code"),
            ("code_verifier", verifier),
        ])
        .send()
        .await?;

    let status = res.status();
    let body = res.text().await?;
    if !status.is_success() {
        // The provider's own message is the only useful diagnostic here —
        // `redirect_uri_mismatch` and `invalid_client` are configuration
        // mistakes that are otherwise invisible.
        anyhow::bail!("token endpoint returned {status}: {body}");
    }

    let token: TokenResponse =
        serde_json::from_str(&body).map_err(|e| anyhow::anyhow!("token response: {e}: {body}"))?;
    claims_of(&token.id_token)
}

/// Decode a JWT's payload. See the module note: the signature is not checked
/// because TLS to the token endpoint already established provenance.
pub fn claims_of(id_token: &str) -> anyhow::Result<Claims> {
    use base64::engine::general_purpose::URL_SAFE_NO_PAD as B64;
    use base64::Engine;

    let mut parts = id_token.split('.');
    let (_header, payload) = (
        parts
            .next()
            .ok_or_else(|| anyhow::anyhow!("id_token is empty"))?,
        parts
            .next()
            .ok_or_else(|| anyhow::anyhow!("id_token has no payload segment"))?,
    );
    let json = B64.decode(payload)?;
    Ok(serde_json::from_slice(&json)?)
}

fn pkce_challenge(verifier: &str) -> String {
    use base64::engine::general_purpose::URL_SAFE_NO_PAD as B64;
    use base64::Engine;
    use sha2::{Digest, Sha256};

    B64.encode(Sha256::digest(verifier.as_bytes()))
}

/// 256 bits from the OS. `state`, `nonce` and the PKCE verifier are all
/// unguessability, so none of them may come from a seeded generator.
fn random_token() -> String {
    use base64::engine::general_purpose::URL_SAFE_NO_PAD as B64;
    use base64::Engine;
    use rand::TryRngCore;

    let mut bytes = [0u8; 32];
    // Straight from the OS, and a hard failure if it will not answer — a
    // predictable `state` or PKCE verifier defeats the whole exchange, so
    // there is no degraded mode worth having here.
    rand::rngs::OsRng
        .try_fill_bytes(&mut bytes)
        .expect("the OS must provide entropy");
    B64.encode(bytes)
}

fn query(pairs: &[(&str, &str)]) -> String {
    pairs
        .iter()
        .map(|(k, v)| format!("{}={}", urlencode(k), urlencode(v)))
        .collect::<Vec<_>>()
        .join("&")
}

/// Percent-encoding for a query value: everything outside the unreserved set,
/// which is stricter than necessary and therefore never wrong.
pub fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(*b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Is `next` somewhere we are willing to send a browser after sign-in?
///
/// Only this estate. An unchecked `next` turns the login endpoint into an open
/// redirect, which is how a phishing link gets to wear the real domain.
pub fn safe_next(next: &str, cookie_domain: &str) -> bool {
    if next.starts_with('/') && !next.starts_with("//") {
        return true; // same-origin path
    }
    let Some(rest) = next
        .strip_prefix("https://")
        .or_else(|| next.strip_prefix("http://"))
    else {
        return false;
    };
    let host = rest
        .split(['/', '?', '#'])
        .next()
        .unwrap_or("")
        .rsplit_once(':')
        .map(|(h, _)| h)
        .unwrap_or_else(|| rest.split(['/', '?', '#']).next().unwrap_or(""))
        .to_ascii_lowercase();

    let apex = cookie_domain.trim_start_matches('.').to_ascii_lowercase();
    host == apex || host.ends_with(&format!(".{apex}"))
}

/// The signed cookie value carrying [`Pending`] between the two requests.
pub fn seal(key: &Key, pending: &Pending) -> String {
    session::sign(key, pending)
}

pub fn unseal(key: &Key, token: &str) -> anyhow::Result<Pending> {
    let p: Pending = session::verify(key, token).map_err(|e| anyhow::anyhow!("{e}"))?;
    if session::now_secs() >= p.exp {
        anyhow::bail!("sign-in took too long — start again");
    }
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider() -> Provider {
        Provider {
            client_id: "cid".into(),
            client_secret_file: "unused".into(),
            redirect_uri: "https://tokera.com/auth/callback".into(),
            auth_endpoint: "https://accounts.google.com/o/oauth2/v2/auth".into(),
            token_endpoint: "https://oauth2.googleapis.com/token".into(),
        }
    }

    #[test]
    fn the_redirect_carries_everything_the_provider_needs() {
        let s = start(&provider(), "/", 600);
        for expect in [
            "client_id=cid",
            "response_type=code",
            "code_challenge_method=S256",
            "scope=openid%20email%20profile",
            "redirect_uri=https%3A%2F%2Ftokera.com%2Fauth%2Fcallback",
        ] {
            assert!(
                s.redirect_to.contains(expect),
                "missing {expect} in {}",
                s.redirect_to
            );
        }
        assert!(s
            .redirect_to
            .contains(&format!("state={}", urlencode(&s.pending.state))));
    }

    #[test]
    fn every_start_is_unguessable_and_different() {
        let a = start(&provider(), "/", 600).pending;
        let b = start(&provider(), "/", 600).pending;
        assert_ne!(a.state, b.state);
        assert_ne!(a.nonce, b.nonce);
        assert_ne!(a.verifier, b.verifier);
        assert!(a.state.len() >= 43, "256 bits base64url is 43 chars");
    }

    #[test]
    fn the_pkce_challenge_is_the_sha256_of_the_verifier() {
        // RFC 7636 A.1's test vector, so this is checked against the spec
        // rather than against itself.
        assert_eq!(
            pkce_challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"),
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        );
    }

    #[test]
    fn next_is_confined_to_the_estate() {
        let d = ".tokera.com";
        assert!(safe_next("/papers", d));
        assert!(safe_next("https://tokera.com/x", d));
        assert!(safe_next("https://bot.tokera.com/", d));
        assert!(safe_next("https://code.tokera.com:8443/s/1", d));

        // The ones that matter.
        assert!(!safe_next("https://evil.com/", d));
        assert!(!safe_next("//evil.com/", d));
        assert!(!safe_next("https://tokera.com.evil.com/", d));
        assert!(!safe_next("https://nottokera.com/", d));
        assert!(!safe_next("javascript:alert(1)", d));
    }

    #[test]
    fn claims_decode_without_needing_the_signature() {
        // {"sub":"42","email":"a@b.c","name":"A"} — a real id_token shape.
        let payload = "eyJzdWIiOiI0MiIsImVtYWlsIjoiYUBiLmMiLCJuYW1lIjoiQSJ9";
        let c = claims_of(&format!("header.{payload}.signature")).unwrap();
        assert_eq!(c.sub, "42");
        assert_eq!(c.email, "a@b.c");
    }

    #[test]
    fn a_malformed_id_token_is_an_error() {
        assert!(claims_of("").is_err());
        assert!(claims_of("only-one-part").is_err());
        assert!(claims_of("a.!!!.c").is_err());
    }
}
