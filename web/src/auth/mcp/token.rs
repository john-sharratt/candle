//! `/oauth/token`: a code, or a refresh token, exchanged for a token pair
//! (`docs/claude_zend_gateway.md` §12).
//!
//! Form-urlencoded, as RFC 6749 and claude.ai require. Public clients (CIMD,
//! `none` authentication) prove themselves with the PKCE verifier on the code
//! exchange, and by holding the refresh token after it. Every exchange issues
//! a fresh pair, and only while the role table still names the person the
//! grant is for: taken off it, their refresh token buys nothing, so access
//! ends within an access token's hour (`docs/guest_access.md` §6). Each
//! exchange logs its outcome on `web::auth::mcp`, never a token.

use std::collections::HashMap;

use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Json, Response};
use serde_json::json;

use super::authorize::may_connect;
use super::grants::{self, Code, Grant, Kind, ACCESS_TTL_SECS, REFRESH_TTL_SECS};
use super::metadata::SCOPE;
use crate::auth::oidc::pkce_challenge;
use crate::auth::session::Key;
use crate::auth::Roles;

/// A token pair as the token endpoint answers it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Issued {
    pub access_token: String,
    pub refresh_token: String,
    pub expires_in: u64,
}

/// Why an exchange was refused, as RFC 6749 §5.2 names it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refused {
    pub error: &'static str,
    pub description: String,
}

fn refused(error: &'static str, description: impl Into<String>) -> Refused {
    Refused {
        error,
        description: description.into(),
    }
}

/// A fresh access and refresh token for `grant`.
pub fn issue(key: &Key, grant: &Grant, now: u64) -> Issued {
    Issued {
        access_token: grants::seal(key, Kind::McpAccess, now + ACCESS_TTL_SECS, grant),
        refresh_token: grants::seal(key, Kind::McpRefresh, now + REFRESH_TTL_SECS, grant),
        expires_in: ACCESS_TTL_SECS,
    }
}

/// Exchange `form` for a token pair, logging the outcome.
pub fn exchange(
    key: &Key,
    roles: &Roles,
    form: &HashMap<String, String>,
    now: u64,
) -> Result<Issued, Refused> {
    let grant_type = form.get("grant_type").map_or("", String::as_str);
    let client = form.get("client_id").map_or("", String::as_str);
    let out = exchange_unlogged(key, roles, form, now);
    match &out {
        Ok((_, grant)) => tracing::info!(
            target: "web::auth::mcp",
            email = %grant.email,
            client,
            grant_type,
            "token issued"
        ),
        Err(r) => tracing::info!(
            target: "web::auth::mcp",
            client,
            grant_type,
            error = r.error,
            detail = %r.description,
            "token refused"
        ),
    }
    out.map(|(issued, _)| issued)
}

/// [`exchange`], answering the grant the pair was issued on beside it.
fn exchange_unlogged(
    key: &Key,
    roles: &Roles,
    form: &HashMap<String, String>,
    now: u64,
) -> Result<(Issued, Grant), Refused> {
    let get = |k: &str| form.get(k).map(String::as_str).filter(|v| !v.is_empty());
    let client_id =
        get("client_id").ok_or_else(|| refused("invalid_request", "`client_id` is missing"))?;
    let grant: Grant = match get("grant_type") {
        Some("authorization_code") => {
            let code =
                get("code").ok_or_else(|| refused("invalid_request", "`code` is missing"))?;
            let verifier = get("code_verifier")
                .ok_or_else(|| refused("invalid_request", "`code_verifier` is missing"))?;
            let redirect = get("redirect_uri")
                .ok_or_else(|| refused("invalid_request", "`redirect_uri` is missing"))?;
            let held: Code = grants::open(key, Kind::McpCode, code, now)
                .ok_or_else(|| refused("invalid_grant", "the code is unknown or expired"))?;
            if held.grant.client_id != client_id {
                return Err(refused(
                    "invalid_grant",
                    "the code was issued to another client",
                ));
            }
            if held.grant.redirect_uri != redirect {
                return Err(refused(
                    "invalid_grant",
                    "the code was issued for another redirect",
                ));
            }
            if pkce_challenge(verifier) != held.challenge {
                return Err(refused("invalid_grant", "the code verifier does not match"));
            }
            held.grant
        }
        Some("refresh_token") => {
            let token = get("refresh_token")
                .ok_or_else(|| refused("invalid_request", "`refresh_token` is missing"))?;
            let held: Grant = grants::open(key, Kind::McpRefresh, token, now).ok_or_else(|| {
                refused("invalid_grant", "the refresh token is unknown or expired")
            })?;
            if held.client_id != client_id {
                return Err(refused(
                    "invalid_grant",
                    "the refresh token was issued to another client",
                ));
            }
            held
        }
        Some(other) => {
            return Err(refused(
                "unsupported_grant_type",
                format!("`{other}` is not a grant this server issues"),
            ))
        }
        None => return Err(refused("invalid_request", "`grant_type` is missing")),
    };
    if let Some(resource) = get("resource") {
        if resource != grant.resource {
            return Err(refused(
                "invalid_target",
                "the token was granted for another resource",
            ));
        }
    }
    if !may_connect(roles, &grant.identity(now)) {
        return Err(refused(
            "invalid_grant",
            "this account is no longer on the list of people who may use Zen",
        ));
    }
    Ok((issue(key, &grant, now), grant))
}

/// The token endpoint's answer to `result`.
pub fn answer(result: Result<Issued, Refused>) -> Response {
    let no_store = [
        (header::CACHE_CONTROL, "no-store"),
        (header::PRAGMA, "no-cache"),
    ];
    match result {
        Ok(issued) => (
            no_store,
            Json(json!({
                "access_token": issued.access_token,
                "token_type": "Bearer",
                "expires_in": issued.expires_in,
                "refresh_token": issued.refresh_token,
                "scope": SCOPE,
            })),
        )
            .into_response(),
        Err(r) => (
            StatusCode::BAD_REQUEST,
            no_store,
            Json(json!({"error": r.error, "error_description": r.description})),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::mcp::grants::testing::{grant, key, listed};
    use crate::auth::mcp::grants::CODE_TTL_SECS;

    const VERIFIER: &str = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";

    fn form(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    /// **Taken off the role table, a person's grant buys nothing**: neither
    /// the code nor the refresh token issued while they were on it.
    #[test]
    fn a_person_off_the_table_is_refused_a_pair() {
        let first = exchange(&key(), &listed(), &by_code(&code(100), VERIFIER), 101).unwrap();
        let gone = Roles::default();
        let err = exchange(&key(), &gone, &by_code(&code(100), VERIFIER), 101).unwrap_err();
        assert_eq!(err.error, "invalid_grant");
        let refresh = form(&[
            ("grant_type", "refresh_token"),
            ("refresh_token", &first.refresh_token),
            (
                "client_id",
                "https://claude.ai/oauth/mcp-oauth-client-metadata",
            ),
        ]);
        assert!(exchange(&key(), &listed(), &refresh, 200).is_ok());
        let err = exchange(&key(), &gone, &refresh, 200).unwrap_err();
        assert_eq!(err.error, "invalid_grant");
    }

    fn code(now: u64) -> String {
        let c = Code {
            grant: grant(),
            challenge: pkce_challenge(VERIFIER),
        };
        grants::seal(&key(), Kind::McpCode, now + CODE_TTL_SECS, &c)
    }

    fn by_code(code: &str, verifier: &str) -> HashMap<String, String> {
        form(&[
            ("grant_type", "authorization_code"),
            ("code", code),
            ("code_verifier", verifier),
            (
                "client_id",
                "https://claude.ai/oauth/mcp-oauth-client-metadata",
            ),
            ("redirect_uri", "https://claude.ai/api/mcp/auth_callback"),
        ])
    }

    /// The RFC 7636 appendix B pair, so the challenge this server computes is
    /// the one every client computes.
    #[test]
    fn the_pkce_challenge_is_rfc_7636s() {
        assert_eq!(
            pkce_challenge(VERIFIER),
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        );
    }

    #[test]
    fn a_code_and_its_verifier_buy_a_pair_that_opens_as_the_grant() {
        let issued = exchange(&key(), &listed(), &by_code(&code(100), VERIFIER), 101).unwrap();
        let access: Grant =
            grants::open(&key(), Kind::McpAccess, &issued.access_token, 102).unwrap();
        assert_eq!(access, grant());
        assert_eq!(issued.expires_in, ACCESS_TTL_SECS);
        let late = exchange(
            &key(),
            &listed(),
            &by_code(&code(100), VERIFIER),
            100 + CODE_TTL_SECS,
        );
        assert_eq!(late.unwrap_err().error, "invalid_grant");
    }

    #[test]
    fn a_wrong_verifier_client_or_redirect_is_invalid_grant() {
        let k = key();
        assert_eq!(
            exchange(&k, &listed(), &by_code(&code(100), "wrong"), 101)
                .unwrap_err()
                .error,
            "invalid_grant"
        );
        let mut f = by_code(&code(100), VERIFIER);
        f.insert("client_id".into(), "https://evil.example/doc".into());
        assert_eq!(
            exchange(&k, &listed(), &f, 101).unwrap_err().error,
            "invalid_grant"
        );
        let mut f = by_code(&code(100), VERIFIER);
        f.insert("redirect_uri".into(), "https://evil.example/cb".into());
        assert_eq!(
            exchange(&k, &listed(), &f, 101).unwrap_err().error,
            "invalid_grant"
        );
    }

    #[test]
    fn a_refresh_token_buys_a_new_pair_and_an_access_token_is_not_one() {
        let first = exchange(&key(), &listed(), &by_code(&code(100), VERIFIER), 101).unwrap();
        let refresh = |token: &str| {
            form(&[
                ("grant_type", "refresh_token"),
                ("refresh_token", token),
                (
                    "client_id",
                    "https://claude.ai/oauth/mcp-oauth-client-metadata",
                ),
            ])
        };
        let second = exchange(&key(), &listed(), &refresh(&first.refresh_token), 200).unwrap();
        assert_ne!(second.access_token, first.access_token);
        assert_eq!(
            exchange(&key(), &listed(), &refresh(&first.access_token), 200)
                .unwrap_err()
                .error,
            "invalid_grant"
        );
    }

    #[test]
    fn an_unknown_grant_type_and_another_resource_are_named() {
        let f = form(&[("grant_type", "client_credentials"), ("client_id", "x")]);
        assert_eq!(
            exchange(&key(), &listed(), &f, 1).unwrap_err().error,
            "unsupported_grant_type"
        );
        let mut f = by_code(&code(100), VERIFIER);
        f.insert("resource".into(), "https://evil.example/mcp".into());
        assert_eq!(
            exchange(&key(), &listed(), &f, 101).unwrap_err().error,
            "invalid_target"
        );
    }
}
