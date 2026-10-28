//! `/oauth/token`: a code, or a refresh token, exchanged for a token pair
//! (`docs/claude_zend_gateway.md` §12).
//!
//! Form-urlencoded, as RFC 6749 and claude.ai require. Public clients (CIMD,
//! `none` authentication) prove themselves with the PKCE verifier on the code
//! exchange, and by holding the refresh token after it. A registered client
//! with a secret (`super::registered`) must also present that secret, in the
//! form or a Basic header, on every exchange — before its code or refresh
//! token is read. Every exchange issues a fresh pair.

use std::collections::HashMap;

use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Json, Response};
use serde_json::json;

use super::client;
use super::grants::{self, Code, Grant, Kind, ACCESS_TTL_SECS, REFRESH_TTL_SECS};
use super::metadata::SCOPE;
use super::registered;
use crate::auth::oidc::pkce_challenge;
use crate::auth::session::Key;
use crate::config::McpClient;

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

/// Exchange `form` for a token pair. `basic` is what an `Authorization: Basic`
/// header carried ([`registered::basic`]).
pub fn exchange(
    key: &Key,
    clients: &[McpClient],
    form: &HashMap<String, String>,
    basic: Option<Result<(String, String), String>>,
    now: u64,
) -> Result<Issued, Refused> {
    let get = |k: &str| form.get(k).map(String::as_str).filter(|v| !v.is_empty());
    let basic = basic
        .transpose()
        .map_err(|why| refused("invalid_client", why))?;
    // One way to authenticate per request (RFC 6749 §2.3): a secret in both
    // places leaves it unclear which was checked.
    if basic.is_some() && get("client_secret").is_some() {
        return Err(refused(
            "invalid_request",
            "a client authenticates one way: the Authorization header or the form, not both",
        ));
    }
    let (client_id, secret) = match (&basic, get("client_id")) {
        (Some((id, _)), Some(named)) if named != id => {
            return Err(refused(
                "invalid_request",
                "the form names another client than the Authorization header",
            ))
        }
        (Some((id, secret)), _) => (id.as_str(), Some(secret.as_str())),
        (None, Some(id)) => (id, get("client_secret")),
        (None, None) => return Err(refused("invalid_request", "`client_id` is missing")),
    };
    // A plain id is a client only while it is registered: one taken off the
    // list is refused here, so its refresh tokens stop with it, rather than
    // being read as a public client.
    match registered::find(clients, client_id) {
        Some(client) if !registered::authenticates(client, secret) => {
            return Err(refused(
                "invalid_client",
                "the client's secret is missing or wrong",
            ))
        }
        Some(_) => {}
        None => {
            client::check_id(client_id).map_err(|why| refused("invalid_client", why))?;
        }
    }
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
    Ok(issue(key, &grant, now))
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
        // A client that failed to authenticate is answered 401 with the
        // scheme it may authenticate by (RFC 6749 §5.2); every other refusal
        // is 400.
        Err(r) if r.error == "invalid_client" => (
            StatusCode::UNAUTHORIZED,
            no_store,
            [(header::WWW_AUTHENTICATE, "Basic")],
            Json(json!({"error": r.error, "error_description": r.description})),
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
    use crate::auth::mcp::grants::testing::{grant, key};
    use crate::auth::mcp::grants::CODE_TTL_SECS;
    use crate::auth::mcp::registered::testing::{grok, GROK_ID, GROK_REDIRECT, GROK_SECRET};

    const VERIFIER: &str = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";

    fn form(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    /// An exchange by a metadata-document client: none registered, no header.
    fn public(key: &Key, form: &HashMap<String, String>, now: u64) -> Result<Issued, Refused> {
        exchange(key, &[], form, None, now)
    }

    fn code(now: u64) -> String {
        let c = Code {
            grant: grant(),
            challenge: pkce_challenge(VERIFIER),
        };
        grants::seal(&key(), Kind::McpCode, now + CODE_TTL_SECS, &c)
    }

    /// A code issued to the registered Grok client, and the form exchanging
    /// it with `secret` in the form.
    fn grok_code(now: u64) -> String {
        let c = Code {
            grant: Grant {
                client_id: GROK_ID.into(),
                redirect_uri: GROK_REDIRECT.into(),
                ..grant()
            },
            challenge: pkce_challenge(VERIFIER),
        };
        grants::seal(&key(), Kind::McpCode, now + CODE_TTL_SECS, &c)
    }

    fn grok_form(code: &str, secret: Option<&str>) -> HashMap<String, String> {
        let mut f = form(&[
            ("grant_type", "authorization_code"),
            ("code", code),
            ("code_verifier", VERIFIER),
            ("client_id", GROK_ID),
            ("redirect_uri", GROK_REDIRECT),
        ]);
        if let Some(s) = secret {
            f.insert("client_secret".into(), s.into());
        }
        f
    }

    /// **A registered client with a secret buys a pair only with it**, in the
    /// form or in a Basic header — and never on the code alone.
    #[test]
    fn a_registered_client_presents_its_secret_in_the_form_or_a_basic_header() {
        let clients = [grok()];
        let code = grok_code(100);
        let by_form = exchange(
            &key(),
            &clients,
            &grok_form(&code, Some(GROK_SECRET)),
            None,
            101,
        );
        assert!(by_form.is_ok());

        let mut bare = grok_form(&code, None);
        bare.remove("client_id");
        let header = Some(Ok((GROK_ID.to_string(), GROK_SECRET.to_string())));
        assert!(exchange(&key(), &clients, &bare, header, 101).is_ok());

        for (f, basic) in [
            (grok_form(&code, None), None),
            (grok_form(&code, Some("wrong")), None),
            (
                bare.clone(),
                Some(Ok((GROK_ID.to_string(), "wrong".to_string()))),
            ),
            (bare.clone(), Some(Err("unreadable".to_string()))),
        ] {
            let err = exchange(&key(), &clients, &f, basic, 101).unwrap_err();
            assert_eq!(err.error, "invalid_client");
        }

        let other = Some(Ok(("zen-other".to_string(), GROK_SECRET.to_string())));
        let err = exchange(&key(), &clients, &grok_form(&code, None), other, 101).unwrap_err();
        assert_eq!(
            err.error, "invalid_request",
            "the header and the form disagree"
        );

        let both = Some(Ok((GROK_ID.to_string(), GROK_SECRET.to_string())));
        let err = exchange(
            &key(),
            &clients,
            &grok_form(&code, Some(GROK_SECRET)),
            both,
            101,
        )
        .unwrap_err();
        assert_eq!(err.error, "invalid_request", "one method at a time");
    }

    /// **A client taken off the list is refused**, refresh token and all —
    /// a plain id is not read as a public client once it is not registered.
    #[test]
    fn a_deregistered_client_is_refused() {
        let first = exchange(
            &key(),
            &[grok()],
            &grok_form(&grok_code(100), Some(GROK_SECRET)),
            None,
            101,
        )
        .unwrap();
        let f = form(&[
            ("grant_type", "refresh_token"),
            ("refresh_token", &first.refresh_token),
            ("client_id", GROK_ID),
        ]);
        let err = exchange(&key(), &[], &f, None, 200).unwrap_err();
        assert_eq!(err.error, "invalid_client");
    }

    /// **A refresh needs the secret too**: the refresh token alone is not the
    /// client.
    #[test]
    fn a_registered_clients_refresh_needs_its_secret() {
        let clients = [grok()];
        let first = exchange(
            &key(),
            &clients,
            &grok_form(&grok_code(100), Some(GROK_SECRET)),
            None,
            101,
        )
        .unwrap();
        let refresh = |secret: Option<&str>| {
            let mut f = form(&[
                ("grant_type", "refresh_token"),
                ("refresh_token", &first.refresh_token),
                ("client_id", GROK_ID),
            ]);
            if let Some(s) = secret {
                f.insert("client_secret".into(), s.into());
            }
            f
        };
        assert!(exchange(&key(), &clients, &refresh(Some(GROK_SECRET)), None, 200).is_ok());
        assert_eq!(
            exchange(&key(), &clients, &refresh(None), None, 200)
                .unwrap_err()
                .error,
            "invalid_client"
        );
    }

    /// **A client that failed to authenticate is told so with 401** and the
    /// scheme to use; any other refusal is 400.
    #[test]
    fn invalid_client_is_401_with_its_scheme() {
        let res = answer(Err(refused("invalid_client", "no")));
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(res.headers()[header::WWW_AUTHENTICATE], "Basic");
        assert_eq!(res.headers()[header::CACHE_CONTROL], "no-store");
        let res = answer(Err(refused("invalid_grant", "no")));
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
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
        let issued = public(&key(), &by_code(&code(100), VERIFIER), 101).unwrap();
        let access: Grant =
            grants::open(&key(), Kind::McpAccess, &issued.access_token, 102).unwrap();
        assert_eq!(access, grant());
        assert_eq!(issued.expires_in, ACCESS_TTL_SECS);
        let late = public(&key(), &by_code(&code(100), VERIFIER), 100 + CODE_TTL_SECS);
        assert_eq!(late.unwrap_err().error, "invalid_grant");
    }

    #[test]
    fn a_wrong_verifier_client_or_redirect_is_invalid_grant() {
        let k = key();
        assert_eq!(
            public(&k, &by_code(&code(100), "wrong"), 101)
                .unwrap_err()
                .error,
            "invalid_grant"
        );
        let mut f = by_code(&code(100), VERIFIER);
        f.insert("client_id".into(), "https://evil.example/doc".into());
        assert_eq!(public(&k, &f, 101).unwrap_err().error, "invalid_grant");
        let mut f = by_code(&code(100), VERIFIER);
        f.insert("redirect_uri".into(), "https://evil.example/cb".into());
        assert_eq!(public(&k, &f, 101).unwrap_err().error, "invalid_grant");
    }

    #[test]
    fn a_refresh_token_buys_a_new_pair_and_an_access_token_is_not_one() {
        let first = public(&key(), &by_code(&code(100), VERIFIER), 101).unwrap();
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
        let second = public(&key(), &refresh(&first.refresh_token), 200).unwrap();
        assert_ne!(second.access_token, first.access_token);
        assert_eq!(
            public(&key(), &refresh(&first.access_token), 200)
                .unwrap_err()
                .error,
            "invalid_grant"
        );
    }

    #[test]
    fn an_unknown_grant_type_and_another_resource_are_named() {
        let f = form(&[
            ("grant_type", "client_credentials"),
            (
                "client_id",
                "https://claude.ai/oauth/mcp-oauth-client-metadata",
            ),
        ]);
        assert_eq!(
            public(&key(), &f, 1).unwrap_err().error,
            "unsupported_grant_type"
        );
        let mut f = by_code(&code(100), VERIFIER);
        f.insert("resource".into(), "https://evil.example/mcp".into());
        assert_eq!(public(&key(), &f, 101).unwrap_err().error, "invalid_target");
    }
}
