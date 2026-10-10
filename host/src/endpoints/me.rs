//! `GET /v1/me` — who the caller is here, and what each tenant lets them do.
//!
//! `role` is always there; a signed-out caller is `200` with `role:
//! "unauthenticated"`, which a page reads to decide what to offer. The rest is
//! the tenants': each contributes top-level fields (npcd's account record,
//! created on first sight) and its own under `tenants.<name>` (zend's tool
//! modes). It is advice to the page, never the enforcement — every route
//! resolves the same role itself.

use std::sync::Arc;

use axum::extract::State;
use axum::http::HeaderMap;
use axum::Json;
use futures::future::join_all;
use serde_json::{json, Map, Value};
use web::auth::forwarded::identify;
use web::auth::session::Identity;
use web::auth::Role;

use super::Shared;
use crate::tenant::MeShare;

pub async fn me(State(s): State<Arc<Shared>>, headers: HeaderMap) -> Json<Value> {
    let identity = identify(&headers).ok();
    let role = s.roles.of(identity.as_ref());
    let shares = join_all(s.tenants.iter().map(|(_, r)| r.me(identity.as_ref(), role))).await;
    let named = s.tenants.iter().map(|(n, _)| *n).zip(shares).collect();
    Json(assemble(identity.as_ref(), role, named))
}

/// The body: the role, who the caller is, and every tenant's share.
///
/// One value per field, by a fixed precedence: the host's `role`, and the
/// signed-in identity's `email`, `display` and `avatar_url` where the gateway
/// sent them, come first; a tenant's field fills only what is still empty —
/// npcd's stored display name, when the gateway sent none — and between two
/// tenants the higher-ranked one's (`shares` is in rank order) stands.
/// `tenants` is the host's: each tenant's own share goes under its name.
fn assemble(identity: Option<&Identity>, role: Role, shares: Vec<(&str, MeShare)>) -> Value {
    let mut top = Map::new();
    top.insert("role".into(), json!(role));
    if let Some(id) = identity {
        for (k, v) in [
            ("email", &id.email),
            ("display", &id.name),
            ("avatar_url", &id.picture),
        ] {
            if !v.is_empty() {
                top.insert(k.into(), json!(v));
            }
        }
    }
    let mut tenants = Map::new();
    for (name, share) in shares {
        for (k, v) in share.top {
            if k != "tenants" && !top.contains_key(&k) {
                top.insert(k, v);
            }
        }
        tenants.insert(name.to_string(), share.own);
    }
    top.insert("tenants".into(), Value::Object(tenants));
    Value::Object(top)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> Identity {
        Identity {
            provider: "google".into(),
            sub: "g-1".into(),
            email: "wren@example.com".into(),
            name: "Wren".into(),
            picture: "https://example.com/w.png".into(),
            exp: 0,
        }
    }

    /// Signed out: the role and each tenant's share for nobody — no account.
    #[test]
    fn a_signed_out_caller_is_an_answer() {
        let body = assemble(
            None,
            Role::Unauthenticated,
            vec![(
                "zend",
                MeShare {
                    top: Map::new(),
                    own: json!({"tool_modes": ["restricted"], "default_tools": "restricted"}),
                },
            )],
        );
        assert_eq!(
            body,
            json!({
                "role": "unauthenticated",
                "tenants": {"zend": {"tool_modes": ["restricted"], "default_tools": "restricted"}},
            })
        );
    }

    /// The host's fields stand over a tenant's copy; a tenant fills what the
    /// gateway left empty; between tenants the higher-ranked one's stands; and
    /// `tenants` is never a tenant's to overwrite.
    #[test]
    fn one_value_per_field_by_precedence() {
        let mut nameless = identity();
        nameless.name = String::new();
        let share = |pairs: &[(&str, Value)]| MeShare {
            top: pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect(),
            own: Value::Null,
        };
        let body = assemble(
            Some(&nameless),
            Role::User,
            vec![
                (
                    "zend",
                    share(&[("theme", json!("zend's")), ("email", json!("x@y"))]),
                ),
                (
                    "npcd",
                    share(&[
                        ("display", json!("Stored Wren")),
                        ("theme", json!("npcd's")),
                        ("tenants", json!("clobber")),
                    ]),
                ),
            ],
        );
        assert_eq!(body["email"], "wren@example.com", "the gateway's");
        assert_eq!(
            body["display"], "Stored Wren",
            "filled where the gateway sent none"
        );
        assert_eq!(body["theme"], "zend's", "the higher-ranked tenant's");
        assert_eq!(body["tenants"], json!({"zend": null, "npcd": null}));
    }

    /// Signed in: npcd's account at the top, zend's modes under its name, the
    /// host's role over any tenant's copy.
    #[test]
    fn each_tenants_share_lands_where_it_belongs() {
        let mut account = Map::new();
        account.insert("user_id".into(), json!("u-7"));
        account.insert("unique_name".into(), json!("wren"));
        account.insert("role".into(), json!("user"));
        let body = assemble(
            Some(&identity()),
            Role::Admin,
            vec![
                (
                    "zend",
                    MeShare {
                        top: Map::new(),
                        own: json!({"tool_modes": ["restricted", "comprehensive"]}),
                    },
                ),
                (
                    "npcd",
                    MeShare {
                        top: account,
                        own: Value::Null,
                    },
                ),
            ],
        );
        assert_eq!(
            body,
            json!({
                "role": "admin",
                "email": "wren@example.com",
                "display": "Wren",
                "avatar_url": "https://example.com/w.png",
                "user_id": "u-7",
                "unique_name": "wren",
                "tenants": {
                    "zend": {"tool_modes": ["restricted", "comprehensive"]},
                    "npcd": null,
                },
            })
        );
    }
}
