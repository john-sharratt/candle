//! The OAuth 2.1 authorization server for the remote MCP endpoint, `/mcp`
//! (`docs/claude_zend_gateway.md` §12).
//!
//! Sign-in belongs to the gateway, and so does this: the gateway serves the
//! discovery documents, asks the signed-in user's consent, and issues codes
//! and tokens — signed with the session key, so every peered gateway honours
//! them ([`grants`]). On `/mcp` a bearer token is an identity like the session
//! cookie is ([`Auth::mcp_identity`]): the gateway states it on `X-Tokera-*`
//! to the member serving `/mcp`, which never sees a token's meaning.
//!
//! A site that names scopes for its `/api` (`sites[].api_scopes`) has a
//! second resource, `https://<host>/api` (`docs/battlecities_app.md` §6): a
//! token for it is anyone's who signs in, carries the scopes its person
//! allowed, and is read on that site's `/api` paths alone
//! ([`Auth::api_grant`]), where the gateway states those scopes upstream as
//! `X-Tokera-Scope`.
//!
//! ```text
//! GET  /.well-known/oauth-protected-resource/mcp
//! GET  /.well-known/oauth-protected-resource/api
//! GET  /.well-known/oauth-authorization-server
//! GET  /oauth/authorize    → sign-in, then the consent page
//! POST /oauth/authorize    → the answer: a code, or access_denied
//! POST /oauth/token        → a code or refresh token for a token pair
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Form, Query, State};
use axum::http::{header, HeaderMap, StatusCode, Uri};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use axum::Router;

use super::session::{self, Identity};
use super::{estate, Auth};
use crate::config::{host_name, hosts_match};

pub mod authorize;
pub mod client;
pub mod grants;
pub mod metadata;
pub mod page;
pub mod token;

use authorize::Api;
use grants::{Grant, Kind};

/// This gateway's public origin, as the client reached it: the scheme the
/// deployment's sign-in names, and the request's `Host`.
pub fn origin(auth: &Auth, headers: &HeaderMap) -> String {
    let scheme = if auth.is_secure() { "https" } else { "http" };
    let host = headers
        .get(header::HOST)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("localhost");
    format!("{scheme}://{host}")
}

/// The resource `/mcp` is on this origin.
pub fn resource(auth: &Auth, headers: &HeaderMap) -> String {
    format!("{}{}", origin(auth, headers), metadata::MCP_PATH)
}

/// The bearer token `headers` carry, if any.
pub fn bearer(headers: &HeaderMap) -> Option<&str> {
    let value = headers.get(header::AUTHORIZATION)?.to_str().ok()?;
    let (scheme, token) = value.split_once(' ')?;
    scheme
        .eq_ignore_ascii_case("bearer")
        .then(|| token.trim())
        .filter(|t| !t.is_empty())
}

/// Whether `path` is under a site's `/api`.
fn under_api(path: &str) -> bool {
    path.strip_prefix(metadata::API_PATH)
        .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
}

impl Auth {
    /// The `/api` of the site `headers` name by their `Host`, when the site
    /// names scopes for one (`sites[].api_scopes`): its resource on this
    /// origin, and those scopes.
    pub fn api(&self, headers: &HeaderMap) -> Option<Api> {
        let host = headers.get(header::HOST)?.to_str().ok()?;
        let name = host_name(host);
        let site = self
            .api_sites
            .iter()
            .find(|site| hosts_match(&site.hosts, &name))?;
        Some(Api {
            resource: format!("{}{}", origin(self, headers), metadata::API_PATH),
            scopes: site.scopes.clone(),
        })
    }

    /// Who a request under a site's `/api` is by its bearer token, and what
    /// the token may do: the person an access token for that site's `/api`
    /// was granted by, and its scopes, space-separated, less any the site
    /// no longer names (`docs/battlecities_app.md` §6). No other path reads
    /// such a token, and no site that names no scopes.
    pub fn api_grant(&self, headers: &HeaderMap, path: &str) -> Option<(Identity, String)> {
        if !under_api(path) {
            return None;
        }
        let api = self.api(headers)?;
        let token = bearer(headers)?;
        let grant: Grant = grants::open(&self.key, Kind::McpAccess, token, session::now_secs())?;
        if grant.resource != api.resource {
            return None;
        }
        let held: Vec<&str> = grant
            .scopes
            .iter()
            .map(String::as_str)
            .filter(|scope| api.scopes.iter().any(|s| s == scope))
            .collect();
        let identity = grant.identity(session::now_secs() + grants::ACCESS_TTL_SECS);
        Some((identity, held.join(" ")))
    }

    /// Who a request to `path` is by its bearer token: only `/mcp` reads one,
    /// and only an access token granted for this origin's `/mcp`.
    pub fn mcp_identity(&self, headers: &HeaderMap, path: &str) -> Option<Identity> {
        if path != metadata::MCP_PATH {
            return None;
        }
        let token = bearer(headers)?;
        let grant: Grant = grants::open(&self.key, Kind::McpAccess, token, session::now_secs())?;
        (grant.resource == resource(self, headers))
            .then(|| grant.identity(session::now_secs() + grants::ACCESS_TTL_SECS))
    }

    /// Who a backend acting for a person is (`server.trusted_backends`,
    /// `docs/mcp_app.md` §2): the person whose `/mcp` access token it
    /// presents, on whatever path and site it calls — a token this gateway
    /// issued, for an MCP resource it serves. A backend that holds no token
    /// is nobody: it cannot name a person it was not handed.
    pub fn mcp_delegated(&self, headers: &HeaderMap) -> Option<Identity> {
        let token = bearer(headers)?;
        let grant: Grant = grants::open(&self.key, Kind::McpAccess, token, session::now_secs())?;
        grant
            .resource
            .ends_with(metadata::MCP_PATH)
            .then(|| grant.identity(session::now_secs() + grants::ACCESS_TTL_SECS))
    }

    /// The authorization server's routes.
    pub fn mcp_router(self: Arc<Self>) -> Router {
        Router::new()
            .route(metadata::RESOURCE_DOC_PATH, get(resource_doc))
            .route(metadata::API_RESOURCE_DOC_PATH, get(api_resource_doc))
            .route(metadata::SERVER_DOC_PATH, get(server_doc))
            .route(
                metadata::AUTHORIZE_PATH,
                get(authorize_ask).post(authorize_decide),
            )
            .route(metadata::TOKEN_PATH, post(token_exchange))
            .with_state(self)
    }
}

async fn resource_doc(State(auth): State<Arc<Auth>>, headers: HeaderMap) -> Response {
    Json(metadata::protected_resource(
        &resource(&auth, &headers),
        &origin(&auth, &headers),
    ))
    .into_response()
}

/// The protected-resource document of this site's `/api`; `404` on a site
/// that names no scopes for one.
async fn api_resource_doc(State(auth): State<Arc<Auth>>, headers: HeaderMap) -> Response {
    match auth.api(&headers) {
        Some(api) => Json(metadata::protected_api(
            &api.resource,
            &origin(&auth, &headers),
            &api.scopes,
        ))
        .into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn server_doc(State(auth): State<Arc<Auth>>, headers: HeaderMap) -> Response {
    Json(metadata::authorization_server(&origin(&auth, &headers))).into_response()
}

async fn authorize_ask(
    State(auth): State<Arc<Auth>>,
    headers: HeaderMap,
    uri: Uri,
    Query(query): Query<HashMap<String, String>>,
) -> Response {
    let path_and_query = uri
        .path_and_query()
        .map_or(metadata::AUTHORIZE_PATH, |p| p.as_str());
    authorize::ask(
        &auth.key,
        &auth.roles,
        auth.identity(&headers),
        path_and_query,
        &query,
        &resource(&auth, &headers),
        auth.api(&headers).as_ref(),
        &estate(&auth, &headers).0.cookie_domain,
        session::now_secs(),
    )
    .await
}

async fn authorize_decide(
    State(auth): State<Arc<Auth>>,
    headers: HeaderMap,
    Form(form): Form<HashMap<String, String>>,
) -> Response {
    authorize::decide(
        &auth.key,
        &auth.roles,
        auth.identity(&headers),
        &form,
        &origin(&auth, &headers),
        session::now_secs(),
    )
}

async fn token_exchange(
    State(auth): State<Arc<Auth>>,
    Form(form): Form<HashMap<String, String>>,
) -> Response {
    token::answer(token::exchange(
        &auth.key,
        &auth.roles,
        &form,
        session::now_secs(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    #[test]
    fn only_a_bearer_token_is_read() {
        let mut h = HeaderMap::new();
        h.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("bearer abc"),
        );
        assert_eq!(bearer(&h), Some("abc"));
        h.insert(header::AUTHORIZATION, HeaderValue::from_static("Basic abc"));
        assert_eq!(bearer(&h), None);
        assert_eq!(bearer(&HeaderMap::new()), None);
    }
}
