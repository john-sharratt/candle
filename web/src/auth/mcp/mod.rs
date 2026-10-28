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
//! ```text
//! GET  /.well-known/oauth-protected-resource/mcp
//! GET  /.well-known/oauth-authorization-server
//! GET  /oauth/authorize    → sign-in, then the consent page
//! POST /oauth/authorize    → the answer: a code, or access_denied
//! POST /oauth/token        → a code or refresh token for a token pair
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Form, Query, State};
use axum::http::{header, HeaderMap, Uri};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use axum::Router;

use super::session::{self, Identity};
use super::Auth;

pub mod authorize;
pub mod client;
pub mod grants;
pub mod metadata;
pub mod page;
pub mod token;

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

impl Auth {
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
