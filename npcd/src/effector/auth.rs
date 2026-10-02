//! The device surface's auth: a bearer token, and nothing else.
//!
//! This is the parallel to [`crate::guard::Api`], and it exists because that one
//! cannot be reused: `Api::route` hardcodes the operator role check over the
//! gateway's `x-tokera-*` headers, with no escape hatch by design. The effector
//! surface has a different notion of *who is acting* — a character resolved from
//! a token, not a human resolved from a role (effector design §8.3) — so it
//! carries its own middleware.
//!
//! # It reads the token and only the token
//!
//! The check, whole: the request must carry `Authorization: Bearer <token>`,
//! the token must resolve in [`Tokens`], and that resolution *is* the caller.
//! No roles, no ownership, and — critically — **`x-tokera-*` is never read**.
//! Two hazards make that non-negotiable, both from the surface sitting behind a
//! gateway that npcd trusts (`behind_gateway`):
//!
//! - The in-process fast path builds a request and stamps the token; it must
//!   never also carry an `x-tokera-*`, or the operator middleware elsewhere
//!   would read a character as a human. This layer never consults those headers,
//!   so a stray one changes nothing here.
//! - The external mount is on the same gateway-fronted domain, so a client can
//!   send *both* a token and `x-tokera-*`. This layer resolves the token and
//!   never falls through to a role check, so the admin headers are inert on this
//!   surface — pinned by [`tests::admin_headers_without_a_token_are_refused`].

use axum::extract::{Request, State};
use axum::http::{header::AUTHORIZATION, StatusCode};
use axum::middleware::Next;
use axum::response::Response;

use npc_map::world::Where;

use crate::api::err;
use crate::effector::token::{Scope, Tokens};

use std::sync::Arc;

/// Who the device surface decided is acting, made available to every handler
/// behind the layer through the request's extensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceCaller {
    pub npc_id: u64,
    pub scope: Scope,
}

/// The standpoint a device call is reach-checked *as-of* — the position the
/// body stood in when the turn's grammar was built, carried on the request so
/// the reach check matches the situation the character actually read.
///
/// **The fix for a body that walks while it thinks.** A character's grammar is
/// snapshotted before the decode ([`crate::engine::runtime::Runtime::within`]),
/// but the world's metronome advances a walking body a leg every 500 ms while
/// the decode takes seconds — so by the time the chosen `query`/`invoke` runs,
/// the body has left the room the address named, and the live reach check 404s
/// an address the grammar had just offered. Pinning the grammar-time standpoint
/// closes that time-of-check/time-of-use gap.
///
/// **It is trusted because it cannot be forged.** This is a Rust request
/// extension, inserted only by the in-process fast path
/// ([`crate::engine::runtime::Runtime::effector_call`]) for an act the engine
/// itself approved this turn. An external socket client cannot inject a typed
/// extension, so an as-npc caller can never use it to reach past its real
/// standpoint — the live check still gates every off-daemon request.
#[derive(Debug, Clone)]
pub struct PinnedStandpoint(pub Where);

/// Resolve the bearer token to a [`DeviceCaller`], or refuse the request.
///
/// On success the caller is inserted into the request extensions and the handler
/// runs; on failure the estate error shape (`{error, detail}`, [`err`]) is
/// returned with `401`, the same refusal for an absent token and a bad one — a
/// device surface says only that the credential did not work, never which half
/// of it was wrong.
pub async fn device_auth(
    State(tokens): State<Arc<Tokens>>,
    mut req: Request,
    next: Next,
) -> Response {
    let Some(grant) = bearer(&req).and_then(|token| tokens.resolve(token)) else {
        return err(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "the effector device needs a valid bearer token",
        );
    };
    req.extensions_mut().insert(DeviceCaller {
        npc_id: grant.npc_id,
        scope: grant.scope,
    });
    next.run(req).await
}

/// The token from an `Authorization: Bearer <token>` header, if there is one.
///
/// Only that header — this is the one place the surface reads a credential, and
/// it reads no other. The scheme match is ASCII-case-insensitive (RFC 7235), and
/// the token is trimmed so a client's stray space is not folded into the secret.
fn bearer(req: &Request) -> Option<&str> {
    let value = req.headers().get(AUTHORIZATION)?.to_str().ok()?;
    let token = value.strip_prefix("Bearer ").or_else(|| {
        // Tolerate the lowercase spelling some clients send; the value after the
        // scheme is the secret and is matched exactly.
        value
            .get(..7)
            .filter(|p| p.eq_ignore_ascii_case("bearer "))
            .map(|_| &value[7..])
    })?;
    let token = token.trim();
    (!token.is_empty()).then_some(token)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request as HttpRequest;
    use axum::middleware::from_fn_with_state;
    use axum::routing::get;
    use axum::Router;
    use tower::ServiceExt;

    /// A tiny router that is nothing but the device-auth layer over a handler
    /// that does no work of its own — so a 200 means the layer let the request
    /// through, and it is the layer under test, not a handler helping it.
    fn app(tokens: Arc<Tokens>) -> Router {
        Router::new()
            .route("/", get(|| async { "ok" }))
            .route_layer(from_fn_with_state(tokens, device_auth))
            .with_state(())
    }

    fn tokens() -> Arc<Tokens> {
        let dir = std::env::temp_dir().join(format!(
            "npcd-effector-auth-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        Arc::new(Tokens::load(dir).unwrap())
    }

    async fn status(app: Router, req: HttpRequest<Body>) -> StatusCode {
        app.oneshot(req).await.unwrap().status()
    }

    /// A valid bearer token passes.
    #[tokio::test]
    async fn a_valid_token_is_let_through() {
        let t = tokens();
        let token = t.mint(1, Scope::AsNpc).unwrap();
        let req = HttpRequest::builder()
            .uri("/")
            .header(AUTHORIZATION, format!("Bearer {token}"))
            .body(Body::empty())
            .unwrap();
        assert_eq!(status(app(t), req).await, StatusCode::OK);
    }

    /// No header at all is refused — the device surface is never open.
    #[tokio::test]
    async fn a_request_with_no_token_is_refused() {
        let req = HttpRequest::builder().uri("/").body(Body::empty()).unwrap();
        assert_eq!(status(app(tokens()), req).await, StatusCode::UNAUTHORIZED);
    }

    /// A token nobody minted is refused, with the same status as none — the
    /// surface does not say which half of the credential was wrong.
    #[tokio::test]
    async fn a_bad_token_is_refused() {
        let req = HttpRequest::builder()
            .uri("/")
            .header(AUTHORIZATION, "Bearer definitely-not-minted")
            .body(Body::empty())
            .unwrap();
        assert_eq!(status(app(tokens()), req).await, StatusCode::UNAUTHORIZED);
    }

    /// **The whole reason a separate auth exists.** A request carrying only the
    /// gateway's admin identity headers and no bearer token is refused: the
    /// device surface never honours `x-tokera-*`, so a human's admin role buys
    /// nothing here. Were this to pass, an operator behind the gateway could act
    /// as any character without a token.
    #[tokio::test]
    async fn admin_headers_without_a_token_are_refused() {
        let req = HttpRequest::builder()
            .uri("/")
            .header("x-tokera-user", "boss")
            .header("x-tokera-provider", "google")
            .header("x-tokera-email", "johnathan.sharratt@gmail.com")
            .body(Body::empty())
            .unwrap();
        assert_eq!(status(app(tokens()), req).await, StatusCode::UNAUTHORIZED);
    }

    /// And the same headers alongside a *valid* token change nothing: the token
    /// is what is read, the headers are inert. This pins the external-mount
    /// hazard — a client sending both must be the character the token names.
    #[tokio::test]
    async fn a_valid_token_wins_and_the_headers_are_ignored() {
        let t = tokens();
        let token = t.mint(9, Scope::AsNpc).unwrap();
        let req = HttpRequest::builder()
            .uri("/")
            .header(AUTHORIZATION, format!("Bearer {token}"))
            .header("x-tokera-user", "boss")
            .header("x-tokera-email", "johnathan.sharratt@gmail.com")
            .body(Body::empty())
            .unwrap();
        assert_eq!(status(app(t), req).await, StatusCode::OK);
    }

    /// The lowercase scheme spelling is accepted, and an empty token is not.
    #[tokio::test]
    async fn the_scheme_is_case_insensitive_and_an_empty_token_is_refused() {
        let t = tokens();
        let token = t.mint(2, Scope::AsNpc).unwrap();
        let lower = HttpRequest::builder()
            .uri("/")
            .header(AUTHORIZATION, format!("bearer {token}"))
            .body(Body::empty())
            .unwrap();
        assert_eq!(status(app(t.clone()), lower).await, StatusCode::OK);

        let empty = HttpRequest::builder()
            .uri("/")
            .header(AUTHORIZATION, "Bearer ")
            .body(Body::empty())
            .unwrap();
        assert_eq!(status(app(t), empty).await, StatusCode::UNAUTHORIZED);
    }
}
