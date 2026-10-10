//! HTTP-01 answers: `/.well-known/acme-challenge/{token}` on every hostname.
//!
//! Routed ahead of the site table — ahead of a redirect site's `301` and of a
//! site that proxies `/` to a daemon — because the CA asks for this path on
//! whichever name it is validating, and the answer is the gateway's, not the
//! site's.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use axum::extract::{Path, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;

pub const PREFIX: &str = "/.well-known/acme-challenge/";

/// Token → the body that proves it, shared between the issuer that sets them
/// and the route that answers them.
#[derive(Clone, Default)]
pub struct Challenges(Arc<RwLock<HashMap<String, String>>>);

impl Challenges {
    pub fn set(&self, token: &str, answer: &str) {
        self.0
            .write()
            .expect("challenge map poisoned")
            .insert(token.to_owned(), answer.to_owned());
    }

    pub fn clear(&self, token: &str) {
        self.0
            .write()
            .expect("challenge map poisoned")
            .remove(token);
    }

    fn get(&self, token: &str) -> Option<String> {
        self.0
            .read()
            .expect("challenge map poisoned")
            .get(token)
            .cloned()
    }

    /// Whether no token is being answered.
    pub fn is_empty(&self) -> bool {
        self.0.read().expect("challenge map poisoned").is_empty()
    }

    /// The route answering every token this map holds.
    pub fn router(&self) -> Router {
        Router::new()
            .route("/.well-known/acme-challenge/:token", get(answer))
            .with_state(self.clone())
    }
}

async fn answer(State(c): State<Challenges>, Path(token): Path<String>) -> Response {
    match c.get(&token) {
        Some(body) => (
            [
                (header::CONTENT_TYPE, "application/octet-stream"),
                (header::CACHE_CONTROL, "no-store"),
            ],
            body,
        )
            .into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::Service;

    async fn get_token(c: &Challenges, token: &str) -> (StatusCode, Vec<u8>) {
        let mut r = c.router();
        let res = r
            .call(
                Request::get(format!("{PREFIX}{token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = res.status();
        (
            status,
            res.into_body().collect().await.unwrap().to_bytes().to_vec(),
        )
    }

    #[tokio::test]
    async fn a_set_token_is_answered_and_a_cleared_one_is_not() {
        let c = Challenges::default();
        c.set("tok-1", "tok-1.thumbprint");
        assert_eq!(
            get_token(&c, "tok-1").await,
            (StatusCode::OK, b"tok-1.thumbprint".to_vec())
        );
        assert_eq!(get_token(&c, "other").await.0, StatusCode::NOT_FOUND);
        c.clear("tok-1");
        assert_eq!(get_token(&c, "tok-1").await.0, StatusCode::NOT_FOUND);
    }
}
