//! `POST /gateway/hooks/github`: GitHub's webhook (`docs/battlecities_app.md`
//! §4.1). Reached with no session, from the internet, on every host: the
//! delivery's signature is its only credential.
//!
//! | The delivery | The answer |
//! |---|---|
//! | Not signed with this gateway's secret, or no secret is configured | `401`, logged |
//! | Signed, and no push of a branch (a ping, a tag, a deleted branch) | `204` |
//! | Signed, a push of a branch | `200`: published on `repo/<repo>`, and a push of `main` of a repository in `cluster.deploy_on_push` told to a member ([`super::pushes`]) |

use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use serde_json::json;

use super::github::{pushed, signed, EVENT, SIGNATURE};
use super::pushes::tell_and_log;
use crate::bus::topic;
use crate::cluster::Gateway;

/// Where GitHub delivers.
pub const HOOK_PATH: &str = "/gateway/hooks/github";

/// The branch whose pushes are deployed.
const DEPLOYED: &str = "main";

pub fn router(gateway: Arc<Gateway>) -> Router {
    Router::new()
        .route(HOOK_PATH, post(github))
        .with_state(gateway)
}

fn said<'a>(headers: &'a HeaderMap, name: &str) -> &'a str {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
}

async fn github(State(gateway): State<Arc<Gateway>>, headers: HeaderMap, body: Bytes) -> Response {
    let believed = gateway
        .github_hook_secret
        .as_deref()
        .is_some_and(|secret| signed(secret, &body, said(&headers, SIGNATURE)));
    if !believed {
        tracing::warn!(
            bytes = body.len(),
            "hook: a delivery that this gateway's secret did not sign was refused"
        );
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let Some(push) = pushed(said(&headers, EVENT), &body) else {
        return StatusCode::NO_CONTENT.into_response();
    };
    let on = format!("repo/{}", push.repo);
    if !topic::valid(&on) {
        return StatusCode::NO_CONTENT.into_response();
    }
    let reached = gateway.bus.publish(
        &on,
        json!({"repo": push.repo, "branch": push.branch, "commit": push.commit, "paths": []}),
    );
    let deploys = push.branch == DEPLOYED && gateway.deploy_on_push.contains(&push.repo);
    tracing::info!(
        repo = %push.repo,
        branch = %push.branch,
        commit = %push.commit,
        reached,
        deploys,
        "hook: a push"
    );
    if deploys {
        // Told on a task of its own: GitHub waits ten seconds for an answer,
        // and a member starting a build may take longer to say so.
        tokio::spawn(async move {
            tell_and_log(&gateway, &push.repo, Some(&push.commit)).await;
        });
    }
    Json(json!({"reached": reached, "deploys": deploys})).into_response()
}

#[cfg(test)]
mod tests {
    use axum::body::{to_bytes, Body};
    use axum::http::Request;
    use cluster::gateway::Hub;
    use cluster::handshake::GroupKey;
    use hmac::{Hmac, Mac};
    use serde_json::Value;
    use sha2::Sha256;
    use tower::ServiceExt;

    use super::*;

    const SECRET: &[u8] = b"a webhook's secret";
    const COMMIT: &str = "a744abc8d1f0c4b7e9a3552f6c1d8e0b9a7f6e5d";

    fn gateway(secret: Option<&[u8]>) -> Arc<Gateway> {
        let hub = Hub::new(GroupKey::from_bytes([7; 32]), "hooks");
        Gateway::with_hub_and_pushes(
            hub,
            secret.map(<[u8]>::to_vec),
            vec!["battle-cities".into()],
        )
    }

    fn signature(secret: &[u8], body: &[u8]) -> String {
        let mut mac = Hmac::<Sha256>::new_from_slice(secret).unwrap();
        mac.update(body);
        let hex: String = mac
            .finalize()
            .into_bytes()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        format!("sha256={hex}")
    }

    fn delivery(event: &str, moved: &str, sign_with: &[u8]) -> Request<Body> {
        let body = json!({"ref": moved, "after": COMMIT, "deleted": false,
            "repository": {"name": "battle-cities"}})
        .to_string();
        Request::post(HOOK_PATH)
            .header(EVENT, event)
            .header(SIGNATURE, signature(sign_with, body.as_bytes()))
            .body(Body::from(body))
            .unwrap()
    }

    async fn answer(gateway: &Arc<Gateway>, req: Request<Body>) -> (StatusCode, Value) {
        let res = router(gateway.clone()).oneshot(req).await.unwrap();
        let status = res.status();
        let bytes = to_bytes(res.into_body(), 1 << 20).await.unwrap();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        )
    }

    /// **A signed push of a branch is published as Zen's own pushes are.**
    /// No `zend` member is joined here, so the push is told to nobody, which
    /// the answer does not wait for.
    #[tokio::test]
    async fn a_signed_push_is_published_on_its_repositorys_topic() {
        let gw = gateway(Some(SECRET));
        let mut heard = gw.bus.subscribe(vec!["repo/battle-cities".into()]);
        let (status, said) = answer(&gw, delivery("push", "refs/heads/main", SECRET)).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(said, json!({"reached": 1, "deploys": true}));
        assert_eq!(
            heard.messages.recv().await.unwrap().payload,
            json!({"repo": "battle-cities", "branch": "main", "commit": COMMIT, "paths": []})
        );
        // A branch that is not main is published and deploys nothing.
        let (status, said) = answer(&gw, delivery("push", "refs/heads/zen/x", SECRET)).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(said, json!({"reached": 1, "deploys": false}));
    }

    /// **Only this gateway's secret is believed**: another's signature, and
    /// any delivery to a gateway with no secret, is `401` and publishes
    /// nothing.
    #[tokio::test]
    async fn a_delivery_not_signed_with_the_secret_is_refused() {
        let gw = gateway(Some(SECRET));
        let mut heard = gw.bus.subscribe(vec!["repo/*".into()]);
        let forged = delivery("push", "refs/heads/main", b"another secret");
        assert_eq!(answer(&gw, forged).await.0, StatusCode::UNAUTHORIZED);
        let unsigned = Request::post(HOOK_PATH)
            .header(EVENT, "push")
            .body(Body::from("{}"))
            .unwrap();
        assert_eq!(answer(&gw, unsigned).await.0, StatusCode::UNAUTHORIZED);
        let none = gateway(None);
        let signed = delivery("push", "refs/heads/main", SECRET);
        assert_eq!(answer(&none, signed).await.0, StatusCode::UNAUTHORIZED);
        assert!(heard.messages.try_recv().is_err());
    }

    /// A ping and a tag are signed and are no push: `204`, nothing published.
    #[tokio::test]
    async fn a_signed_delivery_that_is_no_push_of_a_branch_is_acknowledged() {
        let gw = gateway(Some(SECRET));
        let mut heard = gw.bus.subscribe(vec!["repo/*".into()]);
        for req in [
            delivery("ping", "refs/heads/main", SECRET),
            delivery("push", "refs/tags/v1", SECRET),
        ] {
            assert_eq!(answer(&gw, req).await.0, StatusCode::NO_CONTENT);
        }
        assert!(heard.messages.try_recv().is_err());
    }
}
