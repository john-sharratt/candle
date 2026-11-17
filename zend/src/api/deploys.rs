//! `POST /v1/deploys/pushed`: a gateway's word that a repository's `main`
//! moved (`docs/battlecities_app.md` §4.4), `{"repo", "commit"?,
//! "attempted"?}`. The member deploys what is built from the repository,
//! unless it is already deployed or being deployed
//! ([`crate::mcp::tools::service::on_push`]), and answers what it did:
//! `{"repo", "commit", "did": "deploy" | "nothing", ...}`.
//!
//! Only a gateway says it: the request arrives over a cluster socket and
//! carries `x-tokera-forwarder`, which a gateway sets on its own requests and
//! removes from every one it passes on, and which this member's own port
//! removes from any peer that is no gateway.

use std::sync::Arc;

use axum::body::to_bytes;
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::json;
use web::cluster::ClusterPeer;
use web::hooks::pushes::OWN;

use crate::mcp::tools::service::on_push::pushed;
use crate::session::ZendSession;

/// The most a gateway's word may be.
pub const BODY_LIMIT: usize = 4 << 10;

/// What the gateway says.
#[derive(Debug, Deserialize, PartialEq, Eq)]
pub struct Pushed {
    pub repo: String,
    /// The commit a webhook named. The member reads origin itself, so it is
    /// what the log says, not what is deployed.
    #[serde(default)]
    pub commit: Option<String>,
    /// The commit the gateway last told of this repository at.
    #[serde(default)]
    pub attempted: Option<String>,
}

fn refused(status: StatusCode, error: &str, detail: String) -> Response {
    (status, Json(json!({"error": error, "detail": detail}))).into_response()
}

/// Whether `req` is a gateway's own.
pub fn from_gateway(req: &Request) -> bool {
    req.extensions().get::<ClusterPeer>().is_some() && req.headers().contains_key(OWN)
}

pub async fn post(State(session): State<Arc<ZendSession>>, req: Request) -> Response {
    if !from_gateway(&req) {
        return refused(
            StatusCode::FORBIDDEN,
            "gateways_only",
            "a push is told by a gateway, over a cluster socket".to_string(),
        );
    }
    let body = match to_bytes(req.into_body(), BODY_LIMIT).await {
        Ok(body) => body,
        Err(e) => return refused(StatusCode::BAD_REQUEST, "bad_body", e.to_string()),
    };
    let said: Pushed = match serde_json::from_slice(&body) {
        Ok(said) => said,
        Err(e) => return refused(StatusCode::BAD_REQUEST, "bad_body", e.to_string()),
    };
    match pushed(session, &said.repo, said.attempted.as_deref()).await {
        Ok(did) => Json(did).into_response(),
        Err(why) => refused(StatusCode::SERVICE_UNAVAILABLE, "not_deployed", why),
    }
}

#[cfg(test)]
mod tests {
    use axum::body::Body;

    use super::*;

    fn request(over_socket: bool, own: bool) -> Request {
        let mut b = Request::builder().uri("/v1/deploys/pushed");
        if own {
            b = b.header(OWN, "1");
        }
        let mut req = b.body(Body::empty()).unwrap();
        if over_socket {
            req.extensions_mut().insert(ClusterPeer { cluster: 0 });
        }
        req
    }

    /// **Only a gateway's own request is believed**: one over a cluster
    /// socket that carries the gateway's mark.
    #[test]
    fn only_a_gateways_own_request_tells_of_a_push() {
        assert!(from_gateway(&request(true, true)));
        assert!(!from_gateway(&request(true, false)));
        assert!(!from_gateway(&request(false, true)));
        assert!(!from_gateway(&request(false, false)));
    }

    #[test]
    fn the_gateways_word_names_a_repository_and_what_it_last_told() {
        let webhook: Pushed =
            serde_json::from_str(r#"{"repo": "battle-cities", "commit": "22e9ecf8"}"#).unwrap();
        assert_eq!(
            webhook,
            Pushed {
                repo: "battle-cities".into(),
                commit: Some("22e9ecf8".into()),
                attempted: None,
            }
        );
        let watch: Pushed =
            serde_json::from_str(r#"{"repo": "battle-cities", "attempted": null}"#).unwrap();
        assert_eq!(watch.attempted, None);
        assert!(serde_json::from_str::<Pushed>(r#"{"commit": "22e9ecf8"}"#).is_err());
    }
}
