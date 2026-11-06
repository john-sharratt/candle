//! The bus over HTTP, for a caller that publishes rather than listens
//! (`docs/mcp_file_tools.md` §5.2): a cluster member, over the socket it
//! already holds to the gateway, and a service by its key.
//!
//! - `POST /gateway/events/publish {topic, payload}` answers `{reached}`.
//! - `POST /gateway/events/request {topic, payload, reply, timeout_ms}`
//!   publishes and answers the first payload on `reply` (200), or
//!   `no_subscriber` (404) when nothing listens on `topic`, or `timeout` (504).
//!
//! A member may use any topic; a service only those its key names, `reply`
//! included as one it subscribes to.

use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{ConnectInfo, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Extension, Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};

use super::{socket, topic, Unanswered};
use crate::cluster::service_key::{self, ServiceKey};
use crate::cluster::{Gateway, MemberOrigin};

/// The longest a request waits for its answer.
pub const MAX_WAIT: Duration = Duration::from_secs(30);

#[derive(Clone)]
pub struct BusState {
    pub gateway: Arc<Gateway>,
    pub trusted_backends: Arc<Vec<IpAddr>>,
}

/// Who is using the bus.
pub enum Caller<'a> {
    /// A cluster member, over its socket: any topic.
    Member,
    /// A service, by its key: the topics the key names.
    Service(&'a ServiceKey),
}

impl Caller<'_> {
    pub fn may_publish(&self, topic: &str) -> bool {
        match self {
            Caller::Member => true,
            Caller::Service(k) => k.may_publish(topic),
        }
    }

    pub fn may_subscribe(&self, topic: &str) -> bool {
        match self {
            Caller::Member => true,
            Caller::Service(k) => k.may_subscribe(topic),
        }
    }
}

/// The caller, when the bus admits one: a member's stream, or a service key
/// presented from a trusted backend.
pub fn caller<'a>(
    s: &'a BusState,
    headers: &HeaderMap,
    origin: Option<&MemberOrigin>,
    peer: Option<SocketAddr>,
) -> Option<Caller<'a>> {
    admitted(
        &s.gateway.service_keys,
        &s.trusted_backends,
        headers,
        origin,
        peer,
    )
}

/// [`caller`] over its parts.
fn admitted<'a>(
    keys: &'a [ServiceKey],
    trusted_backends: &[IpAddr],
    headers: &HeaderMap,
    origin: Option<&MemberOrigin>,
    peer: Option<SocketAddr>,
) -> Option<Caller<'a>> {
    if origin.is_some() {
        return Some(Caller::Member);
    }
    let from_backend = peer.is_some_and(|p| trusted_backends.contains(&p.ip()));
    if !from_backend {
        return None;
    }
    service_key::presented(keys, headers).map(Caller::Service)
}

pub fn router(gateway: Arc<Gateway>, trusted_backends: Vec<IpAddr>) -> Router {
    Router::new()
        .route("/gateway/events", get(socket::upgrade))
        .route("/gateway/events/publish", post(publish))
        .route("/gateway/events/request", post(request))
        .with_state(BusState {
            gateway,
            trusted_backends: Arc::new(trusted_backends),
        })
}

fn refused(status: StatusCode, error: &str, detail: String) -> Response {
    (status, Json(json!({"error": error, "detail": detail}))).into_response()
}

fn unadmitted() -> Response {
    refused(
        StatusCode::UNAUTHORIZED,
        "unauthorized",
        "the bus takes a cluster member or a service key".into(),
    )
}

#[derive(Deserialize)]
pub struct Publish {
    pub topic: String,
    #[serde(default)]
    pub payload: Value,
}

async fn publish(
    State(s): State<BusState>,
    origin: Option<Extension<MemberOrigin>>,
    peer: Option<ConnectInfo<SocketAddr>>,
    headers: HeaderMap,
    Json(body): Json<Publish>,
) -> Response {
    let Some(who) = caller(
        &s,
        &headers,
        origin.as_ref().map(|o| &o.0),
        peer.map(|p| p.0),
    ) else {
        return unadmitted();
    };
    if !topic::valid(&body.topic) {
        return refused(StatusCode::BAD_REQUEST, "bad_topic", body.topic);
    }
    if !who.may_publish(&body.topic) {
        return refused(
            StatusCode::FORBIDDEN,
            "forbidden",
            format!("{} is not this key's to publish", body.topic),
        );
    }
    let reached = s.gateway.bus.publish(&body.topic, body.payload);
    Json(json!({"reached": reached})).into_response()
}

#[derive(Deserialize)]
pub struct Ask {
    pub topic: String,
    #[serde(default)]
    pub payload: Value,
    pub reply: String,
    pub timeout_ms: u64,
}

async fn request(
    State(s): State<BusState>,
    origin: Option<Extension<MemberOrigin>>,
    peer: Option<ConnectInfo<SocketAddr>>,
    headers: HeaderMap,
    Json(body): Json<Ask>,
) -> Response {
    let Some(who) = caller(
        &s,
        &headers,
        origin.as_ref().map(|o| &o.0),
        peer.map(|p| p.0),
    ) else {
        return unadmitted();
    };
    if !topic::valid(&body.topic) || !topic::valid(&body.reply) {
        return refused(
            StatusCode::BAD_REQUEST,
            "bad_topic",
            format!("{} → {}", body.topic, body.reply),
        );
    }
    if !who.may_publish(&body.topic) || !who.may_subscribe(&body.reply) {
        return refused(
            StatusCode::FORBIDDEN,
            "forbidden",
            format!("{} → {} is not this key's", body.topic, body.reply),
        );
    }
    let wait = Duration::from_millis(body.timeout_ms).min(MAX_WAIT);
    match s
        .gateway
        .bus
        .request(&body.topic, body.payload, &body.reply, wait)
        .await
    {
        Ok(answer) => Json(answer).into_response(),
        Err(Unanswered::NoSubscriber) => refused(
            StatusCode::NOT_FOUND,
            "no_subscriber",
            format!("nothing listens on {}", body.topic),
        ),
        Err(Unanswered::Timeout) => refused(
            StatusCode::GATEWAY_TIMEOUT,
            "timeout",
            format!("no answer on {} within {} ms", body.reply, wait.as_millis()),
        ),
    }
}

#[cfg(test)]
mod tests {
    use axum::http::{header, HeaderValue};

    use super::*;
    use crate::cluster::bearer_key::BearerKey;
    use crate::config::{ServiceKeyConfig, ServiceTopics};

    fn keys() -> Vec<ServiceKey> {
        vec![ServiceKey::new(
            BearerKey::from_hex(&"5e".repeat(32)).unwrap(),
            &ServiceKeyConfig {
                name: "yenbot".into(),
                key_file: "unused".into(),
                routes: vec![],
                topics: ServiceTopics {
                    subscribe: vec!["repo/yenbot".into()],
                    publish: vec!["gate/yenbot/result/*".into()],
                },
            },
        )]
    }

    fn keyed() -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(
            header::AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", "5e".repeat(32))).unwrap(),
        );
        h
    }

    fn at(ip: &str) -> Option<SocketAddr> {
        Some(SocketAddr::new(ip.parse().unwrap(), 50000))
    }

    /// **The bus admits a member's stream, and a service key from a trusted
    /// backend — nothing else.**
    #[test]
    fn the_bus_admits_members_and_keyed_services_from_a_backend() {
        let keys = keys();
        let trusted: Vec<IpAddr> = vec!["127.0.0.1".parse().unwrap()];
        let member = MemberOrigin {
            app: "zend".into(),
            node: 1,
        };
        assert!(matches!(
            admitted(&keys, &trusted, &HeaderMap::new(), Some(&member), None),
            Some(Caller::Member)
        ));
        let service = admitted(&keys, &trusted, &keyed(), None, at("127.0.0.1")).expect("keyed");
        assert!(service.may_subscribe("repo/yenbot"));
        assert!(!service.may_subscribe("repo/candle"));
        assert!(service.may_publish("gate/yenbot/result/01JD"));
        assert!(!service.may_publish("repo/yenbot"));
        assert!(
            admitted(&keys, &trusted, &keyed(), None, at("203.0.113.9")).is_none(),
            "a key from outside the trusted backends"
        );
        assert!(admitted(&keys, &trusted, &HeaderMap::new(), None, at("127.0.0.1")).is_none());
        assert!(admitted(&keys, &trusted, &keyed(), None, None).is_none());
    }
}
