//! Service keys (`docs/mcp_file_tools.md` §4): how a service on the gateway's
//! machine — yenbot first — acts as itself. A request presenting a service key
//! as `Authorization: Bearer <hex>`, from an address the configuration lists in
//! `server.trusted_backends`, is the key's identity on the routes the key names
//! and on no others; the bus admits it to the topics it names (§5.2). The key
//! itself goes no further than this gateway, like the ops key.
//!
//! The identity's subject is `service:<name>`, its provider [`SERVICE_PROVIDER`]:
//! no sign-in issues one, so it collides with no person, and the role table
//! grants it a role by that subject (`host/roles.yaml`).

use axum::http::HeaderMap;

use super::bearer_key::BearerKey;
use crate::auth::Identity;
use crate::bus::topic::admits;
use crate::config::ServiceKeyConfig;

/// The provider a service key's identity names.
pub const SERVICE_PROVIDER: &str = "service-key";

/// One loaded service key.
pub struct ServiceKey {
    key: BearerKey,
    identity: Identity,
    routes: Vec<String>,
    subscribe: Vec<String>,
    publish: Vec<String>,
}

/// Whether `path` is `route` or under it: `/v1/models` names `/v1/models` and
/// `/v1/models/x`, never `/v1/modelsx`.
fn under(path: &str, route: &str) -> bool {
    let route = route.trim_end_matches('/');
    path == route
        || path
            .strip_prefix(route)
            .is_some_and(|rest| rest.starts_with('/'))
}

impl ServiceKey {
    /// The key `cfg` names, its file read now.
    pub fn load(cfg: &ServiceKeyConfig) -> anyhow::Result<ServiceKey> {
        Ok(ServiceKey::new(BearerKey::load(&cfg.key_file)?, cfg))
    }

    /// `key`, as `cfg` describes it.
    pub fn new(key: BearerKey, cfg: &ServiceKeyConfig) -> ServiceKey {
        ServiceKey {
            key,
            identity: Identity {
                provider: SERVICE_PROVIDER.into(),
                sub: format!("service:{}", cfg.name),
                email: String::new(),
                name: cfg.name.clone(),
                picture: String::new(),
                // As long as the key file lasts.
                exp: u64::MAX,
            },
            routes: cfg.routes.clone(),
            subscribe: cfg.topics.subscribe.clone(),
            publish: cfg.topics.publish.clone(),
        }
    }

    /// Whether `headers` present this key, accepted or not.
    pub fn presented(&self, headers: &HeaderMap) -> bool {
        self.key.presented(headers)
    }

    /// The key's identity, when `headers` present it on a route it names.
    pub fn caller(&self, headers: &HeaderMap, path: &str) -> Option<Identity> {
        (self.presented(headers) && self.routes.iter().any(|r| under(path, r)))
            .then(|| self.identity.clone())
    }

    /// The key's identity, whatever the route — for the bus, which checks
    /// topics rather than paths.
    pub fn identity(&self) -> &Identity {
        &self.identity
    }

    /// Whether the key may subscribe to `topic`.
    pub fn may_subscribe(&self, topic: &str) -> bool {
        self.subscribe.iter().any(|p| admits(p, topic))
    }

    /// Whether the key may publish on `topic`.
    pub fn may_publish(&self, topic: &str) -> bool {
        self.publish.iter().any(|p| admits(p, topic))
    }
}

/// The key `headers` present among `keys`, if any.
pub fn presented<'a>(keys: &'a [ServiceKey], headers: &HeaderMap) -> Option<&'a ServiceKey> {
    keys.iter().find(|k| k.presented(headers))
}

#[cfg(test)]
mod tests {
    use axum::http::{header, HeaderValue};

    use super::*;
    use crate::auth::{Person, Principal, Role, Roles};
    use crate::config::ServiceTopics;

    fn bearer(token: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(
            header::AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {token}")).unwrap(),
        );
        h
    }

    fn yenbot() -> ServiceKey {
        ServiceKey::new(
            BearerKey::from_hex(&"5e".repeat(32)).unwrap(),
            &ServiceKeyConfig {
                name: "yenbot".into(),
                key_file: "unused".into(),
                routes: vec!["/v1/chat/completions".into(), "/v1/models".into()],
                topics: ServiceTopics {
                    subscribe: vec!["repo/yenbot".into(), "gate/yenbot".into()],
                    publish: vec!["gate/yenbot/result/*".into()],
                },
            },
        )
    }

    /// **The key is its service on its own routes, and nobody elsewhere.**
    #[test]
    fn the_key_is_its_service_on_its_routes_only() {
        let key = yenbot();
        let h = bearer(&"5e".repeat(32));
        let id = key.caller(&h, "/v1/chat/completions").expect("its route");
        assert_eq!(
            (id.provider.as_str(), id.sub.as_str(), id.email.as_str()),
            (SERVICE_PROVIDER, "service:yenbot", "")
        );
        assert!(key.caller(&h, "/v1/models/passthrough").is_some());
        assert!(key.caller(&h, "/v1/modelsx").is_none());
        assert!(key.caller(&h, "/v1/conversations").is_none());
        assert!(key.caller(&h, "/mcp").is_none());
        assert!(key
            .caller(&bearer(&"5f".repeat(32)), "/v1/models")
            .is_none());
        assert!(key.caller(&HeaderMap::new(), "/v1/models").is_none());
    }

    /// **The role table grants a service by its subject**, which no sign-in
    /// can ever produce.
    #[test]
    fn the_role_table_grants_a_service_by_its_subject() {
        let roles = Roles::new(vec![Person {
            who: Principal::Sub("service:yenbot".into()),
            role: Role::User,
        }])
        .unwrap();
        let id = yenbot()
            .caller(&bearer(&"5e".repeat(32)), "/v1/models")
            .unwrap();
        assert_eq!(roles.of(Some(&id)), Role::User);
    }

    /// **Topics are the key's, exactly or under a trailing `*`.**
    #[test]
    fn topics_are_admitted_exactly_or_by_prefix() {
        let key = yenbot();
        assert!(key.may_subscribe("repo/yenbot"));
        assert!(!key.may_subscribe("repo/candle"));
        assert!(!key.may_subscribe("repo/yenbot2"));
        assert!(key.may_publish("gate/yenbot/result/01JD"));
        assert!(!key.may_publish("gate/yenbot"));
        assert!(!key.may_publish("repo/yenbot"));
    }
}
