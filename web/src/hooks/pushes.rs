//! Telling the cluster of a push (`docs/battlecities_app.md` §4.2, §4.4).
//!
//! A push to a repository in `cluster.deploy_on_push` is told to one ready
//! `zend` member, `POST /v1/deploys/pushed`, which deploys what is built from
//! that repository unless the commit is already deployed or being deployed.
//! The gateway tells it when GitHub's webhook says so ([`super::http`]), and
//! once a minute on its own ([`Gateway::watch_pushes`]): a webhook can be
//! lost, and the member reads origin itself, so the watch needs no commit.
//!
//! The gateway remembers the commit each repository was last told at
//! ([`Told`]) and says so on the watch's request, so a commit whose build
//! failed is not built again every minute. The request carries
//! `x-tokera-forwarder`, which this gateway removes from every request it
//! passes on for a caller: the member believes it is the gateway's own.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use axum::body::Body;
use axum::http::{header, Method, Request};
use http_body_util::BodyExt;
use serde_json::{json, Value};

use crate::cluster::transport::send;
use crate::cluster::Gateway;

/// The route of a `zend` member that takes a push.
pub const PUSHED_PATH: &str = "/v1/deploys/pushed";

/// The header that marks a request as this gateway's own.
pub const OWN: &str = "x-tokera-forwarder";

/// How often origin is looked at for a push no webhook told of.
pub const WATCH_EVERY: Duration = Duration::from_secs(60);

/// The app whose members deploy.
const DEPLOYER: &str = "zend";

/// The commit each repository was last told at.
#[derive(Default)]
pub struct Told(Mutex<HashMap<String, String>>);

impl Told {
    fn held(&self) -> MutexGuard<'_, HashMap<String, String>> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The commit `repo` was last told at.
    pub fn at(&self, repo: &str) -> Option<String> {
        self.held().get(repo).cloned()
    }

    /// `repo` was told at `commit`.
    pub fn told(&self, repo: &str, commit: &str) {
        self.held().insert(repo.to_string(), commit.to_string());
    }
}

/// What a member is asked: the repository, the commit a webhook named, and
/// the commit it was last told at — the watch's, which names no commit.
pub fn ask(repo: &str, commit: Option<&str>, told: Option<&str>) -> Value {
    match commit {
        Some(commit) => json!({"repo": repo, "commit": commit}),
        None => json!({"repo": repo, "attempted": told}),
    }
}

/// Tell a ready `zend` member that `repo`'s `main` moved — to `commit`, when a
/// webhook named it — and answer what it said.
pub async fn tell(gateway: &Gateway, repo: &str, commit: Option<&str>) -> Result<Value, String> {
    let node = gateway
        .hub
        .any_ready(DEPLOYER)
        .ok_or_else(|| format!("no {DEPLOYER} member is ready to deploy"))?;
    let told = gateway.told.at(repo);
    let body = ask(repo, commit, told.as_deref());
    let req = Request::builder()
        .method(Method::POST)
        .uri(PUSHED_PATH)
        .header(header::CONTENT_TYPE, "application/json")
        .header(OWN, "1")
        .body(Body::from(body.to_string()))
        .map_err(|e| format!("the request does not build: {e}"))?;
    let res = send(&gateway.hub, DEPLOYER, node, req)
        .await
        .map_err(|e| format!("member {node:016x} was not reached: {e}"))?;
    let status = res.status();
    let bytes = res
        .into_body()
        .collect()
        .await
        .map_err(|e| format!("member {node:016x}'s answer did not arrive: {e}"))?
        .to_bytes();
    let said: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    if !status.is_success() {
        return Err(format!("member {node:016x} answered {status}: {said}"));
    }
    if let Some(commit) = said.get("commit").and_then(Value::as_str) {
        gateway.told.told(repo, commit);
    }
    Ok(said)
}

/// Tell of `repo`'s push and log what came of it.
pub async fn tell_and_log(gateway: &Gateway, repo: &str, commit: Option<&str>) {
    match tell(gateway, repo, commit).await {
        Ok(said) => {
            let did = said.get("did").and_then(Value::as_str).unwrap_or("nothing");
            if did != "nothing" {
                tracing::info!(repo, did, "push: {said}");
            }
        }
        Err(e) => tracing::warn!(repo, "push: not told: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A webhook names the commit; the watch names what was last told, so a
    /// commit already tried is not tried again.
    #[test]
    fn a_webhook_names_its_commit_and_the_watch_what_was_told() {
        assert_eq!(
            ask("battle-cities", Some("a744abc8"), Some("0ld")),
            json!({"repo": "battle-cities", "commit": "a744abc8"})
        );
        assert_eq!(
            ask("battle-cities", None, Some("a744abc8")),
            json!({"repo": "battle-cities", "attempted": "a744abc8"})
        );
        assert_eq!(
            ask("battle-cities", None, None),
            json!({"repo": "battle-cities", "attempted": null})
        );
    }

    #[test]
    fn each_repository_is_remembered_at_the_commit_it_was_last_told() {
        let told = Told::default();
        assert_eq!(told.at("battle-cities"), None);
        told.told("battle-cities", "a744abc8");
        told.told("yenbot", "4f1c");
        told.told("battle-cities", "66658636");
        assert_eq!(told.at("battle-cities").as_deref(), Some("66658636"));
        assert_eq!(told.at("yenbot").as_deref(), Some("4f1c"));
    }
}
