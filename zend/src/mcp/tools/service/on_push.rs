//! Deploying on a push (`docs/battlecities_app.md` §4.4): what a gateway's
//! word that a repository's `main` moved comes to.
//!
//! The member reads origin itself, so the word needs no commit, and it is
//! safe to hear twice:
//!
//! | Origin's `main` is | What is done |
//! |---|---|
//! | the commit the gateway says it was last told at | nothing: it was tried, whatever came of it |
//! | the commit the machine's installed binary was built from | nothing |
//! | anything else | `service_deploy` of each service built from the repository, which answers with the deploy already under way for that commit or replaces a build of an older one ([`super::pending`]) |
//!
//! Only the services in [`ON_PUSH`] are deployed so: the daemon, the gateway
//! and the connector stop models or the cluster's front door, and stay a
//! person's decision. The deploy acts as nobody: keep answers the member a
//! request came from, and the build is of `origin/main` or keep refuses it.

use std::sync::Arc;

use axum::http::Method;
use host::machines::machines;
use serde_json::{json, Value};

use super::deploy::{default_host, origin_tip, run};
use super::{keep, repo_of, BATTLECITIES, KEEP_STATE};
use crate::mcp::ops_tools::OpsCall;
use crate::mcp::tools::repo_job::NO_AGENTS;
use crate::session::ZendSession;

/// The services deployed when their repository's `main` moves.
pub const ON_PUSH: [&str; 1] = [BATTLECITIES];

/// The most seconds a deploy started by a push waits on anything.
const CEILING_S: u64 = 30;

/// The services deployed on a push of `repo`.
pub fn services_of(repo: &str) -> Vec<&'static str> {
    ON_PUSH
        .iter()
        .copied()
        .filter(|service| repo_of(service) == repo)
        .collect()
}

/// The commit `service`'s installed binary says it was built from, in keep's
/// `state`: its short hash.
pub fn installed<'a>(state: &'a Value, service: &str) -> Option<&'a str> {
    state
        .get("services")?
        .as_array()?
        .iter()
        .find(|s| s.get("name").and_then(Value::as_str) == Some(service))?
        .get("build")?
        .get("hash")?
        .as_str()
}

/// Whether `service` is to be deployed at origin's `tip`: not when the
/// gateway was last told at it, and not when the installed binary is of it.
pub fn due(tip: &str, attempted: Option<&str>, installed: Option<&str>) -> bool {
    let tried = attempted == Some(tip);
    let current = installed.is_some_and(|hash| !hash.is_empty() && tip.starts_with(hash));
    !tried && !current
}

/// Act on the word that `repo`'s `main` moved, the gateway having last told
/// of it at `attempted`: what was done, as the gateway is answered.
pub async fn pushed(
    session: Arc<ZendSession>,
    repo: &str,
    attempted: Option<&str>,
) -> Result<Value, String> {
    let services = services_of(repo);
    if services.is_empty() {
        return Err(format!("nothing is deployed on a push of `{repo}`"));
    }
    let agents = session.agents().ok_or(NO_AGENTS)?;
    let tip = origin_tip(&agents, repo)
        .await
        .ok_or_else(|| format!("origin's main of `{repo}` could not be read"))?;
    if attempted == Some(tip.as_str()) {
        return Ok(json!({"repo": repo, "commit": tip, "did": "nothing"}));
    }
    let listed = agents
        .listing()
        .await
        .map_err(|e| format!("the agents could not be listed: {e}"))?;
    let host = default_host(&machines(), listed.iter().map(|a| a.host.as_str()))?;
    let call = OpsCall {
        session,
        lineage: format!("push/{repo}"),
        caller: None,
        progress: None,
        ceiling: CEILING_S,
        worker: false,
    };
    let state = keep(&call, Method::GET, KEEP_STATE, &host, None).await?;
    let mut deployed = Vec::new();
    for service in services {
        if !due(&tip, attempted, installed(&state, service)) {
            continue;
        }
        let answer = run(&call, &json!({"service": service, "host": host})).await;
        tracing::info!(
            repo,
            service,
            host,
            commit = %tip,
            refused = answer.is_error,
            "push: {}",
            answer.text
        );
        deployed.push(json!({
            "service": service,
            "answer": answer.text,
            "refused": answer.is_error,
        }));
    }
    let did = match deployed.is_empty() {
        true => "nothing",
        false => "deploy",
    };
    Ok(json!({"repo": repo, "commit": tip, "host": host, "did": did, "services": deployed}))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TIP: &str = "22e9ecf8d1f0c4b7e9a3552f6c1d8e0b9a7f6e5d";

    /// **Only the game is deployed on a push**, and only of its own
    /// repository.
    #[test]
    fn the_game_is_what_a_push_of_its_repository_deploys() {
        assert_eq!(services_of("battle-cities"), ["battlecities"]);
        assert_eq!(services_of("candle"), Vec::<&str>::new());
        assert_eq!(services_of("yenbot"), Vec::<&str>::new());
    }

    /// **A commit is deployed once**: not when the gateway already told of
    /// it, and not when the installed binary was built from it.
    #[test]
    fn a_commit_already_tried_or_installed_is_not_deployed_again() {
        assert!(due(TIP, None, None));
        assert!(due(TIP, Some("0ld"), Some("a744abc8d")));
        assert!(!due(TIP, Some(TIP), Some("a744abc8d")));
        assert!(!due(TIP, None, Some("22e9ecf8d")));
        // A binary that names no commit is of none.
        assert!(due(TIP, None, Some("")));
    }

    #[test]
    fn the_installed_build_is_read_from_keeps_state() {
        let state = json!({"services": [
            {"name": "web", "build": {"count": 4900, "hash": "c0325ac76", "local": ""}},
            {"name": "battlecities", "build": {"count": 812, "hash": "22e9ecf8d", "local": ""}},
            {"name": "mcp", "build": null},
        ]});
        assert_eq!(installed(&state, "battlecities"), Some("22e9ecf8d"));
        assert_eq!(installed(&state, "mcp"), None);
        assert_eq!(installed(&state, "motherd"), None);
        assert_eq!(installed(&json!({}), "battlecities"), None);
    }
}
