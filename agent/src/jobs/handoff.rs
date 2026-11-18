//! A deploy's handoff (`docs/build_agent.md` §7): once a deploy's run has
//! staged its binary, the agent asks its own machine's keep for the trial,
//! before it gives back the build slot, so a restart that waits for the run
//! waits for the handoff too, and no zend member has to be alive for it.

use serde_json::json;

use crate::keep::post;
use crate::wire::Tried;

use super::Jobs;

/// keep's local route that takes a trial as a job.
pub const TRIAL: &str = "/trial";

/// The id a deploy of run `id` on agent `node` goes by on keep's board: what
/// `service_deploy` answered as its handle, without the `deploy:` kind.
pub fn deploy_id(node: u64, id: &str) -> String {
    format!("{node:016x}:{id}")
}

impl Jobs {
    /// Ask keep for the trial of `artifact` as `service`, for run `id`.
    pub(super) async fn hand_off(&self, id: &str, service: &str, artifact: &str) -> Tried {
        let Some(node) = self.node() else {
            return Tried::Refused(
                "this agent has not joined a cluster, so its deploy has no id".into(),
            );
        };
        let deploy = deploy_id(node, id);
        let body = json!({"service": service, "artifact": artifact, "deploy": deploy});
        let tried = match post(TRIAL, body).await {
            Ok(answer) if answer.status == 202 => Tried::Asked,
            Ok(answer) => Tried::Refused(format!("{}: {}", answer.status, answer.body.trim())),
            Err(e) => Tried::Refused(e),
        };
        match &tried {
            Tried::Asked => {
                tracing::info!(deploy, service, artifact, "deploy: keep takes the trial")
            }
            Tried::Refused(why) => {
                tracing::warn!(
                    deploy,
                    service,
                    artifact,
                    "deploy: keep refused the trial: {why}"
                )
            }
        }
        tried
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_deploy_is_named_by_its_agent_and_run() {
        assert_eq!(deploy_id(0xab, "Xe"), "00000000000000ab:Xe");
    }
}
