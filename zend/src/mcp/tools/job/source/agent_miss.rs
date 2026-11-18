//! Why an agent did not answer for a job, and what that means for it
//! (`docs/jobs.md` §6.1).
//!
//! An agent runs inside its machine's daemon, so a trial of a new daemon, an
//! install or a restart takes it away for a while, and its parked queue comes
//! back only once the new process serves (`docs/build_agent.md` §6, §8). For
//! those seconds a job's agent cannot be reached, or answers that it holds no
//! such job because it has not taken its queue back yet. Neither means the job
//! is gone, and on 2026-11-18 and 11-19 both were reported as "lost to a
//! restart … start it again" for deploys keep went on to prove. A job is lost
//! only when an agent that is up and serving says it does not hold it.

use agent::client::{AgentError, Agents};

/// What one failed call to a job's agent says of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Miss {
    /// The agent is away or has not taken its queue back: ask again.
    Restarting,
    /// An agent that is up and serving holds no such job.
    NotHeld,
    /// The agent answered with an error of its own.
    Failed,
}

/// What the error alone says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Said {
    Unreachable,
    NotHeld,
    Failed,
}

fn said(e: &AgentError) -> Said {
    match e {
        AgentError::NoAgent(_) | AgentError::Transport(_) => Said::Unreachable,
        AgentError::Refused {
            status: 502..=504, ..
        } => Said::Unreachable,
        AgentError::Refused { status: 404, .. } => Said::NotHeld,
        AgentError::Refused { .. } => Said::Failed,
    }
}

/// `e`, from the agent whose state the gateway lists as `state` (`None` when
/// it lists no such agent): an agent that is not `ready` has not taken its
/// queue back, so a job it does not hold may be in it.
pub fn miss_of(e: &AgentError, state: Option<&str>) -> Miss {
    match said(e) {
        Said::Unreachable => Miss::Restarting,
        Said::NotHeld if state == Some("ready") => Miss::NotHeld,
        Said::NotHeld => Miss::Restarting,
        Said::Failed => Miss::Failed,
    }
}

/// [`miss_of`], reading agent `node`'s state from the gateway when the error
/// alone does not settle it.
pub async fn miss(agents: &Agents, node: u64, e: &AgentError) -> Miss {
    if said(e) != Said::NotHeld {
        return miss_of(e, None);
    }
    let name = format!("{node:016x}");
    let state = agents
        .listing()
        .await
        .ok()
        .and_then(|rows| rows.into_iter().find(|r| r.node == name).map(|r| r.state));
    miss_of(e, state.as_deref())
}

/// The line a job whose agent is restarting stands at.
pub fn restarting(job: &str, host: &str) -> String {
    format!(
        "{job} — its agent on {host} is restarting, or has not taken its queue back yet; \
         asking again"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refused(status: u16) -> AgentError {
        AgentError::Refused {
            status,
            why: "x".into(),
        }
    }

    /// **Only a serving agent's 404 is a lost job.** An agent that cannot be
    /// reached — no agent, a broken stream, a 502–504 from the gateway — is
    /// restarting; a 404 from one not `ready`, or no longer listed, has not
    /// taken its queue back; any other refusal is the agent's own error.
    #[test]
    fn only_a_serving_agents_404_is_a_lost_job() {
        assert_eq!(
            miss_of(&AgentError::NoAgent("x".into()), None),
            Miss::Restarting
        );
        assert_eq!(
            miss_of(&AgentError::Transport("x".into()), Some("ready")),
            Miss::Restarting
        );
        assert_eq!(miss_of(&refused(503), Some("ready")), Miss::Restarting);
        assert_eq!(miss_of(&refused(502), None), Miss::Restarting);
        assert_eq!(miss_of(&refused(404), Some("ready")), Miss::NotHeld);
        assert_eq!(miss_of(&refused(404), Some("loading")), Miss::Restarting);
        assert_eq!(miss_of(&refused(404), None), Miss::Restarting);
        assert_eq!(miss_of(&refused(500), Some("ready")), Miss::Failed);
        assert_eq!(miss_of(&refused(409), None), Miss::Failed);
    }

    #[test]
    fn the_restarting_line_names_the_job_and_machine() {
        assert_eq!(
            restarting("build:ab:cd", "BlackWorld"),
            "build:ab:cd — its agent on BlackWorld is restarting, or has not taken its queue \
             back yet; asking again"
        );
    }
}
