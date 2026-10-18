//! Which agent runs a job (§11.3).
//!
//! A conversation's jobs go to the agent that last ran them — the holder of
//! its lineage key — which keeps incremental builds warm. While that agent has
//! no free job slot the job waits, up to [`SPILL_AFTER`], and then spills to
//! the **best idle agent**: the one with the most idle cores,
//! `cores × (1 − load / cores)`, among the agents that hold the job's
//! repository (with push rights, for a job that pushes) and its toolchain.
//! A job with no holder goes to the best idle agent at once, and waits like
//! the rest when every agent is busy.
//!
//! **An agent at low priority** — one sharing its host with a model member
//! (§11.5) — is chosen only when no dedicated agent fit for the job has a
//! free slot: it is never the best idle agent beside one, and a lineage it
//! holds moves to a free dedicated agent rather than staying on it.

use std::cmp::Reverse;
use std::time::Duration;

use super::registry::MemberView;

/// How long a job waits for its busy holder before it spills.
pub const SPILL_AFTER: Duration = Duration::from_secs(5);

/// The header a job's lineage travels in: a conversation's id, a fork
/// carrying its parent's.
pub const LINEAGE_HEADER: &str = "x-tokera-lineage";

/// The header naming the repository a job runs in.
pub const REPO_HEADER: &str = "x-tokera-repo";

/// The header naming the program a job starts; absent for a git job.
pub const TOOLCHAIN_HEADER: &str = "x-tokera-toolchain";

/// The header a job that pushes carries, set to `1`.
pub const PUSH_HEADER: &str = "x-tokera-push";

/// The header addressing one member: an admin's on any request, and a
/// member's on a job stream, reading back a job from the agent that ran it —
/// which the gateway names in the same header on the job's answer.
pub const NODE_HEADER: &str = "x-tokera-node";

/// What a job needs of the agent that runs it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Need {
    pub repo: String,
    /// The program it starts; `None` for a git job.
    pub toolchain: Option<String>,
    /// It pushes to the repository's origin.
    pub push: bool,
}

/// The outcome of one look at the agents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Choice {
    Run(u64),
    /// Every agent fit for it is busy: look again shortly.
    Wait,
    /// No ready agent holds its repository and toolchain.
    NoAgent,
}

/// Whether `agent` can run a job needing `need`.
fn fits(agent: &MemberView, need: &Need) -> bool {
    let holds = agent
        .repos
        .iter()
        .any(|r| r.name == need.repo && (r.push || !need.push));
    let has = need.toolchain.as_ref().is_none_or(|t| {
        agent
            .capacity
            .as_ref()
            .is_some_and(|c| c.toolchains.contains(t))
    });
    holds && has
}

/// A free slot the jobs handed to it since its last `CAPACITY` have not
/// already taken.
fn free(agent: &MemberView) -> bool {
    agent
        .capacity
        .as_ref()
        .is_some_and(|c| c.slots_free > agent.handed)
}

/// Its idle cores, less one for each job handed to it since its last
/// `CAPACITY`, which that report does not yet show running.
fn idle_cores(agent: &MemberView) -> i64 {
    agent.capacity.as_ref().map_or(i64::MIN, |c| {
        c.idle_cores_milli() - 1_000 * i64::from(agent.handed)
    })
}

fn dedicated(agent: &MemberView) -> bool {
    agent.capacity.as_ref().is_some_and(|c| !c.low_priority)
}

/// Choose among `ready` agents for a job needing `need`, whose lineage
/// `holder` holds; `waited` once it has waited [`SPILL_AFTER`].
pub fn choose(ready: &[&MemberView], holder: Option<u64>, need: &Need, waited: bool) -> Choice {
    let fit: Vec<&MemberView> = ready.iter().copied().filter(|a| fits(a, need)).collect();
    if fit.is_empty() {
        return Choice::NoAgent;
    }
    let dedicated_free = fit.iter().any(|a| dedicated(a) && free(a));
    if let Some(h) = holder.and_then(|h| fit.iter().find(|a| a.node == h)) {
        if free(h) && (dedicated(h) || !dedicated_free) {
            return Choice::Run(h.node);
        }
        if !waited && dedicated(h) {
            return Choice::Wait;
        }
    }
    let best = |among: &mut dyn Iterator<Item = &&MemberView>| {
        among
            .max_by_key(|a| (dedicated(a), idle_cores(a), Reverse(a.node)))
            .map(|a| a.node)
    };
    if let Some(node) = best(&mut fit.iter().filter(|a| free(a))) {
        return Choice::Run(node);
    }
    if !waited {
        return Choice::Wait;
    }
    // Every agent is still busy: it queues on the one with the most idle
    // cores.
    best(&mut fit.iter()).map_or(Choice::NoAgent, Choice::Run)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{BuildId, Capacity, MemberState, RepoAd};

    fn agent(node: u64, cores: u16, load_milli: u32, slots_free: u32) -> MemberView {
        MemberView {
            node,
            host: format!("a{node}"),
            app: "agent".into(),
            build: BuildId {
                count: 1,
                hash: "h".into(),
                local: String::new(),
            },
            weight: 1,
            session: node,
            repos: vec![
                RepoAd {
                    name: "candle".into(),
                    origin: "git@h:candle.git".into(),
                    push: true,
                },
                RepoAd {
                    name: "site".into(),
                    origin: "git@h:site.git".into(),
                    push: false,
                },
            ],
            capacity: Some(Capacity {
                state: MemberState::Ready,
                slots_free,
                cores,
                load_milli,
                toolchains: vec!["cargo".into()],
                ..Capacity::default()
            }),
            routes: None,
            high_water: 0,
            lagging: false,
            syncing: false,
            in_flight: 0,
            handed: 0,
            closing: false,
        }
    }

    fn need(repo: &str, toolchain: Option<&str>, push: bool) -> Need {
        Need {
            repo: repo.into(),
            toolchain: toolchain.map(str::to_string),
            push,
        }
    }

    fn cargo() -> Need {
        need("candle", Some("cargo"), false)
    }

    /// A lineage's job goes back to its holder while the holder has a slot,
    /// even with an idler agent beside it.
    #[test]
    fn a_job_sticks_to_the_agent_that_last_ran_its_lineage() {
        let (a, b) = (agent(1, 4, 3_000, 1), agent(2, 32, 0, 4));
        assert_eq!(choose(&[&a, &b], Some(1), &cargo(), false), Choice::Run(1));
    }

    /// A busy holder makes the job wait, and after the wait it spills to the
    /// best idle agent.
    #[test]
    fn a_job_waits_for_its_busy_holder_then_spills() {
        let (a, b, c) = (
            agent(1, 8, 0, 0),
            agent(2, 16, 8_000, 2),
            agent(3, 8, 1_000, 2),
        );
        assert_eq!(
            choose(&[&a, &b, &c], Some(1), &cargo(), false),
            Choice::Wait
        );
        // b has 16 − 8 = 8 idle cores, c 8 − 1 = 7.
        assert_eq!(
            choose(&[&a, &b, &c], Some(1), &cargo(), true),
            Choice::Run(2)
        );
    }

    /// Without a holder the job goes to the most idle cores at once; ties go
    /// to the lower node.
    #[test]
    fn the_best_idle_agent_has_the_most_idle_cores() {
        let (a, b, c) = (
            agent(5, 8, 2_000, 1),
            agent(6, 12, 6_500, 1),
            agent(4, 6, 0, 1),
        );
        // 6000, 5500, 6000 idle: 4 and 5 tie, the lower node wins.
        assert_eq!(choose(&[&a, &b, &c], None, &cargo(), false), Choice::Run(4));
    }

    /// **Jobs handed since an agent's last `CAPACITY` count against it**: a
    /// slot one of them took is not free, and each takes a core off its idle
    /// ones — so a burst spreads instead of all reading the same report.
    #[test]
    fn jobs_handed_since_the_last_report_count_against_an_agent() {
        let mut a = agent(1, 8, 0, 1);
        let b = agent(2, 8, 0, 1);
        assert_eq!(choose(&[&a, &b], None, &cargo(), false), Choice::Run(1));
        a.handed = 1;
        assert_eq!(
            choose(&[&a, &b], None, &cargo(), false),
            Choice::Run(2),
            "a's one slot is taken"
        );
        let (mut c, d) = (agent(3, 8, 0, 4), agent(4, 8, 2_500, 4));
        c.handed = 3;
        // c: 8000 − 3000 = 5000 idle; d: 8000 − 2500 = 5500.
        assert_eq!(choose(&[&c, &d], None, &cargo(), false), Choice::Run(4));
    }

    /// Only agents holding the repository, the push right when it pushes, and
    /// the toolchain are considered.
    #[test]
    fn repository_push_and_toolchain_filter_the_agents() {
        let mut plain = agent(1, 64, 0, 4);
        plain.repos.retain(|r| r.name == "site");
        let fit = agent(2, 2, 1_500, 1);
        assert_eq!(
            choose(&[&plain, &fit], None, &cargo(), false),
            Choice::Run(2)
        );
        assert_eq!(
            choose(
                &[&plain, &fit],
                None,
                &need("candle", Some("npm"), false),
                false
            ),
            Choice::NoAgent
        );
        assert_eq!(
            choose(&[&plain, &fit], None, &need("site", None, true), false),
            Choice::NoAgent
        );
        assert_eq!(
            choose(&[&plain, &fit], None, &need("site", None, false), false),
            Choice::Run(1)
        );
        assert_eq!(
            choose(&[&plain, &fit], None, &need("candle", None, true), false),
            Choice::Run(2)
        );
    }

    /// An agent at low priority.
    fn beside(node: u64, cores: u16, load_milli: u32, slots_free: u32) -> MemberView {
        let mut a = agent(node, cores, load_milli, slots_free);
        a.capacity.as_mut().unwrap().low_priority = true;
        a
    }

    /// **An agent at low priority is chosen only when no dedicated agent can
    /// take the job**: never the best idle agent beside a free dedicated one,
    /// however idler; taken at once when the dedicated ones are full; and a
    /// lineage it holds moves to a free dedicated agent.
    #[test]
    fn a_low_priority_agent_is_chosen_last() {
        let (gpu, a) = (beside(1, 32, 0, 4), agent(2, 4, 3_000, 1));
        assert_eq!(choose(&[&gpu, &a], None, &cargo(), false), Choice::Run(2));
        assert_eq!(
            choose(&[&gpu, &a], Some(1), &cargo(), false),
            Choice::Run(2)
        );
        let full = agent(2, 4, 3_000, 0);
        assert_eq!(
            choose(&[&gpu, &full], None, &cargo(), false),
            Choice::Run(1)
        );
        assert_eq!(
            choose(&[&gpu, &full], Some(1), &cargo(), false),
            Choice::Run(1)
        );
        // A dedicated holder that is busy is waited for; a low-priority one is
        // not, when a dedicated agent is free.
        assert_eq!(
            choose(&[&gpu, &full], Some(2), &cargo(), false),
            Choice::Wait
        );
        assert_eq!(
            choose(&[&gpu, &full], Some(2), &cargo(), true),
            Choice::Run(1)
        );
        // Every agent full: a job queues on a dedicated one first.
        let gpu_full = beside(1, 32, 0, 0);
        assert_eq!(
            choose(&[&gpu_full, &full], None, &cargo(), true),
            Choice::Run(2)
        );
    }

    /// With every fit agent busy a job without a holder waits too, then
    /// queues on the most idle; a holder that no longer fits is passed over.
    #[test]
    fn all_busy_waits_then_queues_on_the_most_idle() {
        let (a, b) = (agent(1, 8, 4_000, 0), agent(2, 8, 1_000, 0));
        assert_eq!(choose(&[&a, &b], None, &cargo(), false), Choice::Wait);
        assert_eq!(choose(&[&a, &b], None, &cargo(), true), Choice::Run(2));
        let mut gone = agent(3, 8, 0, 4);
        gone.repos.clear();
        assert_eq!(
            choose(&[&gone, &b], Some(3), &cargo(), true),
            Choice::Run(2)
        );
        assert_eq!(choose(&[], Some(3), &cargo(), true), Choice::NoAgent);
    }
}
