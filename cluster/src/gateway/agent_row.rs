//! The agents a member can reach, as a gateway lists them (§11.7).
//!
//! `GET` [`AGENTS_PATH`] on a stream towards `group:agent` is answered by the
//! gateway itself, from its registry, rather than sent on to an agent: which
//! agents are joined, what each holds and how loaded it is, is the gateway's
//! to know. What each agent is running is then asked of that agent, addressed
//! by its node.

use serde::{Deserialize, Serialize};

use super::registry::MemberView;
use super::status::state_name;
use crate::frame::BuildId;

/// The path a member lists the agents at.
pub const AGENTS_PATH: &str = "/v1/agents";

/// A repository an agent holds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentRepo {
    pub name: String,
    pub origin: String,
    /// It holds the credentials to push to the origin.
    pub push: bool,
}

/// One agent: who it is, what it holds, and its last `CAPACITY`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentRow {
    /// Its node, as sixteen hex digits — what a request addressing it names.
    pub node: String,
    pub host: String,
    pub build: BuildId,
    /// `ready`, `disabled`, `loading`, … ([`state_name`]).
    pub state: String,
    pub os: String,
    pub cores: u16,
    /// The host's load average × 1000.
    pub load_milli: u32,
    pub mem_free_mib: u32,
    pub slots_free: u32,
    /// Jobs running.
    pub jobs: u32,
    pub toolchains: Vec<String>,
    /// It shares its host with a model member, and runs its jobs below
    /// normal priority (§11.5).
    pub low_priority: bool,
    pub repos: Vec<AgentRepo>,
    /// When its socket to this gateway connected, in Unix milliseconds.
    pub connected_ms: u64,
}

impl AgentRow {
    /// `agent` as listed, its socket connected at `connected_ms`.
    pub fn of(agent: &MemberView, connected_ms: u64) -> AgentRow {
        let cap = agent.capacity.clone().unwrap_or_default();
        AgentRow {
            node: format!("{:016x}", agent.node),
            host: agent.host.clone(),
            build: agent.build.clone(),
            state: state_name(agent.state()).to_string(),
            os: cap.os,
            cores: cap.cores,
            load_milli: cap.load_milli,
            mem_free_mib: cap.mem_free_mib,
            slots_free: cap.slots_free,
            jobs: cap.jobs,
            toolchains: cap.toolchains,
            low_priority: cap.low_priority,
            repos: agent
                .repos
                .iter()
                .map(|r| AgentRepo {
                    name: r.name.clone(),
                    origin: r.origin.clone(),
                    push: r.push,
                })
                .collect(),
            connected_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::frame::{Capacity, MemberState, RepoAd};

    /// **An agent is listed as its JSON**: its node in hex, its state by name,
    /// its capacity and its repositories — and one that has sent no
    /// `CAPACITY` yet is listed loading, with nothing measured.
    #[test]
    fn an_agent_is_listed_as_its_json() {
        let mut agent = MemberView {
            node: 0xa1,
            host: "build-1".into(),
            app: "agent".into(),
            build: BuildId {
                count: 7,
                hash: "abc".into(),
                local: String::new(),
            },
            weight: 1,
            session: 1,
            repos: vec![RepoAd {
                name: "candle".into(),
                origin: "git@h:candle.git".into(),
                push: true,
            }],
            capacity: Some(Capacity {
                state: MemberState::Disabled,
                slots_free: 3,
                cores: 16,
                load_milli: 2_500,
                os: "linux".into(),
                mem_free_mib: 9_000,
                jobs: 1,
                toolchains: vec!["cargo".into()],
                ..Capacity::default()
            }),
            routes: None,
            high_water: 0,
            lagging: false,
        };
        assert_eq!(
            serde_json::to_value(AgentRow::of(&agent, 1_700_000_000_000)).unwrap(),
            json!({
                "node": "00000000000000a1", "host": "build-1",
                "build": { "count": 7, "hash": "abc", "local": "" },
                "state": "disabled", "os": "linux", "cores": 16, "load_milli": 2500,
                "mem_free_mib": 9000, "slots_free": 3, "jobs": 1, "toolchains": ["cargo"],
                "low_priority": false,
                "repos": [{ "name": "candle", "origin": "git@h:candle.git", "push": true }],
                "connected_ms": 1_700_000_000_000u64
            })
        );
        agent.capacity = None;
        let row = AgentRow::of(&agent, 0);
        assert_eq!(
            (row.state.as_str(), row.cores, row.os.as_str()),
            ("loading", 0, "")
        );
    }
}
