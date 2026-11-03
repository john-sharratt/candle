//! The `repo_job_*` tools (`docs/build_deploy_tools.md`): a program run on a
//! build agent — a cluster machine that checks out a repository and runs
//! programs on it, never one serving traffic — against a branch as origin holds
//! it, and everything a caller does with it after: watch it, read its log,
//! list it, cancel it. Admins only.
//!
//! | Module | Tool |
//! |---|---|
//! | [`start`] | `repo_job_start` |
//! | [`status`] | `repo_job_status` |
//! | [`log`] | `repo_job_log` |
//! | [`list`] | `repo_job_list` |
//! | [`cancel`] | `repo_job_cancel` |
//!
//! A job is named by one handle, `<node>:<id>` ([`Handle`]), the way every
//! answer writes it and every tool takes it back (§1).

use std::fmt;

use agent::client::{Agents, At};
use agent::wire::{JobView, Phase, Staged};
use cluster::gateway::AgentRow;
use zend_vfs::JobStatus;

use crate::mcp::ops;

pub mod cancel;
pub mod list;
pub mod log;
pub mod start;
pub mod status;

/// What a member with no way to any agent answers.
pub const NO_AGENTS: &str = "this member reaches no agent: it has no cluster and no agent of its own";

/// A job: the agent it runs on and its id there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Handle {
    pub node: u64,
    pub id: String,
}

impl Handle {
    pub fn new(node: u64, id: &str) -> Handle {
        Handle {
            node,
            id: id.to_string(),
        }
    }

    /// `text` as a handle: `<16 hex digits>:<id>`.
    pub fn parse(text: &str) -> Result<Handle, String> {
        let text = text.trim();
        let bad = || {
            format!(
                "`{text}` is not a job: a job is `<node>:<id>` as the answer that started it wrote \
                 it, e.g. `76c3a0f4d39bd9db:XeV_I_QPzpg` — `repo_job_list` lists them"
            )
        };
        let (node, id) = text.split_once(':').ok_or_else(bad)?;
        let node = ops::node(node).map_err(|_| bad())?;
        let valid = !id.is_empty()
            && id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_');
        if !valid {
            return Err(bad());
        }
        Ok(Handle {
            node: u64::from_str_radix(&node, 16).expect("sixteen hex digits"),
            id: id.to_string(),
        })
    }

    /// Where its agent is reached from `agents`.
    pub fn at(&self, agents: &Agents) -> At {
        agents.at(self.node)
    }
}

impl fmt::Display for Handle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:016x}:{}", self.node, self.id)
    }
}

/// An agent's node as an answer writes it.
pub fn place(node: u64) -> String {
    format!("{node:016x}")
}

/// The node of the agent on machine `host` among `agents`: a ready one first.
pub fn on_host(agents: &[AgentRow], host: &str) -> Result<u64, String> {
    let on: Vec<&AgentRow> = agents
        .iter()
        .filter(|a| a.host.eq_ignore_ascii_case(host.trim()))
        .collect();
    let Some(agent) = on.iter().find(|a| a.state == "ready").or(on.first()) else {
        let known: Vec<&str> = agents.iter().map(|a| a.host.as_str()).collect();
        return Err(format!(
            "no agent on {host} has a socket on the gateway; agents are on: {}",
            if known.is_empty() {
                "none".to_string()
            } else {
                known.join(", ")
            }
        ));
    };
    u64::from_str_radix(&agent.node, 16).map_err(|_| format!("{:?} is not a node", agent.node))
}

/// The agent `pin` names — a node id, or the machine it runs on — among
/// `listed`.
pub fn pinned(listed: &[AgentRow], pin: &str) -> Result<u64, String> {
    match ops::node(pin) {
        Ok(n) => Ok(u64::from_str_radix(&n, 16).expect("sixteen hex digits")),
        Err(_) => on_host(listed, pin),
    }
}

/// The machine agent `node` runs on, as `listed` names it.
pub fn host_in(listed: &[AgentRow], node: u64) -> String {
    listed
        .iter()
        .find(|a| a.node == place(node))
        .map_or_else(|| format!("agent {}", place(node)), |a| a.host.clone())
}

/// The machine agent `node` runs on, asking the gateway; its node when the
/// agents cannot be listed.
pub async fn host_of(agents: &Agents, node: u64) -> String {
    match agents.listing().await {
        Ok(listed) => host_in(&listed, node),
        Err(_) => format!("agent {}", place(node)),
    }
}

/// How `status` reads.
pub fn state(status: &JobStatus) -> String {
    match status {
        JobStatus::Queued => "queued".into(),
        JobStatus::Running => "running".into(),
        JobStatus::Exited { code: Some(code) } => format!("exited {code}"),
        JobStatus::Exited { code: None } => "killed by a signal".into(),
        JobStatus::TimedOut => "timed out".into(),
        JobStatus::Cancelled => "cancelled".into(),
        JobStatus::Refused { why } => format!("refused ({why})"),
        JobStatus::Failed { why } => format!("failed ({why})"),
    }
}

/// Where a job has got to, naming the jobs it waits behind by handle.
pub fn phase(phase: &Phase, node: u64) -> String {
    match phase {
        Phase::Fetching => "fetching origin".into(),
        Phase::Waiting { behind } if behind.is_empty() => "waiting for its checkout".into(),
        Phase::Waiting { behind } => format!(
            "waiting for its checkout behind {}",
            behind
                .iter()
                .map(|id| Handle::new(node, id).to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Phase::Running => "running".into(),
        Phase::Staging => "staging what it built".into(),
        Phase::Ended => "ended".into(),
    }
}

/// `ms` as a span: `42 s`, `3 min 5 s`, `1 h 2 min`.
pub fn span(ms: u64) -> String {
    let s = ms / 1000;
    match s {
        0..=59 => format!("{s} s"),
        60..=3_599 => format!("{} min {} s", s / 60, s % 60),
        _ => format!("{} h {} min", s / 3600, (s % 3600) / 60),
    }
}

/// The command a job runs, cut at `max` characters.
pub fn command_line(words: &[String], max: usize) -> String {
    let line = words.join(" ");
    match line.char_indices().nth(max) {
        Some((at, _)) => format!("{}…", &line[..at]),
        None => line,
    }
}

/// One job in a line: its handle, machine, command, where it is, how long it
/// has run, its log's length, and what it came to.
pub fn describe(handle: &Handle, host: &str, view: &JobView, now: u64) -> String {
    let ended = view.status.is_finished();
    let ran = view.ended_ms.unwrap_or(now).saturating_sub(view.started_ms);
    let at = if ended {
        state(&view.status)
    } else {
        phase(&view.phase, handle.node)
    };
    let mut out = format!(
        "job {handle} on {host}: `{}` on {} — {at}, {} {}, {} lines",
        command_line(&view.command, 160),
        view.repo,
        if ended { "took" } else { "for" },
        span(ran),
        view.lines
    );
    match &view.staged {
        Some(Staged::At(path)) => out.push_str(&format!("; staged at {path}")),
        Some(Staged::Not(why)) => out.push_str(&format!("; nothing staged: {why}")),
        None => {}
    }
    out
}

/// What a job that could not be read answers: why, and what that means.
pub fn lost(handle: &Handle, e: &impl fmt::Display) -> String {
    format!(
        "job {handle}: {e} — a job lives in its agent's process, and one lost to a restart is \
         gone: start it again"
    )
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn a_handle_is_its_node_and_its_id() {
        let h = Handle::parse(" 76C3A0F4D39BD9DB:XeV_I_QPzpg ").unwrap();
        assert_eq!(h, Handle::new(0x76c3a0f4d39bd9db, "XeV_I_QPzpg"));
        assert_eq!(h.to_string(), "76c3a0f4d39bd9db:XeV_I_QPzpg");
        for bad in ["XeV_I_QPzpg", "76c3:XeV", "76c3a0f4d39bd9db:", "76c3a0f4d39bd9db:a b"] {
            let err = Handle::parse(bad).unwrap_err();
            assert!(err.contains("`<node>:<id>`"), "{bad}: {err}");
        }
    }

    fn row(host: &str, node: u64, state: &str) -> AgentRow {
        serde_json::from_value(json!({
            "node": place(node), "host": host,
            "build": {"count": 1, "hash": "h", "local": ""},
            "state": state, "os": "windows", "cores": 8, "load_milli": 0,
            "mem_free_mib": 0, "slots_free": 1, "jobs": 0, "toolchains": ["cargo"],
            "low_priority": false, "standby": host == "alienworld", "repos": [],
            "connected_ms": 0
        }))
        .unwrap()
    }

    /// **An agent is pinned by its machine's name**, in any case, a ready one
    /// first; a machine with no agent is refused, naming those there are.
    #[test]
    fn an_agent_is_pinned_by_its_machine() {
        let agents = [
            row("BlackWorld", 0x1, "ready"),
            row("alienworld", 0x2, "disabled"),
            row("alienworld", 0x3, "ready"),
        ];
        assert_eq!(on_host(&agents, "blackworld"), Ok(0x1));
        assert_eq!(on_host(&agents, " AlienWorld "), Ok(0x3));
        assert_eq!(on_host(&agents[..2], "alienworld"), Ok(0x2));
        assert_eq!(pinned(&agents, "00000000000000AB"), Ok(0xab));
        assert_eq!(
            on_host(&agents, "whiteworld").unwrap_err(),
            "no agent on whiteworld has a socket on the gateway; agents are on: BlackWorld, \
             alienworld, alienworld"
        );
        assert_eq!(host_in(&agents, 0x1), "BlackWorld");
        assert_eq!(host_in(&agents, 0x9), "agent 0000000000000009");
    }

    fn view(status: JobStatus, phase: Phase, ended_ms: Option<u64>) -> JobView {
        JobView {
            id: "Xe".into(),
            repo: "candle".into(),
            status,
            lines: 120,
            bytes: 4_000,
            outcome: None,
            pushed: Vec::new(),
            fetched: None,
            listed: None,
            staged: None,
            phase,
            command: vec!["cargo".into(), "test".into(), "-p".into(), "web".into()],
            started_ms: 1_000,
            ended_ms,
        }
    }

    /// **A job reads as its handle, machine, command, phase and time**; one
    /// waiting names the jobs ahead of it by handle, and one ended says how.
    #[test]
    fn a_job_reads_as_where_it_is() {
        let h = Handle::new(0xab, "Xe");
        let waiting = view(
            JobStatus::Queued,
            Phase::Waiting {
                behind: vec!["Yz".into()],
            },
            None,
        );
        assert_eq!(
            describe(&h, "BlackWorld", &waiting, 126_000),
            "job 00000000000000ab:Xe on BlackWorld: `cargo test -p web` on candle — waiting for \
             its checkout behind 00000000000000ab:Yz, for 2 min 5 s, 120 lines"
        );
        let done = JobView {
            staged: Some(Staged::At("D:/keep/stage/web.exe".into())),
            ..view(JobStatus::Exited { code: Some(0) }, Phase::Ended, Some(43_000))
        };
        assert_eq!(
            describe(&h, "BlackWorld", &done, 999_999),
            "job 00000000000000ab:Xe on BlackWorld: `cargo test -p web` on candle — exited 0, took \
             42 s, 120 lines; staged at D:/keep/stage/web.exe"
        );
        assert_eq!(span(3_700_000), "1 h 1 min");
    }

    #[test]
    fn a_long_command_is_cut_on_a_character() {
        let long = vec!["é".repeat(30)];
        let cut = command_line(&long, 10);
        assert_eq!(cut.chars().count(), 11);
        assert!(cut.ends_with('…'));
    }
}
