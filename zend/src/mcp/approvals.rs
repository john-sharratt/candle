//! Approvals: a call a conversation of Zen's may not make alone, kept until a
//! person grants or refuses it (`docs/zen_tool_reach.md` §5).
//!
//! A Decide command, a stop, a delete — and a step the deploy guard refuses —
//! is not run when Zen asks for it. It is recorded here, exactly as asked, and
//! the call answers `needs_approval` with the record's id. The grant comes from
//! outside the conversation: the caller's `approval_grant`, with the user's
//! `confirm`, for a task or chat it started; or the person's own signed-in
//! browser, for a dialogue. Nothing a turn writes grants anything.
//!
//! A record is the call — its tool and arguments — the conversation that asked
//! and why it is held. A grant runs that call once, as recorded, as its
//! granter; changing an argument is a new approval. An approval not settled
//! within [`LIFETIME_MS`] expires and never runs.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::records::Records;

/// How long an approval waits to be settled.
pub const LIFETIME_MS: u64 = 24 * 3_600_000;

/// What an approval's answer starts with, which the GUI draws as one.
pub const NEEDS_APPROVAL: &str = "needs_approval";

/// Where an approval stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum State {
    Asked,
    Granted,
    Refused,
}

/// One call held for a person's approval.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Approval {
    pub id: String,
    /// The key of the person whose approval it waits on.
    pub owner: String,
    /// The conversation that asked.
    pub conversation: String,
    pub tool: String,
    pub args: Value,
    /// Why the call is held rather than run.
    pub why: String,
    pub state: State,
    pub asked_ms: u64,
}

impl Approval {
    /// Whether it is past its time unsettled, at `now_ms`.
    pub fn expired(&self, now_ms: u64) -> bool {
        self.state == State::Asked && now_ms.saturating_sub(self.asked_ms) > LIFETIME_MS
    }

    /// What the conversation that asked is told: the call was not run, why,
    /// and how it will hear.
    pub fn asked(&self) -> String {
        format!(
            "{NEEDS_APPROVAL} {}: `{}` was not run — {}. It is held as approval {} until a person \
             grants or refuses it. Say in your answer what you asked to run and why, and carry on \
             with what does not depend on it; you are told the result when it is granted.",
            self.id, self.tool, self.why, self.id
        )
    }
}

fn key(owner: &str, id: &str) -> String {
    format!("mcp/approval/{owner}/{id}")
}

/// The id of a call asked at `now_ms` by `conversation`: 12 hex digits.
fn id_of(conversation: &str, tool: &str, args: &Value, now_ms: u64) -> String {
    let digest = Sha256::digest(format!("{conversation}\0{tool}\0{args}\0{now_ms}").as_bytes());
    digest[..6].iter().map(|b| format!("{b:02x}")).collect()
}

/// Hold `tool` with `args`, asked by `conversation` at `now_ms`, for `owner`'s
/// approval.
pub fn ask(
    records: &dyn Records,
    owner: &str,
    conversation: &str,
    tool: &str,
    args: &Value,
    why: &str,
    now_ms: u64,
) -> anyhow::Result<Approval> {
    let approval = Approval {
        id: id_of(conversation, tool, args, now_ms),
        owner: owner.to_string(),
        conversation: conversation.to_string(),
        tool: tool.to_string(),
        args: args.clone(),
        why: why.to_string(),
        state: State::Asked,
        asked_ms: now_ms,
    };
    records.put(&key(owner, &approval.id), serde_json::to_value(&approval)?)?;
    Ok(approval)
}

/// `owner`'s approval `id`.
pub fn get(records: &dyn Records, owner: &str, id: &str) -> Option<Approval> {
    records
        .get(&key(owner, id))
        .and_then(|v| serde_json::from_value(v).ok())
}

/// Settle `owner`'s approval `id` at `now_ms`: granted or refused, once. A
/// grant answers the call to run, exactly as it was asked; an approval already
/// settled, expired or unknown is refused with why.
pub fn settle(
    records: &dyn Records,
    owner: &str,
    id: &str,
    grant: bool,
    now_ms: u64,
) -> Result<Approval, String> {
    let Some(mut approval) = get(records, owner, id) else {
        return Err(format!("there is no approval `{id}` of yours"));
    };
    if approval.state != State::Asked {
        return Err(format!(
            "approval {id} is already {}: an approval runs once",
            match approval.state {
                State::Granted => "granted",
                _ => "refused",
            }
        ));
    }
    if approval.expired(now_ms) {
        return Err(format!(
            "approval {id} expired unsettled: ask for the call again"
        ));
    }
    approval.state = if grant {
        State::Granted
    } else {
        State::Refused
    };
    let value = serde_json::to_value(&approval).map_err(|e| e.to_string())?;
    records
        .put(&key(owner, id), value)
        .map_err(|e| e.to_string())?;
    Ok(approval)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::mcp::records::MemRecords;

    const NOW: u64 = 1_800_000_000_000;

    fn asked(r: &MemRecords) -> Approval {
        ask(
            r,
            "o",
            "c1",
            "cluster_command",
            &json!({"member": "a0ac88178cb2b57d", "name": "restart"}),
            "`restart` is a decision",
            NOW,
        )
        .unwrap()
    }

    /// **An approval holds the call exactly as asked, and runs once**: a
    /// grant answers it, and a second settle of either kind is refused.
    #[test]
    fn an_approval_holds_its_call_and_settles_once() {
        let r = MemRecords::default();
        let a = asked(&r);
        assert_eq!(a.id.len(), 12);
        assert_eq!(get(&r, "o", &a.id).unwrap(), a);
        let granted = settle(&r, "o", &a.id, true, NOW + 1).unwrap();
        assert_eq!(granted.state, State::Granted);
        assert_eq!(granted.args, a.args);
        assert!(settle(&r, "o", &a.id, true, NOW + 2)
            .unwrap_err()
            .contains("already granted"));
        assert!(settle(&r, "o", &a.id, false, NOW + 2)
            .unwrap_err()
            .contains("runs once"));
    }

    /// **Only its owner settles it, and an expired one never runs.**
    #[test]
    fn another_persons_or_an_expired_approval_is_refused() {
        let r = MemRecords::default();
        let a = asked(&r);
        assert!(settle(&r, "someone", &a.id, true, NOW)
            .unwrap_err()
            .contains("no approval"));
        assert!(settle(&r, "o", &a.id, true, NOW + LIFETIME_MS + 1)
            .unwrap_err()
            .contains("expired"));
        let refused = settle(&r, "o", &asked(&r).id, false, NOW).unwrap();
        assert_eq!(refused.state, State::Refused);
    }

    /// **The asker is told it was not run, why, and that it is held**, under
    /// the word the GUI draws an approval for.
    #[test]
    fn the_asker_is_told_its_call_is_held() {
        let a = asked(&MemRecords::default());
        let told = a.asked();
        assert!(told.starts_with(&format!(
            "{NEEDS_APPROVAL} {}: `cluster_command` was not run",
            a.id
        )));
        assert!(told.contains("`restart` is a decision"));
    }
}
