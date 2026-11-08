//! What an alone task ended with, kept for the person it worked for
//! (`docs/zen_memory_and_guides.md` Phase 4): nobody waited on it, so its end
//! is raised in their next conversation, from any client, by
//! `get_user_context`.
//!
//! The newest [`RING`] are kept per owner, each in a slot of its own under
//! `mcp/outcome/<owner>/`, the next slot named by a count — so a frame reads
//! at most [`RING`] small records, never a thread's history, and reading marks
//! nothing.

use std::cmp::Reverse;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::records::Records;

/// How many outcomes are kept per owner.
pub const RING: u64 = 10;

/// An alone task's end.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Outcome {
    pub at_ms: u64,
    pub thread: String,
    pub name: String,
    /// How it ended, as its thread's note says it: `Task ended:`, `Task stuck:`.
    pub label: String,
    pub text: String,
}

fn count_key(owner: &str) -> String {
    format!("mcp/outcome-count/{owner}")
}

fn prefix(owner: &str) -> String {
    format!("mcp/outcome/{owner}/")
}

/// Keep `outcome` for `owner`, in place of the oldest past [`RING`].
pub fn keep(records: &dyn Records, owner: &str, outcome: &Outcome) -> anyhow::Result<()> {
    let n = records
        .get(&count_key(owner))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    records.put(
        &format!("{}{:02}", prefix(owner), n % RING),
        serde_json::to_value(outcome)?,
    )?;
    records.put(&count_key(owner), Value::from(n + 1))
}

/// `owner`'s newest `n` outcomes, newest first.
pub fn newest(records: &dyn Records, owner: &str, n: usize) -> Vec<Outcome> {
    let mut all: Vec<Outcome> = records
        .under(&prefix(owner))
        .into_iter()
        .filter_map(|(_, v)| serde_json::from_value(v).ok())
        .collect();
    all.sort_by_key(|o| Reverse(o.at_ms));
    all.truncate(n);
    all
}

/// The block a frame carries for `outcomes`: nothing when there are none.
pub fn block(outcomes: &[Outcome]) -> Option<String> {
    if outcomes.is_empty() {
        return None;
    }
    let mut out = "Finished while nobody waited on it — read and verify before relying on it:".to_string();
    for o in outcomes {
        out.push_str(&format!(
            "\n  - T:{}/{} — {} {}",
            o.thread, o.name, o.label, o.text
        ));
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::records::MemRecords;

    fn outcome(at_ms: u64) -> Outcome {
        Outcome {
            at_ms,
            thread: "dj-set".into(),
            name: format!("doc-{at_ms}"),
            label: "Task ended:".into(),
            text: "Wrote docs/midi_dj_set.md; pushed zen/docs-midi-dj-set; the merge waits on you.".into(),
        }
    }

    /// **The newest are kept, the oldest let go past the ring**, and an
    /// owner's are never another's.
    #[test]
    fn the_newest_are_kept_past_the_ring() {
        let r = MemRecords::default();
        for at in 1..=RING + 3 {
            keep(&r, "o", &outcome(at)).unwrap();
        }
        assert_eq!(r.count_under("mcp/outcome/o/"), RING as usize);
        let kept: Vec<u64> = newest(&r, "o", 3).iter().map(|o| o.at_ms).collect();
        assert_eq!(kept, [RING + 3, RING + 2, RING + 1]);
        assert!(newest(&r, "other", 3).is_empty());
    }

    /// **The block names each task's handle and how it ended**, and is nothing
    /// when nothing ended.
    #[test]
    fn the_block_names_each_task_and_its_end() {
        assert_eq!(block(&[]), None);
        assert_eq!(
            block(&[outcome(5)]).unwrap(),
            "Finished while nobody waited on it — read and verify before relying on it:\n  \
             - T:dj-set/doc-5 — Task ended: Wrote docs/midi_dj_set.md; pushed \
             zen/docs-midi-dj-set; the merge waits on you."
        );
    }
}
