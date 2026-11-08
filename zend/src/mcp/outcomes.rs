//! What an alone task ended with, kept for the person it worked for
//! (`docs/zen_memory_and_guides.md` Phase 4): nobody waited on it, so its end
//! is raised in their next conversation, from any client, by
//! `get_user_context`.
//!
//! Each is kept under `mcp/outcome/<owner>/<at>-<task>` — when it ended, then
//! a digest of the task's handle, so two members keeping outcomes at once
//! each keep their own and neither takes the other's place. Past [`RING`] the
//! oldest are let go as a new one is kept, so a frame reads at most a few more
//! than [`RING`] small records, never a thread's history, and reading marks
//! nothing.

use std::cmp::Reverse;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::records::Records;

/// How many outcomes are kept per owner.
pub const RING: usize = 10;

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

fn prefix(owner: &str) -> String {
    format!("mcp/outcome/{owner}/")
}

/// Where `outcome` is kept: by when it ended, so key order is time order,
/// then by its task, so no two tasks' outcomes share a key.
fn key(owner: &str, outcome: &Outcome) -> String {
    let task = Sha256::digest(format!("{}/{}", outcome.thread, outcome.name).as_bytes());
    let task: String = task[..8].iter().map(|b| format!("{b:02x}")).collect();
    format!("{}{:016}-{task}", prefix(owner), outcome.at_ms)
}

/// Keep `outcome` for `owner`, letting the oldest go past [`RING`].
pub fn keep(records: &dyn Records, owner: &str, outcome: &Outcome) -> anyhow::Result<()> {
    records.put(&key(owner, outcome), serde_json::to_value(outcome)?)?;
    let kept = records.under(&prefix(owner));
    let over = kept.len().saturating_sub(RING);
    for (old, _) in kept.into_iter().take(over) {
        records.cease(&old)?;
    }
    Ok(())
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
    let mut out =
        "Finished while nobody waited on it — read and verify before relying on it:".to_string();
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
            text: "Wrote docs/midi_dj_set.md; pushed zen/docs-midi-dj-set; the merge waits on you."
                .into(),
        }
    }

    /// **The newest are kept, the oldest let go past the ring**, and an
    /// owner's are never another's.
    #[test]
    fn the_newest_are_kept_past_the_ring() {
        let r = MemRecords::default();
        let last = RING as u64 + 3;
        for at in 1..=last {
            keep(&r, "o", &outcome(at)).unwrap();
        }
        assert_eq!(r.count_under("mcp/outcome/o/"), RING);
        let kept: Vec<u64> = newest(&r, "o", 3).iter().map(|o| o.at_ms).collect();
        assert_eq!(kept, [last, last - 1, last - 2]);
        assert!(newest(&r, "other", 3).is_empty());
    }

    /// **Two tasks ending in the same millisecond keep an outcome each**, as
    /// two members keeping theirs at once do.
    #[test]
    fn two_outcomes_at_once_are_both_kept() {
        let r = MemRecords::default();
        let mut a = outcome(7);
        a.name = "a".into();
        let mut b = outcome(7);
        b.name = "b".into();
        keep(&r, "o", &a).unwrap();
        keep(&r, "o", &b).unwrap();
        let mut names: Vec<String> = newest(&r, "o", 5).into_iter().map(|o| o.name).collect();
        names.sort();
        assert_eq!(names, ["a", "b"]);
        let k = key("o", &a);
        assert!(k.starts_with("mcp/outcome/o/0000000000000007-"), "{k}");
        assert_eq!(k.len(), "mcp/outcome/o/0000000000000007-".len() + 16);
        assert_ne!(k, key("o", &b));
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
