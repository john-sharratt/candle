//! The journal entries of each worker already handed to Claude in a tool
//! result, so a later result — the next wait's, or the one that finds the
//! worker done — gives only the entries Claude has not read
//! (`docs/claude_zend_gateway.md` §3.3).
//!
//! Kept as a record of its own rather than on the worker's, which its beat
//! writes while the turn runs: a waiting call marking what it reported never
//! races the worker's own writes. A result marks its entries only once it is
//! delivered ([`Mark`]), so a call whose client went away before its answer
//! leaves them for the next one.

use serde_json::{json, Value};

use super::records::Records;

/// That a result handed over a worker's entries up to `entry`, to be kept
/// once the result is delivered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mark {
    pub thread: String,
    pub name: String,
    pub entry: u64,
}

/// Keep every one of `marks` for `owner`.
pub fn keep(records: &dyn Records, owner: &str, marks: &[Mark]) {
    for m in marks {
        if let Err(e) = mark(records, owner, &m.thread, &m.name, m.entry) {
            tracing::warn!(thread = %m.thread, name = %m.name, "mcp: the entries reported could not be marked: {e}");
        }
    }
}

fn key(owner: &str, thread: &str, name: &str) -> String {
    format!("mcp/reported/{owner}/{thread}/{name}")
}

/// The newest entry of the worker already reported; `None` when none was.
pub fn last(records: &dyn Records, owner: &str, thread: &str, name: &str) -> Option<u64> {
    records
        .get(&key(owner, thread, name))
        .and_then(|v| v.get("entry").and_then(Value::as_u64))
}

/// Record that the worker's entries up to `entry` have been reported.
pub fn mark(
    records: &dyn Records,
    owner: &str,
    thread: &str,
    name: &str,
    entry: u64,
) -> anyhow::Result<()> {
    if last(records, owner, thread, name).is_some_and(|l| l >= entry) {
        return Ok(());
    }
    records.put(&key(owner, thread, name), json!({ "entry": entry }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::records::MemRecords;

    #[test]
    fn the_mark_only_moves_forward() {
        let r = MemRecords::default();
        assert_eq!(last(&r, "o", "review", "verify"), None);
        mark(&r, "o", "review", "verify", 4).unwrap();
        mark(&r, "o", "review", "verify", 2).unwrap();
        assert_eq!(last(&r, "o", "review", "verify"), Some(4));
        assert_eq!(last(&r, "o", "review", "verify/read"), None);
        assert_eq!(
            r.get("mcp/reported/o/review/verify"),
            Some(json!({"entry": 4}))
        );
    }
}
