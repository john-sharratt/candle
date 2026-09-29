//! Keeping each ingested conversation tied to the branches that hold it
//! (`docs/zend_branch_ingest.md` §6.3).
//!
//! A unit's key is its content, so one conversation stands for its version on
//! every branch that carries it — and which branches those are moves whenever
//! a branch does, without the content changing and so without any re-ingest.
//! The branch list is written when the unit commits, and each pass rewrites it
//! on every committed conversation whose list no longer matches the walk.
//!
//! A file's conversation also names the commit it read the file at. That one
//! never moves once written; a conversation committed without it is given a
//! commit the walk found holding the same version — the same bytes, since a
//! file's key is its blob.

use std::collections::HashMap;
use std::sync::Mutex;

use candle_conversation::projection::TimelineId;
use candle_conversation::ConversationEngine;

use super::keys::{branches_value, BRANCHES_KEY, COMMIT_KEY};
use super::plan::Committed;

/// The conversations whose recorded branches differ from `live` — each unit
/// key's current [`BRANCHES_KEY`] value — as `(timeline, value to write)`.
///
/// `recorded` is each committed conversation's current value, `None` when it
/// has none. A key absent from `live` is on no branch this pass walked, and
/// the plan decides its fate; an empty live value (the workspace's own unit)
/// is never written.
pub fn stale(
    committed: &[Committed],
    recorded: &HashMap<TimelineId, Option<String>>,
    live: &HashMap<&str, String>,
) -> Vec<(TimelineId, String)> {
    committed
        .iter()
        .filter_map(|c| {
            let want = live.get(c.key.as_str())?;
            if want.is_empty() {
                return None;
            }
            let have = recorded.get(&c.timeline).and_then(Option::as_deref);
            (have != Some(want.as_str())).then(|| (c.timeline, want.clone()))
        })
        .collect()
}

/// Rewrite [`BRANCHES_KEY`] on each of `committed` whose recorded branches
/// differ from `live` (unit key → the branches holding it, in walk order).
/// Returns how many were rewritten.
pub fn retie(
    engine: &Mutex<ConversationEngine>,
    layer: &str,
    committed: &[Committed],
    live: &HashMap<&str, &[String]>,
) -> usize {
    let live: HashMap<&str, String> = live
        .iter()
        .map(|(key, names)| (*key, branches_value(names)))
        .collect();
    let e = engine.lock().unwrap();
    let recorded: HashMap<TimelineId, Option<String>> = committed
        .iter()
        .map(|c| {
            let value = e
                .conversation_metadata(c.timeline)
                .and_then(|m| m.get(BRANCHES_KEY).cloned());
            (c.timeline, value)
        })
        .collect();
    let mut rewritten = 0usize;
    for (timeline, value) in stale(committed, &recorded, &live) {
        match e.set_conversation_metadata(timeline, BRANCHES_KEY, &value) {
            Ok(()) => rewritten += 1,
            Err(err) => tracing::warn!(
                target: "zend::branch_ingest",
                layer,
                timeline = timeline.raw(),
                "recording the branches a unit is on failed: {err:#}",
            ),
        }
    }
    if rewritten > 0 {
        tracing::info!(
            target: "zend::branch_ingest",
            layer,
            rewritten,
            "recorded the branches that now hold each unit",
        );
    }
    rewritten
}

/// The file conversations with no [`COMMIT_KEY`] whose key the walk found,
/// as `(timeline, commit to write)` — `found` maps each unit key to a commit
/// holding that version. One already naming a commit keeps it.
pub fn without_commit(
    committed: &[Committed],
    recorded: &HashMap<TimelineId, Option<String>>,
    found: &HashMap<&str, String>,
) -> Vec<(TimelineId, String)> {
    committed
        .iter()
        .filter(|c| recorded.get(&c.timeline).is_none_or(Option::is_none))
        .filter_map(|c| Some((c.timeline, found.get(c.key.as_str())?.clone())))
        .collect()
}

/// Write [`COMMIT_KEY`] on each of `committed` that has none, from `found`
/// (unit key → a commit holding that version). Returns how many were written.
pub fn backfill_commits(
    engine: &Mutex<ConversationEngine>,
    layer: &str,
    committed: &[Committed],
    found: &HashMap<&str, String>,
) -> usize {
    let e = engine.lock().unwrap();
    let recorded: HashMap<TimelineId, Option<String>> = committed
        .iter()
        .map(|c| {
            let value = e
                .conversation_metadata(c.timeline)
                .and_then(|m| m.get(COMMIT_KEY).cloned());
            (c.timeline, value)
        })
        .collect();
    let mut written = 0usize;
    for (timeline, commit) in without_commit(committed, &recorded, found) {
        match e.set_conversation_metadata(timeline, COMMIT_KEY, &commit) {
            Ok(()) => written += 1,
            Err(err) => tracing::warn!(
                target: "zend::branch_ingest",
                layer,
                timeline = timeline.raw(),
                "recording the commit a file was read at failed: {err:#}",
            ),
        }
    }
    if written > 0 {
        tracing::info!(
            target: "zend::branch_ingest",
            layer,
            written,
            "recorded the commit each file reading holds",
        );
    }
    written
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tl(n: u64) -> TimelineId {
        TimelineId::from_raw(n).unwrap()
    }

    fn committed(n: u64, key: &str) -> Committed {
        Committed {
            timeline: tl(n),
            key: key.to_string(),
            subject: "s".to_string(),
        }
    }

    /// **Only a list that changed is written**: one never written, one a
    /// branch has joined, and nothing for a list that still matches, a key no
    /// branch holds, or the workspace's own unit.
    #[test]
    fn only_a_changed_branch_list_is_written() {
        let all = [
            committed(1, "fresh"),
            committed(2, "joined"),
            committed(3, "same"),
            committed(4, "gone"),
            committed(5, "workspace"),
        ];
        let recorded = HashMap::from([
            (tl(1), None),
            (tl(2), Some("main".to_string())),
            (tl(3), Some("main,topic".to_string())),
            (tl(4), Some("main".to_string())),
            (tl(5), None),
        ]);
        let live = HashMap::from([
            ("fresh", "main".to_string()),
            ("joined", "main,topic".to_string()),
            ("same", "main,topic".to_string()),
            ("workspace", String::new()),
        ]);
        assert_eq!(
            stale(&all, &recorded, &live),
            [
                (tl(1), "main".to_string()),
                (tl(2), "main,topic".to_string())
            ]
        );
    }

    /// **A commit is written only where none is**: a reading that names one
    /// keeps it however the branches have moved, and a key the walk did not
    /// find is left for the plan.
    #[test]
    fn a_commit_is_backfilled_only_where_none_is() {
        let all = [
            committed(1, "a@1"),
            committed(2, "b@1"),
            committed(3, "gone@1"),
        ];
        let recorded = HashMap::from([
            (tl(1), None),
            (tl(2), Some("aaaa".to_string())),
            (tl(3), None),
        ]);
        let found = HashMap::from([("a@1", "cccc".to_string()), ("b@1", "dddd".to_string())]);
        assert_eq!(
            without_commit(&all, &recorded, &found),
            [(tl(1), "cccc".to_string())]
        );
    }
}
