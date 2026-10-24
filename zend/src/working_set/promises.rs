//! How long a served read's promise lasts (`docs/zend_working_set.md` §4.6).
//!
//! A `file_read` the fast path serves answers `{"status":"in_context",…}`: the
//! file's content is already here, at this anchor. That reply is a turn of the
//! dialogue, and the model reads it on every decode its turn is projected —
//! so the content has to be there for exactly as long. Released any sooner,
//! the model is left with a reply that says it holds the file and a context
//! that does not, and with whatever it already said from the file: asked about
//! it later, it found its own quotations of a paper it could no longer see and
//! concluded it had made them up.
//!
//! So a lock is a promise kept for the life of the reply that made it, the way
//! a real read's text lives as long as the turn that carries it. It ends when:
//!
//! - **the reply leaves the dialogue's projection** ([`departed`]) — the model
//!   no longer reads the promise, so nothing is broken by ending it;
//! - **the file it holds is written** ([`round_release`]) — the content would
//!   be stale, and the model's own write is in its history beside it;
//! - **a call with effects nobody can name runs** — a command, a git write, a
//!   code run — which may have changed any file: every lock goes.
//!
//! A new user message ends none of them. Each ending is marked on the turn
//! that follows it ([`release_one_tag`], [`RELEASE_TAG`]), so a restart
//! rebuilds exactly the promises still standing.

use candle_conversation::projection::{TimelineId, TurnIndex};
use candle_conversation::working_set::marks::{promised_locks, release_one_tag};
use candle_conversation::working_set::WorkingSetConfig;
use candle_conversation::ConversationEngine;
use zend_tools::registry;
use zend_vfs::Workspace;

use crate::branch_ingest::keys::{file_key_path, CONTENT_KEY};
use crate::fast_path::call_file_key;
use crate::tools::ToolResult;
use crate::working_set::releases;

/// The tools whose call names the one file it changes.
const WRITES_ONE_FILE: [&str; 3] = ["write", "file_edit", "file_delete"];

/// What a dispatched round's calls end.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Release {
    /// No call changed anything a lock holds.
    Nothing,
    /// The locks holding these files — workspace-relative, `candle/README.md`.
    Files(Vec<String>),
    /// Every lock: a call ran whose effects cannot be named.
    All,
}

/// What `results` end. A write that names its file ends that file's lock; any
/// other `release_on` call, or a write whose file cannot be read off its call,
/// ends them all.
pub fn round_release(
    config: &WorkingSetConfig,
    results: &[ToolResult],
    workspace: &Workspace,
) -> Release {
    let mut files: Vec<String> = Vec::new();
    for r in results.iter().filter(|r| releases(config, &r.call.name)) {
        let one_file =
            registry::find(&r.call.name).is_some_and(|t| WRITES_ONE_FILE.contains(&t.name));
        match call_file_key(&r.call, workspace).filter(|_| one_file) {
            Some(key) => {
                if !files.contains(&key) {
                    files.push(key);
                }
            }
            None => return Release::All,
        }
    }
    if files.is_empty() {
        Release::Nothing
    } else {
        Release::Files(files)
    }
}

/// The locks standing on `target`, each with the turn whose reply promised it.
fn promised(engine: &ConversationEngine, target: TimelineId) -> Vec<(TimelineId, TurnIndex)> {
    let tags = engine.indexed_turn_tags(target);
    promised_locks(tags.iter().map(|(turn, tags)| (*turn, tags.as_slice())))
}

/// The standing locks whose promising reply the projection of turn `current`
/// left out: a reply on an earlier turn that is not in `selected`. The turn
/// being answered is never among them — it is not in its own projection.
pub fn departed(
    engine: &ConversationEngine,
    target: TimelineId,
    selected: &[u32],
    current: u32,
) -> Vec<TimelineId> {
    gone_from_view(promised(engine, target), selected, current)
}

fn gone_from_view(
    promised: Vec<(TimelineId, TurnIndex)>,
    selected: &[u32],
    current: u32,
) -> Vec<TimelineId> {
    promised
        .into_iter()
        .filter(|(_, turn)| turn.0 < current && !selected.contains(&turn.0))
        .map(|(tl, _)| tl)
        .collect()
}

/// The standing locks whose conversation read one of `files`.
pub fn holding(
    engine: &ConversationEngine,
    target: TimelineId,
    files: &[String],
) -> Vec<TimelineId> {
    promised(engine, target)
        .into_iter()
        .map(|(tl, _)| tl)
        .filter(|&tl| {
            engine
                .conversation_metadata(tl)
                .and_then(|meta| meta.get(CONTENT_KEY).cloned())
                .and_then(|key| file_key_path(&key).map(str::to_string))
                .is_some_and(|path| files.contains(&path))
        })
        .collect()
}

/// End the promises of `locks` on `target` — each becomes provenance at
/// `released` momentum — and return the marks that record it, one per lock
/// that was standing.
pub fn release(
    engine: &ConversationEngine,
    target: TimelineId,
    locks: &[TimelineId],
    released: f32,
) -> Vec<String> {
    locks
        .iter()
        .filter(|&&tl| engine.working_set_release_one(target, tl, released))
        .map(|&tl| release_one_tag(tl))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;

    use serde_json::{json, Value};
    use zend_vfs::RepoSpec;

    use crate::tools::ToolCall;

    fn tl(raw: u64) -> TimelineId {
        TimelineId::from_raw(raw).unwrap()
    }

    fn config() -> WorkingSetConfig {
        WorkingSetConfig {
            budget_tokens: 1_000,
            folder_tokens: 100,
            lock_tokens: 1_000,
            beta: 0.2,
            min_momentum: 100.0,
            max_file_tokens: 500,
            seeds: Vec::new(),
            release_on: ["write", "file_edit", "file_delete", "run_command"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            max_admits: NonZeroUsize::new(2).unwrap(),
        }
    }

    fn workspace() -> Workspace {
        Workspace::new("D:/prog", vec![RepoSpec::named("candle")]).unwrap()
    }

    fn result(name: &str, arguments: Value) -> ToolResult {
        ToolResult {
            call: ToolCall {
                name: name.to_string(),
                arguments,
            },
            response: json!({}),
        }
    }

    /// A write that names its file ends that file's lock alone, under any of
    /// its names, once however often the round wrote it.
    #[test]
    fn a_write_ends_only_the_file_it_names() {
        let results = [
            result("file_read", json!({"repo": "candle", "path": "a.rs"})),
            result(
                "write",
                json!({"repo": "candle", "path": "./zend/src/main.rs"}),
            ),
            result(
                "file_edit",
                json!({"repo": "candle", "path": "zend/src/main.rs"}),
            ),
            result(
                "file_delete",
                json!({"repo": "candle", "path": "docs/old.md"}),
            ),
        ];
        assert_eq!(
            round_release(&config(), &results, &workspace()),
            Release::Files(vec![
                "candle/zend/src/main.rs".to_string(),
                "candle/docs/old.md".to_string(),
            ])
        );
    }

    /// A command may have changed any file, and so may a write whose file
    /// cannot be read off its call: every lock goes.
    #[test]
    fn a_call_with_unnamed_effects_ends_every_lock() {
        let ws = workspace();
        let command = [
            result("write", json!({"repo": "candle", "path": "a.rs"})),
            result(
                "run_command",
                json!({"repo": "candle", "command": "cargo fmt"}),
            ),
        ];
        assert_eq!(round_release(&config(), &command, &ws), Release::All);
        let pathless = [result("write", json!({"repo": "candle"}))];
        assert_eq!(round_release(&config(), &pathless, &ws), Release::All);
        let unknown_repo = [result(
            "write",
            json!({"repo": "elsewhere", "path": "a.rs"}),
        )];
        assert_eq!(round_release(&config(), &unknown_repo, &ws), Release::All);
    }

    /// Reads end nothing; neither does a round with no calls.
    #[test]
    fn a_read_ends_nothing() {
        let reads = [result(
            "file_read",
            json!({"repo": "candle", "path": "a.rs"}),
        )];
        assert_eq!(
            round_release(&config(), &reads, &workspace()),
            Release::Nothing
        );
        assert_eq!(
            round_release(&config(), &[], &workspace()),
            Release::Nothing
        );
    }

    /// The live failure: the paper was served on turn 7, the user asked
    /// something new on turn 10, and the reply that promised the paper was
    /// still in view. Its lock stands until that reply leaves the projection.
    #[test]
    fn a_promise_stands_while_its_reply_is_in_view() {
        let promised = vec![(tl(1), TurnIndex(7)), (tl(2), TurnIndex(3))];
        assert!(gone_from_view(promised.clone(), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10).is_empty());
        assert_eq!(
            gone_from_view(promised, &[0, 1, 2, 4, 5, 6, 7, 8, 9], 10),
            vec![tl(2)],
            "the reply on turn 3 left the projection"
        );
    }

    /// The turn being answered carries a promise it is not projected beside:
    /// it is never taken for one that left.
    #[test]
    fn the_turn_being_answered_is_never_departed() {
        assert!(gone_from_view(vec![(tl(1), TurnIndex(10))], &[0, 1], 10).is_empty());
    }
}
