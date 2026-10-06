//! The dialogue's working set, as the daemon drives it
//! (`docs/zend_working_set.md`): which calls release it, what it is seeded
//! with, and how its locks come back after a restart.
//!
//! The state and the projection rule live in `candle_conversation`; this is
//! the part that knows about tools, repositories and turns.

pub mod restore;
pub mod seeds;

use anyhow::bail;
use candle_conversation::projection::{LayerId, Schema, TimelineId};
use candle_conversation::working_set::marks::{lock_tag, RELEASE_TAG};
use candle_conversation::working_set::WorkingSetConfig;
use zend_tools::registry;

/// The working set the dialogue layer declares, with every `release_on` name
/// checked against the tool registry.
///
/// An unknown name is a load error, not a warning: a release rule that names
/// no tool never fires, so a misspelt write would leave every lock standing
/// across the very change it exists to catch.
pub fn dialogue_config(
    schema: &Schema,
    dialogue: LayerId,
) -> anyhow::Result<Option<WorkingSetConfig>> {
    let Some(config) = schema
        .layers
        .iter()
        .find(|l| l.id == dialogue)
        .and_then(|l| l.working_set.clone())
    else {
        return Ok(None);
    };
    let unknown: Vec<&str> = config
        .release_on
        .iter()
        .map(String::as_str)
        .filter(|name| registry::find(name).is_none())
        .collect();
    if !unknown.is_empty() {
        bail!("working_set.release_on names no registered tool: {unknown:?}");
    }
    Ok(Some(config))
}

/// The marks the turn carrying a round's results gets: one lock per call the
/// working set served, in the order served, then the release when the round
/// made a `release_on` call. Served calls always precede the round's first
/// write, so replaying the marks releases them too — exactly as the live round
/// did.
pub fn round_marks(served: &[TimelineId], released: bool) -> Vec<String> {
    let mut marks: Vec<String> = served.iter().map(|&tl| lock_tag(tl)).collect();
    if released {
        marks.push(RELEASE_TAG.to_string());
    }
    marks
}

/// Whether a call to `tool` — under any name the registry answers to —
/// releases the working set's locks.
pub fn releases(config: &WorkingSetConfig, tool: &str) -> bool {
    let canonical = |name: &str| registry::find(name).map(|t| t.name);
    let Some(called) = canonical(tool) else {
        return false;
    };
    config
        .release_on
        .iter()
        .any(|name| canonical(name) == Some(called))
}

#[cfg(test)]
mod tests {
    use super::*;

    use candle_conversation::models::Dialect;
    use candle_conversation::projection::Builder;

    fn config(release_on: &[&str]) -> WorkingSetConfig {
        WorkingSetConfig {
            budget_tokens: 1_000,
            folder_tokens: 100,
            beta: 0.2,
            min_momentum: 100.0,
            max_file_tokens: 500,
            seeds: Vec::new(),
            release_on: release_on.iter().map(|s| s.to_string()).collect(),
            max_admits: 2,
        }
    }

    /// The results turn carries a lock per served call, in order, then the
    /// release when the round wrote; a round that did neither carries nothing.
    #[test]
    fn a_rounds_marks_are_its_locks_then_its_release() {
        let tl = |raw: u64| TimelineId::from_raw(raw).unwrap();
        assert_eq!(
            round_marks(&[tl(7), tl(3)], true),
            vec![
                "working_set:lock:7".to_string(),
                "working_set:lock:3".to_string(),
                "working_set:release".to_string(),
            ]
        );
        assert_eq!(
            round_marks(&[tl(7)], false),
            vec!["working_set:lock:7".to_string()]
        );
        assert!(round_marks(&[], false).is_empty());
    }

    /// A write releases under its own name and under any alias; a read never.
    #[test]
    fn a_write_releases_under_any_of_its_names() {
        let c = config(&["write", "git_commit"]);
        assert!(releases(&c, "write"));
        assert!(releases(&c, "file_write"));
        assert!(releases(&c, "git_commit"));
        assert!(!releases(&c, "file_read"));
        assert!(!releases(&c, "no_such_tool"));
    }

    /// Every name in the bundled schema's `release_on` is a registered tool —
    /// the check `dialogue_config` makes at load.
    #[test]
    fn the_bundled_release_on_names_real_tools() {
        let dialect = Dialect::chat_ml();
        let b = Builder::from_yaml_with_vars_and_dialect(
            include_str!("prompts/projection.yaml"),
            &[("workspace", "test")],
            Some(&dialect),
        )
        .expect("projection.yaml must parse");
        let dialogue = b.id_for_layer("dialogue").unwrap();
        let c = dialogue_config(b.schema(), dialogue)
            .expect("every release_on name is registered")
            .expect("the dialogue layer declares a working set");
        for name in [
            "write",
            "file_edit",
            "file_delete",
            "git_commit",
            "git_merge",
            "git_ref",
            "git_switch",
            "git_reset",
            "run_command",
            "code_run",
            "code_session_exec",
        ] {
            assert!(c.releases(name), "{name} releases the working set");
        }
    }
}
