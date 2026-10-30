//! The background rebuilds (§5.3), as plain code: which journals want a
//! summary, a period fold, or closing now; the question each puts to the
//! model and the call its answer is held to; and the operation the answer
//! becomes. zend runs them with its model; nothing here needs one.
//!
//! | Module | |
//! |---|---|
//! | [`summary`] | a journal's summary: question, call, operation |
//! | [`fold`] | a window's closed period: question, call, operation |
//! | [`held`] | jobs held back: refused until their question's source moves, or deferred a while |
//!
//! Each call is built in `journal_core::spec`'s flat `{name, params}` form. A
//! grammar guides an object inside an array, and one array level: an array
//! inside an array element decodes free. So a sentence cites with `cite` and
//! an optional `also_cite`, two guided enums, rather than a list.

pub mod fold;
pub mod held;
pub mod summary;

use journal_core::tools::arguments;
use serde_json::{Map, Value};

use crate::path::{JournalPath, Place};
use crate::time::Stamp;
use crate::tree::fingerprint::dirty;
use crate::tree::window::{next_fold, PeriodFold};
use crate::tree::{Journal, Tree};

/// A conversation ends after this long without a call: the connector's own
/// boundary (`docs/mcp_user_context.md` §2.5).
pub const CONVERSATION_GAP_SECS: i64 = 30 * 60;

/// A dirty journal is rebuilt once its window has been quiet this long…
pub const QUIET_SECS: i64 = 30;

/// …or once this many lines are newer than its summary, whichever is first.
pub const NEW_LINES: usize = 20;

/// One rebuild to make.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Job {
    /// A conversation quiet past the gap: closed by zend, so its summary is
    /// its last.
    Close { path: JournalPath },
    /// A window over its size with a closed period to fold.
    Fold { path: JournalPath, plan: PeriodFold },
    /// A dirty journal: its [`summary::plan`] is read when the job runs, from
    /// the tree the earlier jobs of its pass left.
    Summary { path: JournalPath },
}

impl Job {
    /// The journal it is for.
    pub fn path(&self) -> &JournalPath {
        match self {
            Job::Close { path } | Job::Fold { path, .. } | Job::Summary { path } => path,
        }
    }
}

/// The arguments of `answer`, which was to be a `name` call.
pub fn call_of(answer: &str, name: &str) -> Result<Map<String, Value>, String> {
    match arguments(answer) {
        Some((n, args)) if n == name => Ok(args),
        Some((n, _)) => Err(format!("the answer was a {n} call, not {name}")),
        None => Err(format!("the answer was not a {name} call")),
    }
}

/// `path` is a conversation's journal.
fn is_conversation(path: &JournalPath) -> bool {
    path.place().is_ok_and(Place::is_conversation)
}

/// A conversation's journal still open and quiet past the gap at `now`:
/// one to close. Asked when the pass is planned, and again as its close is
/// about to be written, since a line may have come in between.
pub fn gone_quiet(j: &Journal, now: Stamp) -> bool {
    !j.is_closed() && !j.is_folded() && now.secs() - j.last_active().secs() >= CONVERSATION_GAP_SECS
}

/// Everything due at `now`, in the order it should run: closes, then folds —
/// each of which changes what a summary reads — then summaries, deepest
/// first so a parent reads its children's new ones.
pub fn due(tree: &Tree, now: Stamp) -> Vec<Job> {
    let mut closes = Vec::new();
    let mut folds = Vec::new();
    for (path, j) in tree.journals() {
        if !j.problems.is_empty() {
            continue;
        }
        if is_conversation(&path) && gone_quiet(j, now) {
            closes.push(Job::Close { path: path.clone() });
        }
        if let Some(plan) = next_fold(j, now) {
            folds.push(Job::Fold { path, plan });
        }
    }
    let summaries = dirty(tree).into_iter().filter_map(|d| {
        let quiet = d
            .last_line
            .is_none_or(|t| now.secs() - t.secs() >= QUIET_SECS);
        (quiet || d.new_lines >= NEW_LINES).then_some(Job::Summary { path: d.path })
    });
    closes.into_iter().chain(folds).chain(summaries).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actor::Actor;
    use crate::ops::fixture::{john, p, zen};
    use crate::ops::{apply, OnTaken, Op};
    use crate::scope::Channel;
    use crate::testing::{at, slug};
    use crate::tree::WINDOW;

    fn noted(t: &mut Tree, path: &str, text: &str, at: Stamp) {
        apply(
            t,
            &john(),
            &Op::Note {
                path: p(path),
                text: text.into(),
                speaker: None,
            },
            at,
        )
        .unwrap();
    }

    #[test]
    fn a_new_line_is_summarised_once_it_has_been_quiet() {
        let mut t = zen();
        let t0 = at("2026-10-30T12:00:00Z");
        noted(&mut t, "topics", "Builder starts Monday.", t0);
        assert_eq!(due(&t, t0.plus(QUIET_SECS - 1)), vec![]);
        let jobs = due(&t, t0.plus(QUIET_SECS));
        assert!(
            matches!(&jobs[..], [Job::Summary { path, .. }] if path.show() == "@zen/topics"),
            "{jobs:?}"
        );
    }

    #[test]
    fn many_new_lines_are_summarised_without_waiting() {
        let mut t = zen();
        let t0 = at("2026-10-30T12:00:00Z");
        for i in 0..NEW_LINES {
            noted(&mut t, "topics", &format!("n{i}"), t0);
        }
        assert_eq!(due(&t, t0).len(), 1);
    }

    #[test]
    fn a_quiet_conversation_is_closed_and_a_live_one_is_not() {
        let mut t = zen();
        let server = Actor::Server {
            person: slug("john"),
            channel: Channel::Grok,
        };
        let t0 = at("2026-10-30T12:00:00Z");
        let c = apply(
            &mut t,
            &server,
            &Op::Create {
                under: p("conversations"),
                name: slug("2026-10-30-1200-grok"),
                cue: "x".into(),
                on_taken: OnTaken::Number,
            },
            t0,
        )
        .unwrap()
        .created
        .unwrap();
        let closes = |t: &Tree, now| {
            due(t, now)
                .into_iter()
                .filter(|j| matches!(j, Job::Close { .. }))
                .count()
        };
        assert_eq!(closes(&t, t0.plus(CONVERSATION_GAP_SECS - 1)), 0);
        let j = t.get(&c).unwrap();
        assert!(!gone_quiet(j, t0.plus(CONVERSATION_GAP_SECS - 1)));
        assert!(gone_quiet(j, t0.plus(CONVERSATION_GAP_SECS)));
        let jobs = due(&t, t0.plus(CONVERSATION_GAP_SECS));
        assert_eq!(jobs[0], Job::Close { path: c.clone() });
        apply(
            &mut t,
            &Actor::Zend,
            &Op::Close { path: c },
            t0.plus(CONVERSATION_GAP_SECS),
        )
        .unwrap();
        assert_eq!(closes(&t, t0.plus(10 * CONVERSATION_GAP_SECS)), 0);
    }

    #[test]
    fn a_full_window_with_a_closed_day_folds_before_any_summary() {
        let mut t = zen();
        let t0 = at("2026-10-01T00:00:00Z");
        for i in 0..=WINDOW {
            noted(&mut t, "people", &format!("n{i}"), t0.plus(3600 * i as i64));
        }
        let jobs = due(&t, at("2026-10-04T00:00:00Z"));
        assert!(matches!(jobs.first(), Some(Job::Fold { .. })), "{jobs:?}");
        assert!(matches!(jobs.last(), Some(Job::Summary { .. })), "{jobs:?}");
        assert!(jobs.iter().all(|j| j.path() == &p("people")));
    }

    #[test]
    fn an_answer_is_read_as_the_call_it_was_to_be() {
        let call = r#"{"name": "journal_fold", "arguments": {"text": "The day."}}"#;
        assert_eq!(call_of(call, "journal_fold").unwrap()["text"], "The day.");
        assert_eq!(
            call_of(call, "journal_summary").unwrap_err(),
            "the answer was a journal_fold call, not journal_summary"
        );
        assert_eq!(
            call_of("The day was long.", "journal_fold").unwrap_err(),
            "the answer was not a journal_fold call"
        );
    }
}
