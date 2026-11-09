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

use crate::line::Folds;
use crate::path::{JournalPath, Place};
use crate::scope::Section;
use crate::time::Stamp;
use crate::tree::fingerprint::dirty;
use crate::tree::window::{next_fold, PeriodFold};
use crate::tree::{Journal, Tree};

/// A conversation ends after this long without a call: the connector's own
/// boundary (`docs/mcp_user_context.md` §2.5).
pub const CONVERSATION_GAP_SECS: i64 = 30 * 60;

/// A dirty journal is rebuilt once what its summary reads has been quiet this
/// long…
pub const QUIET_SECS: i64 = 30;

/// …or this long, for a journal whose summary reads its children's.
///
/// **A parent waits for its children to settle.** Every child's rebuilt
/// summary dirties its parent, and a master has almost no lines of its own, so
/// timed by its window alone it was rebuilt the moment any child was: on
/// 2026-10-31 `@zen` was rewritten nine times in eighty minutes, three of them
/// within five, each a new `journal.yaml` that zend's code reading read and
/// calibrated again. Timed from its newest child's rebuild, a parent is
/// rebuilt once per lull, after the children it reads have finished.
pub const PARENT_QUIET_SECS: i64 = 10 * 60;

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
        // A bot's journal takes a line for nearly everything it hears, says
        // and does: summarised on every lull it would be summarised all day.
        // It is rebuilt when a period of it folds, and its section when a
        // bot's summary has.
        if d.path.section() == Some(Section::Bots) {
            let j = tree.get(&d.path)?;
            return (refolded(j) || child_rebuilt(j)).then_some(Job::Summary { path: d.path });
        }
        let settle = if d.reads_children {
            PARENT_QUIET_SECS
        } else {
            QUIET_SECS
        };
        let quiet = d
            .last_change
            .is_none_or(|t| now.secs() - t.secs() >= settle);
        (quiet || d.new_lines >= NEW_LINES).then_some(Job::Summary { path: d.path })
    });
    closes.into_iter().chain(folds).chain(summaries).collect()
}

/// A period of `j`'s window folded since its summary was built: its window
/// holds fewer lines than the summary read — a bot's lines are only ever
/// added, so only a fold takes them away — or it has a fold and no summary.
fn refolded(j: &Journal) -> bool {
    match j.summary() {
        Some(s) => j.window.len() < s.lines,
        None => j
            .window
            .iter()
            .any(|l| matches!(l.folds, Some(Folds::Period(_)))),
    }
}

/// A child of `j` was summarised after `j` was, or `j` has no summary and a
/// child has one.
fn child_rebuilt(j: &Journal) -> bool {
    let built = j.summary().map(|s| s.built);
    j.live()
        .filter_map(|(_, c)| c.summary().map(|s| s.built))
        .any(|b| built.is_none_or(|own| b > own))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actor::Actor;
    use crate::experience::{Experience, ExperienceKind};
    use crate::ops::fixture::{john, p, zen};
    use crate::ops::period_fold::lines_over;
    use crate::ops::{apply, Lived, OnTaken, Op};
    use crate::scope::Channel;
    use crate::terms::Terms;
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
                terms: Terms::default(),
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

    /// **A master waits for its children to settle**: a child's rebuilt
    /// summary dirties it at once, and it is rebuilt only once the newest child
    /// rebuild is [`PARENT_QUIET_SECS`] old — not [`QUIET_SECS`] after it, which
    /// rewrote `@zen` once per child.
    #[test]
    fn a_master_is_rebuilt_once_its_children_have_settled() {
        use crate::files::{Meta, Sentence, Summary};
        use crate::tree::fingerprint::fingerprint;

        let mut t = zen();
        let t0 = at("2026-10-30T12:00:00Z");
        noted(&mut t, "topics", "Builder starts Monday.", t0);
        let built = t0.plus(QUIET_SECS);
        let topics = t.get_mut(&p("topics")).unwrap();
        let over = fingerprint(topics);
        topics
            .meta
            .get_or_insert_with(|| Meta::new("topics", None, t0))
            .summary = Some(Summary {
            over,
            built,
            lines: 1,
            children: 0,
            stale: false,
            digest: None,
            sentences: vec![Sentence {
                text: "The builder starts on Monday.".into(),
                cites: vec![],
            }],
        });
        let masters = |now: Stamp| -> Vec<String> {
            due(&t, now)
                .iter()
                .filter(|j| matches!(j, Job::Summary { .. }))
                .map(|j| j.path().show())
                .collect()
        };
        assert_eq!(masters(built.plus(QUIET_SECS)), Vec::<String>::new());
        assert_eq!(
            masters(built.plus(PARENT_QUIET_SECS - 1)),
            Vec::<String>::new()
        );
        assert_eq!(masters(built.plus(PARENT_QUIET_SECS)), ["@zen"]);
    }

    /// **A bot's journal is summarised when a period of it folds, never on a
    /// lull or a count**: quiet for an hour and fifty lines new, it waits.
    #[test]
    fn a_bots_journal_is_summarised_only_when_a_period_folds() {
        let mut t = zen();
        let npcd = Actor::Npcd { group: slug("zen") };
        let t0 = at("2026-10-01T00:00:00Z");
        apply(
            &mut t,
            &npcd,
            &Op::Create {
                under: p("bots"),
                name: slug("zen"),
                cue: "open for Zen".into(),
                on_taken: OnTaken::Refuse,
            },
            t0,
        )
        .unwrap();
        let lines: Vec<Lived> = (0..=WINDOW)
            .map(|i| Lived {
                at: t0.plus(3600 * i as i64),
                text: format!("did {i}"),
                experience: Experience::of(ExperienceKind::Did),
            })
            .collect();
        apply(
            &mut t,
            &npcd,
            &Op::Episodes {
                path: p("bots/zen"),
                lines,
            },
            t0,
        )
        .unwrap();
        let summaries = |t: &Tree, now| -> Vec<String> {
            due(t, now)
                .iter()
                .filter(|j| matches!(j, Job::Summary { .. }))
                .filter(|j| j.path().section() == Some(Section::Bots))
                .map(|j| j.path().show())
                .collect()
        };
        let later = at("2026-10-01T03:00:00Z").plus(10 * PARENT_QUIET_SECS);
        assert_eq!(summaries(&t, later), Vec::<String>::new());
        let now = at("2026-10-04T00:00:00Z");
        let Some(Job::Fold { path, plan }) = due(&t, now).into_iter().next() else {
            panic!("the first closed day folds");
        };
        apply(
            &mut t,
            &Actor::Zend,
            &Op::FoldPeriod {
                over: lines_over(t.get(&path).unwrap(), &plan.ids),
                path,
                period: plan.period,
                ids: plan.ids,
                text: "A day of chores.".into(),
            },
            now,
        )
        .unwrap();
        assert_eq!(summaries(&t, now), ["@zen/bots/zen"]);
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
