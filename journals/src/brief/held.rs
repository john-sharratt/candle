//! The jobs held back from a pass: one whose answer was refused, until what
//! its question was built from changes; one that could not be asked, until a
//! while has passed.
//!
//! The writer asks at temperature 0, so the same question over the same lines
//! gets the same answer and the same refusal — a call that would not parse, a
//! summary over its cap, a grammar that would not build. Asking again spends a
//! model turn for nothing and holds a pass's place from the jobs behind it. A
//! refusal is recorded against the fingerprint of what the question shows —
//! the journal's inputs for a summary or a close, the period's own lines for a
//! fold — and the job is asked again once that moves.
//!
//! A job that could not be asked at all — no slot, a turn that ended early —
//! may go through next time, or may fail the same way for reasons the writer
//! cannot see. It is deferred for [`DEFER_SECS`] rather than retried every
//! pass, so a handful that keep failing never take every pass's place.

use crate::files::Fingerprint;
use crate::ops::period_fold::lines_over;
use crate::time::Stamp;
use crate::tree::fingerprint::fingerprint;
use crate::tree::Tree;

use super::Job;

/// How long a job that could not be asked waits before it is asked again.
pub const DEFER_SECS: i64 = 10 * 60;

#[derive(Clone, Debug, PartialEq, Eq)]
enum Hold {
    /// Refused over this: asked again once it moves.
    Over(Fingerprint),
    /// Could not be asked: asked again from this moment.
    Until(Stamp),
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Held(Vec<(Job, Hold)>);

/// What `job`'s question is built from, as `tree` holds it.
fn over(tree: &Tree, job: &Job) -> Option<Fingerprint> {
    let j = tree.get(job.path())?;
    Some(match job {
        Job::Fold { plan, .. } => lines_over(j, &plan.ids),
        Job::Close { .. } | Job::Summary { .. } => fingerprint(j),
    })
}

impl Held {
    /// `job`'s answer was refused, over what `tree` holds.
    pub fn refused(&mut self, tree: &Tree, job: &Job) {
        let Some(f) = over(tree, job) else {
            return;
        };
        self.put(job, Hold::Over(f));
    }

    /// `job` could not be asked at `now`.
    pub fn deferred(&mut self, job: &Job, now: Stamp) {
        self.put(job, Hold::Until(now.plus(DEFER_SECS)));
    }

    fn put(&mut self, job: &Job, hold: Hold) {
        self.0.retain(|(k, _)| k != job);
        self.0.push((job.clone(), hold));
    }

    /// `due` less the jobs still held at `now` over `tree`. A hold no longer
    /// due, whose question's source has moved, or whose time has come, is
    /// forgotten.
    pub fn sift(&mut self, tree: &Tree, due: Vec<Job>, now: Stamp) -> Vec<Job> {
        self.0.retain(|(job, hold)| {
            due.contains(job)
                && match hold {
                    Hold::Over(f) => over(tree, job).as_ref() == Some(f),
                    Hold::Until(t) => now.secs() < t.secs(),
                }
        });
        due.into_iter()
            .filter(|job| !self.0.iter().any(|(k, _)| k == job))
            .collect()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::fixture::{john, now, p, sharratt};
    use crate::ops::{apply, Op};
    use crate::path::Address;
    use crate::testing::at;
    use crate::tree::window::next_fold;
    use crate::tree::WINDOW;

    fn note(t: &mut Tree, path: &str, text: &str) {
        note_at(t, path, text, now());
    }

    fn note_at(t: &mut Tree, path: &str, text: &str, at: Stamp) {
        apply(
            t,
            &john(),
            &Op::Note {
                path: p(path),
                text: text.into(),
            },
            at,
        )
        .unwrap();
    }

    /// **A refused job waits for its journal to change**, and the jobs behind
    /// it run meanwhile.
    #[test]
    fn a_refused_job_is_held_until_its_journal_changes() {
        let mut t = sharratt();
        note(&mut t, "~", "Prefers tea.");
        note(&mut t, "topics", "Builder starts Monday.");
        let mine = Job::Summary { path: p("~") };
        let theirs = Job::Summary { path: p("topics") };
        let due = vec![mine.clone(), theirs.clone()];
        let mut h = Held::default();
        h.refused(&t, &mine);
        h.refused(&t, &mine);
        assert_eq!(h.len(), 1, "one hold, recorded once");
        assert_eq!(h.sift(&t, due.clone(), now()), vec![theirs.clone()]);
        assert_eq!(
            h.sift(&t, due.clone(), now()),
            vec![theirs.clone()],
            "held while nothing changes"
        );
        note(&mut t, "~", "And green tea most of all.");
        assert_eq!(
            h.sift(&t, due.clone(), now()),
            due,
            "asked again once it changed"
        );
        assert!(h.is_empty());
    }

    /// **A refused fold waits for its own lines**: a new line in the window
    /// does not change what the fold's question shows, so the fold stays held.
    #[test]
    fn a_refused_fold_is_held_until_its_periods_lines_change() {
        let mut t = sharratt();
        let t0 = at("2026-10-01T00:00:00Z");
        for i in 0..=WINDOW {
            note_at(&mut t, "~", &format!("n{i}"), t0.plus(3600 * i as i64));
        }
        let now = at("2026-10-03T12:00:00Z");
        let plan = next_fold(t.get(&p("~")).unwrap(), now).unwrap();
        let fold = Job::Fold {
            path: p("~"),
            plan: plan.clone(),
        };
        let mut h = Held::default();
        h.refused(&t, &fold);
        note_at(&mut t, "~", "a later line", now);
        assert_eq!(h.sift(&t, vec![fold.clone()], now), vec![], "still held");
        apply(
            &mut t,
            &john(),
            &Op::Edit {
                address: Address {
                    path: p("~"),
                    id: plan.ids[0].clone(),
                },
                text: "n0, corrected".into(),
            },
            now,
        )
        .unwrap();
        assert_eq!(h.sift(&t, vec![fold.clone()], now), vec![fold]);
    }

    /// **A job that could not be asked waits a while**, then is asked again.
    #[test]
    fn a_deferred_job_is_asked_again_once_its_time_comes() {
        let mut t = sharratt();
        note(&mut t, "~", "Prefers tea.");
        let job = Job::Summary { path: p("~") };
        let mut h = Held::default();
        h.deferred(&job, now());
        assert_eq!(
            h.sift(&t, vec![job.clone()], now().plus(DEFER_SECS - 1)),
            vec![]
        );
        assert_eq!(
            h.sift(&t, vec![job.clone()], now().plus(DEFER_SECS)),
            vec![job.clone()]
        );
        assert!(h.is_empty());
        h.refused(&t, &job);
        h.deferred(&job, now());
        assert_eq!(h.len(), 1, "a job is held one way at a time");
    }

    #[test]
    fn a_hold_no_longer_due_is_forgotten() {
        let mut t = sharratt();
        note(&mut t, "~", "Prefers tea.");
        let job = Job::Summary { path: p("~") };
        let mut h = Held::default();
        h.refused(&t, &job);
        assert_eq!(h.sift(&t, Vec::new(), now()), vec![]);
        assert!(h.is_empty());
        assert_eq!(h.sift(&t, vec![job.clone()], now()), vec![job]);
    }

    #[test]
    fn a_job_for_a_journal_that_is_gone_records_nothing() {
        let t = sharratt();
        let mut h = Held::default();
        h.refused(
            &t,
            &Job::Summary {
                path: p("topics/nowhere"),
            },
        );
        assert!(h.is_empty());
    }
}
