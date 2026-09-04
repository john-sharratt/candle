//! One generation run: what it is doing, and stopping it.
//!
//! # One job per character, refused rather than queued
//!
//! Two runs against one life would race on the same plan file and the same
//! directory of documents — one writing `1998-09 First Term.md` while the other
//! decides that month no longer exists. The second start is refused with the
//! first job's id, so the console can attach to what is already running instead
//! of starting a duplicate it will have to reconcile.
//!
//! # The run is a thread, not a task
//!
//! `send_turn` blocks, and the fan-out is a pool of blocking calls whose
//! parallelism is the point. A job is therefore an OS thread that owns the
//! ladder from end to end; the API never waits on it and reads
//! [`GenProgress::snapshot`] instead.
//!
//! # A cancelled run keeps what it finished
//!
//! Cancelling stops the next fork from dispatching. Everything already written
//! stays — in the plan and on disk — because a life is generated a stratum at a
//! time and half a ladder is a useful thing to come back to. The alternative,
//! discarding a phase's work because the operator stopped the phase after it,
//! would make the stop button expensive enough that nobody uses it.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use candle_conversation::{ConversationEngine, SequenceConfig};
use serde::Serialize;

use super::generate::run_phase;
use super::plan::{Phase, Plan};
use super::progress::{GenProgress, GenSnapshot, Outcome};

/// A generation in progress, or one that has finished.
#[derive(Debug)]
pub struct Job {
    pub id: String,
    /// The reservation counter this run took, which is the *only* thing that
    /// orders two runs.
    ///
    /// The id embeds it — `zen-11` — and comparing ids is comparing strings, so
    /// `zen-11` sorted before `zen-2` and "the most recent run" became the wrong
    /// run from a character's tenth regeneration onward. A console attaching to
    /// it then watched a finished job and reported a live generation as done.
    seq: u64,
    pub who: String,
    /// The phases this run was asked for, in ladder order.
    pub phases: Vec<Phase>,
    pub progress: Arc<GenProgress>,
    /// The plan under construction. Shared with the API so an operator can read
    /// what has landed so far while the rest is still running.
    pub plan: Arc<Mutex<Plan>>,
}

/// What the console renders for a job.
#[derive(Clone, Debug, Serialize)]
pub struct JobView {
    pub id: String,
    pub who: String,
    pub phases: Vec<Phase>,
    #[serde(flatten)]
    pub progress: GenSnapshot,
}

impl Job {
    pub fn view(&self) -> JobView {
        JobView {
            id: self.id.clone(),
            who: self.who.clone(),
            phases: self.phases.clone(),
            progress: self.progress.snapshot(),
        }
    }
}

/// Why a run could not be started.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "refused", rename_all = "snake_case")]
pub enum NotStarted {
    /// This character is already being generated. Carries the running job's id
    /// so the console attaches rather than starting a duplicate.
    AlreadyRunning { job: String },
    /// No phases were asked for.
    NothingToDo,
}

impl NotStarted {
    pub fn message(&self) -> String {
        match self {
            NotStarted::AlreadyRunning { job } => {
                format!("a generation is already running for this character ({job})")
            }
            NotStarted::NothingToDo => "no phases were requested".to_string(),
        }
    }
}

/// Every generation this daemon has run since it started.
///
/// Finished jobs are kept so the console can read the outcome of a run that
/// ended between polls — a job that vanished on completion is indistinguishable
/// from one that never existed, and the difference matters when it failed.
#[derive(Default)]
pub struct Jobs {
    map: Mutex<HashMap<String, Arc<Job>>>,
    next: AtomicU64,
}

impl Jobs {
    pub fn new() -> Self {
        Self::default()
    }

    /// Claim a character for a run, or refuse with the reason.
    ///
    /// Separate from [`Self::start`] because it is the half with the invariants
    /// — one live run per character, ladder order, a stable id — and the half
    /// that must be testable without a card in the machine. `start` is then
    /// this plus a thread.
    pub fn reserve(&self, plan: Plan, phases: Vec<Phase>) -> Result<Arc<Job>, NotStarted> {
        if phases.is_empty() {
            return Err(NotStarted::NothingToDo);
        }
        let who = plan.seed.who.clone();
        let mut map = self.map.lock().unwrap();
        if let Some(running) = map
            .values()
            .find(|j| j.who == who && !j.progress.is_finished())
        {
            return Err(NotStarted::AlreadyRunning {
                job: running.id.clone(),
            });
        }

        // **Finished runs for this character are dropped as the new one supersedes
        // them.** They are kept between runs on purpose — a job that vanished on
        // completion is indistinguishable from one that never existed, and the
        // difference matters when it failed — but keeping every run this daemon
        // has ever done is a map that only grows, and an operator regenerating a
        // life a hundred times over a long-lived daemon holds a hundred plans.
        // One finished run per character is what the console reads; the older
        // ones are what nothing reads.
        map.retain(|_, j| j.who != who || !j.progress.is_finished());

        let seq = self.next.fetch_add(1, Ordering::Relaxed);
        let id = format!("{who}-{seq}");
        let mut phases = phases;
        // Ladder order regardless of how they were asked for: a months phase
        // that ran before its years would prime a prefix from an outline
        // nothing had expanded yet.
        phases.sort();
        phases.dedup();

        let job = Arc::new(Job {
            id: id.clone(),
            seq,
            who,
            phases,
            progress: Arc::new(GenProgress::new()),
            plan: Arc::new(Mutex::new(plan)),
        });
        map.insert(id, Arc::clone(&job));
        Ok(job)
    }

    /// Reserve a run and drive it on its own thread.
    ///
    /// The plan is handed in already loaded, because deciding what to generate
    /// (which phases, against which seed) belongs to the caller — this owns
    /// running it, not choosing it.
    pub fn start(
        &self,
        engine: Arc<Mutex<ConversationEngine>>,
        cfg: SequenceConfig,
        mind: &Path,
        plan: Plan,
        phases: Vec<Phase>,
    ) -> Result<Arc<Job>, NotStarted> {
        let job = self.reserve(plan, phases)?;
        let run = Arc::clone(&job);
        let phases = job.phases.clone();
        let mind: PathBuf = mind.to_path_buf();
        let spawned = std::thread::Builder::new()
            .name(format!("npcd-lifegen-{}", job.id))
            .spawn(move || {
                let outcome = drive(&engine, &cfg, &mind, &run, &phases);
                run.progress.finish(outcome);
            });
        if let Err(e) = spawned {
            // The thread never started, so nothing will ever finish this job.
            // Recording the failure is what keeps a job's state honest rather
            // than leaving one that reads as running forever.
            job.progress.finish(Outcome::Failed {
                error: format!("could not start the generation thread: {e}"),
            });
        }
        Ok(job)
    }

    pub fn get(&self, id: &str) -> Option<Arc<Job>> {
        self.map.lock().unwrap().get(id).cloned()
    }

    /// The run for a character — the live one if there is one, else the most
    /// recent.
    pub fn for_who(&self, who: &str) -> Option<Arc<Job>> {
        let map = self.map.lock().unwrap();
        let mut mine: Vec<&Arc<Job>> = map.values().filter(|j| j.who == who).collect();
        // By reservation order, not by id: the id embeds the counter as text, so
        // `zen-11` sorts before `zen-2` and the tenth regeneration onward picked
        // the wrong run as the most recent.
        mine.sort_by_key(|j| j.seq);
        mine.iter()
            .rev()
            .find(|j| !j.progress.is_finished())
            .or_else(|| mine.last())
            .map(|j| Arc::clone(j))
    }

    /// Ask a run to stop at its next fork boundary.
    pub fn cancel(&self, id: &str) -> bool {
        match self.get(id) {
            Some(j) => {
                j.progress.cancel();
                true
            }
            None => false,
        }
    }

    pub fn list(&self) -> Vec<JobView> {
        let map = self.map.lock().unwrap();
        // Sorted by reservation order before the views are built: sorting the
        // views means sorting ids, and an id compares as text — `zen-11` before
        // `zen-2`, which is the order this listing was rendered in.
        let mut jobs: Vec<&Arc<Job>> = map.values().collect();
        jobs.sort_by_key(|j| j.seq);
        jobs.into_iter().map(|j| j.view()).collect()
    }
}

/// Run the ladder, rung by rung.
fn drive(
    engine: &Arc<Mutex<ConversationEngine>>,
    cfg: &SequenceConfig,
    mind: &Path,
    job: &Job,
    phases: &[Phase],
) -> Outcome {
    for phase in phases {
        if job.progress.is_cancelled() {
            return Outcome::Cancelled;
        }
        match run_phase(engine, cfg, mind, &job.plan, *phase, &job.progress) {
            Ok(n) => tracing::info!("life {}: {} — {n} node(s) written", job.who, phase.unit()),
            Err(e) => {
                tracing::warn!("life {}: {} failed — {e:#}", job.who, phase.unit());
                return Outcome::Failed {
                    error: format!("{e:#}"),
                };
            }
        }
    }
    if job.progress.is_cancelled() {
        Outcome::Cancelled
    } else {
        Outcome::Done
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lifegen::seed::{check, Cadence, Seed};

    fn plan(who: &str) -> Plan {
        Plan::new(
            &check(&Seed {
                who: who.into(),
                display: "Cindy Tan".into(),
                born: "1998-09-14".into(),
                through: "1999-03-02".into(),
                place: "Nanyang".into(),
                role: "a clerk".into(),
                cadence: Cadence::Even,
                facts: Vec::new(),
                world: Vec::new(),
                cast: Vec::new(),
                eras: Vec::new(),
            })
            .unwrap(),
        )
    }

    /// A job with no phases is refused rather than started and immediately
    /// finished — the console would otherwise show a run that did nothing.
    #[test]
    fn a_run_with_no_phases_is_refused() {
        let jobs = Jobs::new();
        assert_eq!(
            jobs.reserve(plan("a"), Vec::new()).unwrap_err(),
            NotStarted::NothingToDo
        );
        assert!(jobs.list().is_empty(), "a refused run leaves no job behind");
    }

    /// **A months phase that ran before its years would prime a prefix from an
    /// outline nothing had expanded yet.** The order is a dependency, so it is
    /// imposed rather than trusted to the caller.
    #[test]
    fn phases_are_put_in_ladder_order_however_they_were_asked_for() {
        let jobs = Jobs::new();
        let job = jobs
            .reserve(
                plan("a"),
                vec![Phase::Days, Phase::Story, Phase::Months, Phase::Story],
            )
            .unwrap();
        assert_eq!(job.phases, vec![Phase::Story, Phase::Months, Phase::Days]);
    }

    /// Ids are unique across runs, so a second generation of the same character
    /// does not collide with the record of the first.
    #[test]
    fn each_run_gets_its_own_id() {
        let jobs = Jobs::new();
        let first = jobs.reserve(plan("a"), vec![Phase::Story]).unwrap();
        first.progress.finish(Outcome::Done);
        let second = jobs.reserve(plan("a"), vec![Phase::Story]).unwrap();
        assert_ne!(first.id, second.id);
        // The finished one is superseded rather than kept forever: `for_who` and
        // the console read the newest, and a daemon that keeps every run a
        // character ever had holds every plan with it.
        assert_eq!(jobs.list().len(), 1);
        assert_eq!(jobs.list()[0].id, second.id);
    }

    /// **"The most recent run" is the newest reservation, not the largest id.**
    ///
    /// Runs were ordered by comparing their ids, and an id embeds its counter as
    /// text — so from a character's tenth regeneration onward `zen-11` sorted
    /// before `zen-2` and the newest run stopped being the one picked. A console
    /// attaching to a character then watched an old finished job and reported the
    /// live generation as done.
    #[test]
    fn the_newest_run_wins_past_the_tenth() {
        let jobs = Jobs::new();
        // Ten finished runs, so the eleventh's id sorts below several of them.
        for _ in 0..10 {
            jobs.reserve(plan("zen"), vec![Phase::Story])
                .unwrap()
                .progress
                .finish(Outcome::Done);
        }
        let newest = jobs.reserve(plan("zen"), vec![Phase::Story]).unwrap();
        assert_eq!(newest.id, "zen-10");
        assert!(
            newest.id.as_str() < "zen-9",
            "the ids no longer sort the wrong way — this test is not testing anything"
        );
        assert_eq!(jobs.for_who("zen").unwrap().id, newest.id);
    }

    /// Two characters' runs never supersede each other, however they interleave.
    #[test]
    fn superseding_a_run_leaves_other_characters_alone() {
        let jobs = Jobs::new();
        let a = jobs.reserve(plan("a"), vec![Phase::Story]).unwrap();
        a.progress.finish(Outcome::Done);
        let b = jobs.reserve(plan("b"), vec![Phase::Story]).unwrap();
        b.progress.finish(Outcome::Done);
        let a2 = jobs.reserve(plan("a"), vec![Phase::Story]).unwrap();

        assert_eq!(jobs.for_who("a").unwrap().id, a2.id);
        assert_eq!(jobs.for_who("b").unwrap().id, b.id, "b's run was dropped");
        assert_eq!(jobs.list().len(), 2);
    }

    /// A finished job is kept, because a job that vanished on completion is
    /// indistinguishable from one that never existed — and the difference
    /// matters most when it failed.
    #[test]
    fn a_finished_job_is_still_readable() {
        let jobs = Jobs::new();
        let job = Arc::new(Job {
            id: "cindy-0".into(),
            seq: 0,
            who: "cindy".into(),
            phases: vec![Phase::Story],
            progress: Arc::new(GenProgress::new()),
            plan: Arc::new(Mutex::new(plan("cindy"))),
        });
        job.progress.finish(Outcome::Failed {
            error: "no engine".into(),
        });
        jobs.map
            .lock()
            .unwrap()
            .insert(job.id.clone(), Arc::clone(&job));

        let seen = jobs.get("cindy-0").unwrap();
        assert!(seen.progress.is_finished());
        assert_eq!(
            seen.view().progress.outcome,
            Some(Outcome::Failed {
                error: "no engine".into()
            })
        );
        // And it is what `for_who` falls back to once nothing is live.
        assert_eq!(jobs.for_who("cindy").unwrap().id, "cindy-0");
        assert!(jobs.for_who("nobody").is_none());
    }

    /// **Two runs against one life would race on the same plan and the same
    /// directory.** The refusal carries the running id so the console attaches
    /// instead of starting a duplicate.
    #[test]
    fn a_second_run_for_the_same_character_is_refused_with_the_first_ones_id() {
        let jobs = Jobs::new();
        let live = Arc::new(Job {
            id: "cindy-0".into(),
            seq: 0,
            who: "cindy".into(),
            phases: vec![Phase::Story],
            progress: Arc::new(GenProgress::new()),
            plan: Arc::new(Mutex::new(plan("cindy"))),
        });
        jobs.map
            .lock()
            .unwrap()
            .insert(live.id.clone(), Arc::clone(&live));

        let refused = jobs.reserve(plan("cindy"), vec![Phase::Story]).unwrap_err();
        assert_eq!(
            refused,
            NotStarted::AlreadyRunning {
                job: "cindy-0".into()
            }
        );
        assert!(refused.message().contains("cindy-0"));

        // A different character is unaffected — the lock is per life.
        assert!(jobs.reserve(plan("hess"), vec![Phase::Story]).is_ok());

        // Once it has finished, a new run for the same one is allowed.
        live.progress.finish(Outcome::Done);
        assert!(jobs.reserve(plan("cindy"), vec![Phase::Story]).is_ok());
    }

    /// A live job is what `for_who` returns even when a finished one sorts
    /// after it — the console wants what is running.
    #[test]
    fn a_live_job_outranks_a_finished_one_for_the_same_character() {
        let jobs = Jobs::new();
        for (seq, id, finished) in [(0, "cindy-0", false), (1, "cindy-1", true)] {
            let j = Arc::new(Job {
                id: id.into(),
                seq,
                who: "cindy".into(),
                phases: vec![Phase::Story],
                progress: Arc::new(GenProgress::new()),
                plan: Arc::new(Mutex::new(plan("cindy"))),
            });
            if finished {
                j.progress.finish(Outcome::Done);
            }
            jobs.map.lock().unwrap().insert(id.into(), j);
        }
        assert_eq!(jobs.for_who("cindy").unwrap().id, "cindy-0");
    }

    #[test]
    fn cancelling_an_unknown_job_says_so_rather_than_pretending() {
        let jobs = Jobs::new();
        assert!(!jobs.cancel("nope"));
    }

    #[test]
    fn cancelling_marks_the_run_without_ending_it() {
        let jobs = Jobs::new();
        let j = Arc::new(Job {
            id: "cindy-0".into(),
            seq: 0,
            who: "cindy".into(),
            phases: vec![Phase::Story],
            progress: Arc::new(GenProgress::new()),
            plan: Arc::new(Mutex::new(plan("cindy"))),
        });
        jobs.map
            .lock()
            .unwrap()
            .insert(j.id.clone(), Arc::clone(&j));
        assert!(jobs.cancel("cindy-0"));
        assert!(j.progress.is_cancelled());
        assert!(
            !j.progress.is_finished(),
            "a cancel request is not an ending"
        );
    }

    /// A job view carries everything the overlay renders, flattened so the
    /// console reads one object rather than reaching into a nested one.
    #[test]
    fn a_job_view_carries_its_identity_and_its_progress() {
        let j = Job {
            id: "cindy-0".into(),
            seq: 0,
            who: "cindy".into(),
            phases: vec![Phase::Story, Phase::Years],
            progress: Arc::new(GenProgress::new()),
            plan: Arc::new(Mutex::new(plan("cindy"))),
        };
        let v = serde_json::to_value(j.view()).unwrap();
        assert_eq!(v["id"], "cindy-0");
        assert_eq!(v["who"], "cindy");
        assert_eq!(v["phase"], "story");
        assert_eq!(v["stage"], "priming");
        assert_eq!(v["phases"][1], "years");
    }
}
