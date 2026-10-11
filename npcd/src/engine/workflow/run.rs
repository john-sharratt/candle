//! One operation's progress through a workflow, and the pure functions that
//! move it along.
//!
//! A [`Run`] is the whole persisted state: where the operation stands, the
//! outcome that led there, every step taken, and the current round. A route to
//! an earlier step, or to the same one, is a **send-back**: it opens a new
//! round whose first entry is the step that sent the work back. `another` is
//! judged within the round, so whoever sent the work back does not take the
//! fix, while the writer of an earlier round may review. A workflow's
//! `send-backs` bounds how many a run may take.
//!
//! Every actor step also offers [`STUCK`]: the report stays on the step and it
//! is offered to someone else until `stuck-limit` reports are made in the
//! round, and then routes to the step's `stuck` target.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::config::{By, Edits, Step, Workflow, DONE, FAILED, START, STUCK};

/// Who takes a step: the engine itself, or a named actor.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Taker {
    Table,
    Actor(String),
}

/// Where an operation stands.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Where {
    /// Waiting on the named step.
    NextStep(String),
    /// Settled: a step led to `done`.
    Done,
    /// Settled: a step led to `failed`, or the send-backs ran out; the reason.
    Failed(String),
    /// Settled by the engine through [`cancel`]; the reason.
    Cancelled(String),
}

/// One step taken.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Taken {
    pub step: String,
    pub by: Taker,
    /// The outcome reported, or `None` for the plain completion of a step
    /// with a single next step.
    pub outcome: Option<String>,
    /// Where it led: a step name, `done` or `failed`. A stuck report that
    /// leaves the step on offer leads to the step itself.
    pub to: String,
    /// What the taker reported finding.
    pub notes: String,
}

/// An operation's run of a workflow.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Run {
    /// The workflow it runs.
    pub workflow: String,
    pub at: Where,
    /// The outcome that led into the current step: [`START`] on entry, `None`
    /// after a plain completion. It selects the step's prompt and edits.
    pub incoming: Option<String>,
    /// Every step taken, in order.
    pub history: Vec<Taken>,
    /// Where in `history` the current round begins.
    pub round: usize,
    /// How many send-backs it has taken.
    pub send_backs: u32,
    /// How many stuck reports each step has had in the current round.
    pub stuck: BTreeMap<String, u32>,
}

impl Run {
    /// How many steps have been taken.
    pub fn steps_taken(&self) -> usize {
        self.history.len()
    }

    /// The step it waits on, or `None` once settled.
    pub fn current(&self) -> Option<&str> {
        match &self.at {
            Where::NextStep(step) => Some(step),
            Where::Done | Where::Failed(_) | Where::Cancelled(_) => None,
        }
    }

    /// The steps taken in the current round.
    pub fn this_round(&self) -> &[Taken] {
        &self.history[self.round.min(self.history.len())..]
    }

    /// Every actor who has taken a step in the current round. The table is
    /// not an actor.
    ///
    /// **A step reported stuck was not taken.** Whoever gave a step up did
    /// nothing to the work, so it is still another pair of eyes for the steps
    /// after — it is kept off only the step it gave up ([`Run::stuck_on`]).
    /// Counted as acting, three of four Makers had touched two operations, one
    /// of them only by giving up a review, and both waited half a day at canon
    /// on the fourth.
    pub fn actors_this_round(&self) -> BTreeSet<&str> {
        self.this_round()
            .iter()
            .filter(|t| t.outcome.as_deref() != Some(STUCK))
            .filter_map(|t| match &t.by {
                Taker::Actor(a) => Some(a.as_str()),
                Taker::Table => None,
            })
            .collect()
    }

    /// What the most recent step reported finding — what the next step's
    /// `{findings}` answers to, notably when that step sent the work back.
    /// `None` before any step, or when the last step reported nothing.
    pub fn last_findings(&self) -> Option<&str> {
        self.history
            .last()
            .map(|t| t.notes.trim())
            .filter(|n| !n.is_empty())
    }

    /// Whether `actor` reported `step` stuck in the current round.
    fn stuck_on(&self, step: &str, actor: &str) -> bool {
        self.this_round().iter().any(|t| {
            t.step == step
                && t.outcome.as_deref() == Some(STUCK)
                && t.by == Taker::Actor(actor.to_string())
        })
    }
}

/// What the step a run waits on asks of its taker.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Offered<'a> {
    pub step: &'a Step,
    /// The prompt variant for the incoming outcome; placeholders unfilled.
    pub prompt: &'a str,
    pub edits: Edits,
}

/// A fresh run of `workflow`, waiting on its first step.
pub fn start(workflow: &Workflow) -> Run {
    entered(workflow, &workflow.start().name)
}

/// A fresh run of `workflow`, waiting on `step`.
pub fn start_at(workflow: &Workflow, step: &str) -> Result<Run, String> {
    match workflow.step(step) {
        Some(_) => Ok(entered(workflow, step)),
        None => Err(format!("workflow `{}` has no step `{step}`", workflow.name)),
    }
}

fn entered(workflow: &Workflow, step: &str) -> Run {
    Run {
        workflow: workflow.name.clone(),
        at: Where::NextStep(step.to_string()),
        incoming: Some(START.to_string()),
        history: Vec::new(),
        round: 0,
        send_backs: 0,
        stuck: BTreeMap::new(),
    }
}

/// Put a settled run back on `step`, in a new round with its send-backs
/// counted afresh. Its history is kept.
pub fn reopen(workflow: &Workflow, run: &mut Run, step: &str) -> Result<(), String> {
    if run.workflow != workflow.name {
        return Err(wrong_workflow(workflow, run));
    }
    if run.current().is_some() {
        return Err("the operation is not settled, so there is nothing to reopen".to_string());
    }
    if workflow.step(step).is_none() {
        return Err(format!("workflow `{}` has no step `{step}`", workflow.name));
    }
    run.at = Where::NextStep(step.to_string());
    run.incoming = Some(START.to_string());
    run.round = run.history.len();
    run.send_backs = 0;
    run.stuck.clear();
    Ok(())
}

/// Settle an unsettled run as cancelled, for `why`.
pub fn cancel(run: &mut Run, why: &str) -> Result<(), String> {
    if run.current().is_none() {
        return Err("the operation is already settled".to_string());
    }
    run.at = Where::Cancelled(why.to_string());
    Ok(())
}

fn wrong_workflow(workflow: &Workflow, run: &Run) -> String {
    format!(
        "the run is of workflow `{}`, not `{}`",
        run.workflow, workflow.name
    )
}

/// The step `run` waits on, with the prompt and edits its incoming outcome
/// selects.
pub fn offered<'a>(workflow: &'a Workflow, run: &Run) -> Result<Offered<'a>, String> {
    let step = waiting(workflow, run)?;
    let incoming = run.incoming.as_deref();
    let ctx = |e: String| format!("step `{}`: {e}", step.name);
    Ok(Offered {
        step,
        prompt: step.prompt.select(incoming).map_err(ctx)?,
        edits: *step.edits.select(incoming).map_err(ctx)?,
    })
}

/// The step `run` waits on.
fn waiting<'a>(workflow: &'a Workflow, run: &Run) -> Result<&'a Step, String> {
    if run.workflow != workflow.name {
        return Err(wrong_workflow(workflow, run));
    }
    let name = run
        .current()
        .ok_or_else(|| "the operation is settled".to_string())?;
    workflow
        .step(name)
        .ok_or_else(|| format!("workflow `{}` has no step `{name}`", workflow.name))
}

/// Whether `taker` may take the step `run` waits on.
pub fn may_take(workflow: &Workflow, run: &Run, taker: &Taker) -> bool {
    refusal(workflow, run, taker).is_none()
}

/// Why `taker` may not take the step `run` waits on, or `None` when it may.
fn refusal(workflow: &Workflow, run: &Run, taker: &Taker) -> Option<String> {
    let step = match waiting(workflow, run) {
        Ok(step) => step,
        Err(why) => return Some(why),
    };
    let name = &step.name;
    match (step.by, taker) {
        (By::Table, Taker::Table) => None,
        (By::Table, Taker::Actor(a)) => {
            Some(format!("step `{name}` is the table's to take, not `{a}`'s"))
        }
        (By::Maker | By::Another, Taker::Table) => Some(format!(
            "step `{name}` is an actor's to take, not the table's"
        )),
        (_, Taker::Actor(a)) if run.stuck_on(name, a) => Some(format!(
            "`{a}` reported step `{name}` stuck this round; it is offered to someone else"
        )),
        (By::Maker, Taker::Actor(_)) => None,
        (By::Another, Taker::Actor(a)) => match run.actors_this_round().contains(a.as_str()) {
            true => Some(format!(
                "step `{name}` needs an actor who has not acted this round, and `{a}` has"
            )),
            false => None,
        },
    }
}

/// Take the step `run` waits on: `taker` reports `outcome` (`None` for the
/// plain completion of a step with a single next step) with `notes`, and the
/// run moves to where the step leads.
///
/// [`STUCK`] on an actor step stays on the step until `stuck-limit` reports
/// in the round, then follows the step's `stuck` target. A route to an earlier
/// step or the same one is a send-back: it opens a new round, and the one
/// past the workflow's `send-backs` settles the run failed. A refused report
/// leaves the run unchanged.
pub fn advance(
    workflow: &Workflow,
    run: &mut Run,
    taker: &Taker,
    outcome: Option<&str>,
    notes: &str,
) -> Result<Where, String> {
    if let Some(why) = refusal(workflow, run, taker) {
        return Err(why);
    }
    let step = waiting(workflow, run)?;
    let name = step.name.clone();
    let to = if outcome == Some(STUCK) && step.by != By::Table {
        let reports = run.stuck.get(&name).copied().unwrap_or(0) + 1;
        run.stuck.insert(name.clone(), reports);
        if reports < step.stuck_limit {
            run.history.push(taken(&name, taker, outcome, &name, notes));
            return Ok(run.at.clone());
        }
        step.stuck.clone()
    } else {
        step.route(outcome)?.to_string()
    };
    run.history.push(taken(&name, taker, outcome, &to, notes));
    let reported = |what: &str| match notes.trim() {
        "" => what.to_string(),
        found => format!("{what}: {found}"),
    };
    run.at = if to == DONE {
        Where::Done
    } else if to == FAILED {
        Where::Failed(reported(&format!(
            "step `{name}` led to failed on `{}`",
            outcome.unwrap_or("completion")
        )))
    } else {
        let back = workflow.position(&to) <= workflow.position(&name);
        if back {
            run.send_backs += 1;
            run.round = run.history.len() - 1;
            run.stuck.clear();
        }
        if back && run.send_backs > workflow.send_backs {
            Where::Failed(reported(&format!(
                "sent back {} times, more than the {} allowed; step `{name}` last found",
                run.send_backs, workflow.send_backs
            )))
        } else {
            run.incoming = outcome.map(str::to_string);
            Where::NextStep(to)
        }
    };
    Ok(run.at.clone())
}

fn taken(step: &str, by: &Taker, outcome: Option<&str>, to: &str, notes: &str) -> Taken {
    Taken {
        step: step.to_string(),
        by: by.clone(),
        outcome: outcome.map(str::to_string),
        to: to.to_string(),
        notes: notes.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::super::parse::parse_workflows;
    use super::*;

    const W: &str = "workflows:
  w:
    send-backs: 2
    steps:
      draft:
        by: maker
        edits:
          start: new
          default: change
        next: read
        prompt:
          start: Write {objective}.
          default: 'Fix it: {findings}'
      read:
        by: table
        call: reading
        next:
          ok: review
          bad: draft
          bin: failed
        prompt: Read it.
      review:
        by: another
        next:
          pass: done
          fail: draft
        stuck: draft
        stuck-limit: 2
        prompt: Review it.
";

    fn workflow() -> Workflow {
        parse_workflows(W).unwrap().remove(0)
    }

    fn actor(a: &str) -> Taker {
        Taker::Actor(a.to_string())
    }

    fn next(step: &str) -> Result<Where, String> {
        Ok(Where::NextStep(step.to_string()))
    }

    #[test]
    fn a_run_starts_at_the_first_step_with_the_start_variants() {
        let wf = workflow();
        let run = start(&wf);
        assert_eq!(run.at, Where::NextStep("draft".to_string()));
        assert_eq!(run.incoming.as_deref(), Some("start"));
        assert_eq!((run.steps_taken(), run.round, run.send_backs), (0, 0, 0));
        assert!(run.actors_this_round().is_empty());
        assert_eq!(run.last_findings(), None);
        let offer = offered(&wf, &run).unwrap();
        assert_eq!(offer.step.name, "draft");
        assert_eq!(offer.prompt, "Write {objective}.");
        assert_eq!(offer.edits, Edits::New);
    }

    #[test]
    fn a_run_can_start_at_any_step() {
        let wf = workflow();
        let run = start_at(&wf, "review").unwrap();
        assert_eq!(run.current(), Some("review"));
        assert_eq!(run.incoming.as_deref(), Some("start"));
        assert_eq!(
            start_at(&wf, "nowhere"),
            Err("workflow `w` has no step `nowhere`".to_string())
        );
    }

    #[test]
    fn the_table_takes_only_table_steps_and_actors_only_actor_steps() {
        let wf = workflow();
        let mut run = start(&wf);
        assert!(!may_take(&wf, &run, &Taker::Table));
        assert_eq!(
            advance(&wf, &mut run, &Taker::Table, None, ""),
            Err("step `draft` is an actor's to take, not the table's".to_string())
        );
        advance(&wf, &mut run, &actor("ann"), None, "").unwrap();
        assert!(may_take(&wf, &run, &Taker::Table));
        assert_eq!(
            advance(&wf, &mut run, &actor("bob"), Some("ok"), ""),
            Err("step `read` is the table's to take, not `bob`'s".to_string())
        );
    }

    #[test]
    fn an_outcome_the_step_does_not_offer_is_refused_and_changes_nothing() {
        let wf = workflow();
        let mut run = start(&wf);
        assert_eq!(
            advance(&wf, &mut run, &actor("ann"), Some("ok"), ""),
            Err("step `draft` has no outcomes to choose from; it takes a plain completion, not `ok`".to_string())
        );
        advance(&wf, &mut run, &actor("ann"), None, "").unwrap();
        let before = run.clone();
        assert_eq!(
            advance(&wf, &mut run, &Taker::Table, Some("maybe"), ""),
            Err("step `read` offers `ok`, `bad`, `bin`; not `maybe`".to_string())
        );
        assert_eq!(
            advance(&wf, &mut run, &Taker::Table, Some("stuck"), ""),
            Err("step `read` offers `ok`, `bad`, `bin`; not `stuck`".to_string())
        );
        assert_eq!(run, before);
    }

    #[test]
    fn a_send_back_opens_a_round_and_selects_by_its_outcome() {
        let wf = workflow();
        let mut run = start(&wf);
        advance(&wf, &mut run, &actor("ann"), None, "drafted").unwrap();
        assert_eq!(run.incoming, None);
        assert_eq!(
            advance(
                &wf,
                &mut run,
                &Taker::Table,
                Some("bad"),
                "  the dates disagree\n"
            ),
            next("draft")
        );
        assert_eq!((run.round, run.send_backs), (1, 1));
        assert_eq!(run.this_round()[0].step, "read");
        assert_eq!(run.incoming.as_deref(), Some("bad"));
        assert_eq!(run.last_findings(), Some("the dates disagree"));
        let offer = offered(&wf, &run).unwrap();
        assert_eq!(offer.prompt, "Fix it: {findings}");
        assert_eq!(offer.edits, Edits::Change);
        assert!(may_take(&wf, &run, &actor("ann")));
    }

    #[test]
    fn another_is_judged_within_the_round() {
        let wf = workflow();
        let mut run = start(&wf);
        advance(&wf, &mut run, &actor("ann"), None, "").unwrap();
        advance(&wf, &mut run, &Taker::Table, Some("ok"), "").unwrap();
        assert!(!may_take(&wf, &run, &actor("ann")));
        assert_eq!(
            advance(&wf, &mut run, &actor("ann"), Some("pass"), ""),
            Err(
                "step `review` needs an actor who has not acted this round, and `ann` has"
                    .to_string()
            )
        );
        assert_eq!(
            advance(&wf, &mut run, &actor("bob"), Some("fail"), "flat"),
            next("draft")
        );
        assert_eq!(run.actors_this_round(), BTreeSet::from(["bob"]));
        advance(&wf, &mut run, &actor("cara"), None, "").unwrap();
        advance(&wf, &mut run, &Taker::Table, Some("ok"), "").unwrap();
        assert!(!may_take(&wf, &run, &actor("bob")));
        assert!(!may_take(&wf, &run, &actor("cara")));
        assert!(may_take(&wf, &run, &actor("ann")));
        assert_eq!(
            advance(&wf, &mut run, &actor("ann"), Some("pass"), ""),
            Ok(Where::Done)
        );
    }

    #[test]
    fn the_send_back_past_the_allowance_fails_naming_the_findings() {
        let wf = workflow();
        let mut run = start(&wf);
        for _ in 0..2 {
            advance(&wf, &mut run, &actor("ann"), None, "").unwrap();
            assert_eq!(
                advance(&wf, &mut run, &Taker::Table, Some("bad"), "x"),
                next("draft")
            );
        }
        advance(&wf, &mut run, &actor("ann"), None, "").unwrap();
        assert_eq!(
            advance(&wf, &mut run, &Taker::Table, Some("bad"), "the dates still disagree"),
            Ok(Where::Failed(
                "sent back 3 times, more than the 2 allowed; step `read` last found: the dates still disagree"
                    .to_string()
            ))
        );
        assert_eq!(run.current(), None);
    }

    #[test]
    fn stuck_stays_on_the_step_until_its_limit_then_routes() {
        let wf = workflow();
        let mut run = start(&wf);
        advance(&wf, &mut run, &actor("ann"), None, "").unwrap();
        advance(&wf, &mut run, &Taker::Table, Some("ok"), "").unwrap();
        assert_eq!(
            advance(&wf, &mut run, &actor("bob"), Some("stuck"), "no desk"),
            next("review")
        );
        assert_eq!(run.stuck["review"], 1);
        assert_eq!(run.history.last().unwrap().to, "review");
        // Giving the step up is not acting on the work.
        assert_eq!(run.actors_this_round(), BTreeSet::from(["ann"]));
        assert_eq!(
            advance(&wf, &mut run, &actor("bob"), Some("pass"), ""),
            Err(
                "`bob` reported step `review` stuck this round; it is offered to someone else"
                    .to_string()
            )
        );
        assert!(may_take(&wf, &run, &actor("cara")));
        assert_eq!(
            advance(
                &wf,
                &mut run,
                &actor("cara"),
                Some("stuck"),
                "no desk either"
            ),
            next("draft")
        );
        assert_eq!((run.send_backs, run.round), (1, 3));
        assert_eq!(run.this_round()[0].by, actor("cara"));
        assert!(run.stuck.is_empty());
        assert_eq!(run.incoming.as_deref(), Some("stuck"));
        assert_eq!(offered(&wf, &run).unwrap().prompt, "Fix it: {findings}");
    }

    #[test]
    fn stuck_leads_to_failed_by_default() {
        let wf = workflow();
        let mut run = start(&wf);
        assert_eq!(
            advance(&wf, &mut run, &actor("ann"), Some("stuck"), "no paper"),
            Ok(Where::Failed(
                "step `draft` led to failed on `stuck`: no paper".to_string()
            ))
        );
    }

    #[test]
    fn leading_to_failed_settles_it_and_nothing_more_is_taken() {
        let wf = workflow();
        let mut run = start(&wf);
        advance(&wf, &mut run, &actor("ann"), None, "").unwrap();
        assert_eq!(
            advance(&wf, &mut run, &Taker::Table, Some("bin"), ""),
            Ok(Where::Failed(
                "step `read` led to failed on `bin`".to_string()
            ))
        );
        assert_eq!(
            advance(&wf, &mut run, &actor("ann"), None, ""),
            Err("the operation is settled".to_string())
        );
        assert!(!may_take(&wf, &run, &actor("ann")));
        assert_eq!(
            offered(&wf, &run),
            Err("the operation is settled".to_string())
        );
    }

    #[test]
    fn cancel_settles_only_an_unsettled_run() {
        let wf = workflow();
        let mut run = start(&wf);
        cancel(&mut run, "the document is gone").unwrap();
        assert_eq!(run.at, Where::Cancelled("the document is gone".to_string()));
        assert_eq!(
            cancel(&mut run, "again"),
            Err("the operation is already settled".to_string())
        );
    }

    #[test]
    fn reopen_puts_a_settled_run_back_in_a_new_round() {
        let wf = workflow();
        let mut run = start(&wf);
        assert_eq!(
            reopen(&wf, &mut run, "draft"),
            Err("the operation is not settled, so there is nothing to reopen".to_string())
        );
        advance(&wf, &mut run, &actor("ann"), None, "").unwrap();
        advance(&wf, &mut run, &Taker::Table, Some("bad"), "").unwrap();
        advance(&wf, &mut run, &actor("ann"), None, "").unwrap();
        advance(&wf, &mut run, &Taker::Table, Some("bin"), "").unwrap();
        assert_eq!(
            reopen(&wf, &mut run, "nowhere"),
            Err("workflow `w` has no step `nowhere`".to_string())
        );
        reopen(&wf, &mut run, "review").unwrap();
        assert_eq!(run.at, Where::NextStep("review".to_string()));
        assert_eq!(run.incoming.as_deref(), Some("start"));
        assert_eq!((run.round, run.send_backs, run.steps_taken()), (4, 0, 4));
        assert!(may_take(&wf, &run, &actor("ann")));
    }

    #[test]
    fn a_run_of_another_workflow_or_a_vanished_step_is_refused() {
        let wf = workflow();
        let mut run = start(&wf);
        run.workflow = "other".to_string();
        assert_eq!(
            advance(&wf, &mut run, &actor("ann"), None, ""),
            Err("the run is of workflow `other`, not `w`".to_string())
        );
        let mut run = start(&wf);
        run.at = Where::NextStep("gone".to_string());
        assert_eq!(
            advance(&wf, &mut run, &actor("ann"), None, ""),
            Err("workflow `w` has no step `gone`".to_string())
        );
    }

    #[test]
    fn a_missing_variant_is_an_error_at_offer_time() {
        let wf = parse_workflows("workflows:\n  w:\n    steps:\n      a:\n        by: maker\n        next: done\n        prompt:\n          fail: again\n").unwrap().remove(0);
        assert_eq!(
            offered(&wf, &start(&wf)),
            Err("step `a`: has no variant for `start` and no `default`".to_string())
        );
    }

    #[test]
    fn a_run_serializes_to_exact_json_and_back() {
        let wf = workflow();
        let mut run = start(&wf);
        advance(&wf, &mut run, &actor("ann"), None, "drafted").unwrap();
        advance(&wf, &mut run, &Taker::Table, Some("ok"), "").unwrap();
        advance(&wf, &mut run, &actor("bob"), Some("stuck"), "busy").unwrap();
        let json = serde_json::to_string(&run).unwrap();
        assert_eq!(
            json,
            r#"{"workflow":"w","at":{"next_step":"review"},"incoming":"ok","history":[{"step":"draft","by":{"actor":"ann"},"outcome":null,"to":"read","notes":"drafted"},{"step":"read","by":"table","outcome":"ok","to":"review","notes":""},{"step":"review","by":{"actor":"bob"},"outcome":"stuck","to":"review","notes":"busy"}],"round":0,"send_backs":0,"stuck":{"review":1}}"#
        );
        let back: Run = serde_json::from_str(&json).unwrap();
        assert_eq!(back, run);
        assert_eq!(serde_json::to_string(&Where::Done).unwrap(), r#""done""#);
        assert_eq!(
            serde_json::to_string(&Where::Cancelled("x".to_string())).unwrap(),
            r#"{"cancelled":"x"}"#
        );
    }
}
