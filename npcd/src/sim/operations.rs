//! Operations: the objectives the command table holds, each carried through a
//! workflow — a mind's `workflows:` in `missions.yaml` — by Makers and the
//! table.
//!
//! **The workflow is the operation's rules; this is its record.** An operation
//! is opened on a workflow and a document, and its [`Run`] says which step it
//! waits on, every step taken, by whom and with what outcome, and the round it
//! is in. Who may take a step, where an outcome leads, how often work may be
//! sent back and what a stuck report does are the workflow's
//! ([`crate::engine::workflow`]); this module holds the operations, who holds
//! the step on offer, and what the step started from.
//!
//! An operation is kept after it finishes, so what was tried, by whom, and why
//! it stood or failed can be read and edited from the operations tab.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::operation_names::name_for;
use crate::engine::workflow::{
    self as flow, Edits, Offered, OnFailed, Run, Taker, Where, Workflow,
};

/// One objective and what has become of it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Operation {
    pub id: u64,
    /// What it is called — "Operation Iron Lantern".
    pub name: String,
    /// What it is for, in a line.
    pub objective: String,
    /// The generator that found the work, or `operator`.
    pub generator: String,
    /// The ledger key of the target it works on.
    pub target: String,
    /// The mind path of the document it produces or changes.
    pub document: String,
    /// Its progress through its workflow.
    #[serde(default = "before_workflows")]
    pub run: Run,
    /// Who carries the step on offer, once somebody has taken it up.
    #[serde(default)]
    pub holder: Option<String>,
    /// Whether the step it waits on is on the table as a mission.
    #[serde(default)]
    pub offered: bool,
    /// The brief its document was written to — what it was to tell, which
    /// every later step answers to as well.
    #[serde(default)]
    pub brief: String,
    /// The fields of the proposal that opened it, which its steps' prompts are
    /// filled from.
    #[serde(default)]
    pub fields: BTreeMap<String, String>,
    /// What the work reads, beside its own document.
    #[serde(default)]
    pub reads: Vec<String>,
    /// What its document said when it opened, for one that works on a document
    /// the record already held — put back if it fails and its workflow says
    /// `restore`. `None` for a draft.
    #[serde(default)]
    pub before: Option<String>,
    /// What its document said when the step on offer was taken up — what a
    /// step that must change it is judged against.
    #[serde(default)]
    pub found: Option<String>,
    /// Whether a failed operation's document has been settled.
    #[serde(default)]
    pub retired: bool,
}

/// The run of an operation saved before operations ran on workflows: settled,
/// so nothing offers it again.
fn before_workflows() -> Run {
    Run {
        workflow: String::new(),
        at: Where::Cancelled("opened before operations ran on workflows".to_string()),
        incoming: None,
        history: Vec::new(),
        round: 0,
        send_backs: 0,
        stuck: BTreeMap::new(),
    }
}

impl Operation {
    /// The step it waits on, or `None` once settled.
    pub fn step(&self) -> Option<&str> {
        self.run.current()
    }

    /// Whether it is over.
    pub fn settled(&self) -> bool {
        self.run.current().is_none()
    }

    /// Whether it stands: a step led to `done`.
    pub fn succeeded(&self) -> bool {
        self.run.at == Where::Done
    }

    /// Whether it failed.
    pub fn failed(&self) -> bool {
        matches!(self.run.at, Where::Failed(_))
    }

    /// Why it failed or was called off.
    pub fn why(&self) -> Option<&str> {
        match &self.run.at {
            Where::Failed(why) | Where::Cancelled(why) => Some(why),
            Where::NextStep(_) | Where::Done => None,
        }
    }

    /// What the last step found — what the step it waits on answers to.
    pub fn findings(&self) -> Option<&str> {
        self.run.last_findings()
    }

    /// The outcome that led into the step it waits on.
    pub fn incoming(&self) -> Option<&str> {
        self.run.incoming.as_deref()
    }
}

/// Every operation a world's table has held, by id, and the workflows they
/// run on.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Operations {
    /// The id the next operation takes.
    next: u64,
    ops: BTreeMap<u64, Operation>,
    /// The mind's workflows, as last loaded — set by the engine from
    /// `missions.yaml`, and read afresh rather than saved.
    #[serde(skip)]
    workflows: BTreeMap<String, Workflow>,
}

impl Operations {
    /// Set the workflows operations run on, replacing any loaded before.
    pub fn set_workflows(&mut self, workflows: Vec<Workflow>) {
        self.workflows = workflows.into_iter().map(|w| (w.name.clone(), w)).collect();
    }

    /// The workflow called `name`, when loaded.
    pub fn workflow(&self, name: &str) -> Option<&Workflow> {
        self.workflows.get(name)
    }

    /// The workflow operation `op` runs on.
    pub fn workflow_of(&self, op: &Operation) -> Option<&Workflow> {
        self.workflow(&op.run.workflow)
    }

    /// Open an operation on `workflow` for `target`, its work `document`,
    /// waiting on the workflow's first step — or on `at`, when named. Returns
    /// its id.
    pub fn open(
        &mut self,
        workflow: &str,
        at: Option<&str>,
        generator: &str,
        target: &str,
        objective: &str,
        document: &str,
    ) -> Result<u64, String> {
        let wf = self
            .workflow(workflow)
            .ok_or_else(|| format!("there is no workflow called `{workflow}`"))?;
        let run = match at {
            Some(step) => flow::start_at(wf, step)?,
            None => flow::start(wf),
        };
        self.next += 1;
        let id = self.next;
        let name = name_for(id, &|n| self.ops.values().any(|o| o.name == n));
        self.ops.insert(
            id,
            Operation {
                id,
                name,
                objective: objective.to_string(),
                generator: generator.to_string(),
                target: target.to_string(),
                document: document.to_string(),
                run,
                holder: None,
                offered: false,
                brief: String::new(),
                fields: BTreeMap::new(),
                reads: Vec::new(),
                before: None,
                found: None,
                retired: false,
            },
        );
        Ok(id)
    }

    /// Keep `text` as what operation `id`'s document said when it opened.
    pub fn kept_before(&mut self, id: u64, text: &str) {
        if let Some(op) = self.ops.get_mut(&id) {
            op.before = Some(text.to_string());
        }
    }

    /// Keep the brief operation `id` was written to, the proposal fields its
    /// prompts are filled from, and what its work reads.
    pub fn briefed(
        &mut self,
        id: u64,
        brief: &str,
        fields: BTreeMap<String, String>,
        reads: Vec<String>,
    ) {
        if let Some(op) = self.ops.get_mut(&id) {
            op.brief = brief.to_string();
            op.fields = fields;
            op.reads = reads;
        }
    }

    pub fn get(&self, id: u64) -> Option<&Operation> {
        self.ops.get(&id)
    }

    pub fn get_mut(&mut self, id: u64) -> Option<&mut Operation> {
        self.ops.get_mut(&id)
    }

    /// Every operation, newest first.
    pub fn all(&self) -> impl Iterator<Item = &Operation> {
        self.ops.values().rev()
    }

    /// What the step operation `id` waits on asks of its taker: the step, and
    /// the prompt and edits its incoming outcome selects.
    pub fn offer(&self, id: u64) -> Result<Offered<'_>, String> {
        let op = self.ops.get(&id).ok_or("no such operation")?;
        let wf = self
            .workflow_of(op)
            .ok_or_else(|| format!("the workflow `{}` is not loaded", op.run.workflow))?;
        flow::offered(wf, &op.run)
    }

    /// The operations waiting on a step of the table's, oldest first: each id
    /// with the step's call.
    pub fn awaiting_table(&self) -> Vec<(u64, String)> {
        self.ops
            .values()
            .filter_map(|o| {
                let step = self.workflow_of(o)?.step(o.step()?)?;
                Some((o.id, step.call.clone()?))
            })
            .collect()
    }

    /// The operations waiting on a Maker's step that is not yet on the table,
    /// oldest first.
    pub fn awaiting_offer(&self) -> Vec<u64> {
        self.ops
            .values()
            .filter(|o| !o.offered && o.holder.is_none())
            .filter(|o| {
                self.workflow_of(o)
                    .and_then(|w| w.step(o.step()?))
                    .is_some_and(|s| s.call.is_none())
            })
            .map(|o| o.id)
            .collect()
    }

    /// The step operation `id` waits on is on the table.
    pub fn offered(&mut self, id: u64) {
        if let Some(o) = self.ops.get_mut(&id) {
            o.offered = true;
            o.holder = None;
        }
    }

    /// `body` took up the step operation `id` waits on, its document then
    /// saying `found`.
    pub fn taken(&mut self, id: u64, body: &str, found: Option<String>) {
        if let Some(o) = self.ops.get_mut(&id) {
            o.holder = Some(body.to_string());
            o.found = found;
        }
    }

    /// The step's mission left the table without being reported: it is
    /// offered again.
    pub fn released(&mut self, id: u64) {
        if let Some(o) = self.ops.get_mut(&id) {
            o.holder = None;
            o.offered = false;
            o.found = None;
        }
    }

    /// Whether `body` may take up the step operation `id` waits on — by the
    /// step's `by`, judged within the round.
    pub fn may_take(&self, id: u64, body: &str) -> bool {
        let Some(o) = self.ops.get(&id) else {
            return false;
        };
        o.holder.is_none()
            && self
                .workflow_of(o)
                .is_some_and(|w| flow::may_take(w, &o.run, &Taker::Actor(body.to_string())))
    }

    /// Take the step operation `id` waits on: `taker` reports `outcome`
    /// (`None` for a step with one result) with `notes`, and the operation
    /// moves where the step leads.
    pub fn advance(
        &mut self,
        id: u64,
        taker: &Taker,
        outcome: Option<&str>,
        notes: &str,
    ) -> Result<Where, String> {
        let o = self.ops.get_mut(&id).ok_or("no such operation")?;
        let wf = self
            .workflows
            .get(&o.run.workflow)
            .ok_or_else(|| format!("the workflow `{}` is not loaded", o.run.workflow))?;
        let at = flow::advance(wf, &mut o.run, taker, outcome, notes)?;
        o.holder = None;
        o.offered = false;
        o.found = None;
        Ok(at)
    }

    /// What the step operation `id` waits on does to its document.
    pub fn edits(&self, id: u64) -> Option<Edits> {
        self.offer(id).ok().map(|o| o.edits)
    }

    /// Call it off, for `why`. `false` when it was already over.
    pub fn cancel(&mut self, id: u64, why: &str) -> bool {
        let Some(o) = self.ops.get_mut(&id) else {
            return false;
        };
        if flow::cancel(&mut o.run, why).is_err() {
            return false;
        }
        o.holder = None;
        o.offered = false;
        true
    }

    /// Put a settled operation back on `step`, in a new round.
    pub fn reopen(&mut self, id: u64, step: &str) -> Result<(), String> {
        let o = self.ops.get_mut(&id).ok_or("no such operation")?;
        let wf = self
            .workflows
            .get(&o.run.workflow)
            .ok_or_else(|| format!("the workflow `{}` is not loaded", o.run.workflow))?;
        flow::reopen(wf, &mut o.run, step)?;
        o.holder = None;
        o.offered = false;
        o.retired = false;
        Ok(())
    }

    /// What becomes of `op`'s document now that it has failed.
    pub fn on_failed(&self, op: &Operation) -> OnFailed {
        self.workflow_of(op).map_or(OnFailed::Keep, |w| w.on_failed)
    }

    /// The failed operations whose documents have not yet been settled.
    pub fn to_retire(&self) -> Vec<Operation> {
        self.ops
            .values()
            .filter(|o| o.failed() && !o.retired)
            .cloned()
            .collect()
    }

    /// Whether an operation opened after `op` on the same document has
    /// succeeded — whose accepted text putting `op`'s document back would
    /// overwrite.
    pub fn succeeded_after(&self, op: &Operation) -> bool {
        self.ops
            .values()
            .any(|o| o.id > op.id && o.document == op.document && o.succeeded())
    }

    /// Record that operation `id`'s failed document has been settled.
    pub fn mark_retired(&mut self, id: u64) {
        if let Some(o) = self.ops.get_mut(&id) {
            o.retired = true;
        }
    }

    /// Rename it. Refused when another operation already has the name.
    pub fn rename(&mut self, id: u64, name: &str) -> Result<(), String> {
        let name = name.trim();
        if name.is_empty() {
            return Err("an operation needs a name".into());
        }
        if self.ops.values().any(|o| o.id != id && o.name == name) {
            return Err(format!("another operation is already called {name}"));
        }
        let o = self.ops.get_mut(&id).ok_or("no such operation")?;
        o.name = name.to_string();
        Ok(())
    }

    /// Restate what it is for.
    pub fn set_objective(&mut self, id: u64, objective: &str) -> Result<(), String> {
        let objective = objective.trim();
        if objective.is_empty() {
            return Err("an operation needs an objective".into());
        }
        let o = self.ops.get_mut(&id).ok_or("no such operation")?;
        o.objective = objective.to_string();
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::engine::workflow::parse_missions;

    /// The workflows npcd runs today, from the template the mind's file is
    /// written to.
    pub(crate) fn workflows() -> Vec<Workflow> {
        parse_missions(include_str!("../../../docs/npcd_workflows_current.yaml"))
            .unwrap()
            .workflows
    }

    fn ops() -> Operations {
        let mut ops = Operations::default();
        ops.set_workflows(workflows());
        ops
    }

    fn one(workflow: &str, document: &str) -> (Operations, u64) {
        let mut ops = ops();
        let id = ops
            .open(workflow, None, "untold", "era:x", "Tell X", document)
            .unwrap();
        (ops, id)
    }

    fn by(a: &str) -> Taker {
        Taker::Actor(a.to_string())
    }

    fn at(ops: &Operations, id: u64) -> Option<String> {
        ops.get(id).unwrap().step().map(str::to_string)
    }

    /// **A story runs write, read, review, read again, canon — and stands**,
    /// each Maker step by somebody who has not acted in the round.
    #[test]
    fn a_story_runs_its_workflow_and_stands() {
        let (mut ops, id) = one("story", "layers/stories/x.md");
        assert_eq!(ops.get(id).unwrap().name, "Operation Iron Lantern");
        assert_eq!(at(&ops, id).as_deref(), Some("write"));
        assert_eq!(ops.awaiting_offer(), vec![id]);
        assert_eq!(ops.edits(id), Some(Edits::New));
        ops.offered(id);
        assert!(ops.awaiting_offer().is_empty());
        assert!(ops.may_take(id, "wren"));
        ops.taken(id, "wren", None);
        assert!(!ops.may_take(id, "pax"), "held");
        ops.advance(id, &by("wren"), None, "written").unwrap();

        assert_eq!(ops.awaiting_table(), vec![(id, "reading".to_string())]);
        ops.advance(id, &Taker::Table, Some("mend"), "\"x\" — wrong")
            .unwrap();
        assert_eq!(at(&ops, id).as_deref(), Some("review"));
        assert_eq!(ops.edits(id), Some(Edits::Change), "a mend changes it");
        assert!(ops.offer(id).unwrap().prompt.contains("First, mend it."));
        assert!(!ops.may_take(id, "wren"), "not its own draft");
        ops.advance(id, &by("pax"), Some("pass"), "mended").unwrap();

        assert_eq!(ops.awaiting_table(), vec![(id, "reading".to_string())]);
        ops.advance(id, &Taker::Table, Some("sound"), "").unwrap();
        assert_eq!(at(&ops, id).as_deref(), Some("canon"));
        assert!(!ops.may_take(id, "pax"));
        assert_eq!(
            ops.advance(id, &by("bram"), Some("pass"), "agrees"),
            Ok(Where::Done)
        );
        let o = ops.get(id).unwrap();
        assert!(o.succeeded() && o.settled());
        assert!(ops.to_retire().is_empty());
    }

    /// **A rejection sends it to be fixed, by anybody but who rejected it** —
    /// its writer included; sent back past the workflow's allowance, it fails
    /// with what the last step found, and its document is owed settling.
    #[test]
    fn a_rejection_is_fixed_and_too_many_send_backs_fail() {
        let (mut ops, id) = one("life-event", "layers/life/creed/2950 X.md");
        ops.advance(id, &by("wren"), None, "written").unwrap();
        ops.advance(id, &Taker::Table, Some("sound"), "").unwrap();
        assert_eq!(ops.edits(id), Some(Edits::Optional));
        ops.advance(id, &by("pax"), Some("reject"), "set in the wrong era")
            .unwrap();
        assert_eq!(at(&ops, id).as_deref(), Some("fix"));
        assert_eq!(
            ops.get(id).unwrap().findings(),
            Some("set in the wrong era")
        );
        assert!(!ops.may_take(id, "pax"), "not the one who rejected it");
        assert!(ops.may_take(id, "wren"), "its writer may fix it");
        assert!(ops.offer(id).unwrap().prompt.contains("A fix."));
        // Whoever sent it back is of the new round, so each review is by
        // somebody else.
        for reviewer in ["bram", "cara", "bram"] {
            ops.advance(id, &by("wren"), None, "fixed").unwrap();
            ops.advance(id, &Taker::Table, Some("sound"), "").unwrap();
            ops.advance(id, &by(reviewer), Some("reject"), "still wrong")
                .unwrap();
        }
        let o = ops.get(id).unwrap();
        assert!(o.failed(), "{:?}", o.run.at);
        assert!(o.why().unwrap().ends_with("still wrong"));
        assert_eq!(ops.on_failed(o), OnFailed::SetAside);
        assert_eq!(ops.to_retire().len(), 1);
        ops.mark_retired(id);
        assert!(ops.to_retire().is_empty());
    }

    /// **A stuck review goes to somebody else**, and twice stuck, to a fix.
    #[test]
    fn a_stuck_review_goes_to_somebody_else_then_to_a_fix() {
        let (mut ops, id) = one("story", "layers/stories/x.md");
        ops.advance(id, &by("wren"), None, "written").unwrap();
        ops.advance(id, &Taker::Table, Some("mend"), "faults")
            .unwrap();
        ops.advance(id, &by("pax"), Some("stuck"), "no desk")
            .unwrap();
        assert_eq!(at(&ops, id).as_deref(), Some("review"));
        assert!(!ops.may_take(id, "pax"));
        ops.advance(id, &by("bram"), Some("stuck"), "no desk either")
            .unwrap();
        assert_eq!(at(&ops, id).as_deref(), Some("fix"));
    }

    /// **A correction stands on the table's second reading** — no canon check
    /// — and its document is put back if it fails.
    #[test]
    fn a_correction_stands_without_a_canon_check_and_is_restored_on_failure() {
        let (mut ops, id) = one("correction", "layers/eras/a.md");
        ops.kept_before(id, "the era as it was");
        assert_eq!(ops.edits(id), Some(Edits::Change));
        ops.advance(id, &by("wren"), None, "changed").unwrap();
        ops.advance(id, &Taker::Table, Some("sound"), "").unwrap();
        ops.advance(id, &by("pax"), Some("pass"), "agrees").unwrap();
        assert_eq!(
            ops.advance(id, &Taker::Table, Some("sound"), ""),
            Ok(Where::Done)
        );
        let (mut ops, id) = one("correction", "layers/eras/a.md");
        ops.advance(id, &by("wren"), Some("stuck"), "cannot find it")
            .unwrap();
        let o = ops.get(id).unwrap();
        assert!(o.failed());
        assert_eq!(ops.on_failed(o), OnFailed::Restore);
    }

    /// An operation can be opened at any step — a review of a document put
    /// through by hand — reopened once settled, called off, and a mission that
    /// left the table is offered again.
    #[test]
    fn opened_at_a_step_reopened_cancelled_and_released() {
        let mut ops = ops();
        let id = ops
            .open(
                "story",
                Some("read"),
                "operator",
                "doc:x",
                "Review x",
                "layers/stories/x.md",
            )
            .unwrap();
        assert_eq!(at(&ops, id).as_deref(), Some("read"));
        assert!(ops.open("nothing", None, "g", "t", "o", "d").is_err());
        assert!(ops.cancel(id, "not wanted"));
        assert!(!ops.cancel(id, "again"), "already over");
        assert_eq!(ops.get(id).unwrap().why(), Some("not wanted"));
        ops.reopen(id, "canon").unwrap();
        assert_eq!(at(&ops, id).as_deref(), Some("canon"));
        assert_eq!(ops.awaiting_offer(), vec![id]);
        ops.offered(id);
        ops.taken(id, "pax", Some("text".into()));
        assert_eq!(ops.get(id).unwrap().found.as_deref(), Some("text"));
        ops.released(id);
        assert_eq!(ops.awaiting_offer(), vec![id]);
    }

    /// An operation saved before workflows loads settled, and is offered to
    /// nobody.
    #[test]
    fn an_operation_saved_before_workflows_loads_settled() {
        let saved = r#"{"next":1,"ops":{"1":{"id":1,"name":"Operation Old","objective":"o",
            "generator":"g","target":"t","document":"d","phase":"reviewing"}}}"#;
        let mut ops: Operations = serde_json::from_str(saved).unwrap();
        ops.set_workflows(workflows());
        let o = ops.get(1).unwrap();
        assert!(o.settled() && !o.failed());
        assert!(ops.awaiting_offer().is_empty() && ops.awaiting_table().is_empty());
    }

    #[test]
    fn names_are_unique_and_edits_are_checked() {
        let (mut ops, a) = one("story", "layers/stories/a.md");
        let b = ops
            .open(
                "story",
                None,
                "untold",
                "era:y",
                "a story",
                "layers/stories/y.md",
            )
            .unwrap();
        assert_ne!(ops.get(a).unwrap().name, ops.get(b).unwrap().name);
        let taken = ops.get(a).unwrap().name.clone();
        assert!(ops.rename(b, &taken).is_err());
        assert!(ops.rename(b, " ").is_err());
        assert!(ops.rename(b, "Operation Quiet Harbour").is_ok());
        assert!(ops.set_objective(b, "").is_err());
        let newest: Vec<u64> = ops.all().map(|o| o.id).collect();
        assert_eq!(newest, vec![b, a]);
    }
}
