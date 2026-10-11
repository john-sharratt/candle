//! The typed shape of a mind's missions: its shared prompts, its generators,
//! and its workflows — each step, who takes it, what it is told, and where
//! each result leads. [`super::parse`] builds these from YAML.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// The target that settles an operation as succeeded.
pub const DONE: &str = "done";

/// The target that settles an operation as failed.
pub const FAILED: &str = "failed";

/// The end state only the engine sets, through [`super::run::cancel`]. No
/// step may lead to it and no step may be named for it.
pub const CANCELLED: &str = "cancelled";

/// The outcome every actor step offers: the actor cannot do it.
pub const STUCK: &str = "stuck";

/// The variant key for a step entered at the start of a run, by
/// [`super::run::start_at`] or by [`super::run::reopen`].
pub const START: &str = "start";

/// The variant key used when no variant names the incoming outcome.
pub const DEFAULT: &str = "default";

/// How many send-backs a workflow allows when it does not say.
pub const SEND_BACKS: u32 = 3;

/// How many stuck reports a step takes in one round when it does not say.
pub const STUCK_LIMIT: u32 = 1;

/// How often a generator is drawn, relative to the others, when it does not
/// say.
pub const WEIGHT: u32 = 1;

/// Who may take a step.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum By {
    /// Any actor.
    Maker,
    /// An actor who has taken no step in the current round.
    Another,
    /// The engine itself, through the step's `call`; never an actor.
    Table,
}

impl By {
    /// The `by:` value as written, or `None` for a value that names nobody.
    pub fn parse(text: &str) -> Option<By> {
        match text {
            "maker" => Some(By::Maker),
            "another" => Some(By::Another),
            "table" => Some(By::Table),
            _ => None,
        }
    }
}

/// What a step does to its operation's document.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Edits {
    /// It writes the document whole.
    New,
    /// It changes the document; a commit is required.
    Change,
    /// It leaves the document unless it finds a fault.
    #[default]
    Optional,
}

impl Edits {
    /// The `edits:` value as written.
    pub fn parse(text: &str) -> Option<Edits> {
        match text {
            "new" => Some(Edits::New),
            "change" => Some(Edits::Change),
            "optional" => Some(Edits::Optional),
            _ => None,
        }
    }
}

/// What becomes of an operation's document when the operation fails.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OnFailed {
    /// It is moved out of the record and retired from memory.
    SetAside,
    /// It is put back as it was before the operation opened.
    Restore,
    /// It stays as it is.
    #[default]
    Keep,
}

impl OnFailed {
    /// The `on-failed:` value as written.
    pub fn parse(text: &str) -> Option<OnFailed> {
        match text {
            "set-aside" => Some(OnFailed::SetAside),
            "restore" => Some(OnFailed::Restore),
            "keep" => Some(OnFailed::Keep),
            _ => None,
        }
    }
}

/// A step setting that is one value, or varies with the outcome that led into
/// the step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Variants<T> {
    /// One value, whatever led into the step.
    One(T),
    /// Values keyed by the incoming outcome — [`START`] on entry, [`DEFAULT`]
    /// for any outcome no key names — in the order written.
    ByOutcome(Vec<(String, T)>),
}

impl<T> Variants<T> {
    /// The value for a step entered by `incoming`: the outcome that led here,
    /// [`START`] on entry, or `None` after a plain completion, which only
    /// [`DEFAULT`] answers.
    pub fn select(&self, incoming: Option<&str>) -> Result<&T, String> {
        let variants = match self {
            Variants::One(value) => return Ok(value),
            Variants::ByOutcome(variants) => variants,
        };
        let find = |key: &str| variants.iter().find(|(k, _)| k == key).map(|(_, v)| v);
        incoming
            .and_then(find)
            .or_else(|| find(DEFAULT))
            .ok_or_else(|| match incoming {
                Some(key) => format!("has no variant for `{key}` and no `{DEFAULT}`"),
                None => format!("has no `{DEFAULT}` variant for a plain completion"),
            })
    }

    /// The value keyed `key`, for a [`Variants::ByOutcome`].
    pub fn variant(&self, key: &str) -> Option<&T> {
        match self {
            Variants::One(_) => None,
            Variants::ByOutcome(variants) => {
                variants.iter().find(|(k, _)| k == key).map(|(_, v)| v)
            }
        }
    }
}

/// Where a step leads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Next {
    /// `next: <target>` — one result and no choice: completing the step moves
    /// to this target.
    Single(String),
    /// `next:` as a map — a choice: each `(outcome, target)` names a result
    /// and where it leads, in the order listed. On a table step the outcomes
    /// are its call's verdicts.
    Outcomes(Vec<(String, String)>),
}

/// One step of a workflow.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Step {
    pub name: String,
    pub by: By,
    /// The text its taker is given, includes resolved; operation placeholders
    /// are filled by [`super::prompt::fill`].
    pub prompt: Variants<String>,
    pub next: Next,
    /// The Rust-registered decode a table step runs. `Some` exactly when `by`
    /// is [`By::Table`].
    pub call: Option<String>,
    pub edits: Variants<Edits>,
    /// Acts the step adds to the actor's own, in the order written.
    pub tools: Vec<String>,
    /// Checks the step's result must pass, in the order written.
    pub checks: Vec<String>,
    /// Sections the engine puts before the prompt, in the order written.
    pub context: Vec<String>,
    /// Where [`STUCK`] leads once reported `stuck_limit` times in a round.
    pub stuck: String,
    /// How many stuck reports the step takes in one round before it routes.
    pub stuck_limit: u32,
}

impl Step {
    /// Where reporting `outcome` on this step leads, [`STUCK`] aside.
    ///
    /// A step with a single next step takes a plain completion (`None`); a
    /// step with outcomes takes one of the outcomes it offers.
    pub fn route(&self, outcome: Option<&str>) -> Result<&str, String> {
        match (&self.next, outcome) {
            (Next::Single(target), None) => Ok(target),
            (Next::Single(_), Some(o)) => Err(format!(
                "step `{}` has no outcomes to choose from; it takes a plain completion, not `{o}`",
                self.name
            )),
            (Next::Outcomes(outcomes), None) => Err(format!(
                "step `{}` needs an outcome: one of {}",
                self.name,
                outcome_list(outcomes)
            )),
            (Next::Outcomes(outcomes), Some(o)) => outcomes
                .iter()
                .find(|(name, _)| name == o)
                .map(|(_, target)| target.as_str())
                .ok_or_else(|| {
                    format!(
                        "step `{}` offers {}; not `{o}`",
                        self.name,
                        outcome_list(outcomes)
                    )
                }),
        }
    }

    /// Every target this step can lead to, in the order written, its
    /// [`STUCK`] target last for an actor step.
    pub fn targets(&self) -> Vec<&str> {
        let mut targets: Vec<&str> = match &self.next {
            Next::Single(target) => vec![target.as_str()],
            Next::Outcomes(outcomes) => outcomes.iter().map(|(_, t)| t.as_str()).collect(),
        };
        if self.by != By::Table {
            targets.push(&self.stuck);
        }
        targets
    }
}

pub(super) fn outcome_list(outcomes: &[(String, String)]) -> String {
    outcomes
        .iter()
        .map(|(name, _)| format!("`{name}`"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// A named workflow: its steps, the first being the start, and what governs
/// its runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Workflow {
    pub name: String,
    /// How many send-backs a run may take; the one after settles it failed.
    pub send_backs: u32,
    /// Where its work is done, when it names a place.
    pub desk: Option<String>,
    /// When its work is done, when it names a time.
    pub year: Option<String>,
    pub on_failed: OnFailed,
    pub steps: Vec<Step>,
}

impl Workflow {
    /// The step a run starts at. A validated workflow always has one.
    pub fn start(&self) -> &Step {
        &self.steps[0]
    }

    /// The step called `name`.
    pub fn step(&self, name: &str) -> Option<&Step> {
        self.steps.iter().find(|s| s.name == name)
    }

    /// Where `name` stands in the step order.
    pub fn position(&self, name: &str) -> Option<usize> {
        self.steps.iter().position(|s| s.name == name)
    }
}

/// What proposes work: a table call whose accepted proposal opens an
/// operation on `workflow`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Generator {
    pub id: String,
    pub call: String,
    pub workflow: String,
    /// How often it is drawn, relative to the others.
    pub weight: u32,
    pub context: Vec<String>,
    /// Its prompt, includes resolved.
    pub prompt: String,
}

/// Everything a mind's `missions.yaml` declares for operations.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Missions {
    /// How many waiting missions the table is kept stocked with, when it says.
    pub keep: Option<u32>,
    /// The shared prompts, includes resolved.
    pub prompts: BTreeMap<String, String>,
    pub generators: Vec<Generator>,
    pub workflows: Vec<Workflow>,
}

impl Missions {
    /// The workflow called `name`.
    pub fn workflow(&self, name: &str) -> Option<&Workflow> {
        self.workflows.iter().find(|w| w.name == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(by: By, next: Next) -> Step {
        Step {
            name: "s".to_string(),
            by,
            prompt: Variants::One("p".to_string()),
            next,
            call: None,
            edits: Variants::One(Edits::Optional),
            tools: Vec::new(),
            checks: Vec::new(),
            context: Vec::new(),
            stuck: FAILED.to_string(),
            stuck_limit: STUCK_LIMIT,
        }
    }

    fn outcomes(pairs: &[(&str, &str)]) -> Next {
        Next::Outcomes(
            pairs
                .iter()
                .map(|(o, t)| (o.to_string(), t.to_string()))
                .collect(),
        )
    }

    #[test]
    fn route_follows_a_single_next_and_outcomes() {
        let a = step(By::Maker, Next::Single("b".to_string()));
        assert_eq!(a.route(None), Ok("b"));
        assert_eq!(
            a.route(Some("pass")),
            Err(
                "step `s` has no outcomes to choose from; it takes a plain completion, not `pass`"
                    .to_string()
            )
        );
        let b = step(By::Another, outcomes(&[("pass", "done"), ("fail", "a")]));
        assert_eq!(b.route(Some("fail")), Ok("a"));
        assert_eq!(
            b.route(Some("maybe")),
            Err("step `s` offers `pass`, `fail`; not `maybe`".to_string())
        );
        assert_eq!(
            b.route(None),
            Err("step `s` needs an outcome: one of `pass`, `fail`".to_string())
        );
    }

    #[test]
    fn targets_include_stuck_for_actor_steps_only() {
        let mut a = step(By::Maker, outcomes(&[("pass", "done"), ("fail", "a")]));
        a.stuck = "a".to_string();
        assert_eq!(a.targets(), ["done", "a", "a"]);
        let t = step(By::Table, outcomes(&[("sound", "b")]));
        assert_eq!(t.targets(), ["b"]);
    }

    #[test]
    fn a_variant_is_selected_by_incoming_outcome_then_default() {
        let one = Variants::One("one");
        assert_eq!(one.select(Some("fail")), Ok(&"one"));
        assert_eq!(one.select(None), Ok(&"one"));
        assert_eq!(one.variant("fail"), None);

        let by = Variants::ByOutcome(vec![
            ("start".to_string(), "fresh"),
            ("fail".to_string(), "again"),
            ("default".to_string(), "other"),
        ]);
        assert_eq!(by.select(Some("start")), Ok(&"fresh"));
        assert_eq!(by.select(Some("fail")), Ok(&"again"));
        assert_eq!(by.select(Some("mend")), Ok(&"other"));
        assert_eq!(by.select(None), Ok(&"other"));
        assert_eq!(by.variant("fail"), Some(&"again"));

        let no_default = Variants::ByOutcome(vec![("start".to_string(), Edits::New)]);
        assert_eq!(
            no_default.select(Some("fail")),
            Err("has no variant for `fail` and no `default`".to_string())
        );
        assert_eq!(
            no_default.select(None),
            Err("has no `default` variant for a plain completion".to_string())
        );
    }

    #[test]
    fn enums_parse_their_written_values_only() {
        assert_eq!(By::parse("maker"), Some(By::Maker));
        assert_eq!(By::parse("another"), Some(By::Another));
        assert_eq!(By::parse("table"), Some(By::Table));
        assert_eq!(By::parse("Maker"), None);
        assert_eq!(Edits::parse("new"), Some(Edits::New));
        assert_eq!(Edits::parse("change"), Some(Edits::Change));
        assert_eq!(Edits::parse("optional"), Some(Edits::Optional));
        assert_eq!(Edits::parse("maybe"), None);
        assert_eq!(Edits::default(), Edits::Optional);
        assert_eq!(OnFailed::parse("set-aside"), Some(OnFailed::SetAside));
        assert_eq!(OnFailed::parse("restore"), Some(OnFailed::Restore));
        assert_eq!(OnFailed::parse("keep"), Some(OnFailed::Keep));
        assert_eq!(OnFailed::parse("set_aside"), None);
        assert_eq!(OnFailed::default(), OnFailed::Keep);
    }
}
