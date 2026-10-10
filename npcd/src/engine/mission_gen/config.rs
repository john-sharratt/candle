//! What the command table is configured to do: `<mind>/missions.yaml`.
//!
//! The file is authored content, like `projection.yaml`, so how work is found,
//! asked for and carried can change without touching the engine. It names the
//! shared prompts, the **generators** — each a table call that proposes one
//! piece of work, the prompt that asks for it, how heavily it is drawn and the
//! workflow its accepted proposals open — and the **workflows** operations run
//! on ([`crate::engine::workflow`]); and how many missions wait at the table.
//!
//! **Every name is one the engine has.** A step's `call`, its `checks`, its
//! `context` and its `tools`, and a workflow's `year`, name Rust: a name the
//! engine does not implement is refused when the file is read, not discovered
//! when an operation reaches it.

use std::io::ErrorKind;
use std::path::Path;

use super::copied::BRIEF_COPIED;
use super::gates::CHECKS;
use super::target::Kind;
use crate::engine::workflow::{parse_missions, By, Workflow};

/// The file a mind configures its table in.
pub const FILE: &str = "missions.yaml";

/// How many generated missions wait at the table when the file does not say.
const DEFAULT_KEEP: usize = 4;

/// The table calls a workflow step may run.
pub const READING: &str = "reading";

/// The checks a step's result may be held to beyond the gate's own
/// ([`CHECKS`]): its document committed, changed from what the step found,
/// not its brief copied ([`BRIEF_COPIED`]), and a rejection's evidence.
pub const COMMITTED: &str = "committed";
pub const CHANGED: &str = "changed";
pub const REJECTION_QUOTES: &str = "rejection-quotes-the-draft";
pub const REJECTION_NAMES_AN_ERA: &str = "rejection-names-an-era";

/// The sections a Maker's step may put before its prompt.
pub const MAKER_CONTEXT: &[&str] = &[
    "world-then",
    "worlds-words",
    "reads",
    "anchor",
    "voice-example",
    "storyline",
];

/// The sections the table's reading may be shown — see
/// [`super::material::draft`].
pub const READING_CONTEXT: &[&str] = &[
    "draft",
    "when-set",
    "era-it-tells",
    "whose-life",
    "other-events",
    "voice-example",
    "era-it-falls-in",
    "worlds-words",
    "world-now",
    "timeline",
    "told",
];

/// The sections a generator's proposal may be shown — see
/// [`super::material::render`].
pub const PROPOSAL_CONTEXT: &[&str] = &[
    "whose-life",
    "life-story",
    "events-written",
    "latest-event",
    "timeline",
    "where-it-belongs",
    "how-dated",
    "document-a",
    "document-b",
    "era",
    "stories-told",
    "names-taken",
];

/// The acts a step may add to a Maker's own.
pub const TOOLS: &[&str] = &["report_rejected"];

/// When a workflow's work is done: the year its event falls in, or the year
/// the era its story tells opens — both what `canon::set_in` works out.
pub const YEARS: &[&str] = &["event-year", "era-opens"];

/// One configured generator: the call that proposes its work, the prompt that
/// asks for it, and the workflow its proposals open.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Generator {
    /// Its name — what an operation's origin and the ledger carry.
    pub id: String,
    /// What it finds work in, which decides the material it is handed and the
    /// call it answers with.
    pub kind: Kind,
    /// How often it is drawn against the others. Zero switches it off.
    pub weight: u32,
    /// What the model is asked, after the material.
    pub prompt: String,
    /// The sections of material it is shown ([`PROPOSAL_CONTEXT`]).
    pub context: Vec<String>,
    /// The workflow an accepted proposal opens.
    pub workflow: String,
}

/// The whole configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    /// The table's own voice when it proposes work.
    pub system: String,
    /// The table's voice when it reads a document — a reader's, not a
    /// proposer's: told it sets work, a reading answered "nothing here worth
    /// doing" and called the draft sound.
    pub reader: String,
    /// How many generated missions to keep waiting at the table.
    pub keep: usize,
    /// What the table is asked when it reads a document — see
    /// [`super::reading`].
    pub reading: String,
    /// The voice a Maker writes a piece in when it sits down to compose —
    /// see [`crate::engine::compose`].
    pub writing: String,
    pub generators: Vec<Generator>,
    pub workflows: Vec<Workflow>,
}

/// The kind of work a generator's `call` proposes.
fn kind_of(call: &str) -> Option<Kind> {
    match call {
        "life_event" => Some(Kind::LifeEvent),
        "correction" => Some(Kind::Contradiction),
        "story" => Some(Kind::Gap),
        _ => None,
    }
}

impl Config {
    /// Parse and check a configuration.
    pub fn parse(yaml: &str) -> Result<Config, String> {
        let m = parse_missions(yaml).map_err(|e| format!("{FILE}: {e}"))?;
        let prompt = |name: &str| -> Result<String, String> {
            m.prompts
                .get(name)
                .filter(|p| !p.trim().is_empty())
                .cloned()
                .ok_or_else(|| format!("{FILE}: the shared prompt `{name}` is missing or empty"))
        };
        let (system, reader) = (prompt("table")?, prompt("reader")?);
        let (reading, writing) = (prompt("reading")?, prompt("writing")?);
        if m.generators.is_empty() {
            return Err(format!("{FILE}: no generators"));
        }
        let mut generators = Vec::new();
        for g in &m.generators {
            let kind = kind_of(&g.call).ok_or_else(|| {
                format!(
                    "{FILE}: generator `{}` names the call `{}`; the calls that propose are \
                     `life_event`, `correction` and `story`",
                    g.id, g.call
                )
            })?;
            known(
                &format!("generator `{}`", g.id),
                "context",
                &g.context,
                PROPOSAL_CONTEXT,
            )?;
            generators.push(Generator {
                id: g.id.clone(),
                kind,
                weight: g.weight,
                prompt: g.prompt.clone(),
                context: g.context.clone(),
                workflow: g.workflow.clone(),
            });
        }
        if generators.iter().all(|g| g.weight == 0) {
            return Err(format!("{FILE}: every generator has weight 0"));
        }
        for w in &m.workflows {
            check_workflow(w)?;
        }
        Ok(Config {
            system,
            reader,
            keep: m.keep.map_or(DEFAULT_KEEP, |k| k as usize),
            reading,
            writing,
            generators,
            workflows: m.workflows,
        })
    }

    /// Read `<mind>/missions.yaml`. `Ok(None)` when the mind has none — a mind
    /// with no generator gives out the routine bank alone, as it always has.
    pub fn load(mind: &Path) -> Result<Option<Config>, String> {
        let path = mind.join(FILE);
        match std::fs::read_to_string(&path) {
            Ok(text) => Config::parse(&text).map(Some),
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(None),
            Err(e) => Err(format!("{}: {e}", path.display())),
        }
    }

    /// The generator by name.
    pub fn generator(&self, id: &str) -> Option<&Generator> {
        self.generators.iter().find(|g| g.id == id)
    }

    /// The workflow by name.
    pub fn workflow(&self, name: &str) -> Option<&Workflow> {
        self.workflows.iter().find(|w| w.name == name)
    }

    /// The generators in the order the `turn`th draw tries them: weighted
    /// round-robin, so over a full cycle each is first in proportion to its
    /// weight, and the rest follow in declaration order as fallbacks for when the
    /// first finds no work.
    pub fn order(&self, turn: u64) -> Vec<&Generator> {
        let cycle: Vec<&Generator> = self
            .generators
            .iter()
            .flat_map(|g| std::iter::repeat_n(g, g.weight as usize))
            .collect();
        let first = cycle[(turn % cycle.len() as u64) as usize];
        std::iter::once(first)
            .chain(
                self.generators
                    .iter()
                    .filter(|g| g.weight > 0 && g.id != first.id),
            )
            .collect()
    }
}

/// Refuse a workflow naming anything the engine does not implement.
fn check_workflow(w: &Workflow) -> Result<(), String> {
    let at = |step: &str| format!("workflow `{}` step `{step}`", w.name);
    if let Some(year) = w.year.as_deref().filter(|y| !YEARS.contains(y)) {
        return Err(format!(
            "{FILE}: workflow `{}` has the year `{year}`; the years are {}",
            w.name,
            list(YEARS)
        ));
    }
    let checks: Vec<&str> = CHECKS
        .iter()
        .copied()
        .chain([
            COMMITTED,
            CHANGED,
            BRIEF_COPIED,
            REJECTION_QUOTES,
            REJECTION_NAMES_AN_ERA,
        ])
        .collect();
    for s in &w.steps {
        known(&at(&s.name), "checks", &s.checks, &checks)?;
        known(&at(&s.name), "tools", &s.tools, TOOLS)?;
        match (s.by, s.call.as_deref()) {
            (By::Table, Some(READING)) => {
                known(&at(&s.name), "context", &s.context, READING_CONTEXT)?
            }
            (By::Table, Some(call)) => {
                return Err(format!(
                    "{FILE}: {} runs the call `{call}`; a step's call is `{READING}`",
                    at(&s.name)
                ))
            }
            _ => known(&at(&s.name), "context", &s.context, MAKER_CONTEXT)?,
        }
    }
    Ok(())
}

/// Refuse any of `names` not among `known`.
fn known(whose: &str, what: &str, names: &[String], known: &[&str]) -> Result<(), String> {
    match names.iter().find(|n| !known.contains(&n.as_str())) {
        Some(n) => Err(format!(
            "{FILE}: {whose} lists the {what} `{n}`, which the engine does not have; it has {}",
            list(known)
        )),
        None => Ok(()),
    }
}

fn list(names: &[&str]) -> String {
    names
        .iter()
        .map(|n| format!("`{n}`"))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The template npcd's own mind is written to.
    const TEMPLATE: &str = include_str!("../../../../docs/npcd_workflows_current.yaml");

    /// **The template is a configuration the engine runs**: its generators
    /// bound to their workflows, its shared prompts the table's voices, and
    /// every name it gives one the engine has.
    #[test]
    fn the_template_parses_as_the_engines_configuration() {
        let c = Config::parse(TEMPLATE).unwrap();
        assert_eq!(c.keep, 4);
        let generators: Vec<(&str, Kind, &str, u32)> = c
            .generators
            .iter()
            .map(|g| (g.id.as_str(), g.kind, g.workflow.as_str(), g.weight))
            .collect();
        assert_eq!(
            generators,
            [
                ("life-event", Kind::LifeEvent, "life-event", 3),
                ("contradiction", Kind::Contradiction, "correction", 2),
                ("untold", Kind::Gap, "story", 1),
            ]
        );
        assert!(c.system.starts_with("You set work for the Makers"));
        assert!(c.reader.starts_with("You read what the Makers write"));
        assert!(c.reading.starts_with("Above is a draft a Maker wrote"));
        assert!(c.writing.starts_with("You write this world's record"));
        assert!(c.workflow("story").is_some() && c.workflow("nothing").is_none());
    }

    /// **Drawn by weight, never the one switched off, and every other as a
    /// fallback.** Over six turns `life-event` (weight 3) leads three times,
    /// `contradiction` twice and `untold` once.
    #[test]
    fn the_draw_order_follows_the_weights() {
        let c = Config::parse(TEMPLATE).unwrap();
        let firsts: Vec<&str> = (0..6).map(|t| c.order(t)[0].id.as_str()).collect();
        assert_eq!(
            firsts,
            [
                "life-event",
                "life-event",
                "life-event",
                "contradiction",
                "contradiction",
                "untold"
            ]
        );
        let ids: Vec<&str> = c.order(5).iter().map(|g| g.id.as_str()).collect();
        assert_eq!(ids, ["untold", "life-event", "contradiction"]);
    }

    /// **A name the engine does not have is refused when the file is read**,
    /// naming where it is and what the engine has instead.
    #[test]
    fn a_name_the_engine_does_not_have_is_refused() {
        for (from, to, why) in [
            ("call: story\n", "call: saga\n", "names the call `saga`"),
            (
                "          - heading\n",
                "          - spelling\n",
                "the checks `spelling`",
            ),
            (
                "          - storyline\n",
                "          - gossip\n",
                "the context `gossip`",
            ),
            (
                "          - report_rejected\n",
                "          - report_lost\n",
                "the tools `report_lost`",
            ),
            (
                "year: era-opens\n",
                "year: tomorrow\n",
                "has the year `tomorrow`",
            ),
            ("call: reading\n", "call: skim\n", "runs the call `skim`"),
        ] {
            let e = Config::parse(&TEMPLATE.replacen(from, to, 1)).unwrap_err();
            assert!(e.contains(why), "{from:?}: {e}");
        }
        let e = Config::parse(&TEMPLATE.replacen("  writing: |", "  written: |", 1)).unwrap_err();
        assert!(e.contains("the shared prompt `writing` is missing"), "{e}");
    }

    #[test]
    fn a_mind_with_no_file_has_no_generator() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(Config::load(dir.path()), Ok(None));
        std::fs::write(dir.path().join(FILE), TEMPLATE).unwrap();
        assert!(Config::load(dir.path()).unwrap().is_some());
    }
}
