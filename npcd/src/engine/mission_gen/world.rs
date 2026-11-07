//! A world's own missions, and the templates they are filled from.
//!
//! `missions.yaml` is written for the Makers — its generators read the record
//! and its workflows end with a document on it — and every world falls back to
//! it. A world whose characters are not Makers says what its own work is
//! instead, in files named for it, and those **replace** `missions.yaml` for
//! that world: the record's generators do not run there and its workflows are
//! not loaded there.
//!
//! What a world's files set is its **routines** — the missions a character
//! collects at the table when nothing has been lodged for it, and the templates
//! Zen issues to those who take its orders. A routine is authored whole: the
//! brief the character is handed and the steps it works through. A routine may
//! carry **parameters** — names that appear as `{name}` in its brief and steps —
//! and is then a template, filled with arguments when it is issued (see
//! [`Routine::fill`] and `engine::order_table`). It may also lend
//! world-building acts (`grants`) and name a document to write with its
//! completion checks (`work`). The routines take the place of the built-in bank
//! ([`crate::engine::mission::bank`]). A drawn routine ends, like the bank's,
//! with going back to the table to report.
//!
//! A world keeps its routines two ways, read together: the combined
//! `missions/<world>.yaml`, and one template to a file under
//! `missions/<world>/` (grouped into subfolders for browsing — the folder names
//! carry no behaviour). A `missions/<world>/world.yaml` holds the world-level
//! `draw`. The design is `docs/mission_templates.md` and
//! `docs/mission_generator.md` §"A world's own missions".

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};

use crate::engine::mission::{Mission, Origin, Todo, Work};
use crate::mind::files::MindFiles;

/// The folder a world's own missions stand in.
pub const FOLDER: &str = "missions";

/// One authored routine: a mission a character may be handed whole, or a
/// template Zen fills with arguments and issues.
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Routine {
    /// Its name — what the mission's [`Origin::Random`] carries, and what
    /// `order_post` names to issue it.
    pub id: String,
    /// What the character is asked, in a paragraph. May hold `{name}`
    /// placeholders for the routine's parameters.
    pub brief: String,
    /// The steps it works through, in order — each a plain string, or a map
    /// with `critical: true` for an irreversible one ([`Step`]). May hold
    /// `{name}` placeholders.
    pub steps: Vec<Step>,
    /// The parameters the routine takes — names that may appear as `{name}` in
    /// the brief, the steps and the work. Empty for a routine that takes none,
    /// which is issued and drawn exactly as before.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub params: Vec<String>,
    /// The world-building acts the carrier is lent while it carries this — a
    /// standing mission's own ([`Mission::grants`]).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub grants: Vec<String>,
    /// The document the routine is work on, when it is a doc/coding/deploy
    /// template.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub work: Option<RoutineWork>,
}

/// One step of a routine: a plain instruction, or one flagged `critical` — an
/// irreversible move the carrier may take only with an authorizer's go-ahead
/// (`docs/mission_templates.md` §6). In YAML a bare string is a plain step; a
/// map `{ text: "...", critical: true }` flags one.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum Step {
    Plain(String),
    Flagged {
        text: String,
        #[serde(default)]
        critical: bool,
    },
}

impl Step {
    /// The instruction itself, whichever form it took.
    pub fn text(&self) -> &str {
        match self {
            Step::Plain(s) => s,
            Step::Flagged { text, .. } => text,
        }
    }

    /// Whether it is an irreversible step that waits on an authorizer.
    pub fn is_critical(&self) -> bool {
        matches!(self, Step::Flagged { critical: true, .. })
    }
}

impl From<&str> for Step {
    fn from(s: &str) -> Self {
        Step::Plain(s.to_string())
    }
}

impl From<String> for Step {
    fn from(s: String) -> Self {
        Step::Plain(s)
    }
}

/// What a routine's mission is work on: the document to write, the ones to read
/// first, and the checks it is held to. Any field may hold `{name}`
/// placeholders.
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RoutineWork {
    /// The mind path of the document to write or change.
    pub writes: String,
    /// The mind paths to read before writing, in order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reads: Vec<String>,
    /// The checks the document is held to before the mission may be reported
    /// done — the names in `mission_gen::gates::CHECKS`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub checks: Vec<String>,
}

impl RoutineWork {
    /// The [`Work`] this becomes once its placeholders are filled from `args`.
    fn fill(&self, args: &BTreeMap<String, String>) -> Result<Work, String> {
        Ok(Work {
            writes: substitute(&self.writes, args)?,
            reads: self
                .reads
                .iter()
                .map(|r| substitute(r, args))
                .collect::<Result<_, _>>()?,
            min_words: 0,
            edit_optional: false,
            anew: false,
            checks: self.checks.clone(),
            tools: Vec::new(),
        })
    }

    /// Every `{name}` placeholder this names.
    fn placeholders(&self) -> Vec<String> {
        let mut out = placeholders(&self.writes);
        for r in &self.reads {
            out.extend(placeholders(r));
        }
        out
    }
}

/// One filled step: its text with placeholders replaced, and whether it is
/// critical.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FilledStep {
    pub text: String,
    pub critical: bool,
}

/// A routine filled with its arguments: the mission it produces, bar the report
/// step the table adds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Filled {
    pub brief: String,
    pub steps: Vec<FilledStep>,
    pub grants: Vec<String>,
    pub work: Option<Work>,
}

/// Whether a character collecting at the table with nothing set for it draws
/// one of the routines.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Draw {
    /// It draws one — the routines are the table's standing work.
    #[default]
    Routines,
    /// It draws nothing, and waits at the table for work to be posted. The
    /// routines are the catalogue the poster reads (`order_results`), the
    /// standing jobs it may post when the table is empty
    /// (`docs/zen_moon_mind.md` §8.3).
    None,
}

/// A world's own missions, read from its combined file and its folder.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorldMissions {
    pub routines: Vec<Routine>,
    /// What a character with nothing set for it is handed — see [`Draw`].
    #[serde(default)]
    pub draw: Draw,
    /// Who may approve a critical step (`docs/mission_templates.md` §6), by the
    /// name the world writes for them — the Creator and Yen. Empty where none
    /// is set, and then a critical step can only be held, never approved.
    #[serde(default)]
    pub authorizers: Vec<String>,
}

/// The world-level config a `missions/<world>/world.yaml` carries.
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorldConfig {
    #[serde(default)]
    draw: Draw,
    #[serde(default)]
    authorizers: Vec<String>,
}

/// Whether an argument's value is a stand-in for one — empty, or a word that
/// says the value is not known. "none" is not among them: for a filter or a
/// list it is a value.
fn is_no_value(value: &str) -> bool {
    const NONE: [&str; 6] = ["", "unknown", "?", "n/a", "tbd", "unspecified"];
    let v = value
        .trim()
        .trim_matches(|c| c == '"' || c == '\'')
        .to_lowercase();
    NONE.contains(&v.as_str())
}

/// Where world `world`'s combined missions file stands in the mind.
pub fn path(world: &str) -> String {
    format!("{FOLDER}/{world}.yaml")
}

/// Where world `world`'s per-template folder stands in the mind.
pub fn dir(world: &str) -> String {
    format!("{FOLDER}/{world}")
}

impl Routine {
    /// Parse and check one per-template file.
    pub fn parse(file: &str, yaml: &str) -> Result<Routine, String> {
        let r: Routine = serde_yaml::from_str(yaml).map_err(|e| format!("{file}: {e}"))?;
        check_routine(file, &r)?;
        Ok(r)
    }

    /// This routine filled with `args`: its brief, steps and work with every
    /// `{name}` replaced. Every parameter the routine declares must be in
    /// `args`, and no placeholder may name anything else.
    ///
    /// **A value that stands for no value is none.** Zen, with no build to
    /// name, posted the build check with `build: unknown`, and a Drone was
    /// sent to check that "build unknown" had reached every member.
    pub fn fill(&self, args: &BTreeMap<String, String>) -> Result<Filled, String> {
        for p in &self.params {
            let Some(value) = args.get(p) else {
                return Err(format!(
                    "the mission `{}` needs an argument for `{p}`",
                    self.id
                ));
            };
            if is_no_value(value) {
                return Err(format!(
                    "the mission `{}` needs a value for `{p}`, and \"{}\" is not one: name it, \
                     or post something else",
                    self.id,
                    value.trim()
                ));
            }
        }
        Ok(Filled {
            brief: substitute(&self.brief, args)?,
            steps: self
                .steps
                .iter()
                .map(|s| {
                    Ok(FilledStep {
                        text: substitute(s.text(), args)?,
                        critical: s.is_critical(),
                    })
                })
                .collect::<Result<_, String>>()?,
            grants: self.grants.clone(),
            work: self.work.as_ref().map(|w| w.fill(args)).transpose()?,
        })
    }
}

impl WorldMissions {
    /// Parse and check a world's combined file.
    pub fn parse(file: &str, yaml: &str) -> Result<WorldMissions, String> {
        let m: WorldMissions = serde_yaml::from_str(yaml).map_err(|e| format!("{file}: {e}"))?;
        if m.routines.is_empty() {
            return Err(format!("{file}: no routines"));
        }
        for r in &m.routines {
            check_routine(file, r)?;
        }
        check_unique(&m.routines).map_err(|e| format!("{file}: {e}"))?;
        Ok(m)
    }

    /// Read world `world`'s own missions — its combined file and every template
    /// under its folder, together. `Ok(None)` when it has neither, and the
    /// world runs on `missions.yaml` like any other.
    pub fn load(mind: &MindFiles, world: &str) -> Result<Option<WorldMissions>, String> {
        let mut routines: Vec<Routine> = Vec::new();
        let mut draw = Draw::default();
        let mut authorizers: Vec<String> = Vec::new();
        let mut found = false;

        let file = path(world);
        match mind.read(&file) {
            Some(text) => {
                let m = WorldMissions::parse(&file, &text)?;
                routines.extend(m.routines);
                draw = m.draw;
                authorizers.extend(m.authorizers);
                found = true;
            }
            None if mind.is_file(&file) => return Err(format!("{file} is not UTF-8")),
            None => {}
        }

        let dir = dir(world);
        let cfg = format!("{dir}/world.yaml");
        let mut paths = mind.walk(&dir);
        paths.sort();
        for rel in paths {
            if !rel.ends_with(".yaml") {
                continue;
            }
            let Some(text) = mind.read(&rel) else {
                return Err(format!("{rel} is not UTF-8"));
            };
            if rel == cfg {
                let c: WorldConfig =
                    serde_yaml::from_str(&text).map_err(|e| format!("{rel}: {e}"))?;
                draw = c.draw;
                authorizers.extend(c.authorizers);
            } else {
                routines.push(Routine::parse(&rel, &text)?);
            }
            found = true;
        }

        if !found {
            return Ok(None);
        }
        check_unique(&routines)?;
        Ok(Some(WorldMissions {
            routines,
            draw,
            authorizers,
        }))
    }
}

/// Check one routine: a kebab-case id, a brief, at least one non-empty step,
/// well-formed parameter names, and no placeholder that names a parameter the
/// routine did not declare.
fn check_routine(file: &str, r: &Routine) -> Result<(), String> {
    if !is_kebab(&r.id) {
        return Err(format!(
            "{file}: the routine id `{}` is not kebab-case",
            r.id
        ));
    }
    if r.brief.trim().is_empty() {
        return Err(format!("{file}: routine `{}` has no brief", r.id));
    }
    if r.steps.is_empty() || r.steps.iter().any(|s| s.text().trim().is_empty()) {
        return Err(format!(
            "{file}: routine `{}` needs at least one step, and no empty ones",
            r.id
        ));
    }
    let mut seen = BTreeSet::new();
    for p in &r.params {
        if !is_kebab(p) {
            return Err(format!(
                "{file}: routine `{}` has a parameter `{p}` that is not kebab-case",
                r.id
            ));
        }
        if !seen.insert(p.as_str()) {
            return Err(format!(
                "{file}: routine `{}` names the parameter `{p}` twice",
                r.id
            ));
        }
    }
    let declared: BTreeSet<&str> = r.params.iter().map(String::as_str).collect();
    let mut used = placeholders(&r.brief);
    for s in &r.steps {
        used.extend(placeholders(s.text()));
    }
    if let Some(w) = &r.work {
        used.extend(w.placeholders());
    }
    for name in used {
        if !declared.contains(name.as_str()) {
            return Err(format!(
                "{file}: routine `{}` uses `{{{name}}}`, which is not one of its params",
                r.id
            ));
        }
    }
    Ok(())
}

/// No two routines share an id.
fn check_unique(routines: &[Routine]) -> Result<(), String> {
    let mut seen = BTreeSet::new();
    for r in routines {
        if !seen.insert(r.id.as_str()) {
            return Err(format!("two routines are called `{}`", r.id));
        }
    }
    Ok(())
}

/// A non-empty name of ASCII lower-case letters, digits and hyphens.
fn is_kebab(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

/// Every `{name}` placeholder in `text`, in order — a name of the kebab
/// charset between braces. A `{` that does not open one is ordinary text.
fn placeholders(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < text.len() {
        let rest = &text[i..];
        if let Some(name) = opens_placeholder(rest) {
            out.push(name.to_string());
            i += name.len() + 2;
            continue;
        }
        i += rest.chars().next().map_or(1, char::len_utf8);
    }
    out
}

/// Replace every `{name}` in `text` with its argument, or say which argument is
/// missing. A `{` that does not open a placeholder is left as it stands.
fn substitute(text: &str, args: &BTreeMap<String, String>) -> Result<String, String> {
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < text.len() {
        let rest = &text[i..];
        if let Some(name) = opens_placeholder(rest) {
            match args.get(name) {
                Some(value) => out.push_str(value),
                None => return Err(format!("no argument for `{{{name}}}`")),
            }
            i += name.len() + 2;
            continue;
        }
        let ch = rest.chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    Ok(out)
}

/// If `rest` begins with a `{name}` placeholder, its name — `None` otherwise.
fn opens_placeholder(rest: &str) -> Option<&str> {
    let rest = rest.strip_prefix('{')?;
    let close = rest.find('}')?;
    let name = &rest[..close];
    is_kebab(name).then_some(name)
}

/// The report step the table appends to every drawn or issued routine.
pub fn report_step(table: Option<&str>) -> String {
    match table {
        Some(room) => format!("go back to the table in {room} and report it"),
        None => "go back to the table and report it".to_string(),
    }
}

/// Draw one of `routines` by `seed`, ending with the report at the table in
/// `table` when the world has one. Only a routine that takes no parameters is
/// drawn — one with parameters is issued with arguments, never handed out
/// blank. `None` when there are no parameter-less routines.
pub fn draw(routines: &[Routine], table: Option<&str>, seed: u64) -> Option<Mission> {
    let r = routines
        .iter()
        .filter(|r| r.params.is_empty())
        .min_by_key(|r| {
            let mut h = DefaultHasher::new();
            (seed, r.id.as_str()).hash(&mut h);
            h.finish()
        })?;
    let filled = r.fill(&BTreeMap::new()).ok()?;
    Some(mission_of(
        filled,
        Origin::Random {
            routine: r.id.clone(),
        },
        table,
    ))
}

/// The mission a filled routine becomes, with the report step appended and its
/// grants and work carried onto it.
pub fn mission_of(filled: Filled, origin: Origin, table: Option<&str>) -> Mission {
    let mut todo: Vec<Todo> = filled
        .steps
        .iter()
        .map(|s| match s.critical {
            true => Todo::critical(s.text.as_str()),
            false => Todo::new(s.text.as_str()),
        })
        .collect();
    todo.push(Todo::report(report_step(table)));
    let mut m = Mission::new(filled.brief.as_str(), todo, origin);
    if !filled.grants.is_empty() {
        m = m.granting(filled.grants);
    }
    if let Some(work) = filled.work {
        m = m.with_work(work);
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIFE: &str = "\
routines:
  - id: check-on-the-sharratts
    brief: Look in on the Sharratt family.
    steps:
      - go to the Sharratt room
      - settle how the family is today
  - id: ward-shift
    brief: Take a shift on the ward.
    steps:
      - go to the ward room
";

    fn args(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn a_world_file_parses_into_its_routines() {
        let m = WorldMissions::parse("missions/life.yaml", LIFE).unwrap();
        let ids: Vec<&str> = m.routines.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(ids, ["check-on-the-sharratts", "ward-shift"]);
        let steps: Vec<&str> = m.routines[0].steps.iter().map(|s| s.text()).collect();
        assert_eq!(
            steps,
            ["go to the Sharratt room", "settle how the family is today"]
        );
    }

    /// **A file that would hand out a broken mission is refused when read**,
    /// naming the file and what is wrong.
    #[test]
    fn a_broken_routine_is_refused() {
        for (yaml, why) in [
            ("routines: []\n", "no routines"),
            (
                "routines:\n  - { id: Ward, brief: b, steps: [s] }\n",
                "`Ward` is not kebab-case",
            ),
            (
                "routines:\n  - { id: a, brief: b, steps: [s] }\n  - { id: a, brief: c, steps: [t] }\n",
                "two routines are called `a`",
            ),
            (
                "routines:\n  - { id: a, brief: ' ', steps: [s] }\n",
                "`a` has no brief",
            ),
            (
                "routines:\n  - { id: a, brief: b, steps: [] }\n",
                "`a` needs at least one step",
            ),
            (
                "routines:\n  - { id: a, brief: b, steps: [s], weight: 2 }\n",
                "unknown field `weight`",
            ),
            ("generators: []\n", "unknown field `generators`"),
        ] {
            let e = WorldMissions::parse("missions/w.yaml", yaml).unwrap_err();
            assert!(e.starts_with("missions/w.yaml: "), "{e}");
            assert!(e.contains(why), "{yaml:?}: {e}");
        }
    }

    /// **A placeholder must name one of the routine's params**, or the file is
    /// refused; a declared param that is used is fine, whether in brief, step
    /// or work.
    #[test]
    fn a_placeholder_must_be_a_declared_param() {
        let bad = "routines:\n  - id: a\n    brief: Put {title} on {device}.\n    steps: [go]\n    params: [title]\n";
        let e = WorldMissions::parse("missions/w.yaml", bad).unwrap_err();
        assert_eq!(
            e,
            "missions/w.yaml: routine `a` uses `{device}`, which is not one of its params"
        );
        let good = "routines:\n  - id: a\n    brief: Put {title} on {device}.\n    steps: ['play {title} on {device}']\n    params: [title, device]\n";
        assert!(WorldMissions::parse("missions/w.yaml", good).is_ok());
        let dup = "routines:\n  - id: a\n    brief: b\n    steps: [s]\n    params: [x, x]\n";
        assert!(WorldMissions::parse("missions/w.yaml", dup)
            .unwrap_err()
            .contains("names the parameter `x` twice"));
    }

    /// **A routine fills its brief, steps and work from its arguments**, with
    /// every `{name}` replaced exactly.
    #[test]
    fn a_routine_fills_from_its_arguments() {
        let r = Routine::parse(
            "play-a-film.yaml",
            "id: play-a-film\nbrief: Put {title} on {device}.\nsteps:\n  - go to {room}\n  - put {title} on {device}\nparams: [title, device, room]\n",
        )
        .unwrap();
        let filled = r
            .fill(&args(&[
                ("title", "Dune"),
                ("device", "the living room TV"),
                ("room", "the front room"),
            ]))
            .unwrap();
        assert_eq!(filled.brief, "Put Dune on the living room TV.");
        let steps: Vec<&str> = filled.steps.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(
            steps,
            ["go to the front room", "put Dune on the living room TV"]
        );
        assert!(filled.steps.iter().all(|s| !s.critical));
    }

    /// **A step flagged `critical` carries that flag through fill and onto the
    /// mission's Todo** — the irreversible one the gate holds; plain steps stay
    /// ordinary.
    #[test]
    fn a_critical_step_is_carried_through() {
        let r = Routine::parse(
            "clear.yaml",
            "id: clear\nbrief: Clear {what}.\nparams: [what]\nsteps:\n  - check what {what} is safe to remove\n  - text: delete the old {what}\n    critical: true\n",
        )
        .unwrap();
        let filled = r.fill(&args(&[("what", "logs")])).unwrap();
        assert_eq!(filled.steps[0].text, "check what logs is safe to remove");
        assert!(!filled.steps[0].critical);
        assert_eq!(filled.steps[1].text, "delete the old logs");
        assert!(filled.steps[1].critical);
        let m = mission_of(
            filled,
            Origin::Random {
                routine: "clear".into(),
            },
            None,
        );
        assert!(!m.todo[0].critical);
        assert!(m.todo[1].critical, "the delete step is critical");
        assert!(!m.todo[2].critical, "the report step is ordinary");
    }

    /// **A missing argument is named, not guessed**; an unfilled parameter
    /// refuses the fill.
    #[test]
    fn a_missing_argument_is_refused() {
        let r = Routine::parse(
            "f.yaml",
            "id: f\nbrief: Put {title} on it.\nsteps: [go]\nparams: [title]\n",
        )
        .unwrap();
        assert_eq!(
            r.fill(&args(&[])).unwrap_err(),
            "the mission `f` needs an argument for `title`"
        );
        assert_eq!(
            r.fill(&args(&[("title", "Dune")])).unwrap().brief,
            "Put Dune on it."
        );
        assert!(
            r.fill(&args(&[("title", "none")])).is_ok(),
            "none is a value"
        );
        for none in ["unknown", " Unknown ", "", "\"?\"", "TBD"] {
            assert_eq!(
                r.fill(&args(&[("title", none)])).unwrap_err(),
                format!(
                    "the mission `f` needs a value for `title`, and \"{}\" is not one: name it, \
                     or post something else",
                    none.trim()
                ),
                "{none:?}"
            );
        }
    }

    /// **Work is filled too**, each path substituted, into a `Work` the mission
    /// carries.
    #[test]
    fn a_routine_s_work_is_filled() {
        let r = Routine::parse(
            "f.yaml",
            "id: f\nbrief: Write {doc}.\nsteps: ['write {doc}']\nparams: [doc]\nwork:\n  writes: 'notes/{doc}.md'\n  reads: ['notes/index.md']\n  checks: [prose]\n",
        )
        .unwrap();
        let filled = r.fill(&args(&[("doc", "dune")])).unwrap();
        let work = filled.work.unwrap();
        assert_eq!(work.writes, "notes/dune.md");
        assert_eq!(work.reads, ["notes/index.md"]);
        assert_eq!(work.checks, ["prose"]);
    }

    /// **A world draws its routines unless its file says `draw: none`**, and
    /// keeps them either way.
    #[test]
    fn a_world_may_hand_out_nothing_by_draw() {
        let drawn = WorldMissions::parse("missions/life.yaml", LIFE).unwrap();
        assert_eq!(drawn.draw, Draw::Routines);
        let waiting =
            WorldMissions::parse("missions/life.yaml", &format!("draw: none\n{LIFE}")).unwrap();
        assert_eq!(waiting.draw, Draw::None);
        assert_eq!(waiting.routines, drawn.routines);
        let e = WorldMissions::parse("missions/w.yaml", &format!("draw: sometimes\n{LIFE}"))
            .unwrap_err();
        assert!(e.contains("sometimes"), "{e}");
    }

    #[test]
    fn a_world_with_no_file_has_none() {
        let dir = tempfile::tempdir().unwrap();
        let mind = MindFiles::folder(dir.path());
        assert_eq!(WorldMissions::load(&mind, "life"), Ok(None));
        std::fs::create_dir_all(dir.path().join(FOLDER)).unwrap();
        std::fs::write(dir.path().join(path("life")), LIFE).unwrap();
        let m = WorldMissions::load(&mind, "life").unwrap().unwrap();
        assert_eq!(m.routines.len(), 2);
        assert_eq!(WorldMissions::load(&mind, "battle-cities"), Ok(None));
    }

    /// **A world's templates are read one to a file under its folder**, grouped
    /// into subfolders, with the combined file (if any) and a `world.yaml` for
    /// the draw.
    #[test]
    fn a_world_reads_its_templates_from_a_folder() {
        let tmp = tempfile::tempdir().unwrap();
        let mind = MindFiles::folder(tmp.path());
        let root = tmp.path().join(dir("moon"));
        std::fs::create_dir_all(root.join("devices")).unwrap();
        std::fs::write(root.join("world.yaml"), "draw: none\n").unwrap();
        std::fs::write(
            root.join("devices/play-a-film.yaml"),
            "id: play-a-film\nbrief: Put {title} on {device}.\nsteps: ['put {title} on {device}']\nparams: [title, device]\n",
        )
        .unwrap();
        std::fs::write(
            root.join("ward-shift.yaml"),
            "id: ward-shift\nbrief: Take a shift on the ward.\nsteps: [go to the ward room]\n",
        )
        .unwrap();
        let m = WorldMissions::load(&mind, "moon").unwrap().unwrap();
        assert_eq!(m.draw, Draw::None);
        let ids: Vec<&str> = m.routines.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(ids, ["play-a-film", "ward-shift"]);
        assert_eq!(m.routines[0].params, ["title", "device"]);
    }

    /// **The combined file and the folder are read together**, and an id in
    /// both is refused.
    #[test]
    fn the_file_and_folder_are_one_library() {
        let tmp = tempfile::tempdir().unwrap();
        let mind = MindFiles::folder(tmp.path());
        std::fs::create_dir_all(tmp.path().join(dir("moon"))).unwrap();
        std::fs::write(tmp.path().join(path("moon")), LIFE).unwrap();
        std::fs::write(
            tmp.path().join(dir("moon")).join("extra.yaml"),
            "id: extra\nbrief: An extra job.\nsteps: [go]\n",
        )
        .unwrap();
        let m = WorldMissions::load(&mind, "moon").unwrap().unwrap();
        let mut ids: Vec<&str> = m.routines.iter().map(|r| r.id.as_str()).collect();
        ids.sort_unstable();
        assert_eq!(ids, ["check-on-the-sharratts", "extra", "ward-shift"]);

        std::fs::write(
            tmp.path().join(dir("moon")).join("clash.yaml"),
            "id: ward-shift\nbrief: A clashing job.\nsteps: [go]\n",
        )
        .unwrap();
        assert_eq!(
            WorldMissions::load(&mind, "moon").unwrap_err(),
            "two routines are called `ward-shift`"
        );
    }

    /// **A drawn routine is the authored mission, closed by the report**, and
    /// the same seed draws the same one. A routine with parameters is never
    /// drawn.
    #[test]
    fn a_draw_is_the_authored_mission_ending_at_the_table() {
        let m = WorldMissions::parse("missions/life.yaml", LIFE).unwrap();
        let a = draw(&m.routines, Some("the watch room"), 7).unwrap();
        assert_eq!(a, draw(&m.routines, Some("the watch room"), 7).unwrap());
        let r = m
            .routines
            .iter()
            .find(|r| matches!(&a.origin, Origin::Random { routine } if *routine == r.id))
            .unwrap();
        assert_eq!(a.prompt, r.brief);
        let steps: Vec<&str> = a.todo.iter().map(|t| t.text.as_str()).collect();
        let mut want: Vec<&str> = r.steps.iter().map(|s| s.text()).collect();
        want.push("go back to the table in the watch room and report it");
        assert_eq!(steps, want);
        let drawn: BTreeSet<String> = (0..64)
            .filter_map(|seed| match draw(&m.routines, None, seed)?.origin {
                Origin::Random { routine } => Some(routine),
                _ => None,
            })
            .collect();
        assert_eq!(drawn.len(), 2, "over many seeds every routine is drawn");
        assert_eq!(draw(&[], None, 0), None);

        // A routine with parameters is not drawn.
        let parametered = WorldMissions::parse(
            "missions/w.yaml",
            "routines:\n  - id: p\n    brief: Put {x} on it.\n    steps: ['put {x} on it']\n    params: [x]\n",
        )
        .unwrap();
        assert_eq!(draw(&parametered.routines, None, 0), None);
    }
}
