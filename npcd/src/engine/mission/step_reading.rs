//! Reading what a step asks of the world, so the engine can tell which steps it can
//! see done for itself.
//!
//! A step asks for a room to be in, a machine to read, or somebody to be with or to
//! speak to. These are matched against the world by name, with the article and case
//! set aside. A destination that names a level is that room on that level only. A
//! step that matches none of these is one the engine cannot observe, and it never
//! holds a report up. Document steps name the path they read or write, with case
//! and slashes normalised.

use super::*;

/// What a step asks of the world, as far as the engine can tell: a room to be
/// in, a machine to read, or somebody to be with or speak to. What
/// `report_stuck` is checked against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Aim {
    /// A journey: the [`plain`] name of the room.
    Room(String),
    /// A reading: what the step reads, to match against a machine's name.
    Machine(String),
    /// Being with or speaking to somebody, by the step's own words.
    Person(String),
}

impl Aim {
    /// What `step` asks for, or `None` for a step the engine cannot see.
    pub fn of(step: &str) -> Option<Aim> {
        if let Some(of) = reading_of(step) {
            return Some(Aim::Machine(of));
        }
        let lower = step.trim().to_lowercase();
        if MEETING.iter().chain(SPEAKING).any(|v| lower.starts_with(v)) {
            // A journey to a room is a meeting verb too ("go to"); the caller
            // tries the room first and the person after.
            if let Some(room) = journey_to(step) {
                return Some(Aim::Room(room));
            }
            return Some(Aim::Person(lower));
        }
        journey_to(step).map(Aim::Room)
    }

    /// Whether this aim is the room called `name` on the level called `level`.
    pub fn is_room(&self, name: &str, level: &str) -> bool {
        matches!(self, Aim::Room(room) if at_place(room, &plain(name), &plain(level)))
    }

    /// Whether this aim reads the machine called `name`.
    pub fn is_machine(&self, name: &str) -> bool {
        matches!(self, Aim::Machine(of) if names(of, &plain(name)))
    }

    /// Whether this aim is about the person called `who`.
    pub fn is_person(&self, who: &str) -> bool {
        match self {
            Aim::Person(step) | Aim::Room(step) => mentions(step, who),
            Aim::Machine(_) => false,
        }
    }
}

/// A name as it is compared: lower case, trimmed, with no leading article and no
/// closing full stop, so "The plant room." and "the plant room" are one room.
pub(super) fn plain(name: &str) -> String {
    let lower = name.trim().trim_end_matches('.').trim().to_lowercase();
    lower
        .strip_prefix("the ")
        .map(str::to_string)
        .unwrap_or(lower)
}

/// Whether `said` — the rest of a step after its verb — names `thing`: the whole
/// of it, or it followed by more words ("the coolant valve and read its state").
pub(super) fn names(said: &str, thing: &str) -> bool {
    !thing.is_empty()
        && (said == thing
            || said.starts_with(&format!("{thing} "))
            || said.starts_with(&format!("{thing},")))
}

/// Whether the engine signs `step` off itself, from what it sees: a journey, a
/// reading of a machine, somebody met or spoken to, a document read or
/// committed. Nothing else may sign such a step off on the character's word —
/// asked whether it had written a story, a character said it had tried and
/// could not, and the write was struck as thwarted with no document on the
/// record.
pub fn engine_sees(step: &str) -> bool {
    Aim::of(step).is_some()
        || reading_doc(step).is_some()
        || writing_doc(step).is_some()
        || time_step(step).is_some()
        || is_task_step(step)
        || connecting_to(step).is_some()
}

/// The terminal a step connects to, as a [`plain`] name with the rest of the
/// step after it: "connect to the cluster and read its status" → "cluster and
/// read its status", matched against a terminal's name by [`names`]. `None`
/// for any other step. The step of a mind with no body: what it looks at, it
/// looks at through a terminal ([`Mission::worked_terminal`]).
pub(super) fn connecting_to(step: &str) -> Option<String> {
    let step = step.trim().to_lowercase();
    step.strip_prefix("connect to ").map(plain)
}

/// The step that sends a carrier to a time machine to work in `year`.
pub fn time_step_text(year: u32) -> String {
    format!("set your time to {year} at a time machine")
}

/// The year a [`time_step_text`] step asks for. `None` for any other step.
pub fn time_step(step: &str) -> Option<u32> {
    let rest = step.trim().strip_prefix("set your time to ")?;
    let (year, tail) = rest.split_once(' ')?;
    (tail.trim() == "at a time machine")
        .then(|| year.parse().ok())
        .flatten()
}

/// Whether `said` — a journey's [`plain`] destination — is `room` on `level`,
/// both [`plain`]. A destination that names no level is any room so called.
pub(super) fn at_place(said: &str, room: &str, level: &str) -> bool {
    match split_level(said) {
        (r, Some(l)) => names(&r, room) && l == level,
        (r, None) => names(&r, room),
    }
}

/// A destination and the level it names, when it names one: "band one on the
/// casting level" → ("band one", Some("casting level")). Only a trailing "on …
/// level" is a level, so "the room on the left" stays one name.
pub(super) fn split_level(said: &str) -> (String, Option<String>) {
    if let Some((room, level)) = said.rsplit_once(" on ") {
        let level = plain(level);
        if level.ends_with("level") {
            return (room.trim().to_string(), Some(level));
        }
    }
    (said.to_string(), None)
}

/// A mind path as compared: trimmed, forward slashes, lower case, no leading
/// slash — "Layers/eras/x.md" and "/layers/eras/x.md" are one document.
pub(super) fn plain_path(path: &str) -> String {
    path.trim()
        .trim_matches('`')
        .replace('\\', "/")
        .trim_start_matches('/')
        .to_lowercase()
}

/// The document a reading step reads: "read layers/eras/x.md" → its
/// [`plain_path`]. `None` for any other step, or one naming no document.
pub(super) fn reading_doc(step: &str) -> Option<String> {
    doc_after(step, "read ")
}

/// The document a writing step writes: "write layers/life/x/2510 y.md and
/// commit it" → its [`plain_path`]. `None` for any other step.
pub(super) fn writing_doc(step: &str) -> Option<String> {
    doc_after(step, "write ").or_else(|| doc_after(step, "change "))
}

/// The document path a step names after `verb`, up to its `.md`/`.yaml` end, as
/// [`plain_path`] compares it.
pub(super) fn doc_after(step: &str, verb: &str) -> Option<String> {
    raw_doc_after(step, verb).map(|p| plain_path(&p))
}

/// The same path exactly as the step spells it — what a character is told to
/// name, so a new document is created with its title's own capitals.
pub(super) fn raw_doc_after(step: &str, verb: &str) -> Option<String> {
    let rest = step.trim().strip_prefix(verb)?;
    // ASCII lowering keeps every byte where it was, so an offset found in
    // `lower` is a char boundary in `rest`; full Unicode lowering does not
    // (`İ` grows a byte) and a title holding one sliced mid-character.
    let lower = rest.to_ascii_lowercase();
    let end = [".md", ".yaml"]
        .iter()
        .filter_map(|ext| lower.find(ext).map(|at| at + ext.len()))
        .min()?;
    Some(rest[..end].trim().trim_matches('`').to_string())
}

/// A step about a document: one to read, or one to write and commit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DocStep {
    Read(String),
    Write(String),
}

impl DocStep {
    /// The document step `step` is, with its path as the step spells it.
    pub fn of(step: &str) -> Option<DocStep> {
        raw_doc_after(step, "read ").map(DocStep::Read).or_else(|| {
            raw_doc_after(step, "write ")
                .or_else(|| raw_doc_after(step, "change "))
                .map(DocStep::Write)
        })
    }
}

/// Where a journey step leads, as a [`plain`] name: "go to the plant room" →
/// "plant room". `None` for a step that is not a journey.
pub(super) fn journey_to(step: &str) -> Option<String> {
    let step = step.trim().to_lowercase();
    [
        "go to ",
        "go back to ",
        "travel to ",
        "walk to ",
        "head to ",
        "make your way to ",
    ]
    .iter()
    .find_map(|verb| step.strip_prefix(verb))
    .map(plain)
}

/// How a step that is about being with somebody begins.
pub(super) const MEETING: &[&str] = &[
    "find ",
    "go and find ",
    "go to ",
    "travel to ",
    "visit ",
    "seek out ",
    "look in on ",
    "meet ",
];

/// How a step that puts work on the order table begins (`order_post`).
pub(super) const POSTING: &str = "post work";

/// How a step that reads what came back to the order table begins
/// (`order_results`).
pub(super) const READING_RESULTS: &str = "read what came back";

/// How a step that reads the carrier's phone begins (`read`).
pub(super) const READING_MESSAGES: &str = "read your messages";

pub(super) const SPEAKING: &[&str] = &[
    "ask ",
    "tell ",
    "message ",
    "talk to ",
    "talk with ",
    "speak to ",
    "speak with ",
    "give ",
    "draw ",
    "hear ",
];

/// Whether a lower-cased step names `who` — by their whole name, or by their
/// first name as a word of its own ("ask Paxon what he found").
pub(super) fn mentions(step: &str, who: &str) -> bool {
    let who = who.trim().to_lowercase();
    if who.is_empty() {
        return false;
    }
    if step.contains(&who) {
        return true;
    }
    // A first name, never an article: "the channel" once matched every step
    // that said "go to the …", and a message to the channel signed off journeys
    // nobody made.
    let first = who.split_whitespace().next().unwrap_or_default();
    !matches!(first, "the" | "a" | "an")
        && who.split_whitespace().count() > 1
        && first.len() >= 3
        && step
            .split(|c: char| !c.is_alphanumeric() && c != '\'')
            .any(|word| word.trim_end_matches("'s") == first)
}

/// What a reading step reads, as a [`plain`] name: "scan the coolant valve and
/// read what state it is in" → "coolant valve and read what state it is in",
/// matched against a machine's name by [`names`]. `None` for any other step.
pub(super) fn reading_of(step: &str) -> Option<String> {
    let step = step.trim().to_lowercase();
    step.strip_prefix("scan ").map(plain)
}
