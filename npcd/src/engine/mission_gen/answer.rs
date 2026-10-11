//! The generator's answer: one call per kind of work, or a considered "there is
//! nothing here".
//!
//! **The call is the reasoning.** Asked for a free-text brief, the model wrote
//! briefs that recited a character's description back at it with no event in
//! them, addressed the character instead of the Maker, and named a
//! contradiction between "Zen's project began in 2474" and "Zen woke in 2487".
//! Each kind of work now has its own call whose fields are the decisions the
//! work turns on — an event's date, what happens, what it must agree with; a
//! correction's two clashing sentences and which is wrong — and the engine
//! checks them against the corpus and writes the brief and the path from them.
//! A field the model must fill is a decision it must make; a free brief let it
//! skip every one (`docs/asynchronous_mind_hierarchy.md` §1).
//!
//! **Quotes are checked.** A correction names its two sentences exactly, and
//! each must be in its document; a story quotes the era it must agree with. A
//! contradiction the model cannot quote is one it imagined.
//!
//! **Declining is an answer.** `no_mission` is offered beside every kind, and
//! the ledger records it, so honest agreement between two documents is settled
//! rather than "fixed".

use std::collections::{BTreeMap, HashMap};
use std::ops::RangeInclusive;
use std::path::Path;

use candle_conversation::stencil::{Param as CallParam, ParamType, ToolSpec};
use serde_json::{Map, Value};

use super::corpus::{Corpus, Era, Event};
use super::gates::{Voice, LIFE_MIN_WORDS, STORY_MIN_WORDS};
use super::glossary;
use super::leakage;
use super::material::{belongs, cut, strip_calls, ANCHOR_WORDS, ONLY_EVENT_YEARS};
use super::research;
use super::target::{Kind, Subject, Target};
use crate::engine::journal::tools::arguments;
use crate::engine::life;

/// The call that declines.
pub const NOTHING: &str = "no_mission";
/// The call for a life event.
pub const EVENT: &str = "life_event";
/// The call for a correction.
pub const CORRECTION: &str = "correction";
/// The call for a story.
pub const STORY: &str = "story";

/// What every brief for a piece of the record says about where it is written.
///
/// **The writers' room is not the world.** Nine of eleven pieces judged against
/// the lore had the vault in them — its rooms and machines, the Makers by
/// name, a story whose plot was somebody writing — because that was the most
/// concrete thing in front of the Maker writing it. The gate refuses it at the
/// report (`leakage`); this says it before the first word.
const ELSEWHERE: &str = "\n\nYou write this from the vault, and nothing of the vault belongs in \
                         it: not its rooms or machines, not the Makers, not the writing of it. \
                         Nobody in it is writing a record. Tell it from inside its own world, in \
                         its own time, from what the record says was there.";

/// The shortest quote that counts as a quote.
pub(super) const MIN_QUOTE_CHARS: usize = 20;

/// What `turns` asks for, as a refusal names it.
const TURNS: &str = "the act the moment turns on — who does what that cannot be taken back, and \
                     what is different after it";

/// The act an event or story turns on — refused when it is no act at all.
///
/// **An event is decided before its scene.** A brief whose "what happens" had
/// its subject walk into the dark, sit, and wait "for the end of time, or
/// perhaps for nothing at all" was written faithfully as a mood, and the table
/// failed the draft for having no event: the fault was in the brief. Asked to
/// name the turn first, in its own field, before the scene that leads to it,
/// the answer has to have one.
fn turns(field: &Fields) -> Result<String, String> {
    let t = field.words("turns", 6, TURNS_WORDS, TURNS)?;
    let lower = t.to_lowercase();
    // A life is told to its subject as "you", and "You watch Vasko seal the
    // archive, realizing…" is the subject looking on while somebody else acts.
    let still = [
        "nothing",
        "waits",
        "waiting",
        "remembers",
        "remembering",
        "reflects",
        "you watch",
        "you wait",
        "you sit",
        "you remember",
        "you realize",
        "you realise",
    ];
    // **A realization is not the turn either, whoever has it.** "Engineer
    // Kaelen realizes that the solitude of the gate-laying era is over" was
    // written as a man watching a screen fill with manifests — no event, and
    // the table said so. The act is what the realization makes somebody do, so
    // a turn whose subject's verb is a still one is asked for that act.
    let still_verb = lower
        .split_whitespace()
        .take(4)
        .any(|w| STILL_VERBS.contains(&w.trim_matches(|c: char| !c.is_alphabetic())));
    match still.iter().any(|w| lower.starts_with(w)) || still_verb {
        true => Err(format!(
            "`turns` is {TURNS}. \"{t}\" is not an act: name what somebody does, and what is \
             different after it — not what they realize, notice or remember, but what that \
             makes them do."
        )),
        false => Ok(t),
    }
}

/// Verbs of a subject who does nothing: a turn led by one is a moment of
/// understanding, not an act.
const STILL_VERBS: &[&str] = &[
    "realizes",
    "realises",
    "realize",
    "realise",
    "understands",
    "notices",
    "remembers",
    "reflects",
    "wonders",
    "watches",
    "waits",
];

/// The calls one kind of work may answer with — its own, and declining.
pub fn specs(kind: Kind) -> Vec<ToolSpec> {
    let call = match kind {
        // **The long field last.** The grammar emits the fields in this order,
        // and with `happens` before them the model spent itself on the scene and
        // closed `agrees_with` and `leaves` empty, three tries running — the
        // dream call's field-order lesson again. What the event leaves them with
        // is decided before the scene that leads to it is written.
        Kind::LifeEvent => ToolSpec {
            name: EVENT.into(),
            params: vec![
                text("date"),
                text("title"),
                text_within("leaves", LEAVES_WORDS),
                list_within("agrees_with", ENTRY_WORDS, 0..=FACTS_MOST),
                text_within("turns", TURNS_WORDS),
                text_within("happens", HAPPENS_WORDS),
            ],
        },
        Kind::Contradiction => ToolSpec {
            name: CORRECTION.into(),
            params: vec![
                text("quote_a"),
                text("quote_b"),
                choice("wrong", &["A", "B"]),
                text_within("why", WHY_WORDS),
                text_within("change", CHANGE_WORDS),
            ],
        },
        Kind::Gap => ToolSpec {
            name: STORY.into(),
            params: vec![
                text("title"),
                text_within("when", WHEN_WORDS),
                text_within("where", WHERE_WORDS),
                list_within("who", PERSON_WORDS, 1..=PEOPLE_MOST),
                list_within("agrees_with", ENTRY_WORDS, 1..=FACTS_MOST),
                text_within("turns", TURNS_WORDS),
                text_within("happens", HAPPENS_WORDS),
            ],
        },
    };
    vec![
        call,
        ToolSpec {
            name: NOTHING.into(),
            params: vec![text("why")],
        },
    ]
}

pub(super) fn text(name: &str) -> CallParam {
    CallParam {
        name: name.to_string(),
        ty: ParamType::String,
        required: true,
        enum_values: None,
        items: None,
        min_items: 0,
        max_items: None,
        properties: None,
        nullable: false,
        minimum: None,
        max_tokens: None,
        requires: Vec::new(),
        shapes: Vec::new(),
    }
}

/// [`text`] for a field checked against a cap of `words`: the grammar closes
/// it at [`tokens_for`] that many, so a field that degenerates costs a field's
/// length and not the whole call's.
pub(super) fn text_within(name: &str, words: usize) -> CallParam {
    CallParam {
        max_tokens: Some(tokens_for(words)),
        ..text(name)
    }
}

/// The tokens `words` of prose can take, with room to spare: English runs
/// about four tokens to three words, and a field at its cap must close itself
/// rather than be cut.
fn tokens_for(words: usize) -> u32 {
    (words * 2 + 16) as u32
}

/// A field that is a list: an array of strings, each closed at [`tokens_for`]
/// `words`, and `items` of them.
///
/// **A list has an end.** Bounded only by the grammar's own unrolling, a
/// story's `who` went round to 586 words — the same people named again and
/// again — and the answer was refused for it.
///
/// **A list asked for as a string is written as an array, and lost.** A
/// table's reading planned "the faults array" in its thinking and opened the
/// value with `[`; a string value that opens on anything but its quote is the
/// model skipping it, and the grammar writes it empty — `"faults": ""` in 31
/// readings of 32, the faults it had drafted ("Fault 5: …") with them. What is
/// a list is declared one, and read back joined a line each ([`Fields::get`]).
pub(super) fn list_within(name: &str, words: usize, items: RangeInclusive<usize>) -> CallParam {
    CallParam {
        ty: ParamType::Array,
        items: Some(Box::new(CallParam {
            name: String::new(),
            ..text_within(name, words)
        })),
        min_items: *items.start(),
        max_items: Some(*items.end()),
        ..text(name)
    }
}

/// The caps each generated field is checked against, in words.
const TURNS_WORDS: usize = 60;
const HAPPENS_WORDS: usize = 240;
const AGREES_WORDS: usize = 160;
const LEAVES_WORDS: usize = 60;
const WHY_WORDS: usize = 120;
const CHANGE_WORDS: usize = 120;
const WHEN_WORDS: usize = 40;
const WHERE_WORDS: usize = 40;
const WHO_WORDS: usize = 80;
/// The longest one entry of a list may run: a person and who they are, a fact
/// and where it is written.
const ENTRY_WORDS: usize = 60;
/// The most people one scene names — a scene of more is a crowd, not a cast.
const PEOPLE_MOST: usize = 6;
/// One person in `who`: a name and who they are, not their life — the list
/// whole stays inside [`WHO_WORDS`].
const PERSON_WORDS: usize = 12;
/// The most facts a piece is held to.
const FACTS_MOST: usize = 6;

pub(super) fn choice(name: &str, arms: &[&str]) -> CallParam {
    CallParam {
        enum_values: Some(arms.iter().map(|a| a.to_string()).collect()),
        ..text(name)
    }
}

/// What a checked answer said.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Answer {
    /// A mission to carry out.
    Mission(Proposal),
    /// Nothing here needs doing, and why.
    Nothing(String),
}

/// A mission as the engine wrote it from a checked answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proposal {
    /// What the operation is for, in a line.
    pub objective: String,
    /// What the Maker reads under "What has been asked of you".
    pub brief: String,
    /// The mind path to write or change.
    pub writes: String,
    /// The mind paths to read first, every one real.
    pub reads: Vec<String>,
    /// The fewest words the document may be committed at — see
    /// [`crate::engine::mission::Work::min_words`].
    pub min_words: usize,
    /// What the workflow's prompts are filled from: the proposal's own fields
    /// (`title`, `happens`, `turns` …) and the sections the engine wrote for
    /// it (`world-then`, `worlds-words`, `reads` …), by name.
    pub fields: BTreeMap<String, String>,
}

/// Read and check one answer. The error is worded for the model, which is shown
/// it with its refused attempt and answers again.
pub fn check(raw: &str, kind: Kind, target: &Target, corpus: &Corpus) -> Result<Answer, String> {
    let calls: Vec<String> = specs(kind).into_iter().map(|s| s.name).collect();
    let (name, args) = arguments(raw).ok_or_else(|| {
        format!(
            "That was not a call; answer with `{}`.",
            calls.join("` or `")
        )
    })?;
    let field = Fields(&args);
    match (name.as_str(), kind, &target.subject) {
        (NOTHING, _, _) => {
            let why = field.get("why");
            match why.split_whitespace().count() >= 4 {
                true => Ok(Answer::Nothing(why)),
                false => Err("Say in a sentence why there is nothing here to do.".into()),
            }
        }
        (EVENT, Kind::LifeEvent, Subject::Life { who }) => event(&field, who, corpus),
        (CORRECTION, Kind::Contradiction, Subject::Pair { a, b }) => {
            correction(&field, a, b, corpus)
        }
        (STORY, Kind::Gap, Subject::Era { path }) => story(&field, path, corpus),
        (other, _, _) => Err(format!(
            "There is no call named `{other}` here; answer with `{}`.",
            calls.join("` or `")
        )),
    }
}

/// The arguments of one call, read as trimmed text.
pub(super) struct Fields<'a>(pub(super) &'a Map<String, Value>);

impl Fields<'_> {
    /// The field as trimmed text — a list's entries a line each, its empty
    /// entries dropped and one written twice kept once: a list that goes round
    /// repeats its entries whole.
    pub(super) fn get(&self, k: &str) -> String {
        match self.0.get(k) {
            Some(Value::String(s)) => s.trim().to_string(),
            Some(Value::Array(entries)) => {
                let mut kept: Vec<&str> = Vec::new();
                for e in entries.iter().filter_map(Value::as_str).map(str::trim) {
                    if !e.is_empty() && !kept.contains(&e) {
                        kept.push(e);
                    }
                }
                kept.join("\n")
            }
            _ => String::new(),
        }
    }

    /// The field, refused when it is outside `min..=max` words or goes round
    /// in a loop.
    ///
    /// **A loop is the decode failing, not the model choosing.** A `happens`
    /// that began well ended "She is Zen, and nothing matters. She is Zen, and
    /// everything matters. She is Zen…" for a hundred words; the cap and the
    /// repetition check both refuse it, and the retry is asked again.
    pub(super) fn words(
        &self,
        k: &str,
        min: usize,
        max: usize,
        what: &str,
    ) -> Result<String, String> {
        let v = self.get(k);
        let n = v.split_whitespace().count();
        if n < min {
            return Err(format!("`{k}` is {what}; it needs at least {min} words."));
        }
        if n > max {
            return Err(format!(
                "`{k}` is {n} words; it is {what} and must be under {max}. Say it once, plainly."
            ));
        }
        // **A field that ends in a loop is kept up to the loop.** A reading
        // whose `checked` said everything it had to and then could not stop —
        // "I am done. I will output. I am done. I am ready to output…" — was
        // refused three times over its tail, and the reviewer got no reading
        // at all. What came before the loop is the field.
        if loops(&v) {
            let kept = before_the_loop(&v);
            if kept.split_whitespace().count() >= min && !loops(&kept) {
                return Ok(kept);
            }
            return Err(format!(
                "`{k}` repeats itself. Say {what} once, plainly, and stop."
            ));
        }
        Ok(v)
    }
}

impl Fields<'_> {
    /// [`Self::words`] for a field whose opening is its substance: one over
    /// `max` is kept up to the last sentence that ends within it, rather than
    /// refused.
    ///
    /// **A long reading is a reading.** The table's `checked`, its findings
    /// stated, ran to eleven hundred words of going over them again, and was
    /// refused, read again and refused — the reviewer given nothing for a
    /// reading that had said what it found in its first paragraphs.
    pub(super) fn words_kept(
        &self,
        k: &str,
        min: usize,
        max: usize,
        what: &str,
    ) -> Result<String, String> {
        let v = self.get(k);
        if v.split_whitespace().count() <= max {
            return self.words(k, min, max, what);
        }
        let kept = within_words(&v, max);
        match kept.split_whitespace().count() >= min && !loops(&kept) {
            true => Ok(kept),
            false => self.words(k, min, max, what),
        }
    }
}

/// `text` up to the last sentence that ends within its first `max` words.
fn within_words(text: &str, max: usize) -> String {
    let mut words = 0;
    let mut end = 0;
    for (i, c) in text.char_indices() {
        if c.is_whitespace() && i > 0 && !text[..i].ends_with(char::is_whitespace) {
            words += 1;
            if words >= max {
                break;
            }
        }
        if matches!(c, '.' | '!' | '?') {
            end = i + c.len_utf8();
        }
    }
    text[..end].trim().to_string()
}

/// The opening of a sentence as [`loops`] compares it: its first four words,
/// lower-cased — `None` for one too short to tell apart.
fn opening(sentence: &str) -> Option<String> {
    let words: Vec<String> = sentence
        .split_whitespace()
        .take(4)
        .map(|w| w.to_lowercase())
        .collect();
    (words.len() >= 3).then(|| words.join(" "))
}

/// `text` up to the first sentence of its loop: the earliest place an opening
/// that comes back three times or more first appears.
fn before_the_loop(text: &str) -> String {
    let mut sentences: Vec<(usize, Option<String>)> = Vec::new();
    let mut start = 0;
    for (i, c) in text.char_indices() {
        if matches!(c, '.' | '!' | '?' | ';') {
            sentences.push((start, opening(&text[start..i])));
            start = i + c.len_utf8();
        }
    }
    sentences.push((start, opening(&text[start..])));
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for (_, o) in &sentences {
        if let Some(o) = o {
            *counts.entry(o.as_str()).or_default() += 1;
        }
    }
    let cut = sentences
        .iter()
        .find(|(_, o)| o.as_deref().is_some_and(|o| counts[o] >= 3))
        .map_or(text.len(), |(at, _)| *at);
    text[..cut].trim().to_string()
}

/// Whether a text goes round: some sentence, or the opening of one, comes back
/// three times or more.
fn loops(text: &str) -> bool {
    let mut seen: HashMap<String, usize> = HashMap::new();
    for sentence in text.split(['.', '!', '?', ';']) {
        let words: Vec<String> = sentence
            .split_whitespace()
            .take(4)
            .map(|w| w.to_lowercase())
            .collect();
        if words.len() < 3 {
            continue;
        }
        let n = seen.entry(words.join(" ")).or_default();
        *n += 1;
        if *n >= 3 {
            return true;
        }
    }
    false
}

fn event(field: &Fields, who: &str, corpus: &Corpus) -> Result<Answer, String> {
    let life = corpus.life(who).ok_or("That life is not in this world.")?;
    let date = field.get("date");
    let title = file_title(&field.get("title"))?;
    // **A new event has a name of its own.** Asked for Keeper's next event,
    // the model titled it after the event the material quoted, "The Second the
    // Sky Went Out", three centuries later.
    if life
        .events
        .iter()
        .any(|e| e.title.eq_ignore_ascii_case(&title))
    {
        return Err(format!(
            "\"{title}\" is already the title of an event in this life; this one is a different \
             event and needs its own name."
        ));
    }
    let file = format!("{date} {title}.md");
    let named = life::parse_name(Path::new(&file))
        .map_err(|_| format!("`date` must be `YYYY`, `YYYY-MM` or `YYYY-MM-DD`, not `{date}`."))?;
    if life.events.iter().any(|e| e.date == named.date) {
        return Err(format!(
            "{} is already covered by an event in this life; choose a date it does not cover.",
            named.date
        ));
    }
    let year: u32 = named.date[..4]
        .parse()
        .map_err(|_| "The year is not a number.")?;
    let era = corpus.era_of(year).ok_or_else(|| {
        format!("{year} is before any era of this world; the event must fall inside its history.")
    })?;
    // **Nothing is written after the present.** A Mech stationary since 3086
    // was given "five years later", in a world whose latest era opens in 3087.
    if let Some(now) = corpus.present().filter(|now| year > *now) {
        return Err(format!(
            "{year} is after the world's present ({now}); a life is written up to now, not past it."
        ));
    }
    // Where the next event belongs — the rule the material stated.
    match (life.longest_stretch(), life.events.as_slice()) {
        (Some((from, to)), _) if !(from..=to).contains(&year) => {
            return Err(format!(
                "{year} is not in the stretch this event belongs in: {}",
                belongs(life, corpus)
            ))
        }
        (None, [only])
            if only
                .year()
                .is_some_and(|y| y.abs_diff(year) <= ONLY_EVENT_YEARS) =>
        {
            return Err(format!(
                "{year} is too close to the one event written. {}",
                belongs(life, corpus)
            ))
        }
        _ => {}
    }
    let turn = turns(field)?;
    let happens = field.words("happens", 25, HAPPENS_WORDS, "what happens in the event")?;
    let agrees = field.words(
        "agrees_with",
        0,
        AGREES_WORDS,
        "the facts it must agree with",
    )?;
    // **The grounding is the engine's to state; the model's facts are kept
    // when they say where they are written.** Asked for them, the model left the
    // field empty three times running, and elsewhere asserted "the record
    // states" a thing no document says. What the engine knows for certain is
    // the era the date falls in and the nearest written event, and it says so;
    // what the model adds is kept only when it names an era or event by title,
    // which is what lets the Maker check it.
    // Each fact is kept or dropped on its own, and the kept ones are said as
    // one line: spliced in a line each, a brief read "…(2487-03-08). The
    // Zenling Plague era / The Charge / The Salvation era" as one sentence.
    let titles: Vec<String> = corpus
        .eras
        .iter()
        .map(|e| e.title.to_lowercase())
        .chain(life.events.iter().map(|e| e.title.to_lowercase()))
        .collect();
    let names_a_title = |fact: &str| {
        let lower = fact.to_lowercase();
        titles
            .iter()
            .any(|t| lower.contains(t.as_str()) || lower.contains(t.trim_start_matches("the ")))
    };
    let agrees = agrees
        .lines()
        .map(str::trim)
        .filter(|f| !f.is_empty() && names_a_title(f))
        .collect::<Vec<_>>()
        .join("; ");
    let grounded = !agrees.is_empty();
    let leaves = field.words("leaves", 5, LEAVES_WORDS, "what the event leaves them with")?;

    let writes = format!("layers/life/{who}/{file}");
    if corpus.exists(&writes) {
        return Err(format!("`{writes}` is already written; name a new event."));
    }
    // What the Maker reads before it writes, up to the year and nothing after —
    // see `research`. What it must agree with is the era and the entries of
    // this life it reads that came before.
    let read = research::before_writing(corpus, &writes, year, &format!("{title}. {happens}"));
    let reads: Vec<String> = read.iter().map(|r| r.path.clone()).collect();
    let mut must = format!("{} — it falls in that era of the world", era.title);
    let before: Vec<String> = life
        .events
        .iter()
        .filter(|e| reads.contains(&e.path))
        .map(|e| format!("\"{}\" ({})", e.title, e.date))
        .collect();
    if !before.is_empty() {
        must.push_str(&format!(
            ", and what this life holds just before it: {}",
            before.join(", ")
        ));
    }
    must.push('.');
    if grounded {
        must.push(' ');
        must.push_str(&agrees);
    }
    let grain = named.precision.label();
    let name = &life.name;
    // **Whose voice, said outright.** The Maker writing is somebody else — a
    // Keeper with no body wrote a Zenling's year as "I have no body at all",
    // its own anchor bleeding into a machine that has one. So the brief says who
    // the subject is, and quotes how their written events sound.
    let mut s = format!(
        "Write {name}'s life where the record has nothing: the {grain} {date}, \"{title}\".\n\n"
    );
    // **The anchor is about them, and says so.** Quoted bare, "You are
    // Keeper…" reads as addressed to whoever is reading it — the Maker — and a
    // Maker given it wrote the event as itself.
    //
    // **And all of it the reading sees.** Cut to seventy words, Zen's anchor
    // lost "it has no body — it has a structure" and asks for measurements
    // rather than adjectives; the Maker gave Zen hands, a chair and the taste of
    // copper, and the table — shown the whole anchor — failed it for exactly
    // that.
    if !life.anchor.is_empty() {
        s.push_str(&format!(
            "Who {name} is, from {name}'s own anchor (it speaks to {name} as \"you\"; it is about \
             them, not you): {}\n\n",
            cut(&life.anchor, ANCHOR_WORDS)
        ));
    }
    s.push_str(&format!(
        "What happens: {happens}\n\nIt turns on: {turn}\n\nIt must agree with: {must}\n\nWhat it \
         leaves {name} with: {leaves}\n\n"
    ));
    // **The voice is the life's, stated, and its example agrees with it.**
    // The quoted example used to be the nearest event whatever its voice, and
    // the nearest was a Maker's first-person draft in a life the canon tells as
    // "you" — so the brief asked for one voice and the gate refused it.
    let siblings: Vec<(&Event, String)> = life
        .events
        .iter()
        .filter_map(|e| Some((e, strip_calls(&corpus.text(&e.path)?))))
        .collect();
    let voice = majority_voice(siblings.iter().map(|(_, t)| Voice::of(t))).unwrap_or(Voice::Second);
    // Nearest in time, and from before it where there is one: an example quoted
    // from later in the life hands the drafter a piece of its future.
    let example = siblings
        .iter()
        .filter(|(_, t)| Voice::of(t) == voice)
        .min_by_key(|(e, _)| {
            e.year()
                .map_or((true, u32::MAX), |y| (y > year, y.abs_diff(year)))
        })
        .map(|(_, t)| cut(t, 45));
    s.push_str(&format!(
        "Write it in {}, the way every event of {name}'s life is told{}. Not as yourself, and not \
         with a heading: one scene, moment by moment, of about four hundred words, in short \
         paragraphs, ending on what it left {name} believing or meaning to do.",
        voice_said(voice, name),
        match &example {
            Some(v) => format!(" — like this: \"{v}\""),
            None => String::new(),
        }
    ));
    let world = world_then(corpus, era, Some(year)).unwrap_or_default();
    let words = worlds_words(corpus, &format!("{title}. {happens} {turn}"), era);
    let said = research::said(&read);
    s.push_str(&world);
    s.push_str(&words);
    s.push_str(ELSEWHERE);
    s.push_str(&said);
    let anchor = match life.anchor.is_empty() {
        true => String::new(),
        false => format!(
            "Who {name} is, from {name}'s own anchor (it speaks to {name} as \"you\"; it is \
             about them, not you): {}",
            cut(&life.anchor, ANCHOR_WORDS)
        ),
    };
    let voice_example = example
        .map(|v| format!("How every event of {name}'s life is told — like this: \"{v}\""))
        .unwrap_or_default();
    let fields = fields([
        ("subject", name.clone()),
        ("grain", grain.to_string()),
        ("date", date.to_string()),
        ("title", title.to_string()),
        ("happens", happens),
        ("turns", turn),
        ("agrees", must),
        ("leaves", leaves),
        ("voice", voice_said(voice, name)),
        ("anchor", anchor),
        ("voice-example", voice_example),
        ("world-then", world.trim().to_string()),
        ("worlds-words", words.trim().to_string()),
        ("reads", said.trim().to_string()),
    ]);
    Ok(Answer::Mission(Proposal {
        objective: format!("{name}'s life, {date}: {title}"),
        brief: s,
        writes,
        reads,
        min_words: LIFE_MIN_WORDS,
        fields,
    }))
}

/// A proposal's fields, by name.
fn fields<const N: usize>(pairs: [(&str, String); N]) -> BTreeMap<String, String> {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

/// What the world's own terms mean, for those `scene` and its `era` name —
/// the world's account of each, so the piece is written inside it.
///
/// **A word the writer has to guess is a world it invents.** A story of the
/// Contested Cities put its soldiers in bodies that bleed — which the world's
/// avatars are — and was failed by a reader who did not know it; a life of
/// Keeper gave a bodiless intelligence hands. Both terms are defined in the
/// world's own documents, which neither writer nor reader was shown.
fn worlds_words(corpus: &Corpus, scene: &str, era: &Era) -> String {
    let mut around = vec![era.text.as_str()];
    around.extend(corpus.setting.as_deref());
    glossary::render(&glossary::named(&corpus.terms, scene, &around))
}

/// The world's setting, said as the present it is, and the era the work is
/// set in, said as the time it happens — `None` for a world with no setting.
///
/// **The setting describes now.** Handed as "the world it happens in", a
/// setting that opens "three centuries after the Great War" went word for word
/// into stories set a century after that war, and their reviewers rightly threw
/// them out as anachronisms. The brief says which is which.
fn world_then(corpus: &Corpus, era: &Era, year: Option<u32>) -> Option<String> {
    let setting = corpus.setting.as_ref()?;
    let now = corpus
        .present()
        .map(|y| format!(", in {y}"))
        .unwrap_or_default();
    let when = match year {
        Some(y) => format!("in {y}, in {}", era.title),
        None => format!("in {}", era.title),
    };
    Some(format!(
        "\n\nThe world as it stands now{now}: {} That is the present, not when this happens. This \
         happens {when}, and the world then was as that era tells it: nothing of what came after, \
         and no years counted from now. Nothing in it that this world does not have.",
        cut(setting, 60)
    ))
}

/// The voice most of `voices` are in; `None` for none.
fn majority_voice(voices: impl Iterator<Item = Voice>) -> Option<Voice> {
    let mut counts = [
        (Voice::Second, 0usize),
        (Voice::First, 0),
        (Voice::FirstPlural, 0),
        (Voice::Third, 0),
    ];
    for v in voices {
        if let Some(c) = counts.iter_mut().find(|(k, _)| *k == v) {
            c.1 += 1;
        }
    }
    counts
        .into_iter()
        .filter(|(_, n)| *n > 0)
        .max_by_key(|(_, n)| *n)
        .map(|(v, _)| v)
}

/// A voice, said to the Maker who is to write in it.
fn voice_said(v: Voice, name: &str) -> String {
    match v {
        Voice::Second => format!("the second person — \"you\", speaking to {name}"),
        Voice::First => format!("the first person — \"I\", as {name}"),
        Voice::FirstPlural => format!("the first person plural — \"we\", as {name}"),
        Voice::Third => format!("the third person — {name} by name, \"she\" or \"he\" or \"it\""),
    }
}

fn correction(field: &Fields, a: &str, b: &str, corpus: &Corpus) -> Result<Answer, String> {
    let (ta, tb) = (
        corpus.text(a).unwrap_or_default(),
        corpus.text(b).unwrap_or_default(),
    );
    let qa = quoted(&field.get("quote_a"), &ta, "Document A")?;
    let qb = quoted(&field.get("quote_b"), &tb, "Document B")?;
    let (wrong, right) = match field.get("wrong").as_str() {
        "A" => (a, b),
        "B" => (b, a),
        w => return Err(format!("`wrong` is `A` or `B`, not `{w}`.")),
    };
    let why = field.words("why", 8, WHY_WORDS, "why that side is the wrong one")?;
    let change = field.words("change", 6, CHANGE_WORDS, "the change to make")?;
    let brief = format!(
        "Two documents of the record contradict each other, and one must change.\n\n\
         `{a}` says: \"{qa}\"\n\n`{b}` says: \"{qb}\"\n\n\
         `{wrong}` is the one that is wrong: {why}\n\n\
         The change: {change}\n\n\
         Make it with `file_edit` — the clashing words replaced, nothing else in `{wrong}` \
         touched — so that it agrees with `{right}`."
    );
    let fields = fields([
        ("a", a.to_string()),
        ("b", b.to_string()),
        ("qa", qa),
        ("qb", qb),
        ("wrong", wrong.to_string()),
        ("right", right.to_string()),
        ("why", why),
        ("change", change),
    ]);
    Ok(Answer::Mission(Proposal {
        objective: format!("Correct `{wrong}` so it agrees with `{right}`"),
        brief,
        writes: wrong.to_string(),
        reads: vec![a.to_string(), b.to_string()],
        min_words: 0,
        fields,
    }))
}

fn story(field: &Fields, era: &str, corpus: &Corpus) -> Result<Answer, String> {
    let era_text = corpus.text(era).unwrap_or_default();
    let title = field.get("title");
    let slug = slug(&title);
    if slug.split('-').count() < 2 {
        return Err("`title` needs to be a real title of at least two words.".into());
    }
    let writes = format!("layers/stories/{slug}.md");
    if corpus.exists(&writes) {
        return Err(format!(
            "A story called \"{title}\" is already told; tell another."
        ));
    }
    // A year alone is a when: "2203" was refused for being one word, and the
    // generation spent on it was lost.
    let when = field.words("when", 1, WHEN_WORDS, "when it happens")?;
    let place = field.words("where", 2, WHERE_WORDS, "where it happens")?;
    let who = field
        .words("who", 1, WHO_WORDS, "who is there")?
        .replace('\n', "; ");
    let reused = reused_names(corpus, &who);
    if !reused.is_empty() {
        return Err(format!(
            "{} already {} in other stories of the record. The people this story invents are new \
             people: give them names of their own. Only somebody the record itself names — in its \
             eras, its world, or a life — is the same person again.",
            reused.join(", "),
            if reused.len() == 1 {
                "appears"
            } else {
                "appear"
            }
        ));
    }
    let turn = turns(field)?;
    let happens = field.words("happens", 25, HAPPENS_WORDS, "what happens in it")?;
    let agrees = field.get("agrees_with");
    // A list's entries are each a quote, with or without their own marks.
    let entries = agrees
        .lines()
        .map(|l| l.trim().trim_matches(['"', '“', '”']).trim().to_string());
    let quote = quotes_in(&agrees, MIN_QUOTE_CHARS)
        .into_iter()
        .chain(entries)
        .chain(std::iter::once(agrees.trim().to_string()))
        .chain(sentences_of(&agrees))
        .filter(|q| q.chars().count() >= MIN_QUOTE_CHARS)
        .find(|q| contains(&era_text, q))
        .ok_or_else(|| {
            "`agrees_with` must quote, between double quotes, the era's own words the story has \
             to agree with — and they must be its words exactly."
                .to_string()
        })?;
    let mut brief = format!(
        "Tell a story the record passes over: \"{title}\".\n\n\
         When: {when}\nWhere: {place}\nWho is there: {who}\n\n\
         What happens: {happens}\n\n\
         It turns on: {turn}\n\n\
         It must agree with the era, which says: \"{quote}\"\n\n\
         Write it as a story — past tense, six hundred to a thousand words, beginning with a \
         heading of its title. Show it as it happens: what the people in it do and say, moment by \
         moment, in the place itself. Never summarise what the era already says; the story is \
         what the era leaves out."
    );
    // **The world's own setting, so a story stays inside it.** One set in a
    // post-collapse world of towers and machines came back with quills, parchment
    // and silk weights. Said as the present, beside the era it happens in.
    let (mut world, mut words) = (String::new(), String::new());
    if let Some(e) = corpus.eras.iter().find(|e| e.path == era) {
        world = world_then(corpus, e, None).unwrap_or_default();
        words = worlds_words(
            corpus,
            &format!("{title}. {place}. {who}. {happens} {turn}"),
            e,
        );
    }
    brief.push_str(&world);
    brief.push_str(&words);
    let era_title = corpus
        .eras
        .iter()
        .find(|e| e.path == era)
        .map_or(era, |e| e.title.as_str());
    // Read up to the year the Maker works it in — the era's opening, the same
    // year its time step sets (`canon::set_in`). An era with no year is read
    // on its own.
    let read = match corpus
        .eras
        .iter()
        .find(|e| e.path == era)
        .and_then(|e| e.year)
    {
        Some(year) => research::before_writing(
            corpus,
            &writes,
            year,
            &format!("{title}. {place}. {who}. {happens}"),
        ),
        None => Vec::new(),
    };
    let reads: Vec<String> = match read.is_empty() {
        true => vec![era.to_string()],
        false => read.iter().map(|r| r.path.clone()).collect(),
    };
    let said = research::said(&read);
    brief.push_str(ELSEWHERE);
    brief.push_str(&said);
    let fields = fields([
        ("title", title.clone()),
        ("when", when),
        ("where", place),
        ("who", who),
        ("happens", happens),
        ("turns", turn),
        ("quote", quote),
        ("world-then", world.trim().to_string()),
        ("worlds-words", words.trim().to_string()),
        ("reads", said.trim().to_string()),
    ]);
    Ok(Answer::Mission(Proposal {
        objective: format!("Tell what {era_title} passes over: \"{title}\""),
        brief,
        writes,
        reads,
        min_words: STORY_MIN_WORDS,
        fields,
    }))
}

/// How many told stories a name may already be in before a new story taking it
/// is reusing a person rather than inventing one.
const NAME_STORIES: usize = 2;

/// The names in `who` that [`NAME_STORIES`] or more stories already use and
/// the record's canon never does — the generator's own default names.
///
/// **Invented people are new people.** Left to choose, the model named its
/// invented soldiers and engineers from the same handful: three stories in one
/// afternoon had a "Kaelen", beside a "Vasko", a "Senna" and a "Jorik" each in
/// several, until the record read as one cast wandering through every era. A
/// name the eras, the world's own documents or a life use is the record's, and
/// may come back.
fn reused_names(corpus: &Corpus, who: &str) -> Vec<String> {
    let taken = taken_names(corpus);
    let mut reused: Vec<String> = Vec::new();
    for word in names_in(who) {
        if taken.contains(&word) && !reused.contains(&word) {
            reused.push(word);
        }
    }
    reused
}

/// The names the told stories have invented and used in [`NAME_STORIES`] or
/// more of them, most used first — what a new story's cast is told to stay
/// clear of before it is refused for it.
///
/// **A name the model does not know is taken, it takes.** Refused for one
/// handful of reused names, a story's next answer swapped in another — "Thorne,
/// Voss, Aris", then "Senna, Jorik", then "Halloway" — three tries running, and
/// the target was given up. Said up front, the list is avoided.
pub(super) fn taken_names(corpus: &Corpus) -> Vec<String> {
    // The whole lore — eras and every world document, where the ranks and
    // trades a cast is introduced by are named too — and the lives.
    let lore = leakage::lore(&corpus.root);
    let lives: Vec<String> = corpus
        .lives
        .iter()
        .map(|l| format!("{} {}", l.name, l.anchor).to_lowercase())
        .collect();
    let stories: Vec<String> = corpus
        .stories
        .iter()
        .map(|s| s.text.to_lowercase())
        .collect();
    let mut counted: Vec<(String, usize)> = Vec::new();
    for story in &corpus.stories {
        for word in names_in(&story.text) {
            if counted.iter().any(|(w, _)| *w == word) {
                continue;
            }
            let name = word.to_lowercase();
            if has_word(&lore, &name) || lives.iter().any(|l| has_word(l, &name)) {
                continue;
            }
            let n = stories.iter().filter(|s| has_word(s, &name)).count();
            counted.push((word, n));
        }
    }
    counted.retain(|(_, n)| *n >= NAME_STORIES);
    counted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    counted.into_iter().map(|(w, _)| w).collect()
}

/// The capitalised words of `text` three letters or longer — the candidates
/// for a name — each without a possessive "'s", so "Korr's" is Korr.
fn names_in(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '\'' && c != '’')
        .map(|w| {
            w.strip_suffix("'s")
                .or_else(|| w.strip_suffix("’s"))
                .unwrap_or(w)
                .trim_matches(['\'', '’'])
        })
        .filter(|w| w.chars().count() >= 3 && w.chars().next().is_some_and(char::is_uppercase))
        .map(str::to_string)
        .collect()
}

/// Whether `word` stands in `text` as a whole word.
fn has_word(text: &str, word: &str) -> bool {
    text.match_indices(word).any(|(at, _)| {
        let before = text[..at].chars().next_back();
        let after = text[at + word.len()..].chars().next();
        before.is_none_or(|c| !c.is_alphanumeric()) && after.is_none_or(|c| !c.is_alphanumeric())
    })
}

/// A quote from a document, checked to be in it.
fn quoted(q: &str, text: &str, which: &str) -> Result<String, String> {
    let q = q.trim().trim_matches(['"', '“', '”', '\'']).trim();
    if q.chars().count() < MIN_QUOTE_CHARS {
        return Err(format!(
            "The quote from {which} is too short to be checked; quote the whole clashing \
             sentence."
        ));
    }
    match contains(text, q) {
        true => Ok(q.to_string()),
        false => Err(format!(
            "\"{q}\" is not in {which}. Quote its words exactly, or answer `{NOTHING}` if the two \
             do not actually contradict."
        )),
    }
}

/// Whether `text` holds `q`, ignoring case, spacing, markdown emphasis and
/// links, and curly quotes — the differences a faithful copy can still have.
///
/// **Links especially.** An era reads "mobile fortress-[towers](/tower)" on the
/// page and "mobile fortress-towers" to anybody quoting it; a story's quote was
/// refused three times for being faithful.
pub(super) fn contains(text: &str, q: &str) -> bool {
    let norm = |s: &str| -> String {
        unlink(s)
            .chars()
            .filter(|c| !matches!(c, '*' | '_' | '`'))
            .map(|c| match c {
                '“' | '”' => '"',
                '‘' | '’' => '\'',
                '—' | '–' => '-',
                c => c.to_ascii_lowercase(),
            })
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    };
    let q = norm(q);
    !q.is_empty() && norm(text).contains(&q)
}

/// Markdown links reduced to their words: `[towers](/tower)` → `towers`.
pub(super) fn unlink(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(open) = rest.find('[') {
        let after = &rest[open + 1..];
        let link = after.find("](").and_then(|mid| {
            let close = after[mid + 2..].find(')')?;
            Some((mid, mid + 2 + close + 1))
        });
        match link {
            Some((mid, end)) if !after[..mid].contains(['[', ']']) => {
                out.push_str(&rest[..open]);
                out.push_str(&after[..mid]);
                rest = &after[end..];
            }
            _ => {
                out.push_str(&rest[..=open]);
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

/// Each sentence of a text, with its closing stop — a quote given without
/// quote marks is still a quote, and may be several sentences of which some
/// are the model's own.
fn sentences_of(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for c in s.chars() {
        current.push(c);
        if matches!(c, '.' | '!' | '?') {
            out.push(current.trim().trim_matches('"').trim().to_string());
            current.clear();
        }
    }
    if !current.trim().is_empty() {
        out.push(current.trim().trim_matches('"').trim().to_string());
    }
    out
}

/// Every span between single quotes of at least `min` characters — how a
/// reading quoted, as often as not. An apostrophe opens a span too ("Verdi's
/// role as a '…'"), so some spans are noise; the caller keeps only those the
/// document holds.
pub(super) fn single_quoted(s: &str, min: usize) -> Vec<String> {
    let s = s.replace(['‘', '’'], "'");
    s.split('\'')
        .map(str::trim)
        .filter(|q| q.chars().count() >= min)
        .map(str::to_string)
        .collect()
}

/// Every span between double quotes of at least `min` characters.
pub(super) fn quotes_in(s: &str, min: usize) -> Vec<String> {
    let s = s.replace(['“', '”'], "\"");
    s.split('"')
        .skip(1)
        .step_by(2)
        .map(str::trim)
        .filter(|q| q.chars().count() >= min)
        .map(str::to_string)
        .collect()
}

/// A title as a file can carry it: the characters a file name cannot hold are
/// dropped, and it may not end in a dot.
fn file_title(title: &str) -> Result<String, String> {
    let t: String = title
        .chars()
        .filter(|c| !matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|'))
        .collect();
    let t = t.trim().trim_end_matches('.').trim().to_string();
    match t.split_whitespace().count() {
        0 => Err("`title` is empty.".into()),
        n if n > 10 => Err("`title` is a name for the event, ten words at most.".into()),
        _ => Ok(t),
    }
}

/// A title as a story's file name: lower case, words joined by hyphens.
fn slug(title: &str) -> String {
    title
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

/// Where a document is written: the bench that writes it and the room it stands
/// in, as `go to {room} on {level}`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Desk {
    pub room: String,
    pub level: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::mission_gen::corpus::tests::mind;
    use crate::engine::mission_gen::target::next;

    fn call(name: &str, args: serde_json::Value) -> String {
        format!(
            "<tool_call>\n{}\n</tool_call>",
            serde_json::json!({ "name": name, "arguments": args })
        )
    }

    const HAPPENS: &str =
        "The routing table stays empty for a year. Keeper reconciles the colony's \
                           water every shift because nobody has told it to stop, and in the spring \
                           the first message from Alpha Centauri arrives asking which instances \
                           survived; Keeper answers that it does, and is asked to wait for orders.";
    const TURN: &str = "Keeper answers Alpha Centauri that it survived, and is no longer alone.";

    fn keeper() -> (tempfile::TempDir, Corpus, Target) {
        let dir = mind();
        let c = Corpus::read(dir.path(), "test");
        let t = next(Kind::LifeEvent, &c, &|k, _| k == "life:kaelor", 0).unwrap();
        (dir, c, t)
    }

    fn event_call(date: &str, title: &str) -> String {
        call(
            EVENT,
            serde_json::json!({
                "date": date,
                "title": title,
                "turns": TURN,
                "happens": HAPPENS,
                "agrees_with": "The Fall (2487): the sky went out, and Keeper was reconciling a water schedule when it did.",
                "leaves": "It believes the work is what survives.",
            }),
        )
    }

    /// **The engine writes the path, the reads and the brief from the event's
    /// fields**, and the mission's steps are ones it can see done.
    #[test]
    fn a_life_event_becomes_a_dated_document_and_a_mission_the_engine_can_follow() {
        let (_dir, c, t) = keeper();
        let Answer::Mission(p) = check(
            &event_call("2488", "The Empty Table"),
            Kind::LifeEvent,
            &t,
            &c,
        )
        .unwrap() else {
            panic!("a mission");
        };
        assert_eq!(p.writes, "layers/life/keeper/2488 The Empty Table.md");
        assert_eq!(
            p.reads,
            [
                "layers/eras/the-fall.md",
                "layers/life/keeper/2487-03-08 The Second the Sky Went Out.md"
            ]
        );
        assert!(
            p.brief.starts_with(
                "Write Keeper's life where the record has nothing: the year 2488, \"The Empty \
                 Table\"."
            ),
            "{}",
            p.brief
        );
        assert!(p.brief.contains(HAPPENS));
        // Where it is written is not in it.
        assert!(
            p.brief.contains("nothing of the vault belongs in it"),
            "{}",
            p.brief
        );
        // The setting is the present; the event happens in its own era.
        assert!(
            p.brief.contains(
                "The world as it stands now, in 2787: A world of towers"
            ) && p.brief.contains(
                "That is the present, not when this happens. This happens in 2488, in The Fall, \
                 and the world then was as that era tells it"
            ),
            "{}",
            p.brief
        );
        // The anchor is said to be about Keeper, and the voice is stated
        // outright, with an example in that voice.
        assert!(p.brief.contains(
            "Who Keeper is, from Keeper's own anchor (it speaks to Keeper as \"you\"; it is about \
             them, not you):"
        ));
        let voice = Voice::of(&strip_calls(
            &c.text("layers/life/keeper/2786 The Charge.md").unwrap(),
        ));
        assert!(
            p.brief.contains(&format!(
                "Write it in {}, the way every event of Keeper's life is told",
                voice_said(voice, "Keeper")
            )),
            "{}",
            p.brief
        );
        assert_eq!(
            p.reads,
            [
                "layers/eras/the-fall.md",
                "layers/life/keeper/2487-03-08 The Second the Sky Went Out.md",
            ]
        );
        // What the workflow's write prompt is filled from: the proposal's own
        // fields and the sections the engine wrote, each as the brief says it.
        assert_eq!(p.fields["subject"], "Keeper");
        assert_eq!(p.fields["date"], "2488");
        assert_eq!(p.fields["title"], "The Empty Table");
        assert_eq!(p.fields["turns"], TURN);
        assert_eq!(p.fields["voice"], voice_said(voice, "Keeper"));
        for section in ["anchor", "world-then", "reads"] {
            let text = &p.fields[section];
            assert!(
                !text.is_empty() && p.brief.contains(text.as_str()),
                "{section}: {text}"
            );
        }
    }

    /// Every way a life event can be wrong is refused in words the model can
    /// act on.
    #[test]
    fn a_bad_date_a_covered_date_or_a_thin_event_is_refused() {
        let (_dir, c, t) = keeper();
        for (date, why) in [
            ("2786", "already covered"),
            ("the year after", "must be `YYYY`"),
            ("1999", "before any era"),
        ] {
            let e = check(&event_call(date, "A Year"), Kind::LifeEvent, &t, &c).unwrap_err();
            assert!(e.contains(why), "{date}: {e}");
        }
        let thin = call(
            EVENT,
            serde_json::json!({ "date": "2490", "title": "X", "turns": TURN, "happens": "Things.", "agrees_with": "x", "leaves": "y" }),
        );
        assert!(check(&thin, Kind::LifeEvent, &t, &c)
            .unwrap_err()
            .contains("`happens`"));
        // A turn that is no act — waiting, remembering, nothing — is refused.
        for still in [
            "",
            "Nothing changes for Keeper at all.",
            "Waits in the dark for the end.",
            "You watch Colonel Vasko seal the final entry into the archive.",
            "Engineer Kaelen realizes that the solitude of the gate-laying era is over.",
            "Keeper notices the archive has been flagged for deletion.",
        ] {
            let mood = call(
                EVENT,
                serde_json::json!({ "date": "2490", "title": "X", "turns": still, "happens": HAPPENS, "agrees_with": "x", "leaves": "y" }),
            );
            assert!(
                check(&mood, Kind::LifeEvent, &t, &c)
                    .unwrap_err()
                    .contains("`turns`"),
                "{still}"
            );
        }
        // A title carries no character a file name cannot.
        assert_eq!(file_title("Water: the Year?").unwrap(), "Water the Year");
        // A new event does not take an old event's name.
        let e = check(
            &event_call("2488", "the second the sky went out"),
            Kind::LifeEvent,
            &t,
            &c,
        )
        .unwrap_err();
        assert!(
            e.contains("is already the title of an event in this life"),
            "{e}"
        );
    }

    #[test]
    fn a_lifes_voice_is_its_majority_and_is_said_plainly() {
        use Voice::*;
        assert_eq!(
            majority_voice([Second, First, Second].into_iter()),
            Some(Second)
        );
        assert_eq!(majority_voice(std::iter::empty()), None);
        assert_eq!(
            voice_said(Second, "Creed"),
            "the second person — \"you\", speaking to Creed"
        );
    }

    /// **A contradiction the model cannot quote is one it imagined.** Both
    /// quotes must be in their documents; a correction reads both sides.
    #[test]
    fn a_correction_must_quote_both_documents_exactly() {
        let dir = mind();
        let c = Corpus::read(dir.path(), "test");
        let t = next(Kind::Contradiction, &c, &|_, _| false, 0).unwrap();
        let correction = |qa: &str, wrong: &str| {
            call(
                CORRECTION,
                serde_json::json!({
                    "quote_a": qa,
                    "quote_b": "Everyone went underground.",
                    "wrong": wrong,
                    "why": "The fall is when the sky went out, and the retreat cannot precede it.",
                    "change": "Replace the sentence with the corrected order of events.",
                }),
            )
        };
        let Answer::Mission(p) = check(
            &correction("The **sky** went out over every world", "B"),
            Kind::Contradiction,
            &t,
            &c,
        )
        .unwrap() else {
            panic!("a mission");
        };
        assert_eq!(p.writes, "layers/eras/the-retreat.md");
        assert_eq!(
            p.reads,
            ["layers/eras/the-fall.md", "layers/eras/the-retreat.md"]
        );
        assert!(p
            .brief
            .contains("`layers/eras/the-fall.md` says: \"The **sky** went out over every world\""));
        let e = check(
            &correction("The sky stayed lit all year.", "A"),
            Kind::Contradiction,
            &t,
            &c,
        )
        .unwrap_err();
        assert!(e.contains("is not in Document A"), "{e}");
        assert!(
            check(&correction("Short.", "A"), Kind::Contradiction, &t, &c)
                .unwrap_err()
                .contains("too short")
        );
        assert_eq!(
            check(
                &call(
                    NOTHING,
                    serde_json::json!({ "why": "The two eras agree on every date." })
                ),
                Kind::Contradiction,
                &t,
                &c
            ),
            Ok(Answer::Nothing("The two eras agree on every date.".into()))
        );
    }

    #[test]
    fn every_list_asked_for_has_an_end() {
        for kind in [Kind::LifeEvent, Kind::Contradiction, Kind::Gap] {
            for spec in specs(kind) {
                for p in spec.params.iter().filter(|p| p.ty == ParamType::Array) {
                    let max = p
                        .max_items
                        .unwrap_or_else(|| panic!("{} has no end", p.name));
                    assert!(p.min_items <= max && max <= FACTS_MOST, "{}", p.name);
                }
            }
        }
        // A full cast fits inside what `who` is checked against.
        const _: () = assert!(PEOPLE_MOST * PERSON_WORDS <= WHO_WORDS);
        let who = specs(Kind::Gap)[0]
            .params
            .iter()
            .find(|p| p.name == "who")
            .cloned()
            .unwrap();
        assert_eq!((who.min_items, who.max_items), (1, Some(PEOPLE_MOST)));
    }

    /// A possessive names the person it belongs to: "Korr's" is Korr.
    #[test]
    fn a_possessive_is_the_name_it_belongs_to() {
        assert_eq!(
            names_in("Korr's rifle and Korr, with Elara’s map; Al's."),
            ["Korr", "Korr", "Elara"]
        );
    }

    /// **An invented person is a new person**: a name two stories already use
    /// is refused for a story's cast, the record's own people are not, and a
    /// fresh name passes.
    #[test]
    fn a_story_does_not_reuse_the_names_other_stories_invented() {
        let dir = mind();
        for (file, text) in [
            ("a.md", "# A\n\nKaelen held the line."),
            ("b.md", "# B\n\nKaelen shot first."),
        ] {
            std::fs::write(dir.path().join("layers/stories").join(file), text).unwrap();
        }
        let c = Corpus::read(dir.path(), "test");
        assert_eq!(taken_names(&c), ["Kaelen"]);
        assert_eq!(reused_names(&c, "Kaelen, a gunner; Keeper"), ["Kaelen"]);
        assert!(reused_names(&c, "Orsolya Fenn, a clerk; Keeper").is_empty());
        let t = next(Kind::Gap, &c, &|k, _| k != "era:layers/eras/the-fall.md", 0).unwrap();
        let story = call(
            STORY,
            serde_json::json!({
                "title": "The Water Count",
                "when": "the night of 8 March 2487",
                "where": "the colony's water office",
                "who": ["Kaelen, a gunner", "Ila Bren, a clerk"],
                "turns": TURN,
                "happens": HAPPENS,
                "agrees_with": ["The sky went out over every world at once."],
            }),
        );
        assert!(check(&story, Kind::Gap, &t, &c)
            .unwrap_err()
            .starts_with("Kaelen already appears in other stories"));
    }

    #[test]
    fn a_story_quotes_its_era_and_is_new() {
        let dir = mind();
        let c = Corpus::read(dir.path(), "test");
        let t = next(Kind::Gap, &c, &|k, _| k != "era:layers/eras/the-fall.md", 0).unwrap();
        let story = |title: &str, agrees: &str| {
            call(
                STORY,
                serde_json::json!({
                    "title": title,
                    "when": "the night of 8 March 2487",
                    "where": "the colony's water office",
                    "who": "the clerks on shift",
                    "turns": TURN,
                    "happens": HAPPENS,
                    "agrees_with": agrees,
                }),
            )
        };
        let Answer::Mission(p) = check(
            &story(
                "The Water Schedule",
                "The era says \"the sky went out over every world\" and so it must.",
            ),
            Kind::Gap,
            &t,
            &c,
        )
        .unwrap() else {
            panic!("a mission");
        };
        assert_eq!(p.writes, "layers/stories/the-water-schedule.md");
        assert_eq!(p.reads, ["layers/eras/the-fall.md"]);
        // The world's own account of the terms it is set among.
        assert!(
            p.brief.contains(
                "- **tower** (`layers/world/tower.md`): A tower is a mobile fortress the minds \
                 live in."
            ),
            "{}",
            p.brief
        );
        assert!(check(
            &story("The Charge", "\"The sky went out over every world\""),
            Kind::Gap,
            &t,
            &c
        )
        .unwrap_err()
        .contains("already told"));
        assert!(check(
            &story("A Night", "\"The sky was green that night.\""),
            Kind::Gap,
            &t,
            &c
        )
        .unwrap_err()
        .contains("must quote"));
        // A quote given without quote marks is still a quote.
        assert!(check(
            &story(
                "A Clerk's Night",
                "The sky went out over every world at once."
            ),
            Kind::Gap,
            &t,
            &c
        )
        .is_ok());
    }

    /// **The event goes where the life is silent**, and its facts say where
    /// they are written.
    #[test]
    fn an_event_outside_the_stretch_or_with_ungrounded_facts_is_refused() {
        let (_dir, c, t) = keeper();
        // Keeper's longest silence is 2488–2785; the world's present is 2787.
        let e = check(&event_call("2787", "Later"), Kind::LifeEvent, &t, &c).unwrap_err();
        assert!(e.contains("not in the stretch"), "{e}");
        assert!(e.contains("Between 2488 and 2785"), "{e}");
        let e = check(&event_call("2790", "Later"), Kind::LifeEvent, &t, &c).unwrap_err();
        assert!(e.contains("after the world's present (2787)"), "{e}");
        // Facts that say nowhere where they are written are dropped; the
        // engine's own grounding stands — the era, and what the life holds
        // before the event, never the entry nearest it from its future.
        let floating = call(
            EVENT,
            serde_json::json!({
                "date": "2650",
                "title": "Somewhere",
                "turns": TURN,
                "happens": HAPPENS,
                "agrees_with": "It agrees with everything that has been written about this character so far.",
                "leaves": "It believes the work is what survives.",
            }),
        );
        let Answer::Mission(p) = check(&floating, Kind::LifeEvent, &t, &c).unwrap() else {
            panic!("a mission");
        };
        assert!(
            p.brief.contains(
                "It must agree with: The Retreat — it falls in that era of the world, and what \
                 this life holds just before it: \"The Second the Sky Went Out\" \
                 (2487-03-08).\n\n"
            ),
            "{}",
            p.brief
        );
        assert!(!p.brief.contains("everything that has been written"));
    }

    #[test]
    fn a_field_that_goes_round_in_a_loop_is_refused() {
        assert!(loops(
            "She is Zen, and nothing matters. She is Zen, and everything matters. She is Zen, \
             and neither is true."
        ));
        assert!(!loops(HAPPENS));
    }

    /// **A field that ends in a loop is kept up to it**, when what came
    /// before is a field; one that is all loop is still refused.
    #[test]
    fn a_field_is_kept_up_to_its_loop() {
        let tail = "The year 3012 is in the Contested Cities. The draft repeats its first \
                    paragraph. I am done. I will output. I am done. I am ready. I am done. \
                    I will output.";
        assert_eq!(
            before_the_loop(tail),
            "The year 3012 is in the Contested Cities. The draft repeats its first paragraph."
        );
        let args: serde_json::Map<String, serde_json::Value> = serde_json::from_value(
            serde_json::json!({ "checked": tail, "all": "She is Zen, and \
             nothing. She is Zen, and all. She is Zen, and none." }),
        )
        .unwrap();
        let f = Fields(&args);
        assert_eq!(
            f.words("checked", 5, 400, "what you checked").unwrap(),
            "The year 3012 is in the Contested Cities. The draft repeats its first paragraph."
        );
        assert!(f
            .words("all", 5, 400, "what happens")
            .unwrap_err()
            .contains("repeats itself"));
        // Said twice is prose, not a loop: nothing is cut.
        let twice = "The draft repeats a line. The draft repeats a name.";
        assert_eq!(before_the_loop(twice), twice);
    }

    /// **A long field kept to its cap ends on a whole sentence**; one with no
    /// sentence end inside the cap is refused as before.
    #[test]
    fn a_long_field_is_kept_to_its_last_whole_sentence() {
        let long = "It is set in 3012. The Houses fit. The voice is right. And then it goes on";
        assert_eq!(within_words(long, 9), "It is set in 3012. The Houses fit.");
        let args: serde_json::Map<String, serde_json::Value> = serde_json::from_value(
            serde_json::json!({ "checked": long, "run": "one run on sentence with no end at all" }),
        )
        .unwrap();
        let f = Fields(&args);
        assert_eq!(
            f.words_kept("checked", 3, 9, "what you checked").unwrap(),
            "It is set in 3012. The Houses fit."
        );
        assert!(f
            .words_kept("run", 3, 4, "x")
            .unwrap_err()
            .contains("must be under 4"));
        assert_eq!(
            f.words_kept("checked", 3, 400, "x").unwrap(),
            long,
            "under the cap"
        );
    }

    #[test]
    fn a_link_reads_as_its_words_when_a_quote_is_checked() {
        assert_eq!(
            unlink("mobile fortress-[towers](/tower) behind [walls](/w) and [a] note"),
            "mobile fortress-towers behind walls and [a] note"
        );
        assert!(contains(
            "house them in mobile fortress-[towers](/tower) behind dimensional walls",
            "house them in mobile fortress-towers behind dimensional walls"
        ));
    }

    #[test]
    fn an_answer_that_is_not_its_kind_is_refused() {
        let (_dir, c, t) = keeper();
        assert!(check("I think Keeper should…", Kind::LifeEvent, &t, &c)
            .unwrap_err()
            .contains("not a call"));
        let story = call(STORY, serde_json::json!({ "title": "x" }));
        assert!(check(&story, Kind::LifeEvent, &t, &c)
            .unwrap_err()
            .contains("no call named `story`"));
    }

    #[test]
    fn each_kind_answers_with_its_own_call_or_declines() {
        let names = |k| -> Vec<String> { specs(k).into_iter().map(|s| s.name).collect() };
        assert_eq!(names(Kind::LifeEvent), [EVENT, NOTHING]);
        assert_eq!(names(Kind::Contradiction), [CORRECTION, NOTHING]);
        assert_eq!(names(Kind::Gap), [STORY, NOTHING]);
    }
}
