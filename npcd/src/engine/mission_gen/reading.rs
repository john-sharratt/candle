//! The table's reading of a draft, and the review mission it becomes.
//!
//! **The second reading is two readers.** Once a draft is reported done, the
//! table reads it against what it must agree with — its era, the rest of its
//! life, the world — and names its faults, quoting the draft. Then a Maker other
//! than the one who wrote it carries a review: with the table's reading in hand
//! it reads the draft for itself, mends what is wrong in it, and passes the
//! operation, or rejects it.
//!
//! **The reading before the verdict.** Reading closely enough to name faults is
//! the work; a verdict first is a verdict the reading then has to justify. With
//! only a field for problems, a model wrote it empty and kept every document, a
//! Zen dated four years after the Awakening that says "three centuries have
//! passed" among them — so `checked` is its own field and never empty.

use candle_conversation::stencil::ToolSpec;

use super::answer::{choice, contains, quotes_in, single_quoted, text, Desk, Fields};
use super::corpus::Corpus;
use super::gates::{Form, LIFE_MIN_WORDS, STORY_MIN_WORDS};
use crate::engine::journal::tools::arguments;
use crate::engine::mission::{Mission, Origin, Stage, Todo, Work};
use crate::engine::work::what_happens;
use crate::sim::operations::Operation;

/// The call the table reads a draft with.
pub const READING: &str = "reading";

/// The reading call: the reader's working, what was checked, what is wrong,
/// and the table's verdict.
///
/// **The working has its own place.** The call opens straight into its first
/// field, so a reading that had dates to set against the eras worked them out
/// in `checked` — twelve hundred words of "I need to re-read the eras
/// carefully" — and was refused for length three times running, leaving the
/// reviewer nothing. `notes` is where the reading thinks; it is not kept and
/// the reviewer never sees it.
///
/// **What happens is said before the verdict.** The table read a life event as
/// sound in which a door was opened on an empty room and closed again — the
/// prompt's "no event" was one item in a list, and nothing made the reading
/// say what the event was. `event` is that sentence, `none` when there is
/// none, and a draft in which nothing happens is not sound.
pub fn specs() -> Vec<ToolSpec> {
    vec![ToolSpec {
        name: READING.into(),
        params: vec![
            text("notes"),
            text("event"),
            text("checked"),
            text("faults"),
            choice("verdict", &["sound", "mend", "fail"]),
        ],
    }]
}

/// The table's verdict on a draft.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Nothing wrong worth mending.
    Sound,
    /// Faults a reviewer can mend in place.
    Mend,
    /// Faults that run through all of it.
    Fail,
}

impl Verdict {
    fn said(self) -> &'static str {
        match self {
            Verdict::Sound => "sound — the table found nothing worth mending",
            Verdict::Mend => "to be mended — the faults below can be put right in place",
            Verdict::Fail => "failing — the faults below run through all of it",
        }
    }
}

/// A checked reading.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reading {
    /// What happens in the draft, as the table read it — `none` when nothing.
    pub event: String,
    pub checked: String,
    pub faults: String,
    pub verdict: Verdict,
    /// Whether its faults quote the draft. A reading taken without quotes on
    /// the last attempt says so to the reviewer.
    pub quoted: bool,
}

impl Reading {
    /// The reading as the reviewer is given it and the operation keeps it.
    pub fn render(&self) -> String {
        let faults = match (self.faults.trim(), self.verdict) {
            ("", Verdict::Sound) => "Nothing.",
            ("", _) => "See what it checked, above.",
            (f, _) => f,
        };
        // Only a draft to be mended in place is searched for its sentences; a
        // failed one is written anew.
        let unquoted = match (self.quoted, self.verdict) {
            (false, Verdict::Mend) => {
                "\n\nThe table did not point at the sentences themselves; find them in the draft."
            }
            _ => "",
        };
        format!(
            "The table's verdict: {}.\n\nWhat happens in it, as the table read it: {}\n\nWhat \
             the table checked: {}\n\nWhat it found wrong: {faults}{unquoted}",
            self.verdict.said(),
            self.event,
            self.checked
        )
    }
}

/// Read and check the table's reading of the draft at `path`. The error is
/// worded for the model, which is shown it and reads again. `strict` holds a
/// faulted reading to quoting the draft; the last attempt is not strict, so a
/// reading that names real faults without quoting them still reaches the
/// reviewer rather than nothing at all.
pub fn check(raw: &str, path: &str, corpus: &Corpus, strict: bool) -> Result<Reading, String> {
    // **A call that never closed was cut off, not refused.** Told only "that
    // was not a call", a reading whose working had run to the cap wrote the
    // same length again and was cut off again.
    let (name, args) = arguments(raw).ok_or_else(|| match raw.contains("<tool_call>") {
        true => format!(
            "Your answer was cut off before the `{READING}` call closed: it ran too long. Keep \
             `notes` to a few short lines and `checked` under four hundred words."
        ),
        false => format!("That was not a call; answer with `{READING}`."),
    })?;
    if name != READING {
        return Err(format!(
            "There is no call named `{name}`; answer with `{READING}`."
        ));
    }
    let field = Fields(&args);
    let event = field.words(
        "event",
        1,
        80,
        "what happens in the draft, in a sentence — who does what, and how it ends differently \
         than it began; `none` if nothing does",
    )?;
    let checked = field.words(
        "checked",
        20,
        700,
        "what you checked in the draft and what you found — its year and era, its facts, its voice",
    )?;
    // **A good reading is long.** One that caught a date contradicted by the
    // document's own arithmetic, a human motive in a machine's mouth, a clash
    // with the era and a paragraph going round in circles ran to 369 words.
    let faults = field.words("faults", 0, 600, "what is wrong with the draft")?;
    let verdict = match field.get("verdict").as_str() {
        "sound" => Verdict::Sound,
        "mend" => Verdict::Mend,
        "fail" => Verdict::Fail,
        v => {
            return Err(format!(
                "`verdict` is `sound`, `mend` or `fail`, not `{v}`."
            ))
        }
    };
    // **Its own `event` decides, on the last attempt.** A reading that said
    // three times running that nothing happens in a story and called it sound
    // each time left the reviewer no reading at all. The last attempt is taken
    // at its word: nothing happens, so the draft is to be mended for it.
    let (verdict, faults) = match verdict == Verdict::Sound && nothing_happens(&event) {
        true if !strict => (Verdict::Mend, NO_EVENT.to_string()),
        _ => (verdict, faults),
    };
    if verdict == Verdict::Sound && nothing_happens(&event) {
        return Err(
            "Your `event` says nothing happens in it, and a draft in which nothing happens is not \
             sound: the record is made of events. Its verdict is `mend` — name the fault: no \
             event, quoting where the scene should turn — or `fail` if nothing of an event is in \
             it at all."
                .into(),
        );
    }
    // **A fault list that says "nothing" is an empty one.** A reading that
    // failed a story set before the first era put every fault in `checked` and
    // wrote "Nothing." in `faults`, and the reviewer was told the table found
    // nothing wrong under a failing verdict.
    let faults = match verdict != Verdict::Sound && denies_faults(&faults) {
        true => String::new(),
        false => faults,
    };
    // **A fault is quoted from the draft.** One that cannot point at the
    // sentences it objects to is an opinion of the draft's general quality, and
    // the reviewer given it would rewrite what was right along with what was
    // not. The quote is looked for wherever the reading wrote it: one that found
    // Zen in a command room four years after the era has it leave the galaxy
    // quoted the sentence in `checked` and left `faults` empty.
    let quoted = verdict == Verdict::Sound || {
        let text = corpus.text(path).unwrap_or_default();
        quotes_draft(&format!("{faults}\n{checked}"), &text)
    };
    // **Show the shape, not only the rule.** Told only that a fault must
    // quote, readings went on describing faults in their own words three tries
    // running — "the war with Zen was not won by nobody" — and quoted nothing.
    // **Quoting is for mending.** A draft to be mended in place needs its
    // wrong sentences pointed at; one failed through and through is written
    // anew, and its readings were refused twice for not quoting the faults of
    // a piece whose fault was all of it.
    if strict && !quoted && verdict == Verdict::Mend {
        return Err(
            "A fault must quote the draft's own words that are wrong — copied exactly, between \
             double quotes — so the reviewer knows what to mend and what to keep. Write each fault \
             in `faults` like this: \"<the draft's sentence, copied>\" — what is wrong with it and \
             what it should say. If nothing in it is wrong, the verdict is `sound`."
                .into(),
        );
    }
    Ok(Reading {
        event,
        checked,
        faults,
        verdict,
        quoted,
    })
}

/// Whether `said` quotes `text` — a span of at least [`FAULT_QUOTE_CHARS`]
/// between double or single quotes that the text holds.
pub fn quotes_draft(said: &str, text: &str) -> bool {
    quotes_in(said, FAULT_QUOTE_CHARS)
        .iter()
        .chain(&single_quoted(said, FAULT_QUOTE_CHARS))
        .any(|q| quoted_from(text, q))
}

/// The fault a reading that found no event is given, when it would not name
/// one itself.
const NO_EVENT: &str = "No event: nothing happens in it. Nobody does anything that changes how \
                        the moment ends — it is a room, a mood or a memory. Write it again as the \
                        scene its brief asks for: someone wants something, acts, and it costs \
                        them.";

/// Whether `event` says nothing happens — "none", "Nothing happens.".
fn nothing_happens(event: &str) -> bool {
    let said = event
        .trim()
        .trim_end_matches(['.', '!'])
        .trim_matches('`')
        .to_ascii_lowercase();
    matches!(
        said.as_str(),
        "none" | "nothing" | "nothing happens" | "no event" | "n/a"
    )
}

/// Whether `faults` says there are none — "Nothing.", "None", "No faults".
fn denies_faults(faults: &str) -> bool {
    let said = faults
        .trim()
        .trim_end_matches(['.', '!'])
        .to_ascii_lowercase();
    matches!(
        said.as_str(),
        "nothing" | "none" | "n/a" | "no faults" | "nothing wrong" | "nothing found"
    )
}

/// The shortest span a fault may quote. Shorter than a whole sentence: a fault
/// is as often a phrase — "three hundred years" in a life sixty years long —
/// and a reading that quoted exactly that was refused three times.
const FAULT_QUOTE_CHARS: usize = 12;

/// Whether `q` is quoted from `text`, allowing what a reader leaves out with
/// an ellipsis: each piece either side of one must be in the text.
///
/// **A long sentence is quoted shortened.** A reading that rightly failed a
/// draft quoted "Paxon Vael came in while I was at the lift…" three times and
/// was refused three times, because the ellipsis is not in the draft.
fn quoted_from(text: &str, q: &str) -> bool {
    let pieces: Vec<&str> = q
        .split(['…'])
        .flat_map(|p| p.split("..."))
        .map(|p| p.trim().trim_matches(['"', '\'', ',', ';', ':', ' ']))
        .filter(|p| !p.is_empty())
        .collect();
    pieces
        .iter()
        .any(|p| p.chars().count() >= FAULT_QUOTE_CHARS)
        && pieces
            .iter()
            .filter(|p| p.chars().count() >= 8)
            .all(|p| contains(text, p))
}

/// What a review reads besides the draft: what the draft answers to. For a
/// life event, the era its year falls in and the nearest other event of that
/// life; for a story, the era it tells; for a correction, the document it was
/// corrected against.
fn context(op: &Operation, corpus: &Corpus) -> Vec<String> {
    let doc = op.document.as_str();
    let mut reads = Vec::new();
    if let Some(era) = op.target.strip_prefix("era:") {
        reads.push(era.to_string());
    }
    if let Some((a, b)) = op
        .target
        .strip_prefix("pair:")
        .and_then(|p| p.split_once('|'))
    {
        reads.extend([a, b].into_iter().filter(|p| *p != doc).map(str::to_string));
    }
    if let Some(who) = doc
        .strip_prefix("layers/life/")
        .and_then(|rest| rest.split('/').next())
    {
        let year: Option<u32> = doc
            .rsplit('/')
            .next()
            .and_then(|f| f.get(..4))
            .and_then(|y| y.parse().ok());
        if let Some(era) = year.and_then(|y| corpus.era_of(y)) {
            reads.push(era.path.clone());
        }
        if let (Some(l), Some(y)) = (corpus.life(who), year) {
            if let Some(e) = l
                .events
                .iter()
                .filter(|e| e.path != doc)
                .min_by_key(|e| e.year().map_or(u32::MAX, |ey| ey.abs_diff(y)))
            {
                reads.push(e.path.clone());
            }
        }
    }
    reads.dedup();
    reads
}

/// The review mission for operation `op`, carrying the table's `reading`.
/// `mend` is whether the table's reading found faults: when it did, the review
/// carries a repair step — the draft changed and committed — that both
/// verdicts wait for.
///
/// **Repair before verdict.** Left to judge a draft the table had found
/// faults in, reviewers rejected every one — nine of nine, three of them for
/// the wrong voice they had tried and failed to change a line at a time. A
/// reviewer that must first mend the draft, rewriting it whole when its voice
/// is wrong, saves what can be saved, and rejects only what its mending could
/// not.
///
/// **A failed draft is written anew.** Where the table's verdict is `fail` its
/// faults run through all of it, and a reviewer composing with that draft in
/// front of it wrote the same piece back — the vault's light rings and air
/// handlers in a story of the Final Battle. So the review carries `anew`, and
/// the reviewer writes the piece again from what it was to tell.
pub fn review_mission(
    op: &Operation,
    reading: &str,
    verdict: Option<Verdict>,
    corpus: &Corpus,
    desk: Option<&Desk>,
) -> Mission {
    let mend = verdict != Some(Verdict::Sound);
    let anew = verdict == Some(Verdict::Fail);
    let doc = &op.document;
    let mut reads = vec![doc.clone()];
    reads.extend(context(op, corpus));
    let mut todo = Vec::new();
    if let Some(d) = desk {
        todo.push(Todo::new(format!("go to {} on {}", d.room, d.level)));
    }
    for r in &reads {
        todo.push(Todo::new(format!("read {r}")));
    }
    if mend {
        todo.push(Todo::new(format!(
            "change {doc} and commit it, mending what the table found"
        )));
    }
    todo.push(Todo::report("go back to the table and report your verdict"));
    let repair = match (anew, mend) {
        (true, _) => {
            "First, write it anew. The table found its faults run through all of it, so it is \
                 not mended a line at a time: sit down with `compose` and write the piece again \
                 whole, from what it was to tell — the operation's aim above, the era and the \
                 documents you read — as a scene in that world, keeping nothing of the failed \
                 draft but what the table found right in it. Then `bench_commit` it. A draft \
                 that can be saved is saved by you; that is the work."
        }
        (false, true) => {
            "First, mend it. Put right every fault the table found, in the draft itself: a \
                 wrong date or fact with `file_edit`; a line said twice, cut; a paragraph that \
                 runs on, broken where the scene moves. If it is told in the wrong voice, or a \
                 scene is summary, write it again whole with `compose` — the same events, in \
                 the right voice, as a scene — keeping everything that is right in it. Then \
                 `bench_commit` it. A draft that can be saved is saved by you; that is the work."
        }
        (false, false) => {
            "The table found nothing to mend. Read it for yourself; if you find something, \
                  mend it with `file_edit` and `bench_commit` it."
        }
    };
    // **The review answers to what the draft was to tell.** Given only the
    // operation's one-line aim and the table's reading, a reviewer writing a
    // life event anew wrote a mood about a door: the event its brief named was
    // nowhere in front of it.
    let told = match what_happens(&op.brief) {
        Some(h) => format!("What it was to tell, from its brief — What happens: {h}\n\n"),
        None => String::new(),
    };
    let brief = format!(
        "{name} — {objective}.\n\n\
         Another Maker drafted `{doc}`. You are its second reader: nothing of it stands in the \
         record until you pass it.\n\n\
         {told}\
         {reading}\n\n\
         {repair}\n\n\
         Then give your verdict at the table. When it stands, `report_done` with what you \
         checked and what you changed. Only if your mending could not save it — the events \
         themselves set in the wrong era, or nothing of the subject in it at all — \
         `report_rejected` with why. A rejected draft leaves the record.",
        name = op.name,
        objective = op.objective,
    );
    Mission::new(
        brief,
        todo,
        Origin::Generated {
            generator: op.generator.clone(),
            target: op.target.clone(),
            operation: op.id,
            stage: Stage::Review,
        },
    )
    .with_work(Work {
        writes: doc.clone(),
        reads: reads.clone(),
        min_words: match Form::of(doc) {
            Form::LifeEvent => LIFE_MIN_WORDS,
            Form::Story => STORY_MIN_WORDS,
            Form::Other => 0,
        },
        edit_optional: !mend,
        anew,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::mission_gen::corpus::tests::mind;
    use crate::sim::operations::Operations;

    fn call(args: serde_json::Value) -> String {
        format!(
            "<tool_call>\n{}\n</tool_call>",
            serde_json::json!({ "name": READING, "arguments": args })
        )
    }

    #[test]
    fn the_reading_works_first_and_the_reviewer_never_sees_it() {
        let names: Vec<String> = specs()[0].params.iter().map(|p| p.name.clone()).collect();
        assert_eq!(names, ["notes", "event", "checked", "faults", "verdict"]);
        let dir = mind();
        let corpus = Corpus::read(dir.path(), "test");
        let working = "I need to re-read the eras carefully. ".repeat(200);
        let r = check(
            &call(serde_json::json!({
                "notes": working,
                "event": EVENT,
                "checked": CHECKED,
                "faults": "",
                "verdict": "sound"
            })),
            DOC,
            &corpus,
            true,
        )
        .unwrap();
        assert!(!r.render().contains("re-read the eras"));
    }

    const CHECKED: &str = "It is set in 2786, in the Retreat; it says the plan was given, which \
                           the era allows, and it speaks as Keeper's other event does, as we.";
    const DOC: &str = "layers/life/keeper/2786 The Charge.md";
    const EVENT: &str = "Keeper is given the plan and orders the retreat.";

    /// **What happens is said, and a draft in which nothing happens is not
    /// sound.**
    #[test]
    fn a_draft_in_which_nothing_happens_is_not_sound() {
        let dir = mind();
        let c = Corpus::read(dir.path(), "test");
        let read = |event: &str, verdict: &str, strict: bool| {
            check(
                &call(serde_json::json!({
                    "event": event, "checked": CHECKED, "faults": "", "verdict": verdict
                })),
                DOC,
                &c,
                strict,
            )
        };
        for none in ["none", "Nothing happens.", "`none`"] {
            assert!(
                read(none, "sound", true)
                    .unwrap_err()
                    .contains("nothing happens in it"),
                "{none}"
            );
        }
        assert!(read("none", "fail", true).is_ok());
        // The last attempt is taken at its word: nothing happens, so mend.
        let last = read("none", "sound", false).unwrap();
        assert_eq!(last.verdict, Verdict::Mend);
        assert_eq!(last.faults, NO_EVENT);
        assert!(read("", "sound", false).unwrap_err().contains("`event`"));
        // A call cut off at the cap is told so.
        let cut = "<tool_call>\n{\"name\": \"reading\", \"arguments\": {\"notes\": \"2786 is in";
        assert!(check(cut, DOC, &c, true)
            .unwrap_err()
            .contains("cut off before the `reading` call closed"));
        assert!(check("I think it is fine.", DOC, &c, true)
            .unwrap_err()
            .contains("That was not a call"));
        assert!(read(EVENT, "sound", true)
            .unwrap()
            .render()
            .contains("What happens in it, as the table read it: Keeper is given the plan"));
    }

    /// **A fault quotes the draft; a sound draft needs none; a reading shows
    /// its work.**
    #[test]
    fn a_reading_is_checked_against_the_draft() {
        let dir = mind();
        let c = Corpus::read(dir.path(), "test");
        let r = |checked: &str, faults: &str, verdict: &str| {
            check(
                &call(serde_json::json!({
                    "event": EVENT, "checked": checked, "faults": faults, "verdict": verdict
                })),
                DOC,
                &c,
                true,
            )
        };
        assert_eq!(
            r(CHECKED, "", "sound").unwrap(),
            Reading {
                event: EVENT.into(),
                checked: CHECKED.into(),
                faults: String::new(),
                verdict: Verdict::Sound,
                quoted: true,
            }
        );
        assert!(r("", "", "sound").unwrap_err().contains("`checked`"));
        assert!(r(CHECKED, "It is far too thin to be a life.", "mend")
            .unwrap_err()
            .contains("must quote"));
        // A failing reading is not held to quoting: the piece is written anew.
        assert!(r(CHECKED, "Nothing of an event is in it at all.", "fail").is_ok());
        assert!(r(
            CHECKED,
            "Keeper's line 'We were given the plan.' says nothing of who.",
            "mend"
        )
        .is_ok());
        let fail = r(CHECKED, "\"We were given the plan.\" is all of it.", "fail").unwrap();
        assert_eq!(fail.verdict, Verdict::Fail);
        assert!(fail.render().starts_with(
            "The table's verdict: failing — the faults below run through all of it.\n\nWhat \
             happens in it, as the table read it: Keeper is given the plan and orders the \
             retreat.\n\nWhat the table checked: It is set in 2786"
        ));
        assert!(r(CHECKED, "", "keep").unwrap_err().contains("`verdict`"));
        // A failing reading whose faults say "Nothing." has none listed.
        let denied = check(
            &call(serde_json::json!({
                "event": EVENT, "checked": CHECKED, "faults": "Nothing.", "verdict": "fail"
            })),
            DOC,
            &c,
            false,
        )
        .unwrap();
        assert_eq!(denied.faults, "");
        assert!(denied
            .render()
            .contains("What it found wrong: See what it checked, above."));

        // The last attempt takes an unquoted reading, and says it is one.
        let loose = check(
            &call(serde_json::json!({
                "event": EVENT, "checked": CHECKED, "faults": "", "verdict": "mend"
            })),
            DOC,
            &c,
            false,
        )
        .unwrap();
        assert!(!loose.quoted);
        assert!(loose.render().ends_with(
            "What it found wrong: See what it checked, above.\n\nThe table did not point at the \
             sentences themselves; find them in the draft."
        ));
    }

    /// **A quote shortened with an ellipsis is still a quote**, so long as
    /// every piece of it is in the draft.
    #[test]
    fn an_ellipsis_shortens_a_quote_without_breaking_it() {
        let text = "Paxon Vael came in while I was at the lift, and I told him the address system \
                    had cycled.";
        assert!(quoted_from(
            text,
            "Paxon Vael came in while I was at the lift..."
        ));
        assert!(quoted_from(
            text,
            "Paxon Vael came in … the address system had cycled"
        ));
        assert!(!quoted_from(
            text,
            "Paxon Vael came in … the address system had failed"
        ));
        assert!(
            !quoted_from(text, "Paxon … lift"),
            "too short to be a quote"
        );
        // A phrase is a quote.
        assert!(quoted_from(text, "address system"));
        assert_eq!(
            single_quoted("it says 'three hundred years' of it", FAULT_QUOTE_CHARS),
            ["three hundred years"]
        );
    }

    /// **The review reads the draft and what it answers to, may leave it as it
    /// is, and is the operation's review stage.**
    #[test]
    fn a_review_mission_reads_the_draft_and_its_context() {
        let dir = mind();
        let c = Corpus::read(dir.path(), "test");
        let mut ops = Operations::default();
        let id = ops.open("life-event", "life:keeper", "Keeper's charge", DOC);
        let m = review_mission(
            ops.get(id).unwrap(),
            "The table's verdict: sound.",
            Some(Verdict::Sound),
            &c,
            None,
        );
        let steps: Vec<&str> = m.todo.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(
            steps,
            [
                "read layers/life/keeper/2786 The Charge.md",
                "read layers/eras/the-retreat.md",
                "read layers/life/keeper/2487-03-08 The Second the Sky Went Out.md",
                "go back to the table and report your verdict",
            ]
        );
        assert_eq!(m.operation(), Some((id, Stage::Review)));
        let w = m.work.as_ref().unwrap();
        assert!(w.edit_optional);
        assert_eq!(w.min_words, LIFE_MIN_WORDS);
        assert!(m.written_up(), "a review need not change the draft");
        assert!(m
            .prompt
            .starts_with("Operation Iron Lantern — Keeper's charge."));

        let sid = ops.open(
            "untold",
            "era:layers/eras/the-fall.md",
            "a story",
            "layers/stories/x.md",
        );
        let m = review_mission(ops.get(sid).unwrap(), "r", Some(Verdict::Sound), &c, None);
        assert_eq!(
            m.work.unwrap().reads,
            ["layers/stories/x.md", "layers/eras/the-fall.md"]
        );
    }

    /// **A draft the table found faults in is mended before the verdict**:
    /// the review carries a repair step, and nothing is reported until the
    /// draft is committed.
    #[test]
    fn a_faulted_draft_is_mended_before_its_verdict() {
        let dir = mind();
        let c = Corpus::read(dir.path(), "test");
        let mut ops = Operations::default();
        let id = ops.open("life-event", "life:keeper", "Keeper's charge", DOC);
        let mut m = review_mission(
            ops.get(id).unwrap(),
            "to be mended",
            Some(Verdict::Mend),
            &c,
            None,
        );
        assert_eq!(
            m.todo[m.todo.len() - 2].text,
            "change layers/life/keeper/2786 The Charge.md and commit it, mending what the table \
             found"
        );
        assert!(!m.work.as_ref().unwrap().edit_optional);
        assert!(!m.work.as_ref().unwrap().anew);
        assert!(!m.written_up(), "not until it is committed");
        assert!(m.prompt.contains("First, mend it."));
        assert!(m.committed(&[DOC.to_string()]));
        assert!(m.written_up());

        // A reading that would not come still sets a mending review.
        let m = review_mission(ops.get(id).unwrap(), "unread", None, &c, None);
        assert!(m.prompt.contains("First, mend it."));
        assert!(!m.work.unwrap().anew);
    }

    /// **A draft the table failed is written anew**: the review carries `anew`,
    /// so the writer is not shown the failed text, and says to write it again.
    #[test]
    fn a_failed_draft_is_written_anew() {
        let dir = mind();
        let c = Corpus::read(dir.path(), "test");
        let mut ops = Operations::default();
        let id = ops.open("life-event", "life:keeper", "Keeper's charge", DOC);
        ops.briefed(
            id,
            "Write Keeper's charge.\n\nWhat happens: Keeper gives the order to charge, and the \
             line breaks.\n\nGo to a desk.",
        );
        let m = review_mission(
            ops.get(id).unwrap(),
            "failing",
            Some(Verdict::Fail),
            &c,
            None,
        );
        let w = m.work.as_ref().unwrap();
        assert!(w.anew);
        assert!(!w.edit_optional);
        assert!(m.prompt.contains("First, write it anew."), "{}", m.prompt);
        assert!(
            m.prompt.contains(
                "What it was to tell, from its brief — What happens: Keeper gives the order to \
                 charge, and the line breaks.\n\n"
            ),
            "{}",
            m.prompt
        );
    }
}
