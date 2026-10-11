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

use candle_conversation::stencil::{Param as CallParam, ParamType, ToolSpec};
use serde_json::{Map, Value};

use super::answer::{choice, contains, quotes_in, single_quoted, text, text_within, Fields};
use super::corpus::Corpus;
use crate::engine::journal::tools::arguments;
use crate::sim::operations::Operation;

/// The call the table reads a draft with.
pub const READING: &str = "reading";

/// The reading call: what happens, what was checked, what is wrong, and the
/// table's verdict.
///
/// **The working is done before the call, not in it.** Opened straight into
/// its first field, a reading that had dates to set against the eras worked
/// them out in `checked` — twelve hundred words of "I need to re-read the eras
/// carefully". Given a `notes` field to work in, it worked there and could not
/// stop, one sentence going round until the cap. The reading thinks in its
/// reasoning block (`decode_call`'s `think`) and the call holds what it found.
///
/// **What happens is said before the verdict.** The table read a life event as
/// sound in which a door was opened on an empty room and closed again — the
/// prompt's "no event" was one item in a list, and nothing made the reading
/// say what the event was. `event` is that sentence, `none` when there is
/// none, and a draft in which nothing happens is not sound.
///
/// **Each field is closed at its own length.** Bounded only by the whole
/// reading's budget, a one-sentence `event` went "datethe is datethe" for
/// eighteen hundred tokens before the call could close.
pub fn specs() -> Vec<ToolSpec> {
    vec![ToolSpec {
        name: READING.into(),
        params: vec![
            text_within("event", EVENT_WORDS),
            text_within("checked", CHECKED_WORDS),
            fault_list(),
            choice("verdict", &["sound", "mend", "fail"]),
        ],
    }]
}

/// The caps a reading's fields are checked against, in words.
const EVENT_WORDS: usize = 80;
const CHECKED_WORDS: usize = 700;
/// The most faults one reading lists. **A good reading is long** — one that
/// caught a date contradicted by the document's own arithmetic, a human motive
/// in a machine's mouth, a clash with the era and a paragraph going round in
/// circles named four — and a list past this is going round itself: one ran
/// to sixty-four entries, the same two passages again and again.
const FAULTS_MOST: usize = 8;
/// A fault's quote: the draft's own words, a phrase to a few sentences.
const QUOTE_WORDS: usize = 80;
/// What is wrong with the quoted words, and what they should say.
const FAULT_WORDS: usize = 120;

/// The faults: a list of the draft's words, each with what is wrong with them.
///
/// **The quote is its own field.** Asked for faults as text "quoting the
/// draft's sentence between double quotes", readings quoted in double quotes,
/// single quotes, after a dash, or bare before one, and four in thirty-two
/// were refused for not quoting what they had quoted. Held apart, the quote is
/// the draft's words and nothing else, and the engine puts the marks round it.
fn fault_list() -> CallParam {
    CallParam {
        ty: ParamType::Array,
        items: Some(Box::new(CallParam {
            name: String::new(),
            ty: ParamType::Object,
            properties: Some(vec![
                text_within("quote", QUOTE_WORDS),
                text_within("fault", FAULT_WORDS),
            ]),
            ..text("")
        })),
        max_items: Some(FAULTS_MOST),
        ..text("faults")
    }
}

/// A reading's faults as the reviewer reads them: each `"quote" — fault`, a
/// line each, the same one written twice kept once.
pub(super) fn faults_of(args: &Map<String, Value>) -> String {
    let entry = |v: &Value| -> Option<String> {
        match v {
            Value::String(s) => Some(s.trim().to_string()),
            Value::Object(o) => {
                let get = |k: &str| o.get(k).and_then(Value::as_str).unwrap_or_default().trim();
                let quote = get("quote").trim_matches(['"', '“', '”', '\'']).trim();
                Some(match quote {
                    "" => get("fault").to_string(),
                    q => format!("\"{q}\" — {}", get("fault")),
                })
            }
            _ => None,
        }
    };
    let mut seen: Vec<String> = Vec::new();
    let entries: Vec<String> = match args.get("faults") {
        Some(Value::Array(list)) => list.iter().filter_map(entry).collect(),
        Some(v) => entry(v).into_iter().collect(),
        None => Vec::new(),
    };
    for e in entries {
        if !e.is_empty() && !seen.contains(&e) {
            seen.push(e);
        }
    }
    seen.join("\n")
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
    /// The verdict as the call names it, and as a workflow's reading step
    /// routes on it: `sound`, `mend` or `fail`.
    pub fn name(self) -> &'static str {
        match self {
            Verdict::Sound => "sound",
            Verdict::Mend => "mend",
            Verdict::Fail => "fail",
        }
    }

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
            "Your answer was cut off before the `{READING}` call closed: it ran too long. Think \
             before the call, and keep `checked` under four hundred words."
        ),
        false => format!("That was not a call; answer with `{READING}`."),
    })?;
    if name != READING {
        return Err(format!(
            "There is no call named `{name}`; answer with `{READING}`."
        ));
    }
    let field = Fields(&args);
    let event = field.words_kept(
        "event",
        1,
        EVENT_WORDS,
        "what happens in the draft, in a sentence — who does what, and how it ends differently \
         than it began; `none` if nothing does",
    )?;
    let checked = field.words_kept(
        "checked",
        20,
        CHECKED_WORDS,
        "what you checked in the draft and what you found — its year and era, its facts, its voice",
    )?;
    let faults = faults_of(&args);
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
    // A draft in which nothing happens has that for its fault.
    let faults =
        match verdict != Verdict::Sound && faults.trim().is_empty() && nothing_happens(&event) {
            true => NO_EVENT.to_string(),
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
pub(super) fn context(op: &Operation, corpus: &Corpus) -> Vec<String> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::mission_gen::corpus::tests::mind;
    use crate::sim::operations::tests::workflows;
    use crate::sim::operations::Operations;

    fn call(args: serde_json::Value) -> String {
        format!(
            "<tool_call>\n{}\n</tool_call>",
            serde_json::json!({ "name": READING, "arguments": args })
        )
    }

    /// **The call holds what was found, not the working**: the reading thinks
    /// before it, and a field the model would work in is not offered.
    #[test]
    fn the_reading_holds_what_was_found_and_no_working() {
        let names: Vec<String> = specs()[0].params.iter().map(|p| p.name.clone()).collect();
        assert_eq!(names, ["event", "checked", "faults", "verdict"]);
    }

    /// **Each field is closed at its own length**, the sentence well before
    /// the account; the verdict is a choice and needs no bound.
    #[test]
    fn each_field_is_bounded_at_its_own_length() {
        let params = &specs()[0].params;
        let bounds: Vec<Option<u32>> = [&params[0], &params[1], &params[3]]
            .iter()
            .map(|p| p.max_tokens)
            .collect();
        assert_eq!(bounds, [Some(176), Some(1416), None]);
        let fault = params[2].items.as_ref().unwrap();
        let fields: Vec<(&str, Option<u32>)> = fault
            .properties
            .as_ref()
            .unwrap()
            .iter()
            .map(|p| (p.name.as_str(), p.max_tokens))
            .collect();
        assert_eq!(fields, [("quote", Some(176)), ("fault", Some(256))]);
    }

    /// **The faults are a list of quotes and what is wrong with each**, at
    /// most [`FAULTS_MOST`]; the engine marks each quote, and a fault written
    /// twice is kept once.
    #[test]
    fn the_faults_are_quotes_each_with_what_is_wrong() {
        let faults = &specs()[0].params[2];
        assert_eq!(
            (
                faults.name.as_str(),
                faults.ty,
                faults.min_items,
                faults.max_items
            ),
            ("faults", ParamType::Array, 0, Some(FAULTS_MOST))
        );
        let dir = mind();
        let c = Corpus::read(dir.path(), "test");
        let told = serde_json::json!({
            "quote": "We were given the plan.", "fault": "It is told, not shown."
        });
        let r = check(
            &call(serde_json::json!({
                "event": EVENT, "checked": CHECKED,
                "faults": [
                    {"quote": "'We were given the plan.'", "fault": "It says nothing of who."},
                    told,
                    told,
                    {"quote": "", "fault": "The voice drifts."}
                ],
                "verdict": "mend"
            })),
            DOC,
            &c,
            true,
        )
        .unwrap();
        assert_eq!(
            r.faults,
            "\"We were given the plan.\" — It says nothing of who.\n\"We were given the plan.\" — \
             It is told, not shown.\nThe voice drifts."
        );
        assert!(r.quoted);
    }

    /// **An `event` cut off mid-loop is kept to its sentence.**
    #[test]
    fn a_runaway_event_is_kept_to_its_sentence() {
        let dir = mind();
        let c = Corpus::read(dir.path(), "test");
        let runaway = format!("{EVENT} {}", "datethe is ".repeat(60));
        let r = check(
            &call(serde_json::json!({
                "event": runaway, "checked": CHECKED, "faults": "", "verdict": "sound"
            })),
            DOC,
            &c,
            true,
        )
        .unwrap();
        assert_eq!(r.event, EVENT);
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
        // One in which nothing happens has that for its fault.
        let none = check(
            &call(serde_json::json!({
                "event": "none", "checked": CHECKED, "faults": "", "verdict": "fail"
            })),
            DOC,
            &c,
            true,
        )
        .unwrap();
        assert_eq!(none.faults, NO_EVENT);
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

    /// **A review reads what the document answers to**: for a life event, the
    /// era its year falls in and the nearest other event of the life; for a
    /// story, the era it tells.
    #[test]
    fn what_a_review_reads_beside_the_document() {
        let dir = mind();
        let c = Corpus::read(dir.path(), "test");
        let mut ops = Operations::default();
        ops.set_workflows(workflows());
        let id = ops
            .open(
                "life-event",
                None,
                "life-event",
                "life:keeper",
                "Keeper's charge",
                DOC,
            )
            .unwrap();
        assert_eq!(
            context(ops.get(id).unwrap(), &c),
            [
                "layers/eras/the-retreat.md",
                "layers/life/keeper/2487-03-08 The Second the Sky Went Out.md",
            ]
        );
        let sid = ops
            .open(
                "story",
                None,
                "untold",
                "era:layers/eras/the-fall.md",
                "a story",
                "layers/stories/x.md",
            )
            .unwrap();
        assert_eq!(
            context(ops.get(sid).unwrap(), &c),
            ["layers/eras/the-fall.md"]
        );
    }
}
