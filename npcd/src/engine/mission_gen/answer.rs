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

use std::collections::HashMap;
use std::path::Path;

use candle_conversation::stencil::{Param as CallParam, ParamType, ToolSpec};
use serde_json::{Map, Value};

use super::corpus::Corpus;
use super::material::{belongs, cut, strip_calls, ONLY_EVENT_YEARS};
use super::target::{Kind, Subject, Target};
use crate::engine::journal::tools::arguments;
use crate::engine::life;
use crate::engine::mission::{Mission, Origin, Todo, Work};

/// The call that declines.
pub const NOTHING: &str = "no_mission";
/// The call for a life event.
pub const EVENT: &str = "life_event";
/// The call for a correction.
pub const CORRECTION: &str = "correction";
/// The call for a story.
pub const STORY: &str = "story";
/// The call for a review.
pub const REVIEW_CALL: &str = "review";

/// The shortest quote that counts as a quote.
const MIN_QUOTE_CHARS: usize = 20;

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
                text("leaves"),
                text("agrees_with"),
                text("happens"),
            ],
        },
        Kind::Contradiction => ToolSpec {
            name: CORRECTION.into(),
            params: vec![
                text("quote_a"),
                text("quote_b"),
                choice("wrong", &["A", "B"]),
                text("why"),
                text("change"),
            ],
        },
        Kind::Gap => ToolSpec {
            name: STORY.into(),
            params: vec![
                text("title"),
                text("when"),
                text("where"),
                text("who"),
                text("agrees_with"),
                text("happens"),
            ],
        },
        // **The reading before the verdict.** Reading the document closely
        // enough to name its faults is the work; a verdict first is a verdict
        // the reading then has to justify. And the reading is its own field,
        // never empty: with only a `problems` field the model wrote it empty
        // and kept every document, a Zen dated four years after the Awakening
        // that says "three centuries have passed" among them.
        Kind::Review => ToolSpec {
            name: REVIEW_CALL.into(),
            params: vec![
                text("checked"),
                text("faults"),
                choice("verdict", &["keep", "revise"]),
                text("change"),
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

fn text(name: &str) -> CallParam {
    CallParam {
        name: name.to_string(),
        ty: ParamType::String,
        required: true,
        enum_values: None,
        items: None,
        min_items: 0,
        properties: None,
        nullable: false,
        minimum: None,
        requires: Vec::new(),
        shapes: Vec::new(),
    }
}

fn choice(name: &str, arms: &[&str]) -> CallParam {
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
    /// What the Maker reads under "What has been asked of you".
    pub brief: String,
    /// The mind path to write or change.
    pub writes: String,
    /// The mind paths to read first, every one real.
    pub reads: Vec<String>,
    /// The fewest words the document may be committed at — see
    /// [`Work::min_words`].
    pub min_words: usize,
}

/// The floor a life event is committed at: about four hundred words are asked.
const EVENT_MIN_WORDS: usize = 250;
/// The floor a story is committed at: six hundred to a thousand are asked.
const STORY_MIN_WORDS: usize = 380;

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
        (REVIEW_CALL, Kind::Review, Subject::Written { path }) => review(&field, path, corpus),
        (other, _, _) => Err(format!(
            "There is no call named `{other}` here; answer with `{}`.",
            calls.join("` or `")
        )),
    }
}

/// The arguments of one call, read as trimmed text.
struct Fields<'a>(&'a Map<String, Value>);

impl Fields<'_> {
    fn get(&self, k: &str) -> String {
        self.0
            .get(k)
            .and_then(Value::as_str)
            .map(str::trim)
            .unwrap_or_default()
            .to_string()
    }

    /// The field, refused when it is outside `min..=max` words or goes round
    /// in a loop.
    ///
    /// **A loop is the decode failing, not the model choosing.** A `happens`
    /// that began well ended "She is Zen, and nothing matters. She is Zen, and
    /// everything matters. She is Zen…" for a hundred words; the cap and the
    /// repetition check both refuse it, and the retry is asked again.
    fn words(&self, k: &str, min: usize, max: usize, what: &str) -> Result<String, String> {
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
        if loops(&v) {
            return Err(format!(
                "`{k}` repeats itself. Say {what} once, plainly, and stop."
            ));
        }
        Ok(v)
    }
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
    let happens = field.words("happens", 25, 240, "what happens in the event")?;
    let agrees = field.words("agrees_with", 0, 160, "the facts it must agree with")?;
    // **The grounding is the engine's to state; the model's facts are kept
    // when they say where they are written.** Asked for them, the model left the
    // field empty three times running, and elsewhere asserted "the record
    // states" a thing no document says. What the engine knows for certain is
    // the era the date falls in and the nearest written event, and it says so;
    // what the model adds is kept only when it names an era or event by title,
    // which is what lets the Maker check it.
    let lower = agrees.to_lowercase();
    let grounded = corpus
        .eras
        .iter()
        .map(|e| e.title.to_lowercase())
        .chain(life.events.iter().map(|e| e.title.to_lowercase()))
        .any(|t| lower.contains(&t) || lower.contains(t.trim_start_matches("the ")));
    let leaves = field.words("leaves", 5, 60, "what the event leaves them with")?;

    let writes = format!("layers/life/{who}/{file}");
    if corpus.exists(&writes) {
        return Err(format!("`{writes}` is already written; name a new event."));
    }
    // The era it falls in, and the written event nearest it in time — what the
    // Maker must not contradict.
    let mut reads = vec![era.path.clone()];
    let nearest = life
        .events
        .iter()
        .filter_map(|e| Some((e.year()?.abs_diff(year), e)))
        .min_by_key(|(d, _)| *d)
        .map(|(_, e)| e);
    if let Some(near) = nearest {
        reads.push(near.path.clone());
    }
    let mut must = format!("{} — it falls in that era of the world", era.title);
    if let Some(near) = nearest {
        must.push_str(&format!(
            ", and \"{}\" ({}), the written event nearest it",
            near.title, near.date
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
    if !life.anchor.is_empty() {
        s.push_str(&format!("Who {name} is: {}\n\n", cut(&life.anchor, 70)));
    }
    s.push_str(&format!(
        "What happens: {happens}\n\nIt must agree with: {must}\n\nWhat it leaves {name} with: \
         {leaves}\n\n"
    ));
    let voice = nearest
        .and_then(|n| corpus.text(&n.path))
        .map(|t| cut(&strip_calls(&t), 45));
    match voice {
        Some(v) => s.push_str(&format!(
            "{name}'s events are written like this: \"{v}\" Write it the same way — {name}'s own \
             voice and person, not yours — as one scene, moment by moment, of about four hundred \
             words, and end on what it left them believing or meaning to do."
        )),
        None => s.push_str(&format!(
            "Write it in {name}'s own voice, not yours, as one scene, moment by moment, of about \
             four hundred words, and end on what it left them believing or meaning to do."
        )),
    }
    if let Some(world) = &corpus.setting {
        s.push_str(&format!(
            "\n\nThe world it happens in: {} Nothing in it that this world does not have.",
            cut(world, 60)
        ));
    }
    Ok(Answer::Mission(Proposal {
        brief: s,
        writes,
        reads,
        min_words: EVENT_MIN_WORDS,
    }))
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
    let why = field.words("why", 8, 120, "why that side is the wrong one")?;
    let change = field.words("change", 6, 120, "the change to make")?;
    let brief = format!(
        "Two documents of the record contradict each other, and one must change.\n\n\
         `{a}` says: \"{qa}\"\n\n`{b}` says: \"{qb}\"\n\n\
         `{wrong}` is the one that is wrong: {why}\n\n\
         The change: {change}\n\n\
         Make it with `file_edit` — the clashing words replaced, nothing else in `{wrong}` \
         touched — so that it agrees with `{right}`."
    );
    Ok(Answer::Mission(Proposal {
        brief,
        writes: wrong.to_string(),
        reads: vec![a.to_string(), b.to_string()],
        min_words: 0,
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
    let when = field.words("when", 2, 40, "when it happens")?;
    let place = field.words("where", 2, 40, "where it happens")?;
    let who = field.words("who", 1, 80, "who is there")?;
    let happens = field.words("happens", 25, 240, "what happens in it")?;
    let agrees = field.get("agrees_with");
    let quote = quotes_in(&agrees)
        .into_iter()
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
         It must agree with the era, which says: \"{quote}\"\n\n\
         Write it as a story — past tense, six hundred to a thousand words, beginning with a \
         heading of its title. Show it as it happens: what the people in it do and say, moment by \
         moment, in the place itself. Never summarise what the era already says; the story is \
         what the era leaves out."
    );
    // **The world's own setting, so a story stays inside it.** One set in a
    // post-collapse world of towers and machines came back with quills, parchment
    // and silk weights.
    if let Some(world) = &corpus.setting {
        brief.push_str(&format!(
            "\n\nThe world it happens in: {} Nothing in it that this world does not have.",
            cut(world, 60)
        ));
    }
    Ok(Answer::Mission(Proposal {
        brief,
        writes,
        reads: vec![era.to_string()],
        min_words: STORY_MIN_WORDS,
    }))
}

/// A review: kept, or a revision that quotes what is wrong.
///
/// **The fault is quoted from the document.** A revision that cannot point at
/// the sentences it objects to is an opinion of the document's general quality,
/// and the Maker given it would rewrite what was right along with what was not.
fn review(field: &Fields, path: &str, corpus: &Corpus) -> Result<Answer, String> {
    let text = corpus.text(path).unwrap_or_default();
    let checked = field.words(
        "checked",
        20,
        700,
        "what you checked in the document and what you found — its year and era, its facts, its \
         voice",
    )?;
    // **A good review is long.** One that caught a date contradicted by the
    // document's own arithmetic, a human motive in a machine's mouth, a clash
    // with the era and a paragraph going round in circles ran to 369 words.
    let problems = field.words("faults", 0, 600, "what is wrong with the document")?;
    match field.get("verdict").as_str() {
        "keep" => return Ok(Answer::Nothing(format!("A review kept it: {checked}"))),
        "revise" => {}
        v => return Err(format!("`verdict` is `keep` or `revise`, not `{v}`.")),
    }
    let change = field.words("change", 3, 250, "how to revise it")?;
    // **The quote is looked for wherever the review wrote it.** One that found
    // Zen standing in a command room four years after the era has it leave the
    // galaxy quoted the sentence in `checked` and the paragraph to cut in
    // `change`, and left `faults` empty.
    let said = format!("{problems}\n{checked}\n{change}");
    if !quotes_in(&said)
        .iter()
        .chain(&single_quoted(&said))
        .any(|q| contains(&text, q))
    {
        return Err(
            "A revision must quote the document's own sentences that are wrong — its words \
             exactly, between quotes — so the Maker revising it knows what to change and what to \
             keep. If nothing in it is wrong, the verdict is `keep`."
                .into(),
        );
    }
    // What is wrong, as the Maker reads it: the faults when they were listed,
    // the reading when they were found there.
    let problems = match problems.is_empty() {
        true => checked,
        false => problems,
    };
    let life = path
        .strip_prefix("layers/life/")
        .and_then(|rest| rest.split('/').next())
        .and_then(|who| corpus.life(who));
    let year: Option<u32> = path
        .rsplit('/')
        .next()
        .and_then(|f| f.get(..4))
        .and_then(|y| y.parse().ok());
    // What it answers to, read before it is revised: the era it falls in and,
    // for a life, the event of that life nearest it.
    let mut reads: Vec<String> = Vec::new();
    if let Some(era) = year.and_then(|y| corpus.era_of(y)) {
        reads.push(era.path.clone());
    }
    if let (Some(l), Some(y)) = (life, year) {
        if let Some(e) = l
            .events
            .iter()
            .filter(|e| e.path != path)
            .min_by_key(|e| e.year().map_or(u32::MAX, |ey| ey.abs_diff(y)))
        {
            reads.push(e.path.clone());
        }
    }
    let brief = format!(
        "A second reading of `{path}` found it falls short, and it is to be revised.\n\n\
         What is wrong: {problems}\n\n\
         The revision: {change}\n\n\
         Change it with `file_edit`, each wrong passage replaced — or, if the fault runs through \
         all of it, write it again whole with `file_write`. Keep what is right in it."
    );
    Ok(Answer::Mission(Proposal {
        brief,
        writes: path.to_string(),
        reads,
        // Never above what the document already holds: a short document put
        // in line by hand could otherwise not have a one-line fault mended.
        min_words: match life {
            Some(_) => EVENT_MIN_WORDS,
            None => STORY_MIN_WORDS,
        }
        .min(strip_calls(&text).split_whitespace().count()),
    }))
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
fn contains(text: &str, q: &str) -> bool {
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
fn unlink(s: &str) -> String {
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

/// Every span between single quotes, long enough to be a quote — how a review
/// quoted, as often as not. An apostrophe opens a span too ("Verdi's role as a
/// '…'"), so some spans are noise; the caller keeps only those the document
/// holds.
fn single_quoted(s: &str) -> Vec<String> {
    let s = s.replace(['‘', '’'], "'");
    s.split('\'')
        .map(str::trim)
        .filter(|q| q.chars().count() >= MIN_QUOTE_CHARS)
        .map(str::to_string)
        .collect()
}

/// Every span between double quotes, long enough to be a quote.
fn quotes_in(s: &str) -> Vec<String> {
    let s = s.replace(['“', '”'], "\"");
    s.split('"')
        .skip(1)
        .step_by(2)
        .map(str::trim)
        .filter(|q| q.chars().count() >= MIN_QUOTE_CHARS)
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

/// The mission a proposal becomes: its brief, the steps the engine can see done
/// — to the desk, each read, the write and its commit — and the report.
pub fn mission(p: &Proposal, generator: &str, target: &Target, desk: Option<&Desk>) -> Mission {
    let mut todo = Vec::new();
    if let Some(d) = desk {
        todo.push(Todo::new(format!("go to {} on {}", d.room, d.level)));
    }
    for r in &p.reads {
        todo.push(Todo::new(format!("read {r}")));
    }
    // "change" for a document that stands, so the Maker edits it rather than
    // writing a replacement that loses what was right in it.
    let verb = match target.subject {
        Subject::Pair { .. } | Subject::Written { .. } => "change",
        _ => "write",
    };
    todo.push(Todo::new(format!("{verb} {} and commit it", p.writes)));
    todo.push(Todo::report("go back to the table and report it"));
    Mission::new(
        p.brief.clone(),
        todo,
        Origin::Generated {
            generator: generator.to_string(),
            target: target.key.clone(),
        },
    )
    .with_work(Work {
        writes: p.writes.clone(),
        reads: p.reads.clone(),
        min_words: p.min_words,
    })
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
        let desk = Desk {
            room: "band one".into(),
            level: "the casting level".into(),
        };
        let m = mission(&p, "life-event", &t, Some(&desk));
        let steps: Vec<&str> = m.todo.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(
            steps,
            [
                "go to band one on the casting level",
                "read layers/eras/the-fall.md",
                "read layers/life/keeper/2487-03-08 The Second the Sky Went Out.md",
                "write layers/life/keeper/2488 The Empty Table.md and commit it",
                "go back to the table and report it",
            ]
        );
        assert_eq!(
            m.origin,
            Origin::Generated {
                generator: "life-event".into(),
                target: "life:keeper".into()
            }
        );
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
            serde_json::json!({ "date": "2490", "title": "X", "happens": "Things.", "agrees_with": "x", "leaves": "y" }),
        );
        assert!(check(&thin, Kind::LifeEvent, &t, &c)
            .unwrap_err()
            .contains("`happens`"));
        // A title carries no character a file name cannot.
        assert_eq!(file_title("Water: the Year?").unwrap(), "Water the Year");
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
        // engine's own grounding stands.
        let floating = call(
            EVENT,
            serde_json::json!({
                "date": "2650",
                "title": "Somewhere",
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
                "It must agree with: The Retreat — it falls in that era of the world, and \"The \
                 Charge\" (2786), the written event nearest it.\n\n"
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
        assert_eq!(names(Kind::Review), [REVIEW_CALL, NOTHING]);
    }

    /// **A revision quotes what is wrong; a document with nothing wrong is
    /// kept.** The revision reads the era and the nearest sibling event, and
    /// changes the document rather than writing a new one.
    #[test]
    fn a_review_keeps_or_revises_by_quoting_the_document() {
        let dir = mind();
        let mut c = Corpus::read(dir.path(), "test");
        c.reviewable = vec!["layers/life/keeper/2786 The Charge.md".into()];
        let t = next(Kind::Review, &c, &|_, _| false, 0).unwrap();
        const CHECKED: &str = "It is set in 2786, in the Retreat; it says the plan was given, \
                               which the era allows, and it speaks as Keeper's other event does, \
                               as we.";
        let review_with = |checked: &str, problems: &str, verdict: &str| {
            call(
                REVIEW_CALL,
                serde_json::json!({
                    "checked": checked,
                    "faults": problems,
                    "verdict": verdict,
                    "change": "Say who gave the plan and where, in the voice of the earlier event, and end on what Keeper meant to do.",
                }),
            )
        };
        let review = |problems: &str, verdict: &str| review_with(CHECKED, problems, verdict);
        assert_eq!(
            check(&review("", "keep"), Kind::Review, &t, &c),
            Ok(Answer::Nothing(format!("A review kept it: {CHECKED}")))
        );
        // A review that shows no reading is no review.
        assert!(check(&review_with("", "", "keep"), Kind::Review, &t, &c)
            .unwrap_err()
            .contains("`checked`"));
        assert!(check(
            &review("It is far too thin to be a life.", "revise"),
            Kind::Review,
            &t,
            &c
        )
        .unwrap_err()
        .contains("must quote"));
        // A single-quoted fault is a quote.
        assert!(check(
            &review(
                "Keeper's line 'We were given the plan.' says nothing of who gave it.",
                "revise"
            ),
            Kind::Review,
            &t,
            &c
        )
        .is_ok());
        let Answer::Mission(p) = check(
            &review(
                "\"We were given the plan.\" says nothing of who gave it, or where.",
                "revise",
            ),
            Kind::Review,
            &t,
            &c,
        )
        .unwrap() else {
            panic!("a revision");
        };
        assert_eq!(p.writes, "layers/life/keeper/2786 The Charge.md");
        assert_eq!(
            p.reads,
            [
                "layers/eras/the-retreat.md",
                "layers/life/keeper/2487-03-08 The Second the Sky Went Out.md"
            ]
        );
        assert_eq!(p.min_words, 5, "no more than the five words it holds");
        let m = mission(&p, "review", &t, None);
        assert_eq!(
            m.todo[2].text,
            "change layers/life/keeper/2786 The Charge.md and commit it"
        );
    }
}
