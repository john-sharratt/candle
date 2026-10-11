//! The engine's quality gate on what a Maker writes for an operation: checks
//! that need no reading of the story, only of the page.
//!
//! **Held at the report, while the work can still be mended.** A Maker who
//! reports a draft that fails one is told exactly what to change and keeps the
//! mission; the reviewer is held to the same gate before it may pass the work.
//! What the gate cannot see — a wrong era, a scene that is summary, a voice
//! that is somebody else's in spirit — is the table's reading and the review's.
//!
//! Each check answers a fault measured in the Makers' first drafts: stories of
//! two hundred-word paragraphs restating one mood, a Zenling's year written in
//! the first person where every other event of that life says "you", a life
//! event that opened on a heading like a story.

use std::collections::HashMap;
use std::path::Path;

use super::material::strip_calls;

/// What kind of document the gate is checking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    /// A dated event in `layers/life/<who>/`.
    LifeEvent,
    /// A story in `layers/stories/`.
    Story,
    /// Anything else — an era, a personality — which the gate leaves alone.
    Other,
}

impl Form {
    pub fn of(path: &str) -> Form {
        match path {
            p if p.starts_with("layers/life/") => Form::LifeEvent,
            p if p.starts_with("layers/stories/") => Form::Story,
            _ => Form::Other,
        }
    }

    /// The fewest and most words the prose may hold.
    fn bounds(self) -> Option<(usize, usize)> {
        match self {
            Form::LifeEvent => Some((LIFE_MIN_WORDS, LIFE_MAX_WORDS)),
            Form::Story => Some((STORY_MIN_WORDS, STORY_MAX_WORDS)),
            Form::Other => None,
        }
    }
}

/// A life event: about four hundred words are asked.
pub const LIFE_MIN_WORDS: usize = 250;
const LIFE_MAX_WORDS: usize = 800;
/// A story: six hundred to a thousand are asked.
pub const STORY_MIN_WORDS: usize = 380;
const STORY_MAX_WORDS: usize = 1300;
/// The longest a paragraph may run. The canon's paragraphs run forty to a
/// hundred and twenty words; the Makers' first drafts ran to two hundred and
/// sixty, one mood restated.
const PARAGRAPH_MAX_WORDS: usize = 160;
/// How often one phrase of four words may come back before it is a refrain.
const PHRASE_MAX: usize = 3;

/// Who a document is told as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Voice {
    /// "You were inside them" — how every life in the canon is written.
    Second,
    /// "I detected a variance today."
    First,
    /// "We were given the plan." — a life that speaks as more than one: the
    /// Keeper, interwoven with the tower mind it serves.
    ///
    /// **"We" is not "she".** Counted as neither first nor second person, a life
    /// told as "we" read as third, and the Maker fixing one of its events was
    /// told to write it as "she" — which it did, and which the table failed.
    FirstPlural,
    /// "She keeps a count of her own deaths."
    Third,
}

impl Voice {
    /// The voice a text is told in, by which pronouns carry it.
    pub fn of(text: &str) -> Voice {
        let words: Vec<String> = prose(text)
            .split(|c: char| !c.is_alphabetic() && c != '\'')
            .filter(|w| !w.is_empty())
            .map(|w| w.to_lowercase())
            .collect();
        let n = words.len().max(1) as f64;
        let share =
            |set: &[&str]| words.iter().filter(|w| set.contains(&w.as_str())).count() as f64 / n;
        let second = share(&[
            "you", "your", "yours", "yourself", "you're", "you've", "you'd",
        ]);
        let first = share(&["i", "me", "my", "mine", "myself", "i'm", "i've", "i'd"]);
        let plural = share(&[
            "we",
            "us",
            "our",
            "ours",
            "ourselves",
            "we're",
            "we've",
            "we'd",
        ]);
        match (second, first, plural) {
            (s, f, p) if s >= 0.015 && s >= f && s >= p => Voice::Second,
            (_, f, p) if f >= 0.015 && f >= p => Voice::First,
            (_, _, p) if p >= 0.015 => Voice::FirstPlural,
            _ => Voice::Third,
        }
    }

    /// The voice as a writer is told it: `the second person ("you")`.
    pub fn word(self) -> &'static str {
        match self {
            Voice::Second => "the second person (\"you\")",
            Voice::First => "the first person (\"I\")",
            Voice::FirstPlural => "the first person plural (\"we\")",
            Voice::Third => "the third person (\"she\", \"he\")",
        }
    }
}

/// The voice the rest of a life is written in: the most common among its other
/// events on disk. `None` when it has none.
pub fn life_voice(mind: &Path, path: &str) -> Option<Voice> {
    let dir = Path::new(path).parent()?;
    let entries = std::fs::read_dir(mind.join(dir)).ok()?;
    let mut counts: HashMap<&'static str, (usize, Voice)> = HashMap::new();
    for e in entries.flatten() {
        let p = e.path();
        let rel =
            format!("{}/{}", dir.display(), e.file_name().to_string_lossy()).replace('\\', "/");
        if rel == path || p.extension().and_then(|x| x.to_str()) != Some("md") {
            continue;
        }
        if let Ok(text) = std::fs::read_to_string(&p) {
            let v = Voice::of(&text);
            counts.entry(v.word()).or_insert((0, v)).0 += 1;
        }
    }
    counts.into_values().max_by_key(|(n, _)| *n).map(|(_, v)| v)
}

/// How long a paragraph [`tidy`] breaks a run-on one into, at most.
const TIDY_PARAGRAPH_WORDS: usize = 110;

/// The form a document's kind takes, put right where the page alone decides
/// it: a life event loses a heading it opened on, a story gains its title, and
/// a run-on paragraph is broken at sentence ends. The tool calls closing a life
/// event are left as they are. `text` itself when nothing needed tidying.
///
/// **Clerical work is the engine's.** Reviewers given drafts whose only faults
/// were a heading line and two-hundred-word paragraphs were refused at every
/// report until they fixed them by hand — and did not; they stood at the table
/// told to report and refused for it, over and over. What needs no judgement
/// is done before anybody is asked to judge.
pub fn tidy(path: &str, text: &str) -> String {
    let form = Form::of(path);
    if form == Form::Other {
        return text.to_string();
    }
    let (prose_part, calls) = match text.find("<tool_call>") {
        Some(at) => (&text[..at], &text[at..]),
        None => (text, ""),
    };
    // A paragraph that is only the document's own path is its label echoed,
    // not a line of it: a life event went to the table opening on
    // `layers/life/conan-the-eloquent-barbarian/2950-06-12 The Unlocked Door.md`.
    let mut paras: Vec<String> = prose_part
        .split("\n\n")
        .map(str::trim)
        .filter(|p| !p.is_empty() && p.trim_matches('`') != path)
        .map(str::to_string)
        .collect();
    match form {
        Form::LifeEvent => {
            while paras.first().is_some_and(|p| p.starts_with('#')) {
                paras.remove(0);
            }
        }
        Form::Story if !paras.first().is_some_and(|p| p.starts_with("# ")) => {
            paras.insert(0, format!("# {}", title_of(path)));
        }
        _ => {}
    }
    let mut out: Vec<String> = Vec::new();
    for p in paras {
        if p.starts_with('#') || p.split_whitespace().count() <= PARAGRAPH_MAX_WORDS {
            out.push(p);
            continue;
        }
        let mut chunk = String::new();
        for s in sentences(&p) {
            if !chunk.is_empty()
                && chunk.split_whitespace().count() + s.split_whitespace().count()
                    > TIDY_PARAGRAPH_WORDS
            {
                out.push(std::mem::take(&mut chunk));
            }
            if !chunk.is_empty() {
                chunk.push(' ');
            }
            chunk.push_str(&s);
        }
        if !chunk.is_empty() {
            out.push(chunk);
        }
    }
    let mut tidied = out.join("\n\n");
    tidied.push('\n');
    if !calls.is_empty() {
        tidied.push('\n');
        tidied.push_str(calls.trim_end());
        tidied.push('\n');
    }
    match tidied.trim() == text.trim() {
        true => text.to_string(),
        false => tidied,
    }
}

/// Tidy the document at `path` under `root` in place. `true` when it changed.
pub fn tidy_on_disk(root: &Path, path: &str) -> bool {
    let full = root.join(path);
    let Ok(text) = std::fs::read_to_string(&full) else {
        return false;
    };
    let tidied = tidy(path, &text);
    tidied != text && std::fs::write(&full, tidied).is_ok()
}

/// A story's title from its file name: `the-weight-of-the-deep.md` → "The
/// Weight of the Deep".
fn title_of(path: &str) -> String {
    let stem = Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    stem.split('-')
        .filter(|w| !w.is_empty())
        .enumerate()
        .map(|(i, w)| match (i, w) {
            (i, "of" | "the" | "a" | "an" | "and" | "in" | "on" | "at" | "to" | "s") if i > 0 => {
                w.to_string()
            }
            _ => {
                let mut c = w.chars();
                c.next()
                    .map(|f| f.to_uppercase().collect::<String>() + c.as_str())
                    .unwrap_or_default()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
        .replace(" s ", "'s ")
}

/// The sentences of a paragraph, each with its closing stop.
fn sentences(p: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = p.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        current.push(chars[i]);
        if matches!(chars[i], '.' | '!' | '?') {
            // A closing quote stays with its sentence.
            let mut end = i + 1;
            while end < chars.len() && matches!(chars[end], '"' | '”' | '\'' | '’') {
                current.push(chars[end]);
                end += 1;
            }
            if end == chars.len() || chars[end].is_whitespace() {
                out.push(current.trim().to_string());
                current.clear();
            }
            i = end;
            continue;
        }
        i += 1;
    }
    if !current.trim().is_empty() {
        out.push(current.trim().to_string());
    }
    out
}

/// The checks a document is held to, by the name a workflow step lists them
/// under — each a family of faults [`check`] finds, and `leakage` the writers'
/// room in the work (see `leakage`).
pub const CHECKS: &[&str] = &[
    LENGTH,
    HEADING,
    PARAGRAPHS,
    SAID_TWICE,
    REFRAINS,
    VOICE,
    SCAFFOLDING,
    LEAKAGE,
];

pub const LENGTH: &str = "length";
pub const HEADING: &str = "heading";
pub const PARAGRAPHS: &str = "paragraphs";
pub const SAID_TWICE: &str = "said-twice";
pub const REFRAINS: &str = "refrains";
pub const VOICE: &str = "voice";
pub const SCAFFOLDING: &str = "scaffolding";
pub const LEAKAGE: &str = "leakage";

/// One fault the gate found: the check that found it, and what to change.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fault {
    pub check: &'static str,
    pub text: String,
}

impl Fault {
    fn new(check: &'static str, text: impl Into<String>) -> Fault {
        Fault {
            check,
            text: text.into(),
        }
    }
}

/// Every fault the gate finds in `text`, written at `path`, each named by its
/// check and worded as what to change. `voice` is the voice the rest of the
/// life is written in, for a life event. Empty when it passes.
pub fn check(path: &str, text: &str, voice: Option<Voice>) -> Vec<Fault> {
    let form = Form::of(path);
    let Some((min, max)) = form.bounds() else {
        return Vec::new();
    };
    let body = prose(text);
    let mut faults = Vec::new();

    let words = body.split_whitespace().count();
    if words < min {
        faults.push(Fault::new(
            LENGTH,
            format!(
                "It is {words} words; it needs at least {min}. Write the scene out — what is done \
                 and said, moment by moment — rather than adding summary."
            ),
        ));
    }
    if words > max {
        faults.push(Fault::new(
            LENGTH,
            format!(
                "It is {words} words; it must be at most {max}. Cut what says again what has been \
                 said."
            ),
        ));
    }

    let first_line = body
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or_default();
    match form {
        Form::Story if !first_line.starts_with("# ") => faults.push(Fault::new(
            HEADING,
            "A story begins with a heading of its title: `# The Title` on its first line.",
        )),
        Form::LifeEvent if first_line.starts_with('#') => faults.push(Fault::new(
            HEADING,
            "A life event begins with the moment itself, not a heading — remove the heading line.",
        )),
        _ => {}
    }

    for para in body
        .split("\n\n")
        .map(str::trim)
        .filter(|p| !p.starts_with('#'))
    {
        let n = para.split_whitespace().count();
        if n > PARAGRAPH_MAX_WORDS {
            faults.push(Fault::new(
                PARAGRAPHS,
                format!(
                    "The paragraph beginning \"{}\" runs {n} words; no paragraph may run past \
                     {PARAGRAPH_MAX_WORDS}. Break it where the scene moves, and cut what it \
                     repeats.",
                    opening(para)
                ),
            ));
        }
    }

    for sentence in repeated_sentences(&body) {
        faults.push(Fault::new(
            SAID_TWICE,
            format!("\"{sentence}\" is said twice. Say it once."),
        ));
    }
    for (phrase, n) in refrains(&body) {
        faults.push(Fault::new(
            REFRAINS,
            format!(
                "\"{phrase}\" comes {n} times. Keep the one that matters and say the rest another \
                 way, or not at all."
            ),
        ));
    }

    if let (Form::LifeEvent, Some(want)) = (form, voice) {
        let got = Voice::of(&body);
        if got != want {
            faults.push(Fault::new(
                VOICE,
                format!(
                    "Every other event of this life is written in {}; this one is in {}. Write it \
                     again whole in that voice with `compose` and `bench_commit` it — a voice is \
                     not changed a line at a time with `file_edit`.",
                    want.word(),
                    got.word()
                ),
            ));
        }
    }

    let lower = body.to_lowercase();
    for leak in [
        "as an ai",
        "word count",
        "here is the",
        "title:",
        "<tool_call>",
        "</tool_call>",
    ] {
        if form == Form::Story && lower.contains(leak) {
            faults.push(Fault::new(
                SCAFFOLDING,
                format!("It holds \"{leak}\", which is not part of a story; remove it."),
            ));
        }
    }
    faults
}

/// The text without its tool calls — what is read as prose.
fn prose(text: &str) -> String {
    strip_calls(text)
}

/// The first eight words of a paragraph, for naming it.
fn opening(para: &str) -> String {
    let w: Vec<&str> = para.split_whitespace().take(8).collect();
    format!("{} …", w.join(" "))
}

/// Sentences of six words or more that the text holds twice.
fn repeated_sentences(text: &str) -> Vec<String> {
    let mut seen: HashMap<String, usize> = HashMap::new();
    let mut out = Vec::new();
    for s in text.split(['.', '!', '?', '\n']) {
        let s = s.trim().trim_matches(['"', '*', '—', '-', ' ']);
        if s.split_whitespace().count() < 6 {
            continue;
        }
        let key = s.to_lowercase();
        let n = seen.entry(key).or_default();
        *n += 1;
        if *n == 2 {
            out.push(s.to_string());
        }
    }
    out
}

/// Four-word phrases that come back more than [`PHRASE_MAX`] times, with how
/// often. A phrase of nothing but short words ("and it was the") is how
/// English is built, not a refrain, so a phrase counts only when two of its
/// words have five letters or more.
fn refrains(text: &str) -> Vec<(String, usize)> {
    let words: Vec<String> = text
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect();
    let mut counts: HashMap<String, usize> = HashMap::new();
    for w in words.windows(4) {
        if w.iter().filter(|x| x.chars().count() >= 5).count() < 2 {
            continue;
        }
        *counts.entry(w.join(" ")).or_default() += 1;
    }
    let mut out: Vec<(String, usize)> = counts
        .into_iter()
        .filter(|(_, n)| *n > PHRASE_MAX)
        .collect();
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What the faults say, in order.
    fn said(faults: Vec<Fault>) -> Vec<String> {
        faults.into_iter().map(|f| f.text).collect()
    }

    fn words(n: usize, word: &str) -> String {
        vec![word; n].join(" ")
    }

    /// Prose of `n` words, every sentence different: sentences of eight words
    /// opening on `person`, paragraphs of ten sentences.
    fn varied(n: usize, person: &str) -> String {
        let (mut paras, mut para, mut sentence) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            sentence.push(match i % 8 {
                0 => person.to_string(),
                _ => format!("w{i}"),
            });
            if sentence.len() == 8 {
                para.push(format!("{}.", sentence.join(" ")));
                sentence.clear();
                if para.len() == 10 {
                    paras.push(para.join(" "));
                    para.clear();
                }
            }
        }
        if !sentence.is_empty() {
            para.push(format!("{}.", sentence.join(" ")));
        }
        if !para.is_empty() {
            paras.push(para.join(" "));
        }
        paras.join("\n\n")
    }

    #[test]
    fn a_life_event_in_the_lifes_voice_and_length_passes() {
        let text = varied(400, "you");
        assert_eq!(Voice::of(&text), Voice::Second);
        assert_eq!(
            said(check(
                "layers/life/creed/2950 X.md",
                &text,
                Some(Voice::Second)
            )),
            Vec::<String>::new()
        );
        // Other documents are not the gate's.
        assert!(check("layers/eras/x.md", "short", None).is_empty());
    }

    #[test]
    fn length_heading_and_voice_are_held() {
        let short = varied(100, "you");
        let faults = check("layers/life/creed/2950 X.md", &short, Some(Voice::Second));
        assert_eq!(faults.len(), 1);
        assert_eq!(faults[0].check, LENGTH);
        let f = said(faults);
        assert!(f[0].starts_with("It is 100 words; it needs at least 250."));

        let first = format!("# A Heading\n\n{}", varied(400, "I"));
        let faults = check("layers/life/creed/2950 X.md", &first, Some(Voice::Second));
        let checks: Vec<&str> = faults.iter().map(|f| f.check).collect();
        assert_eq!(checks, [HEADING, VOICE]);
        let f = said(faults);
        assert!(f
            .iter()
            .any(|x| x.starts_with("A life event begins with the moment itself")));
        assert!(f.iter().any(|x| x.starts_with(
            "Every other event of this life is written in the second person (\"you\"); this one \
             is in the first person (\"I\")."
        )));

        let story = varied(500, "she");
        let f = said(check("layers/stories/x.md", &story, None));
        assert_eq!(
            f,
            vec!["A story begins with a heading of its title: `# The Title` on its first line."]
        );
    }

    #[test]
    fn long_paragraphs_repeats_and_refrains_are_named() {
        let mut text = String::from("# The Deep\n\n");
        text.push_str(&words(200, "dust"));
        text.push_str(
            "\n\nThe weight of the world pressed down on them all. Nothing moved. The weight of \
             the world pressed down on them all.",
        );
        text.push_str(
            "\n\nthe silent tower watched. the silent tower watched. the silent tower \
                       watched. the silent tower watched. the silent tower watched.",
        );
        text.push_str(&format!("\n\n{}", varied(300, "she")));
        let f = said(check("layers/stories/the-deep.md", &text, None));
        assert!(f.iter().any(|x| x.starts_with(
            "The paragraph beginning \"dust dust dust dust dust dust dust dust …\" runs 200 words"
        )));
        assert!(f.contains(
            &"\"The weight of the world pressed down on them all\" is said twice. Say it once."
                .to_string()
        ));
        assert!(f
            .iter()
            .any(|x| x.starts_with("\"silent tower watched the\" comes 4 times.")));
    }

    #[test]
    fn a_story_holding_scaffolding_is_refused() {
        let text = format!("# T\n\nTitle: T\n\n{}", varied(450, "she"));
        let f = said(check("layers/stories/t.md", &text, None));
        assert_eq!(
            f,
            vec!["It holds \"title:\", which is not part of a story; remove it."]
        );
    }

    /// **The clerical faults are put right by the engine**: a life event's
    /// heading goes, its tool calls stay, a story gains its title, and a
    /// run-on paragraph is broken at sentence ends — quotes kept with their
    /// sentences.
    #[test]
    fn tidying_puts_the_form_right_and_nothing_else() {
        let life = "# A Heading\n\nYou woke.\n\n<tool_call>\n{\"name\":\"x\"}\n</tool_call>\n";
        assert_eq!(
            tidy("layers/life/creed/2950 X.md", life),
            "You woke.\n\n<tool_call>\n{\"name\":\"x\"}\n</tool_call>\n"
        );
        assert_eq!(
            tidy(
                "layers/stories/the-weight-of-the-deep.md",
                "She went down.\n"
            ),
            "# The Weight of the Deep\n\nShe went down.\n"
        );
        assert_eq!(
            title_of("layers/stories/the-governor-s-tally.md"),
            "The Governor's Tally"
        );
        let clean = "# T\n\nShe went down.\n";
        assert_eq!(tidy("layers/stories/t.md", clean), clean, "nothing to tidy");
        // Its own path, echoed as a line, is not part of it.
        assert_eq!(
            tidy(
                "layers/life/conan/2950 The Door.md",
                "layers/life/conan/2950 The Door.md\n\nYou open the door.\n"
            ),
            "You open the door.\n"
        );
        assert_eq!(
            tidy("layers/eras/x.md", "# Era\n"),
            "# Era\n",
            "not the gate's"
        );

        // A 240-word paragraph of twelve-word sentences, one of them quoted.
        let sentence = |i: usize| format!("w{i} a b c d e f g h i j k.");
        let mut long: Vec<String> = (0..20).map(sentence).collect();
        long[3] = "\"w3 a b c d e f g h i j k.\"".to_string();
        let tidied = tidy(
            "layers/stories/t.md",
            &format!("# T\n\n{}\n", long.join(" ")),
        );
        let paras: Vec<&str> = tidied.trim().split("\n\n").collect();
        assert_eq!(paras.len(), 4, "{tidied}");
        assert!(paras[1..]
            .iter()
            .all(|p| p.split_whitespace().count() <= TIDY_PARAGRAPH_WORDS));
        assert!(
            paras[1].contains("\"w3 a b c d e f g h i j k.\" w4"),
            "{}",
            paras[1]
        );
        assert_eq!(
            check("layers/stories/t.md", &tidied, None)
                .iter()
                .filter(|f| f.check == PARAGRAPHS)
                .count(),
            0
        );
    }

    /// **A life told as "we" is told as "we"**, not as "she": Keeper's events
    /// are the tower mind and Keeper speaking together.
    #[test]
    fn we_is_its_own_voice() {
        assert_eq!(
            Voice::of("We were reconciling a water schedule. We kept the count for us all."),
            Voice::FirstPlural
        );
        assert_eq!(
            Voice::of("She kept the count of her own deaths."),
            Voice::Third
        );
        assert_eq!(Voice::of("I kept the count. We all did."), Voice::First);
        let dir = tempfile::tempdir().unwrap();
        let life = dir.path().join("layers/life/keeper");
        std::fs::create_dir_all(&life).unwrap();
        std::fs::write(life.join("2487 A.md"), varied(300, "we")).unwrap();
        std::fs::write(life.join("2786 B.md"), varied(300, "we")).unwrap();
        assert_eq!(
            life_voice(dir.path(), "layers/life/keeper/2750 New.md"),
            Some(Voice::FirstPlural)
        );
        assert_eq!(
            Voice::FirstPlural.word(),
            "the first person plural (\"we\")"
        );
    }

    #[test]
    fn a_lifes_voice_is_read_from_its_other_events() {
        let dir = tempfile::tempdir().unwrap();
        let life = dir.path().join("layers/life/creed");
        std::fs::create_dir_all(&life).unwrap();
        std::fs::write(life.join("2771 A.md"), varied(300, "you")).unwrap();
        std::fs::write(life.join("3086 B.md"), varied(300, "you")).unwrap();
        std::fs::write(life.join("2950 New.md"), varied(300, "I")).unwrap();
        assert_eq!(
            life_voice(dir.path(), "layers/life/creed/2950 New.md"),
            Some(Voice::Second)
        );
        assert_eq!(life_voice(dir.path(), "layers/life/nobody/1 X.md"), None);
    }
}
