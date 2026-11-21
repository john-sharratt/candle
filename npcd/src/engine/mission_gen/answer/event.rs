//! Defines the call a life event is answered with, and the brief the engine builds from a checked answer.
//!
//! A life event is a dated entry the subject's life adds to the record. The brief is built from the fields the model filled, with the grounding stated by the engine rather than trusted from the model. Each fact is kept only when it names where it is written, and the brief says whose voice the piece is in, since a writer given the subject's own anchor unqualified writes it as itself. The example quoted to the writer is the nearest earlier event in the same voice, so the brief asks for one voice and the gate accepts it.

use super::*;

/// The call for a life event.
pub const EVENT: &str = "life_event";

pub(super) fn event(field: &Fields, who: &str, corpus: &Corpus) -> Result<Answer, String> {
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

/// What the world's own terms mean, for those `scene` and its `era` name —
/// the world's account of each, so the piece is written inside it.
///
/// **A word the writer has to guess is a world it invents.** A story of the
/// Contested Cities put its soldiers in bodies that bleed — which the world's
/// avatars are — and was failed by a reader who did not know it; a life of
/// Keeper gave a bodiless intelligence hands. Both terms are defined in the
/// world's own documents, which neither writer nor reader was shown.
pub(super) fn worlds_words(corpus: &Corpus, scene: &str, era: &Era) -> String {
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
pub(super) fn world_then(corpus: &Corpus, era: &Era, year: Option<u32>) -> Option<String> {
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
pub(super) fn majority_voice(voices: impl Iterator<Item = Voice>) -> Option<Voice> {
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
pub(super) fn voice_said(v: Voice, name: &str) -> String {
    match v {
        Voice::Second => format!("the second person — \"you\", speaking to {name}"),
        Voice::First => format!("the first person — \"I\", as {name}"),
        Voice::FirstPlural => format!("the first person plural — \"we\", as {name}"),
        Voice::Third => format!("the third person — {name} by name, \"she\" or \"he\" or \"it\""),
    }
}

/// A title as a file can carry it: the characters a file name cannot hold are
/// dropped, and it may not end in a dot.
pub(super) fn file_title(title: &str) -> Result<String, String> {
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
