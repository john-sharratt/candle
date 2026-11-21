//! Declares the answer grammar for each kind of work and the length each field is held to.
//!
//! Each kind's call is one JSON-shaped grammar, with the fields that decide the work placed in order. The long text field comes last, so the model fills the short fields before spending its length on the scene. Every field's word cap is stated to the writer before it writes, and the grammar closes each text field at the cap, so the length the writer is told and the length the grammar enforces cannot drift apart. A turn must be an act: a turn led by a still verb of understanding is refused, because a realisation is not an event.

use super::*;

/// What `turns` asks for, as a refusal names it.
pub(super) const TURNS: &str =
    "the act the moment turns on — who does what that cannot be taken back, and \
                     what is different after it";

/// The act an event or story turns on — refused when it is no act at all.
///
/// **An event is decided before its scene.** A brief whose "what happens" had
/// its subject walk into the dark, sit, and wait "for the end of time, or
/// perhaps for nothing at all" was written faithfully as a mood, and the table
/// failed the draft for having no event: the fault was in the brief. Asked to
/// name the turn first, in its own field, before the scene that leads to it,
/// the answer has to have one.
pub(super) fn turns(field: &Fields) -> Result<String, String> {
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
pub(super) const STILL_VERBS: &[&str] = &[
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

/// Every field of `kind`'s call that is held to a length, and its cap in words
/// — what the answer is checked against, the same constants the checks use.
/// For a text field it is also what [`specs`] bounds the grammar by; a list's
/// grammar bounds each entry instead, and the cap here is the list's whole.
pub(super) fn caps(kind: Kind) -> &'static [(&'static str, usize)] {
    match kind {
        Kind::LifeEvent => &[
            ("leaves", LEAVES_WORDS),
            ("agrees_with", AGREES_WORDS),
            ("turns", TURNS_WORDS),
            ("happens", HAPPENS_WORDS),
        ],
        Kind::Contradiction => &[("why", WHY_WORDS), ("change", CHANGE_WORDS)],
        Kind::Gap => &[
            ("when", WHEN_WORDS),
            ("where", WHERE_WORDS),
            ("who", WHO_WORDS),
            ("agrees_with", AGREES_WORDS),
            ("turns", TURNS_WORDS),
            ("happens", HAPPENS_WORDS),
        ],
    }
}

/// How long each field of `kind`'s call may be, said before it is written.
///
/// **A cap the writer is not told is a refusal waiting to happen.** The checks
/// held `happens` under 240 words while no prompt said so: answers came back at
/// 250, 264 and 318 words, and each was refused and asked again whole — three
/// minutes a try — to learn what one line up front says.
pub fn lengths(kind: Kind) -> String {
    let each: Vec<String> = caps(kind)
        .iter()
        .map(|(name, words)| format!("`{name}` under {words} words"))
        .collect();
    format!(
        "Keep to these lengths, counted in words: {}.",
        each.join("; ")
    )
}

/// The caps each generated field is checked against, in words.
pub(super) const TURNS_WORDS: usize = 60;

pub(super) const HAPPENS_WORDS: usize = 240;

pub(super) const AGREES_WORDS: usize = 160;

pub(super) const LEAVES_WORDS: usize = 60;

pub(super) const WHY_WORDS: usize = 120;

pub(super) const CHANGE_WORDS: usize = 120;

pub(super) const WHEN_WORDS: usize = 40;

pub(super) const WHERE_WORDS: usize = 40;

pub(super) const WHO_WORDS: usize = 80;

/// The longest one entry of a list may run: a person and who they are, a fact
/// and where it is written.
pub(super) const ENTRY_WORDS: usize = 60;

/// The most people one scene names — a scene of more is a crowd, not a cast.
pub(super) const PEOPLE_MOST: usize = 6;

/// One person in `who`: a name and who they are, not their life — the list
/// whole stays inside [`WHO_WORDS`].
pub(super) const PERSON_WORDS: usize = 12;

/// The most facts a piece is held to.
pub(super) const FACTS_MOST: usize = 6;
