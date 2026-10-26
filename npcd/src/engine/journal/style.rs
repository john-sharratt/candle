//! How a character's journal reads: dated on the world's calendar, in the
//! character's own voice.
//!
//! An entry is rendered once, when it is kept, because the sealed KV a journal
//! section becomes cannot be rendered again when it is gathered: the time it
//! carries is therefore absolute (`14 Jun 2187, 14:20`), never "twenty minutes
//! ago".

use journal_core::period::{utc_bucket, Period};
use journal_core::style::Style;

use crate::clock::{day_of, stamp, WorldTime, DAY_MS};

/// What the character is asked once the stretch is on the table.
///
/// The call is not shown anywhere else, so the question says what each part of it
/// is.
pub const WRITE_QUESTION: &str = "Write it up with journal_write. `claims` is what happened, \
one thing each: its `text` in one short sentence, and `cite`, the number of the turn it rests \
on (`also_cite` for a second). Something you worked out and did not see or hear still cites \
the turn that led you to it. Only a claim that is itself one fact that can be looked up — what \
a gauge read, whose turn it is, where something is — takes `typed`, with its `subject`, \
`attribute` and `value`; any other claim takes none. `summary` is what you were doing in this \
stretch and where it got to, in one short sentence: it is the line your journal shows, and the \
claims are kept behind it. `intend` is what you mean to do next, as short lines, at most three. \
`open` is what \
is still unsettled: its `text`, and `relates` — `new` for something you were not carrying, \
`restates N` to say an open item again, `resolves N` when it has been settled. Keep the whole \
entry brief, and put nothing in that did not happen: a conclusion you only worked out is \
written as one, and something that may have changed since you looked is written as it was \
when you looked.";

/// Offered after [`WRITE_QUESTION`] once the journal holds an entry.
///
/// A new entry is a step change; anything else is more of what the latest
/// entry is about, and a skip files it under that entry.
pub const SKIP_QUESTION: &str = "Write an entry only for a step change: something new asked of \
you, a new errand begun, or a real step in it taken — something found, done, decided or \
refused. If this stretch is more of what your latest entry is about — more walking, looking or \
talking around the same thing — answer with journal_skip instead, its `reason` in one short \
sentence; the stretch is filed under that entry.";

/// Asked once the entries of a world period that is over are in front of the
/// character.
pub const FOLD_QUESTION: &str = "Fold them into one entry with journal_fold. `summary` is the \
line your journal will keep for the whole period, in one short sentence: what you were about \
and where it got to. `digest` is the account kept behind that line: each errand or thing asked \
of you in the order it came, what you found, did and decided, who you dealt with, and what you \
left unfinished, in a few short paragraphs. Put nothing in that the entries do not say.";

const MONTH_NAMES: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];

/// The world `period` that `at_ms` falls in, as the fold question names it:
/// `14 Jun 2187`, `the week of 8 Jun 2187`, `June 2187`, `2187`. The world's
/// calendar is Gregorian, its weeks starting on Monday.
fn period_name(period: Period, at_ms: u64) -> String {
    let time = WorldTime::of(at_ms);
    match period {
        Period::Step | Period::Day => time.date(),
        Period::Week => {
            // 1970-01-01 was a Thursday: three days on from Monday.
            let back = (day_of(at_ms) + 3) % 7;
            format!(
                "the week of {}",
                WorldTime::of(at_ms - back * DAY_MS).date()
            )
        }
        Period::Month => format!("{} {}", MONTH_NAMES[time.month as usize - 1], time.year),
        Period::Year => time.year.to_string(),
    }
}

/// The span an entry covers: `[14 Jun 2187, 14:20–14:41]`, or both ends in full
/// when it crosses midnight.
fn span(from_ms: u64, to_ms: u64) -> String {
    let (a, b) = (WorldTime::of(from_ms), WorldTime::of(to_ms));
    if day_of(from_ms) == day_of(to_ms) {
        format!("[{}, {}–{}]", a.date(), a.clock(), b.clock())
    } else {
        format!("[{} – {}]", a.stamp(), b.stamp())
    }
}

/// The clock part of a stamp: `14:20` out of `14 Jun 2187, 14:20`.
fn clock(at_ms: u64) -> String {
    WorldTime::of(at_ms).clock()
}

/// The world's day `at_ms` falls on: `14 Jun 2187`.
fn day(at_ms: u64) -> String {
    WorldTime::of(at_ms).date()
}

/// The world's hour `at_ms` falls in: `14:00`.
fn hour(at_ms: u64) -> String {
    let clock = WorldTime::of(at_ms).clock();
    format!("{}:00", clock.split(':').next().unwrap_or("00"))
}

fn stretch(from_ms: u64, to_ms: u64) -> String {
    format!(
        "What happened to you, from {} to {}:",
        stamp(from_ms),
        stamp(to_ms)
    )
}

/// The wording a character's journal is kept in.
pub const NPC: Style = Style {
    entry: "Entry",
    latest: "Your latest entry:",
    span,
    day,
    hour,
    saw: "Saw:",
    heard: "Heard:",
    did: "Did:",
    inferred: "Concluded, not seen or heard:",
    meaning_to: "Meaning to:",
    left_open: "Left open:",
    settled: "Settled:",
    corrected: " (the world reads",
    recheck: " (re-check)",
    rests_on: " (per",
    // A character's claims are about the world's state — energy, stockpiles,
    // what a tower is doing — which the world reads, so a fact is checked.
    facts: true,
    open_heading: "What you have left open:",
    nothing_open: "What you have left open: nothing.",
    end: "That is the whole of your journal.",
    clock,
    stretch,
    write_question: WRITE_QUESTION,
    // A character has no journals above it, so `per` is never offered.
    per_question: "",
    skip_question: SKIP_QUESTION,
    bucket: utc_bucket,
    period_name,
    fold_question: FOLD_QUESTION,
};

#[cfg(test)]
mod tests {
    use journal_core::ask;
    use journal_core::entry::{Claim, Entry, Item, Kind, Typed};
    use journal_core::period::Period;
    use journal_core::section::render_open;
    use journal_core::span::{Span, SpanTurn};

    use super::*;
    use crate::clock::DAY_MS;

    const H: u64 = 3_600_000;
    const M: u64 = 60_000;

    fn claim(text: &str, kind: Kind) -> Claim {
        Claim {
            text: text.into(),
            cite: vec![1],
            kind,
            perishable: false,
            typed: None,
            corrected: false,
            per: None,
        }
    }

    fn entry(id: u64) -> Entry {
        Entry {
            id,
            from_turn: 1,
            to_turn: 9,
            from_ms: DAY_MS * 3 + 14 * H + 20 * M,
            to_ms: DAY_MS * 3 + 14 * H + 41 * M,
            summary: "Went to see Pax.".into(),
            claims: vec![],
            intend: vec![],
            opened: vec![],
            resolved: vec![],
            period: Period::Step,
            digest: String::new(),
        }
    }

    #[test]
    fn a_span_inside_a_day_names_the_day_once() {
        assert_eq!(entry(1).span(&NPC), "[4 Jan 1970, 14:20–14:41]");
    }

    #[test]
    fn a_span_across_midnight_names_both_days() {
        let mut e = entry(1);
        e.from_ms = DAY_MS * 3 + 23 * H + 50 * M;
        e.to_ms = DAY_MS * 4 + 10 * M;
        assert_eq!(e.span(&NPC), "[4 Jan 1970, 23:50 – 5 Jan 1970, 00:10]");
    }

    #[test]
    fn claims_read_in_order_each_under_how_it_came_to_be_known() {
        let mut e = entry(1);
        e.claims = vec![
            claim("Somebody is stealing power.", Kind::Inferred),
            claim("I told Pax I was coming.", Kind::Did),
            claim("Pax says the east door is jammed.", Kind::Heard),
            claim("The tower holds 40 ore.", Kind::Observed),
        ];
        assert_eq!(
            e.render(None, &NPC),
            "4 Jan 1970, 14:00\nEntry 1 — Went to see Pax.",
            "the journal shows one line"
        );
        assert_eq!(
            e.detail(&NPC),
            "Entry 1 [4 Jan 1970, 14:20–14:41]\n\
             Went to see Pax.\n\
             - Concluded, not seen or heard: Somebody is stealing power.\n\
             - Did: I told Pax I was coming.\n\
             - Heard: Pax says the east door is jammed.\n\
             - Saw: The tower holds 40 ore."
        );
    }

    #[test]
    fn a_perishable_claim_asks_to_be_rechecked_and_a_corrected_one_says_so() {
        let mut e = entry(1);
        let mut c = claim("Fabricator 5 is running.", Kind::Observed);
        c.perishable = true;
        c.corrected = true;
        c.typed = Some(Typed {
            subject: "fabricator 5".into(),
            attribute: "mode".into(),
            value: "idle".into(),
        });
        e.claims = vec![c];
        assert!(e.detail(&NPC).contains(
            "- Saw: Fabricator 5 is running. (the world reads fabricator 5 mode as idle) \
             (re-check)"
        ));
    }

    #[test]
    fn open_items_carry_their_ids_and_settled_ones_are_named_by_id() {
        let mut e = entry(2);
        e.intend = vec!["Ask Pax about the door.".into()];
        e.opened = vec![Item {
            id: 4,
            text: "Who holds the shield decision?".into(),
        }];
        e.resolved = vec![2, 3];
        let r = e.detail(&NPC);
        assert!(r.contains("Meaning to:\n- Ask Pax about the door."));
        assert!(r.contains("Left open:\n- #4 Who holds the shield decision?"));
        assert!(r.ends_with("Settled:\n- #2\n- #3"));
    }

    #[test]
    fn the_open_items_section_is_headed_in_the_characters_voice() {
        let open = [Item {
            id: 2,
            text: " who holds it? ".into(),
        }];
        assert_eq!(
            render_open(&open, &NPC),
            "What you have left open:\n#2 who holds it?\n\nThat is the whole of your journal."
        );
        assert_eq!(
            render_open(&[], &NPC),
            "What you have left open: nothing.\n\nThat is the whole of your journal."
        );
    }

    #[test]
    fn the_stretch_is_dated_on_the_world_calendar_and_clocked_per_turn() {
        let at = 2 * DAY_MS + 14 * H + 20 * M;
        let span = Span::of(vec![
            SpanTurn {
                id: 1,
                at_ms: at,
                evidence: Some(Kind::Heard),
                text: "Pax: Fabricator 5 is lost.".into(),
                full: "Pax: Fabricator 5 is lost.".into(),
            },
            SpanTurn {
                id: 2,
                at_ms: at + 3 * M,
                evidence: None,
                text: "You: reflect — I feel uneasy".into(),
                full: "You: reflect — I feel uneasy".into(),
            },
        ])
        .unwrap();
        let text = ask::stretch(&span, &NPC);
        assert!(
            text.starts_with("What happened to you, from 3 Jan 1970, 14:20 to 3 Jan 1970, 14:23:"),
            "{text}"
        );
        assert!(
            text.contains("\n#1 14:20 Pax: Fabricator 5 is lost."),
            "{text}"
        );
        assert!(
            text.contains("\n   14:23 (background) You: reflect — I feel uneasy"),
            "{text}"
        );
        assert!(ask::writing(&span, &NPC, None, false, None).ends_with(WRITE_QUESTION));
        let latest = Entry {
            summary: "Went to see Pax.".into(),
            ..entry(3)
        };
        assert!(
            ask::writing(&span, &NPC, Some(&latest), false, None).ends_with(&format!(
            "Your latest entry: Entry 3 — Went to see Pax.\n\n{SKIP_QUESTION}\n\n{WRITE_QUESTION}"
        ))
        );
    }
}
