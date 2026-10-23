//! How a character's journal reads: dated on the world's calendar, in the
//! character's own voice.
//!
//! An entry is rendered once, when it is kept, because the sealed KV a journal
//! section becomes cannot be rendered again when it is gathered: the time it
//! carries is therefore absolute (`14 Jun 2187, 14:20`), never "twenty minutes
//! ago".

use journal_core::style::Style;

use crate::clock::{day_of, stamp, WorldTime};

/// What the character is asked once the stretch is on the table.
///
/// The call is not shown anywhere else, so the question says what each part of it
/// is.
pub const WRITE_QUESTION: &str = "Write it up with journal_write. `claims` is what happened, \
one thing each: its `text` in one short sentence, and `cite`, the number of the turn it rests \
on (`also_cite` for a second). Something you worked out and did not see or hear still cites \
the turn that led you to it. If a claim is a fact that can be looked up — what a gauge read, \
whose turn it is, where something is — add `typed`, with its `subject`, `attribute` and \
`value`. `intend` is what you mean to do next, as short lines, at most three. `open` is what \
is still unsettled: its `text`, and `relates` — `new` for something you were not carrying, \
`restates N` to say an open item again, `resolves N` when it has been settled. Keep the whole \
entry brief, and put nothing in that did not happen: a conclusion you only worked out is \
written as one, and something that may have changed since you looked is written as it was \
when you looked.";

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

fn stretch(from_ms: u64, to_ms: u64) -> String {
    format!(
        "What happened to you, from {} to {}:",
        stamp(from_ms),
        stamp(to_ms)
    )
}

/// The wording a character's journal is kept in.
pub const NPC: Style = Style {
    span,
    saw: "Saw:",
    heard: "Heard:",
    did: "Did:",
    inferred: "Concluded, not seen or heard:",
    meaning_to: "Meaning to:",
    left_open: "Left open:",
    settled: "Settled:",
    corrected: " (the world read differently than I first wrote)",
    recheck: " (re-check)",
    open_heading: "What you have left open:",
    clock,
    stretch,
    write_question: WRITE_QUESTION,
};

#[cfg(test)]
mod tests {
    use journal_core::ask;
    use journal_core::entry::{Claim, Entry, Item, Kind};
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
        }
    }

    fn entry(id: u64) -> Entry {
        Entry {
            id,
            from_turn: 1,
            to_turn: 9,
            from_ms: DAY_MS * 3 + 14 * H + 20 * M,
            to_ms: DAY_MS * 3 + 14 * H + 41 * M,
            claims: vec![],
            intend: vec![],
            opened: vec![],
            resolved: vec![],
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
    fn claims_are_grouped_by_how_they_came_to_be_known() {
        let mut e = entry(1);
        e.claims = vec![
            claim("Somebody is stealing power.", Kind::Inferred),
            claim("I told Pax I was coming.", Kind::Did),
            claim("Pax says the east door is jammed.", Kind::Heard),
            claim("The tower holds 40 ore.", Kind::Observed),
        ];
        assert_eq!(
            e.render(&NPC),
            "[4 Jan 1970, 14:20–14:41]\n\
             Saw:\n- The tower holds 40 ore.\n\
             Heard:\n- Pax says the east door is jammed.\n\
             Did:\n- I told Pax I was coming.\n\
             Concluded, not seen or heard:\n- Somebody is stealing power."
        );
    }

    #[test]
    fn a_perishable_claim_asks_to_be_rechecked_and_a_corrected_one_says_so() {
        let mut e = entry(1);
        let mut c = claim("Fabricator 5 is running.", Kind::Observed);
        c.perishable = true;
        c.corrected = true;
        e.claims = vec![c];
        assert!(e.render(&NPC).contains(
            "- Fabricator 5 is running. (the world read differently than I first wrote) (re-check)"
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
        let r = e.render(&NPC);
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
            render_open(&open, &NPC).as_deref(),
            Some("What you have left open:\n#2 who holds it?")
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
            },
            SpanTurn {
                id: 2,
                at_ms: at + 3 * M,
                evidence: None,
                text: "You: reflect — I feel uneasy".into(),
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
        assert!(ask::writing(&span, &NPC, None).ends_with(WRITE_QUESTION));
    }
}
