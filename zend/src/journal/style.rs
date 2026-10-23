//! How a coding session's journal reads: dated in UTC, in the words of a
//! session with a user and a workspace.
//!
//! An entry is rendered once, when it is kept, because the sealed K/V a journal
//! section becomes cannot be rendered again when it is gathered: each entry
//! therefore carries an absolute time (`[2026-10-23 14:20–14:41 UTC]`), never
//! "twenty minutes ago".

use journal_core::style::Style;

use crate::layers::days::day_of;

const MS_PER_DAY: u64 = 86_400_000;
const MS_PER_MINUTE: u64 = 60_000;

/// What the model is asked once the stretch is on the table.
///
/// The call is shown nowhere else, so the question says what each part of it is.
pub const WRITE_QUESTION: &str = "Write it up with journal_write. `claims` is what this \
stretch established, one thing each — what the user asked for, each file or result read and \
what it showed, each decision, each change made — so a stretch of several steps is several \
claims, as many as it took to get there, up to twelve. Each claim's `text` is one short \
sentence, and `cite` the number of the turn it rests on (`also_cite` for a second). A tool result you read is cited by its \
number; so is what the user said, and so is a call you made. Something you worked out and did \
not see in a result still cites the turn that led you to it. If a claim is about the workspace \
— whether a file exists, which blob a file is at, which commit a branch is on — add `typed`, \
with `subject` `file:<repo>/<path>` and `attribute` `exists` or `blob`, or `subject` \
`branch:<repo>` and `attribute` `head`, and the `value`. `intend` is what you mean to do next, \
as short lines, at most three. `open` is what is still unsettled — a failing test, a question \
the user has not answered, a step not done: its `text`, and `relates` — `new` for something \
you were not carrying, `restates N` to say an open item again, `resolves N` when it has been \
settled. Keep the whole entry brief, and put nothing in that did not happen: a conclusion you \
only worked out is written as one, and a result that may have changed since is written as it \
was when you read it.";

/// `HH:MM` of the wall-clock instant `ms`, UTC.
fn clock(ms: u64) -> String {
    let minutes = (ms % MS_PER_DAY) / MS_PER_MINUTE;
    format!("{:02}:{:02}", minutes / 60, minutes % 60)
}

/// The span an entry covers: `[2026-10-23 14:20–14:41 UTC]`, or both ends in
/// full when it crosses midnight.
fn span(from_ms: u64, to_ms: u64) -> String {
    let (from_day, to_day) = (day_of(from_ms), day_of(to_ms));
    if from_day == to_day {
        format!("[{from_day} {}–{} UTC]", clock(from_ms), clock(to_ms))
    } else {
        format!(
            "[{from_day} {} – {to_day} {} UTC]",
            clock(from_ms),
            clock(to_ms)
        )
    }
}

fn stretch(from_ms: u64, to_ms: u64) -> String {
    format!(
        "This stretch of the session, {}, as numbered turns:",
        span(from_ms, to_ms)
    )
}

/// The wording a coding session's journal is kept in.
pub const CODING: Style = Style {
    span,
    saw: "Saw:",
    heard: "User said:",
    did: "Did:",
    inferred: "Concluded, not seen:",
    meaning_to: "Meaning to:",
    left_open: "Left open:",
    settled: "Settled:",
    corrected: " (the workspace read differently than first written)",
    recheck: " (re-check)",
    open_heading: "Still open:",
    clock,
    stretch,
    write_question: WRITE_QUESTION,
};

#[cfg(test)]
mod tests {
    use journal_core::entry::{Claim, Entry, Item, Kind};
    use journal_core::section::render_open;

    use super::*;

    /// 2026-10-23T14:20:00Z.
    const OCT_23_1420: u64 = 1_792_713_600_000 + 14 * 3_600_000 + 20 * MS_PER_MINUTE;

    fn claim(text: &str, kind: Kind, perishable: bool) -> Claim {
        Claim {
            text: text.into(),
            cite: vec![3],
            kind,
            perishable,
            typed: None,
            corrected: false,
        }
    }

    fn entry() -> Entry {
        Entry {
            id: 1,
            from_turn: 1,
            to_turn: 9,
            from_ms: OCT_23_1420,
            to_ms: OCT_23_1420 + 21 * MS_PER_MINUTE,
            claims: vec![],
            intend: vec![],
            opened: vec![],
            resolved: vec![],
        }
    }

    #[test]
    fn a_span_inside_a_day_names_the_day_once_in_utc() {
        assert_eq!(entry().span(&CODING), "[2026-10-23 14:20–14:41 UTC]");
    }

    #[test]
    fn a_span_across_midnight_names_both_days() {
        let mut e = entry();
        e.from_ms = OCT_23_1420 + 9 * 3_600_000 + 30 * MS_PER_MINUTE;
        e.to_ms = e.from_ms + 40 * MS_PER_MINUTE;
        assert_eq!(e.span(&CODING), "[2026-10-23 23:50 – 2026-10-24 00:30 UTC]");
    }

    #[test]
    fn an_entry_reads_in_the_words_of_a_coding_session() {
        let mut e = entry();
        e.claims = vec![
            claim("cargo test failed in scan.rs.", Kind::Observed, true),
            claim("Asked for the BDP fix first.", Kind::Heard, true),
            claim("Edited scan.rs to bound the loop.", Kind::Did, false),
            claim("The race is in the reclaim path.", Kind::Inferred, true),
        ];
        e.opened = vec![Item {
            id: 2,
            text: "Does the 3090 build pass?".into(),
        }];
        assert_eq!(
            e.render(&CODING),
            "[2026-10-23 14:20–14:41 UTC]\n\
             Saw:\n- cargo test failed in scan.rs. (re-check)\n\
             User said:\n- Asked for the BDP fix first. (re-check)\n\
             Did:\n- Edited scan.rs to bound the loop.\n\
             Concluded, not seen:\n- The race is in the reclaim path. (re-check)\n\
             Left open:\n- #2 Does the 3090 build pass?"
        );
    }

    #[test]
    fn the_open_items_are_headed_as_still_open() {
        let open = [Item {
            id: 4,
            text: "Rerun the gate".into(),
        }];
        assert_eq!(
            render_open(&open, &CODING).as_deref(),
            Some("Still open:\n#4 Rerun the gate")
        );
    }

    #[test]
    fn the_clock_is_hours_and_minutes_of_the_utc_day() {
        assert_eq!(clock(OCT_23_1420), "14:20");
        assert_eq!(clock(OCT_23_1420 + 59 * 1_000), "14:20");
        assert_eq!(clock(0), "00:00");
    }

    #[test]
    fn the_stretch_says_when_it_was() {
        assert_eq!(
            stretch(OCT_23_1420, OCT_23_1420 + MS_PER_MINUTE),
            "This stretch of the session, [2026-10-23 14:20–14:21 UTC], as numbered turns:"
        );
    }
}
