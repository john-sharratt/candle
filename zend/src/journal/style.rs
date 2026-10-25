//! How a coding session's journal reads: dated in UTC, in the words of a
//! session with a user and a workspace.
//!
//! An entry is rendered once, when it is kept, because the sealed K/V a journal
//! section becomes cannot be rendered again when it is gathered: each entry
//! therefore carries an absolute time (`[2026-10-23 14:20–14:41 UTC]`), never
//! "twenty minutes ago".

use journal_core::period::{civil_from_days, utc_bucket, Period};
use journal_core::style::Style;

use crate::layers::days::day_of;
use crate::mcp::threads::conv_names;

const MS_PER_DAY: u64 = 86_400_000;
const MS_PER_MINUTE: u64 = 60_000;

/// The write question, with `$more` said after it.
macro_rules! write_question {
    ($more:literal) => {
        concat!("Write it up with journal_write. An entry records what was \
significant in this stretch, not a retelling of it: whoever reads your journal later sees the \
entry, and goes to the turns themselves when they need the detail. For `claims`, write what \
matters from here on — what was asked for, what was found (a cause, a result that settled a \
question, the definition or line that decided something), each decision, each change made, and \
what was concluded and told — and leave out the steps that only led there: the searches, the \
reads that showed nothing, the calls made again. One fact per claim, in one short, specific \
sentence — names, paths and numbers as they were — and `cite` the number of the turn it rests \
on (`also_cite` for a second). A tool result you read is cited by its number; so is what you \
were told, and so is a call you made. Something you worked out and did \
not see in a result still cites the turn that led you to it. Only a claim that is itself one \
fact about the workspace — that a file exists or not, which blob a file is at, which commit a \
branch is on — takes `typed`, with `subject` `file:<repo>/<path>` and `attribute` `exists` or \
`blob`, or `subject` `branch:<repo>` and `attribute` `head`, and the `value`; a claim about what \
a file says, or what a result showed, takes none. `summary` is the task or step this stretch \
was, in one short sentence — what was being done and where it got to, not each read: it is the \
line your journal shows, and the claims are kept behind it. `intend` is what you mean to do \
next, as short lines, at most three. `open` is what is still unsettled — a failing test, a question \
the user has not answered, a step not done: its `text`, and `relates` — `new` for something \
you were not carrying, `restates N` to say an open item again, `resolves N` when it has been \
settled. Put nothing in that did not happen: a conclusion you only worked out is written as \
one, and a result that may have changed since is written as it was when you read it.", $more)
    };
}

/// What the model is asked once the stretch is on the table.
///
/// The call is shown nowhere else, so the question says what each part of it is.
pub const WRITE_QUESTION: &str = write_question!("");

/// [`WRITE_QUESTION`] for a conversation with journals above it: a claim that
/// rests on one of their entries names it, and the check confirms the entry is
/// there and is an ancestor's (`docs/claude_zend_gateway.md` §5.2).
pub const LINEAGE_WRITE_QUESTION: &str = write_question!(
    " A claim that rests on an entry of a journal above you — a decision in your thread's \
notes, a standing rule of the project, a finding of the worker that started you — says so in \
its text, `per T:<thread>#<n>`, and takes `typed` with `subject` `entry:<journal>#<n>` (the \
journal's path or name as your journals show it), `attribute` `exists` and `value` `true`."
);

/// Offered after [`WRITE_QUESTION`] once the journal holds an entry.
///
/// A new entry is a step change; anything else is more of the task the latest
/// entry is about, and a skip files it under that entry.
pub const SKIP_QUESTION: &str = "Write an entry only for a step change: the user asking for \
something new, a new task begun, or a major step in the task done — a fix made, a cause found, \
a test passing or failing, a decision taken. If this stretch is more of the task your latest \
entry is about — more reading, more searching, more of the same step — answer with \
journal_skip instead, its `reason` in one short sentence; the stretch is filed under that \
entry.";

/// Asked once the entries of a period that is over are in front of the
/// writer. The call is shown nowhere else, so the question says what each part
/// of it is.
pub const FOLD_QUESTION: &str = "Fold them into one entry with journal_fold. `summary` is the \
line your journal will keep for the whole period, in one short sentence: the tasks it was \
and where they got to. `digest` is the account kept behind that line, for when you need more \
than it says: each task or request in the order it came, what was found and decided, what was \
changed — files, names, numbers as they were — what was told to the user, and what was left \
unfinished, in a few short paragraphs. Put nothing in that the entries do not say.";

const WEEKDAYS: [&str; 7] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
];

const MONTHS: [&str; 12] = [
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

/// `2026-10-23` for `days` after 1970-01-01.
fn civil(days: i64) -> String {
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}")
}

/// The UTC `period` that `ms` falls in, as the fold question names it.
fn period_name(period: Period, ms: u64) -> String {
    let day = (ms / MS_PER_DAY) as i64;
    // 1970-01-01 was a Thursday: three days on from Monday.
    let weekday = (day + 3).rem_euclid(7);
    match period {
        Period::Step | Period::Day => {
            format!("{} {}", WEEKDAYS[weekday as usize], civil(day))
        }
        Period::Week => format!("the week of Monday {}", civil(day - weekday)),
        Period::Month => {
            let (y, m, _) = civil_from_days(day);
            format!("{} {y}", MONTHS[(m - 1) as usize])
        }
        Period::Year => civil_from_days(day).0.to_string(),
    }
}

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

/// The UTC day of `ms`: `2026-10-23`.
fn day(ms: u64) -> String {
    day_of(ms)
}

/// The UTC hour `ms` falls in: `14:00 UTC`.
fn hour(ms: u64) -> String {
    format!("{:02}:00 UTC", (ms % MS_PER_DAY) / 3_600_000)
}

fn stretch(from_ms: u64, to_ms: u64) -> String {
    format!(
        "This stretch of the session, {}, as numbered turns:",
        span(from_ms, to_ms)
    )
}

/// The wording a coding session's journal is kept in.
pub const CODING: Style = Style {
    entry: "Entry",
    latest: "Your latest entry:",
    span,
    day,
    hour,
    saw: "Saw:",
    heard: "User said:",
    did: "Did:",
    inferred: "Concluded, not seen:",
    meaning_to: "Meaning to:",
    left_open: "Left open:",
    settled: "Settled:",
    corrected: " (the workspace reads",
    recheck: " (re-check)",
    open_heading: "Still open:",
    nothing_open: "Still open: nothing.",
    end: "That is the whole of your journal. Anything else you know of this conversation comes \
from its turns or from what was retrieved for it, not from your journal.",
    clock,
    stretch,
    write_question: WRITE_QUESTION,
    skip_question: SKIP_QUESTION,
    bucket: utc_bucket,
    period_name,
    fold_question: FOLD_QUESTION,
};

/// The wording of a task's journal — a conversation owned through `/mcp`
/// (`docs/claude_zend_gateway.md` §5): every user turn there is Claude's — the
/// task, an answer, a nudge — so what it heard is what Claude said.
pub const DELEGATED: Style = Style {
    heard: "Claude said:",
    write_question: LINEAGE_WRITE_QUESTION,
    ..CODING
};

/// The wording of a part's journal — what a task's worker started under
/// itself with `sub_run`: its user turns are that worker's.
pub const PART: Style = Style {
    heard: "Its worker said:",
    write_question: LINEAGE_WRITE_QUESTION,
    ..CODING
};

/// The wording `conv_id`'s journal is kept in.
pub fn style_for(conv_id: &str) -> &'static Style {
    match conv_names(conv_id) {
        Some((_, _, lineage)) if lineage.contains('/') => &PART,
        Some(_) => &DELEGATED,
        None => &CODING,
    }
}

#[cfg(test)]
mod tests {
    use journal_core::entry::{Claim, Entry, Item, Kind};
    use journal_core::period::Period;
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
            summary: "Fixed the scan's race.".into(),
            claims: vec![],
            intend: vec![],
            opened: vec![],
            resolved: vec![],
            period: Period::Step,
            digest: String::new(),
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
            e.render(None, &CODING),
            "2026-10-23, 14:00 UTC\nEntry 1 — Fixed the scan's race.",
            "the journal shows one line"
        );
        assert_eq!(
            e.detail(&CODING),
            "Entry 1 [2026-10-23 14:20–14:41 UTC]\n\
             Fixed the scan's race.\n\
             - Saw: cargo test failed in scan.rs. (re-check)\n\
             - User said: Asked for the BDP fix first. (re-check)\n\
             - Did: Edited scan.rs to bound the loop.\n\
             - Concluded, not seen: The race is in the reclaim path. (re-check)\n\
             Left open:\n- #2 Does the 3090 build pass?"
        );
    }

    /// **In a task's journal, what was heard is what Claude said**: every
    /// user turn of a conversation owned through `/mcp` is Claude's. Every
    /// other conversation keeps the user's words.
    #[test]
    fn a_tasks_journal_hears_claude() {
        let mut e = entry();
        e.claims = vec![claim("Asked for the expiry check.", Kind::Heard, false)];
        assert_eq!(
            e.detail(style_for("mcp:abc:review/verify")),
            "Entry 1 [2026-10-23 14:20–14:41 UTC]\n\
             Fixed the scan's race.\n\
             - Claude said: Asked for the expiry check."
        );
        assert_eq!(style_for("c-123").heard, "User said:");
        assert_eq!(
            style_for("mcp:abc:review/verify/read").heard,
            "Its worker said:"
        );
    }

    #[test]
    fn the_journal_ends_with_what_is_still_open_and_where_it_stops() {
        let open = [Item {
            id: 4,
            text: "Rerun the gate".into(),
        }];
        let end = "That is the whole of your journal. Anything else you know of this conversation \
                   comes from its turns or from what was retrieved for it, not from your journal.";
        assert_eq!(
            render_open(&open, &CODING),
            format!("Still open:\n#4 Rerun the gate\n\n{end}")
        );
        assert_eq!(
            render_open(&[], &CODING),
            format!("Still open: nothing.\n\n{end}")
        );
    }

    #[test]
    fn a_period_is_named_on_the_utc_calendar() {
        // 2026-10-23 is a Friday; its week began on Monday the 19th.
        assert_eq!(period_name(Period::Day, OCT_23_1420), "Friday 2026-10-23");
        assert_eq!(
            period_name(Period::Week, OCT_23_1420),
            "the week of Monday 2026-10-19"
        );
        assert_eq!(period_name(Period::Month, OCT_23_1420), "October 2026");
        assert_eq!(period_name(Period::Year, OCT_23_1420), "2026");
        let monday = OCT_23_1420 - 4 * MS_PER_DAY;
        assert_eq!(
            period_name(Period::Week, monday),
            "the week of Monday 2026-10-19"
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
