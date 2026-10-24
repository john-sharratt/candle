//! Folding the entries of a period that is over into one entry.
//!
//! [`next`] says which entries are due: the lowest level first, so a day's
//! steps fold into the day before the day folds into its week, and within a
//! level the earliest period first. A period is over when the journal's
//! calendar ([`Style::bucket`]) has moved past the one each entry *ended* in —
//! a step a skip ran on into today belongs to today.
//!
//! The writer folds them itself: [`question`] puts every entry of the period in
//! front of it, whole, and the answer is one `journal_fold` call — a one-line
//! `summary`, which is all the journal will show of the period, and a `digest`,
//! the detailed account kept behind that line for the journal's tools. Like the
//! draft's calls it is only ever an answer grammar, never in a catalog.

use serde_json::{json, Value};

use crate::entry::Entry;
use crate::period::Period;
use crate::style::Style;
use crate::tools::{arguments, text, text_of};

/// The fold itself.
pub const FOLD: &str = "journal_fold";

/// The entries of one period that is over, and what they fold into.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fold {
    pub into: Period,
    /// Their numbers, in journal order.
    pub ids: Vec<u64>,
}

/// The next fold `entries` are due at `now_ms`, if any.
///
/// A period folds only once nothing below its level is still waiting in it:
/// a month's weeks fold into the month only when its last days have folded
/// into their week, so a week that runs past the month's end is not left to
/// make a second entry for the same month. An entry with no time on it —
/// turns recovered without one read as the epoch — sits on no calendar and
/// never folds.
pub fn next(entries: &[Entry], now_ms: u64, style: &Style) -> Option<Fold> {
    let levels = [Period::Step, Period::Day, Period::Week, Period::Month];
    let dated: Vec<&Entry> = entries.iter().filter(|e| e.to_ms > 0).collect();
    levels.into_iter().find_map(|level| {
        let into = level.folds_into()?;
        let bucket = |e: &Entry| (style.bucket)(into, e.to_ms);
        let now = (style.bucket)(into, now_ms);
        let waiting_below = |b: u64| dated.iter().any(|e| e.period < level && bucket(e) == b);
        let over: Vec<&Entry> = dated
            .iter()
            .copied()
            .filter(|e| e.period == level && bucket(e) < now && !waiting_below(bucket(e)))
            .collect();
        let first = over.iter().map(|e| bucket(e)).min()?;
        Some(Fold {
            into,
            ids: over
                .iter()
                .filter(|e| bucket(e) == first)
                .map(|e| e.id)
                .collect(),
        })
    })
}

/// The entries of `entries` that `fold` takes, in journal order.
fn folded<'a>(fold: &Fold, entries: &'a [Entry]) -> Vec<&'a Entry> {
    entries
        .iter()
        .filter(|e| fold.ids.contains(&e.id))
        .collect()
}

/// `s` with its first letter in capitals.
fn capitalised(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => String::new(),
    }
}

/// What the writer is asked: the period that is over, every entry of it
/// whole, and the call.
pub fn question(fold: &Fold, entries: &[Entry], style: &Style) -> String {
    let taken = folded(fold, entries);
    let at = taken.first().map_or(0, |e| e.to_ms);
    let details: Vec<String> = taken.iter().map(|e| e.detail(style)).collect();
    format!(
        "{} is over. These are your journal's entries from it:\n\n{}\n\n{}",
        capitalised(&(style.period_name)(fold.into, at)),
        details.join("\n\n"),
        style.fold_question
    )
}

/// The `journal_fold` call: the line the journal keeps and the digest behind
/// it.
pub fn spec() -> Value {
    json!({ "name": FOLD, "params": [text("summary", true), text("digest", true)] })
}

/// Read one answer as `(summary, digest)`. The error is worded for the model.
pub fn parse(answer: &str) -> Result<(String, String), String> {
    let (name, args) =
        arguments(answer).ok_or_else(|| "That was not a journal_fold call.".to_string())?;
    if name != FOLD {
        return Err(format!(
            "There is no call named `{name}` here; fold with journal_fold."
        ));
    }
    let summary = args
        .get("summary")
        .and_then(text_of)
        .ok_or("journal_fold needs a one-line `summary`.")?;
    let digest = args
        .get("digest")
        .and_then(text_of)
        .ok_or("journal_fold needs a `digest`.")?;
    Ok((summary, digest))
}

/// The entry `fold` makes of `entries`: it spans all of them, and carries the
/// writer's `summary` and `digest` and nothing else — the open items are the
/// journal's, not any entry's.
pub fn entry(fold: &Fold, entries: &[Entry], summary: String, digest: String) -> Entry {
    let taken = folded(fold, entries);
    let min = |f: fn(&Entry) -> u64| taken.iter().map(|e| f(e)).min().unwrap_or(0);
    let max = |f: fn(&Entry) -> u64| taken.iter().map(|e| f(e)).max().unwrap_or(0);
    Entry {
        id: 0,
        from_turn: min(|e| e.from_turn),
        to_turn: max(|e| e.to_turn),
        from_ms: min(|e| e.from_ms),
        to_ms: max(|e| e.to_ms),
        summary,
        claims: Vec::new(),
        intend: Vec::new(),
        opened: Vec::new(),
        resolved: Vec::new(),
        period: fold.into,
        digest,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entry::testing::entry as step;
    use crate::style::testing::PLAIN;

    /// Entry `id` of `period`, from `from_ms` to `to_ms` (PLAIN's day is 1000 ms).
    fn at(id: u64, period: Period, from_ms: u64, to_ms: u64) -> Entry {
        Entry {
            from_turn: id * 10,
            to_turn: id * 10 + 9,
            from_ms,
            to_ms,
            period,
            ..step(id)
        }
    }

    #[test]
    fn nothing_folds_while_its_period_is_running() {
        let today = [
            at(1, Period::Step, 2_100, 2_200),
            at(2, Period::Step, 2_300, 2_400),
        ];
        assert_eq!(next(&today, 2_999, &PLAIN), None);
    }

    #[test]
    fn a_finished_days_steps_fold_into_the_day_earliest_day_first() {
        let j = [
            at(1, Period::Step, 1_100, 1_200),
            at(2, Period::Step, 1_500, 1_900),
            at(3, Period::Step, 2_100, 2_200),
            at(4, Period::Step, 3_100, 3_200),
        ];
        assert_eq!(
            next(&j, 3_500, &PLAIN),
            Some(Fold {
                into: Period::Day,
                ids: vec![1, 2]
            })
        );
    }

    #[test]
    fn a_step_belongs_to_the_day_it_ended_in() {
        let ran_on = [at(1, Period::Step, 1_900, 2_100)];
        assert_eq!(next(&ran_on, 2_500, &PLAIN), None, "it ended today");
    }

    #[test]
    fn steps_fold_before_days_and_days_before_weeks() {
        let j = [
            at(1, Period::Day, 1_000, 1_999),
            at(2, Period::Step, 8_100, 8_200),
        ];
        assert_eq!(next(&j, 9_000, &PLAIN).unwrap().into, Period::Day);
        let days = [
            at(1, Period::Day, 1_000, 1_999),
            at(3, Period::Day, 8_000, 8_999),
        ];
        assert_eq!(
            next(&days, 9_000, &PLAIN),
            Some(Fold {
                into: Period::Week,
                ids: vec![1]
            }),
            "week 0 is over; week 1 is still running"
        );
    }

    /// The review's case: a month is over while its last week still runs. Its
    /// earlier weeks wait for that week, so the month is made once.
    #[test]
    fn a_month_waits_for_its_last_days_to_fold_into_their_week() {
        // PLAIN: a week is 7000 ms, a month 30 000. Month 0 ends at 29 999;
        // the week 28 000..34 999 runs across its end.
        let j = [
            at(1, Period::Week, 21_000, 27_999),
            at(2, Period::Day, 28_000, 28_999),
        ];
        assert_eq!(
            next(&j, 31_000, &PLAIN),
            None,
            "day 28 still waits for its week, so month 0 does not fold"
        );
        let later = [
            at(1, Period::Week, 21_000, 27_999),
            at(2, Period::Week, 28_000, 28_999),
        ];
        assert_eq!(
            next(&later, 35_000, &PLAIN),
            Some(Fold {
                into: Period::Month,
                ids: vec![1, 2]
            }),
            "once the week has folded, month 0 folds both"
        );
    }

    #[test]
    fn an_entry_with_no_time_never_folds() {
        let undated = [at(1, Period::Step, 0, 0)];
        assert_eq!(next(&undated, 9_000_000, &PLAIN), None);
    }

    #[test]
    fn weeks_fold_into_months_and_months_into_years_and_years_never() {
        let weeks = [at(1, Period::Week, 1_000, 6_999)];
        assert_eq!(next(&weeks, 31_000, &PLAIN).unwrap().into, Period::Month);
        let months = [at(1, Period::Month, 1_000, 29_999)];
        assert_eq!(next(&months, 400_000, &PLAIN).unwrap().into, Period::Year);
        let years = [at(1, Period::Year, 1_000, 359_999)];
        assert_eq!(next(&years, 9_000_000, &PLAIN), None);
    }

    #[test]
    fn the_question_puts_the_period_and_each_entry_whole_before_the_call() {
        let mut a = at(1, Period::Step, 1_100, 1_200);
        a.summary = "Read scan.rs.".into();
        let mut b = at(2, Period::Step, 1_500, 1_900);
        b.summary = "Fixed the race.".into();
        let fold = Fold {
            into: Period::Day,
            ids: vec![1, 2],
        };
        assert_eq!(
            question(&fold, &[a.clone(), b.clone()], &PLAIN),
            format!(
                "Day 1 is over. These are your journal's entries from it:\n\n{}\n\n{}\n\n\
                 Fold them with journal_fold.",
                a.detail(&PLAIN),
                b.detail(&PLAIN)
            )
        );
    }

    #[test]
    fn the_folded_entry_spans_what_it_folds_and_holds_only_its_line_and_digest() {
        let j = [
            at(1, Period::Step, 1_100, 1_200),
            at(2, Period::Step, 1_500, 1_900),
            at(3, Period::Step, 2_100, 2_200),
        ];
        let fold = Fold {
            into: Period::Day,
            ids: vec![1, 2],
        };
        let e = entry(
            &fold,
            &j,
            "Fixed the race.".into(),
            "Read, then fixed.".into(),
        );
        assert_eq!(
            (e.from_turn, e.to_turn, e.from_ms, e.to_ms),
            (10, 29, 1_100, 1_900)
        );
        assert_eq!((e.period, e.id), (Period::Day, 0));
        assert_eq!(
            (e.summary.as_str(), e.digest.as_str()),
            ("Fixed the race.", "Read, then fixed.")
        );
        assert!(e.claims.is_empty() && e.opened.is_empty());
    }

    #[test]
    fn an_answer_is_read_as_its_line_and_digest() {
        let said = r#"{"name": "journal_fold", "arguments": {"summary": "A day.", "digest": "All of it."}}"#;
        assert_eq!(parse(said), Ok(("A day.".into(), "All of it.".into())));
        assert_eq!(
            parse(r#"{"name": "journal_fold", "arguments": {"summary": "A day."}}"#),
            Err("journal_fold needs a `digest`.".into())
        );
        assert_eq!(
            parse(r#"{"name": "journal_write", "arguments": {}}"#),
            Err("There is no call named `journal_write` here; fold with journal_fold.".into())
        );
        assert_eq!(
            parse("A good day."),
            Err("That was not a journal_fold call.".into())
        );
    }

    #[test]
    fn the_fold_call_asks_a_summary_and_a_digest() {
        assert_eq!(
            spec(),
            json!({"name": "journal_fold", "params": [
                {"name": "summary", "type": "string", "required": true},
                {"name": "digest", "type": "string", "required": true},
            ]})
        );
    }
}
