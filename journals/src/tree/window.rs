//! Which closed period of a window folds next (§4).
//!
//! A window over [`WINDOW`] lines folds its oldest lines by period. For the
//! oldest line, the lowest level above its own whose period has closed and
//! holds at least two of the window's lines below that level is the period
//! to fold: those lines' files move to `archive/<period>/` and one `fold` line
//! takes their place. A day of one lone note does not fold on its own; it
//! waits for its week, so every fold shortens the window. If the oldest line
//! has nothing to fold with, the next oldest is tried.

use crate::line::{Line, LineId};
use crate::period::Period;
use crate::time::Stamp;
use crate::tree::{Journal, WINDOW};

/// A fold to make: the period, and the window's lines inside it, oldest
/// first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PeriodFold {
    pub period: Period,
    pub ids: Vec<LineId>,
}

/// The window's lines that a fold of `period` takes: those below its level
/// that it holds.
pub fn inside(j: &Journal, period: Period) -> Vec<LineId> {
    j.window
        .iter()
        .filter(|l| l.level() < period.level() && period.holds(l.unit()))
        .map(|l| l.id.clone())
        .collect()
}

/// The next fold `j`'s window wants at `now`; `None` while it is within
/// [`WINDOW`] lines or nothing closed has two lines to fold.
pub fn next_fold(j: &Journal, now: Stamp) -> Option<PeriodFold> {
    if j.window.len() <= WINDOW {
        return None;
    }
    let mut order: Vec<&Line> = j.window.iter().collect();
    order.sort_by_key(|l| l.at);
    for line in order {
        let mut level = line.level().up();
        while let Some(lv) = level {
            if let Some(period) = Period::of(lv, line.unit().start()) {
                if period.is_closed(now) {
                    let ids = inside(j, period);
                    if ids.len() >= 2 {
                        return Some(PeriodFold { period, ids });
                    }
                } else {
                    // Higher periods hold this one, so are no more closed.
                    break;
                }
            }
            level = lv.up();
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::files::Meta;
    use crate::line::{Folds, Kind, Writer};
    use crate::path::Place;
    use crate::scope::Section;
    use crate::testing::{at, note};

    fn journal(lines: Vec<Line>) -> Journal {
        let mut j = Journal::new(
            Place::Named {
                section: Section::Topics,
                depth: 1,
            },
            Some(Meta::new("t", Some("x".into()), at("2026-01-01T00:00:00Z"))),
        );
        j.window = lines;
        j
    }

    /// `n` notes an hour apart from `start`.
    fn notes(start: &str, n: usize) -> Vec<Line> {
        let t0 = at(start);
        (0..n)
            .map(|i| note(&t0.plus(3600 * i as i64).to_string(), &format!("n{i}")))
            .collect()
    }

    fn fold_line(period: &str) -> Line {
        let p = Period::parse(period).unwrap();
        let mut l = note(&p.start().to_string(), &format!("fold {period}"));
        l.kind = Kind::Fold;
        l.by = Writer::Zend;
        l.folds = Some(Folds::Period(p));
        l
    }

    #[test]
    fn a_window_within_its_size_never_folds() {
        let j = journal(notes("2026-10-01T00:00:00Z", WINDOW));
        assert_eq!(next_fold(&j, at("2027-01-01T00:00:00Z")), None);
    }

    #[test]
    fn the_oldest_closed_day_folds_first() {
        // 51 hourly notes from 1 Oct 00:00: 24 on the 1st, 24 on the 2nd, 3 on the 3rd.
        let lines = notes("2026-10-01T00:00:00Z", WINDOW + 1);
        let j = journal(lines.clone());
        let f = next_fold(&j, at("2026-10-03T12:00:00Z")).unwrap();
        assert_eq!(f.period, Period::parse("2026-10-01").unwrap());
        assert_eq!(
            f.ids,
            lines[..24].iter().map(|l| l.id.clone()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn nothing_folds_while_every_period_is_open() {
        let j = journal(notes("2026-10-01T00:00:00Z", WINDOW + 1));
        assert_eq!(next_fold(&j, at("2026-10-01T23:00:00Z")), None);
    }

    #[test]
    fn lone_days_wait_for_their_week() {
        // One note a day for 51 days: no day holds two, so weeks fold.
        let t0 = at("2026-10-05T12:00:00Z"); // a Monday
        let lines: Vec<_> = (0..WINDOW + 1)
            .map(|i| note(&t0.plus(86_400 * i as i64).to_string(), &format!("d{i}")))
            .collect();
        let j = journal(lines.clone());
        let f = next_fold(&j, at("2027-01-01T00:00:00Z")).unwrap();
        assert_eq!(f.period, Period::parse("2026-W41").unwrap());
        assert_eq!(f.ids.len(), 7);
        assert_eq!(f.ids[0], lines[0].id);
    }

    #[test]
    fn day_folds_gather_into_their_week_then_month() {
        let mut lines: Vec<_> = ["2026-10-05", "2026-10-06", "2026-10-07"]
            .iter()
            .map(|d| fold_line(d))
            .collect();
        lines.extend(notes("2026-11-20T00:00:00Z", WINDOW));
        let j = journal(lines.clone());
        let f = next_fold(&j, at("2026-12-01T00:00:00Z")).unwrap();
        assert_eq!(f.period, Period::parse("2026-W41").unwrap());
        assert_eq!(
            f.ids,
            lines[..3].iter().map(|l| l.id.clone()).collect::<Vec<_>>()
        );
        // Week folds of one month gather into the month.
        let mut lines: Vec<_> = ["2026-W41", "2026-W42"]
            .iter()
            .map(|w| fold_line(w))
            .collect();
        lines.extend(notes("2026-11-20T00:00:00Z", WINDOW));
        let f = next_fold(&journal(lines), at("2026-12-01T00:00:00Z")).unwrap();
        assert_eq!(f.period, Period::parse("2026-10").unwrap());
    }

    #[test]
    fn year_folds_never_fold_and_are_stepped_over() {
        let mut lines = vec![fold_line("2024"), fold_line("2025")];
        lines.extend(notes("2026-10-01T00:00:00Z", WINDOW));
        let j = journal(lines);
        let f = next_fold(&j, at("2026-10-03T12:00:00Z")).unwrap();
        assert_eq!(f.period, Period::parse("2026-10-01").unwrap());
    }

    #[test]
    fn a_fold_takes_only_lines_below_its_level() {
        let mut lines = vec![fold_line("2026-10-01")];
        lines.extend(notes("2026-10-01T00:00:00Z", 2));
        lines.extend(notes("2026-11-01T00:00:00Z", WINDOW));
        let j = journal(lines.clone());
        let f = next_fold(&j, at("2026-12-01T00:00:00Z")).unwrap();
        // The day fold and two notes of that day: the day itself cannot fold
        // its own fold, so the week takes all three.
        assert_eq!(f.period, Period::parse("2026-W40").unwrap());
        assert_eq!(f.ids.len(), 3);
    }
}
