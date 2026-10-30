//! The stretches of the calendar a window folds by (§4): a day, an ISO week,
//! a month, a year — each named the way its archive file is,
//! `2026-10-30`, `2026-W44`, `2026-10`, `2026`.
//!
//! They nest: a day is in one week, a week in the month of its Thursday (the
//! day that also decides its ISO year), a month in its year. So every period
//! below a year has exactly one parent, and a fold at one level always
//! gathers whole folds of the level below.

use std::fmt;

use journal_core::period::{civil_from_days, days_from_civil};

use crate::time::{days_in_month, Stamp};

/// How long a stretch a line stands for. A note or an episode is a `Step`;
/// a fold stands for its period.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Level {
    Step,
    Day,
    Week,
    Month,
    Year,
}

impl Level {
    /// The level above; `None` for a year, which never folds.
    pub fn up(self) -> Option<Level> {
        match self {
            Level::Step => Some(Level::Day),
            Level::Day => Some(Level::Week),
            Level::Week => Some(Level::Month),
            Level::Month => Some(Level::Year),
            Level::Year => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Period {
    Day {
        year: i64,
        month: u32,
        day: u32,
    },
    /// An ISO week: `year` is its ISO year, the year of its Thursday.
    Week {
        year: i64,
        week: u32,
    },
    Month {
        year: i64,
        month: u32,
    },
    Year {
        year: i64,
    },
}

/// Monday = 0 … Sunday = 6, for the day `days` after 1970-01-01 (a Thursday).
fn weekday(days: i64) -> i64 {
    (days + 3).rem_euclid(7)
}

/// The first day (a Monday) of ISO week 1 of `year`: the week holding its
/// first Thursday.
fn iso_week_one(year: i64) -> i64 {
    let jan4 = days_from_civil(year, 1, 4);
    jan4 - weekday(jan4)
}

impl Period {
    pub fn level(self) -> Level {
        match self {
            Period::Day { .. } => Level::Day,
            Period::Week { .. } => Level::Week,
            Period::Month { .. } => Level::Month,
            Period::Year { .. } => Level::Year,
        }
    }

    /// The day `at` falls in.
    pub fn day_of(at: Stamp) -> Period {
        Self::of_days(at.days())
    }

    fn of_days(days: i64) -> Period {
        let (year, month, day) = civil_from_days(days);
        Period::Day { year, month, day }
    }

    /// The period at `level` that `at` falls in; `None` for `Step`.
    pub fn of(level: Level, at: Stamp) -> Option<Period> {
        let mut p = Self::day_of(at);
        while p.level() < level {
            p = p.parent()?;
        }
        (level != Level::Step).then_some(p)
    }

    /// The first instant in it.
    pub fn start(self) -> Stamp {
        Stamp::of_days(match self {
            Period::Day { year, month, day } => days_from_civil(year, month, day),
            Period::Week { year, week } => iso_week_one(year) + 7 * (i64::from(week) - 1),
            Period::Month { year, month } => days_from_civil(year, month, 1),
            Period::Year { year } => days_from_civil(year, 1, 1),
        })
    }

    /// The first instant after it.
    pub fn end(self) -> Stamp {
        Stamp::of_days(match self {
            Period::Day { year, month, day } => days_from_civil(year, month, day) + 1,
            Period::Week { year, week } => iso_week_one(year) + 7 * i64::from(week),
            Period::Month { year, month: 12 } => days_from_civil(year + 1, 1, 1),
            Period::Month { year, month } => days_from_civil(year, month + 1, 1),
            Period::Year { year } => days_from_civil(year + 1, 1, 1),
        })
    }

    /// The first instant after everything it holds. A day and a week hold
    /// only themselves; a month holds the weeks whose Thursday is in it, so
    /// its last week can run days past its calendar end — 2026-10 holds W44,
    /// which ends on 2 November — and a year holds its December's.
    pub fn closes(self) -> Stamp {
        match self {
            Period::Day { .. } | Period::Week { .. } => self.end(),
            Period::Month { .. } => {
                let last_day = Period::day_of(self.end().plus(-1));
                match last_day.parent() {
                    Some(week) if week.parent() == Some(self) => week.end().max(self.end()),
                    _ => self.end(),
                }
            }
            Period::Year { year } => Period::Month { year, month: 12 }.closes(),
        }
    }

    /// Over by `now`: nothing more can fall in it.
    pub fn is_closed(self, now: Stamp) -> bool {
        self.closes() <= now
    }

    /// The period one level up that holds it; `None` for a year.
    pub fn parent(self) -> Option<Period> {
        match self {
            Period::Day { year, month, day } => {
                let days = days_from_civil(year, month, day);
                let thursday = days - weekday(days) + 3;
                let (iso_year, _, _) = civil_from_days(thursday);
                let week = (thursday - iso_week_one(iso_year)) / 7 + 1;
                Some(Period::Week {
                    year: iso_year,
                    week: week as u32,
                })
            }
            Period::Week { year, week } => {
                let thursday = iso_week_one(year) + 7 * (i64::from(week) - 1) + 3;
                let (y, m, _) = civil_from_days(thursday);
                Some(Period::Month { year: y, month: m })
            }
            Period::Month { year, .. } => Some(Period::Year { year }),
            Period::Year { .. } => None,
        }
    }

    /// `other` is this period or lies inside it.
    pub fn holds(self, other: Period) -> bool {
        let mut p = Some(other);
        while let Some(q) = p {
            if q == self {
                return true;
            }
            if q.level() >= self.level() {
                return false;
            }
            p = q.parent();
        }
        false
    }

    /// Weeks in ISO year `year`: 52 or 53.
    fn weeks_in(year: i64) -> u32 {
        ((iso_week_one(year + 1) - iso_week_one(year)) / 7) as u32
    }

    /// `2026-10-30`, `2026-W44`, `2026-10` or `2026`, exactly.
    pub fn parse(s: &str) -> Result<Period, String> {
        let bad = || format!("`{s}` is not a period (2026-10-30, 2026-W44, 2026-10 or 2026)");
        let digits = |t: &str, n: usize| -> Option<u32> {
            (t.len() == n && t.bytes().all(|c| c.is_ascii_digit()))
                .then(|| t.parse().ok())
                .flatten()
        };
        let parts: Vec<&str> = s.split('-').collect();
        let year = i64::from(digits(parts[0], 4).ok_or_else(bad)?);
        let period = match parts[1..] {
            [] => Period::Year { year },
            [w] if w.starts_with('W') => {
                let week = digits(&w[1..], 2).ok_or_else(bad)?;
                if week < 1 || week > Self::weeks_in(year) {
                    return Err(bad());
                }
                Period::Week { year, week }
            }
            [m] => {
                let month = digits(m, 2).ok_or_else(bad)?;
                if !(1..=12).contains(&month) {
                    return Err(bad());
                }
                Period::Month { year, month }
            }
            [m, d] => {
                let month = digits(m, 2).ok_or_else(bad)?;
                let day = digits(d, 2).ok_or_else(bad)?;
                if !(1..=12).contains(&month) || day < 1 || day > days_in_month(year, month) {
                    return Err(bad());
                }
                Period::Day { year, month, day }
            }
            _ => return Err(bad()),
        };
        Ok(period)
    }
}

impl fmt::Display for Period {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Period::Day { year, month, day } => write!(f, "{year:04}-{month:02}-{day:02}"),
            Period::Week { year, week } => write!(f, "{year:04}-W{week:02}"),
            Period::Month { year, month } => write!(f, "{year:04}-{month:02}"),
            Period::Year { year } => write!(f, "{year:04}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(s: &str) -> Stamp {
        Stamp::parse(s).unwrap()
    }

    fn p(s: &str) -> Period {
        Period::parse(s).unwrap()
    }

    #[test]
    fn every_period_is_written_as_its_archive_is_named() {
        for s in [
            "2026-10-30",
            "2026-W44",
            "2026-W01",
            "2026-10",
            "2026",
            "0000",
        ] {
            assert_eq!(p(s).to_string(), s);
        }
        assert_eq!(p("2026-10-30").level(), Level::Day);
        assert_eq!(p("2026-W44").level(), Level::Week);
        assert_eq!(p("2026-10").level(), Level::Month);
        assert_eq!(p("2026").level(), Level::Year);
    }

    #[test]
    fn malformed_periods_are_refused() {
        for bad in [
            "",
            "26",
            "2026-1",
            "2026-13",
            "2026-00",
            "2026-10-32",
            "2026-02-29",
            "2026-W00",
            "2026-W54",
            "2026-w44",
            "2026-W4",
            "2026-10-30-1",
            "2026--10",
            "abcd",
            "2026-1a",
            // 2026 has 53 ISO weeks; 2025 has 52.
            "2025-W53",
        ] {
            assert!(Period::parse(bad).is_err(), "{bad:?}");
        }
        assert!(Period::parse("2026-W53").is_ok());
        assert!(Period::parse("2024-02-29").is_ok());
    }

    #[test]
    fn a_day_is_in_its_iso_week_even_across_a_new_year() {
        // 2026-10-30 is a Friday in week 44.
        assert_eq!(p("2026-10-30").parent(), Some(p("2026-W44")));
        // 2027-01-01 is a Friday: it belongs to 2026's week 53.
        assert_eq!(p("2027-01-01").parent(), Some(p("2026-W53")));
        // 2024-12-30 is a Monday: week 1 of 2025.
        assert_eq!(p("2024-12-30").parent(), Some(p("2025-W01")));
        assert_eq!(p("2025-W01").start(), at("2024-12-30T00:00:00Z"));
        assert_eq!(p("2025-W01").end(), at("2025-01-06T00:00:00Z"));
    }

    #[test]
    fn a_week_is_in_the_month_of_its_thursday() {
        // Week 44 of 2026 runs Mon 26 Oct to Sun 1 Nov; its Thursday is 29 Oct.
        assert_eq!(p("2026-W44").parent(), Some(p("2026-10")));
        // Week 53 of 2026: Thursday 31 Dec 2026.
        assert_eq!(p("2026-W53").parent(), Some(p("2026-12")));
        // Week 1 of 2025: Thursday 2 Jan 2025, though it starts in 2024.
        assert_eq!(p("2025-W01").parent(), Some(p("2025-01")));
        // So a day can sit in a month other than its own by way of its week.
        assert!(p("2026-11").holds(p("2026-W45")));
        assert!(p("2026-10").holds(p("2026-11-01")));
        assert!(!p("2026-11").holds(p("2026-11-01")));
    }

    #[test]
    fn holding_walks_the_chain_and_nothing_holds_a_bigger_period() {
        assert!(p("2026").holds(p("2026-10-30")));
        assert!(p("2026").holds(p("2026")));
        assert!(p("2026-W44").holds(p("2026-10-26")));
        assert!(!p("2026-W44").holds(p("2026-10-25")));
        assert!(!p("2026-10-30").holds(p("2026-10")));
        assert!(!p("2026-10").holds(p("2026")));
        assert!(!p("2025").holds(p("2026-10-30")));
        assert_eq!(p("2026").parent(), None);
    }

    #[test]
    fn periods_start_and_end_at_midnight_and_close_at_their_end() {
        assert_eq!(p("2026-10").start(), at("2026-10-01T00:00:00Z"));
        assert_eq!(p("2026-12").end(), at("2027-01-01T00:00:00Z"));
        assert_eq!(p("2024").end(), at("2025-01-01T00:00:00Z"));
        let day = p("2026-10-30");
        assert!(!day.is_closed(at("2026-10-30T23:59:59Z")));
        assert!(day.is_closed(at("2026-10-31T00:00:00Z")));
    }

    #[test]
    fn a_month_or_year_closes_when_its_last_week_does() {
        // W44 (26 Oct – 1 Nov) is October's: October is open on 1 November.
        assert_eq!(p("2026-10").closes(), at("2026-11-02T00:00:00Z"));
        assert!(!p("2026-10").is_closed(at("2026-11-01T05:00:00Z")));
        assert!(p("2026-10").is_closed(at("2026-11-02T00:00:00Z")));
        // 30 Nov 2026 is a Monday whose week is December's: November closes on time.
        assert_eq!(p("2026-11").closes(), at("2026-12-01T00:00:00Z"));
        // W53 of 2026 runs to 3 Jan 2027 and is December's, so 2026's.
        assert_eq!(p("2026").closes(), at("2027-01-04T00:00:00Z"));
        assert_eq!(
            p("2025").closes(),
            at("2026-01-01T00:00:00Z").max(p("2025-12").closes())
        );
        // Whatever a period holds closes no later than it does.
        let first = p("2020-01-01").start().days();
        for d in first..first + 2200 {
            let day = Period::day_of(Stamp::of_days(d));
            let mut q = Some(day);
            while let Some(x) = q {
                assert!(day.closes() <= x.closes(), "{day} in {x}");
                q = x.parent();
            }
        }
    }

    #[test]
    fn of_finds_the_period_at_each_level() {
        let t = at("2026-10-30T23:59:59Z");
        assert_eq!(Period::of(Level::Step, t), None);
        assert_eq!(Period::of(Level::Day, t), Some(p("2026-10-30")));
        assert_eq!(Period::of(Level::Week, t), Some(p("2026-W44")));
        assert_eq!(Period::of(Level::Month, t), Some(p("2026-10")));
        assert_eq!(Period::of(Level::Year, t), Some(p("2026")));
        assert_eq!(Level::Year.up(), None);
        assert_eq!(Level::Step.up(), Some(Level::Day));
    }

    #[test]
    fn every_day_of_a_decade_nests_consistently() {
        let first = p("2020-01-01").start().days();
        for d in first..first + 3660 {
            let day = Period::day_of(Stamp::of_days(d));
            let week = day.parent().unwrap();
            let month = week.parent().unwrap();
            assert!(
                week.start() <= day.start() && day.end() <= week.end(),
                "{day}"
            );
            assert_eq!(Period::parse(&week.to_string()), Ok(week));
            assert!(month.holds(day) && month.parent().unwrap().holds(day));
            assert_eq!(week.end().secs() - week.start().secs(), 7 * 86_400);
        }
    }
}
