//! The stretch of the calendar an entry stands for, and when a stretch of it
//! is over.
//!
//! A journal compresses itself as its calendar turns. An entry written from
//! turns is a [`Period::Step`]. Once the day it ended on is over, that day's
//! steps fold into one [`Period::Day`] entry; once the week is over, its days
//! fold into a [`Period::Week`]; then a month's weeks into a [`Period::Month`],
//! and a year's months into a [`Period::Year`]. Each fold is lossy on purpose:
//! a one-line summary in the journal and a detailed digest behind it.
//!
//! Whose calendar it is belongs to the product ([`crate::style::Style::bucket`]):
//! a coding session folds on UTC days and weeks, a character on its world's.

use serde::{Deserialize, Serialize};

/// What an entry stands for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Period {
    /// Written from a stretch of turns.
    #[default]
    Step,
    Day,
    Week,
    Month,
    Year,
}

impl Period {
    /// The period entries of this one fold into once it is over; `None` for a
    /// year, which never folds.
    pub fn folds_into(self) -> Option<Period> {
        match self {
            Period::Step => Some(Period::Day),
            Period::Day => Some(Period::Week),
            Period::Week => Some(Period::Month),
            Period::Month => Some(Period::Year),
            Period::Year => None,
        }
    }

    /// The name it is stored and shown under.
    pub fn name(self) -> &'static str {
        match self {
            Period::Step => "step",
            Period::Day => "day",
            Period::Week => "week",
            Period::Month => "month",
            Period::Year => "year",
        }
    }
}

/// Days from 1970-01-01 to the civil date `(year, month, day)` — Howard
/// Hinnant's `days_from_civil`, exact over the proleptic Gregorian calendar.
pub fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let m = i64::from(month);
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + i64::from(day) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The civil date `days` after 1970-01-01 — Hinnant's `civil_from_days`.
pub fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

/// The UTC calendar: which `period` the instant `at_ms` falls in, as a number
/// that grows with time. Weeks start on Monday.
pub fn utc_bucket(period: Period, at_ms: u64) -> u64 {
    const MS_PER_DAY: u64 = 86_400_000;
    let day = at_ms / MS_PER_DAY;
    match period {
        // A step is its own; nothing folds by it.
        Period::Step => at_ms,
        Period::Day => day,
        // 1970-01-01 was a Thursday: three days on, every week starts Monday.
        Period::Week => (day + 3) / 7,
        Period::Month => {
            let (y, m, _) = civil_from_days(day as i64);
            (y as u64) * 12 + u64::from(m - 1)
        }
        Period::Year => civil_from_days(day as i64).0 as u64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS_PER_DAY: u64 = 86_400_000;

    fn at(year: i64, month: u32, day: u32) -> u64 {
        days_from_civil(year, month, day) as u64 * MS_PER_DAY
    }

    #[test]
    fn civil_dates_round_trip_across_leap_years_and_centuries() {
        for days in [-719_468, -1, 0, 59, 60, 11_016, 20_384, 20_750, 47_482] {
            let (y, m, d) = civil_from_days(days);
            assert_eq!(days_from_civil(y, m, d), days, "{y}-{m}-{d}");
        }
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(days_from_civil(2024, 2, 29)), (2024, 2, 29));
        assert_eq!(days_from_civil(2026, 10, 23), 20_749);
    }

    #[test]
    fn a_utc_day_is_midnight_to_midnight() {
        let d = at(2026, 10, 23);
        assert_eq!(
            utc_bucket(Period::Day, d),
            utc_bucket(Period::Day, d + MS_PER_DAY - 1)
        );
        assert_ne!(
            utc_bucket(Period::Day, d),
            utc_bucket(Period::Day, d + MS_PER_DAY)
        );
    }

    #[test]
    fn a_week_runs_monday_to_sunday() {
        // 2026-10-19 is a Monday, 2026-10-25 a Sunday.
        let monday = utc_bucket(Period::Week, at(2026, 10, 19));
        assert_eq!(
            utc_bucket(Period::Week, at(2026, 10, 25) + MS_PER_DAY - 1),
            monday
        );
        assert_eq!(utc_bucket(Period::Week, at(2026, 10, 18)), monday - 1);
        assert_eq!(utc_bucket(Period::Week, at(2026, 10, 26)), monday + 1);
    }

    #[test]
    fn months_and_years_turn_on_the_first() {
        let oct = utc_bucket(Period::Month, at(2026, 10, 31));
        assert_eq!(utc_bucket(Period::Month, at(2026, 11, 1)), oct + 1);
        assert_eq!(utc_bucket(Period::Month, at(2027, 1, 1)), oct + 3);
        assert_eq!(utc_bucket(Period::Year, at(2026, 12, 31)), 2026);
        assert_eq!(utc_bucket(Period::Year, at(2027, 1, 1)), 2027);
    }

    #[test]
    fn each_period_folds_into_the_next_and_a_year_into_none() {
        let chain: Vec<Option<Period>> = [
            Period::Step,
            Period::Day,
            Period::Week,
            Period::Month,
            Period::Year,
        ]
        .map(Period::folds_into)
        .into();
        assert_eq!(
            chain,
            [
                Some(Period::Day),
                Some(Period::Week),
                Some(Period::Month),
                Some(Period::Year),
                None
            ]
        );
    }
}
