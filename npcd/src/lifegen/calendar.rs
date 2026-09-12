//! The date arithmetic a life needs, and nothing else.
//!
//! # Why this is not a dependency
//!
//! Four operations: is this date real, how many days are in this month, walk
//! the years between two dates, walk the months inside a year. A calendar crate
//! brings time zones, leap seconds, parsing of a dozen formats and a `Duration`
//! type, none of which a character's life has any use for — and every one of
//! which is a decision somebody would eventually have to have an opinion about.
//!
//! # Proleptic Gregorian, and that is a world-building decision
//!
//! [`crate::engine::life`] names its documents `YYYY-MM-DD`, so the calendar a
//! character remembers by is already fixed at twelve months with Gregorian leap
//! years. That is a constraint on the settings this engine can host, and it is
//! better stated here than discovered by an author whose world has ten months.

use std::fmt;

/// A calendar date, at day precision.
///
/// Held as three integers rather than a day count: every consumer wants the
/// parts back — a document name, a month to fan out over, a year to label a
/// stratum — and a day count would mean converting on the way in and out for
/// the sake of arithmetic this module barely does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Date {
    pub year: i32,
    pub month: u32,
    pub day: u32,
}

/// Why a date string is not a date.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BadDate {
    /// Not `YYYY-MM-DD` at all.
    Shape(String),
    /// Shaped correctly, but names a day the calendar does not have — a 31st of
    /// February, a 29th of February in a common year. Carries the whole string
    /// because "day out of range" without the date is not actionable.
    NoSuchDay(String),
}

impl BadDate {
    pub fn message(&self) -> String {
        match self {
            BadDate::Shape(s) => format!("`{s}` is not a date — expected YYYY-MM-DD"),
            BadDate::NoSuchDay(s) => format!("`{s}` is not a day the calendar has"),
        }
    }
}

impl fmt::Display for Date {
    /// The wire and filename form. Zero-padded, always — the padding is what
    /// makes lexical order chronological, which is the property every other
    /// module here is built on.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }
}

impl Date {
    /// Parse `YYYY-MM-DD`, rejecting both bad shapes and impossible days.
    ///
    /// Nothing here is lenient. A life's dates end up in filenames that sort,
    /// and a date this function waved through would sort into the wrong place
    /// for the life of the character.
    pub fn parse(s: &str) -> Result<Date, BadDate> {
        let b = s.as_bytes();
        let shaped = b.len() == 10
            && b.iter().enumerate().all(|(i, c)| {
                if i == 4 || i == 7 {
                    *c == b'-'
                } else {
                    c.is_ascii_digit()
                }
            });
        if !shaped {
            return Err(BadDate::Shape(s.to_string()));
        }
        let num = |from: usize, to: usize| s[from..to].parse::<i32>().unwrap_or(0);
        let (year, month, day) = (num(0, 4), num(5, 7) as u32, num(8, 10) as u32);
        if !(1..=12).contains(&month) || day == 0 || day > days_in_month(year, month) {
            return Err(BadDate::NoSuchDay(s.to_string()));
        }
        Ok(Date { year, month, day })
    }

    /// `YYYY` — the year stratum this date falls in.
    pub fn year_key(&self) -> String {
        format!("{:04}", self.year)
    }

    /// `YYYY-MM` — the month stratum this date falls in.
    pub fn month_key(&self) -> String {
        format!("{:04}-{:02}", self.year, self.month)
    }
}

impl Date {
    /// Is `self` the day immediately before `other`?
    ///
    /// What "these two spans touch with nothing between them" means. Spans are inclusive at
    /// both ends, so abutting spans end and begin on consecutive days — and the test has to
    /// cross month and year boundaries, which is why it is arithmetic here rather than a
    /// subtraction at the call site.
    pub fn is_day_before(self, other: Date) -> bool {
        let next = if self.day < days_in_month(self.year, self.month) {
            Date {
                day: self.day + 1,
                ..self
            }
        } else if self.month < 12 {
            Date {
                year: self.year,
                month: self.month + 1,
                day: 1,
            }
        } else {
            Date {
                year: self.year + 1,
                month: 1,
                day: 1,
            }
        };
        next == other
    }
}

/// Gregorian leap year: every fourth, except centuries, except every fourth
/// century.
pub fn is_leap(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// Days in a month. `0` for a month outside 1..=12, which callers reject before
/// they get here — no month is silently treated as having 31 days.
pub fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap(year) => 29,
        2 => 28,
        _ => 0,
    }
}

/// Every year from `born` to `through` inclusive.
///
/// Inclusive at both ends because both are real years of the life: the year
/// they were born in is a year they lived, and so is the one the authored life
/// stops in — usually a partial year, which the generator is told about rather
/// than left to infer from a stratum that mysteriously ends in March.
pub fn years(born: Date, through: Date) -> Vec<i32> {
    if through < born {
        return Vec::new();
    }
    (born.year..=through.year).collect()
}

/// The months of `year` that fall inside the life, as `1..=12`.
///
/// Clipped at both ends: the birth year starts at the birth month, and the
/// final year stops at the month the authored life reaches. A character has no
/// memory of the months before they existed, and a generator handed all twelve
/// would write some.
pub fn months_in_year(year: i32, born: Date, through: Date) -> Vec<u32> {
    if year < born.year || year > through.year {
        return Vec::new();
    }
    let first = if year == born.year { born.month } else { 1 };
    let last = if year == through.year {
        through.month
    } else {
        12
    };
    if last < first {
        return Vec::new();
    }
    (first..=last).collect()
}

/// Whole years elapsed from `born` to `on` — the character's age, counted the
/// way a person counts it: the birthday has to have happened.
pub fn age_on(born: Date, on: Date) -> i32 {
    let mut age = on.year - born.year;
    if (on.month, on.day) < (born.month, born.day) {
        age -= 1;
    }
    age
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(s: &str) -> Date {
        Date::parse(s).unwrap()
    }

    #[test]
    fn a_date_parses_into_its_parts_and_renders_back() {
        let x = d("1998-09-14");
        assert_eq!((x.year, x.month, x.day), (1998, 9, 14));
        assert_eq!(x.to_string(), "1998-09-14");
    }

    /// **Padding is not cosmetic.** Every filename this produces is sorted
    /// lexically, and an unpadded month would file September after October.
    #[test]
    fn rendering_always_pads() {
        let x = Date {
            year: 7,
            month: 1,
            day: 2,
        };
        assert_eq!(x.to_string(), "0007-01-02");
        assert_eq!(d("1998-01-02").month_key(), "1998-01");
        assert_eq!(d("1998-01-02").year_key(), "1998");
    }

    #[test]
    fn a_malformed_date_is_refused_by_shape() {
        for s in ["1998-9-14", "98-09-14", "19980914", "1998-09-14 ", "", "x"] {
            assert!(
                matches!(Date::parse(s), Err(BadDate::Shape(_))),
                "{s:?} accepted"
            );
        }
    }

    /// **A shaped date that names no real day is the dangerous one.** It looks
    /// right in a filename and sorts into a position for a day that does not
    /// exist.
    #[test]
    fn a_day_the_calendar_does_not_have_is_refused() {
        for s in [
            "2001-02-29",
            "2001-04-31",
            "2001-13-01",
            "2001-00-01",
            "2001-01-00",
            "2001-01-32",
        ] {
            assert!(
                matches!(Date::parse(s), Err(BadDate::NoSuchDay(_))),
                "{s} accepted"
            );
        }
        assert!(Date::parse("2000-02-29").is_ok(), "2000 is a leap year");
    }

    #[test]
    fn leap_years_follow_the_gregorian_rule() {
        assert!(is_leap(2000), "divisible by 400");
        assert!(!is_leap(1900), "century, not divisible by 400");
        assert!(is_leap(2024));
        assert!(!is_leap(2023));
        assert_eq!(days_in_month(2024, 2), 29);
        assert_eq!(days_in_month(2023, 2), 28);
    }

    /// A month outside the calendar has no days at all — it must not read as a
    /// 31-day month and let an impossible date through.
    #[test]
    fn a_month_outside_the_calendar_has_no_days() {
        assert_eq!(days_in_month(2024, 0), 0);
        assert_eq!(days_in_month(2024, 13), 0);
    }

    #[test]
    fn the_years_of_a_life_include_both_ends() {
        assert_eq!(
            years(d("1998-09-14"), d("2001-03-02")),
            vec![1998, 1999, 2000, 2001]
        );
        assert_eq!(years(d("1998-09-14"), d("1998-12-31")), vec![1998]);
    }

    /// A life that ends before it starts has no years, rather than a reversed
    /// range that would panic or silently produce nothing surprising later.
    #[test]
    fn a_life_that_ends_before_it_begins_has_no_years() {
        assert!(years(d("2001-01-01"), d("1998-01-01")).is_empty());
    }

    /// **The clip at both ends is the point.** A character has no memory of the
    /// months before they were born, and a generator handed all twelve of their
    /// birth year would write some.
    #[test]
    fn the_first_and_last_years_are_clipped_to_the_life() {
        let (born, through) = (d("1998-09-14"), d("2001-03-02"));
        assert_eq!(months_in_year(1998, born, through), vec![9, 10, 11, 12]);
        assert_eq!(
            months_in_year(1999, born, through),
            (1..=12).collect::<Vec<_>>()
        );
        assert_eq!(months_in_year(2001, born, through), vec![1, 2, 3]);
        assert!(months_in_year(1997, born, through).is_empty());
        assert!(months_in_year(2002, born, through).is_empty());
    }

    /// A life contained in one year is clipped at both ends by the same year.
    #[test]
    fn a_life_inside_one_year_is_clipped_at_both_ends() {
        assert_eq!(
            months_in_year(1998, d("1998-04-01"), d("1998-07-31")),
            vec![4, 5, 6, 7]
        );
        // And one that ends before it starts within a year yields nothing
        // rather than a reversed range.
        assert!(months_in_year(1998, d("1998-07-01"), d("1998-04-01")).is_empty());
    }

    /// Age counts birthdays that have happened, not year subtractions.
    #[test]
    fn age_waits_for_the_birthday() {
        let born = d("1998-09-14");
        assert_eq!(age_on(born, d("2016-09-13")), 17);
        assert_eq!(age_on(born, d("2016-09-14")), 18);
        assert_eq!(age_on(born, d("2016-09-15")), 18);
        assert_eq!(age_on(born, d("1998-09-14")), 0);
    }
}
