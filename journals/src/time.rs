//! An instant, to the second, in UTC — the `at` of a line and the `built` of
//! a summary, written `2026-10-30T09:10:02Z` and nothing else.

use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

use journal_core::period::{civil_from_days, days_from_civil};

const DAY: i64 = 86_400;

/// Seconds since 1970-01-01T00:00:00Z.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Stamp(i64);

/// A stamp's calendar fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Civil {
    pub year: i64,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
}

impl Stamp {
    /// The first instant a stamp may hold: year 0000. Nothing earlier can be
    /// written in four digits.
    pub const MIN: Stamp = Stamp(-62_167_219_200);
    /// The last: 9999-12-31T23:59:59Z.
    pub const MAX: Stamp = Stamp(253_402_300_799);

    /// `secs` since the epoch, kept within [`Self::MIN`]..=[`Self::MAX`].
    pub fn from_secs(secs: i64) -> Stamp {
        Stamp(secs.clamp(Self::MIN.0, Self::MAX.0))
    }

    /// `ms` since the epoch, to the second below.
    pub fn from_ms(ms: u64) -> Stamp {
        Self::from_secs(i64::try_from(ms / 1000).unwrap_or(i64::MAX))
    }

    /// Now, by this machine's clock.
    pub fn now() -> Stamp {
        let ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        Self::from_ms(ms)
    }

    pub fn secs(self) -> i64 {
        self.0
    }

    /// Days since 1970-01-01, the day this stamp falls in.
    pub fn days(self) -> i64 {
        self.0.div_euclid(DAY)
    }

    /// The stamp at midnight starting the day `days` after 1970-01-01.
    pub fn of_days(days: i64) -> Stamp {
        Self::from_secs(days.saturating_mul(DAY))
    }

    pub fn civil(self) -> Civil {
        let (year, month, day) = civil_from_days(self.days());
        let rest = self.0.rem_euclid(DAY) as u32;
        Civil {
            year,
            month,
            day,
            hour: rest / 3600,
            minute: rest / 60 % 60,
            second: rest % 60,
        }
    }

    /// The stamp of a calendar instant; `None` for a date that does not exist
    /// or a year outside 0000..=9999.
    pub fn of_civil(c: Civil) -> Option<Stamp> {
        if !(0..=9999).contains(&c.year)
            || !(1..=12).contains(&c.month)
            || c.day < 1
            || c.day > days_in_month(c.year, c.month)
            || c.hour > 23
            || c.minute > 59
            || c.second > 59
        {
            return None;
        }
        let days = days_from_civil(c.year, c.month, c.day);
        Some(Stamp(
            days * DAY + i64::from(c.hour * 3600 + c.minute * 60 + c.second),
        ))
    }

    /// Exactly `YYYY-MM-DDTHH:MM:SSZ`.
    pub fn parse(s: &str) -> Result<Stamp, String> {
        let bad = || format!("`{s}` is not a time written 2026-10-30T09:10:02Z");
        let b = s.as_bytes();
        if b.len() != 20
            || b[4] != b'-'
            || b[7] != b'-'
            || b[10] != b'T'
            || b[13] != b':'
            || b[16] != b':'
            || b[19] != b'Z'
        {
            return Err(bad());
        }
        let num = |r: std::ops::Range<usize>| -> Result<u32, String> {
            let part = &s[r];
            if part.bytes().all(|c| c.is_ascii_digit()) {
                part.parse().map_err(|_| bad())
            } else {
                Err(bad())
            }
        };
        Stamp::of_civil(Civil {
            year: i64::from(num(0..4)?),
            month: num(5..7)?,
            day: num(8..10)?,
            hour: num(11..13)?,
            minute: num(14..16)?,
            second: num(17..19)?,
        })
        .ok_or_else(bad)
    }

    /// `yyyymmdd-hhmm`: the minute, as a line's id starts.
    pub fn compact_minute(self) -> String {
        let c = self.civil();
        format!(
            "{:04}{:02}{:02}-{:02}{:02}",
            c.year, c.month, c.day, c.hour, c.minute
        )
    }

    /// `secs` later, kept within range.
    pub fn plus(self, secs: i64) -> Stamp {
        Self::from_secs(self.0.saturating_add(secs))
    }
}

/// Days in `month` of `year`, proleptic Gregorian.
pub fn days_in_month(year: i64, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 => 29,
        2 => 28,
        _ => 0,
    }
}

impl fmt::Display for Stamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let c = self.civil();
        write!(
            f,
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
            c.year, c.month, c.day, c.hour, c.minute, c.second
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stamp_is_written_one_way_and_read_back() {
        let s = Stamp::parse("2026-10-30T09:10:02Z").unwrap();
        assert_eq!(s.secs(), 1_793_351_402);
        assert_eq!(s.to_string(), "2026-10-30T09:10:02Z");
        assert_eq!(s.compact_minute(), "20261030-0910");
        assert_eq!(Stamp::from_secs(0).to_string(), "1970-01-01T00:00:00Z");
        assert_eq!(Stamp::from_ms(1_999), Stamp::from_secs(1));
    }

    #[test]
    fn the_range_is_four_digit_years() {
        assert_eq!(Stamp::MIN.to_string(), "0000-01-01T00:00:00Z");
        assert_eq!(Stamp::MAX.to_string(), "9999-12-31T23:59:59Z");
        assert_eq!(Stamp::from_secs(i64::MAX), Stamp::MAX);
        assert_eq!(Stamp::from_secs(i64::MIN), Stamp::MIN);
        assert_eq!(Stamp::MAX.plus(10), Stamp::MAX);
        assert_eq!(Stamp::parse("0000-01-01T00:00:00Z"), Ok(Stamp::MIN));
        assert_eq!(Stamp::parse("9999-12-31T23:59:59Z"), Ok(Stamp::MAX));
    }

    #[test]
    fn leap_days_exist_only_in_leap_years() {
        assert!(Stamp::parse("2024-02-29T00:00:00Z").is_ok());
        assert!(Stamp::parse("2000-02-29T00:00:00Z").is_ok());
        assert!(Stamp::parse("1900-02-29T00:00:00Z").is_err());
        assert!(Stamp::parse("2026-02-29T00:00:00Z").is_err());
        assert_eq!(
            Stamp::parse("2024-02-29T23:59:59Z")
                .unwrap()
                .plus(1)
                .to_string(),
            "2024-03-01T00:00:00Z"
        );
    }

    #[test]
    fn anything_else_is_refused() {
        for bad in [
            "",
            "2026-10-30",
            "2026-10-30T09:10:02",
            "2026-10-30T09:10:02.5Z",
            "2026-10-30 09:10:02Z",
            "2026-10-30T09:10:02+00:00",
            "2026-13-01T00:00:00Z",
            "2026-00-01T00:00:00Z",
            "2026-04-31T00:00:00Z",
            "2026-10-30T24:00:00Z",
            "2026-10-30T23:60:00Z",
            "2026-10-30T23:59:60Z",
            "+026-10-30T09:10:02Z",
            "2026-1a-30T09:10:02Z",
            "２026-10-30T09:10:02Z",
        ] {
            assert!(Stamp::parse(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn days_split_at_midnight_utc_before_and_after_the_epoch() {
        assert_eq!(Stamp::from_secs(-1).days(), -1);
        assert_eq!(Stamp::from_secs(-1).to_string(), "1969-12-31T23:59:59Z");
        assert_eq!(Stamp::from_secs(86_399).days(), 0);
        assert_eq!(Stamp::from_secs(86_400).days(), 1);
        assert_eq!(Stamp::of_days(1).to_string(), "1970-01-02T00:00:00Z");
    }
}
