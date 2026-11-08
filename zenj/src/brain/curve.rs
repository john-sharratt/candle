//! The night's energy curve (§5.1): target energy against the clock, from
//! the set's start to its end, through the fixed points the plan anchors on.
//! Between points the target moves in a straight line; before the first and
//! after the last it holds.

use chrono::{DateTime, Duration, Local, NaiveDate, NaiveTime, TimeZone};
use serde::Deserialize;

/// One point as the config writes it: `{at: "23:30", energy: 0.85}`. Times
/// run forward through the night: one earlier than the point before it is on
/// the next day.
#[derive(Debug, Clone, Deserialize)]
pub struct PointCfg {
    pub at: String,
    pub energy: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Curve {
    points: Vec<(DateTime<Local>, f32)>,
}

impl Curve {
    /// The curve `points` describe for the night starting on `night`.
    pub fn of(night: NaiveDate, points: &[PointCfg]) -> Result<Curve, String> {
        if points.is_empty() {
            return Err("an energy curve needs at least one point".into());
        }
        let mut day = night;
        let mut last: Option<NaiveTime> = None;
        let mut out = Vec::with_capacity(points.len());
        for p in points {
            let t = NaiveTime::parse_from_str(&p.at, "%H:%M")
                .map_err(|e| format!("`{}` is not a time HH:MM: {e}", p.at))?;
            if !(0.0..=1.0).contains(&p.energy) {
                return Err(format!("{}: an energy is 0 to 1, not {}", p.at, p.energy));
            }
            if last.is_some_and(|l| t <= l) {
                day = day.succ_opt().ok_or("the calendar ended")?;
            }
            last = Some(t);
            let at = Local
                .from_local_datetime(&day.and_time(t))
                .earliest()
                .ok_or_else(|| format!("{} does not exist on {day}", p.at))?;
            out.push((at, p.energy));
        }
        Ok(Curve { points: out })
    }

    /// The target energy at `at`.
    pub fn target(&self, at: DateTime<Local>) -> f32 {
        let first = self.points[0];
        if at <= first.0 {
            return first.1;
        }
        for w in self.points.windows(2) {
            let ((t0, e0), (t1, e1)) = (w[0], w[1]);
            if at <= t1 {
                let span = (t1 - t0).num_milliseconds().max(1) as f32;
                let into = (at - t0).num_milliseconds() as f32;
                return e0 + (e1 - e0) * (into / span);
            }
        }
        self.points[self.points.len() - 1].1
    }

    /// The curve with every target from `from` for `over` raised (or
    /// lowered) by `by`, held to `0..=1`: the conversation's "more energy".
    pub fn nudged(&self, from: DateTime<Local>, over: Duration, by: f32) -> Curve {
        let until = from + over;
        let edge = Duration::seconds(1);
        let mut points: Vec<(DateTime<Local>, f32)> = self.points.clone();
        // The window's ends, raised, and a second outside each, not: the
        // curve steps at the window rather than leaning into it.
        for t in [from - edge, from, until, until + edge] {
            points.push((t, self.target(t)));
        }
        points.sort_by_key(|(t, _)| *t);
        points.dedup_by_key(|(t, _)| *t);
        for (t, e) in &mut points {
            if *t >= from && *t <= until {
                *e = (*e + by).clamp(0.0, 1.0);
            }
        }
        Curve { points }
    }

    pub fn points(&self) -> &[(DateTime<Local>, f32)] {
        &self.points
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(points: &[(&str, f32)]) -> Vec<PointCfg> {
        points
            .iter()
            .map(|(at, energy)| PointCfg {
                at: at.to_string(),
                energy: *energy,
            })
            .collect()
    }

    fn night() -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 12, 31).unwrap()
    }

    fn at(day: u32, h: u32, m: u32) -> DateTime<Local> {
        let d = if day == 31 {
            night()
        } else {
            NaiveDate::from_ymd_opt(2027, 1, day).unwrap()
        };
        Local
            .from_local_datetime(&d.and_hms_opt(h, m, 0).unwrap())
            .earliest()
            .unwrap()
    }

    /// **The curve runs through the night**: a time earlier than the one
    /// before it is past midnight, the target is a straight line between
    /// points, and it holds outside them.
    #[test]
    fn the_curve_runs_through_midnight_in_straight_lines() {
        let c = Curve::of(
            night(),
            &cfg(&[("20:00", 0.4), ("23:00", 0.7), ("00:00", 1.0), ("02:00", 0.5)]),
        )
        .unwrap();
        assert_eq!(c.points()[2].0, at(1, 0, 0));
        assert_eq!(c.target(at(31, 19, 0)), 0.4);
        assert!((c.target(at(31, 21, 30)) - 0.55).abs() < 1e-5);
        assert!((c.target(at(31, 23, 30)) - 0.85).abs() < 1e-5);
        assert!((c.target(at(1, 1, 0)) - 0.75).abs() < 1e-5);
        assert_eq!(c.target(at(1, 5, 0)), 0.5);
    }

    #[test]
    fn a_nudge_lifts_the_window_and_is_held_in_range() {
        let c = Curve::of(night(), &cfg(&[("20:00", 0.4), ("00:00", 0.9)])).unwrap();
        let n = c.nudged(at(31, 22, 0), Duration::minutes(30), 0.2);
        assert!((n.target(at(31, 22, 15)) - (0.65 + 0.03125 + 0.2)).abs() < 1e-4);
        assert!((n.target(at(31, 21, 0)) - 0.525).abs() < 1e-4, "before the window");
        let max = c.nudged(at(31, 23, 50), Duration::minutes(30), 0.5);
        assert_eq!(max.target(at(31, 23, 55)), 1.0);
    }

    #[test]
    fn a_bad_curve_is_refused() {
        assert!(Curve::of(night(), &[]).is_err());
        assert_eq!(
            Curve::of(night(), &cfg(&[("25:00", 0.5)])).unwrap_err(),
            "`25:00` is not a time HH:MM: input is out of range"
        );
        assert_eq!(
            Curve::of(night(), &cfg(&[("20:00", 1.5)])).unwrap_err(),
            "20:00: an energy is 0 to 1, not 1.5"
        );
    }
}
