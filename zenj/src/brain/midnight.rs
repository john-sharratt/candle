//! The midnight run-in (§5.1): the midnight track's drop lands on 00:00:00.
//! If the drop is beat *n* of the midnight track, it must start *n* beats
//! before midnight at its tempo; the track before must reach its out point
//! then; and so on back through the run-in. From the lock time only a person
//! changes it.

use chrono::{DateTime, Duration, Local};

use crate::music::{Profile, TrackId};

/// How long before midnight the run-in is locked.
pub const LOCK: Duration = Duration::minutes(20);

/// One track of the run-in: where it enters (its beat when it starts
/// playing) and where the next starts over it — for the midnight track, its
/// drop.
pub struct Leg<'a> {
    pub track: &'a Profile,
    pub enter: f64,
    pub leave: f64,
}

/// When each track of the run-in starts, and when its handover begins.
#[derive(Debug, Clone, PartialEq)]
pub struct Slot {
    pub track: TrackId,
    pub starts: DateTime<Local>,
    pub hands_over: DateTime<Local>,
}

fn beats(track: &Profile, n: f64) -> Duration {
    Duration::microseconds((n * track.grid.ms_per_beat() * 1000.0).round() as i64)
}

/// The run-in `legs` — the midnight track last — scheduled back from
/// `midnight`.
pub fn schedule(midnight: DateTime<Local>, legs: &[Leg<'_>]) -> Vec<Slot> {
    let mut out = Vec::with_capacity(legs.len());
    let mut next_start = midnight;
    for leg in legs.iter().rev() {
        let starts = next_start - beats(leg.track, leg.leave - leg.enter);
        out.push(Slot {
            track: leg.track.id.clone(),
            starts,
            hands_over: next_start,
        });
        next_start = starts;
    }
    out.reverse();
    out
}

/// The playing track's beat at which the next must start to start at
/// `target`, given it is at `beat` now at `bpm`: on the bar nearest. `None`
/// when that is already past.
pub fn out_beat_for(
    target: DateTime<Local>,
    now: DateTime<Local>,
    beat: f64,
    bpm: f32,
) -> Option<f64> {
    let secs = (target - now).num_microseconds()? as f64 / 1e6;
    let exact = beat + secs * bpm as f64 / 60.0;
    let bar = (exact / 4.0).round() * 4.0;
    (bar > beat).then_some(bar)
}

/// Whether the run-in is locked at `now`.
pub fn locked(now: DateTime<Local>, midnight: DateTime<Local>) -> bool {
    now >= midnight - LOCK && now < midnight + Duration::minutes(5)
}

#[cfg(test)]
mod tests {
    use chrono::{NaiveDate, TimeZone};

    use super::*;
    use crate::music::track::fixture::club;

    fn midnight() -> DateTime<Local> {
        Local
            .from_local_datetime(
                &NaiveDate::from_ymd_opt(2027, 1, 1)
                    .unwrap()
                    .and_hms_opt(0, 0, 0)
                    .unwrap(),
            )
            .earliest()
            .unwrap()
    }

    /// **The drop lands on midnight**: the midnight track starts its drop's
    /// beats before it, and each track before starts its play's length before
    /// the next — exact to the microsecond.
    #[test]
    fn the_run_in_is_scheduled_back_from_the_drop() {
        let a = club("a", 120.0, "8A", 0.8); // 500 ms a beat
        let b = club("b", 125.0, "8A", 0.9); // 480 ms
        let m = club("m", 128.0, "8A", 1.0); // 468.75 ms
        let slots = schedule(
            midnight(),
            &[
                Leg { track: &a, enter: 0.0, leave: 224.0 },
                Leg { track: &b, enter: 0.0, leave: 224.0 },
                Leg { track: &m, enter: 0.0, leave: 64.0 },
            ],
        );
        let t = |s: &Slot| (midnight() - s.starts).num_milliseconds();
        assert_eq!(slots[2].track, TrackId("m".into()));
        assert_eq!(t(&slots[2]), 30_000, "64 beats at 128");
        assert_eq!(t(&slots[1]), 30_000 + 107_520, "224 beats at 125");
        assert_eq!(t(&slots[0]), 30_000 + 107_520 + 112_000, "224 at 120");
        assert_eq!(slots[1].hands_over, slots[2].starts);
        assert_eq!(slots[2].hands_over, midnight());
    }

    #[test]
    fn the_out_point_for_a_target_is_the_nearest_bar_ahead() {
        let now = midnight() - Duration::seconds(60);
        // 60 s at 120 BPM is 120 beats; from beat 30.0 that is 150, nearest
        // bar 152.
        assert_eq!(out_beat_for(midnight(), now, 30.0, 120.0), Some(152.0));
        assert_eq!(out_beat_for(now, midnight(), 30.0, 120.0), None, "past");
    }

    #[test]
    fn the_run_in_locks_twenty_minutes_out() {
        assert!(!locked(midnight() - Duration::minutes(21), midnight()));
        assert!(locked(midnight() - Duration::minutes(20), midnight()));
        assert!(locked(midnight() + Duration::minutes(1), midnight()));
        assert!(!locked(midnight() + Duration::minutes(10), midnight()));
    }
}
