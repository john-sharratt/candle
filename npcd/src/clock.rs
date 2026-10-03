//! The narrative clock — what time it is inside a world.
//!
//! A world runs on its own time, and characters date what they remember by it.
//! That is a small amount of arithmetic and no engine at all, which is why it
//! lives here rather than waiting on one: the console has had a clock panel
//! since the beginning, and until now its Save was a toast.
//!
//! # Three numbers, not a ticking counter
//!
//! Nothing counts. A world stores an **anchor** — the narrative time
//! `world_ms`, the real time `at_ms` when that was true, and the `scale` in
//! world-milliseconds per real millisecond — and the current time is computed
//! from it whenever anybody asks:
//!
//! ```text
//! now = world_ms + (real_now - at_ms) * scale
//! ```
//!
//! So a daemon that is restarted, or asleep for a week, resumes at the time the
//! world would have reached — the clock is a function of wall time, not
//! something that stops when the process does. A counter written back on a
//! timer would also mean a disk write per tick, per world, for ever.
//!
//! `scale: 0` is paused, and pausing is not a special case: the arithmetic
//! already stops at zero. It is stored as its own flag as well because a paused
//! world should remember the speed it was going, so resuming does not silently
//! land on 1×.

//! # Years ahead
//!
//! A world may sit in the future: `year_offset` moves the calendar that many
//! years on from the anchor's, and everything that reads the world's time —
//! journal stamps, the console, the day a character wakes into — reads it
//! shifted. The offset is **not** part of the anchor, so changing the pace or
//! jumping to a time does not disturb it, and a jump names the time as the world
//! reads it, offset included.
//!
//! # One place to ask
//!
//! World time is a number of milliseconds since 1970-01-01T00:00 in the world,
//! and [`WorldTime`] is the calendar read off it — the only place in the daemon
//! that turns that number into a year, a date or a clock face. Prose that names
//! a moment goes through [`stamp`], [`WorldTime::date`] or [`WorldTime::clock`];
//! nothing else divides by a day.

use serde_json::{json, Map, Value};

/// World-clock milliseconds in one narrative day.
pub const DAY_MS: u64 = 24 * 60 * 60 * 1000;

/// Which day a world-clock instant falls in, counted from 1970-01-01.
pub fn day_of(world_ms: u64) -> u64 {
    world_ms / DAY_MS
}

const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/// Days from 1970-01-01 to a proleptic-Gregorian civil date.
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The civil date of a day number counted from 1970-01-01.
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

/// The calendar a world-clock instant reads as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorldTime {
    pub year: i64,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
}

impl WorldTime {
    /// The calendar for `world_ms`.
    pub fn of(world_ms: u64) -> Self {
        let days = (world_ms / DAY_MS) as i64;
        let into = world_ms % DAY_MS / 1000;
        let (year, month, day) = civil_from_days(days);
        Self {
            year,
            month: month as u32,
            day: day as u32,
            hour: (into / 3600) as u32,
            minute: (into % 3600 / 60) as u32,
            second: (into % 60) as u32,
        }
    }

    /// `14 Jun 2187`.
    pub fn date(&self) -> String {
        format!(
            "{} {} {}",
            self.day,
            MONTHS[self.month as usize - 1],
            self.year
        )
    }

    /// `14:20`.
    pub fn clock(&self) -> String {
        format!("{:02}:{:02}", self.hour, self.minute)
    }

    /// `14 Jun 2187, 14:20`.
    pub fn stamp(&self) -> String {
        format!("{}, {}", self.date(), self.clock())
    }
}

/// `14 Jun 2187, 14:20` for a world instant.
pub fn stamp(world_ms: u64) -> String {
    WorldTime::of(world_ms).stamp()
}

/// Milliseconds a calendar moves by when it is put `years` on from 1970.
///
/// Whole years from the epoch's own date, so a multiple of 400 years is exact
/// and any other lands within a day of the same calendar date. A fixed amount,
/// so the clock stays monotonic whichever way the leap days fall.
fn years_ms(years: i32) -> i64 {
    days_from_civil(1970 + i64::from(years), 1, 1).saturating_mul(DAY_MS as i64)
}

/// The default pace of a world that has never said: one world-second per real
/// second. A world with no clock in its document reads as though it were
/// started now at 1×, which is what an author who has not thought about it
/// means.
const DEFAULT_SCALE: f64 = 1.0;

/// A world's clock, as stored.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Clock {
    /// Narrative time at the anchor.
    pub world_ms: i64,
    /// Real time when the anchor was set.
    pub at_ms: i64,
    /// World milliseconds per real millisecond.
    pub scale: f64,
    /// Whether the clock is stopped. Held separately from `scale` so a paused
    /// world remembers what speed to resume at.
    pub paused: bool,
    /// Whole years the world's calendar is set ahead of its anchor; negative
    /// for the past.
    pub year_offset: i32,
}

impl Clock {
    /// The clock a world with nothing written has: starting now, at 1×.
    pub fn started_now(now_ms: i64) -> Self {
        Self {
            world_ms: now_ms,
            at_ms: now_ms,
            scale: DEFAULT_SCALE,
            paused: false,
            year_offset: 0,
        }
    }

    /// Read a world document's `time` block, or the default when it has none.
    ///
    /// Every field is taken independently and falls back on its own, because
    /// these are hand-written: a document that says only `scale: 60` means a
    /// world running an hour a minute from now, not a malformed clock.
    pub fn of_world(body: &Value, now_ms: i64) -> Self {
        let Some(t) = body.get("time").and_then(Value::as_object) else {
            return Self::started_now(now_ms);
        };
        let num = |k: &str| t.get(k).and_then(Value::as_i64);
        let world_ms = num("world_ms").unwrap_or(now_ms);
        Self {
            world_ms,
            // An anchor with no real-world timestamp is read as "true now".
            // The alternative — treating it as the epoch — would advance the
            // world by fifty-six years the first time anybody looked.
            at_ms: num("at_ms").unwrap_or(now_ms),
            scale: t
                .get("scale")
                .and_then(Value::as_f64)
                .filter(|s| s.is_finite() && *s >= 0.0)
                .unwrap_or(DEFAULT_SCALE),
            paused: t.get("paused").and_then(Value::as_bool).unwrap_or(false),
            year_offset: num("year_offset")
                .and_then(|y| i32::try_from(y).ok())
                .unwrap_or(0),
        }
    }

    /// What time it is in the world now, the year offset included.
    pub fn now(&self, now_ms: i64) -> i64 {
        self.anchored_now(now_ms)
            .saturating_add(years_ms(self.year_offset))
    }

    /// The time the anchor and pace give, before the calendar is moved.
    ///
    /// Saturating, because the arithmetic is a scaled elapsed time and a world
    /// left running at 1440× for long enough would otherwise overflow into a
    /// date before it started.
    fn anchored_now(&self, now_ms: i64) -> i64 {
        if self.paused || self.scale == 0.0 {
            return self.world_ms;
        }
        let elapsed = (now_ms - self.at_ms).max(0) as f64 * self.scale;
        // `as i64` saturates on overflow, and the clamp keeps a preposterous
        // scale from producing a negative date.
        self.world_ms
            .saturating_add(elapsed.min(i64::MAX as f64) as i64)
    }

    /// Re-anchor so the world reads `world_ms` as of now, keeping the pace and
    /// the offset. `world_ms` is the time as the world reads it, which is what
    /// the console shows.
    pub fn jump_to(self, world_ms: i64, now_ms: i64) -> Self {
        Self {
            world_ms: world_ms.saturating_sub(years_ms(self.year_offset)),
            at_ms: now_ms,
            ..self
        }
    }

    /// Put the calendar `years` ahead of the anchor, without moving the anchor.
    /// The world reads that much later at once.
    pub fn with_year_offset(self, years: i32, now_ms: i64) -> Self {
        Self {
            world_ms: self.anchored_now(now_ms),
            at_ms: now_ms,
            year_offset: years,
            ..self
        }
    }

    /// Change the pace without moving the clock.
    ///
    /// The anchor is re-taken at the *current* narrative time first, so the
    /// elapsed run at the old speed is banked rather than recomputed at the
    /// new one. Without that, changing 1× to 60× would retroactively speed up
    /// every hour the world had already run.
    pub fn set_pace(self, scale: f64, paused: bool, now_ms: i64) -> Self {
        Self {
            world_ms: self.anchored_now(now_ms),
            at_ms: now_ms,
            scale: if scale.is_finite() && scale >= 0.0 {
                scale
            } else {
                self.scale
            },
            paused,
            year_offset: self.year_offset,
        }
    }

    /// What the console reads: the time now, the pace it is running at, and how
    /// many years ahead the calendar is set.
    pub fn wire(&self, now_ms: i64) -> Value {
        json!({
            "world_ms": self.now(now_ms),
            "scale": self.scale,
            "paused": self.paused,
            "year_offset": self.year_offset,
        })
    }

    /// What goes back into the world's YAML.
    pub fn to_document(self) -> Value {
        json!({
            "world_ms": self.world_ms,
            "at_ms": self.at_ms,
            "scale": self.scale,
            "paused": self.paused,
            "year_offset": self.year_offset,
        })
    }
}

/// A world document with its clock replaced.
pub fn with_clock(body: &Value, clock: Clock) -> Map<String, Value> {
    let mut map = body.as_object().cloned().unwrap_or_default();
    map.insert("time".into(), clock.to_document());
    map
}

/// What time it is for one character, given already-taken guards.
///
/// **Guards in, not locks.** The two callers reach the same registries by
/// incompatible routes — the tick driver is a plain OS thread and takes
/// `blocking_read`, an axum handler is inside the runtime and must `.await` —
/// and `blocking_read` from a runtime thread does not contend, it *panics*.
/// That is not a hypothetical: the census route did exactly this, and the whole
/// Pulse page answered with a closed connection while the tick loop it was
/// meant to be showing ran perfectly underneath.
///
/// So the locking is the caller's and the lookup is here, once. Zero for a
/// character with no world, or a world with no clock — a character whose world
/// cannot be read has no time of its own, and inventing one would put it in a
/// day the world has never been in.
pub fn world_ms_for(
    npcs: &crate::npcs::Npcs,
    worlds: &crate::registry::Registry,
    npc_id: u64,
    now_ms: i64,
) -> u64 {
    let Some(world_id) = npcs.world_of(npc_id) else {
        return 0;
    };
    worlds
        .get(world_id)
        .map(|r| Clock::of_world(&r.body, now_ms).now(now_ms).max(0) as u64)
        .unwrap_or(0)
}

/// The calendar for one character's world, by the same route as
/// [`world_ms_for`].
pub fn world_time_for(
    npcs: &crate::npcs::Npcs,
    worlds: &crate::registry::Registry,
    npc_id: u64,
    now_ms: i64,
) -> WorldTime {
    WorldTime::of(world_ms_for(npcs, worlds, npc_id, now_ms))
}

#[cfg(test)]
mod tests {
    use super::*;

    const T0: i64 = 1_700_000_000_000;

    #[test]
    fn the_calendar_is_read_off_the_milliseconds() {
        assert_eq!(stamp(0), "1 Jan 1970, 00:00");
        // 2023-11-14T22:13:20Z.
        assert_eq!(stamp(T0 as u64), "14 Nov 2023, 22:13");
        assert_eq!(stamp(T0 as u64 + 59_000), "14 Nov 2023, 22:14");
        assert_eq!(stamp(DAY_MS - 1), "1 Jan 1970, 23:59");
        // A leap day, and the day after it.
        assert_eq!(stamp(951_782_400_000), "29 Feb 2000, 00:00");
        assert_eq!(stamp(951_868_800_000), "1 Mar 2000, 00:00");
        assert_eq!(WorldTime::of(T0 as u64).second, 20);
    }

    #[test]
    fn a_date_and_a_clock_face_are_the_halves_of_a_stamp() {
        let t = WorldTime::of(T0 as u64);
        assert_eq!(t.date(), "14 Nov 2023");
        assert_eq!(t.clock(), "22:13");
    }

    /// 217 years past 1970 with 53 leap days between, then 164 days into 2187.
    #[test]
    fn a_far_future_date_reads_as_itself() {
        let days = days_from_civil(2187, 6, 14);
        assert_eq!(days, 217 * 365 + 53 + 164);
        let at = days as u64 * DAY_MS + 9 * 3_600_000 + 5 * 60_000;
        assert_eq!(stamp(at), "14 Jun 2187, 09:05");
    }

    #[test]
    fn a_year_offset_moves_the_calendar_and_nothing_else() {
        let base = Clock::of_world(
            &json!({ "time": { "world_ms": 0, "at_ms": T0, "scale": 1 } }),
            T0,
        );
        let ahead = base.with_year_offset(400, T0);
        // 400 Gregorian years is exact: 146 097 days.
        assert_eq!(ahead.now(T0), 146_097 * DAY_MS as i64);
        assert_eq!(stamp(ahead.now(T0) as u64), "1 Jan 2370, 00:00");
        // It keeps running at the same pace.
        assert_eq!(ahead.now(T0 + 60_000) - ahead.now(T0), 60_000);
        assert_eq!(base.now(T0), 0);
    }

    #[test]
    fn a_pace_change_or_a_jump_keeps_the_offset() {
        let c = Clock::started_now(T0).with_year_offset(100, T0);
        let faster = c.set_pace(60.0, false, T0 + 1_000);
        assert_eq!(faster.year_offset, 100);
        assert_eq!(faster.now(T0 + 1_000), c.now(T0 + 1_000));

        // A jump names the time as the world reads it.
        let target = 2_000_000_000_000;
        assert_eq!(c.jump_to(target, T0).now(T0), target);
    }

    #[test]
    fn the_offset_is_stored_in_the_document_and_read_back() {
        let c = Clock::started_now(T0).with_year_offset(-3, T0);
        let body = Value::Object(with_clock(&json!({}), c));
        assert_eq!(body["time"]["year_offset"], -3);
        assert_eq!(Clock::of_world(&body, T0).year_offset, -3);
        // A document that never said has none.
        assert_eq!(Clock::of_world(&json!({ "time": {} }), T0).year_offset, 0);
        assert_eq!(c.wire(T0)["year_offset"], -3);
    }

    #[test]
    fn a_world_with_no_clock_starts_now_at_real_time() {
        let c = Clock::of_world(&json!({ "id": "earth" }), T0);
        assert_eq!(c.now(T0), T0);
        assert_eq!(c.scale, 1.0);
        assert!(!c.paused);
        // And it advances with the wall clock.
        assert_eq!(c.now(T0 + 60_000), T0 + 60_000);
    }

    /// **The clock is a function of wall time, not a counter.** A daemon that
    /// was off for an hour comes back to the time the world reached.
    #[test]
    fn time_passes_while_nothing_is_running() {
        let c = Clock::of_world(
            &json!({ "time": { "world_ms": 0, "at_ms": T0, "scale": 60 } }),
            T0,
        );
        assert_eq!(c.now(T0), 0);
        // One real minute at 60× is one world hour.
        assert_eq!(c.now(T0 + 60_000), 3_600_000);
    }

    #[test]
    fn a_paused_world_does_not_move() {
        let c = Clock::of_world(
            &json!({ "time": { "world_ms": 500, "at_ms": T0, "scale": 60, "paused": true } }),
            T0,
        );
        assert_eq!(c.now(T0 + 10_000_000), 500);
        // And it remembers the pace to resume at, rather than landing on 1×.
        assert_eq!(c.scale, 60.0);
    }

    /// **Changing the pace must not rewrite history.** The elapsed run at the
    /// old speed is banked at the moment of the change.
    #[test]
    fn changing_the_pace_banks_what_already_elapsed() {
        let c = Clock::of_world(
            &json!({ "time": { "world_ms": 0, "at_ms": T0, "scale": 1 } }),
            T0,
        );
        // One real hour has passed at 1×.
        let at = T0 + 3_600_000;
        assert_eq!(c.now(at), 3_600_000);

        let faster = c.set_pace(60.0, false, at);
        // Still the same time at the moment of the change.
        assert_eq!(faster.now(at), 3_600_000);
        // And the next real minute is a world hour, on top of what was banked.
        assert_eq!(faster.now(at + 60_000), 3_600_000 + 3_600_000);
    }

    #[test]
    fn a_jump_moves_the_clock_and_keeps_the_pace() {
        let c = Clock::of_world(
            &json!({ "time": { "world_ms": 0, "at_ms": T0, "scale": 10 } }),
            T0,
        );
        let jumped = c.jump_to(999_000, T0);
        assert_eq!(jumped.now(T0), 999_000);
        assert_eq!(jumped.scale, 10.0);
        assert_eq!(jumped.now(T0 + 1_000), 999_000 + 10_000);
    }

    /// Hand-written documents are partial, and each field falls back on its own.
    #[test]
    fn a_partial_clock_block_is_read_field_by_field() {
        let c = Clock::of_world(&json!({ "time": { "scale": 60 } }), T0);
        assert_eq!(c.scale, 60.0);
        // No anchor written: read as true now, rather than as the epoch — which
        // would advance the world by decades on the first read.
        assert_eq!(c.now(T0), T0);
    }

    /// A scale that is not a number, is negative, or is not finite keeps the
    /// one that was there. A world running backwards is not a thing to store.
    #[test]
    fn a_nonsense_pace_is_refused_rather_than_stored() {
        let c = Clock::of_world(
            &json!({ "time": { "world_ms": 0, "at_ms": T0, "scale": -5 } }),
            T0,
        );
        assert_eq!(c.scale, 1.0, "a negative scale was taken");
        let kept = c.set_pace(f64::NAN, false, T0);
        assert_eq!(kept.scale, 1.0);
    }

    #[test]
    fn the_document_round_trips_through_the_world_body() {
        let c = Clock::started_now(T0).set_pace(60.0, true, T0);
        let body = json!({ "id": "earth", "name": "Earth" });
        let next = Value::Object(with_clock(&body, c));
        assert_eq!(next["id"], "earth", "the rest of the document survived");
        let back = Clock::of_world(&next, T0);
        assert_eq!(back, c);
    }
}
