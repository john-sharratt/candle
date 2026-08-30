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

use serde_json::{json, Map, Value};

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
}

impl Clock {
    /// The clock a world with nothing written has: starting now, at 1×.
    pub fn started_now(now_ms: i64) -> Self {
        Self {
            world_ms: now_ms,
            at_ms: now_ms,
            scale: DEFAULT_SCALE,
            paused: false,
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
        }
    }

    /// What time it is in the world now.
    ///
    /// Saturating, because the arithmetic is a scaled elapsed time and a world
    /// left running at 1440× for long enough would otherwise overflow into a
    /// date before it started.
    pub fn now(&self, now_ms: i64) -> i64 {
        if self.paused || self.scale == 0.0 {
            return self.world_ms;
        }
        let elapsed = (now_ms - self.at_ms).max(0) as f64 * self.scale;
        // `as i64` saturates on overflow, and the clamp keeps a preposterous
        // scale from producing a negative date.
        self.world_ms
            .saturating_add(elapsed.min(i64::MAX as f64) as i64)
    }

    /// Re-anchor so the world reads `world_ms` as of now, keeping the pace.
    pub fn jump_to(self, world_ms: i64, now_ms: i64) -> Self {
        Self {
            world_ms,
            at_ms: now_ms,
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
            world_ms: self.now(now_ms),
            at_ms: now_ms,
            scale: if scale.is_finite() && scale >= 0.0 {
                scale
            } else {
                self.scale
            },
            paused,
        }
    }

    /// What the console reads: the time now, and the pace it is running at.
    pub fn wire(&self, now_ms: i64) -> Value {
        json!({
            "world_ms": self.now(now_ms),
            "scale": self.scale,
            "paused": self.paused,
        })
    }

    /// What goes back into the world's YAML.
    pub fn to_document(self) -> Value {
        json!({
            "world_ms": self.world_ms,
            "at_ms": self.at_ms,
            "scale": self.scale,
            "paused": self.paused,
        })
    }
}

/// A world document with its clock replaced.
pub fn with_clock(body: &Value, clock: Clock) -> Map<String, Value> {
    let mut map = body.as_object().cloned().unwrap_or_default();
    map.insert("time".into(), clock.to_document());
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    const T0: i64 = 1_700_000_000_000;

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
