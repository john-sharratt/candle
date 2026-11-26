//! When this member takes no unit to read at all.
//!
//! Two reasons, both member-wide and both checked by
//! [`crate::ingest_lease`] before it asks the cluster for a unit:
//!
//! - **Ingest is off here** — `ingest: false` in `motherd.yaml`'s `zend:`
//!   section, or `--no-ingest`. The member replicates every reading the others
//!   write and reads none itself: a card that runs the cast, or one too small
//!   to read beside its live turns, leaves the reading to a member that can.
//! - **One of its readings failed a moment ago.** A reading that stalls or
//!   fails hands its lease back as it ends, and a member that then took the
//!   next unit would fail that one for the same reason — a card too full to
//!   move a turn stays full. On 2026-11-21 WhiteWorld held seven folders' leases
//!   through a 25-minute pass that committed none of them, took all seven again
//!   on the next, and BlackWorld, idle, was refused every one. After a failure
//!   the member takes no new unit for [`FAILURE_COOLDOWN`], so another member's
//!   next pass finds the units free.
//!
//! What is already being read goes on: the hold-off refuses new leases, it
//! does not end the readings that hold one.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// How long a member takes no new unit after one of its readings failed.
pub const FAILURE_COOLDOWN: Duration = Duration::from_secs(10 * 60);

/// Whether ingest is on here, and until when a failure holds it off.
pub struct Holdoff {
    enabled: AtomicBool,
    until: Mutex<Option<Instant>>,
}

impl Holdoff {
    /// Ingest on, nothing failed.
    pub const fn new() -> Self {
        Self {
            enabled: AtomicBool::new(true),
            until: Mutex::new(None),
        }
    }

    /// Turn this member's ingest on or off.
    pub fn set_enabled(&self, on: bool) {
        self.enabled.store(on, Ordering::Relaxed);
    }

    /// Whether ingest is on here at all.
    pub fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// A reading failed at `now`: take no new unit for [`FAILURE_COOLDOWN`]
    /// from then. A later failure moves the end later; an earlier one never
    /// shortens it. Returns when the hold-off ends.
    pub fn cool_down(&self, now: Instant) -> Instant {
        let end = now + FAILURE_COOLDOWN;
        let mut until = self.until.lock().unwrap_or_else(|e| e.into_inner());
        let end = until.map_or(end, |u| u.max(end));
        *until = Some(end);
        end
    }

    /// Whether this member leaves every unit at `now`: ingest is off, or a
    /// failure's cooldown has not run out.
    pub fn leaves(&self, now: Instant) -> bool {
        if !self.enabled() {
            return true;
        }
        let until = self.until.lock().unwrap_or_else(|e| e.into_inner());
        until.is_some_and(|u| now < u)
    }

    /// Whether only a failure's cooldown leaves units at `now`: ingest is on
    /// here and the cooldown has not run out — a hold that ends by itself,
    /// which a pass can wait out, where ingest off never does.
    pub fn cooling_down(&self, now: Instant) -> bool {
        self.enabled() && self.leaves(now)
    }
}

impl Default for Holdoff {
    fn default() -> Self {
        Self::new()
    }
}

/// This process's hold-off.
static HOLDOFF: Holdoff = Holdoff::new();

/// This process's hold-off, which every unit lease consults.
pub fn holdoff() -> &'static Holdoff {
    &HOLDOFF
}

/// Turn this member's ingest on or off, as its settings say.
pub fn set_ingest(on: bool) {
    HOLDOFF.set_enabled(on);
}

/// The reading of `unit` failed: this member takes no new unit for
/// [`FAILURE_COOLDOWN`], so another member can.
pub fn cool_down(unit: &str, why: &str) {
    let now = Instant::now();
    let end = HOLDOFF.cool_down(now);
    tracing::warn!(
        target: "zend::ingest",
        unit,
        cooldown_s = end.saturating_duration_since(now).as_secs(),
        "a reading failed here — this member takes no new unit until the cooldown ends, \
         and its lease is free for another member: {why}",
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Off leaves every unit**, at any time, and on again takes them.
    #[test]
    fn ingest_off_leaves_every_unit() {
        let h = Holdoff::new();
        let now = Instant::now();
        assert!(!h.leaves(now));
        h.set_enabled(false);
        assert!(!h.enabled());
        assert!(h.leaves(now));
        assert!(h.leaves(now + Duration::from_secs(86_400)));
        h.set_enabled(true);
        assert!(!h.leaves(now));
    }

    /// **A failure holds new units off for the cooldown**, and no longer.
    #[test]
    fn a_failure_holds_units_off_for_the_cooldown() {
        let h = Holdoff::new();
        let t0 = Instant::now();
        assert_eq!(h.cool_down(t0), t0 + FAILURE_COOLDOWN);
        assert!(h.leaves(t0));
        assert!(h.leaves(t0 + FAILURE_COOLDOWN - Duration::from_secs(1)));
        assert!(!h.leaves(t0 + FAILURE_COOLDOWN));
    }

    /// **A later failure extends the cooldown; an earlier one never cuts it.**
    #[test]
    fn a_later_failure_extends_and_an_earlier_one_never_shortens() {
        let h = Holdoff::new();
        let t0 = Instant::now();
        let later = t0 + Duration::from_secs(120);
        assert_eq!(h.cool_down(later), later + FAILURE_COOLDOWN);
        assert_eq!(h.cool_down(t0), later + FAILURE_COOLDOWN);
        assert!(h.leaves(t0 + FAILURE_COOLDOWN + Duration::from_secs(60)));
        assert!(!h.leaves(later + FAILURE_COOLDOWN));
    }

    /// **Only a running cooldown is one a pass waits out**: ingest off is
    /// never, and a cooldown stops being one when it ends.
    #[test]
    fn only_a_running_cooldown_is_waited_out() {
        let h = Holdoff::new();
        let t0 = Instant::now();
        assert!(!h.cooling_down(t0));
        h.cool_down(t0);
        assert!(h.cooling_down(t0));
        assert!(!h.cooling_down(t0 + FAILURE_COOLDOWN));
        h.set_enabled(false);
        assert!(h.leaves(t0));
        assert!(!h.cooling_down(t0));
    }

    #[test]
    fn the_cooldown_is_ten_minutes() {
        assert_eq!(FAILURE_COOLDOWN, Duration::from_secs(600));
    }
}
