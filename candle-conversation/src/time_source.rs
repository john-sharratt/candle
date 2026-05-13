//! Pluggable time source for temporal marker generation.
//!
//! The [`TimeSource`] trait abstracts wall-clock access so that tests and
//! narrative-time applications can inject deterministic or fictional clocks
//! without touching production code.

use std::time::{Instant, SystemTime};

// ────────────────────────────────────────────────────────────────────────────
// Trait
// ────────────────────────────────────────────────────────────────────────────

/// Provides the current time and temporal coordinate computation for marker
/// generation.
///
/// Implementations capture a reference instant at construction and expose:
/// - The current wall time (`now()`).
/// - The elapsed day count since the reference instant
///   (`days_since_reference()`).
///
/// Override for deterministic testing, narrative time in fiction applications,
/// or replaying historical conversation sequences.
pub trait TimeSource: Send + Sync + 'static {
    /// Returns the current time. Called once per turn at submit time.
    fn now(&self) -> SystemTime;

    /// Compute the day component for a temporal marker.
    ///
    /// Returns elapsed days (i32) since this `TimeSource`'s internal
    /// reference instant. Negative values are allowed for testing and replay
    /// scenarios where `now()` returns a time before the reference.
    fn days_since_reference(&self) -> i32;
}

// ────────────────────────────────────────────────────────────────────────────
// WallClockTimeSource — default implementation
// ────────────────────────────────────────────────────────────────────────────

/// Default [`TimeSource`] backed by the system wall clock.
///
/// Captures a reference [`Instant`] at construction; all subsequent calls
/// to [`days_since_reference`](Self::days_since_reference) measure elapsed
/// days from that instant.
pub struct WallClockTimeSource {
    /// Captured at `new()` time; used as the day-zero reference.
    reference_instant: Instant,
}

impl WallClockTimeSource {
    /// Create a new `WallClockTimeSource` with the reference fixed at the
    /// current moment.
    pub fn new() -> Self {
        Self {
            reference_instant: Instant::now(),
        }
    }
}

impl Default for WallClockTimeSource {
    fn default() -> Self {
        Self::new()
    }
}

impl TimeSource for WallClockTimeSource {
    fn now(&self) -> SystemTime {
        SystemTime::now()
    }

    fn days_since_reference(&self) -> i32 {
        let elapsed_secs = self.reference_instant.elapsed().as_secs();
        (elapsed_secs / 86_400) as i32
    }
}

// ────────────────────────────────────────────────────────────────────────────
// FixedTimeSource — deterministic clock for tests
// ────────────────────────────────────────────────────────────────────────────

/// A deterministic [`TimeSource`] that returns a fixed day value.
///
/// Useful for writing unit tests where exact marker values need to be
/// predictable.
///
/// ```
/// use candle_conversation::tree::FixedTimeSource;
/// use candle_conversation::tree::TimeSource;
///
/// let ts = FixedTimeSource::at_day(3);
/// assert_eq!(ts.days_since_reference(), 3);
/// ```
pub struct FixedTimeSource {
    day: i32,
}

impl FixedTimeSource {
    /// Create a source that always returns `day` as the elapsed day count.
    pub fn at_day(day: i32) -> Self {
        Self { day }
    }
}

impl TimeSource for FixedTimeSource {
    fn now(&self) -> SystemTime {
        // Offset from UNIX_EPOCH by the fixed day count so `now()` is
        // consistent with `days_since_reference()`.
        SystemTime::UNIX_EPOCH
            + std::time::Duration::from_secs(self.day as u64 * 86_400)
    }

    fn days_since_reference(&self) -> i32 {
        self.day
    }
}
