//! The hybrid logical clock a record's `hlc` is read from.
//!
//! A reading is `(wall milliseconds << 16) | counter`. The high 48 bits follow
//! the wall clock, so a reading says roughly when it was taken and a tombstone
//! can be aged by it (§6.8). The low 16 bits count readings taken within one
//! millisecond, so the clock never repeats or runs backwards however fast it is
//! read, and a burst of records does not drag the clock ahead of the wall.
//!
//! Every reading is above every reading this clock has taken or observed, so a
//! record written after another — by this substrate, or after this substrate
//! applied the other — always carries the later reading (§6.2). A wall clock
//! that steps backwards holds the clock where it was, counting up from there.

/// Bits of a reading given to the counter.
pub const COUNTER_BITS: u32 = 16;

/// The wall-clock milliseconds a reading was taken at.
pub fn wall_ms(hlc: u64) -> u64 {
    hlc >> COUNTER_BITS
}

/// A hybrid logical clock.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Hlc {
    /// The greatest reading taken or observed.
    last: u64,
}

impl Hlc {
    /// A clock that has read nothing.
    pub fn new() -> Hlc {
        Hlc::default()
    }

    /// The next reading, with the wall clock at `now_ms`.
    pub fn tick(&mut self, now_ms: u64) -> u64 {
        let wall = now_ms << COUNTER_BITS;
        self.last = if wall > self.last {
            wall
        } else {
            self.last + 1
        };
        self.last
    }

    /// Take `reading` into account: every later reading is above it.
    pub fn observe(&mut self, reading: u64) {
        self.last = self.last.max(reading);
    }

    /// The greatest reading taken or observed.
    pub fn last(&self) -> u64 {
        self.last
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_millisecond_starts_the_counter_at_zero() {
        let mut c = Hlc::new();
        assert_eq!(c.tick(1_000), 1_000 << 16);
        assert_eq!(c.tick(1_001), 1_001 << 16);
        assert_eq!(wall_ms(c.last()), 1_001);
    }

    /// Readings within one millisecond count up, and stay in that millisecond
    /// while the counter lasts.
    #[test]
    fn readings_in_one_millisecond_count_up() {
        let mut c = Hlc::new();
        let a = c.tick(5);
        let b = c.tick(5);
        let d = c.tick(5);
        assert_eq!((a, b, d), (5 << 16, (5 << 16) + 1, (5 << 16) + 2));
        assert_eq!(wall_ms(d), 5);
    }

    /// A wall clock that steps back cannot make a reading repeat or fall.
    #[test]
    fn a_wall_clock_stepping_back_holds_the_clock() {
        let mut c = Hlc::new();
        let a = c.tick(10_000);
        let b = c.tick(9_000);
        assert_eq!(b, a + 1);
    }

    /// An observed reading from another clock pulls this one past it.
    #[test]
    fn an_observed_reading_is_passed() {
        let mut c = Hlc::new();
        c.tick(100);
        c.observe((500 << 16) | 7);
        assert_eq!(c.tick(100), (500 << 16) + 8);
        // An older observation changes nothing.
        c.observe(1);
        assert_eq!(c.tick(100), (500 << 16) + 9);
    }
}
