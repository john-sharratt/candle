//! Where a section seal's time goes inside the scheduler: waiting for
//! admission, preparing its slot (lifting the prefix it attends to hot and
//! injecting it), its forward passes, and the seal that writes it.
//!
//! A conversation's journal is a chain of sections, each sealed after the one
//! before, and its first round waits for the chain. One line per section says
//! which of the four a slow link spent its time in.

use std::time::Instant;

/// The instants a section's ingest passes, from its queueing on.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SectionTiming {
    queued_at: Instant,
    begun_at: Instant,
    prepared_at: Instant,
}

/// A section's time, by part, in microseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SectionSplit {
    pub queue_us: u64,
    pub prepare_us: u64,
    pub forward_us: u64,
    pub seal_us: u64,
}

impl SectionTiming {
    /// A section admitted at `begun_at` after being queued at `queued_at`,
    /// whose slot was prepared by `prepared_at`.
    pub(super) fn new(queued_at: Instant, begun_at: Instant, prepared_at: Instant) -> Self {
        Self {
            queued_at,
            begun_at,
            prepared_at,
        }
    }

    /// The split of a section whose forward passes ended at `sealing` and
    /// whose seal ended at `done`.
    pub(super) fn split(&self, sealing: Instant, done: Instant) -> SectionSplit {
        let us = |from: Instant, to: Instant| to.saturating_duration_since(from).as_micros() as u64;
        SectionSplit {
            queue_us: us(self.queued_at, self.begun_at),
            prepare_us: us(self.begun_at, self.prepared_at),
            forward_us: us(self.prepared_at, sealing),
            seal_us: us(sealing, done),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn each_part_is_the_span_between_its_instants() {
        let t0 = Instant::now();
        let at = |us: u64| t0 + Duration::from_micros(us);
        let timing = SectionTiming::new(t0, at(1_500), at(41_500));
        assert_eq!(
            timing.split(at(98_000), at(110_250)),
            SectionSplit {
                queue_us: 1_500,
                prepare_us: 40_000,
                forward_us: 56_500,
                seal_us: 12_250,
            }
        );
    }

    #[test]
    fn instants_out_of_order_read_as_no_time() {
        let t0 = Instant::now();
        let later = t0 + Duration::from_millis(5);
        let timing = SectionTiming::new(later, t0, t0);
        assert_eq!(
            timing.split(t0, t0),
            SectionSplit {
                queue_us: 0,
                prepare_us: 0,
                forward_us: 0,
                seal_us: 0,
            }
        );
    }
}
