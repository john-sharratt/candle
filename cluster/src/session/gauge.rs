//! What a session holds queued and unwritten, by lane (§4.4): the live events
//! on its ordered lane, and everything on its catch-up lane. The gateway reads
//! them to decide when a member is lagging and when a catch-up frame would
//! pass its cap. Stream frames are not counted: each stream's own credit
//! bounds them (§4.2).

use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Default)]
pub struct Gauges {
    live: AtomicUsize,
    catchup: AtomicUsize,
}

/// The lane a queued or written frame belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lane {
    Control,
    /// A stream frame or `PULSE` on the ordered lane: not gauged.
    Stream,
    /// A live event's fragment on the ordered lane.
    Live,
    CatchUp,
}

impl Gauges {
    pub fn queued(&self, lane: Lane, bytes: usize) {
        match lane {
            Lane::Live => self.live.fetch_add(bytes, Ordering::Relaxed),
            Lane::CatchUp => self.catchup.fetch_add(bytes, Ordering::Relaxed),
            Lane::Control | Lane::Stream => 0,
        };
    }

    pub fn written(&self, lane: Lane, bytes: usize) {
        let gauge = match lane {
            Lane::Live => &self.live,
            Lane::CatchUp => &self.catchup,
            Lane::Control | Lane::Stream => return,
        };
        let _ = gauge.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |held| {
            Some(held.saturating_sub(bytes))
        });
    }

    /// Bytes of live events queued and not yet written.
    pub fn live(&self) -> usize {
        self.live.load(Ordering::Relaxed)
    }

    /// Bytes on the catch-up lane, queued and not yet written.
    pub fn catchup(&self) -> usize {
        self.catchup.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queued_bytes_come_off_as_they_are_written() {
        let g = Gauges::default();
        g.queued(Lane::Live, 3 << 20);
        g.queued(Lane::CatchUp, 100);
        g.queued(Lane::Stream, 64 << 10);
        assert_eq!((g.live(), g.catchup()), (3 << 20, 100));
        g.written(Lane::Live, 1 << 20);
        g.written(Lane::CatchUp, 100);
        g.written(Lane::Control, 9);
        assert_eq!((g.live(), g.catchup()), (2 << 20, 0));
        g.written(Lane::CatchUp, 1);
        assert_eq!(g.catchup(), 0, "never below zero");
    }
}
