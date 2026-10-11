//! The gateway's `RECEIPT` pacing (§4.3): both ends count the frames a member
//! sends on a session, 1, 2, 3 and on. The gateway receipts the count it has
//! relayed — a frame counts once it sits in every recipient's queue — every
//! 100 ms or every 1 MiB, whichever comes first.

use std::time::{Duration, Instant};

pub const EVERY: Duration = Duration::from_millis(100);
pub const EVERY_BYTES: usize = 1 << 20;

#[derive(Debug)]
pub struct ReceiptPacer {
    relayed: u64,
    receipted: u64,
    bytes_since: usize,
    last: Instant,
}

impl ReceiptPacer {
    pub fn new(now: Instant) -> Self {
        Self {
            relayed: 0,
            receipted: 0,
            bytes_since: 0,
            last: now,
        }
    }

    /// The member's next frame, of `bytes`, has been relayed. Returns the
    /// `RECEIPT` to send now, if the byte threshold was reached.
    pub fn relayed(&mut self, bytes: usize, now: Instant) -> Option<u64> {
        self.relayed += 1;
        self.bytes_since += bytes;
        if self.bytes_since >= EVERY_BYTES {
            return self.issue(now);
        }
        None
    }

    /// The timer: a `RECEIPT` when 100 ms have passed with frames unreceipted.
    pub fn tick(&mut self, now: Instant) -> Option<u64> {
        if self.relayed > self.receipted && now.duration_since(self.last) >= EVERY {
            return self.issue(now);
        }
        None
    }

    fn issue(&mut self, now: Instant) -> Option<u64> {
        self.receipted = self.relayed;
        self.bytes_since = 0;
        self.last = now;
        Some(self.relayed)
    }
}

/// The sender's count of the frames it has put on a session.
#[derive(Debug, Default)]
pub struct FrameCounter {
    sent: u64,
}

impl FrameCounter {
    /// Count one more frame sent, returning its number.
    pub fn count_one(&mut self) -> u64 {
        self.sent += 1;
        self.sent
    }

    pub fn sent(&self) -> u64 {
        self.sent
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_receipt_goes_every_mebibyte_or_every_100_ms() {
        let t0 = Instant::now();
        let mut p = ReceiptPacer::new(t0);
        assert_eq!(p.relayed(600 << 10, t0), None);
        assert_eq!(p.relayed(500 << 10, t0), Some(2));
        assert_eq!(p.tick(t0 + EVERY), None, "nothing new since");
        assert_eq!(p.relayed(10, t0 + Duration::from_millis(150)), None);
        assert_eq!(p.tick(t0 + Duration::from_millis(150)), Some(3));
        assert_eq!(p.relayed(10, t0 + Duration::from_millis(160)), None);
        assert_eq!(p.tick(t0 + Duration::from_millis(200)), None);
        assert_eq!(p.tick(t0 + Duration::from_millis(250)), Some(4));
    }

    #[test]
    fn frames_count_from_one() {
        let mut c = FrameCounter::default();
        assert_eq!((c.count_one(), c.count_one(), c.sent()), (1, 2, 2));
    }
}
