//! A counting gate: at most `n` holders at once, the rest waiting.
//!
//! A calibration's draft forks from its unit's whole reading, so drafts are
//! held to the width the layer reads at; its thinking and prefill carry only
//! the system prompt and a question, so more units are in flight than drafts,
//! and a unit thinking does not hold a draft's place.

use std::sync::{Condvar, Mutex};

/// At most `n` holders at once.
pub struct Permits {
    free: Mutex<usize>,
    freed: Condvar,
}

/// A held permit, returned when dropped.
pub struct Permit<'a> {
    of: &'a Permits,
}

impl Permits {
    pub fn new(n: usize) -> Self {
        Permits {
            free: Mutex::new(n),
            freed: Condvar::new(),
        }
    }

    /// Wait for a permit and hold it.
    pub fn acquire(&self) -> Permit<'_> {
        let mut free = self.free.lock().unwrap();
        while *free == 0 {
            free = self.freed.wait(free).unwrap();
        }
        *free -= 1;
        Permit { of: self }
    }

    /// Permits not held now.
    pub fn available(&self) -> usize {
        *self.free.lock().unwrap()
    }
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        *self.of.free.lock().unwrap() += 1;
        self.of.freed.notify_one();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    /// A permit is returned when dropped.
    #[test]
    fn a_permit_is_returned_when_dropped() {
        let p = Permits::new(2);
        let a = p.acquire();
        let b = p.acquire();
        assert_eq!(p.available(), 0);
        drop(a);
        assert_eq!(p.available(), 1);
        drop(b);
        assert_eq!(p.available(), 2);
    }

    /// **Never more than `n` hold at once**, however many threads ask.
    #[test]
    fn never_more_than_n_hold_at_once() {
        let p = Permits::new(3);
        let (now, most) = (AtomicUsize::new(0), AtomicUsize::new(0));
        std::thread::scope(|s| {
            for _ in 0..12 {
                s.spawn(|| {
                    for _ in 0..50 {
                        let _held = p.acquire();
                        let n = now.fetch_add(1, Ordering::SeqCst) + 1;
                        most.fetch_max(n, Ordering::SeqCst);
                        std::thread::yield_now();
                        now.fetch_sub(1, Ordering::SeqCst);
                    }
                });
            }
        });
        let most = most.load(Ordering::SeqCst);
        assert!((1..=3).contains(&most), "{most} held at once");
        assert_eq!(p.available(), 3);
    }
}
