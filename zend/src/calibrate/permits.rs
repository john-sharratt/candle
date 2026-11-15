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
    pub const fn new(n: usize) -> Self {
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

    /// One more place for a holder that already holds `held`: waited for when
    /// it holds none, taken only if one is free when it holds some.
    ///
    /// A holder that waited while it held places would keep them from the
    /// holders it waits on, and holders that each keep some and want more
    /// deadlock with the gate full. Taking without waiting, a holder with
    /// places goes on to finish the work they cover, which hands them back.
    pub fn next_place(&self, held: usize) -> Option<Permit<'_>> {
        if held == 0 {
            return Some(self.acquire());
        }
        let mut free = self.free.lock().unwrap();
        if *free == 0 {
            return None;
        }
        *free -= 1;
        Some(Permit { of: self })
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

    /// **A holder with places never waits for another**, and one with none
    /// does: the gate full, a holder holding some is told there is none.
    #[test]
    fn a_holder_with_places_is_never_made_to_wait() {
        let p = Permits::new(2);
        let first = p.next_place(0).expect("an empty holder waits and gets one");
        let second = p.next_place(1).expect("one is free");
        assert!(p.next_place(2).is_none(), "the gate is full");
        drop(first);
        assert!(p.next_place(1).is_some());
        drop(second);
    }

    /// **Units sharing the gate never hold more than it allows, and all
    /// finish**: each tops its window up without waiting while it holds
    /// places, waits only when it holds none, and gives a place back as each
    /// of its turns ends.
    #[test]
    fn units_sharing_the_gate_stay_within_it_and_all_finish() {
        let p = Permits::new(4);
        let (now, most, done) = (
            AtomicUsize::new(0),
            AtomicUsize::new(0),
            AtomicUsize::new(0),
        );
        std::thread::scope(|s| {
            for _ in 0..9 {
                s.spawn(|| {
                    let (mut submitted, mut held) = (0, Vec::new());
                    while submitted < 30 || !held.is_empty() {
                        while submitted < 30 {
                            let Some(place) = p.next_place(held.len()) else {
                                break;
                            };
                            let n = now.fetch_add(1, Ordering::SeqCst) + 1;
                            most.fetch_max(n, Ordering::SeqCst);
                            held.push(place);
                            submitted += 1;
                        }
                        std::thread::yield_now();
                        // Counted out before the place goes back, so the
                        // count never trails a place another thread holds.
                        if let Some(place) = held.pop() {
                            now.fetch_sub(1, Ordering::SeqCst);
                            done.fetch_add(1, Ordering::SeqCst);
                            drop(place);
                        }
                    }
                });
            }
        });
        assert_eq!(done.load(Ordering::SeqCst), 9 * 30);
        let most = most.load(Ordering::SeqCst);
        assert!((1..=4).contains(&most), "{most} held at once");
        assert_eq!(p.available(), 4);
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
