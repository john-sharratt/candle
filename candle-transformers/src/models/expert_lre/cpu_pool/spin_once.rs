//! A once-cell whose waiters spin.
//!
//! A layer's participants all start on the same expert, and the first to reach
//! it builds the input its tasks share — microseconds of work. `OnceLock` puts
//! the others to sleep on an OS wait meanwhile (`WaitOnAddress` on Windows), and
//! each pays a wake of tens of microseconds, per expert, per layer: the pool
//! computed a decode layer at about half the throughput its participants'
//! matvecs allow. Waiters here spin a few times, then `tpause`, for the
//! microseconds the build takes (`backoff`).

use std::cell::UnsafeCell;
use std::mem::{forget, MaybeUninit};
use std::sync::atomic::{AtomicU8, Ordering};
use std::thread;

use super::backoff::{Backoff, INTRA_LAYER_SPINS};

/// A waiter's wait past its hot spins: `tpause` for ~0.5 µs, against a build
/// of a few microseconds.
const WAIT_PAUSE_CYCLES: u64 = 1_500;

const EMPTY: u8 = 0;
const BUILDING: u8 = 1;
const READY: u8 = 2;
/// The build panicked: there is no value, and none will come.
const POISONED: u8 = 3;

/// Marks the cell poisoned if the build it guards unwinds, so the threads
/// waiting on it panic too rather than spinning for a value that never comes.
struct Building<'a>(&'a AtomicU8);

impl Drop for Building<'_> {
    fn drop(&mut self) {
        if thread::panicking() {
            self.0.store(POISONED, Ordering::Release);
        }
    }
}

/// A value built once, by whichever thread asks first.
pub struct SpinOnce<T> {
    state: AtomicU8,
    value: UnsafeCell<MaybeUninit<T>>,
}

// SAFETY: the value is written once, by the thread that moved the state from
// EMPTY to BUILDING, before its release store of READY; every reader reads it
// only after an acquire load of READY.
unsafe impl<T: Send + Sync> Sync for SpinOnce<T> {}
unsafe impl<T: Send> Send for SpinOnce<T> {}

impl<T> Default for SpinOnce<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> SpinOnce<T> {
    pub const fn new() -> Self {
        Self {
            state: AtomicU8::new(EMPTY),
            value: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }

    /// The value, once built.
    pub fn get(&self) -> Option<&T> {
        (self.state.load(Ordering::Acquire) == READY)
            // SAFETY: READY is stored only after the value is written.
            .then(|| unsafe { (*self.value.get()).assume_init_ref() })
    }

    /// The value, building it with `f` if no thread has begun to; a thread
    /// that finds another building it spins until it is ready.
    ///
    /// # Panics
    /// If `f` panics — in the builder, and in every thread waiting on it or
    /// asking after it, since no value will come.
    pub fn get_or_init(&self, f: impl FnOnce() -> T) -> &T {
        if let Some(v) = self.get() {
            return v;
        }
        if self
            .state
            .compare_exchange(EMPTY, BUILDING, Ordering::Acquire, Ordering::Acquire)
            .is_ok()
        {
            let building = Building(&self.state);
            // SAFETY: the one thread past the compare-exchange writes it, and
            // no reader reads it before READY.
            unsafe { (*self.value.get()).write(f()) };
            forget(building);
            self.state.store(READY, Ordering::Release);
        } else {
            let mut backoff = Backoff::new(INTRA_LAYER_SPINS);
            loop {
                match self.state.load(Ordering::Acquire) {
                    READY => break,
                    POISONED => panic!("spin once: the build of this value panicked"),
                    _ => backoff.pause(WAIT_PAUSE_CYCLES),
                }
            }
        }
        // SAFETY: READY.
        unsafe { (*self.value.get()).assume_init_ref() }
    }

    /// Sets the value if no thread has; returns whether this call set it.
    pub fn set(&self, v: T) -> bool {
        if self
            .state
            .compare_exchange(EMPTY, BUILDING, Ordering::Acquire, Ordering::Acquire)
            .is_err()
        {
            return false;
        }
        // SAFETY: as `get_or_init`.
        unsafe { (*self.value.get()).write(v) };
        self.state.store(READY, Ordering::Release);
        true
    }
}

impl<T> Drop for SpinOnce<T> {
    fn drop(&mut self) {
        if *self.state.get_mut() == READY {
            // SAFETY: READY means written, and `&mut self` means no reader.
            unsafe { self.value.get_mut().assume_init_drop() };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;
    use std::time::Duration;

    /// A build that panics poisons the cell: the threads waiting on it, and
    /// any that ask after it, panic rather than spin, `get` stays empty, and
    /// the cell drops nothing.
    #[test]
    fn a_panicking_build_poisons_the_waiters() {
        let cell = Arc::new(SpinOnce::<Vec<u32>>::new());
        let builder = {
            let cell = cell.clone();
            thread::spawn(move || {
                cell.get_or_init(|| {
                    thread::sleep(Duration::from_millis(30));
                    panic!("the build fails")
                })
                .len()
            })
        };
        thread::sleep(Duration::from_millis(5));
        let waiters: Vec<_> = (0..4)
            .map(|_| {
                let cell = cell.clone();
                thread::spawn(move || cell.get_or_init(|| vec![1]).len())
            })
            .collect();
        assert!(builder.join().is_err(), "the builder's own panic");
        for w in waiters {
            assert!(w.join().is_err(), "a waiter panics, not spins");
        }
        assert!(cell.get().is_none());
        let late = catch_unwind(AssertUnwindSafe(|| cell.get_or_init(|| vec![2]).len()));
        assert!(late.is_err(), "a later caller panics too");
        assert!(!cell.set(vec![3]), "a poisoned cell takes no value");
    }

    /// Many threads at once: one builds, every one gets the same value.
    #[test]
    fn one_builds_and_all_read_it() {
        for _ in 0..200 {
            let cell = Arc::new(SpinOnce::<Vec<u32>>::new());
            let builds = Arc::new(AtomicUsize::new(0));
            let handles: Vec<_> = (0..8)
                .map(|i| {
                    let (cell, builds) = (cell.clone(), builds.clone());
                    thread::spawn(move || {
                        cell.get_or_init(|| {
                            builds.fetch_add(1, Ordering::Relaxed);
                            vec![i, 1, 2]
                        })[1..]
                            .to_vec()
                    })
                })
                .collect();
            for h in handles {
                assert_eq!(h.join().unwrap(), vec![1, 2]);
            }
            assert_eq!(builds.load(Ordering::Relaxed), 1);
        }
    }

    /// `get` is empty until set; a second set is refused and the first stands;
    /// the value is dropped with the cell.
    #[test]
    fn set_get_and_drop() {
        let drops = Arc::new(AtomicUsize::new(0));
        struct D(Arc<AtomicUsize>, u32);
        impl Drop for D {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }
        {
            let cell = SpinOnce::new();
            assert!(cell.get().is_none());
            assert!(cell.set(D(drops.clone(), 7)));
            assert!(!cell.set(D(drops.clone(), 8)));
            assert_eq!(
                drops.load(Ordering::Relaxed),
                1,
                "the refused value dropped"
            );
            assert_eq!(cell.get().unwrap().1, 7);
        }
        assert_eq!(drops.load(Ordering::Relaxed), 2);
        let empty: SpinOnce<D> = SpinOnce::new();
        drop(empty);
        assert_eq!(
            drops.load(Ordering::Relaxed),
            2,
            "an empty cell drops nothing"
        );
    }
}
