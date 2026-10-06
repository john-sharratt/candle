//! Host bookkeeping over every layer at once.
//!
//! A session keeps one KV backing per layer, and structural operations — a view
//! borrowing its parent's prefix, a freed view dropping it — repeat the same
//! per-chunk refcount work in each. At a ~10K-block prefix that is ~20 ms a
//! layer-loop on one thread, paid on every reprojection. The layers are
//! independent (each backing has its own lock and its own chunk table), so the
//! loop splits across scoped threads that are joined before the call returns.
//!
//! Only for work that makes **no device call**: a scoped worker has no CUDA
//! context bound.

use std::thread;

/// Worker threads for one call — enough to split a stack of layers, few enough
/// that spawning them costs nothing against the work.
const MAX_WORKERS: usize = 8;

/// Fewer items than this run on the caller's thread.
const MIN_PARALLEL: usize = 4;

fn workers_for(n: usize) -> usize {
    let cores = thread::available_parallelism().map_or(1, |c| c.get());
    cores.min(MAX_WORKERS).min(n).max(1)
}

/// `f` over every item, results in item order.
pub fn map<B: Sync, T: Send>(items: &[B], f: impl Fn(&B) -> T + Sync) -> Vec<T> {
    let workers = workers_for(items.len());
    if items.len() < MIN_PARALLEL || workers == 1 {
        return items.iter().map(f).collect();
    }
    let per = items.len().div_ceil(workers);
    let f = &f;
    thread::scope(|s| {
        let handles: Vec<_> = items
            .chunks(per)
            .map(|group| s.spawn(move || group.iter().map(f).collect::<Vec<T>>()))
            .collect();
        handles
            .into_iter()
            .flat_map(|h| match h.join() {
                Ok(out) => out,
                Err(panic) => std::panic::resume_unwind(panic),
            })
            .collect()
    })
}

/// Drop every item, the drops split across threads.
pub fn drop_all<T: Send>(items: Vec<T>) {
    let workers = workers_for(items.len());
    if items.len() < MIN_PARALLEL || workers == 1 {
        return;
    }
    let per = items.len().div_ceil(workers);
    let mut items = items;
    thread::scope(|s| {
        while !items.is_empty() {
            let rest = items.split_off(items.len().saturating_sub(per));
            s.spawn(move || drop(rest));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Results come back in item order whether the call splits or not.
    #[test]
    fn results_keep_item_order() {
        for n in [0usize, 1, 3, 4, 7, 48, 61] {
            let items: Vec<usize> = (0..n).collect();
            let out = map(&items, |&i| i * 10);
            assert_eq!(out, (0..n).map(|i| i * 10).collect::<Vec<_>>(), "n = {n}");
        }
    }

    struct Counted(Arc<AtomicUsize>);

    impl Drop for Counted {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Every item is dropped, exactly once, before the call returns.
    #[test]
    fn every_item_is_dropped_before_return() {
        for n in [0usize, 2, 4, 9, 48] {
            let dropped = Arc::new(AtomicUsize::new(0));
            let items: Vec<Counted> = (0..n).map(|_| Counted(dropped.clone())).collect();
            drop_all(items);
            assert_eq!(dropped.load(Ordering::Relaxed), n, "n = {n}");
        }
    }
}
