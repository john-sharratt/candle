//! Feature-gated scoped timing for the projection / reproject hot path.
//!
//! With `--features profile`: each `let _g = profile::span("name");` accumulates
//! its scope duration into a thread-local table; `report()` dumps a
//! sorted-by-cost breakdown and clears it. Without the feature every hook is a
//! ZST no-op the optimizer removes — **zero cost**, no branches, no allocation.
//!
//! Call sites stay one line each (`let _g = profile::span("…");`) so the hot
//! path isn't bloated.

#[cfg(feature = "profile")]
mod imp {
    use std::cell::RefCell;
    use std::collections::BTreeMap;
    use std::time::{Duration, Instant};

    thread_local! {
        static ACC: RefCell<BTreeMap<&'static str, (Duration, u64)>> =
            RefCell::new(BTreeMap::new());
    }

    /// RAII timing guard; on drop adds its elapsed time to the named bucket.
    pub struct Span {
        name: &'static str,
        start: Instant,
    }

    impl Drop for Span {
        #[inline]
        fn drop(&mut self) {
            let d = self.start.elapsed();
            ACC.with(|a| {
                let mut a = a.borrow_mut();
                let e = a.entry(self.name).or_insert((Duration::ZERO, 0));
                e.0 += d;
                e.1 += 1;
            });
        }
    }

    #[inline]
    pub fn span(name: &'static str) -> Span {
        Span {
            name,
            start: Instant::now(),
        }
    }

    /// Clear the thread-local table (start of a fresh measurement scope).
    pub fn reset() {
        ACC.with(|a| a.borrow_mut().clear());
    }

    /// Emit the accumulated breakdown (sorted by total descending) at INFO under
    /// `candle_conversation::scheduler::profile`, then clear.
    pub fn report(header: &str) {
        ACC.with(|a| {
            let map = a.borrow();
            if map.is_empty() {
                return;
            }
            let mut rows: Vec<(&&'static str, &(Duration, u64))> = map.iter().collect();
            rows.sort_by(|x, y| y.1 .0.cmp(&x.1 .0));
            let mut out = String::new();
            for (name, (total, count)) in rows {
                let ms = total.as_secs_f64() * 1000.0;
                let avg = ms / (*count as f64).max(1.0);
                out.push_str(&format!(
                    "\n  {name:<30} {ms:>9.2}ms  ×{count:<4} avg {avg:>7.3}ms"
                ));
            }
            tracing::info!(
                target: "candle_conversation::scheduler::profile",
                "── profile [{header}] ──{out}"
            );
        });
        reset();
    }
}

#[cfg(not(feature = "profile"))]
mod imp {
    /// Zero-sized no-op guard; constructing and dropping it compiles to nothing.
    pub struct Span;

    #[inline(always)]
    pub fn span(_name: &'static str) -> Span {
        Span
    }

    #[inline(always)]
    pub fn reset() {}

    #[inline(always)]
    pub fn report(_header: &str) {}
}

pub(crate) use imp::{report, reset, span};
