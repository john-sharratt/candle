//! Feature-gated scoped timing for the scheduler's main loop.
//!
//! With `--features profile`: each `let _g = profile::span("name");` accumulates
//! its scope duration into the **process-wide** pipeline profiler in
//! `candle_transformers::models::profile`. Without the feature every hook is a
//! ZST no-op the optimizer removes — no branches, no allocation.
//!
//! Recording into the transformers accumulator rather than a table of our own is
//! what makes a single readout answer the whole question. The scheduler's
//! housekeeping and the model's kernels are two halves of one wave, and a
//! breakdown that could show either but never both is exactly the shape that
//! sends you looking in the wrong half: a wave whose time is neither in the
//! forward nor in any phase this crate names is only visibly so when both sets
//! of spans are in the same table, in the same units, over the same interval.
//!
//! Call sites stay one line each (`let _g = profile::span("…");`).

#[cfg(feature = "profile")]
mod imp {
    use candle_transformers::models::profile::pipeline_record_duration;
    use std::time::Instant;

    /// RAII timing guard; on drop adds its elapsed time to the named bucket.
    pub struct Span {
        name: &'static str,
        start: Instant,
    }

    impl Drop for Span {
        #[inline]
        fn drop(&mut self) {
            pipeline_record_duration(self.name, self.start.elapsed(), 1);
        }
    }

    impl Span {
        /// Close the span here rather than at end of scope.
        #[inline]
        pub fn end(self) {
            drop(self);
        }
    }

    #[inline]
    pub fn span(name: &'static str) -> Span {
        Span {
            name,
            start: Instant::now(),
        }
    }
}

#[cfg(not(feature = "profile"))]
mod imp {
    /// Zero-sized no-op guard; constructing and dropping it compiles to nothing.
    pub struct Span;

    impl Span {
        /// Consumes the (zero-sized) span and does nothing. Split from the
        /// timing arm because `Span` only implements `Drop` when there is
        /// something to record, and `drop()` on a non-`Drop` type is a lint
        /// rather than a no-op.
        #[inline(always)]
        pub fn end(self) {}
    }

    #[inline(always)]
    pub fn span(_name: &'static str) -> Span {
        Span
    }
}

pub(crate) use imp::span;
