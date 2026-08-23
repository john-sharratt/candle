//! Zero-cost profiling infrastructure for hot-path instrumentation.
//!
//! Everything here compiles away completely when the `profile` feature is off:
//! the types become zero-sized and every method an `#[inline(always)]` no-op
//! that LLVM eliminates, including the parameters at call sites.
//!
//! # Two timers, and which to reach for
//!
//! | | measures | cost | use for |
//! |---|---|---|---|
//! | [`span`] / [`pipeline_record`] | host wall time | an `Instant` + a `RefCell` borrow | host-side work, waits, anything not enqueued |
//! | [`gpu_span`] | GPU time, stream-ordered | two enqueued `cuEventRecord` | device work |
//!
//! The distinction matters because CUDA work is asynchronous: a host timer
//! around a launch measures the *launch*, not the work. The obvious fix — drain
//! the device at the span boundary — is the one this module deliberately does
//! NOT do, because it serialises the pipeline and hides the inter-op gaps that
//! are frequently the thing being hunted. [`gpu_span`] brackets work with CUDA
//! events instead and reads them later, at a boundary that already
//! synchronises. See [`gpu`] for the full argument.
//!
//! # Feature interaction
//!
//! `profile` and `nvtx` are independent on purpose. `profile` produces the
//! summary table; `nvtx` annotates the trace so
//! `nsys stats --report nvtx_kern_sum` attributes each kernel to the span that
//! launched it. A capture run usually wants the second without paying for the
//! first. [`gpu_span`] additionally needs `cuda` — without it there is no stream
//! to record into — and degrades to a no-op rather than a compile error, so call
//! sites need no `cfg` of their own.
//!
//! # Usage
//!
//! ```rust,ignore
//! let s = span("decode:select");        // host time + NVTX range
//! let g = gpu_span("decode:kernel", x.device());  // GPU time, no sync
//! // ... work ...
//! g.end();
//! s.end();
//! // at a boundary that already synchronises:
//! gpu_drain_blocking();
//! ```

mod accumulator;
mod gpu;
mod mark;
mod pipeline;
mod span;

#[cfg(test)]
mod tests;

pub use accumulator::{ProfileAccumulator, ProfileSnapshot};
pub use gpu::{gpu_drain, gpu_drain_blocking, gpu_span, gpu_span_if, gpu_span_phase, GpuSpan};
pub use mark::{profile_now, ProfileMark};
pub use pipeline::{pipeline_record, pipeline_record_duration, pipeline_snapshot_and_reset};
pub use span::{span, span_if, Span};
