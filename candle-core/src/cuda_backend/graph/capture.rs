//! Beginning and ending a stream capture — the only place either is done.
//!
//! A capture the record gate does not know about is one its synchronises and
//! module loads walk straight into: the gate waits for the captures it has
//! counted, runs, and the driver refuses the operation and invalidates the
//! capture it never heard of. Every capture is therefore begun here, counted in
//! for as long as it records, and the raw driver call is in `clippy.toml`'s
//! `disallowed-methods`. Measured: three driver-level tests and a benchmark
//! began their own captures, and `candle-core`'s suite lost captures to module
//! loads the gate had cleared.

#![allow(clippy::disallowed_methods)]

use cudarc::driver::{result, sys, CudaStream};

use super::record_gate;
use crate::cuda_backend::WrapErr;
use crate::Result;

/// Start a `ThreadLocal` capture on `stream`, counted into the record gate.
/// Waits while a gated operation runs. Every success is paired with one
/// [`end`].
///
/// # Safety
///
/// `stream` is a created stream — not the legacy null stream, which cannot
/// capture — that is not already capturing, owned by the caller for as long as
/// the capture is open.
pub(crate) unsafe fn begin(stream: &CudaStream) -> Result<()> {
    stream.context().bind_to_thread().w()?;
    record_gate::recording_begins();
    let begun = result::stream::begin_capture(
        stream.cu_stream(),
        sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
    )
    .w();
    if begun.is_err() {
        record_gate::recording_ends();
    }
    begun
}

/// End the capture [`begin`] started on `stream`, and count it out of the gate
/// whether or not the driver hands back a graph.
///
/// # Safety
///
/// `stream` is capturing, begun by [`begin`] on this thread.
pub(crate) unsafe fn end(stream: &CudaStream) -> Result<sys::CUgraph> {
    let graph = result::stream::end_capture(stream.cu_stream()).w();
    record_gate::recording_ends();
    graph
}
