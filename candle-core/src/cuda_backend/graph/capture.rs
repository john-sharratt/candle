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

/// Start a `ThreadLocal` capture on `stream` — a stream of its own, which no
/// other thread issues on — counted into the record gate. Waits while a gated
/// operation runs. Every success is paired with one [`end`]. For the
/// driver-level tests; a wave records through [`try_begin_on_stream`].
///
/// # Safety
///
/// `stream` is a created stream that is not already capturing, owned by the
/// caller for as long as the capture is open.
#[cfg(test)]
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

/// Start a `ThreadLocal` capture on a device's stream — the one every thread
/// issues on — counted into the record gate, unless another thread holds a
/// stream lease: `Ok(false)`, with nothing begun, and the caller runs the
/// segment as issued. Every `Ok(true)` is paired with one [`end`].
///
/// # Safety
///
/// `stream` is the device's created stream, not already capturing, and stays
/// the caller's to end for as long as the capture is open.
pub(crate) unsafe fn try_begin_on_stream(stream: &CudaStream) -> Result<bool> {
    stream.context().bind_to_thread().w()?;
    if !record_gate::try_stream_recording_begins() {
        return Ok(false);
    }
    let begun = result::stream::begin_capture(
        stream.cu_stream(),
        sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
    )
    .w();
    if begun.is_err() {
        record_gate::recording_ends();
    }
    begun.map(|()| true)
}

/// End the capture [`begin`] started on `stream`, and count it out of the gate
/// whether or not the driver hands back a graph.
///
/// # Safety
///
/// `stream` is capturing, begun on this thread.
#[cfg(test)]
pub(crate) unsafe fn end(stream: &CudaStream) -> Result<sys::CUgraph> {
    let graph = result::stream::end_capture(stream.cu_stream()).w();
    record_gate::recording_ends();
    graph
}

/// End the capture [`try_begin_on_stream`] started on a device's stream,
/// leaving it counted in the gate until `counted` drops: the segment's graph is
/// launched onto the stream in between, and a lease another thread is waiting
/// for must not be granted ahead of that launch, or its work would land ahead
/// of the segment it waited on.
///
/// # Safety
///
/// `stream` is capturing, begun on this thread by [`try_begin_on_stream`].
pub(crate) unsafe fn end_on_stream(
    stream: &CudaStream,
    counted: &record_gate::StreamRecording,
) -> Result<sys::CUgraph> {
    let _ = counted;
    result::stream::end_capture(stream.cu_stream()).w()
}
