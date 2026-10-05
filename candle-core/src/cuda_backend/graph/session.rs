//! One open capture on its own stream — the driver-level tests' recorder.

use super::{CaptureStream, ComputeStream, GraphError, GraphExec};
use crate::cuda_backend::WrapErr;
use crate::Result;
use cudarc::driver::{result, sys, CudaStream};
use std::sync::Arc;

/// An open capture on a [`CaptureStream`], recording every launch issued on
/// [`Self::stream`] until [`Self::finish`].
///
/// Holds the stream by `&mut` for its whole life. Dropping it unfinished ends
/// the capture and discards what was recorded, so an error part-way through a
/// region leaves the stream usable for the next one.
pub struct CaptureSession<'s> {
    cap: &'s mut CaptureStream,
    open: bool,
}

impl<'s> CaptureSession<'s> {
    /// Start recording on `cap`, in `ThreadLocal` mode.
    pub fn begin(cap: &'s mut CaptureStream) -> Result<Self> {
        cap.stream.context().bind_to_thread().w()?;
        // SAFETY: a stream this value owns, not already capturing — a session
        // holds it by `&mut`, so no second one can be open on it.
        unsafe {
            result::stream::begin_capture(
                cap.stream.cu_stream(),
                sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
            .w()?;
        }
        Ok(Self { cap, open: true })
    }

    /// The stream launches are recorded on while this session is open.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.cap.stream
    }

    /// End the capture, instantiate it, and upload it ahead of its first launch
    /// on `on`.
    ///
    /// `expected_nodes` is how many launches the caller issued on
    /// [`Self::stream`]; a graph that recorded any other number is refused
    /// ([`GraphError::NodeCount`]), because a launcher that skipped its launch
    /// would otherwise be missing from every replay without a word.
    pub fn finish(mut self, on: &ComputeStream, expected_nodes: usize) -> Result<GraphExec> {
        self.open = false;
        let graph = self.end()?;
        if graph.is_null() {
            return Err(crate::Error::wrap(GraphError::NoGraph));
        }
        // SAFETY: `graph` is the capture's own graph, handed to `instantiate`,
        // which destroys it on every path.
        unsafe { GraphExec::instantiate(graph, on, expected_nodes) }
    }

    /// End the capture and fold it into `slot`: an empty slot is instantiated,
    /// an occupied one updated in place ([`GraphExec::update`]). Returns `true`
    /// when an existing executable was updated without re-instantiation.
    pub fn finish_into(
        mut self,
        slot: &mut Option<GraphExec>,
        on: &ComputeStream,
        expected_nodes: usize,
    ) -> Result<bool> {
        self.open = false;
        let graph = self.end()?;
        if graph.is_null() {
            return Err(crate::Error::wrap(GraphError::NoGraph));
        }
        match slot {
            // SAFETY: the capture's own graph; `update` destroys it.
            Some(exec) => unsafe { exec.update(graph, on, expected_nodes) },
            None => {
                // SAFETY: the capture's own graph; `instantiate` destroys it.
                *slot = Some(unsafe { GraphExec::instantiate(graph, on, expected_nodes)? });
                Ok(false)
            }
        }
    }

    fn end(&self) -> Result<sys::CUgraph> {
        self.cap.stream.context().bind_to_thread().w()?;
        // SAFETY: this session began the capture on this stream.
        unsafe { result::stream::end_capture(self.cap.stream.cu_stream()).w() }
    }
}

impl Drop for CaptureSession<'_> {
    fn drop(&mut self) {
        if !self.open {
            return;
        }
        // Ending is what returns the stream to normal operation; the recorded
        // graph, if the capture was still valid, is discarded.
        if let Ok(graph) = self.end() {
            if !graph.is_null() {
                // SAFETY: a graph this capture produced and nothing else owns.
                let _ = unsafe { result::graph::destroy(graph) };
            }
        }
    }
}
