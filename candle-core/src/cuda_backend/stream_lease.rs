//! The device's one stream, as a thread borrows it to issue work.
//!
//! A device has a single created stream: every launch, copy, allocation and
//! graph launch goes on it, from every thread, so stream order is issue order
//! and nothing needs an event to be ordered against anything else. A wave
//! capture records on that same stream, and a capture takes in whatever is
//! issued on its stream from any thread — so a thread borrows the stream
//! through a [`StreamLease`], which keeps it off the stream while another
//! thread's segment records (`graph::record_gate`).
//!
//! **Hold a lease only while issuing.** A lease another thread holds makes the
//! recording thread run its segments as issued rather than wait, so a lease held
//! across unrelated work costs the wave its graphs for that long.

use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::Arc;

use cudarc::driver::CudaStream;

use super::graph::record_gate;

/// The device's stream, borrowed by this thread for as long as this lives.
///
/// Not `Send`: the lease is counted against the thread that took it, which is
/// the thread whose launches a capture must not take in.
#[must_use = "a lease keeps other threads' captures waiting only while it is held"]
pub struct StreamLease {
    stream: Arc<CudaStream>,
    /// Counted in the record gate, and returned there on drop. A thread
    /// recording on the stream takes uncounted leases: its launches are the
    /// segment's.
    counted: bool,
    _thread_bound: PhantomData<*const ()>,
}

impl StreamLease {
    /// Borrow `stream` for this thread, waiting while another thread's segment
    /// records on it.
    #[track_caller]
    pub(crate) fn take(stream: &Arc<CudaStream>) -> Self {
        Self {
            stream: stream.clone(),
            counted: record_gate::lease_begins(),
            _thread_bound: PhantomData,
        }
    }
}

impl Deref for StreamLease {
    type Target = Arc<CudaStream>;

    fn deref(&self) -> &Self::Target {
        &self.stream
    }
}

impl Drop for StreamLease {
    fn drop(&mut self) {
        if self.counted {
            record_gate::lease_ends();
        }
    }
}
