//! The wave captures open on this thread, for host-memory calls that hold no
//! device.
//!
//! The driver refuses to pin or unpin host pages (`cuMemHostAlloc`,
//! `cuMemFreeHost`) on a thread that is capturing, and the refusal invalidates
//! the capture: the segment is lost and the next wave step faults. A device
//! method pauses its own wave before such a call
//! ([`CudaDevice::pause_capture`]); pinned staging holds no device, so a wave
//! capture registers its device here for as long as it is open, and
//! [`outside_capture`] pauses whichever of them is recording around the call.

use std::cell::RefCell;

use crate::cuda_backend::CudaDevice;
use crate::Result;

thread_local! {
    /// The devices with a wave capture open on this thread, innermost last.
    static WAVES: RefCell<Vec<CudaDevice>> = const { RefCell::new(Vec::new()) };
}

/// A wave capture on `dev` opened on this thread.
pub(crate) fn enter(dev: CudaDevice) {
    let _ = WAVES.try_with(|w| w.borrow_mut().push(dev));
}

/// The wave capture on `dev` this thread opened has closed.
pub(crate) fn leave(dev: &CudaDevice) {
    let _ = WAVES.try_with(|w| {
        let mut w = w.borrow_mut();
        if let Some(i) = w.iter().rposition(|d| d.id() == dev.id()) {
            w.remove(i);
        }
    });
}

/// Run `f` — a call the driver refuses while this thread records, such as
/// pinning or unpinning host pages — with every wave this thread is recording
/// paused around it: each recorded segment is launched first and recording
/// resumes after `f` returns. A thread recording nothing runs `f` directly.
pub fn outside_capture<R>(f: impl FnOnce() -> R) -> Result<R> {
    // Cloned out, so `f` and the pauses run with no borrow of the registry: a
    // pause launches a segment, and what that segment releases may drop
    // pinned memory, which comes back here.
    let waves = WAVES.try_with(|w| w.borrow().clone()).unwrap_or_default();
    let mut paused = Vec::new();
    for dev in &waves {
        if dev.recording_segment().is_some() {
            paused.push(dev.pause_capture()?);
        }
    }
    let out = f();
    drop(paused);
    Ok(out)
}
