//! Keeps the pinned host buffers that recurrent state is read back into and
//! uploaded from, and releases them at shutdown.
//!
//! The seal buffer is shared by the whole process, since one thread seals at a
//! time. An upload buffer returns to its pool only after its copies have
//! completed, so an import never waits for the forward in flight. Pinned memory
//! does not shrink while an engine runs, so an engine's shutdown frees these
//! buffers, which would otherwise stay pinned for the life of the process.

use super::*;

/// The pinned host buffer every store's seal readback lands in, kept between
/// seals and grown only when a store needs more than it holds.
///
/// One per process rather than one per store: the scheduler seals one turn at a
/// time on its loop thread, so a single buffer serves them all, where a buffer per
/// conversation would pin ~65 MB of host memory for every sequence alive. It never
/// shrinks while an engine runs, and has nothing to shrink from: a store's size is
/// its model's geometry, identical for every sequence, so the buffer settles at one
/// store's size on the first seal. An engine's shutdown frees it
/// ([`release_recurrent_host_buffers`]).
#[cfg(feature = "cuda")]
pub(super) static READBACK: Mutex<Option<PinnedBuf>> = Mutex::new(None);

/// The pinned host buffers stores' restores upload from, kept between imports
/// like [`READBACK`] and for the same reasons, each with the event recorded
/// after its last uploads.
///
/// **A pool, because an import never waits.** A restore's uploads are queued
/// behind the forward still running, so a buffer one restore has just filled is
/// not read until that forward ends, and an admission pass restores several
/// sequences back to back: one buffer made every restore after a pass's first
/// wait out the forward — 213 ms of a 260 ms admission begin on BlackWorld
/// (2026-11-15). An import takes a buffer whose uploads have completed, asked
/// without blocking, and allocates one only when none has, so the pool settles
/// at the number of restores one forward's admissions make.
#[cfg(feature = "cuda")]
pub(super) static UPLOAD: Mutex<Vec<PinnedUpload>> = Mutex::new(Vec::new());

/// One of [`UPLOAD`]'s buffers and the point on the stream past which nothing
/// reads it.
#[cfg(feature = "cuda")]
pub(super) struct PinnedUpload {
    pub(super) buf: PinnedBuf,
    /// Recorded after the last import's uploads from it were enqueued.
    pub(super) done: Option<CudaEvent>,
}

#[cfg(feature = "cuda")]
impl PinnedUpload {
    /// Whether nothing reads the buffer any more, asked without waiting.
    pub(super) fn is_free(&self) -> bool {
        self.done.as_ref().is_none_or(CudaEvent::is_complete)
    }

    /// Wait until the last import's uploads have read the buffer.
    pub(super) fn settle(&mut self) -> Result<()> {
        if let Some(done) = self.done.take() {
            done.synchronize()
                .map_err(|e| Error::Msg(format!("recurrent store: upload settle: {e:?}")))?;
        }
        Ok(())
    }
}

/// Free the pinned buffers a store's seal reads back into ([`READBACK`]) and
/// its restore uploads from ([`UPLOAD`]), unpinning their pages.
///
/// The buffers are process-wide, so they outlive the engine whose seals and
/// restores sized them: without this, a shut-down engine leaves one store's
/// whole recurrent state — 19.3 MiB on the 0.8B, ~65 MB on the 35B-A3B —
/// pinned twice for the life of the process. Called once the engine's
/// scheduler, the only thread that seals or restores, has joined; the next
/// engine's first seal and first restore allocate them again.
pub fn release_recurrent_host_buffers() {
    #[cfg(feature = "cuda")]
    {
        release_readback_in(&READBACK);
        let held = std::mem::take(&mut *UPLOAD.lock().unwrap_or_else(|e| e.into_inner()));
        for mut up in held {
            // Uploads still reading it would read freed pages.
            if let Err(e) = up.settle() {
                tracing::warn!("{e}");
            }
        }
    }
}

/// Empty `slot`, freeing its buffer outside the lock (the drop is a
/// `cuMemFreeHost`). A poisoned lock still releases: the buffer is only ever
/// replaced whole, so a panicking holder cannot leave it half-written.
#[cfg(feature = "cuda")]
pub(super) fn release_readback_in(slot: &Mutex<Option<PinnedBuf>>) {
    let held = slot.lock().unwrap_or_else(|e| e.into_inner()).take();
    drop(held);
}
