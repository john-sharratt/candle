//! The pinned staging buffers that uploads are copied through, and the rule for
//! which one an upload may use.
//!
//! A staging buffer is write-combined pinned memory that the CPU fills and the
//! copy engine reads. An event recorded after its last copy lets its reuse be
//! fenced without draining the stream. An upload reuses the buffer the previous
//! upload used while that buffer is free, and falls to the other one only when
//! it is not. The only wait is when both are still in flight, and then it waits
//! for the older one.

use super::*;

/// One pinned staging buffer and the event recorded after the copies out of it.
pub(super) struct Staging {
    /// Write-combined pinned memory — the CPU only writes it, the copy engine
    /// only reads it. Grow-only; replaced only once no copy reads it.
    pub(super) buf: PinnedBuf,
    /// Recorded after the most recent copies out of [`Self::buf`]; `None` until
    /// the first. Created once and re-recorded: `cuEventRecord` overwrites the
    /// previous recording, and a create/destroy pair per upload would put two
    /// driver object lifecycles on every 32-token boundary of every sequence.
    pub(super) done: Option<CudaEvent>,
}

impl Staging {
    pub(super) fn empty() -> Self {
        Self {
            buf: Self::no_buf(),
            done: None,
        }
    }

    /// The zero-length placeholder a `Staging` holds before its first upload.
    /// `alloc_owned(0)` returns a zero-len `Bump` variant — no CUDA call.
    pub(super) fn no_buf() -> PinnedBuf {
        PinnedBuf::alloc_owned(0).expect("zero-len PinnedBuf alloc cannot fail")
    }

    /// Take the pinned buffer out, leaving the placeholder.
    ///
    /// The caller must have fenced this staging's copies first: the recycler hands
    /// the same pages to the next taker, so a copy still reading them would read
    /// another owner's bytes.
    pub(super) fn take_buf(&mut self) -> PinnedBuf {
        std::mem::replace(&mut self.buf, Self::no_buf())
    }

    /// Whether a copy out of this buffer may still be waiting in the stream.
    /// A non-blocking query — the host never waits here.
    pub(super) fn busy(&self) -> bool {
        self.done.as_ref().is_some_and(|event| !event.is_complete())
    }
}

/// Which staging buffer the next upload goes through, and whether it must first
/// wait for that buffer's previous copy: `busy[i]` is whether a queued copy may
/// still read buffer `i`, and `last` is the buffer the previous upload used.
///
/// The previous upload's buffer when it is free — so an upload that never
/// collides with a queued copy (every decode-path upload) keeps reusing one
/// buffer, and the second is never grown. Otherwise the other one; if that is
/// busy too it is the older of the two, and the upload waits for it — the only
/// wait, and it takes two uploads queued behind unfinished work.
pub(super) fn choose_staging(busy: [bool; 2], last: Option<usize>) -> (usize, bool) {
    match last {
        None => (0, false),
        Some(l) if !busy[l] => (l, false),
        Some(l) => {
            let other = 1 - l;
            (other, busy[other])
        }
    }
}
