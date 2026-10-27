//! The pinned buffers a guest checkpoint crosses to the card through.
//!
//! **Pinned, so the copy is the link's and not the host's.** A copy out of
//! pageable memory is staged by the driver through a bounce buffer of its own,
//! at about half the link, and cudarc's slice copy then synchronises the whole
//! stream before it returns — so every tensor of a checkpoint waited for every
//! kernel before it, and the disk sat idle while the card worked and the card
//! while the disk did. A copy out of pinned memory is one DMA and returns at
//! once.
//!
//! **Two, so the disk and the card overlap.** While the card drains one
//! window's copies and repacks, the next window is read off the disk into the
//! other buffer. A buffer is refilled only once the event recorded after the
//! copies out of it has passed — one wait per window, not per tensor.
//!
//! **For the length of a load, not resident.** The buffers are taken when a
//! load starts and freed when it ends; a member between draws holds none of
//! the checkpoint anywhere, host or card.

use std::sync::Arc;

use candle::direct_io::round_up_sector;
use candle::quantized::cuda::{alloc_host_mapped, HostMappedAlloc};
use cudarc::driver::{CudaEvent, CudaStream};

/// How many buffers a load stages through: one filling, one draining.
pub const SLOTS: usize = 2;

struct Slot {
    ptr: *mut u8,
    _alloc: HostMappedAlloc,
    /// Recorded after the last copy issued out of this buffer.
    drained: Option<CudaEvent>,
}

/// [`SLOTS`] pinned buffers, taken in turn.
pub struct Staging {
    slots: Vec<Slot>,
    len: usize,
    next: usize,
}

// SAFETY: the buffers are pinned host memory, valid from any thread for as
// long as the allocations they view, which `Staging` owns.
unsafe impl Send for Staging {}

impl Staging {
    /// [`SLOTS`] buffers of at least `len` bytes each, sector-rounded so a
    /// direct read may land in them.
    pub fn new(len: usize) -> candle::Result<Self> {
        let len = round_up_sector(len.max(1));
        let slots = (0..SLOTS)
            .map(|_| {
                let (ptr, _, alloc) = alloc_host_mapped(len)?;
                Ok(Slot {
                    ptr,
                    _alloc: alloc,
                    drained: None,
                })
            })
            .collect::<candle::Result<Vec<_>>>()?;
        Ok(Self {
            slots,
            len,
            next: 0,
        })
    }

    /// The next buffer in turn and its index, once every copy issued out of it
    /// before has landed.
    pub fn acquire(&mut self) -> candle::Result<(usize, &mut [u8])> {
        let at = self.next;
        self.next = (self.next + 1) % self.slots.len();
        let slot = &mut self.slots[at];
        if let Some(event) = slot.drained.take() {
            event.synchronize().map_err(candle::Error::wrap)?;
        }
        // SAFETY: `ptr` is `len` bytes of pinned memory this `Staging` owns, and
        // no copy out of it is in flight (waited for above).
        Ok((at, unsafe {
            std::slice::from_raw_parts_mut(slot.ptr, self.len)
        }))
    }

    /// Every copy out of buffer `at` has been issued on `stream`: the buffer is
    /// refilled only once they have landed.
    pub fn issued(&mut self, at: usize, stream: &Arc<CudaStream>) -> candle::Result<()> {
        self.slots[at].drained = Some(stream.record_event(None).map_err(candle::Error::wrap)?);
        Ok(())
    }
}

/// A copy may still be reading a buffer when a load fails part-way; the memory
/// is not freed under it.
impl Drop for Staging {
    fn drop(&mut self) {
        for slot in &mut self.slots {
            if let Some(event) = slot.drained.take() {
                let _ = event.synchronize();
            }
        }
    }
}
