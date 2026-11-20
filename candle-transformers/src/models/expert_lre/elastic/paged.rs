//! The warm tier's pageable part as equal chunks of ordinary heap memory
//! (`docs/elastic_warm_tier.md` §5.1): what the elastic tier grows and
//! shrinks. The pinned part beside it is sized once at startup and never
//! touched after.
//!
//! Paged slot `p` is slot `p % per` of chunk `p / per`, so growing appends a
//! chunk and shrinking frees the last one: the slots below never move. Nothing
//! on the device addresses a pageable slot — the stager copies one into the pad
//! and the CPU pool reads it on the host — so a chunk is freed once the stager
//! has drained its readers, with no kernel or driver call.

use std::alloc::{alloc, dealloc, Layout};
use std::sync::{RwLock, RwLockReadGuard};

use candle::direct_io::DIRECT_IO_SECTOR;

/// A chunk's size, before it is cut to whole slots.
pub(crate) const CHUNK_BYTES: usize = 1024 * 1024 * 1024;

/// Slots in a chunk of `stride`-byte slots for a tier of at most `most`
/// pageable slots: a [`CHUNK_BYTES`]'s worth, but no more than the tier can
/// hold, and at least one.
pub(crate) fn chunk_slots(stride: usize, most: usize) -> usize {
    (CHUNK_BYTES / stride.max(1)).clamp(1, most.max(1))
}

/// One sector-aligned allocation of `per` slots.
struct Chunk {
    base: *mut u8,
    layout: Layout,
}

impl Chunk {
    /// `bytes` uninitialised — the grow fills every slot before any is
    /// published. `None` when the allocator refuses.
    fn new(bytes: usize) -> Option<Self> {
        let layout = Layout::from_size_align(bytes, DIRECT_IO_SECTOR)
            .ok()
            .filter(|l| l.size() > 0)?;
        // SAFETY: `layout` has a non-zero size.
        let base = unsafe { alloc(layout) };
        (!base.is_null()).then_some(Self { base, layout })
    }
}

impl Drop for Chunk {
    fn drop(&mut self) {
        // SAFETY: `base` came from `alloc(layout)`.
        unsafe { dealloc(self.base, self.layout) };
    }
}

// SAFETY: plain owned memory; who may write a slot, and when a chunk may go,
// is the stager's protocol (`stager.rs`, the paged shrink).
unsafe impl Send for Chunk {}
unsafe impl Sync for Chunk {}

pub(crate) struct PagedChunks {
    stride: usize,
    per: usize,
    chunks: RwLock<Vec<Chunk>>,
}

impl PagedChunks {
    /// Empty, for chunks of `per` slots of `stride` bytes.
    pub(crate) fn new(stride: usize, per: usize) -> Self {
        Self {
            stride,
            per: per.max(1),
            chunks: RwLock::new(Vec::new()),
        }
    }

    fn read(&self) -> RwLockReadGuard<'_, Vec<Chunk>> {
        self.chunks.read().unwrap_or_else(|e| e.into_inner())
    }

    /// Slots a chunk holds.
    pub(crate) fn per_chunk(&self) -> usize {
        self.per
    }

    /// Bytes a chunk holds.
    pub(crate) fn chunk_bytes(&self) -> u64 {
        (self.per * self.stride) as u64
    }

    pub(crate) fn chunks(&self) -> usize {
        self.read().len()
    }

    /// Slots held, every chunk's.
    pub(crate) fn slots(&self) -> usize {
        self.chunks() * self.per
    }

    /// Bytes held, every chunk's.
    pub(crate) fn held_bytes(&self) -> u64 {
        self.chunks() as u64 * self.chunk_bytes()
    }

    /// Append one chunk. `false` when the allocator refuses it.
    pub(crate) fn grow(&self) -> bool {
        let Some(chunk) = Chunk::new(self.per * self.stride) else {
            return false;
        };
        self.chunks
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .push(chunk);
        true
    }

    /// Free the last chunk — only once nothing reads it: the stager has
    /// drained its slots.
    pub(crate) fn pop(&self) -> bool {
        self.chunks
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .pop()
            .is_some()
    }

    fn addr(&self, p: usize) -> *mut u8 {
        let chunks = self.read();
        let chunk = chunks.get(p / self.per).unwrap_or_else(|| {
            panic!(
                "warm tier: paged slot {p} is past the {} held",
                chunks.len() * self.per
            )
        });
        // SAFETY: `p % per` slots of `stride` lie inside the chunk.
        unsafe { chunk.base.add((p % self.per) * self.stride) }
    }

    /// Paged slot `p`'s first `len` bytes.
    pub(crate) fn slot(&self, p: usize, len: usize) -> &[u8] {
        assert!(
            len <= self.stride,
            "warm tier: a {len} B read past the stride"
        );
        // SAFETY: inside a held chunk (`addr` checks), which the stager frees
        // only once no reader of it remains.
        unsafe { std::slice::from_raw_parts(self.addr(p), len) }
    }

    /// Paged slot `p`'s bytes, for the writer filling it.
    ///
    /// # Safety
    ///
    /// No other reference to the slot may be live while the returned slice
    /// is, and its chunk must not be popped while it is.
    #[allow(clippy::mut_from_ref)]
    pub(crate) unsafe fn slot_mut(&self, p: usize) -> &mut [u8] {
        std::slice::from_raw_parts_mut(self.addr(p), self.stride)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A chunk is a gigabyte cut to whole slots, never more than the tier
    /// can hold and never none.**
    #[test]
    fn a_chunk_is_a_gigabyte_of_whole_slots() {
        let stride = 1_998_848; // Flash-Next's expert record, 1.91 MiB.
        assert_eq!(chunk_slots(stride, 24_064), 537);
        assert_eq!(chunk_slots(stride, 100), 100, "a small tier is one chunk");
        assert_eq!(chunk_slots(stride, 0), 1);
        assert_eq!(
            chunk_slots(4 * CHUNK_BYTES, 10),
            1,
            "a slot larger than a chunk"
        );
    }

    /// **Chunks grow and shrink at the end**: slots index across them in
    /// order, and what is written to one reads back.
    #[test]
    fn chunks_grow_and_shrink_at_the_end() {
        let c = PagedChunks::new(4096, 4);
        assert!(c.grow() && c.grow());
        assert_eq!((c.chunks(), c.slots(), c.held_bytes()), (2, 8, 8 * 4096));
        for p in 0..8 {
            // SAFETY: one slot at a time, nothing else borrows it.
            unsafe { c.slot_mut(p) }.fill(p as u8 + 1);
        }
        for p in 0..8 {
            assert_eq!(c.slot(p, 4096), &[p as u8 + 1; 4096][..], "slot {p}");
        }
        assert_eq!(c.addr(5) as usize - c.addr(4) as usize, 4096);
        assert!(c.pop());
        assert_eq!(c.slots(), 4);
        assert_eq!(c.slot(3, 1)[0], 4, "the slots below never move");
        assert!(c.pop() && !c.pop());
    }

    /// A slot past the held chunks is refused, not read.
    #[test]
    #[should_panic(expected = "past the 4 held")]
    fn a_slot_past_the_chunks_is_refused() {
        let c = PagedChunks::new(4096, 4);
        assert!(c.grow());
        c.slot(4, 1);
    }
}
