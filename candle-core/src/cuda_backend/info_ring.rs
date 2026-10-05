//! Device copies of the small layout tables strided kernels read.
//!
//! A strided kernel reads its operand's `dims ++ strides` (and, for a binary
//! op, the second operand's strides) from device memory. Those tables repeat
//! across launches, so each distinct one is uploaded once and memoized by
//! contents — per-call uploads were a measured WDDM submission storm, tens of
//! thousands of 24–128 B copies per wave sweep.
//!
//! **Carved from one ring, not allocated per table.** Wave widths change from
//! one forward to the next, so new shapes — and new tables — keep arriving for
//! the life of the process. Giving each its own pool allocation put a driver
//! allocation inside the forward every time a width was seen for the first
//! time. Every table is instead written into a ring buffer allocated once; a
//! miss is one small host→device copy into the next free words.
//!
//! **Rewinding is stream-ordered.** When the ring is full it starts again at
//! word 0 of the same buffer and forgets every memoized table. That is safe
//! because every kernel that read an old table was launched on this handle's
//! stream before the copy that overwrites it, so it has read the old words by
//! the time they change. The one thing stream order cannot protect is a table
//! a caller still holds on the host and has not launched against yet; the
//! entry's `Arc` count says so, and when any is held the ring moves to a fresh
//! buffer instead, which the held entries keep alive through their anchor.

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::{CudaSlice, DevicePtr};

use super::{CudaDevice, Uploaded, WrapErr};
use crate::Result;

/// Words in one ring buffer (512 KiB). A table is `dims ++ strides` for one
/// or two operands — under 32 words at any rank in use — so the ring holds
/// thousands of distinct shapes before it wraps.
pub(crate) const RING_WORDS: usize = 1 << 16;

/// Where the next table is written.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Place {
    /// At this word offset of the current buffer.
    At(usize),
    /// At word 0 of the current buffer, forgetting every earlier table.
    Rewind,
    /// At word 0 of a fresh buffer of this many words.
    Fresh(usize),
}

/// Place a table of `len` words in a ring of `capacity` words (`None` before
/// the first buffer exists) whose first `used` words are taken. `held` is
/// whether any earlier table is still held by a caller on the host.
pub(crate) fn place(used: usize, len: usize, capacity: Option<usize>, held: bool) -> Place {
    match capacity {
        Some(cap) if used + len <= cap => Place::At(used),
        Some(cap) if !held && len <= cap => Place::Rewind,
        _ => Place::Fresh(RING_WORDS.max(len)),
    }
}

/// The memo and the ring it is carved from. One per stream handle.
#[derive(Default)]
pub(crate) struct InfoRing {
    buf: Option<Arc<CudaSlice<usize>>>,
    used: usize,
    tables: HashMap<Vec<usize>, Arc<Uploaded<usize>>>,
}

impl InfoRing {
    /// The device copy of `info`, uploading it on a miss.
    pub(crate) fn table(
        &mut self,
        dev: &CudaDevice,
        info: &[usize],
    ) -> Result<Arc<Uploaded<usize>>> {
        if let Some(t) = self.tables.get(info) {
            return Ok(t.clone());
        }
        // A miss uploads (and may rewind over words recorded launches read),
        // so it runs eagerly behind everything recorded so far.
        let _eager = dev.pause_capture()?;
        let capacity = self.buf.as_ref().map(|b| b.len());
        let wraps = capacity.is_some_and(|cap| self.used + info.len() > cap);
        let held = wraps && self.tables.values().any(|t| Arc::strong_count(t) > 1);
        let at = match place(self.used, info.len(), capacity, held) {
            Place::At(at) => at,
            Place::Rewind => {
                self.tables.clear();
                0
            }
            Place::Fresh(words) => {
                self.tables.clear();
                // SAFETY: a word is read only through a table, and each table's
                // words are written by its own upload below before any launch.
                self.buf = Some(Arc::new(unsafe { dev.alloc::<usize>(words)? }));
                0
            }
        };
        let buf = self.buf.as_ref().expect("placed above");
        let stream = dev.compute_stream();
        let (base, _guard) = buf.device_ptr(&stream);
        let addr = base + (at * std::mem::size_of::<usize>()) as u64;
        // SAFETY: `[at, at + len)` lies inside `buf`, and the entry's anchor
        // keeps `buf` alive for as long as the view exists.
        let mut view = unsafe { stream.upgrade_device_ptr::<usize>(addr, info.len()) };
        stream.memcpy_htod(info, &mut view).w()?;
        self.used = at + info.len();
        let t = Arc::new(Uploaded::leased(view, Some(buf.clone())));
        self.tables.insert(info.to_vec(), t.clone());
        Ok(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_table_opens_a_ring() {
        assert_eq!(place(0, 12, None, false), Place::Fresh(RING_WORDS));
    }

    #[test]
    fn a_table_that_fits_appends_after_the_last() {
        assert_eq!(place(40, 12, Some(64), false), Place::At(40));
        assert_eq!(place(52, 12, Some(64), true), Place::At(52));
    }

    #[test]
    fn a_full_ring_with_no_holder_rewinds_in_place() {
        assert_eq!(place(60, 12, Some(64), false), Place::Rewind);
    }

    #[test]
    fn a_full_ring_with_a_holder_moves_to_a_fresh_buffer() {
        assert_eq!(place(60, 12, Some(64), true), Place::Fresh(RING_WORDS));
    }

    #[test]
    fn a_table_wider_than_the_ring_gets_a_buffer_of_its_own_width() {
        let wide = RING_WORDS + 3;
        assert_eq!(place(0, wide, Some(RING_WORDS), false), Place::Fresh(wide));
    }

    /// `[i, i+1, …, i+15]` — sixteen words, so `RING_WORDS / 16` of them fill
    /// the ring exactly.
    fn table_words(i: usize) -> Vec<usize> {
        (i..i + 16).collect()
    }

    fn addr(dev: &CudaDevice, t: &Uploaded<usize>) -> u64 {
        t.device_ptr(&dev.cuda_stream()).0
    }

    fn read(dev: &CudaDevice, t: &Uploaded<usize>) -> Vec<usize> {
        dev.cuda_stream().memcpy_dtov(&**t).unwrap()
    }

    fn device() -> Option<CudaDevice> {
        use crate::backend::BackendDevice;
        CudaDevice::new(0).ok()
    }

    #[test]
    fn a_full_ring_rewinds_to_its_base_and_the_new_table_reads_back() {
        let Some(dev) = device() else { return };
        let mut ring = InfoRing::default();
        let first = ring.table(&dev, &table_words(0)).unwrap();
        let base = addr(&dev, &first);
        assert!(Arc::ptr_eq(
            &first,
            &ring.table(&dev, &table_words(0)).unwrap()
        ));
        drop(first);
        for i in 1..RING_WORDS / 16 {
            ring.table(&dev, &table_words(i)).unwrap();
        }
        let wrapped = ring.table(&dev, &table_words(RING_WORDS)).unwrap();
        assert_eq!(addr(&dev, &wrapped), base);
        assert_eq!(read(&dev, &wrapped), table_words(RING_WORDS));
    }

    #[test]
    fn a_held_table_survives_the_wrap_in_its_own_buffer() {
        let Some(dev) = device() else { return };
        let mut ring = InfoRing::default();
        let held = ring.table(&dev, &table_words(0)).unwrap();
        let base = addr(&dev, &held);
        for i in 1..RING_WORDS / 16 {
            ring.table(&dev, &table_words(i)).unwrap();
        }
        let next = ring.table(&dev, &table_words(RING_WORDS)).unwrap();
        assert_ne!(addr(&dev, &next), base);
        assert_eq!(read(&dev, &held), table_words(0));
        assert_eq!(read(&dev, &next), table_words(RING_WORDS));
    }
}
