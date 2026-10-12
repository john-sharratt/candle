//! Device memory for a [`super::pinned_staging::PinnedStager`]'s resident
//! submissions — the slot headers, slice arrays and position maps every block
//! of an attention kernel walks first, copied out of the pinned arena so the
//! reads are L2 hits rather than PCIe round trips.
//!
//! Each of those copies used to take a buffer of its own from the driver, on
//! every forward: the most frequent driver allocation left in the serving loop.
//! They live exactly as long as the pinned bytes they were copied from — until
//! the stager's fence fires and its arena resets — so they are carved from
//! slabs the stager holds and rewind with that same reset. A generation that
//! outgrows its slab takes another; the next reset folds them into one slab as
//! wide as all of them, so a steady workload settles on one slab and allocates
//! nothing.

/// Alignment of every carve: what a fresh device allocation guarantees and
/// what the kernels' vectorised loads assume of a base pointer.
pub(crate) const RESIDENT_ALIGN: usize = 256;

/// The first slab's width.
pub(crate) const FIRST_SLAB_BYTES: usize = 4 << 20;

/// Where a carve lands, decided on the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Carve {
    /// At this byte offset of the current slab.
    At(usize),
    /// At offset 0 of a new slab of this many bytes, which becomes current.
    NewSlab(usize),
}

/// The slabs' bookkeeping, with no device in it.
#[derive(Debug, Clone)]
pub(crate) struct RingPlan {
    /// Width of each held slab, oldest first; the last is current.
    slabs: Vec<usize>,
    /// Bytes used in the current slab.
    cursor: usize,
    /// The width the next first slab opens at.
    first_width: usize,
}

impl Default for RingPlan {
    fn default() -> Self {
        Self {
            slabs: Vec::new(),
            cursor: 0,
            first_width: FIRST_SLAB_BYTES,
        }
    }
}

impl RingPlan {
    /// Place `len` bytes.
    pub(crate) fn carve(&mut self, len: usize) -> Carve {
        let len = len.max(1);
        if let Some(&cap) = self.slabs.last() {
            let at = self.cursor.next_multiple_of(RESIDENT_ALIGN);
            if at + len <= cap {
                self.cursor = at + len;
                return Carve::At(at);
            }
        }
        let floor = match self.slabs.last() {
            Some(&last) => 2 * last,
            None => self.first_width,
        };
        let width = floor.max(len.next_multiple_of(RESIDENT_ALIGN));
        self.slabs.push(width);
        self.cursor = len;
        Carve::NewSlab(width)
    }

    /// Rewind for the next generation. `true` when the held slabs must be
    /// dropped: several were held, and the next generation opens one slab as
    /// wide as all of them, so a generation of the same shape fits in one.
    pub(crate) fn reset(&mut self) -> bool {
        self.cursor = 0;
        if self.slabs.len() <= 1 {
            return false;
        }
        self.first_width = self.slabs.iter().sum();
        self.slabs.clear();
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Carves follow each other on the alignment inside one slab.
    #[test]
    fn carves_are_aligned_and_sequential() {
        let mut r = RingPlan::default();
        assert_eq!(r.carve(100), Carve::NewSlab(FIRST_SLAB_BYTES));
        assert_eq!(r.carve(10), Carve::At(256));
        assert_eq!(r.carve(0), Carve::At(512));
    }

    /// A carve that does not fit opens a slab at least twice the last, and
    /// one wider than that gets a slab of its own width.
    #[test]
    fn an_overflow_opens_a_wider_slab() {
        let mut r = RingPlan::default();
        r.carve(FIRST_SLAB_BYTES - 10);
        assert_eq!(r.carve(100), Carve::NewSlab(2 * FIRST_SLAB_BYTES));
        let huge = 40 * FIRST_SLAB_BYTES + 1;
        assert_eq!(
            r.carve(huge),
            Carve::NewSlab(huge.next_multiple_of(RESIDENT_ALIGN))
        );
    }

    /// A reset after an overflow drops the slabs and the next carve opens one
    /// as wide as all of them; a reset with one slab keeps it and rewinds.
    #[test]
    fn a_reset_folds_the_overflow_into_one_slab() {
        let mut r = RingPlan::default();
        r.carve(FIRST_SLAB_BYTES);
        r.carve(1);
        assert!(r.reset());
        assert_eq!(
            r.carve(2 * FIRST_SLAB_BYTES),
            Carve::NewSlab(3 * FIRST_SLAB_BYTES)
        );
        assert_eq!(r.carve(1), Carve::At(2 * FIRST_SLAB_BYTES));
        assert!(!r.reset());
        assert_eq!(r.carve(1), Carve::At(0));
    }
}
