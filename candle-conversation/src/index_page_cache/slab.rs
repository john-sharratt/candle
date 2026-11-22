//! The cache's host memory: its regions, carved into blocks of the size classes
//! as each is first needed, with a free list per class.
//!
//! The slab hands out and takes back whole blocks and knows nothing about what
//! they hold. It is only ever touched under the cache's lock, so a block is in
//! exactly one place at a time: the free list, the cache's map, or the hands of
//! the one reader filling it.

use std::sync::Arc;

use candle::host_region::{HostRegion, RegionKind};

use super::classes::{class_bytes, CLASSES};

/// A block of the slab: `class_bytes(class)` bytes at `offset` of region
/// `region`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Block {
    pub region: usize,
    pub offset: usize,
    pub class: usize,
}

/// See the module documentation.
pub(super) struct Slab {
    /// Pinned regions first: a block is carved from the first with room.
    /// Shared with every page's bytes, so no page outlives its memory.
    regions: Arc<Vec<HostRegion>>,
    /// How much of each region has been carved.
    carved: Vec<usize>,
    free: [Vec<Block>; CLASSES],
}

impl Slab {
    pub fn new(mut regions: Vec<HostRegion>) -> Self {
        regions.sort_by_key(|r| r.kind() != RegionKind::Pinned);
        let carved = vec![0; regions.len()];
        Self {
            regions: Arc::new(regions),
            carved,
            free: Default::default(),
        }
    }

    /// The regions, for a page's bytes to keep alive.
    pub fn regions(&self) -> &Arc<Vec<HostRegion>> {
        &self.regions
    }

    /// A free block of `class`: one returned earlier, else one carved from the
    /// first region with room. `None` when neither exists.
    pub fn take(&mut self, class: usize) -> Option<Block> {
        if let Some(block) = self.free[class].pop() {
            return Some(block);
        }
        let size = class_bytes(class);
        for (region, r) in self.regions.iter().enumerate() {
            let offset = self.carved[region];
            if offset + size <= r.len() {
                self.carved[region] = offset + size;
                return Some(Block {
                    region,
                    offset,
                    class,
                });
            }
        }
        None
    }

    /// `block` is free again.
    pub fn give(&mut self, block: Block) {
        self.free[block.class].push(block);
    }

    /// The block's first byte.
    pub fn base(&self, block: Block) -> *mut u8 {
        // SAFETY: `offset` was carved inside the region, which lives as long
        // as the slab.
        unsafe { self.regions[block.region].base().add(block.offset) }
    }

    /// Bytes of each kind of region: `(pinned, pageable)`.
    pub fn capacity(&self) -> (usize, usize) {
        self.regions
            .iter()
            .fold((0, 0), |(p, q), r| match r.kind() {
                RegionKind::Pinned => (p + r.len(), q),
                RegionKind::Pageable => (p, q + r.len()),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slab(bytes: usize) -> Slab {
        Slab::new(vec![HostRegion::pageable(bytes).unwrap()])
    }

    /// Blocks are carved end to end until the region is spent, then refused.
    #[test]
    fn blocks_are_carved_until_the_region_is_spent() {
        let mut s = slab(32 * 1024);
        let a = s.take(2).unwrap();
        let b = s.take(1).unwrap();
        let c = s.take(1).unwrap();
        assert_eq!((a.offset, b.offset, c.offset), (0, 16_384, 24_576));
        assert_eq!(s.take(0), None, "32 KiB carved: nothing left");
    }

    /// A block given back is the next one of its class taken, and of no other.
    #[test]
    fn a_block_given_back_is_taken_again_in_its_class() {
        let mut s = slab(16 * 1024);
        let a = s.take(1).unwrap();
        let b = s.take(1).unwrap();
        assert_eq!(s.take(1), None);
        s.give(a);
        assert_eq!(s.take(0), None, "a class-1 block is not split");
        assert_eq!(s.take(1), Some(a));
        assert_ne!(a, b);
    }

    /// Blocks start on the region's base plus their offset.
    #[test]
    fn a_blocks_base_is_its_offset_into_its_region() {
        let mut s = slab(8192);
        let a = s.take(0).unwrap();
        let b = s.take(0).unwrap();
        assert_eq!(s.base(b) as usize - s.base(a) as usize, 4096);
        assert_eq!(s.capacity(), (0, 8192));
    }
}
