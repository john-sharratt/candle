//! The block sizes the cache carves its slab into: powers of two from one disk
//! sector to [`LARGEST_CLASS`]. A record is held in the smallest class that fits
//! its whole size, header and padding included, so the log's read lands in the
//! block without a second buffer.

use candle::host_region::REGION_ALIGN;

/// How many classes there are: 4 KiB, 8 KiB, … 8 MiB.
pub(super) const CLASSES: usize = 12;

/// The largest block. A record larger than this is read into the heap and not
/// cached.
pub(super) const LARGEST_CLASS: usize = REGION_ALIGN << (CLASSES - 1);

/// The class a record of `len` bytes is held in, or `None` when it is larger
/// than [`LARGEST_CLASS`] or empty.
pub(super) fn class_of(len: usize) -> Option<usize> {
    if len == 0 || len > LARGEST_CLASS {
        return None;
    }
    let blocks = len.div_ceil(REGION_ALIGN);
    Some(blocks.next_power_of_two().trailing_zeros() as usize)
}

/// The bytes of a block of `class`.
pub(super) fn class_bytes(class: usize) -> usize {
    REGION_ALIGN << class
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each length goes to the smallest power-of-two block that holds it.
    #[test]
    fn a_record_takes_the_smallest_block_that_holds_it() {
        assert_eq!(class_of(1), Some(0));
        assert_eq!(class_of(4096), Some(0));
        assert_eq!(class_of(4097), Some(1));
        assert_eq!(class_of(8192), Some(1));
        assert_eq!(class_of(12_288), Some(2));
        assert_eq!(class_of(1 << 20), Some(8));
        assert_eq!(class_of(8 << 20), Some(11));
        assert_eq!(class_bytes(0), 4096);
        assert_eq!(class_bytes(2), 16_384);
        assert_eq!(class_bytes(11), 8 << 20);
        assert_eq!(LARGEST_CLASS, 8 << 20);
    }

    /// Nothing is no record, and past the largest block is not cached.
    #[test]
    fn empty_and_oversized_records_have_no_class() {
        assert_eq!(class_of(0), None);
        assert_eq!(class_of((8 << 20) + 1), None);
    }
}
