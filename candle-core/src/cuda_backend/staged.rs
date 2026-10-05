//! Several host arrays a launch reads, laid out as one staging upload.
//!
//! A kernel fed by a descriptor table and the side arrays it indexes would
//! otherwise upload each array into its own fresh allocation — one driver
//! allocation per array per launch. Packed end to end, they go up in one copy
//! into the device's reused staging buffer
//! ([`super::CudaDevice::with_staged_segments`]) and allocate nothing.

use cudarc::driver::DeviceRepr;

/// Every segment starts on this byte boundary, which covers the alignment of
/// any element type a launch table holds (at most 8 bytes).
pub const SEGMENT_ALIGN: usize = 16;

/// `data`'s bytes, for packing as a segment.
pub fn segment<T: DeviceRepr>(data: &[T]) -> &[u8] {
    // SAFETY: `DeviceRepr` types are plain values with no padding the kernel
    // does not also see; read as bytes they are exactly what is uploaded.
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) }
}

/// Lay `segments` end to end, each starting on a [`SEGMENT_ALIGN`] boundary,
/// returning the packed bytes and each segment's offset into them. The gaps
/// are zero. An empty segment still gets an offset — a launch needs a valid
/// address for an input it never reads.
pub fn pack_segments(segments: &[&[u8]]) -> (Vec<u8>, Vec<usize>) {
    let mut bytes = Vec::with_capacity(
        segments
            .iter()
            .map(|s| s.len().next_multiple_of(SEGMENT_ALIGN))
            .sum(),
    );
    let mut offsets = Vec::with_capacity(segments.len());
    for s in segments {
        bytes.resize(bytes.len().next_multiple_of(SEGMENT_ALIGN), 0);
        offsets.push(bytes.len());
        bytes.extend_from_slice(s);
    }
    (bytes, offsets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segments_start_on_the_alignment_with_zero_gaps() {
        let a = [1u8, 2, 3];
        let b = [0x0102_0304_0506_0708i64];
        let (bytes, offsets) = pack_segments(&[&a, segment(&b)]);
        assert_eq!(offsets, vec![0, 16]);
        let mut want = vec![1u8, 2, 3];
        want.resize(16, 0);
        want.extend_from_slice(&[8, 7, 6, 5, 4, 3, 2, 1]);
        assert_eq!(bytes, want);
    }

    #[test]
    fn an_empty_segment_gets_the_next_aligned_offset() {
        let a = [9u8; 17];
        let (bytes, offsets) = pack_segments(&[&a, &[], &[5u8]]);
        assert_eq!(offsets, vec![0, 32, 32]);
        assert_eq!(bytes.len(), 33);
        assert_eq!(bytes[32], 5);
        assert!(bytes[17..32].iter().all(|&b| b == 0));
    }
}
