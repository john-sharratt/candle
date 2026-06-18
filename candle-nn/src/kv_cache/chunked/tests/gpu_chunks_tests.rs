//! Unit tests for `gpu_chunks` serialisation helpers.
//!
//! Only compiled when the `cuda` feature is enabled because the helpers live
//! in the `#[cfg(feature = "cuda")]`-gated `gpu_chunks` module.

#[cfg(feature = "cuda")]
mod tests {
    use crate::kv_cache::chunked::gpu_chunks::{
        kv_head_serialized_size, token_slice_serialized_size, write_identity_pal_map,
        write_slice_header, SLICE_HEADER_BYTES,
    };
    /// head_dim=8, N_PALETTE=4 → sub_hd=2
    /// dims 0,1 → pal 0; 2,3 → pal 1; 4,5 → pal 2; 6,7 → pal 3
    /// byte 0: (0<<0)|(0<<2)|(1<<4)|(1<<6) = 0x50
    /// byte 1: (2<<0)|(2<<2)|(3<<4)|(3<<6) = 0xFA
    #[test]
    fn test_identity_pal_map_head_dim_8() {
        let mut buf = [0u8; 2];
        write_identity_pal_map(8, &mut buf);
        assert_eq!(buf, [0x50, 0xFA]);
    }

    /// head_dim=4, N_PALETTE=4 → sub_hd=1
    /// dim 0→0, 1→1, 2→2, 3→3 packed: (0<<0)|(1<<2)|(2<<4)|(3<<6) = 0xE4
    #[test]
    fn test_identity_pal_map_head_dim_4() {
        let mut buf = [0u8; 1];
        write_identity_pal_map(4, &mut buf);
        assert_eq!(buf, [0xE4]);
    }

    #[test]
    fn test_serialized_sizes() {
        // kv_head_size(8) = 8/4*2 + 32 + 32 + 4 + 4 + 16 + 16 = 4 + 104 = 108
        // k_pal[2] + v_pal[2] + k_ptr[4×8] + v_ptr[4×8] + k_fmt[4] + v_fmt[4]
        //   + k_scale[4×f32] + v_scale[4×f32]
        assert_eq!(kv_head_serialized_size(8), 108);
        // Two-section footprint: 16-byte slice header + the out-of-line record.
        assert_eq!(token_slice_serialized_size(1, 8), 16 + 108);
        assert_eq!(token_slice_serialized_size(2, 8), 16 + 2 * 108);
    }

    /// The 16-byte slice header (offset/len/rope/kvheads_ptr) the kernel reads.
    #[test]
    fn test_serialize_header() {
        let mut buf = vec![0u8; SLICE_HEADER_BYTES];
        write_slice_header(&mut buf, 3, 20, 0xDEAD_BEEF, 0x1234_5678_9ABC_DEF0);
        assert_eq!(
            u16::from_le_bytes(buf[0..2].try_into().unwrap()),
            3,
            "offset"
        );
        assert_eq!(u16::from_le_bytes(buf[2..4].try_into().unwrap()), 20, "len");
        assert_eq!(
            u32::from_le_bytes(buf[4..8].try_into().unwrap()),
            0xDEAD_BEEF,
            "rope"
        );
        assert_eq!(
            u64::from_le_bytes(buf[8..16].try_into().unwrap()),
            0x1234_5678_9ABC_DEF0,
            "kvheads_ptr"
        );
    }
}
