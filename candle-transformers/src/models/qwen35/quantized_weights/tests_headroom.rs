//! Tests for which tensors count toward a load's device headroom.
//!
//! The embedding lives on the host and is never counted, while a head read whole
//! is. A head repacked from the mapping a band at a time is left out, tied or
//! not, so the headroom is the widest projection read whole.

use super::*;

#[cfg(test)]
pub(super) mod headroom_tests {
    use super::load_headroom_bytes;
    use candle::quantized::cuda::REPACK_BAND_BYTES;
    use candle::quantized::gguf_file::{Content, TensorInfo, VersionedMagic};
    use candle::quantized::{GgmlDType, Int8Mode};
    use candle::Shape;
    use std::collections::HashMap;

    /// Q8_0 tensors: `[64, 256]` is 512 blocks of 34 bytes, `[32, 128]` 128.
    fn header(names: &[(&str, (usize, usize))]) -> Content {
        Content {
            magic: VersionedMagic::GgufV3,
            metadata: HashMap::new(),
            tensor_infos: names
                .iter()
                .map(|&(name, dims)| {
                    let info = TensorInfo {
                        ggml_dtype: GgmlDType::Q8_0,
                        shape: Shape::from(dims),
                        offset: 0,
                    };
                    (name.to_string(), info)
                })
                .collect(),
            tensor_data_offset: 0,
        }
    }

    const BANDS: usize = 2 * REPACK_BAND_BYTES;

    /// The embedding lives on the host and is never counted; a head read whole —
    /// no mapping to band it from, or no int8 twin — is.
    #[test]
    fn the_embedding_is_never_counted_and_a_head_read_whole_is() {
        let untied = header(&[
            ("token_embd.weight", (64, 256)),
            ("output.weight", (64, 256)),
            ("blk.0.attn_qkv.weight", (32, 128)),
        ]);
        for mode in [Int8Mode::Off, Int8Mode::Precision] {
            assert_eq!(
                load_headroom_bytes(&untied, None, false, mode),
                512 * 34 + BANDS
            );
        }
        assert_eq!(
            load_headroom_bytes(&untied, None, true, Int8Mode::Off),
            512 * 34 + BANDS
        );
        // Tied: the head is the embedding read again through the projection route.
        let tied = header(&[
            ("token_embd.weight", (64, 256)),
            ("blk.0.attn_qkv.weight", (32, 128)),
        ]);
        assert_eq!(
            load_headroom_bytes(&tied, None, false, Int8Mode::Precision),
            512 * 34 + BANDS
        );
    }

    /// A head repacked from the mapping a band at a time is left out, tied or
    /// not: what remains is the widest projection read whole.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_banded_head_is_left_out() {
        let untied = header(&[
            ("token_embd.weight", (64, 256)),
            ("output.weight", (64, 256)),
            ("blk.0.attn_qkv.weight", (32, 128)),
        ]);
        assert_eq!(
            load_headroom_bytes(&untied, None, true, Int8Mode::Precision),
            128 * 34 + BANDS
        );
        let tied = header(&[
            ("token_embd.weight", (64, 256)),
            ("blk.0.attn_qkv.weight", (32, 128)),
        ]);
        assert_eq!(
            load_headroom_bytes(&tied, None, true, Int8Mode::Performance),
            128 * 34 + BANDS
        );
    }
}
