//! Which of a GGUF's tensors a guest places, and which of those it builds int8
//! twins of as it places them.
//!
//! A two-dimensional tensor the q8a128 matmul can tile is placed as its KO
//! twin — a lane-major repack only that matmul reads — when the placement's
//! mode builds twins. A tensor the file already stores as a KO twin (a twinned
//! model pack, `model_pack::twins`) is placed as it is: it is the twin.

use candle::quantized::ko_quant::ko_tileable;
use candle::quantized::{GgmlDType, Int8Mode};

/// The tensors a guest leaves out.
#[derive(Clone, Copy, Debug, Default)]
pub struct Pick<'a> {
    /// Tensors the model never asks for: not placed at all.
    pub skip: &'a [&'a str],
}

impl Pick<'_> {
    /// Whether a `dtype` tensor of `dims` is placed as an int8 twin built under
    /// `mode`.
    pub fn twins(&self, dims: &[usize], dtype: GgmlDType, mode: Int8Mode) -> bool {
        mode.is_int8()
            && !dtype.is_ko()
            && match dims {
                [nrows, ncols] => ko_tileable(*nrows, *ncols),
                _ => false,
            }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A tileable matrix is twinned under an int8 mode**; nothing is with
    /// int8 off, when its shape does not tile, or when the file already holds
    /// it as a twin.
    #[test]
    fn only_a_tileable_source_is_twinned_under_int8() {
        let pick = Pick::default();
        let int8 = Int8Mode::Precision;
        let f16 = GgmlDType::F16;
        assert!(pick.twins(&[3456, 1152], f16, int8));
        assert!(!pick.twins(&[4304, 1152], f16, int8));
        assert!(!pick.twins(&[1152], f16, int8));
        assert!(!pick.twins(&[3456, 1152], f16, Int8Mode::Off));
        assert!(!pick.twins(&[3456, 1152], GgmlDType::Q8_KO, int8));
    }
}
