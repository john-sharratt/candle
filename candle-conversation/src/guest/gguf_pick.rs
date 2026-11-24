//! Which of a GGUF's tensors a guest places, and which of those become int8
//! twins.
//!
//! A two-dimensional tensor the q8a128 matmul can tile is placed as its KO
//! twin — a lane-major repack only that matmul reads. A model that reads such
//! a tensor any other way, a table it dequantizes and gathers from, needs its
//! file bytes instead: the twin is not a layout anything else decodes, and a
//! dequantize over it reads a different format's blocks from a buffer of
//! another length.

use candle::quantized::ko_quant::ko_tileable;
use candle::quantized::Int8Mode;

/// The tensors a guest leaves out, and the tileable ones it keeps verbatim.
#[derive(Clone, Copy, Debug, Default)]
pub struct Pick<'a> {
    /// Tensors the model never asks for: not placed at all.
    pub skip: &'a [&'a str],
    /// Tileable tensors the model reads densely: placed as their file bytes.
    pub verbatim: &'a [&'a str],
}

impl Pick<'_> {
    /// Whether `name`, of `dims`, is placed as its int8 twin under `mode`.
    pub fn twins(&self, name: &str, dims: &[usize], mode: Int8Mode) -> bool {
        mode.is_int8()
            && !self.verbatim.contains(&name)
            && match dims {
                [nrows, ncols] => ko_tileable(*nrows, *ncols),
                _ => false,
            }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A table the model dequantizes keeps its file bytes**; every other
    /// tileable matrix is twinned, and nothing is twinned with int8 off or
    /// when its shape does not tile.
    #[test]
    fn a_verbatim_tensor_is_never_twinned() {
        let pick = Pick {
            skip: &[],
            verbatim: &["v.position_embd.weight"],
        };
        let int8 = Int8Mode::Precision;
        assert!(!pick.twins("v.position_embd.weight", &[2304, 1152], int8));
        assert!(pick.twins("v.blk.0.attn_qkv.weight", &[3456, 1152], int8));
        assert!(!pick.twins("v.blk.0.ffn_up.weight", &[4304, 1152], int8));
        assert!(!pick.twins("v.blk.0.ln1.weight", &[1152], int8));
        assert!(!pick.twins("v.blk.0.attn_qkv.weight", &[3456, 1152], Int8Mode::Off));
    }
}
