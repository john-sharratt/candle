//! A quantized weight's input columns, split in two at a block boundary —
//! exactly, by cutting each row's bytes, with nothing re-quantized.
//!
//! The draft head's `eh_proj` is one `[n_embd, 2·n_embd]` weight over the
//! concat `[enorm(e) ; hnorm(h)]`. Applied as stored, the head has to build that
//! concat — the embedding broadcast across every stream and copied, then the
//! two halves joined — which is two wide copies per wave for an operand the
//! weight only ever reads in two halves. Split at load, each half is its own
//! weight: `fc_embedding · e` runs once per row instead of once per stream, and
//! `fc_hidden · h` reads the norm output where it lies.
//!
//! A row of a block-quantized weight is its blocks in column order, so a cut
//! at a column that starts a block is a cut in each row's bytes, and the two
//! halves hold the very blocks the whole did.

use candle::quantized::ggml_file::qtensor_from_ggml;
use candle::quantized::QTensor;
use candle::{Device, Result};

/// `w`'s columns `0..at` and `at..`, as two weights of `w`'s encoding.
///
/// `w` is read through [`QTensor::data`], so it should be on the host: a device
/// weight would be copied back whole to be cut.
pub fn split_columns(w: &QTensor, at: usize, device: &Device) -> Result<(QTensor, QTensor)> {
    let (rows, cols) = w.shape().dims2()?;
    let dtype = w.dtype();
    let block = dtype.block_size();
    if at == 0 || at >= cols {
        candle::bail!("split_columns: a cut at column {at} of {cols} leaves an empty half");
    }
    if !at.is_multiple_of(block) || !cols.is_multiple_of(block) {
        candle::bail!(
            "split_columns: {dtype:?} blocks are {block} wide, so column {at} of {cols} does \
             not start a block — the cut would split one"
        );
    }
    let row_bytes = cols / block * dtype.type_size();
    let left_bytes = at / block * dtype.type_size();
    let data = w.data()?;
    if data.len() != rows * row_bytes {
        candle::bail!(
            "split_columns: {} bytes for {rows} rows of {row_bytes}",
            data.len()
        );
    }
    let mut left = Vec::with_capacity(rows * left_bytes);
    let mut right = Vec::with_capacity(rows * (row_bytes - left_bytes));
    for row in data.chunks_exact(row_bytes) {
        left.extend_from_slice(&row[..left_bytes]);
        right.extend_from_slice(&row[left_bytes..]);
    }
    Ok((
        qtensor_from_ggml(dtype, &left, vec![rows, at], device)?,
        qtensor_from_ggml(dtype, &right, vec![rows, cols - at], device)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::quantized::GgmlDType;

    /// Q8_0: a block is 32 columns in 34 bytes (an f16 scale, then 32 `i8`).
    const BLOCK_BYTES: usize = 34;

    /// Two rows of two blocks each, every byte distinct.
    fn weight() -> (Vec<u8>, QTensor) {
        let bytes: Vec<u8> = (0..2 * 2 * BLOCK_BYTES).map(|i| i as u8).collect();
        let w = qtensor_from_ggml(GgmlDType::Q8_0, &bytes, vec![2, 64], &Device::Cpu).unwrap();
        (bytes, w)
    }

    #[test]
    fn each_half_holds_its_rows_blocks() {
        let (bytes, w) = weight();
        let (l, r) = split_columns(&w, 32, &Device::Cpu).unwrap();
        assert_eq!(l.shape().dims2().unwrap(), (2, 32));
        assert_eq!(r.shape().dims2().unwrap(), (2, 32));
        let mut want_l = bytes[..BLOCK_BYTES].to_vec();
        want_l.extend_from_slice(&bytes[2 * BLOCK_BYTES..3 * BLOCK_BYTES]);
        let mut want_r = bytes[BLOCK_BYTES..2 * BLOCK_BYTES].to_vec();
        want_r.extend_from_slice(&bytes[3 * BLOCK_BYTES..]);
        assert_eq!(l.data().unwrap().as_ref(), want_l.as_slice());
        assert_eq!(r.data().unwrap().as_ref(), want_r.as_slice());
    }

    #[test]
    fn a_cut_inside_a_block_is_refused() {
        let (_, w) = weight();
        assert!(split_columns(&w, 16, &Device::Cpu).is_err());
        assert!(split_columns(&w, 0, &Device::Cpu).is_err());
        assert!(split_columns(&w, 64, &Device::Cpu).is_err());
    }
}
