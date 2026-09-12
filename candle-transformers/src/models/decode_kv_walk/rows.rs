//! Which rows of a q8a1024 operand carry a non-finite per-128 scale.
//!
//! The int8 payload of a q8a1024 buffer cannot encode a non-finite value, so a
//! row that dequantizes to NaN or inf has a non-finite `{scale, sum}` in one of
//! its tiles' `ds[0]`. Reading those on the host names the rows — on the decode
//! path, the sequences — from the bytes themselves rather than from any count
//! the device reported.

use candle::Result;
use half::f16;

/// Bytes per q8a1024 super-block: 8 × 128 int8 tiles, then 8 × 16-byte ds slots.
pub(crate) const Q8A1024_BLK_BYTES: usize = 1152;
/// Offset of the ds section inside a super-block.
pub(crate) const Q8A1024_META_OFF: usize = 1024;
/// Elements per q8a1024 tile.
const TILE_ELEMS: usize = 128;

/// Byte offset of tile `flat_tile`'s ds slot, whose first four bytes are the
/// tile's `{scale, sum}` as a little-endian half2 (`q8a1024_ds_off` in
/// `blocks.cuh`).
pub(crate) fn ds_off(flat_tile: usize) -> usize {
    (flat_tile >> 3) * Q8A1024_BLK_BYTES + Q8A1024_META_OFF + (flat_tile & 7) * 16
}

/// Rows of a `rows × cols` q8a1024 operand whose scale or sum is non-finite in
/// any of its tiles, in ascending order.
///
/// A buffer too short to hold every row's meta is an error rather than a row
/// reported bad: a truncated read would otherwise name sequences that were
/// never examined.
pub(crate) fn nonfinite_rows(bytes: &[u8], rows: usize, cols: usize) -> Result<Vec<usize>> {
    if !cols.is_multiple_of(TILE_ELEMS) {
        candle::bail!("q8a1024 rows: {cols} columns is not a whole number of 128-element tiles");
    }
    let tiles_per_row = cols / TILE_ELEMS;
    let need = ds_off((rows * tiles_per_row).saturating_sub(1)) + 4;
    if rows > 0 && tiles_per_row > 0 && bytes.len() < need {
        candle::bail!(
            "q8a1024 rows: {} bytes cannot hold the meta of {rows} rows × {tiles_per_row} tiles \
             (need {need})",
            bytes.len()
        );
    }
    let half_at = |o: usize| f16::from_le_bytes([bytes[o], bytes[o + 1]]);
    Ok((0..rows)
        .filter(|&r| {
            (0..tiles_per_row).any(|k| {
                let o = ds_off(r * tiles_per_row + k);
                !half_at(o).is_finite() || !half_at(o + 2).is_finite()
            })
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `rows × cols` operand whose every tile holds `{scale 1.0, sum 0.0}`.
    fn clean(rows: usize, cols: usize) -> Vec<u8> {
        let tiles = rows * cols / TILE_ELEMS;
        let mut b = vec![0u8; tiles.div_ceil(8) * Q8A1024_BLK_BYTES];
        for t in 0..tiles {
            let o = ds_off(t);
            b[o..o + 2].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
            b[o + 2..o + 4].copy_from_slice(&f16::from_f32(0.0).to_le_bytes());
        }
        b
    }

    #[test]
    fn ds_slots_follow_the_flat_grouped_layout() {
        assert_eq!(ds_off(0), 1024);
        assert_eq!(ds_off(7), 1024 + 7 * 16);
        assert_eq!(ds_off(8), 1152 + 1024);
        assert_eq!(ds_off(19), 2 * 1152 + 1024 + 3 * 16);
    }

    #[test]
    fn a_clean_operand_names_no_row() {
        assert!(nonfinite_rows(&clean(3, 256), 3, 256).unwrap().is_empty());
    }

    #[test]
    fn a_nan_scale_names_its_row_only() {
        let mut b = clean(3, 256);
        // Row 1, tile 1 → flat tile 3.
        let o = ds_off(3);
        b[o..o + 2].copy_from_slice(&0x7E00u16.to_le_bytes());
        assert_eq!(nonfinite_rows(&b, 3, 256).unwrap(), vec![1]);
    }

    /// The capture's own geometry: 10 rows × 4096 columns is 320 tiles in 40
    /// super-blocks, 46,080 bytes. An inf *sum* counts as much as a bad scale.
    #[test]
    fn an_inf_sum_in_the_last_tile_of_a_row_names_that_row() {
        let mut b = clean(10, 4096);
        assert_eq!(b.len(), 46080);
        let o = ds_off(7 * 32 + 31);
        b[o + 2..o + 4].copy_from_slice(&0x7C00u16.to_le_bytes());
        assert_eq!(nonfinite_rows(&b, 10, 4096).unwrap(), vec![7]);
    }

    #[test]
    fn poison_scales_name_every_poisoned_row() {
        let mut b = clean(4, 128);
        for r in [0usize, 3] {
            let o = ds_off(r);
            b[o..o + 4].copy_from_slice(&[0xFF; 4]);
        }
        assert_eq!(nonfinite_rows(&b, 4, 128).unwrap(), vec![0, 3]);
    }

    #[test]
    fn a_buffer_too_short_for_the_meta_is_refused() {
        let b = clean(10, 4096);
        assert!(nonfinite_rows(&b[..46080 - 1152], 10, 4096).is_err());
    }

    #[test]
    fn a_partial_tile_is_refused() {
        assert!(nonfinite_rows(&clean(1, 256), 1, 200).is_err());
    }
}
