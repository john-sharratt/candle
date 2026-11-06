//! Part of a sealed index page, as a page of its own — what a span placed from
//! the middle of a sealed turn pushes in place of the turn's whole page
//! (`docs/inline_read_injection.md` §3).
//!
//! A page is one [`SealedIndex`] per KV layer ([`encode_aux`]). A layer's
//! completed rows each pool `ratio` tokens from the page's start, the last one
//! `last_cells`, so a range of the page's tokens is a run of whole rows exactly
//! when it starts on a row and ends on one or at the rows' end. The open block —
//! tokens not yet pooled when the page was sealed — is never taken: no row of it
//! exists to give.

use std::ops::Range;

use candle::Result;

use super::paged_index::{decode_aux, encode_aux, SealedIndex};

/// The rows of the page `blob` over its tokens `tokens`. `ratios` are the
/// page's layers' QSA ratios in its layer order, 0 for a layer that indexes
/// nothing.
pub fn slice_page(blob: &[u8], ratios: &[usize], tokens: Range<usize>) -> Result<Vec<u8>> {
    if tokens.is_empty() {
        candle::bail!("index page: an empty range {tokens:?} has no rows");
    }
    let (_, layers) = decode_aux(blob)?;
    if layers.len() != ratios.len() {
        candle::bail!(
            "index page: {} layers against {} ratios",
            layers.len(),
            ratios.len()
        );
    }
    let sliced = layers
        .iter()
        .zip(ratios)
        .map(|(layer, &ratio)| slice_layer(layer, ratio, tokens.clone()))
        .collect::<Result<Vec<_>>>()?;
    encode_aux(&[], &sliced)
}

/// One layer's rows over `tokens`.
fn slice_layer(layer: &SealedIndex, ratio: usize, tokens: Range<usize>) -> Result<SealedIndex> {
    if ratio == 0 {
        return Ok(SealedIndex {
            rows: Vec::new(),
            dim: layer.dim,
            last_cells: layer.last_cells,
            open: Vec::new(),
        });
    }
    let covered = layer.page_tokens(ratio);
    if tokens.end > covered {
        candle::bail!(
            "index page: tokens {tokens:?} reach past the {covered} its completed rows cover"
        );
    }
    if !tokens.start.is_multiple_of(ratio) {
        candle::bail!("index page: tokens {tokens:?} do not start on a {ratio}-token row");
    }
    if !tokens.end.is_multiple_of(ratio) && tokens.end != covered {
        candle::bail!(
            "index page: tokens {tokens:?} end inside a {ratio}-token row short of the rows' end"
        );
    }
    let first = tokens.start / ratio;
    let last = tokens.end.div_ceil(ratio);
    Ok(SealedIndex {
        rows: layer.rows[first * layer.dim..last * layer.dim].to_vec(),
        dim: layer.dim,
        last_cells: tokens.end - (last - 1) * ratio,
        open: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ratio-4 layer of three rows, the last pooling 3 tokens — 11 tokens —
    /// with one open row; and a layer that indexes nothing.
    fn page() -> Vec<u8> {
        encode_aux(
            &[],
            &[
                SealedIndex {
                    rows: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    dim: 2,
                    last_cells: 3,
                    open: vec![7.0, 8.0],
                },
                SealedIndex {
                    rows: Vec::new(),
                    dim: 2,
                    last_cells: 1,
                    open: Vec::new(),
                },
            ],
        )
        .unwrap()
    }

    fn layers(blob: &[u8]) -> Vec<SealedIndex> {
        decode_aux(blob).unwrap().1
    }

    /// Rows 1..3 to the rows' end keep the short last row; rows 0..2 end on a
    /// boundary, so their last row is whole. The open row never comes along.
    #[test]
    fn a_slice_is_the_whole_rows_over_its_range() {
        let tail = layers(&slice_page(&page(), &[4, 0], 4..11).unwrap());
        assert_eq!(tail[0].rows, vec![3.0, 4.0, 5.0, 6.0]);
        assert_eq!(tail[0].last_cells, 3);
        assert!(tail[0].open.is_empty());
        assert_eq!(tail[0].page_tokens(4), 7);
        assert!(tail[1].rows.is_empty());

        let head = layers(&slice_page(&page(), &[4, 0], 0..8).unwrap());
        assert_eq!(head[0].rows, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(head[0].last_cells, 4);
        assert_eq!(head[0].page_tokens(4), 8);
    }

    /// The encoded slice, byte for byte: version 3, no PLE, two layers — the
    /// first `[2 rows, dim 2, last 4, open 0]` then its four floats, the second
    /// `[0, 2, 1, 0]`.
    #[test]
    fn a_slice_encodes_to_its_raw_bytes() {
        let blob = slice_page(&page(), &[4, 0], 0..8).unwrap();
        let mut want: Vec<u8> = Vec::new();
        let u32s = |out: &mut Vec<u8>, vs: &[u32]| {
            for v in vs {
                out.extend_from_slice(&v.to_le_bytes());
            }
        };
        u32s(&mut want, &[3, 0, 2]);
        u32s(&mut want, &[16 + 4 * 4, 2, 2, 4, 0]);
        for v in [1.0f32, 2.0, 3.0, 4.0] {
            want.extend_from_slice(&v.to_le_bytes());
        }
        u32s(&mut want, &[16, 0, 2, 1, 0]);
        assert_eq!(blob, want);
    }

    #[test]
    fn a_range_off_the_rows_is_refused() {
        let err = |r: Range<usize>| slice_page(&page(), &[4, 0], r).unwrap_err().to_string();
        assert!(err(2..8).contains("do not start on"));
        assert!(err(4..9).contains("end inside"));
        assert!(err(4..12).contains("reach past"));
        assert!(err(4..4).contains("empty range"));
        assert!(slice_page(&page(), &[4], 0..8)
            .unwrap_err()
            .to_string()
            .contains("2 layers against 1 ratios"));
    }
}
