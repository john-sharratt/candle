//! Resident tokens by storage format.
//!
//! A chunk holds `token_count` tokens of one layer, stored as bands — one per
//! `(head, palette band, K/V)` — each in its own format ([`SealedChunk::k_fmt`],
//! [`SealedChunk::v_fmt`]). Every band carries an equal share of its chunk's
//! tokens, and a sequence's layers share its tokens, so a format's count is the
//! token-equivalents stored in it: the counts over every format add up to the
//! tokens resident, and a sequence stored half in `Q4_0` and half in `Q8_0`
//! reads as half of its tokens in each.
//!
//! A chunk with no format tags at all is counted as [`ArenaFormatTag::Invalid`]
//! rather than dropped, so the total stays the tokens resident.

use crate::kv_cache::chunked::types::SealedSequence;
use crate::kv_cache::ArenaFormatTag;

/// Token-equivalents per format tag.
#[derive(Clone, Debug)]
pub struct FormatCensus {
    tokens: [f64; 256],
}

impl Default for FormatCensus {
    fn default() -> Self {
        Self { tokens: [0.0; 256] }
    }
}

impl FormatCensus {
    /// Count one chunk of a sequence `layers` layers deep.
    pub fn add_chunk(&mut self, token_count: u16, k_fmt: &[u8], v_fmt: &[u8], layers: usize) {
        if token_count == 0 || layers == 0 {
            return;
        }
        let tokens = token_count as f64 / layers as f64;
        let bands = k_fmt.len() + v_fmt.len();
        if bands == 0 {
            self.tokens[ArenaFormatTag::Invalid.as_u8() as usize] += tokens;
            return;
        }
        let share = tokens / bands as f64;
        for &tag in k_fmt.iter().chain(v_fmt) {
            self.tokens[tag as usize] += share;
        }
    }

    /// Count one residence: its sealed sequences, one per layer.
    pub fn add_layers(&mut self, layers: &[SealedSequence]) {
        let n = layers.len();
        for seq in layers {
            for c in &seq.chunks {
                self.add_chunk(c.token_count, &c.k_fmt, &c.v_fmt, n);
            }
        }
    }

    /// Every format holding anything, most tokens first; equal counts in tag
    /// order.
    pub fn rows(&self) -> Vec<(ArenaFormatTag, f64)> {
        let mut rows: Vec<(ArenaFormatTag, f64)> = self
            .tokens
            .iter()
            .enumerate()
            .filter(|(_, &t)| t > 0.0)
            .map(|(tag, &t)| (ArenaFormatTag::from_u8(tag as u8), t))
            .collect();
        rows.sort_by(|a, b| b.1.total_cmp(&a.1));
        rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const Q4: u8 = ArenaFormatTag::Q4_0 as u8;
    const Q8: u8 = ArenaFormatTag::Q8_0 as u8;
    const BF16: u8 = ArenaFormatTag::BF16 as u8;

    /// A chunk's tokens split evenly over its bands, and a sequence's over its
    /// layers: two layers of one full chunk, K in Q4_0 and V in Q8_0 on the
    /// first and BF16 throughout on the second, are 32 tokens — 8 in each
    /// quantized format and 16 in BF16.
    #[test]
    fn tokens_split_over_bands_and_layers() {
        let mut c = FormatCensus::default();
        c.add_chunk(32, &[Q4, Q4], &[Q8, Q8], 2);
        c.add_chunk(32, &[BF16, BF16], &[BF16, BF16], 2);
        assert_eq!(
            c.rows(),
            vec![
                (ArenaFormatTag::BF16, 16.0),
                (ArenaFormatTag::Q8_0, 8.0),
                (ArenaFormatTag::Q4_0, 8.0),
            ]
        );
    }

    /// A partial chunk counts only the tokens it holds, and a chunk with no
    /// tags is kept, as untagged, so the total stays the tokens resident.
    #[test]
    fn partial_and_untagged_chunks_keep_the_total() {
        let mut c = FormatCensus::default();
        c.add_chunk(5, &[Q4], &[Q4], 1);
        c.add_chunk(3, &[], &[], 1);
        assert_eq!(
            c.rows(),
            vec![(ArenaFormatTag::Q4_0, 5.0), (ArenaFormatTag::Invalid, 3.0)]
        );
    }

    /// Nothing counted is nothing reported.
    #[test]
    fn an_empty_census_has_no_rows() {
        assert!(FormatCensus::default().rows().is_empty());
    }
}
