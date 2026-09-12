//! What one KV band holds over the token window a decode kernel read from it.
//!
//! A band is one `(head, palette, K|V)` sub-head of one chunk: `SUB_HEAD_DIM`
//! dimensions × 32 tokens. The kernel addresses it two ways (`ArenaAccessor`):
//!
//! * **float formats** are token-major — element `(t, d)` at `(t·SUB + d)·esz`;
//! * **block formats** are dim-major, one 32-token block per dimension — dim
//!   `d`'s block at `d·block_bytes`. `R16` holds the 32 raw f16 values first and
//!   32 reserved u16 after, and only the first half is ever read.
//!
//! Only the window `offset..offset + len` is examined. Under `tensor-assert`
//! every freshly claimed slot is stamped `0xFF`, so tokens outside the window are
//! legitimately poison; inside it, poison means a value the kernel read was never
//! written.

use std::ops::Range;

use candle::Result;
use candle_nn::kv_cache::ArenaFormatTag;

/// Tokens per chunk, and so per band block.
const CHUNK_TOKENS: usize = 32;

/// What a band held inside the read window.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct BandScan {
    /// Whether this format's values could be examined at all.
    pub scanned: bool,
    /// Values examined: elements for float formats and R16, per-dimension
    /// block scales for the scaled block formats.
    pub elems: usize,
    /// Of those, how many are NaN or inf.
    pub nonfinite: usize,
    /// Of those, how many are exactly the allocation poison.
    pub poison: usize,
}

/// How a format's bytes are laid out, as far as finiteness can be judged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Layout {
    /// Token-major floats of `esz` bytes.
    Float { esz: usize },
    /// Dim-major `R16`: 32 f16 then 32 reserved u16 per dimension.
    R16,
    /// Dim-major blocks whose first two bytes are an f16 scale.
    ScaledBlock { block_bytes: usize },
    /// A layout whose values this scan does not decode.
    Opaque,
}

fn layout(fmt: u8) -> Layout {
    let is = |t: ArenaFormatTag| fmt == t.as_u8();
    if is(ArenaFormatTag::F32) {
        Layout::Float { esz: 4 }
    } else if is(ArenaFormatTag::F16) || is(ArenaFormatTag::BF16) {
        Layout::Float { esz: 2 }
    } else if is(ArenaFormatTag::F8E4M3) || is(ArenaFormatTag::F8E5M2) {
        Layout::Float { esz: 1 }
    } else if is(ArenaFormatTag::R16) {
        Layout::R16
    } else {
        // Block sizes from `ArenaAccessor::get_quant_block_bytes`; each of these
        // opens with an f16 `d`.
        let block_bytes = [
            (ArenaFormatTag::Q4_0, 18),
            (ArenaFormatTag::Q4_1, 20),
            (ArenaFormatTag::Q5_0, 22),
            (ArenaFormatTag::Q5_1, 24),
            (ArenaFormatTag::Q8_0, 34),
            (ArenaFormatTag::Q8_1, 36),
            (ArenaFormatTag::Q8_KS, 36),
            (ArenaFormatTag::Q2_0, 10),
            (ArenaFormatTag::Q3_0, 14),
        ]
        .into_iter()
        .find(|(t, _)| fmt == t.as_u8())
        .map(|(_, b)| b);
        match block_bytes {
            Some(block_bytes) => Layout::ScaledBlock { block_bytes },
            None => Layout::Opaque,
        }
    }
}

/// Bytes a band of `fmt` occupies for `sub_head_dim` dimensions, or `None` for a
/// format this scan cannot size.
pub(crate) fn band_payload_bytes(fmt: u8, sub_head_dim: usize) -> Option<usize> {
    match layout(fmt) {
        Layout::Float { esz } => Some(CHUNK_TOKENS * sub_head_dim * esz),
        Layout::R16 => Some(sub_head_dim * 128),
        Layout::ScaledBlock { block_bytes } => Some(sub_head_dim * block_bytes),
        Layout::Opaque => None,
    }
}

fn f16_bad(h: u16) -> bool {
    h & 0x7C00 == 0x7C00
}
fn bf16_bad(h: u16) -> bool {
    h & 0x7F80 == 0x7F80
}

/// Examine `bytes` — one band of `fmt` — over tokens `window`.
pub(crate) fn scan_band(
    fmt: u8,
    bytes: &[u8],
    sub_head_dim: usize,
    window: Range<usize>,
) -> Result<BandScan> {
    if window.end > CHUNK_TOKENS || window.start > window.end {
        candle::bail!("band scan: window {window:?} is not inside a {CHUNK_TOKENS}-token chunk");
    }
    let Some(need) = band_payload_bytes(fmt, sub_head_dim) else {
        return Ok(BandScan::default());
    };
    if bytes.len() < need {
        candle::bail!("band scan: {} bytes, a band of format {fmt} needs {need}", bytes.len());
    }
    let u16_at = |o: usize| u16::from_le_bytes([bytes[o], bytes[o + 1]]);
    let mut s = BandScan { scanned: true, ..BandScan::default() };
    match layout(fmt) {
        Layout::Float { esz } => {
            let bf16 = fmt == ArenaFormatTag::BF16.as_u8();
            let e5m2 = fmt == ArenaFormatTag::F8E5M2.as_u8();
            for t in window {
                for d in 0..sub_head_dim {
                    let o = (t * sub_head_dim + d) * esz;
                    let (bad, poison) = match esz {
                        4 => {
                            let w = u32::from_le_bytes(bytes[o..o + 4].try_into().expect("4"));
                            (!f32::from_bits(w).is_finite(), w == u32::MAX)
                        }
                        2 => {
                            let h = u16_at(o);
                            (if bf16 { bf16_bad(h) } else { f16_bad(h) }, h == u16::MAX)
                        }
                        _ => {
                            let b = bytes[o];
                            let bad = if e5m2 { b & 0x7C == 0x7C } else { b & 0x7F == 0x7F };
                            (bad, b == u8::MAX)
                        }
                    };
                    s.elems += 1;
                    s.nonfinite += bad as usize;
                    s.poison += poison as usize;
                }
            }
        }
        Layout::R16 => {
            for d in 0..sub_head_dim {
                for t in window.clone() {
                    let h = u16_at(d * 128 + t * 2);
                    s.elems += 1;
                    s.nonfinite += f16_bad(h) as usize;
                    s.poison += (h == u16::MAX) as usize;
                }
            }
        }
        Layout::ScaledBlock { block_bytes } => {
            if !window.is_empty() {
                for d in 0..sub_head_dim {
                    let h = u16_at(d * block_bytes);
                    s.elems += 1;
                    s.nonfinite += f16_bad(h) as usize;
                    s.poison += (h == u16::MAX) as usize;
                }
            }
        }
        Layout::Opaque => unreachable!("sized above"),
    }
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    const F16_NAN: u16 = 0x7E00;
    const F16_ONE: u16 = 0x3C00;

    fn put16(b: &mut [u8], o: usize, v: u16) {
        b[o..o + 2].copy_from_slice(&v.to_le_bytes());
    }

    /// A band filled with the allocation poison, as a freshly claimed slot is.
    fn poisoned(fmt: u8, sub: usize) -> Vec<u8> {
        vec![0xFF; band_payload_bytes(fmt, sub).unwrap()]
    }

    #[test]
    fn payload_sizes_follow_the_accessor() {
        assert_eq!(band_payload_bytes(ArenaFormatTag::F16.as_u8(), 64), Some(32 * 64 * 2));
        assert_eq!(band_payload_bytes(ArenaFormatTag::R16.as_u8(), 64), Some(64 * 128));
        assert_eq!(band_payload_bytes(ArenaFormatTag::Q8_0.as_u8(), 64), Some(64 * 34));
        assert_eq!(band_payload_bytes(ArenaFormatTag::Q4_KS.as_u8(), 64), None);
    }

    /// Token-major f16: tokens 1..3 written, everything else still poison.
    #[test]
    fn an_f16_band_counts_only_its_window() {
        let f16 = ArenaFormatTag::F16.as_u8();
        let mut b = poisoned(f16, 2);
        for t in 1..3 {
            for d in 0..2 {
                put16(&mut b, (t * 2 + d) * 2, F16_ONE);
            }
        }
        let clean = scan_band(f16, &b, 2, 1..3).unwrap();
        assert_eq!(clean, BandScan { scanned: true, elems: 4, nonfinite: 0, poison: 0 });

        // A NaN the kernel read, at (t = 2, d = 1).
        put16(&mut b, (2 * 2 + 1) * 2, F16_NAN);
        let bad = scan_band(f16, &b, 2, 1..3).unwrap();
        assert_eq!(bad, BandScan { scanned: true, elems: 4, nonfinite: 1, poison: 0 });

        // Widening the window onto unwritten token 3 reads poison.
        let past = scan_band(f16, &b, 2, 1..4).unwrap();
        assert_eq!(past, BandScan { scanned: true, elems: 6, nonfinite: 3, poison: 2 });
    }

    /// Dim-major R16: the reserved half of each block is never counted.
    #[test]
    fn an_r16_band_reads_values_not_the_reserved_half() {
        let r16 = ArenaFormatTag::R16.as_u8();
        let mut b = poisoned(r16, 2);
        for d in 0..2 {
            for t in 0..5 {
                put16(&mut b, d * 128 + t * 2, F16_ONE);
            }
        }
        // The reserved u16 space (bytes 64..128 of each block) stays 0xFF.
        assert_eq!(
            scan_band(r16, &b, 2, 0..5).unwrap(),
            BandScan { scanned: true, elems: 10, nonfinite: 0, poison: 0 }
        );
        // An unwritten value inside the window is poison.
        put16(&mut b, 128 + 4 * 2, u16::MAX);
        assert_eq!(
            scan_band(r16, &b, 2, 0..5).unwrap(),
            BandScan { scanned: true, elems: 10, nonfinite: 1, poison: 1 }
        );
    }

    /// Each 16-bit format is judged by its own exponent field. `0x7C00` is f16
    /// inf but a finite bf16 (~2^121); the reverse cannot be shown, because
    /// every bf16 non-finite pattern also fills the f16 exponent.
    #[test]
    fn f16_and_bf16_are_judged_by_their_own_exponents() {
        let f16 = ArenaFormatTag::F16.as_u8();
        let bf16 = ArenaFormatTag::BF16.as_u8();
        let mut b = vec![0u8; band_payload_bytes(f16, 1).unwrap()];
        put16(&mut b, 0, 0x7C00);
        assert_eq!(scan_band(f16, &b, 1, 0..1).unwrap().nonfinite, 1);
        assert_eq!(scan_band(bf16, &b, 1, 0..1).unwrap().nonfinite, 0);
        put16(&mut b, 0, 0x7F80);
        assert_eq!(scan_band(bf16, &b, 1, 0..1).unwrap().nonfinite, 1);
    }

    #[test]
    fn a_scaled_block_is_judged_by_each_dimensions_scale() {
        let q8 = ArenaFormatTag::Q8_0.as_u8();
        let mut b = vec![0u8; band_payload_bytes(q8, 2).unwrap()];
        put16(&mut b, 0, F16_NAN);
        put16(&mut b, 34, F16_ONE);
        assert_eq!(
            scan_band(q8, &b, 2, 0..9).unwrap(),
            BandScan { scanned: true, elems: 2, nonfinite: 1, poison: 0 }
        );
        // Nothing read, nothing judged.
        assert_eq!(scan_band(q8, &b, 2, 4..4).unwrap().elems, 0);
    }

    #[test]
    fn an_opaque_format_is_reported_unscanned_not_clean() {
        let q4ks = ArenaFormatTag::Q4_KS.as_u8();
        assert_eq!(scan_band(q4ks, &[0u8; 8], 2, 0..8).unwrap(), BandScan::default());
    }

    #[test]
    fn a_short_band_or_a_window_past_the_chunk_is_refused() {
        let f16 = ArenaFormatTag::F16.as_u8();
        assert!(scan_band(f16, &[0u8; 10], 2, 0..1).is_err());
        assert!(scan_band(f16, &poisoned(f16, 2), 2, 30..33).is_err());
    }
}
