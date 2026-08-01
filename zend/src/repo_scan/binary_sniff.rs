//! Statistical text/binary classification of a byte sample.
//!
//! An extension allowlist does not guarantee a file's *content* is text — a
//! compiled object / CUDA fatbin / image checked in under a text extension
//! (e.g. `candle-flash-attn/precompiled/*.txt`) sails past the extension and
//! size gates, then carves into hundreds of garbage scopes that blow the ingest
//! co-batch's VRAM budget. This module answers "is this sample text?" the way
//! `git` and `file(1)` do — but by COUNTING non-text bytes and thresholding on
//! the ratio, not tripping on the first one: a real text file may carry a stray
//! NUL or a byte or two of mixed encoding, whereas a binary is *dense* with
//! them. It stays UTF-8 aware so non-English text (CJK, accents, emoji) is kept.

/// A sample is binary if its non-text bytes (NULs, non-whitespace control
/// characters, and invalid-UTF-8 bytes) exceed this percent of its length.
/// Real text — including non-English UTF-8 and legacy single-byte prose — sits
/// far below; a binary blob sits far above. Perl's `-T` heuristic uses the same
/// one-third cutoff.
const MAX_NONTEXT_PCT: usize = 30;

/// NULs tolerated regardless of the ratio. A text file may carry a stray NUL
/// (a trailing pad byte, an editor artifact); more than this is decisive even
/// when the overall non-text ratio stays low (a large file with a NUL-dense
/// region that the ratio alone would dilute below the cutoff).
const NUL_TOLERANCE: usize = 2;

/// Whether a byte `sample` (a file prefix, or a whole small file) is binary.
///
/// One UTF-8-aware pass tallies the non-text **bytes** (`nontext`), measured
/// against the sample's byte length:
///  * **NULs** — text never needs one; counted, and more than [`NUL_TOLERANCE`]
///    is decisive on its own (a NUL-dense region the ratio might dilute).
///  * **non-whitespace control scalars** (C0/C1 controls, DEL) — a binary that
///    still decodes as UTF-8 is dense with these; each contributes its UTF-8
///    byte length so the tally stays commensurate with the byte-length total.
///  * **invalid-UTF-8 bytes** — a truncated trailing multibyte sequence (the
///    sample boundary cutting a real char) is NOT counted; genuine mid-sample
///    invalid bytes are.
///
/// Binary iff NULs exceed [`NUL_TOLERANCE`] or the non-text ratio exceeds
/// [`MAX_NONTEXT_PCT`]. Both bounds are checked incrementally so a blob is
/// rejected the moment it is decided, without scanning the rest. An empty
/// sample is text (an empty file carves to nothing anyway).
///
/// Note: UTF-16/UTF-32-encoded text is NUL-dense and so classifies as binary —
/// intentional, since the ingest reads UTF-8 and would carve such a file to
/// garbage anyway.
pub fn is_binary_sample(sample: &[u8]) -> bool {
    if sample.is_empty() {
        return false;
    }
    let total = sample.len();
    // `nontext` grows monotonically and `total` is fixed, so once either bound
    // trips it stays tripped — every check below is a valid early exit.
    let over_ratio = |nontext: usize| nontext * 100 / total > MAX_NONTEXT_PCT;
    let mut nul = 0usize;
    let mut nontext = 0usize;

    let mut rest = sample;
    loop {
        let (valid, tail) = match std::str::from_utf8(rest) {
            Ok(valid) => (valid, None),
            // SAFETY: `valid_up_to()` bytes are validated UTF-8 by definition.
            Err(e) => (
                unsafe { std::str::from_utf8_unchecked(&rest[..e.valid_up_to()]) },
                Some((e.valid_up_to(), e.error_len())),
            ),
        };
        for c in valid.chars() {
            if c == '\0' {
                nul += 1;
                if nul > NUL_TOLERANCE {
                    return true; // NUL-dense — decided immediately
                }
                nontext += 1;
            } else if c.is_control() && !matches!(c, '\t' | '\n' | '\r' | '\u{0B}' | '\u{0C}') {
                nontext += c.len_utf8();
            }
        }
        match tail {
            // `bad` genuine invalid bytes mid-sample — count and step past them.
            Some((at, Some(bad))) => {
                nontext += bad;
                if over_ratio(nontext) {
                    return true;
                }
                rest = &rest[at + bad..];
            }
            // A multibyte char truncated by the sample boundary — not a binary
            // signal; stop.
            Some((_, None)) => break,
            None => break,
        }
    }

    over_ratio(nontext)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_is_text() {
        assert!(!is_binary_sample(b""));
    }

    #[test]
    fn ascii_source_is_text() {
        assert!(!is_binary_sample(
            b"fn main() {\n    println!(\"hello\");\n}\n"
        ));
    }

    #[test]
    fn non_english_utf8_is_text() {
        // Japanese + accents + emoji — all valid UTF-8, zero non-text bytes.
        assert!(!is_binary_sample(
            "こんにちは — café — 🚀 données\n".as_bytes()
        ));
    }

    #[test]
    fn a_couple_stray_nuls_are_tolerated() {
        // A long ASCII file with two stray NULs (e.g. trailing pad) stays text.
        let mut s = b"perfectly normal text file, mostly ascii\n".to_vec();
        s.extend(std::iter::repeat(b'x').take(500));
        s.push(0);
        s.push(0);
        assert!(!is_binary_sample(&s));
    }

    #[test]
    fn many_nuls_are_binary() {
        // Three NULs exceeds NUL_TOLERANCE even without a high overall ratio.
        let mut s = b"header text".to_vec();
        s.extend(std::iter::repeat(b'a').take(500));
        s.extend_from_slice(&[0, 0, 0]);
        assert!(is_binary_sample(&s));
    }

    #[test]
    fn nul_dense_blob_is_binary() {
        // The shape of a fatbin `*.txt`: ELF magic + lots of NULs + some ASCII.
        assert!(is_binary_sample(b"\x7fELF\x00\x00\x00\x00fatbin\x00\x00 code"));
    }

    #[test]
    fn c1_controls_counted_by_byte_length() {
        // 100 C1 controls (U+0080, 2 UTF-8 bytes each = 200 bytes) + 200 ASCII
        // bytes = 400 total. Byte-accurate: 200 non-text / 400 = 50% ⇒ binary. A
        // naive 1-per-char tally would undercount to 100/400 = 25% and wrongly
        // pass it as text — this pins the `len_utf8()` accounting.
        let mut s = String::new();
        for _ in 0..100 {
            s.push('\u{80}'); // C1 control, 2 UTF-8 bytes, not whitespace
        }
        s.push_str(&"a".repeat(200));
        assert_eq!(s.len(), 400);
        assert!(is_binary_sample(s.as_bytes()));
    }

    #[test]
    fn control_dense_utf8_is_binary() {
        // NUL-free but packed with C0 control bytes (0x01..0x08, 0x0E..0x1F).
        let mut blob = Vec::new();
        for _ in 0..200 {
            blob.extend_from_slice(&[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]);
            blob.push(b'A');
        }
        assert!(is_binary_sample(&blob));
    }

    #[test]
    fn high_byte_non_utf8_blob_is_binary() {
        // NUL-free invalid-UTF-8 bytes dominated by high/control bytes.
        let blob: Vec<u8> = (0u16..2000).map(|i| (0x80 | (i & 0x7F)) as u8).collect();
        assert!(is_binary_sample(&blob));
    }

    #[test]
    fn a_few_invalid_utf8_bytes_are_tolerated() {
        // Legacy single-byte prose: mostly ASCII with a handful of lone 0xE9
        // ('é' in Latin-1) bytes MID-text — invalid UTF-8, but far below the
        // ratio, so kept. Exercises the mid-sample invalid-byte counting.
        let mut s = b"The ".to_vec();
        for _ in 0..5 {
            s.push(0xE9); // lone e-acute, invalid UTF-8
            s.extend_from_slice(b" cafe was busy. ");
            s.extend(std::iter::repeat(b'x').take(40));
        }
        assert!(!is_binary_sample(&s));
    }

    #[test]
    fn truncated_trailing_multibyte_is_text() {
        // A 4-byte emoji sliced after its first 3 bytes must not read as binary.
        let full = "ok 🚀".as_bytes();
        let cut = &full[..full.len() - 1];
        assert!(std::str::from_utf8(cut).is_err(), "precondition: cut is invalid");
        assert!(!is_binary_sample(cut));
    }
}
