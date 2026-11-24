//! An image's rows as bytes, for the substrate: the grid, the row width, then
//! every value in bf16 little-endian — the precision the reference runs the
//! tower at, half the bytes of F32.
//!
//! `ZVR1 | height u32 | width u32 | row width u32 | values bf16…`

use candle_transformers::models::qwen_vision::{Grid, ImageRows};
use half::bf16;

const MAGIC: &[u8; 4] = b"ZVR1";

/// `rows` as bytes.
pub fn encode_rows(rows: &ImageRows) -> Vec<u8> {
    let mut out = Vec::with_capacity(16 + rows.values.len() * 2);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(rows.grid.height as u32).to_le_bytes());
    out.extend_from_slice(&(rows.grid.width as u32).to_le_bytes());
    out.extend_from_slice(&(rows.width as u32).to_le_bytes());
    for v in &rows.values {
        out.extend_from_slice(&bf16::from_f32(*v).to_le_bytes());
    }
    out
}

/// The rows `bytes` hold, or why not.
pub fn decode_rows(bytes: &[u8]) -> Result<ImageRows, String> {
    if bytes.len() < 16 || &bytes[..4] != MAGIC {
        return Err("not an image's rows".into());
    }
    let word = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap()) as usize;
    let grid = Grid {
        height: word(4),
        width: word(8),
    };
    let width = word(12);
    let want = grid.tokens() * width;
    let body = &bytes[16..];
    if body.len() != want * 2 {
        return Err(format!("{} bytes of rows for {want} values", body.len()));
    }
    let values = body
        .chunks_exact(2)
        .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect();
    Ok(ImageRows {
        grid,
        width,
        values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_transformers::models::qwen_vision::geometry::FACTOR;

    /// The exact bytes for a 32 × 32 image — one token — with two-wide rows.
    #[test]
    fn rows_encode_to_their_exact_bytes() {
        let rows = ImageRows {
            grid: Grid {
                height: FACTOR,
                width: FACTOR,
            },
            width: 2,
            values: vec![1.0, -2.0],
        };
        let bytes = encode_rows(&rows);
        assert_eq!(
            bytes,
            [
                b'Z', b'V', b'R', b'1', 32, 0, 0, 0, 32, 0, 0, 0, 2, 0, 0, 0, 0x80, 0x3f, 0x00,
                0xc0
            ]
        );
        assert_eq!(decode_rows(&bytes).unwrap(), rows);
    }

    #[test]
    fn short_or_foreign_bytes_are_refused() {
        assert!(decode_rows(b"ZVR1").is_err());
        assert!(decode_rows(&[0u8; 20]).is_err());
        let mut bytes = encode_rows(&ImageRows {
            grid: Grid {
                height: FACTOR,
                width: FACTOR,
            },
            width: 2,
            values: vec![1.0, 2.0],
        });
        bytes.pop();
        assert!(decode_rows(&bytes).is_err());
    }
}
