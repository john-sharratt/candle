//! The band codec's layout: a quantized band is token-oriented — block r is
//! rank r's 32 tokens, encoded on their own — which is what every attention
//! kernel addresses. These tests pin that byte for byte against the formats'
//! own block codecs, and pin Q0_V's side.

use super::super::band_codec::{band_bytes, decode_band, encode_band, patch_band};
use super::super::palette_layout::{is_identity, palette_columns};
use crate::kv_cache::QuantFormat;
use crate::CHUNK_SIZE;
use candle::quantized::k_quants::{decode_blocks_q0_v, encode_block_q0_v, BlockQ0V};
use candle::quantized::QTensor;
use candle::{Device, Result, Tensor};
use strum::IntoEnumIterator;

const SUB: usize = 32;

/// A token-major `[CHUNK_SIZE][SUB]` band whose every (token, rank) cell is
/// distinct and whose ranks differ in scale, so a transposed or mis-sided
/// codec produces different bytes.
fn band_values() -> Vec<f32> {
    (0..CHUNK_SIZE * SUB)
        .map(|i| {
            let (t, r) = (i / SUB, i % SUB);
            ((t as f32 * 0.37 + r as f32 * 1.13).sin()) * (0.05 + r as f32 * 0.03)
        })
        .collect()
}

/// Rank r's 32 tokens.
fn rank_column(values: &[f32], r: usize) -> Vec<f32> {
    (0..CHUNK_SIZE).map(|t| values[t * SUB + r]).collect()
}

#[test]
fn quantized_band_is_one_block_per_rank_of_its_tokens() -> Result<()> {
    let values = band_values();
    for fmt in [QuantFormat::Q8_0, QuantFormat::Q4_0, QuantFormat::Q2_0] {
        let got = encode_band(&values, fmt, false, SUB)?;
        let mut want = Vec::new();
        for r in 0..SUB {
            let col = Tensor::from_vec(rank_column(&values, r), CHUNK_SIZE, &Device::Cpu)?;
            want.extend_from_slice(&QTensor::quantize(&col, fmt.to_ggml_dtype())?.data()?);
        }
        assert_eq!(got.len(), band_bytes(fmt, SUB), "{fmt:?} band size");
        assert_eq!(got, want, "{fmt:?} bytes are not rank-major token blocks");
    }
    Ok(())
}

#[test]
fn decode_places_each_rank_block_in_its_column() -> Result<()> {
    let values = band_values();
    let fmt = QuantFormat::Q8_0;
    let bytes = encode_band(&values, fmt, false, SUB)?;
    let got = decode_band(&bytes, fmt, false, SUB)?;
    for r in 0..SUB {
        let col = Tensor::from_vec(rank_column(&values, r), CHUNK_SIZE, &Device::Cpu)?;
        let deq = QTensor::quantize(&col, fmt.to_ggml_dtype())?
            .dequantize(&Device::Cpu)?
            .to_vec1::<f32>()?;
        for t in 0..CHUNK_SIZE {
            assert_eq!(
                got[t * SUB + r].to_bits(),
                deq[t].to_bits(),
                "token {t} rank {r}"
            );
        }
    }
    Ok(())
}

/// Every KV format the arena can tag a band with has a band codec: the bytes
/// are the format's own block encoding of each rank's tokens, and they decode
/// back into that rank's column. Q0_V is the side-aware exception, pinned by
/// its own test below.
#[test]
fn every_kv_format_round_trips_through_its_block_codec() -> Result<()> {
    let values = band_values();
    for fmt in QuantFormat::iter().filter(|&f| f != QuantFormat::Q0_V) {
        let ggml = fmt.to_ggml_dtype();
        let bytes = encode_band(&values, fmt, true, SUB)?;
        assert_eq!(bytes.len(), band_bytes(fmt, SUB), "{fmt:?} band size");
        let got = decode_band(&bytes, fmt, true, SUB)?;
        for r in 0..SUB {
            let col = Tensor::from_vec(rank_column(&values, r), CHUNK_SIZE, &Device::Cpu)?;
            let q = QTensor::quantize(&col, ggml)?;
            let block = band_bytes(fmt, 1);
            assert_eq!(
                &bytes[r * block..(r + 1) * block],
                &q.data()?[..],
                "{fmt:?} rank {r} block"
            );
            let deq = q.dequantize(&Device::Cpu)?.to_vec1::<f32>()?;
            for t in 0..CHUNK_SIZE {
                assert_eq!(
                    got[t * SUB + r].to_bits(),
                    deq[t].to_bits(),
                    "{fmt:?} token {t} rank {r}"
                );
            }
        }
    }
    Ok(())
}

/// Q0_V's codebook depends on the side: a K band encodes and decodes against
/// the K codebook, a V band against the V one, and the two differ.
#[test]
fn q0_v_band_uses_its_sides_codebook() -> Result<()> {
    let values = band_values();
    let k = encode_band(&values, QuantFormat::Q0_V, true, SUB)?;
    let v = encode_band(&values, QuantFormat::Q0_V, false, SUB)?;
    let want_k: Vec<u8> = (0..SUB)
        .flat_map(|r| encode_block_q0_v::<true>(&rank_column(&values, r)).to_le_bytes())
        .collect();
    let want_v: Vec<u8> = (0..SUB)
        .flat_map(|r| encode_block_q0_v::<false>(&rank_column(&values, r)).to_le_bytes())
        .collect();
    assert_eq!(k, want_k, "K band");
    assert_eq!(v, want_v, "V band");
    assert_ne!(k, v, "the corpus must tell the two codebooks apart");

    let got = decode_band(&k, QuantFormat::Q0_V, true, SUB)?;
    let blocks: Vec<BlockQ0V> = k
        .chunks_exact(2)
        .map(|b| BlockQ0V::from_le_bytes([b[0], b[1]]))
        .collect();
    let mut rank_major = vec![0f32; CHUNK_SIZE * SUB];
    decode_blocks_q0_v::<true>(&blocks, &mut rank_major);
    for r in 0..SUB {
        for t in 0..CHUNK_SIZE {
            assert_eq!(
                got[t * SUB + r].to_bits(),
                rank_major[r * CHUNK_SIZE + t].to_bits()
            );
        }
    }
    Ok(())
}

/// R16 patches write the K halves of the named tokens only; the captured-Q
/// halves and every other token's bytes stay as they were.
#[test]
fn r16_patch_writes_only_its_tokens_k_halves() -> Result<()> {
    let fmt = QuantFormat::R16;
    let mut bytes: Vec<u8> = (0..band_bytes(fmt, SUB)).map(|i| (i % 251) as u8).collect();
    let before = bytes.clone();
    let rows: Vec<f32> = (0..2 * SUB).map(|i| i as f32 * 0.5 - 7.0).collect();
    patch_band(&mut bytes, fmt, true, SUB, 5, &rows)?;
    let block = bytes.len() / SUB;
    for r in 0..SUB {
        for t in 0..CHUNK_SIZE {
            let at = r * block + t * 2;
            let k = [bytes[at], bytes[at + 1]];
            if t == 5 || t == 6 {
                let want = half::f16::from_f32(rows[(t - 5) * SUB + r]).to_le_bytes();
                assert_eq!(k, want, "token {t} rank {r} K half");
            } else {
                assert_eq!(
                    k,
                    [before[at], before[at + 1]],
                    "token {t} rank {r} untouched"
                );
            }
            let q = r * block + 64 + t * 2;
            assert_eq!(
                [bytes[q], bytes[q + 1]],
                [before[q], before[q + 1]],
                "Q capture kept"
            );
        }
    }
    Ok(())
}

#[test]
fn palette_columns_follow_the_map() {
    // head_dim 16, 4 palettes of 4: dims assigned 3,0,1,2 repeating.
    let pal: Vec<u8> = (0..4).map(|_| 0b10_01_00_11u8).collect();
    let cols = palette_columns(&pal, 16, 4);
    // dim 0 → palette 3 rank 0 → column 12; dim 1 → palette 0 rank 0 → 0; …
    assert_eq!(
        cols,
        vec![12, 0, 4, 8, 13, 1, 5, 9, 14, 2, 6, 10, 15, 3, 7, 11]
    );
    assert!(!is_identity(&cols));
    // The identity map: dims 0..4 in palette 0, 4..8 in palette 1, …
    let ident = [0b00_00_00_00u8, 0b01_01_01_01, 0b10_10_10_10, 0b11_11_11_11];
    assert!(is_identity(&palette_columns(&ident, 16, 4)));
}
