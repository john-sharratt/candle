//! One quantized band chunk's bytes ⇄ floats, in the layout the attention
//! kernels read.
//!
//! A band holds `sub` of a head's dims (its palette's ranks) for the chunk's
//! `CHUNK_SIZE` tokens. Quantized, it is **token-oriented**: one 32-element
//! block per rank, the block holding that rank's 32 tokens, blocks in rank
//! order — what the palette seal writes and every attention kernel's
//! `ArenaAccessor` / `I8BlockQuad` addresses (`block = rank`, element =
//! token). The floats on this side are **token-major** `[token][rank]`, the
//! layout a float band stores, so a band reads and writes the same way
//! whatever its tag.
//!
//! The values are the band's STORED values — outer-scaled, as the encoder was
//! handed them. Dividing by the palette's outer scale is the reader's job
//! (`read_contiguous`), exactly as the kernels divide at read time.
//!
//! Q0_V is calibrated per side, so its codec takes `is_k`: a K band decodes
//! and encodes against the K codebook. The generic `GgmlType` codec has no
//! side and would read every Q0_V band against the V codebook.

use candle::quantized::k_quants::{decode_blocks_q0_v, encode_block_q0_v, BlockQ0V};
use candle::quantized::{ggml_file::qtensor_from_ggml, GgmlDType, QTensor};
use candle::{Device, Result, Tensor};

use crate::kv_cache::QuantFormat;
use crate::CHUNK_SIZE;

/// Decode one quantized band's `bytes` into token-major `[CHUNK_SIZE][sub]`
/// floats.
pub(crate) fn decode_band(
    bytes: &[u8],
    fmt: QuantFormat,
    is_k: bool,
    sub: usize,
) -> Result<Vec<f32>> {
    let rank_major = decode_rank_major(bytes, fmt, is_k, sub)?;
    Ok(transpose(&rank_major, sub, CHUNK_SIZE))
}

/// Encode token-major `[CHUNK_SIZE][sub]` floats into one quantized band's
/// bytes.
pub(crate) fn encode_band(
    values: &[f32],
    fmt: QuantFormat,
    is_k: bool,
    sub: usize,
) -> Result<Vec<u8>> {
    if values.len() != CHUNK_SIZE * sub {
        candle::bail!(
            "encode_band: {} values for a {CHUNK_SIZE}×{sub} band",
            values.len()
        );
    }
    let rank_major = transpose(values, CHUNK_SIZE, sub);
    encode_rank_major(&rank_major, fmt, is_k, sub)
}

/// Write token-major `rows` (`[n][sub]`, tokens `tok0..tok0 + n`) into one
/// quantized band's `bytes` in place.
///
/// R16 stores each token raw — a rank's block is its 32 K halves, then 32
/// captured-Q halves — so its tokens are written as they are and every other
/// byte, the Q capture included, is left alone. Any other format's block is
/// one encoding of all 32 tokens, so the band is decoded, the rows replaced
/// and the band encoded again.
pub(crate) fn patch_band(
    bytes: &mut [u8],
    fmt: QuantFormat,
    is_k: bool,
    sub: usize,
    tok0: usize,
    rows: &[f32],
) -> Result<()> {
    let n = rows.len() / sub;
    if rows.len() != n * sub || tok0 + n > CHUNK_SIZE {
        candle::bail!(
            "patch_band: {} values at token {tok0} do not fit a {CHUNK_SIZE}×{sub} band",
            rows.len()
        );
    }
    let want = band_bytes(fmt, sub);
    if bytes.len() < want {
        candle::bail!(
            "patch_band: {} bytes for a {want}-byte {fmt:?} band",
            bytes.len()
        );
    }
    if fmt.to_ggml_dtype() == GgmlDType::R16 {
        let block = fmt.to_ggml_dtype().type_size();
        for t in 0..n {
            for r in 0..sub {
                let at = r * block + (tok0 + t) * 2;
                bytes[at..at + 2]
                    .copy_from_slice(&half::f16::from_f32(rows[t * sub + r]).to_le_bytes());
            }
        }
        return Ok(());
    }
    let mut values = decode_band(&bytes[..want], fmt, is_k, sub)?;
    values[tok0 * sub..(tok0 + n) * sub].copy_from_slice(rows);
    bytes[..want].copy_from_slice(&encode_band(&values, fmt, is_k, sub)?);
    Ok(())
}

/// The byte size of one quantized band of `sub` ranks.
pub(crate) fn band_bytes(fmt: QuantFormat, sub: usize) -> usize {
    let ggml = fmt.to_ggml_dtype();
    (CHUNK_SIZE * sub / ggml.block_size()) * ggml.type_size()
}

/// `[rows][cols]` → `[cols][rows]`.
fn transpose(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = src[r * cols + c];
        }
    }
    out
}

/// Rank-major `[sub][CHUNK_SIZE]`: block `r` is rank `r`'s tokens.
fn decode_rank_major(bytes: &[u8], fmt: QuantFormat, is_k: bool, sub: usize) -> Result<Vec<f32>> {
    check_block_geometry(fmt)?;
    let want = band_bytes(fmt, sub);
    if bytes.len() < want {
        candle::bail!(
            "decode_band: {} bytes for a {want}-byte {fmt:?} band",
            bytes.len()
        );
    }
    let ggml = fmt.to_ggml_dtype();
    if ggml == GgmlDType::Q0_V {
        let blocks: Vec<BlockQ0V> = bytes[..want]
            .chunks_exact(2)
            .map(|b| BlockQ0V::from_le_bytes([b[0], b[1]]))
            .collect();
        let mut out = vec![0f32; CHUNK_SIZE * sub];
        if is_k {
            decode_blocks_q0_v::<true>(&blocks, &mut out);
        } else {
            decode_blocks_q0_v::<false>(&blocks, &mut out);
        }
        return Ok(out);
    }
    qtensor_from_ggml(ggml, &bytes[..want], vec![CHUNK_SIZE * sub], &Device::Cpu)?
        .dequantize(&Device::Cpu)?
        .to_vec1::<f32>()
}

fn encode_rank_major(values: &[f32], fmt: QuantFormat, is_k: bool, sub: usize) -> Result<Vec<u8>> {
    check_block_geometry(fmt)?;
    let ggml = fmt.to_ggml_dtype();
    if ggml == GgmlDType::Q0_V {
        let encode = if is_k {
            encode_block_q0_v::<true>
        } else {
            encode_block_q0_v::<false>
        };
        return Ok(values
            .chunks_exact(CHUNK_SIZE)
            .flat_map(|block| encode(block).to_le_bytes())
            .collect());
    }
    let src = Tensor::from_slice(values, CHUNK_SIZE * sub, &Device::Cpu)?;
    Ok(QTensor::quantize(&src, ggml)?.data()?.into_owned())
}

/// A band's block is one rank's `CHUNK_SIZE` tokens, so the codec's block must
/// be exactly that long.
fn check_block_geometry(fmt: QuantFormat) -> Result<()> {
    let block = fmt.to_ggml_dtype().block_size();
    if block != CHUNK_SIZE {
        candle::bail!(
            "band codec: {fmt:?} has {block}-element blocks; a band's block is one rank's \
             {CHUNK_SIZE} tokens"
        );
    }
    Ok(())
}
