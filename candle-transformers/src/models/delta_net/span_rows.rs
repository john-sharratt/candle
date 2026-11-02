//! A span's rows as the fused prefill scan reads them, per DeltaNet layer: the
//! convolved `[Q|K|V]` row and the raw gate projections
//! (`docs/recurrent_reply_calibration.md` §3).
//!
//! The scan derives every per-token quantity of the delta rule from these and
//! the layer's own weights, and none of them depends on the state it runs
//! against. So the rows a span produced in one conversation can be run through
//! another conversation's state: the correction `β·(v − S·k)` is computed again
//! against that state, never carried.
//!
//! On the device a layer's rows are F32, as the scan reads them
//! ([`SpanRows`]). Kept, they are BF16 ([`ReplyRows`]), in one blob:
//!
//! ```text
//! "RRC1" | n_layers u32 | n_tokens u32 | conv_dim u32 | n_v_heads u32 | layers [n_layers] u32
//! per layer: qkv [n_tokens][conv_dim] | alpha [n_tokens][n_v_heads] | beta [n_tokens][n_v_heads]
//! ```
//!
//! little-endian throughout, each element a BF16.

use candle::{DType, Device, Result, Tensor};
use half::bf16;

const MAGIC: &[u8; 4] = b"RRC1";

/// One DeltaNet layer's rows of a span, on the device: `qkv [T, conv_dim]`,
/// `alpha [T, n_v_heads]`, `beta [T, n_v_heads]`, all F32.
#[derive(Debug, Clone)]
pub struct SpanRows {
    pub qkv: Tensor,
    pub alpha: Tensor,
    pub beta: Tensor,
}

impl SpanRows {
    /// Uninitialised rows for a span of `len` tokens: a capture writes every one.
    pub fn alloc(len: usize, conv_dim: usize, n_v_heads: usize, dev: &Device) -> Result<Self> {
        Ok(Self {
            qkv: Tensor::empty((len, conv_dim), DType::F32, dev)?,
            alpha: Tensor::empty((len, n_v_heads), DType::F32, dev)?,
            beta: Tensor::empty((len, n_v_heads), DType::F32, dev)?,
        })
    }

    /// How many tokens the span holds.
    pub fn tokens(&self) -> Result<usize> {
        Ok(self.qkv.dims2()?.0)
    }
}

/// A span's rows across every DeltaNet layer, as kept: BF16, one run per layer.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplyRows {
    /// The trunk index of each layer, in the order the runs are held.
    pub layers: Vec<u32>,
    pub tokens: usize,
    pub conv_dim: usize,
    pub n_v_heads: usize,
    /// Per layer: `qkv`, then `alpha`, then `beta`, each token-major.
    pub data: Vec<Vec<bf16>>,
}

impl ReplyRows {
    fn layer_elems(&self) -> usize {
        self.tokens * (self.conv_dim + 2 * self.n_v_heads)
    }

    /// Read every layer's rows off the device, narrowed to BF16 — the format
    /// the rows are kept in.
    pub fn from_device(layers: &[(u32, SpanRows)]) -> Result<Self> {
        let Some((_, first)) = layers.first() else {
            candle::bail!("span rows: no layer to keep");
        };
        let (tokens, conv_dim) = first.qkv.dims2()?;
        let n_v_heads = first.alpha.dims2()?.1;
        let mut data = Vec::with_capacity(layers.len());
        for (layer, rows) in layers {
            if rows.qkv.dims2()? != (tokens, conv_dim)
                || rows.alpha.dims2()? != (tokens, n_v_heads)
                || rows.beta.dims2()? != (tokens, n_v_heads)
            {
                candle::bail!("span rows: layer {layer} holds rows of another shape");
            }
            let mut run = Vec::with_capacity(tokens * (conv_dim + 2 * n_v_heads));
            for t in [&rows.qkv, &rows.alpha, &rows.beta] {
                run.extend(t.to_dtype(DType::BF16)?.flatten_all()?.to_vec1::<bf16>()?);
            }
            data.push(run);
        }
        Ok(Self {
            layers: layers.iter().map(|(l, _)| *l).collect(),
            tokens,
            conv_dim,
            n_v_heads,
            data,
        })
    }

    /// Every layer's rows on `dev`, widened to the F32 the scan reads.
    pub fn to_device(&self, dev: &Device) -> Result<Vec<(u32, SpanRows)>> {
        let (t, c, h) = (self.tokens, self.conv_dim, self.n_v_heads);
        self.layers
            .iter()
            .zip(&self.data)
            .map(|(&layer, run)| {
                let field = |from: usize, cols: usize| -> Result<Tensor> {
                    Tensor::from_slice(&run[from..from + t * cols], (t, cols), dev)?
                        .to_dtype(DType::F32)
                };
                Ok((
                    layer,
                    SpanRows {
                        qkv: field(0, c)?,
                        alpha: field(t * c, h)?,
                        beta: field(t * (c + h), h)?,
                    },
                ))
            })
            .collect()
    }

    /// The blob these rows are kept as.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            20 + 4 * self.layers.len() + 2 * self.data.len() * self.layer_elems(),
        );
        out.extend_from_slice(MAGIC);
        for n in [
            self.layers.len(),
            self.tokens,
            self.conv_dim,
            self.n_v_heads,
        ] {
            out.extend_from_slice(&(n as u32).to_le_bytes());
        }
        for l in &self.layers {
            out.extend_from_slice(&l.to_le_bytes());
        }
        for run in &self.data {
            for v in run {
                out.extend_from_slice(&v.to_bits().to_le_bytes());
            }
        }
        out
    }

    /// Rows read back from [`Self::encode`]'s blob.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let word = |at: usize| -> Result<u32> {
            bytes
                .get(at..at + 4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .ok_or_else(|| candle::Error::Msg(format!("span rows: blob ends at {at}")))
        };
        if bytes.get(..4) != Some(MAGIC) {
            candle::bail!("span rows: the blob does not open with {MAGIC:?}");
        }
        let n_layers = word(4)? as usize;
        let mut rows = Self {
            layers: Vec::with_capacity(n_layers),
            tokens: word(8)? as usize,
            conv_dim: word(12)? as usize,
            n_v_heads: word(16)? as usize,
            data: Vec::with_capacity(n_layers),
        };
        for i in 0..n_layers {
            rows.layers.push(word(20 + 4 * i)?);
        }
        let mut at = 20 + 4 * n_layers;
        let elems = rows.layer_elems();
        if bytes.len() != at + 2 * elems * n_layers {
            candle::bail!(
                "span rows: a blob of {} bytes holds {n_layers} layers of {elems} elements",
                bytes.len()
            );
        }
        for _ in 0..n_layers {
            rows.data.push(
                bytes[at..at + 2 * elems]
                    .chunks_exact(2)
                    .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])))
                    .collect(),
            );
            at += 2 * elems;
        }
        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_by_two() -> ReplyRows {
        // Two tokens, conv_dim 2, one V head: 2·(2 + 2·1) = 8 elements a layer.
        let run =
            |base: f32| -> Vec<bf16> { (0..8).map(|i| bf16::from_f32(base + i as f32)).collect() };
        ReplyRows {
            layers: vec![0, 3],
            tokens: 2,
            conv_dim: 2,
            n_v_heads: 1,
            data: vec![run(1.0), run(-1.0)],
        }
    }

    /// The blob, byte for byte: header, layer indices, then each layer's BF16s.
    #[test]
    fn rows_encode_to_their_blob() {
        let blob = two_by_two().encode();
        let mut want: Vec<u8> = b"RRC1".to_vec();
        for w in [2u32, 2, 2, 1, 0, 3] {
            want.extend_from_slice(&w.to_le_bytes());
        }
        // bf16(1.0) = 0x3F80, bf16(2.0) = 0x4000, …; bf16(-1.0) = 0xBF80.
        for bits in [
            0x3F80u16, 0x4000, 0x4040, 0x4080, 0x40A0, 0x40C0, 0x40E0, 0x4100,
        ] {
            want.extend_from_slice(&bits.to_le_bytes());
        }
        for bits in [
            0xBF80u16, 0x0000, 0x3F80, 0x4000, 0x4040, 0x4080, 0x40A0, 0x40C0,
        ] {
            want.extend_from_slice(&bits.to_le_bytes());
        }
        assert_eq!(blob, want);
        assert_eq!(ReplyRows::decode(&blob).unwrap(), two_by_two());
    }

    #[test]
    fn a_short_or_foreign_blob_is_refused() {
        let blob = two_by_two().encode();
        assert!(ReplyRows::decode(&blob[..blob.len() - 1]).is_err());
        assert!(ReplyRows::decode(b"RRC2").is_err());
    }

    /// Off the device and back: the fields land where the scan reads them.
    #[test]
    fn rows_round_trip_through_the_device() {
        let dev = Device::Cpu;
        let rows = two_by_two();
        let on = rows.to_device(&dev).unwrap();
        assert_eq!(
            on[0].1.qkv.to_vec2::<f32>().unwrap(),
            [[1.0, 2.0], [3.0, 4.0]]
        );
        assert_eq!(on[0].1.alpha.to_vec2::<f32>().unwrap(), [[5.0], [6.0]]);
        assert_eq!(on[0].1.beta.to_vec2::<f32>().unwrap(), [[7.0], [8.0]]);
        assert_eq!(on[1].0, 3);
        assert_eq!(ReplyRows::from_device(&on).unwrap(), rows);
    }
}
