//! Rotary position embeddings with YaRN scaling.
//!
//! Mirrors `precompute_freqs_cis` / `apply_rotary_emb` in `inference/model.py`. The
//! rotation is **interleaved** (adjacent pairs `(x[2k], x[2k+1])` form a complex number),
//! matching `torch.view_as_complex(x.unflatten(-1, (-1, 2)))`. RoPE is applied only to
//! the trailing `rope_head_dim` dims of each head; the output projection later
//! de-rotates those dims (the `inverse` path here).

use candle::{DType, Device, Result, Tensor, D};

/// Host-side RoPE for one `(rope_dim, theta, original_seq_len)` setting.
///
/// Holds ONLY the YaRN-adjusted frequencies — `cos`/`sin` are computed per
/// call for exactly the requested positions. Nothing position-sized is ever
/// allocated, so the addressable context is unbounded (the §L fixed-budget
/// principle): a query at position 10⁶ costs the same as one at position 10.
/// The paged kernels compute their own RoPE in-kernel from the same
/// [`yarn_freqs`] values; this host path serves the indexer/compressor
/// projections and the reference forwards.
#[derive(Debug, Clone)]
pub struct RotaryCache {
    freqs: Vec<f64>,
    rope_dim: usize,
    device: Device,
}

/// The YaRN-adjusted inverse frequencies (`rope_dim / 2` values). When
/// `original_seq_len == 0` YaRN is disabled and these are the plain RoPE
/// frequencies `1 / theta^(2i/dim)`. This is the single source of truth — the
/// [`RotaryCache`] tables and the paged kernels' in-kernel RoPE both derive
/// from it.
pub fn yarn_freqs(
    rope_dim: usize,
    theta: f64,
    original_seq_len: usize,
    factor: f64,
    beta_fast: f64,
    beta_slow: f64,
) -> Vec<f64> {
    let half = rope_dim / 2;
    let mut freqs: Vec<f64> = (0..half)
        .map(|i| 1.0 / theta.powf((2 * i) as f64 / rope_dim as f64))
        .collect();

    if original_seq_len > 0 {
        let (low, high) =
            RotaryCache::correction_range(beta_fast, beta_slow, rope_dim, theta, original_seq_len);
        for (i, f) in freqs.iter_mut().enumerate() {
            // smooth = 1 - ramp; interpolate between scaled (/factor) and raw freqs.
            let ramp = RotaryCache::ramp(low, high, i);
            let smooth = 1.0 - ramp;
            *f = *f / factor * (1.0 - smooth) + *f * smooth;
        }
    }
    freqs
}

impl RotaryCache {
    /// `factor`/`beta_fast`/`beta_slow` are the YaRN parameters; when
    /// `original_seq_len == 0` YaRN is disabled and plain RoPE frequencies are
    /// used.
    pub fn new(
        rope_dim: usize,
        theta: f64,
        original_seq_len: usize,
        factor: f64,
        beta_fast: f64,
        beta_slow: f64,
        device: &Device,
    ) -> Result<Self> {
        let freqs = yarn_freqs(
            rope_dim,
            theta,
            original_seq_len,
            factor,
            beta_fast,
            beta_slow,
        );
        Ok(Self {
            freqs,
            rope_dim,
            device: device.clone(),
        })
    }

    /// `cos`/`sin` rows for exactly these positions, shape `[n, rope_dim/2]`.
    /// Same math the precomputed table used (`f64` angle, direct sincos), so
    /// every consumer's numerics are unchanged.
    fn cos_sin_for(&self, positions: &[u32]) -> Result<(Tensor, Tensor)> {
        let half = self.rope_dim / 2;
        let n = positions.len();
        let mut cos = vec![0f32; n * half];
        let mut sin = vec![0f32; n * half];
        for (t, &pos) in positions.iter().enumerate() {
            for (i, &f) in self.freqs.iter().enumerate() {
                let angle = pos as f64 * f;
                cos[t * half + i] = angle.cos() as f32;
                sin[t * half + i] = angle.sin() as f32;
            }
        }
        let cos = Tensor::from_vec(cos, (n, half), &self.device)?;
        let sin = Tensor::from_vec(sin, (n, half), &self.device)?;
        Ok((cos, sin))
    }

    fn correction_dim(num_rotations: f64, dim: usize, base: f64, max_seq: usize) -> f64 {
        dim as f64 * ((max_seq as f64) / (num_rotations * 2.0 * std::f64::consts::PI)).ln()
            / (2.0 * base.ln())
    }

    fn correction_range(
        low_rot: f64,
        high_rot: f64,
        dim: usize,
        base: f64,
        max_seq: usize,
    ) -> (f64, f64) {
        let low = Self::correction_dim(low_rot, dim, base, max_seq).floor();
        let high = Self::correction_dim(high_rot, dim, base, max_seq).ceil();
        (low.max(0.0), high.min((dim - 1) as f64))
    }

    /// `linear_ramp_factor(min, max, dim)[i]`, clamped to `[0, 1]`.
    fn ramp(min: f64, max: f64, i: usize) -> f64 {
        let max = if min == max { max + 0.001 } else { max };
        (((i as f64) - min) / (max - min)).clamp(0.0, 1.0)
    }

    /// Apply the rotation to the trailing `rope_dim` dims of `x` at the given contiguous
    /// position range `[start, start + seq)`. `x` has shape `[..., seq, rope_dim]`
    /// (the seq axis is `D::Minus2`). When `inverse` is true the conjugate rotation is
    /// applied (used to de-rotate attention output).
    pub fn apply(&self, x: &Tensor, start: usize, inverse: bool) -> Result<Tensor> {
        let seq = x.dim(D::Minus2)?;
        let positions: Vec<u32> = (start as u32..(start + seq) as u32).collect();
        let (cos, sin) = self.cos_sin_for(&positions)?;
        self.apply_with(x, &cos, &sin, inverse)
    }

    /// As [`Self::apply`] but at arbitrary (e.g. strided) positions — used for
    /// compressed KV entries, whose RoPE positions are the group-start token
    /// positions `0, ratio, 2·ratio, …`.
    pub fn apply_positions(&self, x: &Tensor, positions: &[u32], inverse: bool) -> Result<Tensor> {
        let (cos, sin) = self.cos_sin_for(positions)?;
        self.apply_with(x, &cos, &sin, inverse)
    }

    fn apply_with(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, inverse: bool) -> Result<Tensor> {
        let seq = x.dim(D::Minus2)?;
        let dims = x.dims().to_vec();
        let rope_dim = *dims.last().unwrap();
        debug_assert_eq!(rope_dim, self.rope_dim);
        let half = rope_dim / 2;

        // cos/sin for these positions, shaped to broadcast over every leading axis:
        // [1, .., seq, half].
        let mut bshape = vec![1usize; dims.len() - 1];
        bshape[dims.len() - 2] = seq;
        bshape.push(half);
        let cos = cos.reshape(bshape.clone())?.to_dtype(DType::F32)?;
        let sin = sin.reshape(bshape)?.to_dtype(DType::F32)?;

        let orig_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        // Split interleaved pairs: last dim -> [half, 2].
        let mut pshape = dims.clone();
        pshape.pop();
        pshape.push(half);
        pshape.push(2);
        let xp = x.reshape(pshape)?;
        let x0 = xp.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?; // even
        let x1 = xp.narrow(D::Minus1, 1, 1)?.squeeze(D::Minus1)?; // odd

        let (rot0, rot1) = if inverse {
            // conj: sin -> -sin
            (
                (x0.broadcast_mul(&cos)? + x1.broadcast_mul(&sin)?)?,
                (x1.broadcast_mul(&cos)? - x0.broadcast_mul(&sin)?)?,
            )
        } else {
            (
                (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?,
                (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?,
            )
        };

        // Re-interleave: stack -> [..., half, 2] -> [..., rope_dim].
        let stacked = Tensor::stack(&[rot0, rot1], D::Minus1)?;
        let out = stacked.reshape(dims)?;
        out.to_dtype(orig_dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{Device, IndexOp};

    /// Forward rotation then its inverse (conjugate) must recover the input — this is
    /// exactly the output-projection de-rotation the model relies on.
    #[test]
    fn forward_then_inverse_is_identity() -> Result<()> {
        let dev = Device::Cpu;
        let cache = RotaryCache::new(8, 10000.0, 0, 1.0, 32.0, 1.0, &dev)?;
        // [b=2, heads=3, seq=5, rope_dim=8] with seq on Minus2.
        let x = Tensor::randn(0f32, 1.0, (2, 3, 5, 8), &dev)?;
        let rotated = cache.apply(&x, 7, false)?;
        let back = cache.apply(&rotated, 7, true)?;
        let diff = (back - &x)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-4, "round-trip diff {diff}");
        Ok(())
    }

    /// GROUND TRUTH vs `model.py` `apply_rotary_emb`: for the 4D query, `freqs_cis` is indexed by
    /// the SEQUENCE position and broadcast over heads (`view(1, seq, 1, rd)`), so every head at
    /// token `t` rotates by position `t`. The attention ropes the query in `[b, h, s, rd]` layout
    /// (seq at `Minus2`); this asserts that layout rotates each head at token `t` by exactly
    /// position `t` — head-independent — matching the scalar complex-multiply. This is the
    /// absolute check the earlier "query roped by head index" bug failed.
    #[test]
    fn query_rope_is_token_position_broadcast_over_heads() -> Result<()> {
        let dev = Device::Cpu;
        let (rope_dim, theta) = (8usize, 10000.0f64);
        let cache = RotaryCache::new(rope_dim, theta, 0, 1.0, 32.0, 1.0, &dev)?;
        let (h, s) = (3usize, 5usize);
        // Query in the attention's roped layout: [b=1, heads, seq, rope_dim].
        let q = Tensor::randn(0f32, 1.0, (1, h, s, rope_dim), &dev)?;
        let out = cache.apply(&q, 0, false)?; // rope by seq position (Minus2 = s)
        let qv = q.i(0)?.to_vec3::<f32>()?; // [h, s, rd]
        let ov = out.i(0)?.to_vec3::<f32>()?; // [h, s, rd]
        let half = rope_dim / 2;
        for head in 0..h {
            for t in 0..s {
                for k in 0..half {
                    let freq = 1.0 / theta.powf((2 * k) as f64 / rope_dim as f64);
                    let angle = t as f64 * freq; // position = token t, INDEPENDENT of head
                    let (c, sn) = (angle.cos() as f32, angle.sin() as f32);
                    let (x0, x1) = (qv[head][t][2 * k], qv[head][t][2 * k + 1]);
                    let (e, o) = (x0 * c - x1 * sn, x0 * sn + x1 * c);
                    assert!(
                        (ov[head][t][2 * k] - e).abs() < 1e-5,
                        "head{head} t{t} k{k} even: {} vs {e}",
                        ov[head][t][2 * k]
                    );
                    assert!(
                        (ov[head][t][2 * k + 1] - o).abs() < 1e-5,
                        "head{head} t{t} k{k} odd: {} vs {o}",
                        ov[head][t][2 * k + 1]
                    );
                }
            }
        }
        Ok(())
    }

    /// The interleaved rotation matches the scalar complex-multiply formula at a known
    /// position, and YaRN-off frequencies equal `1/theta^(2i/dim)`.
    #[test]
    fn rotation_matches_scalar_formula() -> Result<()> {
        let dev = Device::Cpu;
        let (rope_dim, theta) = (8usize, 10000.0f64);
        let cache = RotaryCache::new(rope_dim, theta, 0, 1.0, 32.0, 1.0, &dev)?;
        let pos = 3usize;
        // single vector [1,1,1,rope_dim]
        let vals: Vec<f32> = (0..rope_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let x = Tensor::from_vec(vals.clone(), (1, 1, 1, rope_dim), &dev)?;
        let got = cache
            .apply(&x, pos, false)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        // Scalar reference.
        let half = rope_dim / 2;
        for k in 0..half {
            let freq = 1.0 / theta.powf((2 * k) as f64 / rope_dim as f64);
            let angle = pos as f64 * freq;
            let (c, s) = (angle.cos() as f32, angle.sin() as f32);
            let x0 = vals[2 * k];
            let x1 = vals[2 * k + 1];
            let e = x0 * c - x1 * s;
            let o = x0 * s + x1 * c;
            assert!(
                (got[2 * k] - e).abs() < 1e-5,
                "even {k}: {} vs {e}",
                got[2 * k]
            );
            assert!(
                (got[2 * k + 1] - o).abs() < 1e-5,
                "odd {k}: {} vs {o}",
                got[2 * k + 1]
            );
        }
        Ok(())
    }
}
