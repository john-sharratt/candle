//! Making a *partial* rotary run on kernels that only know full-width RoPE.
//!
//! The paged attention kernels rotate every head dim, pairing `d` with
//! `d + head_dim/2` and reading `(cos, sin)` from a `[pos, head_dim]` table
//! at entry `d`. Qwen3.5 sets `partial_rotary_factor = 0.25`: only the first
//! `rope_dim` (64 of 256) dims rotate, pairing `j` with `j + rope_dim/2`, and
//! the remaining 192 pass through untouched.
//!
//! Two things therefore disagree — which dims rotate, and which dims pair —
//! and a table of `cos = 1, sin = 0` only fixes the first. The pairing is
//! fixed instead by **permuting the head dims**, so that the kernel's own
//! `(d, d + head_dim/2)` pairing lands on the dims this model wants paired:
//!
//! ```text
//!   kernel slot      0 ..  r/2   holds model dims   0     ..  r/2
//!   kernel slot  hd/2 .. hd/2+r/2 holds model dims   r/2   ..  r
//!   every other slot                holds a pass-through dim,
//!                                   with a (1, 0) table entry
//! ```
//!
//! so kernel pair `(j, j + hd/2)` is model pair `(j, j + r/2)` for
//! `j < r/2`, at table frequency `j` — exactly ggml's `rope_neox` over
//! `n_rot` — and every other kernel pair rotates by the identity.
//!
//! This is safe because attention only ever contracts Q against K over the
//! head dim: `q · k` is invariant under any permutation applied to *both*.
//! `V` is deliberately **not** permuted — its dims flow through to the
//! context and out through the output projection, so they must stay in model
//! order.
//!
//! **Where it is applied.** On the projection *outputs*, per forward, not on
//! the weights at load. Folding it into the weight rows would be free at run
//! time, and rows are independent block sequences so it is meaningful on a
//! quantized tensor — but candle has no row-gather for `QTensor` (only
//! `concat_rows_cuda`), and dequantizing to permute would either cost a
//! requantization's worth of quality or keep an F32 copy of every Q/K
//! projection resident. So Q and K are gathered after projection, which is
//! exact and cheap to reason about.
//!
//! The standing optimisation, when it is worth the kernel work, is to
//! template the paged kernels on `n_rot` and delete this module: the
//! permutation exists only because those kernels hard-code the pairing at
//! `head_dim/2`.

use candle::{DType, Device, LiveTensor, Result, Tensor};

/// The head-dim permutation and matching RoPE table for one geometry.
#[derive(Debug, Clone)]
pub struct RotaryLayout {
    head_dim: usize,
    rope_dim: usize,
    /// `perm[k]` is the model dim the kernel sees at its slot `k`.
    perm: Vec<usize>,
    /// `perm` as a device tensor, built once at load.
    ///
    /// The gather runs per attention layer per forward; rebuilding its index
    /// each time would be a host→device transfer on the hot path, which is
    /// exactly what invariant 3 forbids. `None` for an identity layout,
    /// which never gathers.
    index: Option<Tensor>,
}

impl RotaryLayout {
    /// Build the layout for `head_dim` with `rope_dim` rotating dims,
    /// materialising the gather index on `dev`.
    ///
    /// `rope_dim == head_dim` yields the identity permutation, so a
    /// full-rotary model pays nothing and reads the same as it always did.
    pub fn new(head_dim: usize, rope_dim: usize, dev: &Device) -> Result<Self> {
        if !head_dim.is_multiple_of(2) || head_dim == 0 {
            candle::bail!("head_dim {head_dim} must be even and nonzero");
        }
        if !rope_dim.is_multiple_of(2) || rope_dim == 0 || rope_dim > head_dim {
            candle::bail!(
                "rope_dim {rope_dim} must be even, nonzero and at most head_dim {head_dim}"
            );
        }
        let (half, r_half) = (head_dim / 2, rope_dim / 2);

        // Slots that carry rotating dims, and their model dims.
        let mut perm = vec![usize::MAX; head_dim];
        for j in 0..r_half {
            perm[j] = j; // low half of the rotary block
            perm[half + j] = r_half + j; // high half, the kernel's partner
        }
        // Everything else is a pass-through dim; the order among the free
        // slots is arbitrary (they all rotate by the identity), so fill them
        // in ascending model order to keep the mapping reproducible.
        let mut free = (rope_dim..head_dim).collect::<Vec<_>>().into_iter();
        for slot in perm.iter_mut() {
            if *slot == usize::MAX {
                *slot = free.next().expect("free dims exactly fill the free slots");
            }
        }
        debug_assert!(free.next().is_none());
        let index = if rope_dim == head_dim {
            None
        } else {
            Some(Tensor::from_vec(
                perm.iter().map(|&d| d as u32).collect::<Vec<_>>(),
                head_dim,
                dev,
            )?)
        };
        Ok(Self {
            head_dim,
            rope_dim,
            perm,
            index,
        })
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn rope_dim(&self) -> usize {
        self.rope_dim
    }

    /// `perm[k]` = the model dim occupying kernel slot `k`.
    pub fn permutation(&self) -> &[usize] {
        &self.perm
    }

    /// Whether this layout is a no-op (full-width rotary).
    pub fn is_identity(&self) -> bool {
        self.rope_dim == self.head_dim
    }

    /// The gather index, or `None` when this layout is the identity.
    pub fn index(&self) -> Option<&Tensor> {
        self.index.as_ref()
    }

    fn check_last(&self, last_dim: usize) -> Result<()> {
        if last_dim != self.head_dim {
            candle::bail!(
                "rotary permute: last dim {last_dim} is not head_dim {}",
                self.head_dim
            );
        }
        Ok(())
    }

    /// Reorder the last axis of `x` (`[.., head_dim]`) into kernel order.
    pub fn permute_last_dim(&self, x: &Tensor) -> Result<Tensor> {
        let Some(idx) = &self.index else {
            return Ok(x.clone());
        };
        let last = x.rank() - 1;
        self.check_last(x.dim(last)?)?;
        x.index_select(idx, last)
    }

    /// [`Self::permute_last_dim`] over a wave-scoped activation.
    ///
    /// The gather's result inherits `'w` from its operand, so a Q/K
    /// projection carved from the wave stays on the wave.
    pub fn permute_last_dim_live<'w>(&self, x: &LiveTensor<'w>) -> Result<LiveTensor<'w>> {
        let Some(idx) = &self.index else {
            return Ok(x.clone());
        };
        let last = x.rank() - 1;
        self.check_last(x.dim(last)?)?;
        x.index_select(idx, last)
    }

    /// The `[max_pos, head_dim]` interleaved `(cos, sin)` table the kernels
    /// read, with identity entries on every pass-through pair.
    pub fn rope_table(
        &self,
        max_pos: usize,
        theta: f32,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor> {
        let half = self.head_dim / 2;
        let r_half = self.rope_dim / 2;
        let mut vals = Vec::with_capacity(max_pos * self.head_dim);
        for pos in 0..max_pos {
            for j in 0..half {
                if j < r_half {
                    // ggml's `rope_neox` frequency over the ROTARY width.
                    let inv = 1f32 / theta.powf(2.0 * j as f32 / self.rope_dim as f32);
                    let ang = pos as f32 * inv;
                    vals.push(ang.cos());
                    vals.push(ang.sin());
                } else {
                    // Pass-through pair: rotation by zero.
                    vals.push(1.0);
                    vals.push(0.0);
                }
            }
        }
        Tensor::from_vec(vals, (max_pos, self.head_dim), dev)?.to_dtype(dtype)
    }

    /// Split `(cos, sin)` at the given absolute positions, `[n, head_dim/2]`
    /// each, with identity entries on every pass-through pair.
    ///
    /// The same angles [`Self::rope_table`] interleaves, de-interleaved. The
    /// paged kernels read the interleaved table and never touch these, but the
    /// attention parameters carry both and a caller that reached for the split
    /// form would otherwise get another model's frequencies. Built per wave
    /// over the wave's own positions, which is a few hundred rows — not the
    /// whole context, which is what the interleaved table has to cover.
    pub fn rope_cos_sin(
        &self,
        positions: &[u32],
        theta: f32,
        dtype: DType,
        dev: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let half = self.head_dim / 2;
        let r_half = self.rope_dim / 2;
        let mut cos = Vec::with_capacity(positions.len() * half);
        let mut sin = Vec::with_capacity(positions.len() * half);
        for &pos in positions {
            for j in 0..half {
                if j < r_half {
                    let inv = 1f32 / theta.powf(2.0 * j as f32 / self.rope_dim as f32);
                    let ang = pos as f32 * inv;
                    cos.push(ang.cos());
                    sin.push(ang.sin());
                } else {
                    cos.push(1.0);
                    sin.push(0.0);
                }
            }
        }
        let shape = (positions.len(), half);
        Ok((
            Tensor::from_vec(cos, shape, dev)?.to_dtype(dtype)?,
            Tensor::from_vec(sin, shape, dev)?.to_dtype(dtype)?,
        ))
    }
}

// The oracle these compare against is `qwen35::attention::RopeTables`, which
// lives in the CUDA-gated hybrid lineage.
#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::models::qwen35::attention::RopeTables;

    fn dev() -> Device {
        Device::Cpu
    }

    /// What the paged kernel does: pair `d` with `d + head_dim/2`, taking
    /// `(cos, sin)` from table entry `d`, over the WHOLE head.
    fn kernel_rope(x: &[f32], table: &[f32], head_dim: usize) -> Vec<f32> {
        let half = head_dim / 2;
        let mut out = vec![0f32; head_dim];
        for j in 0..half {
            let (c, s) = (table[2 * j], table[2 * j + 1]);
            let (lo, hi) = (x[j], x[j + half]);
            out[j] = lo * c - hi * s;
            out[j + half] = hi * c + lo * s;
        }
        out
    }

    #[test]
    fn permutation_is_a_bijection() {
        let l = RotaryLayout::new(256, 64, &dev()).unwrap();
        let mut seen = vec![false; 256];
        for &d in l.permutation() {
            assert!(!seen[d], "dim {d} appears twice");
            seen[d] = true;
        }
        assert!(seen.into_iter().all(|s| s));
    }

    #[test]
    fn full_width_rotary_is_the_identity_layout() {
        let l = RotaryLayout::new(128, 128, &dev()).unwrap();
        assert!(l.is_identity());
        assert_eq!(l.permutation(), (0..128).collect::<Vec<_>>());
        assert!(
            l.index().is_none(),
            "an identity layout must not allocate a gather index"
        );
    }

    #[test]
    fn refuses_impossible_geometries() {
        let d = dev();
        assert!(RotaryLayout::new(256, 0, &d).is_err());
        assert!(RotaryLayout::new(256, 65, &d).is_err(), "odd rope_dim");
        assert!(
            RotaryLayout::new(256, 512, &d).is_err(),
            "wider than the head"
        );
        assert!(RotaryLayout::new(0, 0, &d).is_err());
    }

    /// THE property this whole module exists for: running the kernel's
    /// full-width RoPE over permuted dims, with this table, reproduces the
    /// model's partial rotary exactly — up to the permutation, which
    /// attention cancels because it contracts Q against K.
    #[test]
    fn kernel_rope_over_permuted_dims_equals_partial_rotary() {
        let (head_dim, rope_dim, theta) = (256usize, 64usize, 1e7f32);
        let l = RotaryLayout::new(head_dim, rope_dim, &dev()).unwrap();
        let table = l
            .rope_table(8, theta, DType::F32, &dev())
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        // The validated reference: partial rotary over the model's dims.
        let reference = RopeTables::new(rope_dim, theta, 8, &dev()).unwrap();

        for pos in [0usize, 1, 5, 7] {
            let x: Vec<f32> = (0..head_dim)
                .map(|i| ((i * 37 % 101) as f32) - 50.0)
                .collect();

            // Reference path: partial rotary in model order.
            let xt = Tensor::from_vec(x.clone(), (1, 1, head_dim), &dev()).unwrap();
            let want = reference
                .apply(&xt, pos)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();

            // Kernel path: permute, then full-width rotate with our table.
            let permuted: Vec<f32> = l.permutation().iter().map(|&d| x[d]).collect();
            let rotated = kernel_rope(&permuted, &table[pos], head_dim);
            // Undo the permutation to compare in model order.
            let mut got = vec![0f32; head_dim];
            for (slot, &d) in l.permutation().iter().enumerate() {
                got[d] = rotated[slot];
            }

            for i in 0..head_dim {
                assert!(
                    (got[i] - want[i]).abs() < 1e-4,
                    "pos {pos} dim {i}: kernel path {} vs partial rotary {}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    /// The pass-through dims must be untouched at every position — this is
    /// what the `(1, 0)` table entries buy, and a wrong entry would show up
    /// as a slow drift rather than an obvious break.
    #[test]
    fn pass_through_dims_are_invariant_to_position() {
        let (head_dim, rope_dim) = (256usize, 64usize);
        let l = RotaryLayout::new(head_dim, rope_dim, &dev()).unwrap();
        let table = l
            .rope_table(16, 1e7, DType::F32, &dev())
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let x: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.5 - 3.0).collect();
        let permuted: Vec<f32> = l.permutation().iter().map(|&d| x[d]).collect();
        for pos in [0usize, 3, 15] {
            let rotated = kernel_rope(&permuted, &table[pos], head_dim);
            for (slot, &d) in l.permutation().iter().enumerate() {
                if d >= rope_dim {
                    assert!(
                        (rotated[slot] - x[d]).abs() < 1e-6,
                        "pass-through dim {d} moved at pos {pos}"
                    );
                }
            }
        }
    }

    /// The runtime gather reorders the last axis and leaves the leading
    /// axes (tokens, heads) alone.
    #[test]
    fn permute_last_dim_reorders_only_the_head_axis() {
        let (head_dim, rope_dim, rows) = (8usize, 4usize, 3usize);
        let l = RotaryLayout::new(head_dim, rope_dim, &dev()).unwrap();
        // Element (r, d) holds r*100 + d, so both axes are readable.
        let vals: Vec<f32> = (0..rows)
            .flat_map(|r| (0..head_dim).map(move |d| (r * 100 + d) as f32))
            .collect();
        let x = Tensor::from_vec(vals, (rows, head_dim), &dev()).unwrap();
        let got = l.permute_last_dim(&x).unwrap().to_vec2::<f32>().unwrap();
        for (r, row) in got.iter().enumerate() {
            for (slot, &d) in l.permutation().iter().enumerate() {
                assert_eq!(row[slot] as usize, r * 100 + d, "row {r} slot {slot}");
            }
        }
    }

    /// A full-rotary geometry must be a true no-op, so adopting this
    /// machinery cannot perturb an existing model.
    #[test]
    fn identity_layout_returns_the_input_untouched() {
        let l = RotaryLayout::new(64, 64, &dev()).unwrap();
        let x = Tensor::from_vec(
            (0..128).map(|r| r as f32).collect::<Vec<_>>(),
            (2, 64),
            &dev(),
        )
        .unwrap();
        let p = l.permute_last_dim(&x).unwrap();
        assert_eq!(
            x.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            p.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }
}
