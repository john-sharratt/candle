//! Turning an image's rows of Q and K by their place in the image.
//!
//! The attention kernels rotate every key by its slot at read time, and keys
//! are stored un-rotated so a chunk can be placed at any slot. An image row's
//! multimodal position is its slot plus a fixed offset per axis that depends
//! only on the row's place in its image ([`image_delta`]), and RoPE composes:
//! `R(slot) · R(δ) = R(slot + δ)`. So each image row's Q and K are turned once,
//! in place, by `δ` — before the kernel rotates Q and before K is written — and
//! every reader's rotation by the slot then lands on the multimodal position,
//! wherever the chunk is later placed (`docs/flash_next_vision.md` §4.4).
//!
//! The head is in the kernels' slot order (`rotary_layout`): rotary pair `f`
//! is slots `(f, f + head_dim/2)` for `f < rope_dim/2`, at frequency `f`, and
//! every other pair passes through. So one NeoX rotation over the whole head,
//! with `(cos, sin)` of the offset at the rotary pairs and `(1, 0)` elsewhere,
//! is the turn — `candle_nn::rotary_emb::rope_thd` over `[1, rows, heads, dim]`.

use std::cell::RefCell;

use candle::{DType, LiveTensor, Result, Tensor};

use super::geometry::{image_delta, trunk_axis, Grid};

/// A run of one image's consecutive rows in a group of rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageRun {
    /// The run's first row in the group.
    pub row: usize,
    /// Rows in the run.
    pub len: usize,
    /// The image's grid.
    pub grid: Grid,
    /// The run's first row's index in its image.
    pub first: usize,
}

/// The `(cos, sin)` tables for one run, `[len, head_dim / 2]` each, row-major.
pub fn run_tables(run: &ImageRun, head_dim: usize, rope_dim: usize, theta: f64) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let pairs = rope_dim / 2;
    let inv: Vec<f64> = (0..pairs)
        .map(|f| 1.0 / theta.powf(2.0 * f as f64 / rope_dim as f64))
        .collect();
    let mut cos = vec![1.0f32; run.len * half];
    let mut sin = vec![0.0f32; run.len * half];
    for j in 0..run.len {
        let d = image_delta(run.grid, run.first + j);
        for (f, w) in inv.iter().enumerate() {
            let a = d[trunk_axis(f)] as f64 * w;
            cos[j * half + f] = a.cos() as f32;
            sin[j * half + f] = a.sin() as f32;
        }
    }
    (cos, sin)
}

/// A group's image runs and their tables, built once per forward and read by
/// every attention layer.
pub struct ImageRotation {
    runs: Vec<ImageRun>,
    /// Each run's `(cos, sin)`, `[len, head_dim / 2]` F32, uploaded where the
    /// caller placed them (a wave's forward span).
    tables: Vec<(Tensor, Tensor)>,
    /// The tables in the type of the rows they turn, converted at the first
    /// layer that turns and reused by every later one. A conversion inherits
    /// its operand's arena, so these sit beside the tables.
    typed: RefCell<Option<(DType, Vec<(Tensor, Tensor)>)>>,
}

impl ImageRotation {
    /// Build every run's tables and hand each to `upload` (`values, (rows,
    /// cols)` → an F32 tensor where it should live).
    pub fn new(
        runs: Vec<ImageRun>,
        head_dim: usize,
        rope_dim: usize,
        theta: f64,
        upload: impl Fn(Vec<f32>, (usize, usize)) -> Result<Tensor>,
    ) -> Result<Self> {
        let half = head_dim / 2;
        let mut tables = Vec::with_capacity(runs.len());
        for r in &runs {
            let (c, s) = run_tables(r, head_dim, rope_dim, theta);
            tables.push((upload(c, (r.len, half))?, upload(s, (r.len, half))?));
        }
        Ok(Self {
            runs,
            tables,
            typed: RefCell::new(None),
        })
    }

    pub fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }

    /// Turn the runs' rows of `x` (`[rows, heads, head_dim]`, contiguous) in
    /// place.
    pub fn apply<'w>(&self, x: &LiveTensor<'w>) -> Result<()> {
        if self.runs.is_empty() {
            return Ok(());
        }
        let dtype = x.dtype();
        let mut cell = self.typed.borrow_mut();
        if cell.as_ref().map(|(d, _)| *d) != Some(dtype) {
            // The one conversion the design calls for: a table built in F32,
            // into the type of the rows it turns, once per forward.
            let typed = self
                .tables
                .iter()
                .map(|(c, s)| Ok((c.to_dtype(dtype)?, s.to_dtype(dtype)?)))
                .collect::<Result<Vec<_>>>()?;
            *cell = Some((dtype, typed));
        }
        let (_, up) = cell.as_ref().expect("converted above");
        for (run, (c, s)) in self.runs.iter().zip(up) {
            let rows = x.narrow(0, run.row, run.len)?.unsqueeze(0)?;
            let turned = candle_nn::rotary_emb::rope_thd(&rows, c, s)?;
            x.slice_set(&turned.squeeze(0)?, 0, run.row)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qwen_vision::geometry::FACTOR;
    use candle::Device;

    /// A 1 × 3 image's last row: each rotary pair turns by its axis's offset,
    /// every other pair by nothing.
    #[test]
    fn a_rows_table_turns_each_pair_by_its_axis() {
        let grid = Grid {
            height: FACTOR,
            width: 3 * FACTOR,
        };
        // n 3, m 3: row 2 is (r 0, c 2) → δ = (3 − 3 − 2, 0 + 0 − 2, 0 + 2 − 2).
        assert_eq!(image_delta(grid, 2), [-2, -2, 0]);
        let run = ImageRun {
            row: 0,
            len: 1,
            grid,
            first: 2,
        };
        let (cos, sin) = run_tables(&run, 8, 6, 100.0);
        assert_eq!(cos.len(), 4);
        // Pair 0: T, δ −2, frequency 1.
        assert!((cos[0] - (-2f64).cos() as f32).abs() < 1e-7);
        assert!((sin[0] - (-2f64).sin() as f32).abs() < 1e-7);
        // Pair 1: H, δ −2, frequency 100^(-1/3).
        let f1 = 100f64.powf(-2.0 / 6.0);
        assert!((sin[1] - (-2.0 * f1).sin() as f32).abs() < 1e-7);
        // Pair 2: W, δ 0.
        assert_eq!((cos[2], sin[2]), (1.0, 0.0));
        // Pair 3 is past the rotary pairs.
        assert_eq!((cos[3], sin[3]), (1.0, 0.0));
    }

    /// Turning by δ and then by the slot is turning by the slot plus δ: the
    /// property the whole design rests on, checked on the actual op.
    #[test]
    fn a_turn_then_the_slot_rotation_is_the_multimodal_rotation() {
        let dev = Device::Cpu;
        let grid = Grid {
            height: 2 * FACTOR,
            width: 2 * FACTOR,
        };
        let (head_dim, rope_dim, theta) = (8, 8, 50.0);
        let run = ImageRun {
            row: 1,
            len: 4,
            grid,
            first: 0,
        };
        let rows = 6;
        let vals: Vec<f32> = (0..rows * 2 * head_dim)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
            .collect();
        let x = Tensor::from_vec(vals.clone(), (rows, 2, head_dim), &dev).unwrap();
        let rot = ImageRotation::new(vec![run], head_dim, rope_dim, theta, |v, s| {
            Tensor::from_vec(v, s, &Device::Cpu)
        })
        .unwrap();
        rot.apply(&x).unwrap();
        let turned = x.to_vec3::<f32>().unwrap();
        let orig = Tensor::from_vec(vals, (rows, 2, head_dim), &dev)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();
        // Rows outside the run are untouched.
        assert_eq!(turned[0], orig[0]);
        assert_eq!(turned[5], orig[5]);
        // Inside: slot rotation of the turned row equals a direct rotation at
        // slot + δ per axis.
        let half = head_dim / 2;
        let rotate = |v: &[f32], angle: &dyn Fn(usize) -> f64| -> Vec<f32> {
            let mut out = v.to_vec();
            for f in 0..half {
                let (c, s) = (angle(f).cos(), angle(f).sin());
                let (a, b) = (v[f] as f64, v[f + half] as f64);
                out[f] = (a * c - b * s) as f32;
                out[f + half] = (a * s + b * c) as f32;
            }
            out
        };
        let inv = |f: usize| 1.0 / theta.powf(2.0 * f as f64 / rope_dim as f64);
        for j in 0..4 {
            let slot = 10.0 + j as f64;
            let d = image_delta(grid, j);
            for h in 0..2 {
                let via_turn = rotate(&turned[1 + j][h], &|f| slot * inv(f));
                let direct = rotate(&orig[1 + j][h], &|f| {
                    (slot + d[trunk_axis(f)] as f64) * inv(f)
                });
                for (a, b) in via_turn.iter().zip(&direct) {
                    assert!((a - b).abs() < 1e-5, "row {j} head {h}: {a} vs {b}");
                }
            }
        }
    }
}
