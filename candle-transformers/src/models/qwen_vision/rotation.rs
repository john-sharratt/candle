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
//! every other pair passes through. The turn is one launch per attention
//! layer over every image row of the group, Q and K together
//! (`simple/image_rope_rows.cu`), from a table of `(cos, sin)` per listed row
//! and rotary pair built once per forward.

use std::ffi::c_void;

use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::{CudaStorage, DType, LiveTensor, Result, Storage, Tensor};
use candle_kernels::simple::image_rope_rows::{
    run_image_rope_rows, IMAGE_ROPE_BAD_SHAPE, IMAGE_ROPE_BF16, IMAGE_ROPE_F16, IMAGE_ROPE_F32,
    IMAGE_ROPE_UNSUPPORTED_DTYPE,
};
use half::{bf16, f16};

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

/// A group's image rows and their turns, on the host: each listed row, and
/// its `(cos, sin)` at every rotary pair, `[rows.len(), pairs]` row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct RotationTables {
    pub rows: Vec<u32>,
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub pairs: usize,
    /// One past the last listed row: the rows a tensor must have to be turned.
    pub rows_end: usize,
}

/// The turns of every row of `runs`, for a rotary width of `rope_dim` at
/// base `theta`.
pub fn rotation_tables(runs: &[ImageRun], rope_dim: usize, theta: f64) -> RotationTables {
    let pairs = rope_dim / 2;
    let inv: Vec<f64> = (0..pairs)
        .map(|f| 1.0 / theta.powf(2.0 * f as f64 / rope_dim as f64))
        .collect();
    let n: usize = runs.iter().map(|r| r.len).sum();
    let mut out = RotationTables {
        rows: Vec::with_capacity(n),
        cos: Vec::with_capacity(n * pairs),
        sin: Vec::with_capacity(n * pairs),
        pairs,
        rows_end: runs.iter().map(|r| r.row + r.len).max().unwrap_or(0),
    };
    for run in runs {
        for j in 0..run.len {
            out.rows.push((run.row + j) as u32);
            let d = image_delta(run.grid, run.first + j);
            for (f, w) in inv.iter().enumerate() {
                let a = d[trunk_axis(f)] as f64 * w;
                out.cos.push(a.cos() as f32);
                out.sin.push(a.sin() as f32);
            }
        }
    }
    out
}

/// A group's image rows and their tables on the device, built once per
/// forward and read by every attention layer.
pub struct ImageRotation {
    /// U32 `[n]`: the listed rows.
    rows: Tensor,
    /// One past the last of them.
    rows_end: usize,
    /// F32 `[n, pairs]` each.
    cos: Tensor,
    sin: Tensor,
}

/// `t`'s dense `[rows, heads, dim]` geometry, or why it cannot be turned.
fn heads_of(t: &LiveTensor<'_>, what: &str) -> Result<(usize, usize, usize)> {
    if !t.is_contiguous() {
        candle::bail!(
            "image rotation: {what} is {:?} stride {:?}, not dense",
            t.dims(),
            t.stride()
        );
    }
    t.dims3()
}

/// A storage's CUDA side, for a pointer taken in the caller's scope.
fn cuda_of<'s>(s: &'s Storage, what: &str) -> Result<&'s CudaStorage> {
    match s {
        Storage::Cuda(c) => Ok(c),
        _ => candle::bail!("image rotation: {what} is not on a CUDA device"),
    }
}

impl ImageRotation {
    /// The tables `rotation_tables` built, uploaded where the caller placed
    /// them (a wave's forward span), with their `rows_end`.
    pub fn from_tables(rows: Tensor, rows_end: usize, cos: Tensor, sin: Tensor) -> Result<Self> {
        let n = rows.elem_count();
        if rows.dtype() != DType::U32 || !rows.is_contiguous() {
            candle::bail!("image rotation: rows must be dense U32");
        }
        for (t, what) in [(&cos, "cos"), (&sin, "sin")] {
            if t.dtype() != DType::F32 || !t.is_contiguous() || t.dim(0)? != n {
                candle::bail!(
                    "image rotation: {what} is {:?} {:?}, not dense F32 [{n}, pairs]",
                    t.dtype(),
                    t.dims()
                );
            }
        }
        if cos.dims() != sin.dims() {
            candle::bail!(
                "image rotation: cos {:?} and sin {:?} differ",
                cos.dims(),
                sin.dims()
            );
        }
        Ok(Self {
            rows,
            rows_end,
            cos,
            sin,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.rows.elem_count() == 0
    }

    /// Turn the listed rows of `q` and `k` (`[rows, heads, head_dim]` each,
    /// dense, one type) in place.
    pub fn apply(&self, q: &LiveTensor<'_>, k: &LiveTensor<'_>) -> Result<()> {
        if self.is_empty() {
            return Ok(());
        }
        let (q_rows, q_heads, dim) = heads_of(q, "q")?;
        let (k_rows, k_heads, k_dim) = heads_of(k, "k")?;
        if (k_rows, k_dim) != (q_rows, dim) || k.dtype() != q.dtype() {
            candle::bail!(
                "image rotation: q is {:?} {:?} and k {:?} {:?}",
                q.dtype(),
                q.dims(),
                k.dtype(),
                k.dims()
            );
        }
        if self.rows_end > q_rows {
            candle::bail!(
                "image rotation: rows up to {} listed for tensors of {q_rows}",
                self.rows_end
            );
        }
        let (n, pairs) = self.cos.dims2()?;
        let code = match q.dtype() {
            DType::F32 => IMAGE_ROPE_F32,
            DType::F16 => IMAGE_ROPE_F16,
            DType::BF16 => IMAGE_ROPE_BF16,
            other => candle::bail!("image rotation: q is {other:?}, not a float type"),
        };
        // Every borrow and device-pointer guard is held until the launch has
        // been issued.
        let (qs, ql) = q.storage_and_layout();
        let (ks, kl) = k.storage_and_layout();
        let (rs, rl) = self.rows.storage_and_layout();
        let (cs, cl) = self.cos.storage_and_layout();
        let (ss, sl) = self.sin.storage_and_layout();
        let (qc, kc) = (cuda_of(&qs, "q")?, cuda_of(&ks, "k")?);
        let (rc, cc, sc) = (
            cuda_of(&rs, "the row list")?,
            cuda_of(&cs, "cos")?,
            cuda_of(&ss, "sin")?,
        );
        let stream = qc.device().cuda_stream();
        let rows_v = rc.as_cuda_slice::<u32>()?.slice(rl.start_offset()..);
        let cos_v = cc.as_cuda_slice::<f32>()?.slice(cl.start_offset()..);
        let sin_v = sc.as_cuda_slice::<f32>()?.slice(sl.start_offset()..);
        let (rows_p, _rg) = rows_v.device_ptr(&stream);
        let (cos_p, _cg) = cos_v.device_ptr(&stream);
        let (sin_p, _sg) = sin_v.device_ptr(&stream);
        let launch = |qp: u64, kp: u64| {
            candle::set_kernel_breadcrumb("run_image_rope_rows", file!(), line!());
            unsafe {
                run_image_rope_rows(
                    code,
                    qp as *mut c_void,
                    q_heads as i32,
                    kp as *mut c_void,
                    k_heads as i32,
                    rows_p as *const u32,
                    cos_p as *const f32,
                    sin_p as *const f32,
                    n as i32,
                    dim as i32,
                    pairs as i32,
                    stream.cu_stream() as *mut c_void,
                )
            }
        };
        macro_rules! typed {
            ($t:ty) => {{
                let qv = qc.as_cuda_slice::<$t>()?.slice(ql.start_offset()..);
                let kv = kc.as_cuda_slice::<$t>()?.slice(kl.start_offset()..);
                let (qp, _qg) = qv.device_ptr(&stream);
                let (kp, _kg) = kv.device_ptr(&stream);
                launch(qp, kp)
            }};
        }
        let status = match q.dtype() {
            DType::F32 => typed!(f32),
            DType::F16 => typed!(f16),
            _ => typed!(bf16),
        };
        match status {
            0 => Ok(()),
            IMAGE_ROPE_UNSUPPORTED_DTYPE => {
                candle::bail!("image rotation: no kernel for {:?}", q.dtype())
            }
            IMAGE_ROPE_BAD_SHAPE => {
                candle::bail!("image rotation: {pairs} pairs do not fit a {dim}-wide head")
            }
            other => candle::bail!("image rotation: unexpected launcher status {other}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::gpu_test_lock::gpu_serial;
    use crate::models::qwen_vision::geometry::FACTOR;
    use candle::Device;

    /// A 1 × 3 image's last row: each rotary pair turns by its axis's offset.
    #[test]
    fn a_rows_table_turns_each_pair_by_its_axis() {
        let grid = Grid {
            height: FACTOR,
            width: 3 * FACTOR,
        };
        // n 3, m 3: row 2 is (r 0, c 2) → δ = (3 − 3 − 2, 0 + 0 − 2, 0 + 2 − 2).
        assert_eq!(image_delta(grid, 2), [-2, -2, 0]);
        let run = ImageRun {
            row: 5,
            len: 1,
            grid,
            first: 2,
        };
        let t = rotation_tables(&[run], 6, 100.0);
        assert_eq!(
            (t.rows.clone(), t.pairs, t.cos.len(), t.rows_end),
            (vec![5], 3, 3, 6)
        );
        // Pair 0: T, δ −2, frequency 1.
        assert_eq!(t.cos[0], (-2f64).cos() as f32);
        assert_eq!(t.sin[0], (-2f64).sin() as f32);
        // Pair 1: H, δ −2, frequency 100^(-1/3).
        let f1 = 100f64.powf(-2.0 / 6.0);
        assert_eq!(t.sin[1], (-2.0 * f1).sin() as f32);
        // Pair 2: W, δ 0.
        assert_eq!((t.cos[2], t.sin[2]), (1.0, 0.0));
    }

    /// Two runs list their rows in order, each by its row in the group.
    #[test]
    fn runs_list_their_rows_in_order() {
        let grid = Grid {
            height: FACTOR,
            width: 2 * FACTOR,
        };
        let a = ImageRun {
            row: 1,
            len: 2,
            grid,
            first: 0,
        };
        let b = ImageRun {
            row: 7,
            len: 1,
            grid,
            first: 1,
        };
        let t = rotation_tables(&[a, b], 4, 10.0);
        assert_eq!(t.rows, vec![1, 2, 7]);
        assert_eq!((t.cos.len(), t.rows_end), (3 * 2, 8));
    }

    /// Turning by δ and then by the slot is turning by the slot plus δ — the
    /// property the whole design rests on — checked on the kernel, in F32,
    /// for Q and K in one launch, with the rows outside the image and the
    /// pairs past the rotary width left bit for bit as they were.
    #[test]
    fn a_turn_then_the_slot_rotation_is_the_multimodal_rotation() {
        let _gpu = gpu_serial();
        let Ok(dev) = Device::new_cuda(0) else { return };
        let grid = Grid {
            height: 2 * FACTOR,
            width: 2 * FACTOR,
        };
        // Heads 8 wide, 2 of their 4 pairs rotary; Q 2 heads, K 1.
        let (head_dim, rope_dim, theta) = (8, 4, 50.0);
        let run = ImageRun {
            row: 1,
            len: 4,
            grid,
            first: 0,
        };
        let rows = 6;
        let values = |heads: usize, seed: usize| -> Vec<f32> {
            (0..rows * heads * head_dim)
                .map(|i| (((i + seed) * 37 % 101) as f32 - 50.0) / 25.0)
                .collect()
        };
        let (qv, kv) = (values(2, 0), values(1, 11));
        let q = Tensor::from_vec(qv.clone(), (rows, 2, head_dim), &dev).unwrap();
        let k = Tensor::from_vec(kv.clone(), (rows, 1, head_dim), &dev).unwrap();
        let t = rotation_tables(&[run], rope_dim, theta);
        let n = t.rows.len();
        let rot = ImageRotation::from_tables(
            Tensor::from_vec(t.rows, n, &dev).unwrap(),
            t.rows_end,
            Tensor::from_vec(t.cos, (n, t.pairs), &dev).unwrap(),
            Tensor::from_vec(t.sin, (n, t.pairs), &dev).unwrap(),
        )
        .unwrap();
        rot.apply(&q, &k).unwrap();
        let half = head_dim / 2;
        let pairs = rope_dim / 2;
        let rotate = |v: &[f32], angle: &dyn Fn(usize) -> f64| -> Vec<f32> {
            let mut out = v.to_vec();
            for f in 0..pairs {
                let (c, s) = (angle(f).cos(), angle(f).sin());
                let (a, b) = (v[f] as f64, v[f + half] as f64);
                out[f] = (a * c - b * s) as f32;
                out[f + half] = (a * s + b * c) as f32;
            }
            out
        };
        let inv = |f: usize| 1.0 / theta.powf(2.0 * f as f64 / rope_dim as f64);
        for (x, vals, heads) in [(&q, qv, 2), (&k, kv, 1)] {
            let turned = x.to_vec3::<f32>().unwrap();
            let orig = Tensor::from_vec(vals, (rows, heads, head_dim), &Device::Cpu)
                .unwrap()
                .to_vec3::<f32>()
                .unwrap();
            assert_eq!(turned[0], orig[0]);
            assert_eq!(turned[5], orig[5]);
            for j in 0..4 {
                let slot = 10.0 + j as f64;
                let d = image_delta(grid, j);
                for h in 0..heads {
                    for f in pairs..half {
                        assert_eq!(turned[1 + j][h][f], orig[1 + j][h][f]);
                        assert_eq!(turned[1 + j][h][f + half], orig[1 + j][h][f + half]);
                    }
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

    /// A list past the tensors' rows is refused before anything launches.
    #[test]
    fn rows_past_the_tensors_are_refused() {
        let _gpu = gpu_serial();
        let Ok(dev) = Device::new_cuda(0) else { return };
        let grid = Grid {
            height: FACTOR,
            width: FACTOR,
        };
        let run = ImageRun {
            row: 3,
            len: 1,
            grid,
            first: 0,
        };
        let t = rotation_tables(&[run], 4, 10.0);
        let rot = ImageRotation::from_tables(
            Tensor::from_vec(t.rows, 1, &dev).unwrap(),
            t.rows_end,
            Tensor::from_vec(t.cos, (1, t.pairs), &dev).unwrap(),
            Tensor::from_vec(t.sin, (1, t.pairs), &dev).unwrap(),
        )
        .unwrap();
        let q = Tensor::zeros((3, 1, 8), DType::F32, &dev).unwrap();
        let k = Tensor::zeros((3, 1, 8), DType::F32, &dev).unwrap();
        assert!(rot.apply(&q, &k).is_err());
    }
}
