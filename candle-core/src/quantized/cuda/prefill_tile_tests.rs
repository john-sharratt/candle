//! The int8 dense and grouped matmuls at prefill width: bit-identity of the prefill tiles
//! against the narrower ones, device-time benches over the Flash-Next and DeepSeek-V4-Flash
//! projection and expert shapes, and ncu targets.

use std::ffi::c_void;

use candle_kernels::quantized::{
    run_grouped_quantized_matmul, run_quantized_matmul, VxSegment, YType,
};
use cudarc::driver::{CudaSlice, DevicePtr};
use half::{bf16, f16};
use rand::{Rng, SeedableRng};

use super::super::int8_matmul_mode::{q8a128_dense_tile, DenseTile};
use super::super::ko_quant::{
    ko_chunk_bytes, mxfp4_ko_to_gpu_chunk, quantize_ko, quantize_mxfp4_ko,
};
use super::super::{GgmlDType, Int8Mode, SumScale};
use super::graph_bench::{time_graph, Rotation};
use super::{
    check_matmul_status, dtype_to_qtype, grouped_int8_n_sub, grouped_int8_row_fast,
    grouped_matmul_gemx_q8a128_with_mode, out_dtype_code, q8a128_dense_matmul, to_dynamic,
    CudaDevice, DynamicActs, Q8a128Operand,
};
use crate::backend::BackendDevice;
use crate::cuda_backend::Backing;
use crate::{DType, Device, Result, Tensor};

/// `len` pseudo-random bytes (xorshift64*), for weights whose values a timing does not read.
fn noise_bytes(len: usize, seed: u64) -> Vec<u8> {
    let mut s = seed | 1;
    let mut out = Vec::with_capacity(len + 8);
    while out.len() < len {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        out.extend_from_slice(&s.wrapping_mul(0x2545_F491_4F6C_DD1D).to_le_bytes());
    }
    out.truncate(len);
    out
}

/// A KO weight's byte length: `K/128` k-tiles × `N/8` row groups × the format's chunk.
fn ko_weight_bytes(dtype: GgmlDType, n: usize, k: usize) -> usize {
    (k / 128) * (n / 8) * ko_chunk_bytes(dtype)
}

/// The device address of a slice, read once.
fn addr<T>(dev: &CudaDevice, s: &CudaSlice<T>) -> u64 {
    let stream = dev.cuda_stream();
    let (p, _g) = s.device_ptr(&stream);
    p
}

/// The KO chunk bytes of a seeded random `[n, k]` weight of `dtype`, uploaded: the affine
/// formats through [`quantize_ko`], MXFP4 through its exact repack and GPU chunk layout.
fn ko_weight(
    dev: &CudaDevice,
    dtype: GgmlDType,
    n: usize,
    k: usize,
    seed: u64,
) -> Result<CudaSlice<u8>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let w: Vec<f32> = (0..n * k).map(|_| rng.random_range(-0.1f32..0.1)).collect();
    let bytes = if dtype == GgmlDType::MXFP4_KO {
        mxfp4_ko_to_gpu_chunk(&quantize_mxfp4_ko(&w, n, k), n, k)
    } else {
        quantize_ko(&w, n, k, dtype)
    };
    assert_eq!(bytes.len(), ko_weight_bytes(dtype, n, k));
    dev.memcpy_stod(&bytes)
}

/// The raw bits of every element of a dense output stored at F32, BF16 or F16.
fn out_bits(t: &Tensor) -> Result<Vec<u32>> {
    let t = t.flatten_all()?;
    Ok(match t.dtype() {
        DType::F32 => t.to_vec1::<f32>()?.iter().map(|v| v.to_bits()).collect(),
        DType::BF16 => t
            .to_vec1::<bf16>()?
            .iter()
            .map(|v| v.to_bits() as u32)
            .collect(),
        DType::F16 => t
            .to_vec1::<f16>()?
            .iter()
            .map(|v| v.to_bits() as u32)
            .collect(),
        other => unreachable!("dense outputs are F32, BF16 or F16, not {other:?}"),
    })
}

/// The value an output element's bits hold, as F32.
fn out_value(bits: u32, out: DType) -> f32 {
    match out {
        DType::F32 => f32::from_bits(bits),
        DType::BF16 => bf16::from_bits(bits as u16).to_f32(),
        DType::F16 => f16::from_bits(bits as u16).to_f32(),
        other => unreachable!("dense outputs are F32, BF16 or F16, not {other:?}"),
    }
}

/// A reference output is a real product, so a comparison against it proves something: every
/// element finite and at most one in a hundred an exact zero.
fn assert_written(bits: &[u32], out: DType) {
    let finite = bits.iter().all(|&b| out_value(b, out).is_finite());
    let zero = bits.iter().filter(|&&b| out_value(b, out) == 0.0).count();
    assert!(finite, "the reference holds a non-finite element");
    assert!(
        zero * 100 <= bits.len(),
        "{zero} of {} reference elements are zero",
        bits.len()
    );
}

/// How many elements of `got` differ from `want`, bit for bit.
fn bit_diffs(got: &[u32], want: &[u32]) -> usize {
    assert_eq!(got.len(), want.len());
    got.iter().zip(want).filter(|(a, b)| a != b).count()
}

/// Every KO weight format, by dtype.
const KO_FORMATS: [GgmlDType; 7] = [
    GgmlDType::Q2_KO,
    GgmlDType::Q3_KO,
    GgmlDType::Q4_KO,
    GgmlDType::Q5_KO,
    GgmlDType::Q6_KO,
    GgmlDType::Q8_KO,
    GgmlDType::MXFP4_KO,
];

/// The mode-4 prefill tile is mode-1 and mode-2 bit for bit — every KO format, every output
/// width, both Σx conventions — at ragged M (one row, a partial m16 sub-tile, a partial and a
/// whole 64-row tile, a tail past several tiles) and N that overhangs the 128-row tile by one,
/// two and three warps (96, 416, 2592) as well as N that fills it (256).
#[test]
fn mode4_dense_is_mode1_and_mode2_bit_for_bit() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let device = Device::Cuda(dev.clone());
    let shapes = [(96usize, 512usize), (256, 256), (416, 768), (2592, 2560)];
    for (fi, dtype) in KO_FORMATS.into_iter().enumerate() {
        for (si, &(n, k)) in shapes.iter().enumerate() {
            let w = ko_weight(&dev, dtype, n, k, (fi * 31 + si) as u64)?;
            let (ptr, len) = (addr(&dev, &w), ko_weight_bytes(dtype, n, k));
            for m in [1usize, 9, 40, 64, 200] {
                let x = Tensor::randn(0f32, 1.0, (m, k), &device)?;
                for sum in [SumScale::Raw, SumScale::ByAmax] {
                    let acts = to_dynamic(&x, Int8Mode::Performance, &dev, sum)?;
                    let DynamicActs::Int8(op) = &acts else {
                        unreachable!("an int8 mode quantizes")
                    };
                    for out in [DType::F32, DType::BF16, DType::F16] {
                        let run = |tile: DenseTile| -> Result<Vec<u32>> {
                            let t = q8a128_dense_matmul(op, ptr, dtype, n, len, tile, out, &dev)?;
                            out_bits(&t)
                        };
                        let m1 = run(DenseTile::Mode1)?;
                        assert_eq!(m1.len(), m * n);
                        assert_written(&m1, out);
                        for tile in [DenseTile::Mode2, DenseTile::Mode4] {
                            let diff = bit_diffs(&run(tile)?, &m1);
                            assert_eq!(
                                diff,
                                0,
                                "{dtype:?} [{m}x{k}]·[{n}x{k}]ᵀ {sum:?} {out:?}: {tile:?} \
                                 differs from mode-1 on {diff} of {} outputs",
                                m1.len()
                            );
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// The grouped prefill tiles — mode-4 (Bm 64 × 128 rows) and mode-8 (Bm 128 × 64 rows) — are the
/// mode-2 tile (Bm 32 × 32 rows) bit for bit, in both grid orders, for every KO format: experts
/// below one sub-tile, between sub-tiles, across several tiles and empty, at N that overhangs the
/// wide row tile by whole warps (96, 416) and N that fills it (640).
#[test]
fn wide_grouped_tiles_are_mode2_bit_for_bit() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let device = Device::Cuda(dev.clone());
    let k = 512usize;
    let batches = [1usize, 40, 0, 100, 200, 129];
    let total: usize = batches.iter().sum();
    let mut offsets = vec![0i32];
    for &b in &batches {
        offsets.push(offsets.last().copied().unwrap_or(0) + b as i32);
    }
    let x = Tensor::randn(0f32, 1.0, (total, k), &device)?;
    for (fi, dtype) in KO_FORMATS.into_iter().enumerate() {
        for n in [96usize, 416, 640] {
            let weights: Vec<CudaSlice<u8>> = (0..batches.len())
                .map(|e| ko_weight(&dev, dtype, n, k, (fi * 97 + e * 7 + n) as u64))
                .collect::<Result<_>>()?;
            let ptrs: Vec<u64> = weights.iter().map(|w| addr(&dev, w)).collect();
            for sum in [SumScale::Raw, SumScale::ByAmax] {
                let acts = to_dynamic(&x, Int8Mode::Performance, &dev, sum)?;
                let DynamicActs::Int8(op) = &acts else {
                    unreachable!("an int8 mode quantizes")
                };
                let run = |n_sub: usize, row_fast: bool| -> Result<Vec<u32>> {
                    let out = op.with_device_ptr(&dev, |act| {
                        grouped_matmul_gemx_q8a128_with_mode(
                            act,
                            &ptrs,
                            dtype,
                            n,
                            k,
                            total,
                            &offsets,
                            &dev,
                            Backing::Owned,
                            n_sub,
                            row_fast,
                            sum,
                        )
                    })?;
                    out_bits(&out)
                };
                let m2 = run(2, true)?;
                assert_eq!(m2.len(), total * n);
                assert_written(&m2, DType::F32);
                for n_sub in [4usize, 8] {
                    for row_fast in [true, false] {
                        let diff = bit_diffs(&run(n_sub, row_fast)?, &m2);
                        assert_eq!(
                            diff,
                            0,
                            "{dtype:?} N={n} {sum:?}: n_sub={n_sub} row_fast={row_fast} \
                             differs from mode-2 on {diff} of {} outputs",
                            m2.len()
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

/// The dense tilings a bench row times.
const DENSE_TILES: [(&str, DenseTile); 3] = [
    ("m1", DenseTile::Mode1),
    ("m2", DenseTile::Mode2),
    ("m4", DenseTile::Mode4),
];

/// One dense projection a bench times: `n` output rows, `k` columns, its weight format and the
/// width the output is stored at.
#[derive(Clone, Copy)]
struct DenseShape {
    dtype: GgmlDType,
    n: usize,
    k: usize,
    out: DType,
}

/// Device µs per launch of the dense matmul `op [M, K] × Wᵀ` at `tile`, the weight rotated
/// through `weights` (copies past the L2), into `dst` (`M × N` elements at `shape.out`).
fn time_dense(
    dev: &CudaDevice,
    op: &Q8a128Operand<'_>,
    shape: DenseShape,
    weights: &[u64],
    dst: u64,
    tile: DenseTile,
) -> Result<f64> {
    let DenseShape { dtype, n, k, out } = shape;
    let (m, len) = (op.rows, ko_weight_bytes(dtype, n, k));
    let qtype = dtype_to_qtype(dtype)? as i32;
    let out_code = out_dtype_code(out)?;
    let sum_norm = op.sum_scale.as_code();
    op.with_device_ptr(dev, |act| {
        time_graph(dev, weights, |wp, cs| {
            let seg = VxSegment {
                weights: wp as *const c_void,
                batch_count: m as i32,
            };
            // SAFETY: a KO weight of `len` bytes and `n` rows; the operand and `dst` hold
            // `m` rows.
            let status = unsafe {
                run_quantized_matmul(
                    &seg,
                    1,
                    act as *const c_void,
                    dst as *mut c_void,
                    k as i32,
                    n as i32,
                    m as i32,
                    n as i32,
                    qtype,
                    YType::Q8A128 as i32,
                    len,
                    tile.code(),
                    out_code,
                    sum_norm,
                    cs as *mut c_void,
                )
            };
            check_matmul_status(status, "dense")
        })
    })
}

/// A weight of `shape` rotated past the L2, an `[m, K]` operand and an output wide enough for
/// it, and `f` called with them.
fn with_dense_operands<R>(
    dev: &CudaDevice,
    shape: DenseShape,
    ms: &[usize],
    mut f: impl FnMut(usize, &Q8a128Operand<'_>, &[u64], u64) -> Result<R>,
) -> Result<Vec<R>> {
    let device = Device::Cuda(dev.clone());
    let len = ko_weight_bytes(shape.dtype, shape.n, shape.k);
    let w = dev.memcpy_stod(&noise_bytes(len, (shape.n * shape.k) as u64))?;
    let rot = Rotation::of_device_bytes(dev, addr(dev, &w), len)?;
    let weights = rot.ptrs(dev);
    let mut results = Vec::with_capacity(ms.len());
    for &m in ms {
        let x = Tensor::randn(0f32, 1.0, (m, shape.k), &device)?;
        let acts = to_dynamic(&x, Int8Mode::Performance, dev, SumScale::Raw)?;
        let DynamicActs::Int8(op) = &acts else {
            unreachable!("an int8 mode quantizes")
        };
        // SAFETY: fully written by every launch before it is read; F32-sized, so wide enough
        // for any output dtype.
        let dst = unsafe { dev.alloc::<f32>(m * shape.n)? };
        results.push(f(m, op, &weights, addr(dev, &dst))?);
    }
    Ok(results)
}

/// The dense int8 matmul on the Flash-Next prefill projections: device µs per launch (captured
/// and replayed, weights rotated past the L2) at every tiling, and the int8 TOPS it reaches.
#[test]
#[ignore = "benchmark — run explicitly with --ignored, card to itself"]
fn bench_prefill_dense() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let shape = |dtype, n, k, out| DenseShape { dtype, n, k, out };
    let (q8, q4, f32_, bf16_) = (GgmlDType::Q8_KO, GgmlDType::Q4_KO, DType::F32, DType::BF16);
    let shapes = [
        ("deltanet in q8", shape(q8, 16_480, 2560, f32_)),
        ("deltanet out q8", shape(q8, 2560, 6144, f32_)),
        ("attn qkv q8", shape(q8, 13_312, 2560, f32_)),
        ("attn qkv q8 bf16", shape(q8, 13_312, 2560, bf16_)),
        ("attn out q8", shape(q8, 2560, 4096, f32_)),
        ("shexp up q8", shape(q8, 640, 2560, f32_)),
        ("shexp down q8", shape(q8, 2560, 640, f32_)),
        ("deltanet out q4", shape(q4, 2560, 6144, f32_)),
    ];
    let mut header = format!("{:<16} {:>5}", "shape", "M");
    for (name, _) in DENSE_TILES {
        header.push_str(&format!(" {:>9} {:>5}", name, "TOPS"));
    }
    println!("{header}  (device µs per launch)");
    for (label, shape) in shapes {
        with_dense_operands(&dev, shape, &[2048, 4096, 7500], |m, op, weights, dst| {
            let mut line = format!("{label:<16} {m:>5}");
            for (_, tile) in DENSE_TILES {
                let us = time_dense(&dev, op, shape, weights, dst, tile)?;
                let tops = 2.0 * (m * shape.n * shape.k) as f64 / (us * 1e6);
                line.push_str(&format!(" {us:>9.1} {tops:>5.0}"));
            }
            println!("{line}");
            Ok(())
        })?;
    }
    Ok(())
}

/// Mode-1, mode-2 and mode-4 across M and N at K = 2560 (Q8_KO, weights rotated past the L2):
/// device µs per launch, with `*` on the tiling `q8a128_dense_tile` picks — the surface the
/// mode-4 gate is fitted to.
#[test]
#[ignore = "benchmark — run explicitly with --ignored, card to itself"]
fn bench_dense_tile_crossover() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let sm = dev.multiprocessor_count()?;
    println!(
        "{:>6} {:>5} {:>9} {:>9} {:>9}  (device µs per launch; * = the rule's pick)",
        "N", "M", "m1", "m2", "m4"
    );
    let ms = [64usize, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048];
    for n in [512usize, 1280, 2560, 6144, 13_312] {
        let shape = DenseShape {
            dtype: GgmlDType::Q8_KO,
            n,
            k: 2560,
            out: DType::F32,
        };
        with_dense_operands(&dev, shape, &ms, |m, op, weights, dst| {
            let pick = q8a128_dense_tile(m, n, shape.k, sm);
            let mut line = format!("{n:>6} {m:>5}");
            for (_, tile) in DENSE_TILES {
                let us = time_dense(&dev, op, shape, weights, dst, tile)?;
                let mark = if tile == pick { "*" } else { " " };
                line.push_str(&format!(" {us:>8.1}{mark}"));
            }
            println!("{line}");
            Ok(())
        })?;
    }
    Ok(())
}

/// One launch of mode-2 and mode-4 on the DeltaNet out-projection at 4096 rows (Q8_KO, F32): the
/// ncu target for the dense prefill tile — `--kernel-name regex:dense_m[24] --launch-count 2`.
#[test]
#[ignore = "profiling target — run explicitly under ncu"]
fn profile_dense_tiles_once() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let device = Device::Cuda(dev.clone());
    let (dtype, m, n, k) = (GgmlDType::Q8_KO, 4096usize, 2560usize, 6144usize);
    let len = ko_weight_bytes(dtype, n, k);
    let w = dev.memcpy_stod(&noise_bytes(len, 1))?;
    let x = Tensor::randn(0f32, 1.0, (m, k), &device)?;
    let acts = to_dynamic(&x, Int8Mode::Performance, &dev, SumScale::Raw)?;
    let DynamicActs::Int8(op) = &acts else {
        unreachable!("an int8 mode quantizes")
    };
    for tile in [DenseTile::Mode2, DenseTile::Mode4] {
        q8a128_dense_matmul(op, addr(&dev, &w), dtype, n, len, tile, DType::F32, &dev)?;
    }
    dev.synchronize()?;
    Ok(())
}

/// Expert row counts of `tokens` tokens each routed to `top_k` distinct experts of `experts`,
/// uniformly at random — the shape a prefill wave hands the routed GEMM.
fn routed_counts(tokens: usize, experts: usize, top_k: usize, seed: u64) -> Vec<usize> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut counts = vec![0usize; experts];
    let mut picked = Vec::with_capacity(top_k);
    for _ in 0..tokens {
        picked.clear();
        while picked.len() < top_k {
            let e = rng.random_range(0..experts);
            if !picked.contains(&e) {
                picked.push(e);
            }
        }
        for &e in &picked {
            counts[e] += 1;
        }
    }
    counts
}

/// The grouped tile tables at token-tile width `tile_w`: `(expert, batch start, batch count)` per
/// tile, each expert's rows cut into tiles in order.
fn tile_tables(counts: &[usize], tile_w: usize) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let (mut te, mut tbs, mut tbc) = (Vec::new(), Vec::new(), Vec::new());
    let mut start = 0usize;
    for (e, &c) in counts.iter().enumerate() {
        let mut s = 0usize;
        while s < c {
            let cnt = (c - s).min(tile_w);
            te.push(e as i32);
            tbs.push((start + s) as i32);
            tbc.push(cnt as i32);
            s += cnt;
        }
        start += c;
    }
    (te, tbs, tbc)
}

/// One routed-expert layer a grouped bench runs: the model's label, expert format, expert count,
/// experts per token, its projections as `(label, N, K)`, and the wave widths in tokens.
struct ExpertLayer {
    model: &'static str,
    dtype: GgmlDType,
    experts: usize,
    top_k: usize,
    projections: &'static [(&'static str, usize, usize)],
    tokens: &'static [usize],
}

/// Qwen3.8-Flash-Next's routed experts: 512, top-10, expert FFN 640, Q4_KO.
const FLASH_NEXT_EXPERTS: ExpertLayer = ExpertLayer {
    model: "flash-next",
    dtype: GgmlDType::Q4_KO,
    experts: 512,
    top_k: 10,
    projections: &[("gate/up", 640, 2560), ("down", 2560, 640)],
    tokens: &[2048, 4096, 7500],
};

/// DeepSeek-V4-Flash's routed experts: 256, top-6, `[2048, 7168]`, MXFP4_KO.
const DEEPSEEK_EXPERTS: ExpertLayer = ExpertLayer {
    model: "deepseek",
    dtype: GgmlDType::MXFP4_KO,
    experts: 256,
    top_k: 6,
    projections: &[("gate/up", 2048, 7168), ("down", 7168, 2048)],
    tokens: &[2048, 3900, 8192],
};

/// Distinct expert weights a grouped bench allocates per projection: enough to span 384 MiB, so
/// the working set is past the L2 as the real layer's is, and the experts cycle through them.
const EXPERT_POOL_BYTES: usize = 384 << 20;

/// The device tables of one grouped launch: every expert's weight address, and the tile table
/// at one token-tile width.
struct GroupedTables {
    weights: CudaSlice<u64>,
    tile_expert: CudaSlice<i32>,
    tile_b_start: CudaSlice<i32>,
    tile_b_cnt: CudaSlice<i32>,
    tiles: usize,
}

/// Device µs per launch of the grouped matmul `op × experts` over `tables` at token-tile mode
/// `n_sub` and grid order `row_fast`, into `dst` (`rows × N` F32).
#[allow(clippy::too_many_arguments)]
fn time_grouped(
    dev: &CudaDevice,
    op: &Q8a128Operand<'_>,
    dtype: GgmlDType,
    (n, k): (usize, usize),
    tables: &GroupedTables,
    dst: u64,
    n_sub: usize,
    row_fast: i32,
) -> Result<f64> {
    let qtype = dtype_to_qtype(dtype)? as i32;
    let ptrs = [
        addr(dev, &tables.weights),
        addr(dev, &tables.tile_expert),
        addr(dev, &tables.tile_b_start),
        addr(dev, &tables.tile_b_cnt),
    ];
    op.with_device_ptr(dev, |act| {
        time_graph(dev, &[()], |(), cs| {
            // SAFETY: device tables of every expert's weight and `tables.tiles` tiles; the
            // operand and `dst` hold every row the tiles name.
            let status = unsafe {
                run_grouped_quantized_matmul(
                    ptrs[0] as *const c_void,
                    ptrs[1] as *const c_void,
                    ptrs[2] as *const c_void,
                    ptrs[3] as *const c_void,
                    act as *const c_void,
                    dst as *mut c_void,
                    k as i32,
                    n as i32,
                    k as i32,
                    n as i32,
                    tables.tiles as i32,
                    qtype,
                    YType::Q8A128 as i32,
                    n_sub as i32,
                    row_fast,
                    SumScale::Raw.as_code(),
                    std::ptr::null(),
                    cs as *mut c_void,
                )
            };
            check_matmul_status(status, "grouped")
        })
    })
}

/// The grouped int8 matmul on the routed experts of Qwen3.8-Flash-Next and DeepSeek-V4-Flash:
/// device µs per launch at every token-tile mode (`<` on the one `grouped_int8_n_sub` picks), in
/// both grid orders (`*` on the one `grouped_int8_row_fast` picks).
#[test]
#[ignore = "benchmark — run explicitly with --ignored, card to itself"]
fn bench_prefill_grouped() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let device = Device::Cuda(dev.clone());
    println!(
        "{:<10} {:<8} {:>5} {:>6} {:>5} {:>9} {:>5}  (device µs per launch)",
        "model", "shape", "M", "rows", "n_sub", "µs", "TOPS"
    );
    for layer in [FLASH_NEXT_EXPERTS, DEEPSEEK_EXPERTS] {
        let (experts, top_k, dtype) = (layer.experts, layer.top_k, layer.dtype);
        for &(label, n, k) in layer.projections {
            let wbytes = ko_weight_bytes(dtype, n, k);
            let pool = EXPERT_POOL_BYTES.div_ceil(wbytes).clamp(1, experts);
            let w = dev.memcpy_stod(&noise_bytes(pool * wbytes, (n + k) as u64))?;
            let base = addr(&dev, &w);
            let wptrs: Vec<u64> = (0..experts)
                .map(|e| base + ((e % pool) * wbytes) as u64)
                .collect();
            for &m in layer.tokens {
                let counts = routed_counts(m, experts, top_k, m as u64);
                let rows = m * top_k;
                let x = Tensor::randn(0f32, 1.0, (rows, k), &device)?;
                let acts = to_dynamic(&x, Int8Mode::Performance, &dev, SumScale::Raw)?;
                let DynamicActs::Int8(op) = &acts else {
                    unreachable!("an int8 mode quantizes")
                };
                // SAFETY: fully written by every launch before it is read.
                let dst = unsafe { dev.alloc::<f32>(rows * n)? };
                let active = counts.iter().filter(|&&c| c > 0).count();
                let picked = grouped_int8_n_sub(rows / active.max(1), &[dtype]);
                for n_sub in [2usize, 4, 8] {
                    let (te, tbs, tbc) = tile_tables(&counts, 16 * n_sub);
                    let tables = GroupedTables {
                        weights: dev.memcpy_stod(&wptrs)?,
                        tiles: te.len(),
                        tile_expert: dev.memcpy_stod(&te)?,
                        tile_b_start: dev.memcpy_stod(&tbs)?,
                        tile_b_cnt: dev.memcpy_stod(&tbc)?,
                    };
                    let row_fast = grouped_int8_row_fast(rows, k, n, n_sub, &dev) as i32;
                    for order in [row_fast, 1 - row_fast] {
                        let us = time_grouped(
                            &dev,
                            op,
                            dtype,
                            (n, k),
                            &tables,
                            addr(&dev, &dst),
                            n_sub,
                            order,
                        )?;
                        let tops = 2.0 * (rows * n * k) as f64 / (us * 1e6);
                        let pick = if n_sub == picked { "<" } else { " " };
                        let mark = if order == row_fast { "*" } else { " " };
                        println!(
                            "{:<10} {label:<8} {m:>5} {rows:>6} {n_sub:>4}{pick} {order}{mark} \
                             {us:>9.1} {tops:>5.0}",
                            layer.model
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

/// One launch of each grouped prefill tile on Flash-Next's routed `down` projection at a
/// 7500-token wave (N 2560, K 640): the ncu target for the grouped tiles —
/// `--kernel-name regex:grouped_m --launch-count 2`.
#[test]
#[ignore = "profiling target — run explicitly under ncu"]
fn profile_grouped_tiles_once() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let device = Device::Cuda(dev.clone());
    let layer = FLASH_NEXT_EXPERTS;
    let (experts, top_k, dtype, tokens) = (layer.experts, layer.top_k, layer.dtype, 7500usize);
    let (n, k) = (2560usize, 640usize);
    let wbytes = ko_weight_bytes(dtype, n, k);
    let w = dev.memcpy_stod(&noise_bytes(experts * wbytes, 3))?;
    let base = addr(&dev, &w);
    let wptrs: Vec<u64> = (0..experts).map(|e| base + (e * wbytes) as u64).collect();
    let counts = routed_counts(tokens, experts, top_k, 7);
    let mut offsets = vec![0i32];
    for &c in &counts {
        offsets.push(offsets.last().copied().unwrap_or(0) + c as i32);
    }
    let rows = tokens * top_k;
    let x = Tensor::randn(0f32, 1.0, (rows, k), &device)?;
    let acts = to_dynamic(&x, Int8Mode::Performance, &dev, SumScale::Raw)?;
    let DynamicActs::Int8(op) = &acts else {
        unreachable!("an int8 mode quantizes")
    };
    for n_sub in [4usize, 8] {
        let row_fast = grouped_int8_row_fast(rows, k, n, n_sub, &dev);
        op.with_device_ptr(&dev, |act| {
            grouped_matmul_gemx_q8a128_with_mode(
                act,
                &wptrs,
                dtype,
                n,
                k,
                rows,
                &offsets,
                &dev,
                Backing::Owned,
                n_sub,
                row_fast,
                SumScale::Raw,
            )
        })?;
    }
    dev.synchronize()?;
    Ok(())
}
