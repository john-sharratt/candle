//! Performance harness for the W4A16 → Q4_KO import — the §0.4 rule-4 harness
//! for the converter's three CPU kernels: `decode_packed` (nibble unpack +
//! scale gate), `pack_q4_ko` (the chunk-permutation pack) and the
//! losslessness verify.
//!
//! Every run carries a **correctness gate**: for a sample of experts per
//! iteration the packed output is dequantized through the *untouched*
//! `dequant_ko` and compared bit-for-bit against the source affine — the
//! independent path that anchors the fast codes-roundtrip verify (and the
//! shared permutation table) to the real layout. A fast wrong kernel cannot
//! pass.
//!
//! Sizing: one iteration converts `experts` slabs at the real expert
//! geometry (`640 × 2560`, ~1.6 M codes each). At the default 64 experts the
//! working set is ~420 MB of codes/packed/scales — past any CPU cache, so
//! the numbers are memory-system numbers, not cache numbers (the §0.4 L2
//! lesson, applied to the host).
//!
//! Driven by `examples/w4a16_convert_bench.rs`; pure CPU, rayon across
//! experts exactly as the converter runs.

use candle::quantized::ko_quant::{dequant_ko, pack_q4_ko};
use candle::quantized::GgmlDType;
use candle::{Device, Result};
use half::bf16;
use rayon::prelude::*;

use super::convert::{decode_packed, gpu_repack_tensor, DecodedExpert};

/// Harness configuration.
#[derive(Debug, Clone, Copy)]
pub struct ConvertBenchCfg {
    /// Expert slabs per iteration (each `nrows × ncols`).
    pub experts: usize,
    pub nrows: usize,
    pub ncols: usize,
    pub warmup: usize,
    pub iters: usize,
    pub seed: u64,
}

impl Default for ConvertBenchCfg {
    fn default() -> Self {
        Self {
            // The real routed-expert geometry (gate/up orientation).
            experts: 64,
            nrows: 640,
            ncols: 2560,
            warmup: 2,
            iters: 10,
            seed: 0x5EED_C0DE,
        }
    }
}

/// One synthetic W4A16 expert: raw `weight_packed` and `weight_scale` bytes.
fn synth_expert(cfg: &ConvertBenchCfg, seed: u64) -> (Vec<u8>, Vec<u8>) {
    let (nrows, ncols) = (cfg.nrows, cfg.ncols);
    let mut lcg = seed;
    let mut next = move || {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
        (lcg >> 33) as u32
    };
    let mut packed = Vec::with_capacity(nrows * ncols / 2);
    for _ in 0..nrows * ncols / 8 {
        let mut word = 0u32;
        for i in 0..8 {
            word |= (next() % 16) << (4 * i);
        }
        packed.extend_from_slice(&word.to_le_bytes());
    }
    let scales: Vec<u8> = (0..nrows * ncols / 128)
        // f16-exact power-of-two scales, as bf16 weight scales in range are.
        .flat_map(|_| bf16::from_f32(2f32.powi((next() % 8) as i32 - 12)).to_le_bytes())
        .collect();
    (packed, scales)
}

/// The independent gate: `dequant_ko(pack(...))` must equal the source
/// affine exactly. Runs the untouched reference dequant, so it is also the
/// harness's slowest stage — sampled, not swept, after iteration 0.
fn gate_expert(packed_ko: &[u8], d: &DecodedExpert) -> Result<()> {
    let back = dequant_ko(packed_ko, d.nrows, d.ncols, GgmlDType::Q4_KO);
    let k_groups = d.ncols / 128;
    for (i, &got) in back.iter().enumerate() {
        let (s, mn) = d.dm[(i / d.ncols) * k_groups + (i % d.ncols) / 128];
        let want = s * d.codes[i] as f32 + mn;
        if got != want {
            candle::bail!(
                "convert_bench gate: elem {i} = {got} against source {want} — a fast \
                 wrong kernel just tried to pass"
            );
        }
    }
    Ok(())
}

/// Run the harness. Prints per-stage times and aggregate throughput.
pub fn run_convert_bench(cfg: ConvertBenchCfg) -> Result<()> {
    let inputs: Vec<(Vec<u8>, Vec<u8>)> = (0..cfg.experts)
        .map(|e| synth_expert(&cfg, cfg.seed.wrapping_add(e as u64)))
        .collect();
    let in_bytes: usize = inputs.iter().map(|(p, s)| p.len() + s.len()).sum();
    let out_bytes = cfg.experts * cfg.nrows * cfg.ncols / GgmlDType::Q4_KO.block_size()
        * GgmlDType::Q4_KO.type_size();
    println!(
        "[convert_bench] {} experts of [{} × {}] per iter — in {:.1} MB, out {:.1} MB",
        cfg.experts,
        cfg.nrows,
        cfg.ncols,
        in_bytes as f64 / 1e6,
        out_bytes as f64 / 1e6
    );

    let (mut t_decode, mut t_pack, mut t_verify) = (0f64, 0f64, 0f64);
    let mut gated = 0usize;
    for it in 0..cfg.warmup + cfg.iters {
        let measured = it >= cfg.warmup;

        let t = std::time::Instant::now();
        let decoded: Vec<DecodedExpert> = inputs
            .par_iter()
            .map(|(p, s)| decode_packed(p, s, cfg.nrows, cfg.ncols, "bench"))
            .collect::<Result<_>>()?;
        if measured {
            t_decode += t.elapsed().as_secs_f64();
        }

        let t = std::time::Instant::now();
        let packed: Vec<Vec<u8>> = decoded
            .par_iter()
            .map(|d| pack_q4_ko(&d.codes, &d.dm, d.nrows, d.ncols))
            .collect();
        if measured {
            t_pack += t.elapsed().as_secs_f64();
        }

        let t = std::time::Instant::now();
        decoded
            .par_iter()
            .zip(packed.par_iter())
            .try_for_each(|(d, p)| super::convert::verify_packed(p, d, "bench"))?;
        if measured {
            t_verify += t.elapsed().as_secs_f64();
        }

        // The independent correctness gate: every expert on the first
        // iteration, one per iteration after — enough to catch a kernel that
        // went fast by going wrong, without the reference dequant dominating
        // the harness it is guarding.
        if it == 0 {
            decoded
                .par_iter()
                .zip(packed.par_iter())
                .try_for_each(|(d, p)| gate_expert(p, d))?;
            gated += decoded.len();
        } else {
            let pick = it % decoded.len();
            gate_expert(&packed[pick], &decoded[pick])?;
            gated += 1;
        }
    }

    let n = cfg.iters as f64;
    let per_iter = (t_decode + t_pack + t_verify) / n;
    let params = (cfg.experts * cfg.nrows * cfg.ncols) as f64;
    println!(
        "[convert_bench] CPU: decode {:.1} ms  pack {:.1} ms  verify {:.1} ms  per iter \
         ({:.2} ms/expert end-to-end)",
        t_decode / n * 1e3,
        t_pack / n * 1e3,
        t_verify / n * 1e3,
        per_iter * 1e3 / cfg.experts as f64,
    );
    println!(
        "[convert_bench] CPU: {:.2} G codes/s, in {:.2} GB/s, out {:.2} GB/s  (gate ran \
         on {gated} experts)",
        params / per_iter / 1e9,
        in_bytes as f64 / per_iter / 1e9,
        out_bytes as f64 / per_iter / 1e9,
    );

    // ── The GPU path: the fused repack kernel, host-to-host ──────────────────
    //
    // Timed as the converter pays it — gathered host bytes in, Q4_KO bytes
    // back on the host — so the number includes H2D, kernel and D2H. The gate
    // is byte-identity against the CPU pack for EVERY expert on the first
    // iteration and one thereafter: the permutation is data-independent, so
    // identity on randomized inputs pins the kernel's layout to the CPU
    // reference (which the dequant gate above anchors to the real format).
    let device = Device::new_cuda(0)?;
    let mut words_all = Vec::with_capacity(cfg.experts * cfg.nrows * cfg.ncols / 2);
    let mut scales_all = Vec::with_capacity(cfg.experts * cfg.nrows * cfg.ncols / 64);
    for (p, s) in &inputs {
        words_all.extend_from_slice(p);
        scales_all.extend_from_slice(s);
    }
    let cpu_packed: Vec<Vec<u8>> = inputs
        .par_iter()
        .map(|(p, s)| {
            let d = decode_packed(p, s, cfg.nrows, cfg.ncols, "bench")?;
            Ok(pack_q4_ko(&d.codes, &d.dm, d.nrows, d.ncols))
        })
        .collect::<Result<_>>()?;
    let mut t_gpu = 0f64;
    for it in 0..cfg.warmup + cfg.iters {
        let t = std::time::Instant::now();
        let out = gpu_repack_tensor(
            &device,
            &words_all,
            &scales_all,
            cfg.experts,
            cfg.nrows,
            cfg.ncols,
        )?;
        if it >= cfg.warmup {
            t_gpu += t.elapsed().as_secs_f64();
        }
        let per = out.len() / cfg.experts;
        let check = |e: usize| -> Result<()> {
            if cpu_packed[e].as_slice() != &out[e * per..(e + 1) * per] {
                candle::bail!(
                    "convert_bench gate: expert {e} GPU repack diverges from the CPU \
                     reference bytes"
                );
            }
            Ok(())
        };
        if it == 0 {
            (0..cfg.experts).try_for_each(check)?;
        } else {
            check(it % cfg.experts)?;
        }
    }
    let g = t_gpu / n;
    println!(
        "[convert_bench] GPU (H2D + kernel + D2H): {:.1} ms per iter ({:.3} ms/expert, \
         {:.2} G codes/s, out {:.2} GB/s)",
        g * 1e3,
        g * 1e3 / cfg.experts as f64,
        params / g / 1e9,
        out_bytes as f64 / g / 1e9,
    );
    Ok(())
}
