//! Microbench + `ncu` target for the hyper-connection low-rank projections'
//! int8 GEMMs: KO-quantized, on the int8 tensor cores.
//!
//! [`super::bench`] times the three fused kernels and the F32 GEMMs at prefill
//! width. This times the two KO GEMMs a decode wave runs about 110 times a step,
//! **stage by stage**: each column isolates one stage with a standalone quantize
//! in front of it, so a GEMM's own time is the column minus its quantize. The
//! engine runs the same two GEMMs fed by fused producers instead (`hc_mix_q8`:
//! the norm and the SiLU emit the q8a128 operands themselves), so the `q(..)` and
//! `silu` columns are launches the engine does not make, and `lowrank` is the
//! unfused chain's cost — an upper bound on the engine's. The split-K sweep at
//! the end times the GEMMs alone, as the engine issues them.
//!
//! # What the harness has to get right
//!
//! **Weights rotate past the L2.** One module's int8 weights fit in the 96 MiB
//! L2 with room to spare, so timing one module in a loop measures cache. A
//! forward reads ~110 distinct modules a step; the bench cycles through
//! [`MODULES`] of them, enough that every call reads its weights from DRAM.
//!
//! **Gate before timing.** The KO path is held to the F32 module's output at
//! quantization error (the bound `ko::tests` uses) before anything is timed, so
//! a fast wrong answer cannot be reported.
//!
//! **Decode and prefill widths both.** A change made for decode must not cost
//! prefill, and this is where that is visible first: every row prints its
//! weight-read bandwidth, so a prefill width that slows down shows here before
//! it shows in a gate.

use std::time::Instant;

use candle::quantized::cuda::{produce_q8a128, to_dynamic, DynamicTensor};
use candle::quantized::int8_split_k::{
    dense_k_split_depth, dense_k_split_fits, narrow_smem_bound, q8a128_dense_plan, DensePlan,
    NARROW_MAX_ROWS, NARROW_SMEM_CAP,
};
use candle::quantized::{Int8Mode, SumScale};
use candle::{DType, Device, Result, Tensor};

use super::cuda_fused;
use super::ko::HcWeightsKo;
use super::{hc_mix_with_operand, HcProject, HcWeights};
use crate::models::quantized_matmul::QMatMul;

/// Distinct modules the timed loop cycles through: 16 × ~8.7 MB ≈ 139 MB of
/// weights against the 96 MiB L2.
const MODULES: usize = 16;

/// The KO module's output against the F32 module's: Q8 quantization error,
/// the same bound `ko::tests` holds it to.
const GATE: f32 = 0.02;

/// Bytes of a Q8 weight per element: 34-byte blocks of 32.
const Q8_BYTES_PER_ELEM: f64 = 34.0 / 32.0;

fn lcg(shape: &[usize], seed: u64, scale: f32, dev: &Device) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let v: Vec<f32> = (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5) * scale
        })
        .collect();
    Tensor::from_vec(v, shape, dev)
}

/// Root-mean-square gap relative to the reference's own magnitude.
fn rel_gap(got: &Tensor, want: &Tensor) -> Result<f32> {
    let a = got.flatten_all()?.to_vec1::<f32>()?;
    let b = want.flatten_all()?.to_vec1::<f32>()?;
    let num: f32 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
    let den: f32 = b.iter().map(|y| y * y).sum();
    Ok((num / den.max(1e-30)).sqrt())
}

/// µs per call of `f(i)` over `iters` calls after `warmup`, `i` cycling
/// through the modules.
fn time_rotating(
    dev: &Device,
    warmup: usize,
    iters: usize,
    mut f: impl FnMut(usize) -> Result<()>,
) -> Result<f64> {
    for i in 0..warmup {
        f(i % MODULES)?;
    }
    dev.synchronize()?;
    let t0 = Instant::now();
    for i in 0..iters {
        f(i % MODULES)?;
    }
    dev.synchronize()?;
    Ok(t0.elapsed().as_secs_f64() * 1e6 / iters as f64)
}

/// Gate the KO module against the F32 module, then time each launch of
/// `hc_mix:lowrank` at every width in `widths`.
pub fn run_ko_projections(dev: &Device, widths: &[usize], iters: usize) -> Result<()> {
    let (hc, n_embd, low_rank) = (4usize, 2560usize, 320usize);
    let hc_dim = hc * n_embd;
    let mode = Int8Mode::auto(dev);
    let Device::Cuda(cuda) = dev else {
        candle::bail!("the KO projection bench runs on CUDA");
    };

    let mut f32_modules = Vec::with_capacity(MODULES);
    let mut ko_modules = Vec::with_capacity(MODULES);
    for m in 0..MODULES as u64 {
        let w = HcWeights::from_checkpoint(
            lcg(&[hc_dim], 0x100 + m, 0.4, dev)?.affine(1.0, 1.0)?,
            lcg(&[low_rank + hc, hc_dim], 0x200 + m, 0.1, dev)?,
            lcg(&[hc_dim, low_rank], 0x300 + m, 0.1, dev)?,
            hc,
        )?;
        ko_modules.push(HcWeightsKo::from_weights(&w, mode)?);
        f32_modules.push(w);
    }
    let gate_cols = ko_modules[0].gate_cols();
    let down_rows = (gate_cols + hc).div_ceil(32) * 32;
    let down_bytes = (down_rows * hc_dim) as f64 * Q8_BYTES_PER_ELEM;
    let up_bytes = (hc_dim * gate_cols) as f64 * Q8_BYTES_PER_ELEM;
    println!(
        "KO hyper-connection projections | {MODULES} modules rotated, {:.1} MB of weights \
         (L2 96 MiB) | down {down_rows}x{hc_dim} ({:.2} MB), up {hc_dim}x{gate_cols} ({:.2} MB) \
         | mode {mode:?}",
        MODULES as f64 * (down_bytes + up_bytes) / 1e6,
        down_bytes / 1e6,
        up_bytes / 1e6,
    );

    {
        let xn = lcg(&[8, hc_dim], 0x400, 2.0, dev)?;
        let want = f32_modules[0].down(&xn)?;
        let got = ko_modules[0].down(&xn)?;
        let gap_down = rel_gap(&got.narrow(1, 0, low_rank)?, &want.narrow(1, 0, low_rank)?)?;
        let lo = lcg(&[8, gate_cols], 0x401, 2.0, dev)?;
        let want_up = f32_modules[0].up(&lo.narrow(1, 0, low_rank)?.contiguous()?)?;
        let got_up = ko_modules[0].up(&lo)?;
        // KO pads the rank to 384 with zero weight columns, so `lo`'s columns
        // 320..384 multiply into nothing and the full operand through KO is the
        // same product as its rank-320 slice through F32.
        let gap_up = rel_gap(&got_up, &want_up)?;
        let worst = gap_down.max(gap_up);
        if worst > GATE {
            candle::bail!(
                "KO bench: the KO projections disagree with F32 by {worst} (down {gap_down}, \
                 up {gap_up}) — past Q8 quantization error, so timing them measures nothing"
            );
        }
        println!("correctness gate: KO vs F32, down {gap_down:.2e}, up {gap_up:.2e}");
    }

    println!(
        "{:>6} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>10} {:>10} {:>9}",
        "rows",
        "q(xn)",
        "down",
        "silu",
        "q(lo)",
        "up",
        "lowrank",
        "down GB/s",
        "up GB/s",
        "up_silu"
    );
    for &t in widths {
        let warmup = MODULES;
        let xn = lcg(&[t, hc_dim], 0x500 + t as u64, 2.0, dev)?;
        let lo = lcg(&[t, gate_cols], 0x600 + t as u64, 2.0, dev)?;
        let proj = ko_modules[0].down(&xn)?;

        let q_xn = time_rotating(dev, warmup, iters, |_| {
            to_dynamic(&xn, mode, cuda, SumScale::Raw)?;
            Ok(())
        })?;
        let q_lo = time_rotating(dev, warmup, iters, |_| {
            to_dynamic(&lo, mode, cuda, SumScale::Raw)?;
            Ok(())
        })?;
        // `down` and `up` each quantize their operand first; the GEMM alone is
        // the difference, and `ncu` gives the kernel's own time beside it.
        let down = time_rotating(dev, warmup, iters, |m| {
            ko_modules[m].down(&xn)?;
            Ok(())
        })?;
        let silu = time_rotating(dev, warmup, iters, |_| {
            proj.narrow(1, 0, gate_cols)?.silu()?;
            Ok(())
        })?;
        let up = time_rotating(dev, warmup, iters, |m| {
            ko_modules[m].up(&lo)?;
            Ok(())
        })?;
        let lowrank = time_rotating(dev, warmup, iters, |m| {
            let p = ko_modules[m].down(&xn)?;
            let l = p.narrow(1, 0, gate_cols)?.silu()?;
            ko_modules[m].up(&l)?;
            Ok(())
        })?;
        // `up` as the engine runs it: the SiLU and the quantize inside the matmul.
        let up_silu = time_rotating(dev, warmup, iters, |m| {
            ko_modules[m]
                .matmuls()
                .1
                .forward_silu_f32(&proj, gate_cols)?;
            Ok(())
        })?;
        let gbps = |bytes: f64, us: f64| bytes / (us.max(1e-3) * 1e-6) / 1e9;
        println!(
            "{t:>6} {q_xn:>9.1} {down:>9.1} {silu:>9.1} {q_lo:>9.1} {up:>9.1} {lowrank:>9.1} \
             {:>10.0} {:>10.0} {up_silu:>9.1}",
            gbps(down_bytes, down - q_xn),
            gbps(up_bytes, up - q_lo),
        );
    }
    println!(
        "(µs per call; `down`/`up` include their operand's quantize, the GB/s columns \
         subtract it — weight bytes over GEMM time)"
    );

    // The production pre-mix, as a decode layer calls it: norm → down → SiLU → up →
    // collapse with the block operand, five launches. Timed twice — synchronised at
    // the end (the GPU's pace, when it is the bound), and as the host's issue time
    // alone (the loop's own wall clock before the final sync, which is the bound
    // whenever the GPU drains faster than the host can feed it).
    println!("\n{:>6} {:>11} {:>11}", "rows", "hc_mix µs", "host µs");
    for &t in widths {
        let x = lcg(&[t, hc, n_embd], 0x800 + t as u64, 2.0, dev)?;
        for module in &ko_modules {
            hc_mix_with_operand(&x, module, 1e-6, None)?;
        }
        dev.synchronize()?;
        let t0 = Instant::now();
        for i in 0..iters {
            hc_mix_with_operand(&x, &ko_modules[i % MODULES], 1e-6, None)?;
        }
        let host = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        dev.synchronize()?;
        let total = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        println!("{t:>6} {total:>11.1} {host:>11.1}");
    }

    // The same steps one at a time, host issue time only — which of them carries
    // the host cost the pre-mix pays beyond its launches.
    println!(
        "\n{:>6} {:>8} {:>8} {:>8} {:>8}   (host µs per op)",
        "rows", "norm_q8", "down", "up_silu", "mix_q8"
    );
    for &t in widths {
        let x = lcg(&[t, hc, n_embd], 0x900 + t as u64, 2.0, dev)?;
        let ss = SumScale::Raw;
        let (xn, xn_q8) = cuda_fused::norm_q8(&x, ko_modules[0].norm(), 1e-6, None, ss)?;
        let (d, u, _) = ko_modules[0].matmuls();
        let proj = d.forward_dynamic(DynamicTensor::Int8(&xn_q8), DType::F32)?;
        let gate_raw = u.forward_silu_f32(&proj, gate_cols)?;
        let host_us = |f: &mut dyn FnMut() -> Result<()>| -> Result<f64> {
            dev.synchronize()?;
            let t0 = Instant::now();
            for _ in 0..iters {
                f()?;
            }
            let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            dev.synchronize()?;
            Ok(us)
        };
        let norm = host_us(&mut || {
            cuda_fused::norm_q8(&x, ko_modules[0].norm(), 1e-6, None, ss).map(|_| ())
        })?;
        let down = host_us(&mut || {
            d.forward_dynamic(DynamicTensor::Int8(&xn_q8), DType::F32)
                .map(|_| ())
        })?;
        let up = host_us(&mut || u.forward_silu_f32(&proj, gate_cols).map(|_| ()))?;
        let mix =
            host_us(&mut || cuda_fused::mix_q8(&xn, &gate_raw, hc, n_embd, None, ss).map(|_| ()))?;
        // The operand's allocation and wrapping with no kernel behind it: what a
        // producer costs the host before its launch.
        let produce =
            host_us(&mut || produce_q8a128(&proj, t, gate_cols, ss, |_| Ok(())).map(|_| ()))?;
        println!(
            "{t:>6} {norm:>8.1} {down:>8.1} {up:>8.1} {mix:>8.1}   produce alone {produce:.1}"
        );
    }

    let down = Gemm {
        name: "down",
        k: hc_dim,
        n: down_rows,
        bytes: down_bytes,
    };
    let up = Gemm {
        name: "up",
        k: gate_cols,
        n: hc_dim,
        bytes: up_bytes,
    };
    sweep_splits(dev, &ko_modules, widths, iters, [down, up])
}

/// One of the module's two GEMMs, as the split sweep times it: `[t, k] × [n, k]ᵀ`.
struct Gemm {
    name: &'static str,
    k: usize,
    n: usize,
    /// Its weight's bytes, for the bandwidth column.
    bytes: f64,
}

/// The `down` and `up` GEMMs alone — operands quantized once, outside the timing — unsplit,
/// split into K's slices and, up to eight rows, narrow at four and eight warps; every form gated
/// equal to the unsplit one bit for bit (they all sum in one order) before it is timed.
fn sweep_splits(
    dev: &Device,
    ko: &[HcWeightsKo],
    widths: &[usize],
    iters: usize,
    gemms: [Gemm; 2],
) -> Result<()> {
    let Device::Cuda(cuda) = dev else {
        candle::bail!("the split sweep runs on CUDA");
    };
    let sm = cuda.multiprocessor_count()?;
    let (_, _, mode) = ko[0].matmuls();
    println!(
        "\nsplit-K / narrow sweep — GEMM only, µs per call (GB/s of weights), * = the rule's \
         choice; `s:` slices (1 = unsplit), `nW:` narrow at W warps"
    );
    for Gemm { name, k, n, bytes } in gemms {
        for &t in widths {
            let x = lcg(&[t, k], 0x700 + t as u64, 2.0, dev)?;
            let acts = to_dynamic(&x, mode, cuda, SumScale::Raw)?;
            let pick = |m: usize| -> &QMatMul {
                let (d, u, _) = ko[m].matmuls();
                if name == "down" {
                    d
                } else {
                    u
                }
            };
            let want = pick(0).inner().forward_dynamic_plan(
                acts.as_dynamic(),
                DType::F32,
                DensePlan::Unsplit,
            )?;
            let want = want.flatten_all()?.to_vec1::<f32>()?;
            let rule = q8a128_dense_plan(t, n, k, sm, true);
            let mut row = format!("{name:>4} {t:>5} rows |");
            // Split forms only where the fixed scratch holds them — the launcher refuses the
            // rest, and the rule never picks them. Each asked-for depth is launched at the
            // count that covers K with no empty slice.
            let mut tried = vec![DensePlan::Unsplit];
            if dense_k_split_fits(t, n, k) {
                let mut depths: Vec<usize> = [8usize, 16, 27, 40]
                    .iter()
                    .map(|&want| dense_k_split_depth(k, want))
                    .collect();
                if let DensePlan::SplitK(s) = rule {
                    depths.push(s);
                }
                depths.sort_unstable();
                depths.dedup();
                tried.extend(depths.into_iter().filter(|&d| d > 1).map(DensePlan::SplitK));
            }
            // Narrow forms where a launch carries the rows and its shared memory fits.
            if t <= NARROW_MAX_ROWS {
                for warps in [4usize, 8, 16] {
                    if narrow_smem_bound(k, warps) <= NARROW_SMEM_CAP {
                        tried.push(DensePlan::Narrow { warps });
                    }
                }
                if matches!(rule, DensePlan::Narrow { .. }) && !tried.contains(&rule) {
                    tried.push(rule);
                }
            }
            for &plan in &tried {
                let got =
                    pick(0)
                        .inner()
                        .forward_dynamic_plan(acts.as_dynamic(), DType::F32, plan)?;
                if got.flatten_all()?.to_vec1::<f32>()? != want {
                    candle::bail!(
                        "split sweep: {name} at {t} rows, {plan:?}, differs from unsplit — \
                         every plan must sum in one order"
                    );
                }
                let us = time_rotating(dev, MODULES, iters, |m| {
                    pick(m)
                        .inner()
                        .forward_dynamic_plan(acts.as_dynamic(), DType::F32, plan)?;
                    Ok(())
                })?;
                let mark = if plan == rule { "*" } else { " " };
                let label = match plan {
                    DensePlan::Unsplit => "1".to_string(),
                    DensePlan::SplitK(s) => s.to_string(),
                    DensePlan::Narrow { warps } => format!("n{warps}"),
                };
                row.push_str(&format!(
                    " {label}:{us:.1}({:.0}){mark}",
                    bytes / (us * 1e-6) / 1e9
                ));
            }
            println!("{row}");
        }
    }
    Ok(())
}
