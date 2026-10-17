//! **The per-expert accuracy gate** (`docs/cpu_expert_compute.md` §6): one
//! expert, the same slot bytes and the same q8 activation codes through the
//! GPU's grouped chain (gather, gate, up, fused SwiGLU-quantize, down) and
//! through the pool's kernel. On synthetic experts both are measured against
//! the f32 chain over the dequantized weights, and the CPU's error may be at
//! most 1.1× the GPU's; on the model pack's own experts the two are compared
//! with each other.

use std::path::PathBuf;

use candle::cuda_backend::{Backing, CudaDevice};
use candle::quantized::cuda::{
    fused_moe_gather_q8a128, grouped_qmatmul_dev_q8a128, moe_bucketize, quantize_acts_q8a128,
    silu_mul_q8a128, MoeBucketizeWorkspace,
};
use candle::quantized::decode_rows::DecodeRows;
use candle::quantized::ko_quant::{dequant_ko, quantize_ko};
use candle::quantized::{GgmlDType, Int8Mode, SumScale};
use candle::{Device, Result, Tensor};
use cudarc::driver::DevicePtr;
use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};

use super::kernel::act::{ActTiles, TILE};
use super::kernel::expert::{device_exp, device_silu, forward, Expert};
use super::kernel::ko::Ko;
use super::kernel::simd::Isa;
use super::rings::Mapped;
use crate::models::expert_lre::section::{geometries_of, open_section};
use crate::models::model_pack::ModelPack;
use crate::models::quantized_qwen38_moe::prepared_engine_pack;

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

/// `||a − b|| / ||b||`.
fn rel(a: &[f32], b: &[f64]) -> f64 {
    let (mut e, mut n) = (0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        e += (x as f64 - y).powi(2);
        n += y.powi(2);
    }
    (e / n).sqrt()
}

fn widen(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

/// One expert in `slot` (projections at `offsets`, KO width `ko` / `dt`) over
/// the activations `x` (`tokens × hidden`): the GPU's grouped chain, and the
/// pool's kernel on the device's own activation bytes. Returns `(cpu, gpu)`,
/// each `tokens × hidden`.
#[allow(clippy::too_many_arguments)]
fn both_chains(
    device: &Device,
    dev: &CudaDevice,
    slot: &[u8],
    offsets: (usize, usize, usize),
    hidden: usize,
    ff: usize,
    ko: Ko,
    dt: GgmlDType,
    x: &[f32],
    tokens: usize,
    n_sub: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let stream = dev.cuda_stream();
    let xd = dev.memcpy_stod(x)?;
    let op = {
        let (p, _g) = xd.device_ptr(&stream);
        quantize_acts_q8a128(p, 2, tokens, hidden, dev, Backing::Owned, SumScale::Raw)?
    };
    let wdev = dev.memcpy_stod(slot)?;
    let table = {
        let (base, _g) = wdev.device_ptr(&stream);
        dev.memcpy_stod(&[
            base + offsets.0 as u64,
            base + offsets.1 as u64,
            base + offsets.2 as u64,
        ])?
    };
    let ids = Tensor::from_vec(vec![0u32; tokens], (tokens, 1), device)?;
    let mut ws = MoeBucketizeWorkspace::new(dev, tokens, 1)?;
    moe_bucketize(&ids, 1, 16 * n_sub, &mut ws, None, &DecodeRows::none())?;
    let stacked = fused_moe_gather_q8a128(&op, &ws.tok_ids, tokens, dev, Backing::Owned)?;
    let launch_tiles = tokens.div_ceil(16 * n_sub) + 1;
    let proj = |acts, base: usize, nrows: usize| {
        grouped_qmatmul_dev_q8a128(
            acts,
            &table,
            base,
            1,
            dt,
            nrows,
            &ws.tile_expert,
            &ws.tile_b_start,
            &ws.tile_b_cnt,
            launch_tiles,
            n_sub,
            None,
            dev,
        )
    };
    let g = proj(&stacked, 0, ff)?;
    let u = proj(&stacked, 1, ff)?;
    let inter = silu_mul_q8a128(&g, &u, dev, Backing::Owned, SumScale::Raw)?;
    let y = proj(&inter, 2, hidden)?;
    let gpu: Vec<f32> = y.flatten_all()?.to_vec1()?;

    let bytes = op.with_device_ptr(dev, |p| {
        let mut v = vec![0u8; op.byte_len()];
        // SAFETY: `p` is the operand's live allocation of `byte_len` bytes.
        unsafe { dev.read_raw(p, &mut v) }?;
        Ok(v)
    })?;
    let rows: Vec<usize> = (0..tokens).collect();
    let act = ActTiles::from_q8a1024(&bytes, &rows, hidden / TILE, false);
    let e = Expert::from_slot(slot, offsets, hidden, ff, ko);
    let mut cpu = vec![0f32; tokens * hidden];
    forward(Isa::detect(), &e, &act, &mut cpu);
    Ok((cpu, gpu))
}

#[test]
fn the_cpu_expert_is_within_the_gpus_error() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let Device::Cuda(dev) = &device else {
        unreachable!("a CUDA device")
    };
    // Flash-Next's expert: 2560 → 640 → 2560.
    let (hidden, ff, tokens) = (2560usize, 640usize, 6usize);
    for (ko, dt) in [(Ko::Q3, GgmlDType::Q3_KO), (Ko::Q4, GgmlDType::Q4_KO)] {
        let mut seed = 0x9a71_7e57;
        let mut w = |n: usize| -> Vec<f32> { (0..n).map(|_| lcg(&mut seed) * 0.05).collect() };
        let (wg, wu, wd) = (w(ff * hidden), w(ff * hidden), w(hidden * ff));
        let (bg, bu, bd) = (
            quantize_ko(&wg, ff, hidden, dt),
            quantize_ko(&wu, ff, hidden, dt),
            quantize_ko(&wd, hidden, ff, dt),
        );
        let offsets = (0, bg.len(), bg.len() + bu.len());
        let slot = [bg.clone(), bu.clone(), bd.clone()].concat();
        let mut seed = 0x0ac7_5eed;
        let x: Vec<f32> = (0..tokens * hidden).map(|_| lcg(&mut seed)).collect();
        let (cpu, gpu) = both_chains(
            &device, dev, &slot, offsets, hidden, ff, ko, dt, &x, tokens, 2,
        )?;

        // The f32 chain over the dequantized weights and the raw activations.
        let (dg, du, dd) = (
            dequant_ko(&bg, ff, hidden, dt),
            dequant_ko(&bu, ff, hidden, dt),
            dequant_ko(&bd, hidden, ff, dt),
        );
        let mv = |w: &[f32], rows: usize, cols: usize, v: &[f64]| -> Vec<f64> {
            (0..rows)
                .map(|r| (0..cols).map(|k| w[r * cols + k] as f64 * v[k]).sum())
                .collect()
        };
        let mut want = Vec::with_capacity(tokens * hidden);
        for t in 0..tokens {
            let xs = widen(&x[t * hidden..(t + 1) * hidden]);
            let (gv, uv) = (mv(&dg, ff, hidden, &xs), mv(&du, ff, hidden, &xs));
            let h: Vec<f64> = gv
                .iter()
                .zip(&uv)
                .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
                .collect();
            want.extend(mv(&dd, hidden, ff, &h));
        }
        let (e_cpu, e_gpu) = (rel(&cpu, &want), rel(&gpu, &want));
        let apart = rel(&cpu, &widen(&gpu));
        println!(
            "{ko:?}: rel L2 to f32 — CPU {e_cpu:.3e}, GPU {e_gpu:.3e}; CPU vs GPU {apart:.3e}"
        );
        assert!(
            e_cpu <= 1.1 * e_gpu,
            "{ko:?}: the CPU's error {e_cpu:.3e} exceeds 1.1× the GPU's {e_gpu:.3e}"
        );
    }
    Ok(())
}

/// **Where the CPU's SiLU parts from the device's, and why.** Four million
/// gate values in [−24, 24] through the device's own F32 SiLU (the eager
/// fused SiLU-mul at `up = 1`, which multiplies by one exactly) and through
/// the CPU's `device_silu`, compared bit for bit. Every disagreement must be
/// the sigmoid's division alone: fast math divides as `rcp.approx` /
/// `div.approx` — a reciprocal within an ulp, and `e · rcp(1 + e)` for the
/// negative branch — and some reciprocal within an ulp of the true one,
/// carried through those steps, must reproduce the device's value exactly.
/// The exponential, the add, the final multiply and the branch are then the
/// device's bit for bit, and the division is the whole difference.
#[test]
fn the_cpu_silu_parts_from_the_device_only_at_the_division() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let n = 1usize << 22;
    let mut seed = 0x511u64;
    let x: Vec<f32> = (0..n).map(|_| lcg(&mut seed) * 24.0).collect();
    let g = Tensor::from_vec(x.clone(), n, &device)?;
    let one = Tensor::ones(n, candle::DType::F32, &device)?;
    let gpu: Vec<f32> = candle_nn::ops::silu_mul(&g, &one)?.to_vec1()?;
    let (mut equal, mut by_division, mut other) = (0usize, 0usize, Vec::new());
    for (&xi, &gi) in x.iter().zip(&gpu) {
        let cpu = device_silu(xi);
        if cpu.to_bits() == gi.to_bits() {
            equal += 1;
            continue;
        }
        // Fast math divides as PTX's `rcp.approx` / `div.approx` do: the
        // reciprocal of the denominator, approximate to within an ulp, and for
        // `e / (1 + e)` that reciprocal times `e`, rounded — two roundings, not
        // one quotient. Every candidate reciprocal within an ulp of the true
        // one, carried through those same steps.
        let (num, den) = if xi >= 0.0 {
            (1.0f32, 1.0 + device_exp(-xi))
        } else {
            let e = device_exp(xi);
            (e, 1.0 + e)
        };
        let r = 1.0 / den;
        let reciprocals = [
            r,
            f32::from_bits(r.to_bits() + 1),
            f32::from_bits(r.to_bits() - 1),
        ];
        let explained = reciprocals.iter().any(|&rc| {
            let s = if xi >= 0.0 { rc } else { num * rc };
            (xi * s).to_bits() == gi.to_bits()
        });
        if explained {
            by_division += 1;
        } else {
            other.push((xi, cpu, gi));
        }
    }
    println!(
        "{n} gate values: {equal} equal, {by_division} the device's approximate division \
         reproduces, {} otherwise apart{}",
        other.len(),
        other
            .first()
            .map(|(x, c, g)| format!(", e.g. x {x:e}: CPU {c:e} device {g:e}"))
            .unwrap_or_default()
    );
    assert!(
        other.is_empty(),
        "disagreements the division does not explain"
    );
    Ok(())
}

fn tokenizer_json(repo: &str, rev: &str) -> Result<String> {
    let repo = Repo::with_revision(repo.to_string(), RepoType::Model, rev.to_string());
    let p: PathBuf = Api::new()
        .map_err(|e| candle::Error::Msg(format!("hf api: {e}")))?
        .repo(repo)
        .get("tokenizer.json")
        .map_err(|e| candle::Error::Msg(format!("fetch tokenizer.json: {e}")))?;
    std::fs::read_to_string(&p).map_err(|e| candle::Error::Msg(format!("read {p:?}: {e}")))
}

/// **The model pack's own experts**: records read straight from the expert
/// section — the bytes the warm tier and every slot hold — through both
/// chains, at every small count and at the prompt widths. The pool reads a
/// real slot image exactly as the GPU does, and its SwiGLU and down-input
/// quantization are the device's, so the two agree bit for bit outside the
/// rare value the device's approximate division in its SiLU rounds across a
/// code boundary.
#[test]
#[ignore = "reads the Flash-Next model pack"]
fn the_cpu_expert_matches_the_gpu_on_the_packs_experts() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let Device::Cuda(dev) = &device else {
        unreachable!("a CUDA device")
    };
    let path = prepared_engine_pack(&device, Some(Int8Mode::auto(&device)), &tokenizer_json)?;
    let pack = ModelPack::open(&path)?;
    let at = pack.experts.expect("the pack has an expert section");
    let section = open_section(&pack.path, at.offset, dev)?;
    let geoms = geometries_of(section.header())?;
    let record = Mapped::new(section.stride())?;
    let mut worst = 0f64;
    // Every small count at decode's narrow tiles — the counts the split hands
    // the pool — and a prompt's wide tiles over prompt-sized counts.
    let shapes = (1..=9usize).map(|t| (2usize, t)).chain([(4, 40), (8, 130)]);
    for (n_sub, tokens) in shapes {
        for layer in [0usize, 1, 23, geoms.len() - 1] {
            let g = &geoms[layer];
            let (hidden, ff) = (g.gate_shape[1], g.gate_shape[0]);
            let ko = match g.gate_dtype {
                GgmlDType::Q3_KO => Ko::Q3,
                GgmlDType::Q4_KO => Ko::Q4,
                other => panic!("layer {layer}: experts are {other:?}"),
            };
            let lay = section.layout(layer);
            let offsets = (lay.gate.offset, lay.up.offset, lay.down.offset);
            for expert in [0usize, 7, 300] {
                // SAFETY: one stride of pinned, page-aligned memory, owned here.
                let dest =
                    unsafe { std::slice::from_raw_parts_mut(record.host(), section.stride()) };
                section.read_into(layer, expert, dest)?;
                let mut seed = 0x9e7 + (layer * 1000 + expert) as u64;
                let x: Vec<f32> = (0..tokens * hidden).map(|_| lcg(&mut seed) * 2.0).collect();
                let (cpu, gpu) = both_chains(
                    &device,
                    dev,
                    dest,
                    offsets,
                    hidden,
                    ff,
                    ko,
                    g.gate_dtype,
                    &x,
                    tokens,
                    n_sub,
                )?;
                let apart = rel(&cpu, &widen(&gpu));
                println!(
                "n_sub {n_sub} × {tokens}: L{layer} E{expert} ({:?}/{:?}, {hidden}→{ff}): CPU vs GPU {apart:.3e}",
                g.gate_dtype, g.down_dtype
            );
                worst = worst.max(apart);
            }
        }
    }
    // Bit-identical but where the device's approximate division in its SiLU
    // puts a SwiGLU value an ulp across an int8 code boundary: one flipped
    // code over 130 rows measured 2.3e-4.
    assert!(
        worst < 1e-3,
        "the pool and the GPU part on the pack's experts: {worst:.3e}"
    );
    Ok(())
}
