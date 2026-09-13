//! Individual-kernel microbench + `ncu` target for the fused Gated-DeltaNet
//! prefill scan (`dn:mix`).
//!
//! §0.4 rule 4: a kernel that has not been measured is not finished, and this
//! applies to kernels we *reuse* at a new geometry as much as to ones we
//! write. The fused scan was tuned for the Qwen3.5/3.6 lineage; Qwen3.8-Flash-Next
//! runs it at **48 V heads / 16 K heads @ 128** over 36 of its 48 layers, and
//! nothing established its behaviour there. A span timer says `dn:mix` is 5.9%
//! of the profile; only a profiler says whether that is the bound or the floor.
//!
//! # The two rules this harness exists to satisfy
//!
//! **A per-run correctness gate**, so a fast wrong kernel cannot pass: the
//! fused CUDA path is checked against the tensor-op path — the reference the
//! kernels are parity-locked to — on the CPU, at a small size, before anything
//! is timed.
//!
//! **Sized past the L2**, because this card has 96 MiB of it and a microbench
//! that fits is a microbench that lies. [`DeltaNetBenchCfg::working_set_bytes`]
//! reports what the timed configuration actually touches, and
//! [`run_delta_net_kernels`] prints it beside the L2 so the reader can see
//! which side of the line the number came from.

use candle::{Device, Result, Tensor};

use super::mix::{delta_net_mix, delta_net_mix_spans, DeltaNetProjections, DeltaNetSeq};
use super::types::{DeltaNetDims, ZGate};
use super::{DeltaNetConstants, DeltaNetState};

/// The card's L2. A timed working set below this measures cache, not memory.
const L2_BYTES: usize = 96 * 1024 * 1024;

/// One benchmark point.
#[derive(Clone, Copy, Debug)]
pub struct DeltaNetBenchCfg {
    /// Tokens in the packed buffer, split evenly across `seqs`.
    pub tokens: usize,
    /// Concurrent sequences — the wave shape, since each carries its own state
    /// and the scan walks its own span.
    pub seqs: usize,
    pub dims: DeltaNetDims,
    pub warmup: usize,
    pub iters: usize,
    pub seed: u64,
}

impl DeltaNetBenchCfg {
    /// Qwen3.8-Flash-Next's Gated-DeltaNet geometry (§12.6 / §3 of the design
    /// doc): 48 V heads, 16 K heads, head_dim 128, conv kernel 4.
    pub fn qwen4exp(tokens: usize, seqs: usize) -> Self {
        Self {
            tokens,
            seqs,
            dims: DeltaNetDims {
                head_dim: 128,
                n_k_heads: 16,
                n_v_heads: 48,
                conv_kernel: 4,
            },
            warmup: 10,
            iters: 50,
            seed: 0xD3_17A_4E7,
        }
    }

    /// Bytes the timed call reads or writes once: the four projections, the
    /// output, and the per-sequence state.
    ///
    /// The state is the interesting term — `n_v_heads · head_dim²` F32 per
    /// sequence, 3 MiB here — because the chunked form's whole claim is that it
    /// pays that once per chunk rather than once per token.
    pub fn working_set_bytes(&self) -> usize {
        let d = &self.dims;
        let per_token = d.conv_dim() + d.value_dim() + 2 * d.n_v_heads + d.value_dim();
        let state = self.seqs * d.n_v_heads * d.head_dim * d.head_dim;
        (self.tokens * per_token + state) * std::mem::size_of::<f32>()
    }
}

fn lcg(shape: &[usize], seed: u64, dev: &Device) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let vals: Vec<f32> = (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect();
    Tensor::from_vec(vals, shape, dev)
}

fn projections(t: usize, dims: &DeltaNetDims, seed: u64, dev: &Device) -> Result<Projections> {
    Ok(Projections {
        qkv: lcg(&[t, dims.conv_dim()], seed ^ 0x91, dev)?,
        z: lcg(&[t, dims.value_dim()], seed ^ 0x92, dev)?,
        beta_lin: lcg(&[t, dims.n_v_heads], seed ^ 0x93, dev)?,
        alpha_lin: lcg(&[t, dims.n_v_heads], seed ^ 0x94, dev)?,
    })
}

/// Owned projections; `DeltaNetProjections` borrows, so the bench holds these.
struct Projections {
    qkv: Tensor,
    z: Tensor,
    beta_lin: Tensor,
    alpha_lin: Tensor,
}

impl Projections {
    fn view(&self) -> DeltaNetProjections<'_> {
        DeltaNetProjections {
            qkv: self.qkv.clone(),
            z: self.z.clone(),
            beta_lin: self.beta_lin.clone(),
            alpha_lin: self.alpha_lin.clone(),
        }
    }
}

fn constants(
    dims: &DeltaNetDims,
    seed: u64,
    dev: &Device,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let dt_bias = lcg(&[dims.n_v_heads], seed ^ 0x95, dev)?;
    // `a` is `−exp(A_log)` in the checkpoint, so strictly negative.
    let a = lcg(&[dims.n_v_heads], seed ^ 0x96, dev)?.abs()?.neg()?;
    let conv = lcg(&[dims.conv_dim(), dims.conv_kernel], seed ^ 0x97, dev)?;
    let norm = lcg(&[dims.head_dim], seed ^ 0x98, dev)?.affine(0.2, 1.0)?;
    Ok((dt_bias, a, conv, norm))
}

/// The gate: the fused CUDA scan against the tensor-op path on the CPU, which
/// is the reference these kernels are parity-locked to.
///
/// Small on purpose — the point is to catch a wrong kernel, and the CPU
/// reference is a scalar walk. A gate that took as long as the benchmark would
/// simply not be run.
fn correctness_gate(dev: &Device, dims: &DeltaNetDims, seed: u64, eps: f64) -> Result<f32> {
    const GATE_TOKENS: usize = 96;
    let cpu = Device::Cpu;

    let p_cpu = projections(GATE_TOKENS, dims, seed, &cpu)?;
    let (dt_bias, a, conv, norm) = constants(dims, seed, &cpu)?;
    let want = {
        let c = DeltaNetConstants {
            dt_bias: &dt_bias,
            a: &a,
            conv: &conv,
            norm: &norm,
        };
        let mut st = DeltaNetState::zeros(dims, &cpu)?;
        delta_net_mix(&p_cpu.view(), &c, dims, &mut st, eps, ZGate::Sigmoid)?
    };

    let p_gpu = projections(GATE_TOKENS, dims, seed, dev)?;
    let (dt_bias, a, conv, norm) = constants(dims, seed, dev)?;
    let got = {
        let c = DeltaNetConstants {
            dt_bias: &dt_bias,
            a: &a,
            conv: &conv,
            norm: &norm,
        };
        let mut st = DeltaNetState::zeros(dims, dev)?;
        let out = st.solo_out()?;
        let mut seqs = [DeltaNetSeq {
            start: 0,
            len: GATE_TOKENS,
            state: &mut st,
            out,
            stash: None,
        }];
        delta_net_mix_spans(
            &p_gpu.view(),
            &c,
            dims,
            &mut seqs,
            eps,
            None,
            ZGate::Sigmoid,
        )?
        .to_owned_tensor()?
    };

    let g = got.to_device(&cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let w = want.flatten_all()?.to_vec1::<f32>()?;
    let scale = w.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
    Ok(g.iter()
        .zip(&w)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()))
        / scale)
}

/// Gate, then time the fused scan.
///
/// Prints µs per call, the working set against the L2, and the achieved
/// bandwidth — enough to say whether a later change moved anything, and enough
/// to tell a bandwidth bound from a latency one before reaching for `ncu`.
pub fn run_delta_net_kernels(dev: &Device, cfg: DeltaNetBenchCfg) -> Result<()> {
    let eps = 1e-6;
    let dims = cfg.dims;

    let gap = correctness_gate(dev, &dims, cfg.seed, eps)?;
    if gap > 2e-4 {
        candle::bail!(
            "delta_net bench: the fused scan disagrees with the tensor-op reference \
             by {gap} — timing a wrong kernel measures nothing"
        );
    }
    println!("correctness gate: fused vs tensor-op reference, rel gap {gap:.2e}");

    let per_seq = cfg.tokens / cfg.seqs;
    let total = per_seq * cfg.seqs;
    let p = projections(total, &dims, cfg.seed, dev)?;
    let (dt_bias, a, conv, norm) = constants(&dims, cfg.seed, dev)?;
    let c = DeltaNetConstants {
        dt_bias: &dt_bias,
        a: &a,
        conv: &conv,
        norm: &norm,
    };
    let mut states: Vec<DeltaNetState> = (0..cfg.seqs)
        .map(|_| DeltaNetState::zeros(&dims, dev))
        .collect::<Result<_>>()?;

    let ws = cfg.working_set_bytes();
    println!(
        "geometry {}h_v/{}h_k @ {} | {total} tokens over {} seqs | working set {:.1} MiB ({} L2)",
        dims.n_v_heads,
        dims.n_k_heads,
        dims.head_dim,
        cfg.seqs,
        ws as f64 / (1024.0 * 1024.0),
        if ws > L2_BYTES {
            "PAST"
        } else {
            "INSIDE — measuring cache"
        },
    );

    let once = |states: &mut Vec<DeltaNetState>| -> Result<()> {
        let mut outs = Vec::with_capacity(states.len());
        for st in states.iter_mut() {
            outs.push(st.solo_out()?);
        }
        let mut seqs: Vec<DeltaNetSeq<'_>> = states
            .iter_mut()
            .zip(outs)
            .enumerate()
            .map(|(i, (state, out))| DeltaNetSeq {
                start: i * per_seq,
                len: per_seq,
                state,
                out,
                stash: None,
            })
            .collect();
        let _ = delta_net_mix_spans(&p.view(), &c, &dims, &mut seqs, eps, None, ZGate::Sigmoid)?;
        Ok(())
    };

    for _ in 0..cfg.warmup {
        once(&mut states)?;
    }
    dev.synchronize()?;

    let t0 = std::time::Instant::now();
    for _ in 0..cfg.iters {
        once(&mut states)?;
    }
    dev.synchronize()?;
    let us = t0.elapsed().as_secs_f64() * 1e6 / cfg.iters as f64;

    // The chunked scan's arithmetic, per (chunk, V head): the A build, the
    // solve, `v_new = u − wSᵀ`, the two output terms and the state update.
    let chunks = total.div_ceil(64);
    let per_chunk_head = 64 * 64 * dims.head_dim          // A
        + 64 * dims.head_dim * dims.head_dim              // v_new
        + 64 * dims.head_dim * dims.head_dim              // q Sᵀ
        + 64 * 64 * dims.head_dim                         // kq · v_new
        + dims.head_dim * 64 * dims.head_dim; // S update
    let gflop = 2.0 * (per_chunk_head as f64) * (chunks as f64) * (dims.n_v_heads as f64) / 1e9;

    println!(
        "dn:mix  {us:>9.1} us/call   {:>6.1} GFLOP/s   {:>6.1} GB/s (working set / call)",
        gflop / (us * 1e-6) / 1e9 * 1e9,
        ws as f64 / (us * 1e-6) / 1e9,
    );
    Ok(())
}
