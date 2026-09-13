//! Individual-kernel microbench + `ncu` target for the three Gated-Residual
//! kernels (`gr_norm`, `gr_mix`, `gr_combine`).
//!
//! §0.4 rule 4: a kernel that has not been measured is not finished. The
//! fusion landed on a *span* timer — `hc_mix:norm` −65%, `gate_mean` −72%,
//! combine −50% — and a span timer can only say a kernel got faster than the
//! chain it replaced. It cannot say whether what remains is the memory floor
//! or three quarters of it, and that is the only question worth asking next.
//!
//! # What the harness has to get right
//!
//! **Gate before timing, fused against eager, on the same device.** Not
//! against the CPU. candle's CPU sigmoid is a precise `1/(1+exp(−x))` while
//! its CUDA sigmoid is `fast_exp::sigmoid`, a cubic polynomial — so the two
//! eager paths already disagree by ~2e-5, and a tolerance wide enough to admit
//! that is wide enough to admit a kernel computing a different formula. That
//! is not hypothetical here: the first cut of these kernels rolled its own
//! `1/(1 + __expf(-x))`, was ~400× *more* accurate than the path it replaced,
//! and the first symptom was a KV calibration rung going red two runs later.
//! [`GATE`] is set where reassociation lives and nowhere near where a changed
//! formula does.
//!
//! **Size past the L2.** This card has 96 MiB of it. These kernels are
//! memory-shaped — one or two reads and a write of the wide `[T, hc·n_embd]`
//! residual — so a working set that fits in L2 reports the cache's bandwidth
//! and calls it the kernel's. [`GrBenchCfg::working_set_bytes`] reports what
//! the timed configuration touches and [`run_gr_kernels`] prints it against
//! the L2, so a cache-resident number cannot be read as a memory-bound one.
//!
//! The reported GB/s is each kernel's own traffic — the bytes it must read and
//! write at minimum — over its own time. For kernels this shape that figure is
//! the whole verdict: at the card's achievable bandwidth there is nothing left
//! to win, and well under it there is.

use candle::{DType, Device, Result, Tensor};

use super::cuda_fused;
use super::{eager_combine, eager_gate_mean, eager_grouped_norm};

/// The card's L2. A timed working set below this measures cache, not memory.
const L2_BYTES: usize = 96 * 1024 * 1024;

/// Admissible fused-vs-eager gap: reassociation only. The block tree-reduction
/// sums `n_embd` squares in a different order than `rms_norm` does, and that
/// is the entire licence — see the module header for what a wider one would
/// let through.
const GATE: f32 = 2e-6;

/// One benchmark point.
#[derive(Clone, Copy, Debug)]
pub struct GrBenchCfg {
    /// Tokens in the wide residual — the axis that decides the working set.
    pub tokens: usize,
    /// Parallel residual streams (`hc`).
    pub hc: usize,
    pub n_embd: usize,
    /// The read gate's bottleneck (`low_rank` in §12.3): the two projections are
    /// `hc_dim → low_rank → hc_dim`.
    pub low_rank: usize,
    pub warmup: usize,
    pub iters: usize,
    pub seed: u64,
}

impl GrBenchCfg {
    /// Qwen3.8-Flash-Next's Gated Residual geometry (§12.3): 4 streams over
    /// `n_embd` 2560, so the wide residual is `[T, 10240]`, through a rank-320
    /// gate.
    pub fn qwen4exp(tokens: usize) -> Self {
        Self {
            tokens,
            hc: 4,
            n_embd: 2560,
            low_rank: 320,
            warmup: 10,
            iters: 50,
            seed: 0x9E37_79B9,
        }
    }

    fn hc_dim(&self) -> usize {
        self.hc * self.n_embd
    }

    /// Bytes one pass over the wide residual reads and writes. All three
    /// kernels are within a few percent of this — they differ only in the
    /// narrow `[T, n_embd]` / `[T, hc]` operands beside it — so it is the
    /// figure that decides which side of the L2 the whole bench sits on.
    pub fn working_set_bytes(&self) -> usize {
        2 * self.tokens * self.hc_dim() * std::mem::size_of::<f32>()
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

/// Largest elementwise gap, relative to the reference's own magnitude.
fn rel_gap(got: &Tensor, want: &Tensor) -> Result<f32> {
    let g = got.flatten_all()?.to_vec1::<f32>()?;
    let w = want.flatten_all()?.to_vec1::<f32>()?;
    let scale = w.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
    Ok(g.iter()
        .zip(&w)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()))
        / scale)
}

/// Time `f` after `warmup` untimed calls, returning µs per call.
fn time_call(dev: &Device, cfg: &GrBenchCfg, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..cfg.warmup {
        f()?;
    }
    dev.synchronize()?;
    let t0 = std::time::Instant::now();
    for _ in 0..cfg.iters {
        f()?;
    }
    dev.synchronize()?;
    Ok(t0.elapsed().as_secs_f64() * 1e6 / cfg.iters as f64)
}

/// Gate all three kernels against the eager reference, then time them.
///
/// The gate runs at a small width — it exists to catch a wrong kernel, and one
/// that took as long as the benchmark would simply not be run — but at the
/// production `n_embd`, because the reduction width is what the reassociation
/// bound is about.
pub fn run_gr_kernels(dev: &Device, cfg: GrBenchCfg) -> Result<()> {
    let eps = 1e-6;
    let (hc, d) = (cfg.hc, cfg.n_embd);
    let hc_dim = cfg.hc_dim();

    {
        const GATE_TOKENS: usize = 8;
        let x = lcg(&[GATE_TOKENS, hc, d], cfg.seed ^ 0x11, dev)?;
        let wn = lcg(&[hc_dim], cfg.seed ^ 0x12, dev)?.affine(0.2, 1.0)?;
        let gate_raw = lcg(&[GATE_TOKENS, hc_dim], cfg.seed ^ 0x13, dev)?;
        let out = lcg(&[GATE_TOKENS, d], cfg.seed ^ 0x14, dev)?;
        let inj = lcg(&[GATE_TOKENS, hc], cfg.seed ^ 0x15, dev)?;

        let checks: [(&str, f32); 3] = [
            (
                "gr_norm",
                rel_gap(
                    &cuda_fused::norm(&x, &wn, eps)?,
                    &eager_grouped_norm(&x, &wn, eps)?,
                )?,
            ),
            (
                "gr_mix",
                rel_gap(
                    &cuda_fused::mix(&x, &gate_raw, hc, d)?,
                    &eager_gate_mean(
                        &x.reshape((GATE_TOKENS, hc_dim))?,
                        &gate_raw,
                        GATE_TOKENS,
                        hc,
                        d,
                    )?,
                )?,
            ),
            (
                "gr_combine",
                rel_gap(
                    &cuda_fused::combine(&x, &out, &inj)?,
                    &eager_combine(&x, &out, &inj)?,
                )?,
            ),
        ];
        for (name, gap) in checks {
            if gap > GATE {
                candle::bail!(
                    "gr bench: {name} disagrees with the eager reference by {gap} — that is \
                     past reassociation, so the kernel computes something else and timing it \
                     measures nothing"
                );
            }
        }
        let worst = checks.iter().fold(0f32, |m, (_, g)| m.max(*g));
        println!("correctness gate: fused vs eager on device, worst rel gap {worst:.2e}");
    }

    let t = cfg.tokens;
    let x = lcg(&[t, hc, d], cfg.seed ^ 0x21, dev)?;
    let wn = lcg(&[hc_dim], cfg.seed ^ 0x22, dev)?.affine(0.2, 1.0)?;
    let gate_raw = lcg(&[t, hc_dim], cfg.seed ^ 0x23, dev)?;
    let out = lcg(&[t, d], cfg.seed ^ 0x24, dev)?;
    let inj = lcg(&[t, hc], cfg.seed ^ 0x25, dev)?;

    let ws = cfg.working_set_bytes();
    println!(
        "geometry {hc} streams x {d} = {hc_dim} wide | {t} tokens | wide-residual pass {:.1} MiB ({})",
        ws as f64 / (1024.0 * 1024.0),
        if ws > L2_BYTES {
            "PAST L2"
        } else {
            "INSIDE L2 — measuring cache"
        },
    );

    let f4 = std::mem::size_of::<f32>() as f64;
    let wide = (t * hc_dim) as f64 * f4;
    let narrow = (t * d) as f64 * f4;

    // Each kernel's own minimum traffic, not the shared figure above: `mix`
    // reads two wide operands and writes one narrow, `combine` reads one wide
    // and one narrow and writes one wide.
    let rows: [(&str, f64, f64); 3] = [
        (
            "gr_norm",
            time_call(dev, &cfg, || {
                cuda_fused::norm(&x, &wn, eps)?;
                Ok(())
            })?,
            2.0 * wide,
        ),
        (
            "gr_mix",
            time_call(dev, &cfg, || {
                cuda_fused::mix(&x, &gate_raw, hc, d)?;
                Ok(())
            })?,
            2.0 * wide + narrow,
        ),
        (
            "gr_combine",
            time_call(dev, &cfg, || {
                cuda_fused::combine(&x, &out, &inj)?;
                Ok(())
            })?,
            2.0 * wide + narrow,
        ),
    ];

    // What a call costs before its kernel runs. Every one of these launches
    // allocates its own output, and allocation is a synchronising call — so the
    // loop cannot overlap iteration n's allocation with iteration n−1's kernel,
    // and this time sits in front of every launch rather than behind it. It is
    // charged to the wide `[T, hc_dim]` shape, which is what `norm` and
    // `combine` return; `mix` returns the narrow one and pays less.
    //
    // `zeros` is timed beside `empty` because the difference is exactly one
    // memset over the wide residual — the pass hot-path invariant 6 exists to
    // refuse, priced at this geometry.
    let alloc_us = time_call(dev, &cfg, || {
        Tensor::empty((t, hc, d), DType::F32, dev)?;
        Ok(())
    })?;
    let zero_us = time_call(dev, &cfg, || {
        Tensor::zeros((t, hc, d), DType::F32, dev)?;
        Ok(())
    })?;

    for (name, us, bytes) in rows {
        println!(
            "{name:<11} {us:>9.1} us/call   {:>7.1} GB/s   ({:.1} MiB moved)",
            bytes / (us * 1e-6) / 1e9,
            bytes / (1024.0 * 1024.0),
        );
    }

    // ── The projections, which the fused kernels deliberately do not touch ──
    //
    // §0.4 rule 4 covers kernels we *reuse*, and these are cuBLAS at a geometry
    // nothing established: `down` is tall-skinny (K = hc_dim, N = low_rank) and
    // `up` is its transpose-shaped twin (K = low_rank, N = hc_dim). Together
    // they are `hc_mix:lowrank`, the largest GR span and the one the fusion was
    // predicted — correctly — to leave alone.
    //
    // `inject` is measured beside them because it is **the same contraction
    // against the same operand**: `xn_flat @ W'` with `W` `[hc, hc_dim]` where
    // `down` has `[low_rank, hc_dim]`. Stacking the two weights once at load
    // makes one GEMM of `[low_rank + hc, hc_dim]` and drops a whole read of the
    // wide `xn_flat`. The `merged` row is that candidate, timed against
    // `down + inject` so the claim is a measurement rather than an argument.
    let lr = cfg.low_rank;
    let w_down = lcg(&[lr, hc_dim], cfg.seed ^ 0x31, dev)?.affine(0.05, 0.)?;
    let w_up = lcg(&[hc_dim, lr], cfg.seed ^ 0x32, dev)?.affine(0.05, 0.)?;
    let w_inj = lcg(&[hc, hc_dim], cfg.seed ^ 0x33, dev)?.affine(0.05, 0.)?;
    // The stacked weight is built ONCE, exactly as the loader would build it —
    // timing a per-call concatenation would measure a design nobody proposed.
    let w_stacked = Tensor::cat(&[&w_down, &w_inj], 0)?.contiguous()?;
    let xn_flat = cuda_fused::norm(&x, &wn, eps)?.reshape((t, hc_dim))?;
    let lo_t = lcg(&[t, lr], cfg.seed ^ 0x34, dev)?;

    let down_us = time_call(dev, &cfg, || {
        xn_flat.matmul(&w_down.t()?)?;
        Ok(())
    })?;
    let up_us = time_call(dev, &cfg, || {
        lo_t.matmul(&w_up.t()?)?;
        Ok(())
    })?;
    let inject_us = time_call(dev, &cfg, || {
        xn_flat.matmul(&w_inj.t()?)?;
        Ok(())
    })?;
    let merged_us = time_call(dev, &cfg, || {
        xn_flat.matmul(&w_stacked.t()?)?;
        Ok(())
    })?;

    let gflop = |n: usize| 2.0 * (t as f64) * (n as f64) * (hc_dim as f64) / 1e9;
    println!(
        "down  ({hc_dim}->{lr}) {down_us:>8.1} us/call   {:>6.1} GFLOP/s",
        gflop(lr) / (down_us * 1e-6)
    );
    println!(
        "up    ({lr}->{hc_dim}) {up_us:>8.1} us/call   {:>6.1} GFLOP/s",
        2.0 * (t as f64) * (lr as f64) * (hc_dim as f64) / 1e9 / (up_us * 1e-6)
    );
    println!(
        "inject({hc_dim}->{hc:>4}) {inject_us:>8.1} us/call   {:>6.1} GFLOP/s",
        gflop(hc) / (inject_us * 1e-6)
    );
    println!(
        "merged({hc_dim}->{:>4}) {merged_us:>8.1} us/call   vs down+inject {:.1} us  =>  {:+.1} us/call",
        lr + hc,
        down_us + inject_us,
        merged_us - (down_us + inject_us),
    );

    // What the merge actually costs its CONSUMERS. Splitting `[t, low_rank+hc]`
    // gives both halves a row stride of `low_rank+hc`, so neither slice is
    // dense — and a saving bought by handing the next GEMM a layout it has to
    // copy is not a saving (invariant 2). The `up` GEMM is the one that matters:
    // its operand is `[t, low_rank]`, 2.5 MiB at this width, and whether candle
    // reads that stride in place or materialises it is the difference between
    // this idea working and merely relocating the cost.
    let merged_out = xn_flat.matmul(&w_stacked.t()?)?;
    let lo_view = merged_out.narrow(1, 0, lr)?;
    let inj_view = merged_out.narrow(1, lr, hc)?;
    println!(
        "  split views: lo {:?} stride {:?} dense={} | inject {:?} stride {:?} dense={}",
        lo_view.dims(),
        lo_view.stride(),
        lo_view.is_contiguous(),
        inj_view.dims(),
        inj_view.stride(),
        inj_view.is_contiguous(),
    );
    // candle's matmul REFUSES a strided operand rather than silently copying it
    // ("matmul is only supported for contiguous tensors"), which is the honest
    // behaviour and makes the cost visible instead of hidden. So the `up` GEMM's
    // operand has to be compacted, and this is what that compaction costs —
    // `[t, low_rank]`, against the wide `[t, hc_dim]` read the merge removes.
    let lo_compact_us = time_call(dev, &cfg, || {
        lo_view.contiguous()?;
        Ok(())
    })?;
    let inj_compact_us = time_call(dev, &cfg, || {
        inj_view.contiguous()?;
        Ok(())
    })?;
    println!(
        "  compaction: lo [{t},{lr}] {lo_compact_us:.1} us | inject [{t},{hc}] {inj_compact_us:.1} us"
    );
    println!(
        "  NET merged+compaction {:.1} us vs down+inject {:.1} us  =>  {:+.1} us/call",
        merged_us + lo_compact_us + inj_compact_us,
        down_us + inject_us,
        (merged_us + lo_compact_us + inj_compact_us) - (down_us + inject_us),
    );
    println!(
        "wide alloc  {alloc_us:>10.1} us/call   (empty)   {zero_us:.1} us (zeros) — \
         in front of each launch, not kernel time"
    );

    // The eager chain the fusion replaced, at the same width — so the bench
    // reports the win it is defending, not just the absolute number.
    let xn = eager_grouped_norm(&x, &wn, eps)?;
    let xn_flat = xn.reshape((t, hc_dim))?;
    let eager: [(&str, f64); 3] = [
        (
            "gr_norm",
            time_call(dev, &cfg, || {
                eager_grouped_norm(&x, &wn, eps)?;
                Ok(())
            })?,
        ),
        (
            "gr_mix",
            time_call(dev, &cfg, || {
                eager_gate_mean(&xn_flat, &gate_raw, t, hc, d)?;
                Ok(())
            })?,
        ),
        (
            "gr_combine",
            time_call(dev, &cfg, || {
                eager_combine(&x, &out, &inj)?;
                Ok(())
            })?,
        ),
    ];
    for ((name, fused_us, _), (_, eager_us)) in rows.iter().zip(eager) {
        println!(
            "{name:<11} eager {eager_us:>9.1} us/call   fused is {:.2}x",
            eager_us / fused_us,
        );
    }

    Ok(())
}
