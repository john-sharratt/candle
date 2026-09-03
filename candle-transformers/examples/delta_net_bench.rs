//! Individual-kernel microbench + `ncu` target for the fused Gated-DeltaNet
//! prefill scan (`dn:mix`), at Qwen3.8-Flash-Next's geometry.
//!
//! The scan was tuned for the Qwen3.5/3.6 lineage; this model runs it at
//! 48 V heads / 16 K heads @ 128 across 36 of its 48 layers, where its
//! behaviour was never established (§0.4 rule 4 — reusing a kernel at a new
//! geometry means extending its harness to cover that geometry).
//!
//! Gates against the tensor-op reference before timing, and reports the timed
//! working set against the card's 96 MiB L2 so a cache-resident number cannot
//! be mistaken for a memory-bound one.
//!
//! Usage:
//!   cargo run -p candle-transformers --example delta_net_bench \
//!       --features cuda --release -- [tokens] [seqs] [iters]
//!
//! Profile one kernel of the scan (the setup launches are excluded by name):
//!   ncu -k delta_net_prefill_state_f32_kernel --launch-count 3 --set full \
//!       target/release/examples/delta_net_bench 4096 2 2
//!
//! The three kernels are `delta_net_conv_prefill_f32_kernel`,
//! `delta_net_prefill_intra_f32_kernel` and `delta_net_prefill_state_f32_kernel`.
//! `-k` takes a plain substring or `regex:<expr>`, but on Windows the `.bat`
//! wrapper hands its arguments to `cmd`, which eats `(`, `)` and `|` — so
//! profile one kernel per invocation rather than writing an alternation.
//! Note the FIRST profiled launch is the correctness gate's (96 tokens): its
//! grid is a fraction of the timed one, so read the later launches.

#[cfg(feature = "cuda")]
fn main() -> candle::Result<()> {
    use candle::Device;
    use candle_transformers::models::delta_net::bench::{run_delta_net_kernels, DeltaNetBenchCfg};

    let a: Vec<String> = std::env::args().collect();
    let parse = |i: usize, d: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    // 8192 tokens over 4 sequences puts the working set past the L2; the
    // bench prints which side of that line it landed on.
    let mut cfg = DeltaNetBenchCfg::qwen4exp(parse(1, 8192), parse(2, 4));
    cfg.iters = parse(3, 50);

    let t = std::time::Instant::now();
    let dev = Device::new_cuda(0)?;
    eprintln!("[bench] cuda init {:.2}s", t.elapsed().as_secs_f64());
    run_delta_net_kernels(&dev, cfg)?;
    eprintln!("[bench] total {:.2}s", t.elapsed().as_secs_f64());
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("delta_net_bench requires the `cuda` feature");
}
