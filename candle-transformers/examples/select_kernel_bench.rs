//! Individual-kernel microbench + ncu target for the two batched Stage-1
//! corpus-selection kernels (`bdp_recall_batched`, `topm_select_batched`).
//!
//! Builds a realistic deep decode batch (`sessions` concurrent sessions, each
//! with `entries` compressed gallery rows) and loops ONLY those two kernels, so
//! a profiler attaches to them and nothing else. Seconds to run.
//!
//! Usage:
//!   cargo run -p candle-transformers --example select_kernel_bench \
//!       --features cuda --release -- [sessions] [entries] [iters]
//!
//! Profile just the two kernels (skip the one-time setup launches):
//!   ncu --kernel-name-base regex \
//!       --kernel-name "bdp_recall_batched_kernel|topm_.*_batched_kernel" \
//!       --launch-count 8 --set full \
//!       target/release/examples/select_kernel_bench 64 8192 64

#[cfg(feature = "cuda")]
fn main() -> candle::Result<()> {
    use candle::Device;
    use candle_transformers::models::latent_moe::select_bench::{run_select_kernels, SelectCfg};

    let a: Vec<String> = std::env::args().collect();
    let parse = |i: usize, d: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let cfg = SelectCfg {
        sessions: parse(1, 64),
        entries: parse(2, 8192),
        top_k: 512,
        warmup: 20,
        iters: 200,
        seed: 0x5E1E_C7ED,
    };
    let iters = parse(3, 200);

    let t = std::time::Instant::now();
    let dev = Device::new_cuda(0)?;
    eprintln!("[bench] cuda init {:.2}s", t.elapsed().as_secs_f64());
    run_select_kernels(&dev, cfg, iters)?;
    eprintln!("[bench] total {:.2}s", t.elapsed().as_secs_f64());
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("select_kernel_bench requires the `cuda` feature");
}
