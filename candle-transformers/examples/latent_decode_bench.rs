//! Microbenchmark + profiling target for the paged latent-attention DECODE
//! kernel (batched hybrid window + compressed attention, single latent K≡V).
//!
//! Builds a realistic deep-context wave — `slots` concurrent decode sessions,
//! each a 128-token window at depth `D` plus a scattered top-`topk` selection
//! into a `D/ratio`-entry compressed gallery — without loading the model, then
//! times a warmup+launch loop and validates the output against a table-faithful
//! reference. Seconds to run.
//!
//! Usage:
//!   cargo run -p candle-transformers --example latent_decode_bench \
//!       --features cuda --release -- [slots] [depth_tokens] [topk] [iters]
//!
//! Profile just the attention kernels (skip the one-time setup launches):
//!   ncu --kernel-name-base regex --kernel-name "latent_(decode|combine)_kernel" \
//!       --launch-count 4 \
//!       target/release/examples/latent_decode_bench 64 200000 512 32

#[cfg(feature = "cuda")]
fn main() -> candle::Result<()> {
    use candle::Device;
    use candle_transformers::models::deepseek4::bench::{run_decode, DecodeCfg};

    let a: Vec<String> = std::env::args().collect();
    let parse = |i: usize, d: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let cfg = DecodeCfg {
        slots: parse(1, 64),
        depth_tokens: parse(2, 200_000),
        topk: parse(3, 512),
        iters: parse(4, 200),
        splits: parse(5, 0),
        ..DecodeCfg::default()
    };

    let t = std::time::Instant::now();
    let dev = Device::new_cuda(0)?;
    eprintln!("[bench] cuda init {:.2}s", t.elapsed().as_secs_f64());
    let report = run_decode(&dev, cfg)?;
    report.print();
    eprintln!("[bench] total {:.2}s", t.elapsed().as_secs_f64());
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("latent_decode_bench requires the `cuda` feature");
}
