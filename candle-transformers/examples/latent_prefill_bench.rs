//! Microbenchmark + profiling target for the paged latent-attention PREFILL
//! kernel (many queries over a settled slot, fresh-diagonal key source).
//!
//! Builds a new prompt of `total_q` tokens absorbed over an existing
//! `depth`-token context — a 128-token settled window plus a `depth/ratio`-entry
//! compressed gallery, each query selecting a scattered top-`topk` — without
//! loading the model, then times a warmup+launch loop and validates against a
//! table-faithful reference. Seconds to run.
//!
//! Usage:
//!   cargo run -p candle-transformers --example latent_prefill_bench \
//!       --features cuda --release -- [total_q] [depth_tokens] [topk] [iters]
//!
//! Profile just the attention kernels:
//!   ncu --kernel-name-base regex --kernel-name "latent_prefill.*kernel" \
//!       --launch-count 8 \
//!       target/release/examples/latent_prefill_bench 4096 200000 512 8

#[cfg(feature = "cuda")]
fn main() -> candle::Result<()> {
    use candle::Device;
    use candle_transformers::models::latent_moe::bench::{run_prefill, PrefillCfg};

    let a: Vec<String> = std::env::args().collect();
    let parse = |i: usize, d: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let cfg = PrefillCfg {
        total_q: parse(1, 4096),
        depth_tokens: parse(2, 200_000),
        topk: parse(3, 512),
        iters: parse(4, 50),
        ..PrefillCfg::default()
    };

    let dev = Device::new_cuda(0)?;
    let report = run_prefill(&dev, cfg)?;
    report.print();
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("latent_prefill_bench requires the `cuda` feature");
}
