//! Individual-kernel microbench + ncu target for the fused corpus gather
//! (`corpus_gather_rows_kernel`).
//!
//! Builds `sessions` galleries and loops the per-session `gather_corpus_into`
//! that the decode wave uses to assemble its selected corpus block.
//!
//! Usage:
//!   cargo run -p candle-transformers --example corpus_gather_bench \
//!       --features cuda --release -- [sessions] [entries] [top_k] [iters]
//!
//! Profile just the gather kernel:
//!   ncu --kernel-name "regex:corpus_gather_rows_kernel" --launch-count 8 \
//!       --set full \
//!       target/release/examples/corpus_gather_bench 64 8192 512 64

#[cfg(feature = "cuda")]
fn main() -> candle::Result<()> {
    use candle::Device;
    use candle_transformers::models::latent_moe::select_bench::{
        run_corpus_gather_kernels, SelectCfg,
    };

    let a: Vec<String> = std::env::args().collect();
    let parse = |i: usize, d: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let cfg = SelectCfg {
        sessions: parse(1, 64),
        entries: parse(2, 8192),
        top_k: parse(3, 512),
        warmup: 20,
        iters: 200,
        seed: 0x5E1E_C7ED,
    };
    let iters = parse(4, 200);

    let t = std::time::Instant::now();
    let dev = Device::new_cuda(0)?;
    eprintln!("[bench] cuda init {:.2}s", t.elapsed().as_secs_f64());
    run_corpus_gather_kernels(&dev, cfg, iters)?;
    eprintln!("[bench] total {:.2}s", t.elapsed().as_secs_f64());
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("corpus_gather_bench requires the `cuda` feature");
}
