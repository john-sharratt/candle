//! Micro harness for the W4A16 → Q4_KO import kernels (`decode_packed`,
//! `pack_q4_ko`, the losslessness verify) — §0.4 rule 4 for the converter.
//!
//! Pure CPU (rayon across experts, exactly as the converter runs); every run
//! carries a bit-exact correctness gate through the untouched `dequant_ko`.
//! Defaults use the real routed-expert geometry and a working set past any
//! CPU cache.
//!
//! Usage:
//!   cargo run -p candle-transformers --example w4a16_convert_bench \
//!       --features cuda --release -- [experts] [nrows] [ncols] [iters]

#[cfg(feature = "cuda")]
fn main() -> candle::Result<()> {
    use candle_transformers::models::qwen4exp::convert_bench::{
        run_convert_bench, ConvertBenchCfg,
    };

    let a: Vec<String> = std::env::args().collect();
    let parse = |i: usize, d: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let mut cfg = ConvertBenchCfg::default();
    cfg.experts = parse(1, cfg.experts);
    cfg.nrows = parse(2, cfg.nrows);
    cfg.ncols = parse(3, cfg.ncols);
    cfg.iters = parse(4, cfg.iters);

    let t = std::time::Instant::now();
    run_convert_bench(cfg)?;
    eprintln!("[bench] total {:.2}s", t.elapsed().as_secs_f64());
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!(
        "w4a16_convert_bench requires the `cuda` feature (the qwen4exp module is gated on it)"
    );
}
