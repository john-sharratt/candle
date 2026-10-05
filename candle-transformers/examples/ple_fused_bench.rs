//! Individual-kernel microbench + `ncu` target for the fused PLE launches
//! (`ple_gate_kernel`, `ple_conv_kernel`, `ple_history_kernel`) at
//! Qwen3.8-Flash-Next's geometry: 4 residual streams over `n_embd` 2560.
//!
//! Gates fused against eager **on the device** before timing, across decode
//! rows, segments shorter and longer than the conv history, and an odd width
//! that takes the scalar path (see `qwen4exp::ple_bench`), and reports the timed
//! working set against the card's 96 MiB L2.
//!
//! Usage:
//!   cargo run -p candle-transformers --example ple_fused_bench \
//!       --features cuda --release -- [tokens] [spans] [iters]
//!
//! Profile one kernel (the gate's eager launches are excluded by name):
//!   ncu -k ple_gate_kernel --launch-skip 10 --launch-count 3 --set full \
//!       target/release/examples/ple_fused_bench 2048 8 20
//!
//! On Windows the `.bat` wrapper hands `-k` to `cmd`, so profile one kernel
//! per invocation rather than with a regex alternation.

#[cfg(feature = "cuda")]
fn main() -> candle::Result<()> {
    use candle::Device;
    use candle_transformers::models::qwen4exp::ple_bench::{run_ple_kernels, PleBenchCfg};

    let a: Vec<String> = std::env::args().collect();
    let parse = |i: usize, d: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    // 2048 rows puts the gate's pass at ~340 MiB, past the 96 MiB L2.
    let mut cfg = PleBenchCfg::qwen4exp(parse(1, 2048), parse(2, 8));
    cfg.iters = parse(3, 50);

    let t = std::time::Instant::now();
    let dev = Device::new_cuda(0)?;
    eprintln!("[bench] cuda init {:.2}s", t.elapsed().as_secs_f64());
    run_ple_kernels(&dev, cfg)?;
    eprintln!("[bench] total {:.2}s", t.elapsed().as_secs_f64());
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("ple_fused_bench requires the `cuda` feature");
}
