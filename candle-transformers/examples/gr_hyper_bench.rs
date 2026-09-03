//! Individual-kernel microbench + `ncu` target for the three Gated-Residual
//! kernels (`gr_norm`, `gr_mix`, `gr_combine`) at Qwen3.8-Flash-Next's
//! geometry: 4 residual streams over `n_embd` 2560.
//!
//! The fusion was accepted on a span timer, which can only say these kernels
//! beat the eager chain they replaced. This says how far they are from the
//! memory floor, which is the question that decides whether there is anything
//! left in them.
//!
//! Gates fused against eager **on the device** before timing (a CPU reference
//! cannot police these — see the module header of `hyper::bench`), and reports
//! the timed working set against the card's 96 MiB L2.
//!
//! Usage:
//!   cargo run -p candle-transformers --example gr_hyper_bench \
//!       --features cuda --release -- [tokens] [iters]
//!
//! Profile one kernel (the setup launches are excluded by name):
//!   ncu -k gr_mix_kernel --launch-count 6 --set full \
//!       target/release/examples/gr_hyper_bench 2048 20
//!
//! The kernel names are `gr_norm_kernel`, `gr_mix_kernel`, `gr_combine_kernel`.
//! Note that `-k` takes a plain substring or `regex:<expr>`; on Windows the
//! `.bat` wrapper hands the argument to `cmd`, so a regex containing `(`, `)`
//! or `|` must be avoided — profile one kernel per invocation instead.

#[cfg(feature = "cuda")]
fn main() -> candle::Result<()> {
    use candle::Device;
    use candle_transformers::models::qwen4exp::hyper::bench::{run_gr_kernels, GrBenchCfg};

    let a: Vec<String> = std::env::args().collect();
    let parse = |i: usize, d: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    // 2048 tokens puts one pass over the wide residual at 160 MiB, past the
    // card's 96 MiB L2; the bench prints which side of that line it landed on.
    let mut cfg = GrBenchCfg::qwen4exp(parse(1, 2048));
    cfg.iters = parse(2, 50);

    let t = std::time::Instant::now();
    let dev = Device::new_cuda(0)?;
    eprintln!("[bench] cuda init {:.2}s", t.elapsed().as_secs_f64());
    run_gr_kernels(&dev, cfg)?;
    eprintln!("[bench] total {:.2}s", t.elapsed().as_secs_f64());
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("gr_hyper_bench requires the `cuda` feature");
}
