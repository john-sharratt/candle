//! Micro harness for the widening gather (`moe_gather_bf16_f32`): the embedding
//! lookup Flash-Next runs once per wave, BF16 table rows out as F32.
//!
//! Times the kernel against the two-pass `index_select` + `to_dtype` it
//! replaced, over the released table (248,320 × 2,560 BF16, 1.27 GB) at the row
//! counts a wave actually gathers — a decode step, a draft block, a verify
//! block, a prompt, and the widest prefill. The figure of merit is effective
//! bandwidth: 2 bytes read and 4 written per element, against the card's DRAM
//! peak. Run it alone, then under `ncu` for occupancy, register and memory
//! throughput. Each row count launches the kernel 210 times, so skipping 860
//! profiles the widest (5,748-row) case past its warmup:
//!
//! ```bash
//! cargo test --release -p candle-core --features cuda --test gather_widen_bench -- --ignored --nocapture
//! ncu --kernel-name moe_gather_bf16_f32 --launch-skip 860 --launch-count 1 \
//!     --section SpeedOfLight --section Occupancy --section LaunchStats --section MemoryWorkloadAnalysis \
//!     target/release/deps/gather_widen_bench-<hash>.exe --ignored --nocapture
//! ```

#![cfg(feature = "cuda")]

use candle_core::quantized::cuda::gather_rows_bf16_to_f32;
use candle_core::{DType, Device, Result, Tensor};
use std::time::Instant;

const VOCAB: usize = 248_320;
const COLS: usize = 2_560;
const ITERS: usize = 200;

fn time<F: FnMut() -> Result<Tensor>>(device: &Device, mut f: F) -> Result<f64> {
    for _ in 0..10 {
        f()?;
    }
    device.synchronize()?;
    let t = Instant::now();
    for _ in 0..ITERS {
        f()?;
    }
    device.synchronize()?;
    Ok(t.elapsed().as_secs_f64() / ITERS as f64)
}

#[test]
#[ignore = "micro benchmark: allocates a 1.27 GB table on the GPU; run alone"]
fn widening_gather_bandwidth() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let table = Tensor::zeros((VOCAB, COLS), DType::BF16, &device)?;
    println!(
        "{:>6} {:>12} {:>10} {:>12} {:>10} {:>8}",
        "rows", "fused µs", "GB/s", "two-pass µs", "GB/s", "speedup"
    );
    for rows in [1usize, 8, 56, 713, 5_748] {
        let ids: Vec<u32> = (0..rows)
            .map(|i| ((i * 2_654_435_761usize) % VOCAB) as u32)
            .collect();
        let ids = Tensor::from_vec(ids, (rows,), &device)?;
        let bytes = (rows * COLS * (2 + 4)) as f64;
        let fused = time(&device, || gather_rows_bf16_to_f32(&table, &ids))?;
        let two_pass = time(&device, || {
            table.index_select(&ids, 0)?.to_dtype(DType::F32)
        })?;
        println!(
            "{rows:>6} {:>12.2} {:>10.1} {:>12.2} {:>10.1} {:>7.2}×",
            fused * 1e6,
            bytes / fused / 1e9,
            two_pass * 1e6,
            bytes / two_pass / 1e9,
            two_pass / fused
        );
    }
    Ok(())
}
