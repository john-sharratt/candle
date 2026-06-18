//! Baseline for the quantized expert matmul (`QMatMul`).
//!
//! The MoE experts spend the bulk of decode time in three quantized matmuls
//! (gate / up / down).  This module captures a **single real expert projection
//! from Qwen3-30B-A3B** — layer 0, expert 0 `ffn_gate_exps` (`[intermediate,
//! hidden] = [768, 2048]`, Q4_K) — straight out of the GGUF onto disk, then
//! reconstructs it as a standalone `QMatMul` (GEMX K/128 repacked, exactly as the
//! pipeline does) so the kernel can be:
//!
//! 1. **validated** — its output matches a full-precision `dequantize → f32
//!    matmul` of the *same* quantized weights, within accumulation tolerance, and
//! 2. **benchmarked** — timed across batch sizes to establish the baseline any
//!    expert-matmul optimisation must beat.
//!
//! ```bash
//! # one-time capture (reads the cached GGUF, writes the fixture to disk):
//! cargo test --release --features cuda -p candle-transformers \
//!   expert_lre::matmul_baseline::extract_expert_weight_fixture -- --ignored --nocapture
//! # correctness (asserts against the real weights):
//! cargo test --release --features cuda -p candle-transformers \
//!   expert_lre::matmul_baseline::expert_matmul_matches_dequant_reference -- --nocapture
//! # baseline benchmark (prints the table):
//! cargo test --release --features cuda -p candle-transformers \
//!   expert_lre::matmul_baseline::expert_matmul_baseline_bench -- --ignored --nocapture
//! ```

use crate::models::quantized_matmul::QMatMul;
use candle::quantized::{gguf_file, GgmlDType, QTensor};
use candle::{Device, Module, Result, Tensor};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

const REPO: &str = "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF";
const GGUF: &str = "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf";
/// Layer-0 gate experts (3D merged tensor `[n_expert, intermediate, hidden]`).
const TENSOR: &str = "blk.0.ffn_gate_exps.weight";
/// Which expert within the layer to capture.
const EXPERT_IDX: usize = 0;

/// On-disk capture of one real expert projection (raw GGML quantized bytes).
#[derive(Serialize, Deserialize)]
struct ExpertProj {
    /// `GgmlDType` discriminant (the type is `#[repr(u32)]`).
    dtype: u32,
    /// `[out, in]` — e.g. `[768, 2048]` for gate/up.
    shape: Vec<usize>,
    /// Raw GGML quantized bytes for this single expert.
    ggml: Vec<u8>,
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src/models/expert_lre/fixtures/qwen3_gate_expert.bin")
}

/// Reconstruct a `GgmlDType` from its discriminant (candle's `from_u32` is
/// crate-private).  Covers the formats GGUF MoE experts actually use.
fn dtype_from_u32(u: u32) -> Result<GgmlDType> {
    Ok(match u {
        7 => GgmlDType::Q8_0,
        11 => GgmlDType::Q6_K,
        14 => GgmlDType::Q5_K,
        15 => GgmlDType::Q4_0,
        17 => GgmlDType::Q4_K,
        other => candle::bail!("unsupported expert dtype code {other} in fixture"),
    })
}

/// Capture one real expert projection from the Qwen3-30B-A3B GGUF onto disk.
///
/// Reads only the GGUF header + one expert's byte-slice — no model / VRAM load.
#[test]
#[ignore = "reads the cached Qwen3-30B-A3B GGUF and writes the expert-weight fixture"]
fn extract_expert_weight_fixture() -> Result<()> {
    use crate::models::batch_test::test_helpers::hf_get;
    use memmap2::MmapOptions;
    use std::io::Cursor;

    let path = hf_get(REPO, hf_hub::RepoType::Model, "main", GGUF)
        .map_err(|e| candle::Error::Msg(format!("hf_get GGUF: {e}")))?;
    let file = std::fs::File::open(&path)?;
    let mmap = unsafe { MmapOptions::new().map(&file).map_err(candle::Error::wrap)? };
    let ct = gguf_file::Content::read(&mut Cursor::new(&mmap[..]))?;

    let info = ct
        .tensor_infos
        .get(TENSOR)
        .ok_or_else(|| candle::Error::Msg(format!("tensor {TENSOR} not found in GGUF")))?;
    let dims = info.shape.dims();
    if dims.len() != 3 {
        candle::bail!("{TENSOR} is not a 3D merged expert tensor: {dims:?}");
    }
    let (n_expert, out, inn) = (dims[0], dims[1], dims[2]);
    let dtype = info.ggml_dtype;
    let expert_bytes = (out * inn) / dtype.block_size() * dtype.type_size();
    let base = (ct.tensor_data_offset + info.offset) as usize + EXPERT_IDX * expert_bytes;
    let ggml = mmap[base..base + expert_bytes].to_vec();

    let fx = ExpertProj {
        dtype: dtype as u32,
        shape: vec![out, inn],
        ggml,
    };
    let bytes = bincode::serialize(&fx).map_err(|e| candle::Error::Msg(format!("bincode: {e}")))?;
    let out_path = fixture_path();
    std::fs::create_dir_all(out_path.parent().unwrap())?;
    std::fs::write(&out_path, &bytes)?;

    println!(
        "captured expert {EXPERT_IDX} of {TENSOR}: [{out}x{inn}] {dtype:?} \
         (1 of {n_expert} experts), {} KB → {}",
        bytes.len() / 1024,
        out_path.display(),
    );
    Ok(())
}

/// Load the captured expert and reconstruct it the way the pipeline does:
/// GEMX K/128 repack on the GPU.  Returns `(qmm, dequantised_weight, weight_bytes)`
/// or `None` if the fixture has not been captured yet.
fn load_expert(dev: &Device) -> Result<Option<(QMatMul, Tensor, usize)>> {
    let path = fixture_path();
    let Ok(bytes) = std::fs::read(&path) else {
        println!(
            "[expert matmul] no fixture at {} — run extract_expert_weight_fixture first",
            path.display()
        );
        return Ok(None);
    };
    let fx: ExpertProj =
        bincode::deserialize(&bytes).map_err(|e| candle::Error::Msg(format!("bincode: {e}")))?;
    let dtype = dtype_from_u32(fx.dtype)?;
    let (n, k) = (fx.shape[0], fx.shape[1]);
    let weight_bytes = fx.ggml.len();

    // Reference weights: dequantise the raw GGML on CPU (unambiguous), to device.
    let cpu_qt =
        candle::quantized::ggml_file::qtensor_from_ggml(dtype, &fx.ggml, vec![n, k], &Device::Cpu)?;
    let w_deq = cpu_qt.dequantize(&Device::Cpu)?.to_device(dev)?; // [n, k]

    // Production reconstruction: repack to the GEMX K/128 format on the GPU.
    let cuda_dev = match dev {
        Device::Cuda(d) => d,
        _ => candle::bail!("expert matmul baseline requires a CUDA device"),
    };
    let repacked = candle::quantized::repack_to_host(cuda_dev, &fx.ggml, n, k, dtype)?;
    let storage = candle::quantized::load_repacked(cuda_dev, &repacked, dtype)?;
    let qt = QTensor::new(storage, vec![n, k])?;
    let qmm = QMatMul::from_qtensor_repacked(qt)?;
    Ok(Some((qmm, w_deq, weight_bytes)))
}

/// Relative L2 error `||a - b|| / ||b||`.
fn rel_l2(a: &Tensor, b: &Tensor) -> Result<f32> {
    let num = (a - b)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
    let den = b.sqr()?.sum_all()?.to_scalar::<f32>()?.max(1e-12);
    Ok((num / den).sqrt())
}

#[test]
fn expert_matmul_matches_dequant_reference() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let Some((qmm, w_deq, _)) = load_expert(&dev)? else {
        return Ok(()); // fixture not captured — skip
    };
    let (_n, k) = w_deq.dims2()?;
    let wt = w_deq.t()?.contiguous()?; // [k, n]

    for &m in &[1usize, 8, 64, 256] {
        let x = Tensor::randn(0f32, 1f32, (m, k), &dev)?;
        let out = qmm.forward(&x)?; // [m, n] — quantized GEMX kernel
        let reference = x.matmul(&wt)?; // [m, n] — f32 dequant matmul
        let err = rel_l2(&out, &reference)?;
        assert!(
            err < 0.03,
            "M={m}: expert matmul diverged from dequant reference (rel L2 = {err:.4})"
        );
    }
    Ok(())
}

/// Baseline benchmark — establishes the performance bar for expert-matmul work.
#[test]
#[ignore = "GPU benchmark; run explicitly with --ignored --nocapture"]
fn expert_matmul_baseline_bench() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let Some((qmm, w_deq, weight_bytes)) = load_expert(&dev)? else {
        return Ok(()); // fixture not captured — skip
    };
    let (n, k) = w_deq.dims2()?;
    let weight_bytes = weight_bytes as f64;

    println!("\n=== Expert matmul baseline — Qwen3-30B-A3B gate_proj [{n}x{k}] (captured) ===",);
    println!("weight = {:.0} KB\n", weight_bytes / 1024.0);

    for in_dtype in [candle::DType::F32, candle::DType::BF16] {
        println!("-- input dtype {in_dtype:?} --");
        println!(
            "{:>6} {:>11} {:>13} {:>12}",
            "tokens", "latency(ms)", "GFLOP/s", "weight GB/s"
        );
        for &m in &[1usize, 8, 32, 64, 128, 256, 512] {
            let x = Tensor::randn(0f32, 1f32, (m, k), &dev)?.to_dtype(in_dtype)?;

            for _ in 0..20 {
                let _ = qmm.forward(&x)?;
            }
            dev.synchronize()?;

            // Min over several timed batches — filters transient GPU stalls
            // (clock changes, scheduler) for a reproducible best-case baseline.
            let iters = 100;
            let mut dt = f64::MAX;
            for _ in 0..5 {
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    let _ = qmm.forward(&x)?;
                }
                dev.synchronize()?;
                dt = dt.min(t0.elapsed().as_secs_f64() / iters as f64);
            }

            let flops = 2.0 * m as f64 * n as f64 * k as f64;
            println!(
                "{:>6} {:>11.4} {:>13.1} {:>12.1}",
                m,
                dt * 1e3,
                flops / dt / 1e9,
                weight_bytes / dt / 1e9,
            );
        }
        println!();
    }
    Ok(())
}
