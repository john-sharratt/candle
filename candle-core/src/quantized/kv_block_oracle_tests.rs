//! The KV block oracles: every block format a sealed KV band can take must
//! mean the same thing on the host as it does to the kernels.
//!
//! The kernels read a band through `dequant_element_inline` and the palette
//! seal writes it with `p4c_encode_quant_block`; the host reads and writes the
//! same bytes through each format's `GgmlType` codec (`read_contiguous`, the
//! band codec, the CPU fallbacks). Two implementations of one format drift
//! silently — a host decoder reading FP8 centroids where the kernels store
//! INT8 ones returns plausible floats, not an error — so both directions are
//! pinned here:
//!
//! - **decode**: blocks the GPU encoder produced, decoded by the kernels'
//!   element path and by the host codec, must agree bit for bit;
//! - **encode**: the host codec's bytes must equal the GPU encoder's, block for
//!   block.
//!
//! Q0_V is calibrated per side and has its own oracles
//! (`q0_v_decode_oracle_tests`, `q0_v_encode_oracle_tests`).

use super::*;
use crate::quantized::ggml_file::qtensor_from_ggml;
use crate::quantized::QTensor;
use crate::{Device, Tensor};
use candle_kernels::simple::quantized::{run_kv_decode_oracle, run_kv_encode_oracle};
use cudarc::driver::DevicePtr;
use std::ffi::c_void;

const ELEMS: usize = 32;

/// Every block format a KV band can be sealed in, Q0_V aside.
const KV_FORMATS: [GgmlDType; 21] = [
    GgmlDType::R16,
    GgmlDType::Q4_0,
    GgmlDType::Q4_1,
    GgmlDType::Q5_0,
    GgmlDType::Q5_1,
    GgmlDType::Q8_0,
    GgmlDType::Q8_1,
    GgmlDType::Q4_KS,
    GgmlDType::Q8_KS,
    GgmlDType::Q3_0,
    GgmlDType::Q3_1,
    GgmlDType::Q2_0,
    GgmlDType::Q2_1,
    GgmlDType::Q2_A,
    GgmlDType::Q2_S,
    GgmlDType::Q1_S,
    GgmlDType::Q0,
    GgmlDType::Q1_A,
    GgmlDType::Q0_X,
    GgmlDType::Q0_M2,
    GgmlDType::Q0_M4,
];

/// xorshift64*: a deterministic corpus without a dependency.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// Block-major f32 blocks: uniform and bell-shaped over four decades of
/// amplitude with an offset, an attention-sink block (four large leading
/// tokens), flat blocks and single spikes.
fn corpus() -> Vec<f32> {
    let mut rng = Rng(0x51A7_C0DE_D00D_F00D);
    let mut out = Vec::new();
    for i in 0..2_048 {
        let amp = [1.0f32, 0.3, 0.05, 0.004][i % 4];
        let offset = rng.unit() * 0.5 * amp;
        let bell = i % 2 == 1;
        for _ in 0..ELEMS {
            let x = if bell {
                (rng.unit() + rng.unit() + rng.unit()) / 3.0
            } else {
                rng.unit()
            };
            out.push(x * amp + offset);
        }
    }
    for _ in 0..64 {
        for e in 0..ELEMS {
            let x = rng.unit() * 0.05;
            out.push(if e < 4 { x * 60.0 } else { x });
        }
    }
    for i in 0..32 {
        let v = (i as f32 - 16.0) / 16.0;
        out.extend(std::iter::repeat_n(v, ELEMS));
    }
    for pos in 0..ELEMS {
        for sign in [1.0f32, -1.0] {
            out.extend((0..ELEMS).map(|e| if e == pos { sign * 0.9 } else { 0.01 }));
        }
    }
    out
}

fn arena_code(dtype: GgmlDType) -> i32 {
    ggml_dtype_to_arena_fmt_code(dtype).expect("a KV format has an arena code") as i32
}

/// `src` encoded on the device by the palette seal's block encoder.
fn encode_on_device(dtype: GgmlDType, src: &[f32], is_k: bool) -> Result<Vec<u8>> {
    let dev = CudaDevice::new(0)?;
    let n = src.len() / ELEMS;
    let bb = dtype.type_size();
    let src_d = dev.memcpy_stod(src)?;
    let dst_d = unsafe { dev.alloc::<u8>(n * bb)? };
    {
        let stream = dev.cuda_stream();
        let (s, _gs) = src_d.device_ptr(&stream);
        let (d, _gd) = dst_d.device_ptr(&stream);
        // The oracle launches on the legacy default stream: order it after the
        // upload, and the download after it, with device-wide fences.
        dev.synchronize()?;
        unsafe {
            run_kv_encode_oracle(
                s as *const c_void,
                d as *mut c_void,
                n as i32,
                bb as i32,
                arena_code(dtype),
                is_k as i32,
                std::ptr::null_mut(),
            )
        };
        dev.synchronize()?;
    }
    dev.memcpy_dtov(&dst_d)
}

/// `bytes` decoded by the kernels' element path.
fn decode_on_device(dtype: GgmlDType, bytes: &[u8], is_k: bool) -> Result<Vec<f32>> {
    let dev = CudaDevice::new(0)?;
    let bb = dtype.type_size();
    let n = bytes.len() / bb;
    let src_d = dev.memcpy_stod(bytes)?;
    let dst_d = unsafe { dev.alloc::<f32>(n * ELEMS)? };
    {
        let stream = dev.cuda_stream();
        let (s, _gs) = src_d.device_ptr(&stream);
        let (d, _gd) = dst_d.device_ptr(&stream);
        dev.synchronize()?;
        unsafe {
            run_kv_decode_oracle(
                s as *const c_void,
                d as *mut c_void,
                n as i32,
                bb as i32,
                arena_code(dtype),
                is_k as i32,
                std::ptr::null_mut(),
            )
        };
        dev.synchronize()?;
    }
    dev.memcpy_dtov(&dst_d)
}

fn decode_on_host(dtype: GgmlDType, bytes: &[u8]) -> Result<Vec<f32>> {
    let n = bytes.len() / dtype.type_size() * ELEMS;
    qtensor_from_ggml(dtype, bytes, vec![n], &Device::Cpu)?
        .dequantize(&Device::Cpu)?
        .to_vec1::<f32>()
}

fn encode_on_host(dtype: GgmlDType, src: &[f32]) -> Result<Vec<u8>> {
    let t = Tensor::from_slice(src, src.len(), &Device::Cpu)?;
    Ok(QTensor::quantize(&t, dtype)?.data()?.into_owned())
}

#[test]
fn kv_block_decode_matches_the_kernels_bit_exact() -> Result<()> {
    let src = corpus();
    let mut failures = Vec::new();
    for dtype in KV_FORMATS {
        for is_k in [true, false] {
            let bytes = encode_on_device(dtype, &src, is_k)?;
            let gpu = decode_on_device(dtype, &bytes, is_k)?;
            let host = decode_on_host(dtype, &bytes)?;
            let bad: Vec<usize> = (0..gpu.len())
                .filter(|&i| gpu[i].to_bits() != host[i].to_bits())
                .collect();
            if let Some(&i) = bad.first() {
                failures.push(format!(
                    "{dtype:?} {}: {} of {} elements differ; first block {} element {}: \
                     kernels {} host {}",
                    if is_k { "K" } else { "V" },
                    bad.len(),
                    gpu.len(),
                    i / ELEMS,
                    i % ELEMS,
                    gpu[i],
                    host[i],
                ));
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
    Ok(())
}

#[test]
fn kv_block_encode_matches_the_seal_byte_exact() -> Result<()> {
    let src = corpus();
    let mut failures = Vec::new();
    for dtype in KV_FORMATS {
        let bb = dtype.type_size();
        let mut gpu = encode_on_device(dtype, &src, true)?;
        let mut host = encode_on_host(dtype, &src)?;
        if dtype == GgmlDType::Q8_1 {
            // Bytes 2..4 are `s`, which no KV decoder reads: the seal stores
            // Σx, the host `d · Σq` for the Q4_1/Q5_1 vec-dot. Everything the
            // band decodes from must match.
            for b in 0..src.len() / ELEMS {
                gpu[b * bb + 2..b * bb + 4].fill(0);
                host[b * bb + 2..b * bb + 4].fill(0);
            }
        }
        let bad: Vec<usize> = (0..src.len() / ELEMS)
            .filter(|&b| gpu[b * bb..(b + 1) * bb] != host[b * bb..(b + 1) * bb])
            .collect();
        if let Some(&b) = bad.first() {
            failures.push(format!(
                "{dtype:?}: {} of {} blocks differ; first block {b}: seal {:02x?} host {:02x?}",
                bad.len(),
                src.len() / ELEMS,
                &gpu[b * bb..(b + 1) * bb],
                &host[b * bb..(b + 1) * bb],
            ));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
    Ok(())
}
