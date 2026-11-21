//! Scalar KV-path quantizer tests, pinned to the bytes each format defines.
//!
//! The cache write path quantizes through single-block functions that differ
//! from the multi-block reference, so each test drives that path on one
//! 32-element block and checks the stored bytes. Each test was written to fail
//! before its fix. They cover a Q4_0 and Q5_0 scale that lost the sign of the
//! dominant element, a Q1_S scale taken from the amax rather than the mean
//! magnitude, and INT8 scale bytes that must round to the value stored.

use super::*;

// KV-path quantization scalar-function bug regression tests.
//
// The KV cache write path calls scalar single-block functions (quantize_block_*)
// via transpose_batch.cuh, which are distinct from the multi-block reference
// path (quantize_blocks_*) exercised by the existing tests.  These tests drive
// that scalar path directly through `quantize_transposed_batched_to_dtype` with
// geometry (n_head=1, chunk_size=32, head_dim=1) — a no-op transpose that
// processes exactly one 32-element block — and check the raw quantized bytes.
//
// Each test is written to FAIL before the corresponding kernel fix is applied
// and to PASS afterwards.

pub(super) fn kv_path_quantize_one_block(
    src_data: &[f32],
    dtype: GgmlDType,
    dev: &CudaDevice,
) -> Result<Vec<u8>> {
    assert_eq!(
        src_data.len(),
        32,
        "KV-path tests require exactly 32 elements"
    );
    assert_eq!(dtype.block_size(), 32, "dtype must have 32-element blocks");

    let quant_size = dtype.type_size();
    let src_gpu = dev.memcpy_stod(src_data)?;
    let mut kv_buf = unsafe { dev.alloc::<u8>(quant_size)? };

    {
        let stream = dev.cuda_stream();
        let (src_ptr, _sg) = src_gpu.device_ptr(&stream);
        let (dst_ptr, _dg) = kv_buf.device_ptr_mut(&stream);
        unsafe {
            quantize_transposed_batched_to_dtype(
                src_ptr as *const f32,
                dst_ptr as *mut u8,
                None,
                None,
                1,  // num_chunks
                1,  // n_head
                32, // chunk_size (= one full block)
                1,  // head_dim
                dtype,
                dev,
            )?;
        }
    }
    dev.synchronize()?;
    Ok(dev.memcpy_dtov(&kv_buf.slice(..))?)
}

// Q4_0 scalar: d = -amax/8  (always negative)
// Fix:         d = max_val/-8  (preserves sign of dominant element)
//
// Input: v[0]=-2.0 (dominant negative), v[1..31]=0.1.
// Bug:  d = -0.25  → q[0]=15 (saturated), dequant=-1.75, error=0.25
// Fix:  d = +0.25  → q[0]=0,              dequant=-2.0,  error=0
//
// Detected via the f16 scale byte: 0x34 = +0.25 (correct), 0xB4 = -0.25 (bug).
#[test]
pub(super) fn kv_path_sign_bug_q4_0() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let mut src = vec![0.1f32; 32];
    src[0] = -2.0;

    let bytes = kv_path_quantize_one_block(&src, GgmlDType::Q4_0, &dev)?;

    // Q4_0 block layout: [d: f16 LE (2 bytes), qs: [u8;16]]
    // +0.25 f16 = 0x3400 → [0x00, 0x34]; -0.25 f16 = 0xB400 → [0x00, 0xB4]
    assert_eq!(
        bytes[1], 0x34,
        "Q4_0 KV scalar sign bug: expected d=+0.25 (byte 0x34), got 0x{:02X} \
         (negative scale; bug is d=-amax/8 instead of d=max_val/-8)",
        bytes[1]
    );
    Ok(())
}

// Q5_0 scalar: d = -amax/16  (always negative).
// Fix:         d = max_val/-16.
//
// Same sign-preservation defect as Q4_0 but with 5-bit offset=16.
// Detected via the f16 scale byte: 0x30 = +0.125, 0xB0 = -0.125.
#[test]
pub(super) fn kv_path_sign_bug_q5_0() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let mut src = vec![0.1f32; 32];
    src[0] = -2.0;

    let bytes = kv_path_quantize_one_block(&src, GgmlDType::Q5_0, &dev)?;

    // Q5_0 block layout: [d: f16 LE (2 bytes), qh: u32, ql: [u8;16]]
    // +0.125 f16 = 0x3000 → [0x00, 0x30]; -0.125 f16 = 0xB000 → [0x00, 0xB0]
    assert_eq!(
        bytes[1], 0x30,
        "Q5_0 KV scalar sign bug: expected d=+0.125 (byte 0x30), got 0x{:02X} \
         (negative scale; bug is d=-amax/16 instead of d=max_val/-16)",
        bytes[1]
    );
    Ok(())
}

// Q1_S scalar/vec/multi-block: the INT8 scale is the block's mean |x|, not
// its amax.
//
// Input: v[0]=1.0 (outlier), v[1..31]=0.1.
//   mean_abs = (1.0 + 31*0.1) / 32 = 0.128125 → rint(· 127) = 16
//   amax     = 1.0                              → rint(· 127) = 127
//
// With amax-scale: all elements reconstruct as ±1.0.  v[1..31] error = 0.9 each,
// MSE ≈ 0.785.  With mean-scale: v[1..31] error = 0.025 each, MSE ≈ 0.025.
// Detected via the stored scale byte (bytes[0]).
#[test]
pub(super) fn kv_path_scale_q1_s() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let mut src = vec![0.1f32; 32];
    src[0] = 1.0;

    let bytes = kv_path_quantize_one_block(&src, GgmlDType::Q1_S, &dev)?;

    // Q1_S block layout: [scale: i8, qs: [u8;4]]
    // mean_abs = (31*0.1 + 1.0)/32 = 4.1/32 = 0.128125
    // INT8 scale = round(0.128125 * 127) = round(16.27) = 16
    // Bug (amax): round(1.0 * 127) = 127
    let scale = bytes[0] as i8;
    assert_eq!(
        scale, 16,
        "Q1_S INT8 scale: expected round(mean_abs*127)=16, got {} (bug would store amax→127)",
        scale
    );
    Ok(())
}

// Q2_S scalar: encode stores INT8(round(d*127)) and quantizes using d_int8=scale/127.
//
// Input: v[0]=0.9 (sets amax), v[1..31]=0.61.
//   d = 0.9/1.5 = 0.6
//   INT8 scale = round(0.6 * 127) = round(76.2) = 76
//
// Detected via the stored scale byte (bytes[0]).
#[test]
pub(super) fn kv_path_int8_roundtrip_q2_s() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let mut src = vec![0.61f32; 32];
    src[0] = 0.9;

    let bytes = kv_path_quantize_one_block(&src, GgmlDType::Q2_S, &dev)?;

    // Q2_S block: [scale: i8, qs: [u8;8]]
    // amax=0.9, d=0.6, INT8 scale = round(0.6*127) = 76
    let scale = bytes[0] as i8;
    assert_eq!(
        scale, 76,
        "Q2_S INT8 scale: expected round(d*127)=76, got {}",
        scale
    );
    Ok(())
}

// Q2_A scalar: encode stores INT8 scale=round(delta*127) and INT8 bias=round(vmin*127).
//
// Input: v[0]=1.1875 (vmax), v[1]=-0.59375 (vmin), v[2..31]=-0.31.
//   delta = (1.1875-(-0.59375))/3 = 1.78125/3 = 0.59375
//   INT8 scale = round(0.59375 * 127) = round(75.41) = 75
//   INT8 bias  = round(-0.59375 * 127) = round(-75.41) = -75
//
// Detected via the stored scale and bias bytes (bytes[0], bytes[1]).
#[test]
pub(super) fn kv_path_int8_roundtrip_q2_a() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let mut src = vec![-0.31f32; 32];
    src[0] = 1.1875; // sets vmax
    src[1] = -0.59375; // sets vmin

    let bytes = kv_path_quantize_one_block(&src, GgmlDType::Q2_A, &dev)?;

    // Q2_A block: [scale: i8, bias: i8, qs: [u8;8]]
    let scale = bytes[0] as i8;
    let bias = bytes[1] as i8;
    assert_eq!(
        scale, 75,
        "Q2_A INT8 scale: expected round(delta*127)=75, got {}",
        scale
    );
    assert_eq!(
        bias, -75,
        "Q2_A INT8 bias:  expected round(vmin*127)=-75, got {}",
        bias
    );
    Ok(())
}

// Q1_A: 1-bit asymmetric — separate amplitude per sign + 32 sign bits.
// Input: lanes 0-15 = +1.0 (positive), lanes 16-31 = -0.25 (negative).
//   scale_pos = round(mean(positive) * 127) = round(1.0 * 127) = 127
//   scale_neg = round(mean(|negative|) * 127) = round(0.25 * 127) = round(31.75) = 32
//   qmask: bits 0..15 set (positive), bits 16..31 clear (negative)
//          → bytes[2..6] = [0xFF, 0xFF, 0x00, 0x00]
#[test]
pub(super) fn kv_path_q1_a_asymmetric() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let mut src = [-0.25f32; 32];
    for v in &mut src[..16] {
        *v = 1.0;
    }

    let bytes = kv_path_quantize_one_block(&src, GgmlDType::Q1_A, &dev)?;

    assert_eq!(bytes.len(), 6, "Q1_A block must be 6 bytes");
    assert_eq!(
        bytes[0] as i8, 127,
        "Q1_A scale_pos (mean(+) = 1.0): expected 127, got {}",
        bytes[0] as i8
    );
    assert_eq!(
        bytes[1] as i8, 32,
        "Q1_A scale_neg (mean(|-|) = 0.25): expected 32, got {}",
        bytes[1] as i8
    );
    assert_eq!(
        bytes[2], 0xFF,
        "Q1_A qs[0] (lanes 0-7 positive): expected 0xFF, got 0x{:02X}",
        bytes[2]
    );
    assert_eq!(
        bytes[3], 0xFF,
        "Q1_A qs[1] (lanes 8-15 positive): expected 0xFF, got 0x{:02X}",
        bytes[3]
    );
    assert_eq!(
        bytes[4], 0x00,
        "Q1_A qs[2] (lanes 16-23 negative): expected 0x00, got 0x{:02X}",
        bytes[4]
    );
    assert_eq!(
        bytes[5], 0x00,
        "Q1_A qs[3] (lanes 24-31 negative): expected 0x00, got 0x{:02X}",
        bytes[5]
    );
    Ok(())
}

// Q0_X: flat block + one outlier escape.
//
// Input: 31 zeros + one outlier value 1.0 at lane 7.
//   sum = 1.0; mean = 1/32 = 0.03125
//   bulk_anchor = round(0.03125 * 127) = round(3.96875) = 4
//   x_i8: 0 everywhere, except lane 7 = round(1.0 * 127) = 127
//   residuals: -4 elsewhere, +123 at lane 7 → argmax = lane 7
//   delta_raw = round(123 / 32) = 4 → clamped to outlier_delta = 3
//   packed byte 1 = (lane_idx 7 in low 5 bits) | (delta 3 in top 3 bits)
//                 = 0x07 | (0x03 << 5) = 0x67
#[test]
pub(super) fn kv_path_q0_x_outlier() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let mut src = [0.0f32; 32];
    src[7] = 1.0;

    let bytes = kv_path_quantize_one_block(&src, GgmlDType::Q0_X, &dev)?;

    assert_eq!(bytes.len(), 2, "Q0_X block must be 2 bytes");
    assert_eq!(
        bytes[0] as i8, 4,
        "Q0_X bulk_anchor (mean=1/32 → INT8 4): got {}",
        bytes[0] as i8
    );
    assert_eq!(
        bytes[1], 0x67,
        "Q0_X outlier_packed (idx=7, delta=3): expected 0x67, got 0x{:02X}",
        bytes[1]
    );
    Ok(())
}

// Q0_M2: 2-centroid + 8-bit quartet mask.
// Input: 8 quartets alternating [-1.0]*4 and [1.0]*4.
//   Init: c0=min=-1.0, c1=max=1.0.  Lloyd converges immediately.
//   Encoding: c0=-1.0 → INT8 round(-1.0*127)=-127; c1=1.0 → INT8 127.
//   Mask: even quartets (0,2,...) → c0 (bit=0), odd quartets → c1 (bit=1).
//   Expected qmask byte: 0b_10101010 = 0xAA.
// Block layout: [centroid[0], centroid[1], qmask: u8]
#[test]
pub(super) fn kv_path_q0_m2_alternating() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let mut src = [0.0f32; 32];
    for i in 0..8 {
        let base = i * 4;
        let val = if i % 2 == 0 { -1.0 } else { 1.0 };
        src[base..base + 4].fill(val);
    }

    let bytes = kv_path_quantize_one_block(&src, GgmlDType::Q0_M2, &dev)?;

    assert_eq!(bytes.len(), 3, "Q0_M2 block must be 3 bytes");
    // Quartet assignments: [0→c0, 1→c1, 2→c0, 3→c1, 4→c0, 5→c1, 6→c0, 7→c1]
    // qmask bit k = assignment of quartet k: 0b10101010 = 0xAA
    assert_eq!(
        bytes[0] as i8, -127,
        "Q0_M2 c0 (-1.0): expected INT8 -127, got {}",
        bytes[0] as i8
    );
    assert_eq!(
        bytes[1] as i8, 127,
        "Q0_M2 c1 (+1.0): expected INT8 127, got {}",
        bytes[1] as i8
    );
    assert_eq!(
        bytes[2], 0xAA,
        "Q0_M2 mask: expected 0xAA (alternating), got 0x{:02X}",
        bytes[2]
    );
    Ok(())
}

// Q0_M4: 4-centroid + 32-bit pair mask.
// Input: 8 quartets cycling through [-0.75, -0.25, 0.25, 0.75] (2 quartets each value).
//   Init: vmin=-0.75, vmax=0.75, step=0.5 → c[0..3]=-0.75,-0.25,0.25,0.75.
//   Lloyd converges immediately (pair means map cleanly to centroids).
//   INT8 encoding: round(v*127): -0.75→-95, -0.25→-32, 0.25→32, 0.75→95.
//   2-bit assignment per pair (16 pairs of 2): pair k → (k/2) % 4
//     pairs 0,1→0; 2,3→1; 4,5→2; 6,7→3; 8,9→0; 10,11→1; 12,13→2; 14,15→3.
//   qmask u32 LE: bits [15:0] = 11_11_10_10_01_01_00_00 = 0xFA50,
//                 bits [31:16] = same pattern → 0xFA50FA50.
#[test]
pub(super) fn kv_path_q0_m4_four_levels() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let mut src = [0.0f32; 32];
    for i in 0..8 {
        let val = [-0.75f32, -0.25, 0.25, 0.75][i % 4];
        src[i * 4..i * 4 + 4].fill(val);
    }

    let bytes = kv_path_quantize_one_block(&src, GgmlDType::Q0_M4, &dev)?;

    assert_eq!(bytes.len(), 8, "Q0_M4 block must be 8 bytes");
    // INT8: round(v*127) — all within [-127,127] so no clamping
    assert_eq!(
        bytes[0] as i8, -95,
        "Q0_M4 c[0]=-0.75: expected INT8 -95, got {}",
        bytes[0] as i8
    );
    assert_eq!(
        bytes[1] as i8, -32,
        "Q0_M4 c[1]=-0.25: expected INT8 -32, got {}",
        bytes[1] as i8
    );
    assert_eq!(
        bytes[2] as i8, 32,
        "Q0_M4 c[2]=+0.25: expected INT8 32, got {}",
        bytes[2] as i8
    );
    assert_eq!(
        bytes[3] as i8, 95,
        "Q0_M4 c[3]=+0.75: expected INT8 95, got {}",
        bytes[3] as i8
    );
    let qmask = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    assert_eq!(
        qmask, 0xFA50_FA50,
        "Q0_M4 mask: expected 0xFA50FA50, got 0x{:08X}",
        qmask
    );
    Ok(())
}
