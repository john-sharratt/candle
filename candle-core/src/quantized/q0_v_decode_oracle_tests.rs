//! The Q0_V decode oracle: every production GPU read path that reaches a Q0_V
//! block must reproduce the reference decoder, bit for bit, on both sides and
//! at any palette scale.
//!
//! A Q0_V block is one 16-bit code, so there are exactly 65,536 blocks and the
//! test decodes all of them — every element of every code — through the four
//! paths `run_q0_v_decode_oracle` exposes: the ArenaAccessor element dispatch,
//! the INT8 prefill element decoder, the INT8 tile decoder's four-token quad,
//! and the block header the decode kernels hoist out of their token loops.
//! Each is compared by bit pattern against `k_quants::q0_v_elem_scaled`, which
//! reads the 128-curve reference tables; the GPU reads the compact four-base
//! codebook, so this is also what pins the two codebooks to each other.
//!
//! A read at palette scale `scale` divides by it, and every path folds that
//! division into the block's scale and centroid as a multiply by the
//! correctly rounded reciprocal — so the scales below include ones whose
//! reciprocal is inexact, where a path that divided per element instead would
//! round differently.
//!
//! The side matters: K and V are calibrated separately, and the tile and INT8
//! prefill paths once decoded every K block against the V codebook. Running
//! the K side here is what catches that.

use super::*;
use crate::quantized::k_quants::{q0_v_elem_scaled, BlockQ0V};
use candle_kernels::simple::quantized::run_q0_v_decode_oracle;
use cudarc::driver::DevicePtr;
use std::ffi::c_void;

const CODES: usize = 1 << 16;
const ELEMS: usize = 32;
const PATHS: [&str; 4] = [
    "element dispatch",
    "INT8 prefill element",
    "INT8 tile quad",
    "hoisted header",
];
/// Unit scale, and scales with inexact reciprocals from small to large.
const SCALES: [f32; 4] = [1.0, 0.37, 3.0, 0.0071];

/// Every 16-bit code once, in order, as the blocks' two little-endian bytes.
fn every_code() -> Vec<u8> {
    (0..CODES).flat_map(|c| (c as u16).to_le_bytes()).collect()
}

/// The four GPU paths' decodes of every code under side `is_k` at `scale`.
fn decode_on_device(is_k: bool, scale: f32) -> Result<[Vec<f32>; 4]> {
    let dev = CudaDevice::new(0)?;
    let src = dev.memcpy_stod(&every_code())?;
    let n = CODES * ELEMS;
    let outs = [
        unsafe { dev.alloc::<f32>(n)? },
        unsafe { dev.alloc::<f32>(n)? },
        unsafe { dev.alloc::<f32>(n)? },
        unsafe { dev.alloc::<f32>(n)? },
    ];
    {
        let stream = dev.cuda_stream();
        let (s, _gs) = src.device_ptr(&stream);
        let (d, _gd) = outs[0].device_ptr(&stream);
        let (i, _gi) = outs[1].device_ptr(&stream);
        let (q, _gq) = outs[2].device_ptr(&stream);
        let (h, _gh) = outs[3].device_ptr(&stream);
        // The oracle launches on the legacy default stream: order it after the
        // upload, and the downloads after it, with device-wide fences.
        dev.synchronize()?;
        unsafe {
            run_q0_v_decode_oracle(
                s as *const c_void,
                d as *mut c_void,
                i as *mut c_void,
                q as *mut c_void,
                h as *mut c_void,
                CODES as i32,
                is_k as i32,
                scale,
            );
        }
        dev.synchronize()?;
    }
    Ok([
        dev.memcpy_dtov(&outs[0])?,
        dev.memcpy_dtov(&outs[1])?,
        dev.memcpy_dtov(&outs[2])?,
        dev.memcpy_dtov(&outs[3])?,
    ])
}

/// The reference decode of every code under side `IS_K` at `scale`,
/// element-major per code.
fn decode_reference<const IS_K: bool>(scale: f32) -> Vec<f32> {
    let r = 1.0 / scale;
    let mut out = Vec::with_capacity(CODES * ELEMS);
    for c in 0..CODES {
        let block = BlockQ0V::from_le_bytes((c as u16).to_le_bytes());
        for e in 0..ELEMS {
            out.push(q0_v_elem_scaled::<IS_K>(&block, e, r));
        }
    }
    out
}

fn assert_bit_exact(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mismatches: Vec<usize> = (0..got.len())
        .filter(|&i| got[i].to_bits() != want[i].to_bits())
        .collect();
    if let Some(&first) = mismatches.first() {
        let (code, e) = (first / ELEMS, first % ELEMS);
        panic!(
            "{what}: {} of {} elements differ; first at code {code:#06x} element {e}: \
             got {} ({:#010x}), want {} ({:#010x})",
            mismatches.len(),
            got.len(),
            got[first],
            got[first].to_bits(),
            want[first],
            want[first].to_bits(),
        );
    }
}

fn check_side<const IS_K: bool>(side: &str) -> Result<()> {
    for scale in SCALES {
        let want = decode_reference::<IS_K>(scale);
        let got = decode_on_device(IS_K, scale)?;
        for (path, g) in PATHS.iter().zip(got.iter()) {
            assert_bit_exact(&format!("{side} {path} @ scale {scale}"), g, &want);
        }
    }
    Ok(())
}

#[test]
fn q0_v_k_decode_matches_reference_bit_exact() -> Result<()> {
    check_side::<true>("K")
}

#[test]
fn q0_v_v_decode_matches_reference_bit_exact() -> Result<()> {
    check_side::<false>("V")
}
