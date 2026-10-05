//! The Q0_V encode oracle: the production GPU encoder must produce the
//! reference encoder's bytes, block for block, on both sides.
//!
//! The GPU encoder searches the 128 curves as 64 (base, phase) correlations
//! spread across the warp's lanes; the reference `k_quants::encode_block_q0_v`
//! scores every curve of the 128-curve table in turn. The two meet only if
//! the codebook really is four signed, rotated base curves and every rounding
//! step agrees, so equal bytes over a varied corpus pin both.
//!
//! The corpus covers what the search must get right beyond typical data:
//! blocks that are exact codebook reconstructions (the answer is known), flat
//! blocks (every curve ties, so the tie rule decides), single spikes, and
//! magnitudes from 1e-4 to 1.

use super::*;
use crate::quantized::k_quants::{encode_block_q0_v, q0_v_elem, BlockQ0V};
use candle_kernels::simple::quantized::run_q0_v_encode_oracle;
use cudarc::driver::DevicePtr;
use std::ffi::c_void;

const ELEMS: usize = 32;

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

/// The encode corpus for side `IS_K`, block-major.
fn corpus<const IS_K: bool>() -> Vec<f32> {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ IS_K as u64);
    let mut out = Vec::new();
    // Random blocks: uniform, and a sum-of-uniforms bell, over five decades
    // of amplitude, with an offset so the centroid search is exercised.
    for i in 0..12_288 {
        let amp = [1.0f32, 0.3, 0.05, 0.01, 1e-4][i % 5];
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
    // Exact reconstructions of random codes.
    for _ in 0..4_096 {
        let code = (rng.next_u64() & 0xFFFF) as u16;
        let block = BlockQ0V::from_le_bytes(code.to_le_bytes());
        out.extend((0..ELEMS).map(|e| q0_v_elem::<IS_K>(&block, e)));
    }
    // Flat blocks, including zero: every curve scores the same.
    for i in 0..64 {
        let v = (i as f32 - 32.0) / 32.0;
        out.extend(std::iter::repeat_n(v, ELEMS));
    }
    // One spike on a flat floor, at every position and both signs.
    for pos in 0..ELEMS {
        for sign in [1.0f32, -1.0] {
            out.extend((0..ELEMS).map(|e| if e == pos { sign * 0.9 } else { 0.01 }));
        }
    }
    out
}

fn encode_on_device(src: &[f32], is_k: bool) -> Result<Vec<u8>> {
    let dev = CudaDevice::new(0)?;
    let n = src.len() / ELEMS;
    let src_d = dev.memcpy_stod(src)?;
    let dst_d = unsafe { dev.alloc::<u8>(n * 2)? };
    {
        let stream = dev.cuda_stream();
        let (s, _gs) = src_d.device_ptr(&stream);
        let (d, _gd) = dst_d.device_ptr(&stream);
        // The oracle launches on the legacy default stream: order it after the
        // upload, and the download after it, with device-wide fences.
        dev.synchronize()?;
        unsafe {
            run_q0_v_encode_oracle(
                s as *const c_void,
                d as *mut c_void,
                n as i32,
                is_k as i32,
                std::ptr::null_mut(),
            )
        };
        dev.synchronize()?;
    }
    dev.memcpy_dtov(&dst_d)
}

fn check_side<const IS_K: bool>(side: &str) -> Result<()> {
    let src = corpus::<IS_K>();
    let got = encode_on_device(&src, IS_K)?;
    let mismatches: Vec<usize> = src
        .chunks_exact(ELEMS)
        .enumerate()
        .filter(|(i, block)| {
            let want = encode_block_q0_v::<IS_K>(block).to_le_bytes();
            got[2 * i..2 * i + 2] != want
        })
        .map(|(i, _)| i)
        .collect();
    if let Some(&first) = mismatches.first() {
        let block = &src[first * ELEMS..(first + 1) * ELEMS];
        panic!(
            "{side}: {} of {} blocks differ; first is block {first}: got {:02x?}, want {:02x?}, input {block:?}",
            mismatches.len(),
            src.len() / ELEMS,
            &got[2 * first..2 * first + 2],
            encode_block_q0_v::<IS_K>(block).to_le_bytes(),
        );
    }
    Ok(())
}

#[test]
fn q0_v_k_encode_matches_reference() -> Result<()> {
    check_side::<true>("K")
}

#[test]
fn q0_v_v_encode_matches_reference() -> Result<()> {
    check_side::<false>("V")
}
