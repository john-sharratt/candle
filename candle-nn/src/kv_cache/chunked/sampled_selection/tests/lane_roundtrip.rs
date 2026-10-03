//! The selection search measures a format by quantizing a block and decoding it
//! again. Wherever `lane_roundtrip` has a register path it skips the bytes and
//! computes each lane's reconstruction directly; that is only sound if the two
//! agree bit for bit, so the search still measures exactly what the attention
//! kernel will decode. This drives both paths over the same blocks
//! (`run_select_roundtrip_parity`) and asserts they do.

#![cfg(feature = "cuda")]

use std::ffi::{c_int, c_void};

use candle::cuda_backend::cudarc::driver::{CudaSlice, DevicePtr};

use super::mirror::unit;
use super::{SampleFormat, CHUNK_SIZE};
use crate::kv_cache::chunked::gpu_test_lock::gpu_serial;
use crate::kv_cache::{KvFormat, QuantFormat};

extern "C" {
    fn run_select_roundtrip_parity(
        src: *const c_void,
        recon_bytes: *mut c_void,
        recon_lane: *mut c_void,
        num_blocks: c_int,
        outer: f32,
        fmt: c_int,
        is_k: c_int,
    );
}

/// Every format with a register round trip.
const FORMATS: [QuantFormat; 17] = [
    QuantFormat::Q0,
    QuantFormat::Q0_V,
    QuantFormat::Q0_X,
    QuantFormat::Q0_M2,
    QuantFormat::Q0_M4,
    QuantFormat::Q1_A,
    QuantFormat::Q1_S,
    QuantFormat::Q2_A,
    QuantFormat::Q2_S,
    QuantFormat::Q3_0,
    QuantFormat::Q3_1,
    QuantFormat::Q4_0,
    QuantFormat::Q4_1,
    QuantFormat::Q5_0,
    QuantFormat::Q5_1,
    QuantFormat::Q8_0,
    QuantFormat::Q8_1,
];

/// The outer scales the search multiplies blocks by: identity, and values like
/// the 1/amax and 1/percentile candidates on either side of it.
const OUTERS: [f32; 4] = [1.0, 0.37, 2.9, 127.0 / 91.0];

/// Block `b`: one of nine shapes chosen to hit the encoders' edges — values
/// spread over six decades (sums that round), uniform, flat, a single outlier,
/// two maxima of equal magnitude and opposite sign (the signed-maximum tie
/// rules), all zero, all positive, all negative, and values on rounding
/// midpoints.
fn block(b: usize) -> [f32; CHUNK_SIZE] {
    let seed = b as u64;
    let mut x = [0.0f32; CHUNK_SIZE];
    for (i, v) in x.iter_mut().enumerate() {
        let u = unit(seed, i);
        *v = match b % 9 {
            0 => u.signum() * 10f32.powf(-6.0 * unit(seed ^ 0x55, i).abs()),
            1 => u,
            2 => 0.25 + 1e-4 * u,
            3 => {
                if i == (b / 9) % CHUNK_SIZE {
                    0.9 * u.signum()
                } else {
                    -0.1 + 0.01 * u
                }
            }
            4 => 0.5 * u,
            5 => 0.0,
            6 => u.abs(),
            7 => -u.abs(),
            _ => ((u * 120.0).round() + 0.5) / 127.0,
        };
    }
    if b % 9 == 4 {
        let (i, j) = ((b / 9) % CHUNK_SIZE, (b / 9 + 7) % CHUNK_SIZE);
        x[i] = 0.75;
        x[j] = -0.75;
    }
    x
}

#[test]
fn the_register_round_trip_matches_the_bytes_bit_for_bit() {
    let _gpu = gpu_serial();
    let Ok(candle::Device::Cuda(dev)) = candle::Device::cuda_if_available(0) else {
        return;
    };
    let stream = dev.cuda_stream();

    const N_BLOCKS: usize = 9 * 256;
    let blocks: Vec<[f32; CHUNK_SIZE]> = (0..N_BLOCKS).map(block).collect();
    let flat: Vec<f32> = blocks.iter().flatten().copied().collect();
    let src: CudaSlice<f32> = dev.memcpy_stod(&flat).expect("upload blocks");
    let bytes_gpu: CudaSlice<f32> = stream.alloc_zeros(flat.len()).expect("alloc");
    let lane_gpu: CudaSlice<f32> = stream.alloc_zeros(flat.len()).expect("alloc");

    for fmt in FORMATS {
        let tag = SampleFormat::from_kv_format(KvFormat::Quantized(fmt))
            .expect("sample format")
            .to_cuda_tag();
        for is_k in [1, 0] {
            for outer in OUTERS {
                {
                    let (src_ptr, _g0) = src.device_ptr(&stream);
                    let (bytes_ptr, _g1) = bytes_gpu.device_ptr(&stream);
                    let (lane_ptr, _g2) = lane_gpu.device_ptr(&stream);
                    unsafe {
                        run_select_roundtrip_parity(
                            src_ptr as *const c_void,
                            bytes_ptr as *mut c_void,
                            lane_ptr as *mut c_void,
                            N_BLOCKS as c_int,
                            outer,
                            tag,
                            is_k,
                        );
                    }
                }
                stream.synchronize().expect("sync");
                let bytes = dev.memcpy_dtov(&bytes_gpu).expect("download");
                let lanes = dev.memcpy_dtov(&lane_gpu).expect("download");
                for (i, (b, l)) in bytes.iter().zip(&lanes).enumerate() {
                    assert_eq!(
                        b.to_bits(),
                        l.to_bits(),
                        "{fmt:?} is_k={is_k} outer={outer}: block {} lane {}: bytes {b:e}, register {l:e}",
                        i / CHUNK_SIZE,
                        i % CHUNK_SIZE
                    );
                }
            }
        }
    }
}
