//! Unit tests for `gather_r16_kv_probe` and the underlying CUDA kernel.
//!
//! The fast path replaces O(n_head × N_PALETTE × n_blocks) synchronous
//! `memcpy_dtov` calls with a single kernel launch + one DtoH copy.
//!
//! Output layout contract tested here:
//!   Combined buffer  out_kqv[0..N) = K,  [N..2N) = Q,  [2N..3N) = V
//!   where N = n_warps × CHUNK_SIZE × sub_head_dim.
//!
//!   Within each section the kernel writes D-MAJOR per warp:
//!     index(warp_id, d, token) = warp_id × CHUNK_SIZE × sub_head_dim
//!                              + d × CHUNK_SIZE + token
//!
//!   Callers (backing.rs) transpose d-major → token-major during the
//!   F16→F32 pass so the consumer (r16_block_to_turn_signatures) sees the
//!   standard [token × sub_head_dim + d] layout.

#[cfg(feature = "cuda")]
mod tests {
    /// Smoke test: manually construct a minimal R16 K chunk and a float-F16 V chunk,
    /// upload both to GPU, run the kernel, and verify the d-major output is correct.
    ///
    /// Config: n_warps = 1 (one (block=0, head=0, palette=0) triple)
    ///         sub_head_dim = 2, CHUNK_SIZE = 32
    #[test]
    fn test_gather_r16_kv_kernel_correctness() {
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::cuda_backend::kernels;
        use candle::Device;

        let dev = match Device::cuda_if_available(0) {
            Ok(d) => d,
            Err(_) => return, // no GPU, skip
        };
        let Device::Cuda(cuda_dev) = &dev else {
            return;
        };

        const CHUNK_SIZE: usize = 32;
        let sub_head_dim: usize = 2;
        let n_warps: usize = 1;

        // Build R16 K data: sub_head_dim=2 groups × 128 bytes = 256 bytes.
        // Group d: bytes 0..63 = K F16[32], bytes 64..127 = Q F16[32].
        let mut k_data: Vec<u8> = vec![0u8; sub_head_dim * 128];
        for d in 0..sub_head_dim {
            for t in 0..CHUNK_SIZE {
                let k_val = half::f16::from_f32((d * 100 + t) as f32);
                let q_val = half::f16::from_f32((1000 * (d + 1) + t) as f32);
                let blk_off = d * 128;
                let [k0, k1] = k_val.to_le_bytes();
                let [q0, q1] = q_val.to_le_bytes();
                k_data[blk_off + t * 2] = k0;
                k_data[blk_off + t * 2 + 1] = k1;
                k_data[blk_off + 64 + t * 2] = q0;
                k_data[blk_off + 64 + t * 2 + 1] = q1;
            }
        }

        // Build V F16 data: token-major [CHUNK_SIZE, sub_head_dim].
        let mut v_data: Vec<u8> = vec![0u8; CHUNK_SIZE * sub_head_dim * 2];
        for t in 0..CHUNK_SIZE {
            for d in 0..sub_head_dim {
                let v_val = half::f16::from_f32((t * 10 + d) as f32);
                let [b0, b1] = v_val.to_le_bytes();
                let idx = (t * sub_head_dim + d) * 2;
                v_data[idx] = b0;
                v_data[idx + 1] = b1;
            }
        }

        let k_gpu = cuda_dev.memcpy_stod(&k_data).expect("k HtoD");
        let v_gpu = cuda_dev.memcpy_stod(&v_data).expect("v HtoD");

        let stream = cuda_dev.cuda_stream();
        let k_raw = k_gpu.device_ptr(&stream).0 as i64;
        let v_raw = v_gpu.device_ptr(&stream).0 as i64;
        drop(stream);

        let k_ptrs_gpu = cuda_dev.memcpy_stod(&vec![k_raw]).expect("k_ptrs HtoD");
        let v_ptrs_gpu = cuda_dev.memcpy_stod(&vec![v_raw]).expect("v_ptrs HtoD");

        let total_elems = n_warps * CHUNK_SIZE * sub_head_dim;
        let out_kqv = unsafe {
            cuda_dev
                .alloc::<half::f16>(3 * total_elems)
                .expect("out_kqv alloc")
        };

        let stream = cuda_dev.cuda_stream();
        {
            let (kp, _kg) = k_ptrs_gpu.device_ptr(&stream);
            let (vp, _vg) = v_ptrs_gpu.device_ptr(&stream);
            let (okqv, _og) = out_kqv.device_ptr(&stream);
            unsafe {
                kernels::simple::gather_r16_kv::run_gather_r16_kv_f16(
                    kp as *const i64,
                    vp as *const i64,
                    okqv as *mut std::ffi::c_void,
                    n_warps as i32,
                    sub_head_dim as i32,
                    stream.cu_stream() as *mut _,
                );
            }
        }

        let kqv_cpu = cuda_dev.memcpy_dtov(&out_kqv).expect("kqv DtoH");
        let out_k_cpu = &kqv_cpu[..total_elems];
        let out_q_cpu = &kqv_cpu[total_elems..2 * total_elems];
        let out_v_cpu = &kqv_cpu[2 * total_elems..];

        // Kernel writes d-major: index(warp=0, d, token) = d * CHUNK_SIZE + token.
        for t in 0..CHUNK_SIZE {
            for d in 0..sub_head_dim {
                let idx = d * CHUNK_SIZE + t; // d-major
                let got_k = out_k_cpu[idx].to_f32();
                let got_q = out_q_cpu[idx].to_f32();
                let got_v = out_v_cpu[idx].to_f32();

                let exp_k = (d * 100 + t) as f32;
                let exp_q = (1000 * (d + 1) + t) as f32;
                let exp_v = (t * 10 + d) as f32;

                assert_eq!(got_k, exp_k, "K mismatch at token={t} dim={d}");
                assert_eq!(got_q, exp_q, "Q mismatch at token={t} dim={d}");
                assert_eq!(got_v, exp_v, "V mismatch at token={t} dim={d}");
            }
        }
    }

    /// Multi-warp test: 2 warps (two (block, head, palette) triples) in one launch.
    /// Verifies that warp_id indexing separates outputs correctly in d-major layout.
    #[test]
    fn test_gather_r16_kv_multi_warp() {
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::cuda_backend::kernels;
        use candle::Device;

        let dev = match Device::cuda_if_available(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let Device::Cuda(cuda_dev) = &dev else {
            return;
        };

        const CHUNK_SIZE: usize = 32;
        let sub_head_dim: usize = 2;
        let n_warps: usize = 2;

        let build_r16 = |offset: f32| -> Vec<u8> {
            let mut buf = vec![0u8; sub_head_dim * 128];
            for d in 0..sub_head_dim {
                for t in 0..CHUNK_SIZE {
                    let k_val = half::f16::from_f32(offset + (d * 100 + t) as f32);
                    let q_val = half::f16::from_f32(offset + 500.0 + (d * 100 + t) as f32);
                    let blk_off = d * 128;
                    let [k0, k1] = k_val.to_le_bytes();
                    let [q0, q1] = q_val.to_le_bytes();
                    buf[blk_off + t * 2] = k0;
                    buf[blk_off + t * 2 + 1] = k1;
                    buf[blk_off + 64 + t * 2] = q0;
                    buf[blk_off + 64 + t * 2 + 1] = q1;
                }
            }
            buf
        };

        let build_v = |offset: f32| -> Vec<u8> {
            let mut buf = vec![0u8; CHUNK_SIZE * sub_head_dim * 2];
            for t in 0..CHUNK_SIZE {
                for d in 0..sub_head_dim {
                    let v_val = half::f16::from_f32(offset + (t * 10 + d) as f32);
                    let [b0, b1] = v_val.to_le_bytes();
                    let idx = (t * sub_head_dim + d) * 2;
                    buf[idx] = b0;
                    buf[idx + 1] = b1;
                }
            }
            buf
        };

        // All offsets < 2048: every value is exactly representable in F16.
        let k0_data = build_r16(0.0);
        let k1_data = build_r16(400.0);
        let v0_data = build_v(0.0);
        let v1_data = build_v(700.0);

        let k0_gpu = cuda_dev.memcpy_stod(&k0_data).unwrap();
        let k1_gpu = cuda_dev.memcpy_stod(&k1_data).unwrap();
        let v0_gpu = cuda_dev.memcpy_stod(&v0_data).unwrap();
        let v1_gpu = cuda_dev.memcpy_stod(&v1_data).unwrap();

        let stream = cuda_dev.cuda_stream();
        let k_ptrs: Vec<i64> = vec![
            k0_gpu.device_ptr(&stream).0 as i64,
            k1_gpu.device_ptr(&stream).0 as i64,
        ];
        let v_ptrs: Vec<i64> = vec![
            v0_gpu.device_ptr(&stream).0 as i64,
            v1_gpu.device_ptr(&stream).0 as i64,
        ];
        drop(stream);

        let k_ptrs_gpu = cuda_dev.memcpy_stod(&k_ptrs).unwrap();
        let v_ptrs_gpu = cuda_dev.memcpy_stod(&v_ptrs).unwrap();

        let total_elems = n_warps * CHUNK_SIZE * sub_head_dim;
        let out_kqv = unsafe { cuda_dev.alloc::<half::f16>(3 * total_elems).unwrap() };

        let stream = cuda_dev.cuda_stream();
        {
            let (kp, _kg) = k_ptrs_gpu.device_ptr(&stream);
            let (vp, _vg) = v_ptrs_gpu.device_ptr(&stream);
            let (okqv, _og) = out_kqv.device_ptr(&stream);
            unsafe {
                kernels::simple::gather_r16_kv::run_gather_r16_kv_f16(
                    kp as *const i64,
                    vp as *const i64,
                    okqv as *mut std::ffi::c_void,
                    n_warps as i32,
                    sub_head_dim as i32,
                    stream.cu_stream() as *mut _,
                );
            }
        }

        let kqv_cpu = cuda_dev.memcpy_dtov(&out_kqv).unwrap();
        let out_k_cpu = &kqv_cpu[..total_elems];
        let out_q_cpu = &kqv_cpu[total_elems..2 * total_elems];
        let out_v_cpu = &kqv_cpu[2 * total_elems..];

        // D-major: index(warp_id=wi, d, token=t) = wi * CHUNK_SIZE * sub_head_dim
        //                                         + d * CHUNK_SIZE + t
        let warp_stride = CHUNK_SIZE * sub_head_dim;
        for (wi, &(k_off, v_off)) in [(0.0f32, 0.0f32), (400.0, 700.0)].iter().enumerate() {
            for t in 0..CHUNK_SIZE {
                for d in 0..sub_head_dim {
                    let idx = wi * warp_stride + d * CHUNK_SIZE + t; // d-major
                    let got_k = out_k_cpu[idx].to_f32();
                    let got_q = out_q_cpu[idx].to_f32();
                    let got_v = out_v_cpu[idx].to_f32();
                    let exp_k = k_off + (d * 100 + t) as f32;
                    let exp_q = k_off + 500.0 + (d * 100 + t) as f32;
                    let exp_v = v_off + (t * 10 + d) as f32;
                    assert_eq!(got_k, exp_k, "warp={wi} t={t} d={d} K");
                    assert_eq!(got_q, exp_q, "warp={wi} t={t} d={d} Q");
                    assert_eq!(got_v, exp_v, "warp={wi} t={t} d={d} V");
                }
            }
        }
    }
}
