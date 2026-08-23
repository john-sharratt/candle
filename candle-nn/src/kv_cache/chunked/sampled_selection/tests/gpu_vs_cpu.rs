// Test code: expressions are kept in the shape of the formula under comparison.
#![allow(clippy::identity_op, clippy::map_flatten)]

#[allow(unused_imports)]
use super::*;
#[cfg(feature = "cuda")]
use crate::kv_cache::arena_table::N_PALETTE;
#[allow(unused_imports)]
use candle::quantized::pinned_staging::GpuBuf;

#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn gpu_matches_cpu_real_data() {
    // One lock per test, taken before the first device touch: the crate-wide
    // guard is not reentrant, and several of these acquire a device more than
    // once. See `crate::kv_cache::chunked::gpu_test_lock`.
    #[cfg(feature = "cuda")]
    let _gpu = crate::kv_cache::chunked::gpu_test_lock::gpu_serial();
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::quantized::GgmlDType;

    let path = match r16_dump_path().or_else(dump_path) {
        Some(p) => p,
        None => {
            println!("SKIP: no dump file found at {R16_DUMP_REL_PATH} or {DUMP_REL_PATH}");
            return;
        }
    };
    let (header, chunks) = match load_dump(&path) {
        Some(v) => v,
        None => {
            println!("SKIP: failed to parse dump");
            return;
        }
    };
    let use_chunks = chunks.iter().take(8).collect::<Vec<_>>();
    let has_q_capture = use_chunks.iter().any(|chunk| {
        chunk
            .q
            .as_ref()
            .is_some_and(|q| q.iter().any(|&v| v != 0.0))
    });

    let dev = candle::Device::cuda_if_available(0).expect("cuda_if_available");
    let cuda_dev = match &dev {
        candle::Device::Cuda(d) => d.clone(),
        _ => {
            println!("SKIP: no CUDA device available");
            return;
        }
    };
    let stream = cuda_dev.cuda_stream();

    struct ChunkGpu {
        k_gpu: candle::cuda_backend::cudarc::driver::CudaSlice<u8>,
        v_gpu: candle::cuda_backend::cudarc::driver::CudaSlice<u8>,
    }

    let chunk_gpus: Vec<ChunkGpu> = use_chunks
        .iter()
        .map(|chunk| {
            let q = chunk
                .q
                .as_ref()
                .expect("q-bearing dump required for GPU relevance test");
            let k_bytes = pack_r16_blocks(&chunk.k, q);
            let v_bytes = pack_f16(&chunk.v);
            ChunkGpu {
                k_gpu: cuda_dev.memcpy_stod(&k_bytes).expect("upload K R16"),
                v_gpu: cuda_dev.memcpy_stod(&v_bytes).expect("upload V F16"),
            }
        })
        .collect();

    let k_chunk_byte_stride = ((header.chunk_size * header.head_dim) / CHUNK_SIZE * 128) as i64;
    let v_chunk_byte_stride = (header.chunk_size * header.head_dim * 2) as i64;
    // Unity outer scale, and one `Palette4PerHeadEntry` row per (chunk, head):
    // four 9-value sub-entries. Every band of this fixture shares one buffer,
    // so the four are identical — but each must be present, because the kernel
    // resolves a band through its own sub-entry, not through palette 0.
    let outer_one_bits = 1.0_f32.to_bits() as i64;
    let per_head_table_host: Vec<i64> = chunk_gpus
        .iter()
        .map(|cg| {
            let (k_ptr, _) = cg.k_gpu.device_ptr(&stream);
            let (v_ptr, _) = cg.v_gpu.device_ptr(&stream);
            let metadata = (39i64 << 16) | (1i64 << 8) | 0i64;
            let sub = [
                k_ptr as i64,
                v_ptr as i64,
                0i64,
                0i64,
                k_chunk_byte_stride,
                v_chunk_byte_stride,
                metadata,
                outer_one_bits,
                outer_one_bits,
            ];
            sub.repeat(N_PALETTE)
        })
        .flatten()
        .collect();
    let per_head_table_gpu = cuda_dev.memcpy_stod(&per_head_table_host).expect("table");
    let per_head_table_buf = {
        let (ptr, _) = per_head_table_gpu.device_ptr(&stream);
        GpuBuf::from_borrowed(ptr, per_head_table_host.len() * std::mem::size_of::<i64>())
    };

    const TEST_ARENA_CHUNKS: i64 = 8192;
    let mut head_gids = Vec::with_capacity(use_chunks.len() * 2);
    for (i, _) in use_chunks.iter().enumerate() {
        head_gids.push(i as i64 * TEST_ARENA_CHUNKS);
        head_gids.push(i as i64 * TEST_ARENA_CHUNKS);
    }
    let head_gids_buf = {
        let gpu_u8 = cuda_dev
            .memcpy_stod(unsafe {
                std::slice::from_raw_parts(
                    head_gids.as_ptr() as *const u8,
                    std::mem::size_of_val(head_gids.as_slice()),
                )
            })
            .expect("head_gids upload");
        GpuBuf::from_raw_owned(gpu_u8, &cuda_dev)
    };

    let candidates = vec![
        GgmlDType::Q0,
        GgmlDType::Q0_V,
        GgmlDType::Q1_A,
        GgmlDType::Q0_X,
        GgmlDType::Q0_M2,
        GgmlDType::Q1_S,
        GgmlDType::Q0_M4,
        GgmlDType::Q2_S,
        GgmlDType::Q2_0,
        GgmlDType::Q2_A,
        GgmlDType::Q2_1,
        GgmlDType::Q3_0,
        GgmlDType::Q3_1,
        GgmlDType::Q4_0,
        GgmlDType::Q4_1,
        GgmlDType::Q4_KS,
        GgmlDType::Q8_0,
        GgmlDType::Q8_1,
        GgmlDType::Q8_KS,
        GgmlDType::BF16,
        GgmlDType::F16,
    ];

    for (side_is_k, sample_side) in [(true, SampleSide::Key), (false, SampleSide::Value)] {
        let gpu = sample_error_surface_gpu_paged(
            &per_head_table_buf,
            &head_gids_buf,
            &candidates,
            0,
            sample_side,
            use_chunks.len(),
            1,
            header.head_dim,
            TEST_ARENA_CHUNKS as usize,
            &cuda_dev,
            None,
        )
        .expect("gpu sampled errors");
        let gpu_vals: Vec<f32> = gpu.data.clone();
        if side_is_k {
            assert!(
                has_q_capture,
                "GPU relevance test requires a q-bearing R16 dump"
            );
            let q_response = gpu.q_relevance.as_ref().expect("gpu q relevance response");
            assert_eq!(q_response.len(), use_chunks.len() * header.head_dim);
            let all_ones = q_response.iter().all(|&v| (v - 1.0).abs() <= 1e-6);
            let all_zeros = q_response.iter().all(|&v| v.abs() <= 1e-6);
            assert!(!all_ones, "degenerate GPU q relevance response: all 1.0");
            assert!(!all_zeros, "degenerate GPU q relevance response: all 0.0");
        }

        let mut data = Vec::with_capacity(use_chunks.len() * header.chunk_size * header.head_dim);
        for chunk in &use_chunks {
            let first_head_len = header.chunk_size * header.head_dim;
            let src = if side_is_k {
                &chunk.k[..first_head_len]
            } else {
                &chunk.v[..first_head_len]
            };
            data.extend(src.iter().map(|&v| f16::from_f32(v).to_f32()));
        }
        assert!(data.iter().any(|v| *v > 0.0));
        assert!(data.iter().any(|v| *v < 0.0));
        let cpu = sample_error_surface_cpu(
            &data,
            use_chunks.len(),
            1,
            header.chunk_size,
            header.head_dim,
            0,
            &candidate_formats(),
            sample_side,
            None,
        )
        .expect("cpu surface");

        assert_eq!(gpu_vals.len(), cpu.data.len());
        let mut max_diff = 0.0f32;
        for (g, c) in gpu_vals.iter().zip(cpu.data.iter()) {
            max_diff = max_diff.max((g - c).abs());
        }
        assert!(max_diff <= 1e-3, "max diff too large: {max_diff}");
    }
}
