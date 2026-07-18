//! GPU integration tests for the palette4_convert kernel.
//!
//! These tests construct KvHead metadata on the CPU, allocate CUDA device
//! memory for arenas, fill src arenas with known data, invoke the kernel
//! via FFI, and verify the output matches the CPU reference.
#![cfg(feature = "cuda")]

use candle_core::quantized::pinned_staging::PinnedStager;
use candle_core::quantized::GgmlDType;
use candle_core::{Device, Result};
use cudarc::driver::DevicePtr;
use half::{bf16, f16};

// KvHead layout constants for HD=128
const HD: usize = 128;
const N_PAL: usize = 4;
const PAL_DIM: usize = HD / N_PAL; // 32
const CHUNK_SIZE: usize = 32;

// R16 block size: half d[32] + uint16_t q[32] = 128 bytes
const R16_BLOCK_BYTES: usize = 128;

/// Build an identity pal_map: dims 0..31 → pal 0, 32..63 → pal 1, etc.
fn identity_pal_map() -> [u8; HD / 4] {
    let mut out = [0u8; HD / 4];
    for d in 0..HD {
        let p = (d / PAL_DIM) as u8;
        out[d / 4] |= (p & 0x3) << (2 * (d % 4));
    }
    out
}

/// Fill R16 arena data for one palette: `num_dims` dims × `num_chunks` chunks.
/// R16 layout: block(local_d, chunk_c) = arena_base + (local_d * num_chunks + c) * 128
/// Within block: half d[32] at offset 0 (tokens 0..31).
fn fill_r16_arena(num_dims: usize, num_chunks: usize, base_val: f32) -> Vec<u8> {
    let total_blocks = num_dims * num_chunks;
    let mut data = vec![0u8; total_blocks * R16_BLOCK_BYTES];
    for ld in 0..num_dims {
        for c in 0..num_chunks {
            let blk_off = (ld * num_chunks + c) * R16_BLOCK_BYTES;
            for t in 0..CHUNK_SIZE {
                let val = base_val + (c * CHUNK_SIZE + t) as f32 * 1000.0 + ld as f32;
                let h = f16::from_f32(val);
                let off = blk_off + t * 2;
                data[off..off + 2].copy_from_slice(&h.to_le_bytes());
            }
        }
    }
    data
}

/// Read back R16 arena and extract f16 values.
fn read_r16_arena(data: &[u8], num_dims: usize, num_chunks: usize) -> Vec<Vec<Vec<f32>>> {
    // [dim][chunk][token]
    let mut out = vec![vec![vec![0.0f32; CHUNK_SIZE]; num_chunks]; num_dims];
    for ld in 0..num_dims {
        for c in 0..num_chunks {
            let blk_off = (ld * num_chunks + c) * R16_BLOCK_BYTES;
            for t in 0..CHUNK_SIZE {
                let off = blk_off + t * 2;
                let h = f16::from_le_bytes([data[off], data[off + 1]]);
                out[ld][c][t] = h.to_f32();
            }
        }
    }
    out
}

/// Helper to get CudaDevice and allocate/copy to device.
fn get_cuda_dev() -> Result<Device> {
    Device::new_cuda(0)
}

/// Convert an arena format code (u8) back to GgmlDType for use with the buffered API.
fn fmt_code_to_ggml_dtype(code: u8) -> GgmlDType {
    match code {
        0 => GgmlDType::F32,
        1 => GgmlDType::F16,
        2 => GgmlDType::BF16,
        3 => GgmlDType::R16,
        4 => GgmlDType::P2,
        5 => GgmlDType::QAWQ,
        6 => GgmlDType::QAWQ_G64,
        7 => GgmlDType::Q8_0,
        8 => GgmlDType::Q8_1,
        9 => GgmlDType::Q8_K,
        10 => GgmlDType::Q8_KS,
        11 => GgmlDType::Q6_K,
        12 => GgmlDType::Q5_0,
        13 => GgmlDType::Q5_1,
        14 => GgmlDType::Q5_K,
        15 => GgmlDType::Q4_0,
        16 => GgmlDType::Q4_1,
        17 => GgmlDType::Q4_K,
        18 => GgmlDType::Q4_KS,
        19 => GgmlDType::Q3_0,
        20 => GgmlDType::Q3_1,
        21 => GgmlDType::Q3_K,
        22 => GgmlDType::Q2_0,
        23 => GgmlDType::Q2_1,
        24 => GgmlDType::Q2_K,
        25 => GgmlDType::Q2_S,
        26 => GgmlDType::Q2_A,
        27 => GgmlDType::Q1_S,
        28 => GgmlDType::Q0_V,
        29 => GgmlDType::Q1_A,
        30 => GgmlDType::Q0_X,
        31 => GgmlDType::Q0_M2,
        32 => GgmlDType::Q0_M4,
        33 => GgmlDType::Q0,
        c => panic!("unknown arena fmt code {c}"),
    }
}

#[test]
fn palette4_convert_r16_identity_roundtrip() -> Result<()> {
    // R16 src → R16 dst with identity pal_map: output should match input exactly.
    let dev = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()), // skip if no GPU
    };
    let cuda_dev = match &dev {
        Device::Cuda(d) => d,
        _ => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();
    use candle_core::quantized::cuda::{quantize_palette4_convert_buffered, PalHeadDesc};
    let stream = cuda_dev.cuda_stream();

    let num_chunks = 2usize;

    let mut src_arena_data = Vec::new();
    for p in 0..N_PAL {
        src_arena_data.push(fill_r16_arena(PAL_DIM, num_chunks, p as f32 * 10000.0));
    }
    let arena_size = PAL_DIM * num_chunks * R16_BLOCK_BYTES;

    let mut src_gpu = Vec::new();
    let mut dst_gpu = Vec::new();
    for p in 0..N_PAL {
        src_gpu.push(cuda_dev.memcpy_stod(&src_arena_data[p])?);
        dst_gpu.push(cuda_dev.memcpy_stod(&vec![0u8; arena_size])?);
    }

    let sptrs: [u64; N_PAL] = std::array::from_fn(|p| src_gpu[p].device_ptr(&stream).0 as u64);
    let dptrs: [u64; N_PAL] = std::array::from_fn(|p| dst_gpu[p].device_ptr(&stream).0 as u64);
    let ident = identity_pal_map();

    quantize_palette4_convert_buffered(
        &[PalHeadDesc {
            k_src_arena_ptrs: sptrs,
            v_src_arena_ptrs: sptrs,
            k_src_fmts: [GgmlDType::R16; N_PAL],
            v_src_fmts: [GgmlDType::R16; N_PAL],
            k_src_pal_map: ident,
            v_src_pal_map: ident,
            k_src_scales: [1.0f32; N_PAL],
            v_src_scales: [1.0f32; N_PAL],
            k_dst_arena_ptrs: dptrs,
            v_dst_arena_ptrs: dptrs,
            k_dst_fmts: [GgmlDType::R16; N_PAL],
            v_dst_fmts: [GgmlDType::R16; N_PAL],
            k_dst_pal_map: ident,
            v_dst_pal_map: ident,
            k_dst_scales: [1.0f32; N_PAL],
            v_dst_scales: [1.0f32; N_PAL],
        }],
        1,
        1,
        num_chunks,
        &PinnedStager::new(cuda_dev).begin_generation(),
        &stream,
    )?;
    dev.synchronize()?;

    // Read back dst arenas
    for p in 0..N_PAL {
        let dst_data = cuda_dev.memcpy_dtov(&dst_gpu[p])?;

        let src_vals = read_r16_arena(&src_arena_data[p], PAL_DIM, num_chunks);
        let dst_vals = read_r16_arena(&dst_data, PAL_DIM, num_chunks);

        for ld in 0..PAL_DIM {
            for c in 0..num_chunks {
                for t in 0..CHUNK_SIZE {
                    let src_v = src_vals[ld][c][t];
                    let dst_v = dst_vals[ld][c][t];
                    // R16→R16 with identity map through f16 staging: should be exact
                    assert_eq!(
                        dst_v, src_v,
                        "pal={p} dim={ld} chunk={c} token={t}: got {dst_v}, want {src_v}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn palette4_convert_f16_to_r16_identity() -> Result<()> {
    // F16 channel-oriented src → R16 token-oriented dst with identity pal_map.
    // Tests the cp.async F16 fast path and format conversion.
    let dev = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let cuda_dev = match &dev {
        Device::Cuda(d) => d,
        _ => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();

    let num_chunks = 1usize;
    let _num_layers = 1usize;
    let _num_kv_heads = 1usize;

    // F16 channel-oriented layout: (c * CHUNK * PAL_DIM + t * PAL_DIM + local_d) * 2
    let f16_arena_size = num_chunks * CHUNK_SIZE * PAL_DIM * 2;
    let r16_arena_size = PAL_DIM * num_chunks * R16_BLOCK_BYTES;

    let mut src_gpu = Vec::new();
    let mut dst_gpu = Vec::new();
    let mut src_data_all = Vec::new();

    for p in 0..N_PAL {
        let mut src_data = vec![0u8; f16_arena_size];
        for c in 0..num_chunks {
            for t in 0..CHUNK_SIZE {
                for ld in 0..PAL_DIM {
                    let val =
                        (p as f32) * 10000.0 + (c * CHUNK_SIZE + t) as f32 * 100.0 + ld as f32;
                    let h = f16::from_f32(val);
                    let off = (c * CHUNK_SIZE * PAL_DIM + t * PAL_DIM + ld) * 2;
                    src_data[off..off + 2].copy_from_slice(&h.to_le_bytes());
                }
            }
        }
        src_data_all.push(src_data.clone());
        src_gpu.push(cuda_dev.memcpy_stod(&src_data)?);
        dst_gpu.push(cuda_dev.memcpy_stod(&vec![0u8; r16_arena_size])?);
    }

    let mut src_arena_ptrs = [0u64; N_PAL];
    let mut dst_arena_ptrs = [0u64; N_PAL];
    for p in 0..N_PAL {
        src_arena_ptrs[p] = src_gpu[p].device_ptr(&stream).0 as u64;
        dst_arena_ptrs[p] = dst_gpu[p].device_ptr(&stream).0 as u64;
    }

    use candle_core::quantized::cuda::{quantize_palette4_convert_buffered, PalHeadDesc};
    let ident = identity_pal_map();
    quantize_palette4_convert_buffered(
        &[PalHeadDesc {
            k_src_arena_ptrs: src_arena_ptrs,
            v_src_arena_ptrs: src_arena_ptrs,
            k_src_fmts: [GgmlDType::F16; N_PAL],
            v_src_fmts: [GgmlDType::F16; N_PAL],
            k_src_pal_map: ident,
            v_src_pal_map: ident,
            k_src_scales: [1.0f32; N_PAL],
            v_src_scales: [1.0f32; N_PAL],
            k_dst_arena_ptrs: dst_arena_ptrs,
            v_dst_arena_ptrs: dst_arena_ptrs,
            k_dst_fmts: [GgmlDType::R16; N_PAL],
            v_dst_fmts: [GgmlDType::R16; N_PAL],
            k_dst_pal_map: ident,
            v_dst_pal_map: ident,
            k_dst_scales: [1.0f32; N_PAL],
            v_dst_scales: [1.0f32; N_PAL],
        }],
        1,
        1,
        num_chunks,
        &PinnedStager::new(cuda_dev).begin_generation(),
        &stream,
    )?;
    dev.synchronize()?;

    // Read back dst R16 arenas and compare with src F16 values
    for p in 0..N_PAL {
        let dst_data = cuda_dev.memcpy_dtov(&dst_gpu[p])?;
        let dst_vals = read_r16_arena(&dst_data, PAL_DIM, num_chunks);

        for ld in 0..PAL_DIM {
            for c in 0..num_chunks {
                for t in 0..CHUNK_SIZE {
                    // Expected: f16 staging is lossless for F16 src, so values should be exact
                    let src_off = (c * CHUNK_SIZE * PAL_DIM + t * PAL_DIM + ld) * 2;
                    let src_h = f16::from_le_bytes([
                        src_data_all[p][src_off],
                        src_data_all[p][src_off + 1],
                    ]);
                    let expected = src_h.to_f32();
                    let got = dst_vals[ld][c][t];
                    assert_eq!(
                        got, expected,
                        "F16→R16 pal={p} dim={ld} chunk={c} token={t}: got {got}, want {expected}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn palette4_convert_multi_layer_multi_head() -> Result<()> {
    // R16→R16 with 2 layers, 2 heads to test grid indexing.
    let dev = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let cuda_dev = match &dev {
        Device::Cuda(d) => d,
        _ => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();

    let num_chunks = 1usize;
    let num_layers = 2usize;
    let num_kv_heads = 2usize;
    let total_jobs = num_layers * num_kv_heads; // 4

    let arena_size = PAL_DIM * num_chunks * R16_BLOCK_BYTES;

    // Allocate arenas for each job
    let mut src_arena_gpus = Vec::new();
    let mut dst_arena_gpus = Vec::new();
    let mut src_data_cpu = Vec::new();

    for job in 0..total_jobs {
        let mut job_src_arenas = Vec::new();
        let mut job_dst_arenas = Vec::new();
        let mut src_arena_ptrs = [0u64; N_PAL];
        let mut dst_arena_ptrs = [0u64; N_PAL];

        // Each job gets unique data based on job index
        let mut job_src_data = Vec::new();
        for p in 0..N_PAL {
            let base = (job * N_PAL + p) as f32 * 10000.0;
            let data = fill_r16_arena(PAL_DIM, num_chunks, base);
            let src_g = cuda_dev.memcpy_stod(&data)?;
            let dst_g = cuda_dev.memcpy_stod(&vec![0u8; arena_size])?;
            src_arena_ptrs[p] = src_g.device_ptr(&stream).0 as u64;
            dst_arena_ptrs[p] = dst_g.device_ptr(&stream).0 as u64;
            job_src_arenas.push(src_g);
            job_dst_arenas.push(dst_g);
            job_src_data.push(data);
        }

        let k_fmts = [FMT_R16; N_PAL];
        let _ = k_fmts; // arena pointers captured above

        src_arena_gpus.push(job_src_arenas);
        dst_arena_gpus.push(job_dst_arenas);
        src_data_cpu.push(job_src_data);
    }

    use candle_core::quantized::cuda::{quantize_palette4_convert_buffered, PalHeadDesc};
    let ident = identity_pal_map();
    let mut descs: Vec<PalHeadDesc> = Vec::new();
    for job in 0..total_jobs {
        let sp: [u64; N_PAL] =
            std::array::from_fn(|p| src_arena_gpus[job][p].device_ptr(&stream).0 as u64);
        let dp: [u64; N_PAL] =
            std::array::from_fn(|p| dst_arena_gpus[job][p].device_ptr(&stream).0 as u64);
        descs.push(PalHeadDesc {
            k_src_arena_ptrs: sp,
            v_src_arena_ptrs: sp,
            k_src_fmts: [GgmlDType::R16; N_PAL],
            v_src_fmts: [GgmlDType::R16; N_PAL],
            k_src_pal_map: ident,
            v_src_pal_map: ident,
            k_src_scales: [1.0f32; N_PAL],
            v_src_scales: [1.0f32; N_PAL],
            k_dst_arena_ptrs: dp,
            v_dst_arena_ptrs: dp,
            k_dst_fmts: [GgmlDType::R16; N_PAL],
            v_dst_fmts: [GgmlDType::R16; N_PAL],
            k_dst_pal_map: ident,
            v_dst_pal_map: ident,
            k_dst_scales: [1.0f32; N_PAL],
            v_dst_scales: [1.0f32; N_PAL],
        });
    }
    quantize_palette4_convert_buffered(
        &descs,
        num_kv_heads,
        num_layers,
        num_chunks,
        &PinnedStager::new(cuda_dev).begin_generation(),
        &stream,
    )?;
    dev.synchronize()?;

    // Verify each job
    for job in 0..total_jobs {
        for p in 0..N_PAL {
            let dst_data = cuda_dev.memcpy_dtov(&dst_arena_gpus[job][p])?;

            let src_vals = read_r16_arena(&src_data_cpu[job][p], PAL_DIM, num_chunks);
            let dst_vals = read_r16_arena(&dst_data, PAL_DIM, num_chunks);

            for ld in 0..PAL_DIM {
                for c in 0..num_chunks {
                    for t in 0..CHUNK_SIZE {
                        assert_eq!(
                            dst_vals[ld][c][t], src_vals[ld][c][t],
                            "job={job} pal={p} dim={ld} chunk={c} token={t}"
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

/// A/B equivalence: converting N chunks in ONE batched launch (`num_layers=N`)
/// must be **bit-identical** to N separate per-chunk launches (`num_layers=1`,
/// looped) — the production hot→warm quantize batches this way to collapse the
/// per-launch overhead. Each `(chunk, head)` job carries its OWN dst pal_map
/// (shuffled, seeded by job) so this genuinely exercises the per-job "dst-side
/// state variance" the per-chunk path existed to preserve; if batching bled
/// state across grid blocks, the dst bytes would diverge.
#[test]
fn palette4_convert_batched_matches_per_chunk() -> Result<()> {
    use candle_core::quantized::cuda::{
        quantize_palette4_convert_buffered, shuffled_pal_map_128, PalHeadDesc,
    };
    let dev = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let cuda_dev = match &dev {
        Device::Cuda(d) => d,
        _ => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();

    let num_chunks = 1usize; // each grid "layer" is one independent chunk
    let n_layers = 4usize; // 4 chunks batched into one launch in mode B
    let num_kv_heads = 2usize;
    let total_jobs = n_layers * num_kv_heads;
    let arena_size = PAL_DIM * num_chunks * R16_BLOCK_BYTES;
    let ident = identity_pal_map();

    // Shared src arenas; separate dst sets for A (per-chunk) and B (batched).
    let mut src_gpus = Vec::new();
    let mut dst_a_gpus = Vec::new();
    let mut dst_b_gpus = Vec::new();
    for job in 0..total_jobs {
        let mut sj = Vec::new();
        let mut da = Vec::new();
        let mut db = Vec::new();
        for p in 0..N_PAL {
            let base = (job * N_PAL + p) as f32 * 100.0;
            let data = fill_r16_arena(PAL_DIM, num_chunks, base);
            sj.push(cuda_dev.memcpy_stod(&data)?);
            da.push(cuda_dev.memcpy_stod(&vec![0u8; arena_size])?);
            db.push(cuda_dev.memcpy_stod(&vec![0u8; arena_size])?);
        }
        src_gpus.push(sj);
        dst_a_gpus.push(da);
        dst_b_gpus.push(db);
    }

    // Build descriptors against a given dst arena set. Identical per-job state
    // (a shuffled dst pal_map seeded by job) for A and B — only the dst arena
    // pointers and the launch grouping differ.
    let build = |dst_gpus: &[Vec<cudarc::driver::CudaSlice<u8>>]| -> Vec<PalHeadDesc> {
        (0..total_jobs)
            .map(|job| {
                let sp: [u64; N_PAL] =
                    std::array::from_fn(|p| src_gpus[job][p].device_ptr(&stream).0 as u64);
                let dp: [u64; N_PAL] =
                    std::array::from_fn(|p| dst_gpus[job][p].device_ptr(&stream).0 as u64);
                let dst_pal = shuffled_pal_map_128(job as u64 + 1);
                PalHeadDesc {
                    k_src_arena_ptrs: sp,
                    v_src_arena_ptrs: sp,
                    k_src_fmts: [GgmlDType::R16; N_PAL],
                    v_src_fmts: [GgmlDType::R16; N_PAL],
                    k_src_pal_map: ident,
                    v_src_pal_map: ident,
                    k_src_scales: [1.0f32; N_PAL],
                    v_src_scales: [1.0f32; N_PAL],
                    k_dst_arena_ptrs: dp,
                    v_dst_arena_ptrs: dp,
                    k_dst_fmts: [GgmlDType::R16; N_PAL],
                    v_dst_fmts: [GgmlDType::R16; N_PAL],
                    k_dst_pal_map: dst_pal,
                    v_dst_pal_map: dst_pal,
                    k_dst_scales: [1.0f32; N_PAL],
                    v_dst_scales: [1.0f32; N_PAL],
                }
            })
            .collect()
    };
    let descs_a = build(&dst_a_gpus);
    let descs_b = build(&dst_b_gpus);

    // Mode A: N per-chunk launches (num_layers=1), the old production path.
    for layer in 0..n_layers {
        let start = layer * num_kv_heads;
        let end = start + num_kv_heads;
        quantize_palette4_convert_buffered(
            &descs_a[start..end],
            num_kv_heads,
            1,
            num_chunks,
            &PinnedStager::new(cuda_dev).begin_generation(),
            &stream,
        )?;
    }
    // Mode B: one batched launch (num_layers=N), the new production path.
    quantize_palette4_convert_buffered(
        &descs_b,
        num_kv_heads,
        n_layers,
        num_chunks,
        &PinnedStager::new(cuda_dev).begin_generation(),
        &stream,
    )?;
    dev.synchronize()?;

    // Bit-identical dst bytes for every job/palette.
    for job in 0..total_jobs {
        for p in 0..N_PAL {
            let a = cuda_dev.memcpy_dtov(&dst_a_gpus[job][p])?;
            let b = cuda_dev.memcpy_dtov(&dst_b_gpus[job][p])?;
            assert_eq!(a, b, "batched != per-chunk at job={job} pal={p}");
        }
    }
    Ok(())
}

#[test]
fn palette4_convert_r16_identity_copy_check() -> Result<()> {
    // Basic correctness test: copy R16 palette chunks through the buffered API
    // with identity pal_maps and verify the output matches the source.
    let dev = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let cuda_dev = match &dev {
        Device::Cuda(d) => d,
        _ => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();

    let num_chunks = 1usize;
    let arena_size = PAL_DIM * num_chunks * R16_BLOCK_BYTES;

    let mut src_gpu = Vec::new();
    let mut dst_gpu = Vec::new();
    let mut src_cpu = Vec::new();
    for p in 0..N_PAL {
        let src = fill_r16_arena(PAL_DIM, num_chunks, (p as f32 + 1.0) * 1000.0);
        src_gpu.push(cuda_dev.memcpy_stod(&src)?);
        dst_gpu.push(cuda_dev.memcpy_stod(&vec![0u8; arena_size])?);
        src_cpu.push(src);
    }

    let mut src_arena_ptrs = [0u64; N_PAL];
    let mut dst_arena_ptrs = [0u64; N_PAL];
    for p in 0..N_PAL {
        src_arena_ptrs[p] = src_gpu[p].device_ptr(&stream).0 as u64;
        dst_arena_ptrs[p] = dst_gpu[p].device_ptr(&stream).0 as u64;
    }

    {
        use candle_core::quantized::cuda::{quantize_palette4_convert_buffered, PalHeadDesc};
        let ident = identity_pal_map();
        quantize_palette4_convert_buffered(
            &[PalHeadDesc {
                k_src_arena_ptrs: src_arena_ptrs,
                v_src_arena_ptrs: src_arena_ptrs,
                k_src_fmts: [GgmlDType::R16; N_PAL],
                v_src_fmts: [GgmlDType::R16; N_PAL],
                k_src_pal_map: ident,
                v_src_pal_map: ident,
                k_src_scales: [1.0f32; N_PAL],
                v_src_scales: [1.0f32; N_PAL],
                k_dst_arena_ptrs: dst_arena_ptrs,
                v_dst_arena_ptrs: dst_arena_ptrs,
                k_dst_fmts: [GgmlDType::R16; N_PAL],
                v_dst_fmts: [GgmlDType::R16; N_PAL],
                k_dst_pal_map: ident,
                v_dst_pal_map: ident,
                k_dst_scales: [1.0f32; N_PAL],
                v_dst_scales: [1.0f32; N_PAL],
            }],
            1,
            1,
            num_chunks,
            &PinnedStager::new(cuda_dev).begin_generation(),
            &stream,
        )?;
    }

    dev.synchronize()?;

    for p in 0..N_PAL {
        let out = cuda_dev.memcpy_dtov(&dst_gpu[p])?;
        let got = read_r16_arena(&out, PAL_DIM, num_chunks);
        let want = read_r16_arena(&src_cpu[p], PAL_DIM, num_chunks);
        for ld in 0..PAL_DIM {
            for c in 0..num_chunks {
                for t in 0..CHUNK_SIZE {
                    assert_eq!(
                        got[ld][c][t], want[ld][c][t],
                        "mismatch pal={p} dim={ld} chunk={c} token={t}"
                    );
                }
            }
        }
    }
    Ok(())
}

// =============================================================================
// EXTENDED HELPERS AND CONSTANTS
// =============================================================================

// ArenaFormat code reference table (not all entries are used in every test)
#[allow(dead_code)]
const FMT_F32: u8 = 0;
#[allow(dead_code)]
const FMT_F16: u8 = 1;
#[allow(dead_code)]
const FMT_BF16: u8 = 2;
#[allow(dead_code)]
const FMT_R16: u8 = 3;
#[allow(dead_code)]
const FMT_P2: u8 = 4;
#[allow(dead_code)]
const FMT_QAWQ: u8 = 5;
#[allow(dead_code)]
const FMT_QAWQ_G64: u8 = 6;
#[allow(dead_code)]
const FMT_Q8_0: u8 = 7;
#[allow(dead_code)]
const FMT_Q8_1: u8 = 8;
#[allow(dead_code)]
const FMT_Q8_K: u8 = 9;
#[allow(dead_code)]
const FMT_Q8_KS: u8 = 10;
#[allow(dead_code)]
const FMT_Q6_K: u8 = 11;
#[allow(dead_code)]
const FMT_Q5_0: u8 = 12;
#[allow(dead_code)]
const FMT_Q5_1: u8 = 13;
#[allow(dead_code)]
const FMT_Q5_K: u8 = 14;
#[allow(dead_code)]
const FMT_Q4_0: u8 = 15;
#[allow(dead_code)]
const FMT_Q4_1: u8 = 16;
#[allow(dead_code)]
const FMT_Q4_K: u8 = 17;
#[allow(dead_code)]
const FMT_Q4_KS: u8 = 18;
#[allow(dead_code)]
const FMT_Q3_0: u8 = 19;
#[allow(dead_code)]
const FMT_Q3_1: u8 = 20;
#[allow(dead_code)]
const FMT_Q3_K: u8 = 21;
#[allow(dead_code)]
const FMT_Q2_0: u8 = 22;
#[allow(dead_code)]
const FMT_Q2_1: u8 = 23;
#[allow(dead_code)]
const FMT_Q2_K: u8 = 24;
#[allow(dead_code)]
const FMT_Q2_S: u8 = 25;
#[allow(dead_code)]
const FMT_Q2_A: u8 = 26;
#[allow(dead_code)]
const FMT_Q1_S: u8 = 27;
#[allow(dead_code)]
const FMT_Q0_V: u8 = 28;
#[allow(dead_code)]
const FMT_Q1_A: u8 = 29;
#[allow(dead_code)]
const FMT_Q0_X: u8 = 30;
#[allow(dead_code)]
const FMT_Q0_M2: u8 = 31;
#[allow(dead_code)]
const FMT_Q0_M4: u8 = 32;
#[allow(dead_code)]
const FMT_Q0: u8 = 33;

/// All quant formats supported by the palette4 kernel (32-element blocks).
const ALL_QUANT_FMTS: &[(u8, &str, f32)] = &[
    (FMT_R16, "R16", 0.0), // lossless f16 store
    (FMT_Q8_0, "Q8_0", 0.012),
    (FMT_Q8_1, "Q8_1", 0.012),
    (FMT_Q8_KS, "Q8_KS", 0.012),
    (FMT_Q5_0, "Q5_0", 0.08),
    (FMT_Q5_1, "Q5_1", 0.08),
    (FMT_Q4_0, "Q4_0", 0.2),
    (FMT_Q4_1, "Q4_1", 0.2),
    (FMT_Q4_KS, "Q4_KS", 0.2),
    (FMT_Q3_0, "Q3_0", 0.4),
    (FMT_Q3_1, "Q3_1", 0.4),
    (FMT_Q2_0, "Q2_0", 0.85),
    (FMT_Q2_1, "Q2_1", 0.85),
    (FMT_Q2_S, "Q2_S", 0.85),
    (FMT_Q2_A, "Q2_A", 0.85),
    (FMT_Q1_S, "Q1_S", 1.0), // sign-only; check for crash + plausible sign
    (FMT_Q0, "Q0", 999.0),   // constant block; only verify kernel doesn't crash
    (FMT_Q0_V, "Q0_V", 999.0),
    (FMT_Q1_A, "Q1_A", 999.0),
    (FMT_Q0_X, "Q0_X", 999.0),
    (FMT_Q0_M2, "Q0_M2", 999.0),
    (FMT_Q0_M4, "Q0_M4", 999.0),
];

/// Block byte size for quant-format (token-oriented) arenas.
fn quant_block_bytes(fmt: u8) -> usize {
    match fmt {
        FMT_R16 => 128,
        FMT_Q4_0 => 18,
        FMT_Q4_1 => 20,
        FMT_Q5_0 => 22,
        FMT_Q5_1 => 24,
        FMT_Q8_0 => 34,
        FMT_Q8_1 => 36,
        FMT_Q4_KS => 20,
        FMT_Q8_KS => 36,
        FMT_Q2_0 => 10,
        FMT_Q3_0 => 14,
        FMT_Q0 => 1,
        FMT_Q1_S => 5,
        FMT_Q2_S => 9,
        FMT_Q2_A => 10,
        FMT_Q2_1 => 12,
        FMT_Q3_1 => 16,
        FMT_Q0_V => 2,
        FMT_Q1_A => 6,
        FMT_Q0_X => 2,
        FMT_Q0_M2 => 3,
        FMT_Q0_M4 => 8,
        _ => panic!("quant_block_bytes: unknown fmt {fmt}"),
    }
}

/// Float element byte size; 0 for quant formats.
fn float_elem_bytes(fmt: u8) -> usize {
    match fmt {
        FMT_F32 => 4,
        FMT_F16 => 2,
        FMT_BF16 => 2,
        _ => 0,
    }
}

/// Arena size in bytes for one palette of `num_dims` dims × `num_chunks` chunks.
fn arena_bytes(fmt: u8, num_dims: usize, num_chunks: usize) -> usize {
    let esz = float_elem_bytes(fmt);
    if esz > 0 {
        num_chunks * CHUNK_SIZE * num_dims * esz
    } else {
        num_dims * num_chunks * quant_block_bytes(fmt)
    }
}

/// Reversed-block pal_map: global dims 0..31→pal3, 32..63→pal2, 64..95→pal1, 96..127→pal0.
fn reversed_block_pal_map() -> [u8; HD / 4] {
    let mut out = [0u8; HD / 4];
    for d in 0..HD {
        let p = (3 - d / PAL_DIM) as u8;
        out[d / 4] |= (p & 0x3) << (2 * (d % 4));
    }
    out
}

/// Striped pal_map: dim d → palette d % 4.
fn striped_pal_map() -> [u8; HD / 4] {
    let mut out = [0u8; HD / 4];
    for d in 0..HD {
        let p = (d % N_PAL) as u8;
        out[d / 4] |= (p & 0x3) << (2 * (d % 4));
    }
    out
}

/// Rust mirror of CUDA pal_map_get.
fn pmg(pal_map: &[u8; HD / 4], d: usize) -> usize {
    ((pal_map[d / 4] >> (2 * (d % 4))) & 0x3) as usize
}

/// Rust mirror of CUDA find_nth_dim_in_pal.
fn nth_dim_in_pal(pal_map: &[u8; HD / 4], p: usize, n: usize) -> usize {
    let mut count = 0usize;
    for g in 0..HD {
        if pmg(pal_map, g) == p {
            if count == n {
                return g;
            }
            count += 1;
        }
    }
    panic!("nth_dim_in_pal: pal={p} n={n} not found");
}

/// Rust mirror of CUDA rank_in_pal.
fn rank_in_pal(pal_map: &[u8; HD / 4], p: usize, global_d: usize) -> usize {
    let mut rank = 0usize;
    for g in 0..global_d {
        if pmg(pal_map, g) == p {
            rank += 1;
        }
    }
    rank
}

/// Fill channel-oriented F16 arena: layout[c][t][ld] = base + (c*CHUNK+t) + ld*0.01
fn fill_f16_channel_arena(num_dims: usize, num_chunks: usize, base_val: f32) -> Vec<u8> {
    let mut data = vec![0u8; num_chunks * CHUNK_SIZE * num_dims * 2];
    for c in 0..num_chunks {
        for t in 0..CHUNK_SIZE {
            for ld in 0..num_dims {
                // Use values in [base, base+32+num_dims*0.01] - moderate range
                let val = base_val + (c * CHUNK_SIZE + t) as f32 * 0.01 + ld as f32 * 0.001;
                let h = f16::from_f32(val);
                let off = (c * CHUNK_SIZE * num_dims + t * num_dims + ld) * 2;
                data[off..off + 2].copy_from_slice(&h.to_le_bytes());
            }
        }
    }
    data
}

/// Read channel-oriented F16 arena → [dim][chunk][token]
fn read_f16_channel_arena(data: &[u8], num_dims: usize, num_chunks: usize) -> Vec<Vec<Vec<f32>>> {
    let mut out = vec![vec![vec![0.0f32; CHUNK_SIZE]; num_chunks]; num_dims];
    for c in 0..num_chunks {
        for t in 0..CHUNK_SIZE {
            for ld in 0..num_dims {
                let off = (c * CHUNK_SIZE * num_dims + t * num_dims + ld) * 2;
                let h = f16::from_le_bytes([data[off], data[off + 1]]);
                out[ld][c][t] = h.to_f32();
            }
        }
    }
    out
}

/// Fill channel-oriented F32 arena
fn fill_f32_channel_arena(num_dims: usize, num_chunks: usize, base_val: f32) -> Vec<u8> {
    let mut data = vec![0u8; num_chunks * CHUNK_SIZE * num_dims * 4];
    for c in 0..num_chunks {
        for t in 0..CHUNK_SIZE {
            for ld in 0..num_dims {
                let val = base_val + (c * CHUNK_SIZE + t) as f32 * 0.01 + ld as f32 * 0.001;
                let off = (c * CHUNK_SIZE * num_dims + t * num_dims + ld) * 4;
                data[off..off + 4].copy_from_slice(&val.to_le_bytes());
            }
        }
    }
    data
}

/// Fill channel-oriented BF16 arena
fn fill_bf16_channel_arena(num_dims: usize, num_chunks: usize, base_val: f32) -> Vec<u8> {
    let mut data = vec![0u8; num_chunks * CHUNK_SIZE * num_dims * 2];
    for c in 0..num_chunks {
        for t in 0..CHUNK_SIZE {
            for ld in 0..num_dims {
                let val = base_val + (c * CHUNK_SIZE + t) as f32 * 0.01 + ld as f32 * 0.001;
                let h = bf16::from_f32(val);
                let off = (c * CHUNK_SIZE * num_dims + t * num_dims + ld) * 2;
                data[off..off + 2].copy_from_slice(&h.to_le_bytes());
            }
        }
    }
    data
}

/// Read channel-oriented F32 arena → [dim][chunk][token]
fn read_f32_channel_arena(data: &[u8], num_dims: usize, num_chunks: usize) -> Vec<Vec<Vec<f32>>> {
    let mut out = vec![vec![vec![0.0f32; CHUNK_SIZE]; num_chunks]; num_dims];
    for c in 0..num_chunks {
        for t in 0..CHUNK_SIZE {
            for ld in 0..num_dims {
                let off = (c * CHUNK_SIZE * num_dims + t * num_dims + ld) * 4;
                out[ld][c][t] =
                    f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
            }
        }
    }
    out
}

// =============================================================================
// PALETTE MAP TESTS
// =============================================================================

/// Run a single convert pass through `quantize_palette4_convert_buffered`.
/// Both K and V channels use the same arenas so both execute; callers only verify K output.
/// The `_is_k` parameter is retained for call-site compatibility but is otherwise unused.
fn run_kernel_pass(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    dev: &Device,
    src_arenas: &[cudarc::driver::CudaSlice<u8>],
    dst_arenas: &[cudarc::driver::CudaSlice<u8>],
    src_fmts: [u8; N_PAL],
    dst_fmts: [u8; N_PAL],
    src_pal_map: &[u8; HD / 4],
    dst_pal_map: &[u8; HD / 4],
    num_chunks: usize,
    _is_k: bool,
) -> Result<()> {
    use candle_core::quantized::cuda::{quantize_palette4_convert_buffered, PalHeadDesc};
    // `cuda_dev` derived from the live `Device` so the `PinnedStager`
    // construction below still has its CudaDevice handle.
    let cuda_dev = match dev {
        Device::Cuda(d) => d,
        _ => unreachable!("run_kernel_pass requires a CUDA device"),
    };
    let src_ptrs: [u64; N_PAL] = std::array::from_fn(|p| src_arenas[p].device_ptr(stream).0 as u64);
    let dst_ptrs: [u64; N_PAL] = std::array::from_fn(|p| dst_arenas[p].device_ptr(stream).0 as u64);
    let sg: [GgmlDType; N_PAL] = std::array::from_fn(|i| fmt_code_to_ggml_dtype(src_fmts[i]));
    let dg: [GgmlDType; N_PAL] = std::array::from_fn(|i| fmt_code_to_ggml_dtype(dst_fmts[i]));
    let desc = PalHeadDesc {
        k_src_arena_ptrs: src_ptrs,
        v_src_arena_ptrs: src_ptrs,
        k_src_fmts: sg,
        v_src_fmts: sg,
        k_src_pal_map: *src_pal_map,
        v_src_pal_map: *src_pal_map,
        k_src_scales: [1.0f32; N_PAL],
        v_src_scales: [1.0f32; N_PAL],
        k_dst_arena_ptrs: dst_ptrs,
        v_dst_arena_ptrs: dst_ptrs,
        k_dst_fmts: dg,
        v_dst_fmts: dg,
        k_dst_pal_map: *dst_pal_map,
        v_dst_pal_map: *dst_pal_map,
        k_dst_scales: [1.0f32; N_PAL],
        v_dst_scales: [1.0f32; N_PAL],
    };
    quantize_palette4_convert_buffered(
        &[desc],
        1,
        1,
        num_chunks,
        &PinnedStager::new(cuda_dev).begin_generation(),
        stream,
    )?;
    dev.synchronize()
}

#[test]
fn palette4_convert_uniform_to_reversed_block_pal() -> Result<()> {
    // src: identity pal_map (pal0=dims0..31, pal1=dims32..63, etc.)
    // dst: reversed-block pal_map (pal0=dims96..127, pal1=dims64..95, etc.)
    // Expected: dst_pal[p][ld] contains the value from src_pal[3-p][ld]
    let dev = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let cuda_dev = match &dev {
        Device::Cuda(d) => d,
        _ => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();
    let num_chunks = 2usize;
    let arena_sz = PAL_DIM * num_chunks * R16_BLOCK_BYTES;

    let mut src_gpu = Vec::new();
    let mut dst_gpu = Vec::new();
    let mut src_cpu = Vec::new();
    for p in 0..N_PAL {
        let data = fill_r16_arena(PAL_DIM, num_chunks, (p as f32 + 1.0) * 100.0);
        src_gpu.push(cuda_dev.memcpy_stod(&data)?);
        dst_gpu.push(cuda_dev.memcpy_stod(&vec![0u8; arena_sz])?);
        src_cpu.push(data);
    }

    let ident = identity_pal_map();
    let rev = reversed_block_pal_map();
    run_kernel_pass(
        &stream,
        &dev,
        &src_gpu,
        &dst_gpu,
        [FMT_R16; N_PAL],
        [FMT_R16; N_PAL],
        &ident,
        &rev,
        num_chunks,
        true,
    )?;

    for dst_p in 0..N_PAL {
        let src_p = 3 - dst_p;
        let dst_data = cuda_dev.memcpy_dtov(&dst_gpu[dst_p])?;
        let dst_vals = read_r16_arena(&dst_data, PAL_DIM, num_chunks);
        let src_vals = read_r16_arena(&src_cpu[src_p], PAL_DIM, num_chunks);
        for ld in 0..PAL_DIM {
            for c in 0..num_chunks {
                for t in 0..CHUNK_SIZE {
                    assert_eq!(
                        dst_vals[ld][c][t], src_vals[ld][c][t],
                        "uniform→reversed: dst_pal={dst_p} src_pal={src_p} ld={ld} c={c} t={t}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn palette4_convert_reversed_block_pal_to_uniform() -> Result<()> {
    // src: reversed-block pal_map, dst: identity pal_map.
    // Expected mapping is the same formula: dst_pal[p][ld] == src_pal[3-p][ld].
    let dev = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let cuda_dev = match &dev {
        Device::Cuda(d) => d,
        _ => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();
    let num_chunks = 2usize;
    let arena_sz = PAL_DIM * num_chunks * R16_BLOCK_BYTES;

    let mut src_gpu = Vec::new();
    let mut dst_gpu = Vec::new();
    let mut src_cpu = Vec::new();
    for p in 0..N_PAL {
        let data = fill_r16_arena(PAL_DIM, num_chunks, (p as f32 + 1.0) * 100.0);
        src_gpu.push(cuda_dev.memcpy_stod(&data)?);
        dst_gpu.push(cuda_dev.memcpy_stod(&vec![0u8; arena_sz])?);
        src_cpu.push(data);
    }

    let ident = identity_pal_map();
    let rev = reversed_block_pal_map();
    run_kernel_pass(
        &stream,
        &dev,
        &src_gpu,
        &dst_gpu,
        [FMT_R16; N_PAL],
        [FMT_R16; N_PAL],
        &rev,
        &ident,
        num_chunks,
        true,
    )?;

    for dst_p in 0..N_PAL {
        let src_p = 3 - dst_p;
        let dst_data = cuda_dev.memcpy_dtov(&dst_gpu[dst_p])?;
        let dst_vals = read_r16_arena(&dst_data, PAL_DIM, num_chunks);
        let src_vals = read_r16_arena(&src_cpu[src_p], PAL_DIM, num_chunks);
        for ld in 0..PAL_DIM {
            for c in 0..num_chunks {
                for t in 0..CHUNK_SIZE {
                    assert_eq!(
                        dst_vals[ld][c][t], src_vals[ld][c][t],
                        "reversed→uniform: dst_pal={dst_p} src_pal={src_p} ld={ld} c={c} t={t}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn palette4_convert_uniform_to_uniform_value_verification() -> Result<()> {
    // Identity→identity: verify that specific global dim values reach the expected pal/local_d.
    // At identity map: global dim d → pal = d/32, local_d = d%32.
    // So dst_pal[p] local_dim[ld] == src_pal[p] local_dim[ld] for all p,ld.
    // We write unique per-(pal,dim) values and verify they're in the right place.
    let dev = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let cuda_dev = match &dev {
        Device::Cuda(d) => d,
        _ => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();
    let num_chunks = 1usize;
    let arena_sz = PAL_DIM * num_chunks * R16_BLOCK_BYTES;

    let mut src_gpu = Vec::new();
    let mut dst_gpu = Vec::new();
    let mut src_cpu = Vec::new();
    for p in 0..N_PAL {
        // base_val chosen so pal p has values in a unique range: p*1000
        let data = fill_r16_arena(PAL_DIM, num_chunks, p as f32 * 1000.0);
        src_gpu.push(cuda_dev.memcpy_stod(&data)?);
        dst_gpu.push(cuda_dev.memcpy_stod(&vec![0u8; arena_sz])?);
        src_cpu.push(data);
    }

    let ident = identity_pal_map();
    run_kernel_pass(
        &stream,
        &dev,
        &src_gpu,
        &dst_gpu,
        [FMT_R16; N_PAL],
        [FMT_R16; N_PAL],
        &ident,
        &ident,
        num_chunks,
        true,
    )?;

    for p in 0..N_PAL {
        let dst_data = cuda_dev.memcpy_dtov(&dst_gpu[p])?;
        let dst_vals = read_r16_arena(&dst_data, PAL_DIM, num_chunks);
        let src_vals = read_r16_arena(&src_cpu[p], PAL_DIM, num_chunks);
        for ld in 0..PAL_DIM {
            for c in 0..num_chunks {
                for t in 0..CHUNK_SIZE {
                    assert_eq!(
                        dst_vals[ld][c][t], src_vals[ld][c][t],
                        "identity→identity: pal={p} ld={ld} c={c} t={t}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn palette4_convert_striped_identity_roundtrip() -> Result<()> {
    // identity→striped(pass1)→identity(pass2) should be exact roundtrip.
    let dev = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let cuda_dev = match &dev {
        Device::Cuda(d) => d,
        _ => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();
    let num_chunks = 2usize;
    let r16_sz = PAL_DIM * num_chunks * R16_BLOCK_BYTES;
    // Striped: each palette has 32 dims but non-contiguous.
    // For striped pal_map, dims 0,4,8,...,124 => pal0, etc. (each pal gets 32 dims)
    // Arena size is the same: 32 dims, 2 chunks, R16 128 bytes/block.

    let mut src_gpu = Vec::new();
    let mut mid_gpu = Vec::new();
    let mut dst_gpu = Vec::new();
    let mut src_cpu = Vec::new();
    for p in 0..N_PAL {
        let data = fill_r16_arena(PAL_DIM, num_chunks, (p as f32 + 1.0) * 500.0);
        src_gpu.push(cuda_dev.memcpy_stod(&data)?);
        mid_gpu.push(cuda_dev.memcpy_stod(&vec![0u8; r16_sz])?);
        dst_gpu.push(cuda_dev.memcpy_stod(&vec![0u8; r16_sz])?);
        src_cpu.push(data);
    }

    let ident = identity_pal_map();
    let striped = striped_pal_map();

    // Pass 1: identity → striped
    run_kernel_pass(
        &stream,
        &dev,
        &src_gpu,
        &mid_gpu,
        [FMT_R16; N_PAL],
        [FMT_R16; N_PAL],
        &ident,
        &striped,
        num_chunks,
        true,
    )?;
    // Pass 2: striped → identity
    run_kernel_pass(
        &stream,
        &dev,
        &mid_gpu,
        &dst_gpu,
        [FMT_R16; N_PAL],
        [FMT_R16; N_PAL],
        &striped,
        &ident,
        num_chunks,
        true,
    )?;

    for p in 0..N_PAL {
        let dst_data = cuda_dev.memcpy_dtov(&dst_gpu[p])?;
        let dst_vals = read_r16_arena(&dst_data, PAL_DIM, num_chunks);
        let src_vals = read_r16_arena(&src_cpu[p], PAL_DIM, num_chunks);
        for ld in 0..PAL_DIM {
            for c in 0..num_chunks {
                for t in 0..CHUNK_SIZE {
                    assert_eq!(
                        dst_vals[ld][c][t], src_vals[ld][c][t],
                        "striped roundtrip: pal={p} ld={ld} c={c} t={t}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn palette4_convert_nonuniform_to_nonuniform_striped_to_reversed() -> Result<()> {
    // src: striped pal_map, dst: reversed-block pal_map.
    // Verify expected values using Rust reference mapping helpers.
    let dev = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let cuda_dev = match &dev {
        Device::Cuda(d) => d,
        _ => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();
    let num_chunks = 1usize;
    let r16_sz = PAL_DIM * num_chunks * R16_BLOCK_BYTES;

    let striped = striped_pal_map();
    let rev = reversed_block_pal_map();

    // Fill src with unique per-(pal, local_dim, chunk, token) values
    let mut src_gpu = Vec::new();
    let mut dst_gpu = Vec::new();
    let mut src_cpu = Vec::new();
    for p in 0..N_PAL {
        // each pal gets a distinct base so we can uniquely identify values
        let data = fill_r16_arena(PAL_DIM, num_chunks, (p as f32 + 1.0) * 100.0);
        src_gpu.push(cuda_dev.memcpy_stod(&data)?);
        dst_gpu.push(cuda_dev.memcpy_stod(&vec![0u8; r16_sz])?);
        src_cpu.push(read_r16_arena(&data, PAL_DIM, num_chunks));
    }

    run_kernel_pass(
        &stream,
        &dev,
        &src_gpu,
        &dst_gpu,
        [FMT_R16; N_PAL],
        [FMT_R16; N_PAL],
        &striped,
        &rev,
        num_chunks,
        true,
    )?;

    // Compute expected values using Rust reference mapping.
    // For each dst (p, ld): global_d = nth_dim_in_pal(rev, p, ld)
    //   sp = pmg(striped, global_d) = global_d % 4
    //   s_local_d = rank_in_pal(striped, sp, global_d) = global_d / 4
    for dst_p in 0..N_PAL {
        let dst_data = cuda_dev.memcpy_dtov(&dst_gpu[dst_p])?;
        let dst_vals = read_r16_arena(&dst_data, PAL_DIM, num_chunks);
        for ld in 0..PAL_DIM {
            let global_d = nth_dim_in_pal(&rev, dst_p, ld);
            let sp = pmg(&striped, global_d);
            let s_local_d = rank_in_pal(&striped, sp, global_d);
            for c in 0..num_chunks {
                for t in 0..CHUNK_SIZE {
                    let expected = src_cpu[sp][s_local_d][c][t];
                    let got = dst_vals[ld][c][t];
                    assert_eq!(got, expected,
                        "striped→reversed: dst_p={dst_p} ld={ld} global_d={global_d} sp={sp} s_local_d={s_local_d} c={c} t={t}");
                }
            }
        }
    }
    Ok(())
}

// =============================================================================
// FORMAT ROUNDTRIP TESTS (F16 and R16 → every quant type → R16)
// =============================================================================

/// Diagnostic: single-format two-pass sweep using same architecture as multi_format_sweep.
/// Isolates whether the 17-head batching itself is the issue.
#[test]
fn palette4_convert_f16_to_r16_single_fmt_sweep_diag() -> Result<()> {
    let dev = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let cuda_dev = match &dev {
        Device::Cuda(d) => d,
        _ => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();
    let num_chunks = 1usize;
    let ident = identity_pal_map();
    let stream = cuda_dev.cuda_stream();
    let r16_sz = arena_bytes(FMT_R16, PAL_DIM, num_chunks);

    // Upload src (F16)
    let src_cpu: Vec<Vec<u8>> = (0..N_PAL)
        .map(|p| fill_f16_channel_arena(PAL_DIM, num_chunks, 1.0 + p as f32 * 0.05))
        .collect();
    let src_gpu: Vec<_> = src_cpu
        .iter()
        .map(|d| cuda_dev.memcpy_stod(d).unwrap())
        .collect();
    let src_ptrs_shared: [u64; N_PAL] =
        std::array::from_fn(|p| src_gpu[p].device_ptr(&stream).0 as u64);

    // Allocate mid (R16) and out (R16) for 1 format
    let mid_sz = arena_bytes(FMT_R16, PAL_DIM, num_chunks);
    let mid_gpu: Vec<cudarc::driver::CudaSlice<u8>> = (0..N_PAL)
        .map(|_| cuda_dev.memcpy_stod(&vec![0u8; mid_sz]).unwrap())
        .collect();
    let out_gpu: Vec<cudarc::driver::CudaSlice<u8>> = (0..N_PAL)
        .map(|_| cuda_dev.memcpy_stod(&vec![0u8; r16_sz]).unwrap())
        .collect();
    let mid_ptrs: [u64; N_PAL] = std::array::from_fn(|p| mid_gpu[p].device_ptr(&stream).0 as u64);
    let out_ptrs: [u64; N_PAL] = std::array::from_fn(|p| out_gpu[p].device_ptr(&stream).0 as u64);

    use candle_core::quantized::cuda::{quantize_palette4_convert_buffered, PalHeadDesc};
    eprintln!("[diag] pass 1 F16→R16");
    quantize_palette4_convert_buffered(
        &[PalHeadDesc {
            k_src_arena_ptrs: src_ptrs_shared,
            v_src_arena_ptrs: src_ptrs_shared,
            k_src_fmts: [GgmlDType::F16; N_PAL],
            v_src_fmts: [GgmlDType::F16; N_PAL],
            k_src_pal_map: ident,
            v_src_pal_map: ident,
            k_src_scales: [1.0f32; N_PAL],
            v_src_scales: [1.0f32; N_PAL],
            k_dst_arena_ptrs: mid_ptrs,
            v_dst_arena_ptrs: mid_ptrs,
            k_dst_fmts: [GgmlDType::R16; N_PAL],
            v_dst_fmts: [GgmlDType::R16; N_PAL],
            k_dst_pal_map: ident,
            v_dst_pal_map: ident,
            k_dst_scales: [1.0f32; N_PAL],
            v_dst_scales: [1.0f32; N_PAL],
        }],
        1,
        1,
        num_chunks,
        &PinnedStager::new(cuda_dev).begin_generation(),
        &stream,
    )?;
    dev.synchronize()?;
    eprintln!("[diag] pass 2 R16→R16");
    quantize_palette4_convert_buffered(
        &[PalHeadDesc {
            k_src_arena_ptrs: mid_ptrs,
            v_src_arena_ptrs: mid_ptrs,
            k_src_fmts: [GgmlDType::R16; N_PAL],
            v_src_fmts: [GgmlDType::R16; N_PAL],
            k_src_pal_map: ident,
            v_src_pal_map: ident,
            k_src_scales: [1.0f32; N_PAL],
            v_src_scales: [1.0f32; N_PAL],
            k_dst_arena_ptrs: out_ptrs,
            v_dst_arena_ptrs: out_ptrs,
            k_dst_fmts: [GgmlDType::R16; N_PAL],
            v_dst_fmts: [GgmlDType::R16; N_PAL],
            k_dst_pal_map: ident,
            v_dst_pal_map: ident,
            k_dst_scales: [1.0f32; N_PAL],
            v_dst_scales: [1.0f32; N_PAL],
        }],
        1,
        1,
        num_chunks,
        &PinnedStager::new(cuda_dev).begin_generation(),
        &stream,
    )?;
    dev.synchronize()?;
    eprintln!("[diag] done");

    // Verify
    for p in 0..N_PAL {
        let data = cuda_dev.memcpy_dtov(&out_gpu[p])?;
        let got = read_r16_arena(&data, PAL_DIM, num_chunks);
        let ref_f16 = read_f16_channel_arena(&src_cpu[p], PAL_DIM, num_chunks);
        for ld in 0..PAL_DIM {
            for t in 0..CHUNK_SIZE {
                let expected = ref_f16[ld][0][t];
                let got_v = got[ld][0][t];
                assert!(
                    (got_v - expected).abs() < 1e-3,
                    "pal={p} ld={ld} t={t} expected={expected} got={got_v}"
                );
            }
        }
    }
    Ok(())
}

/// Two-pass roundtrip: src_fmt arenas → mid_fmt arenas → R16 arenas.
/// Returns R16 output values [pal][dim][chunk][token].
fn multi_format_sweep(
    cuda_dev: &candle_core::CudaDevice,
    dev: &Device,
    src_fmt: u8,
    // Shared src arenas [N_PAL]: same data reused for all format heads (src is read-only).
    src_arenas_cpu: &[Vec<u8>],
    num_chunks: usize,
) -> Result<Vec<Vec<Vec<Vec<Vec<f32>>>>>> {
    // Returns [n_fmts][N_PAL][dim][chunk][token] R16 values.
    // Two calls to quantize_palette4_convert_buffered: each with N_FMTS heads.
    let n_fmts = ALL_QUANT_FMTS.len();
    use candle_core::quantized::cuda::{quantize_palette4_convert_buffered, PalHeadDesc};
    let ident = identity_pal_map();
    let stream = cuda_dev.cuda_stream();
    let r16_sz = arena_bytes(FMT_R16, PAL_DIM, num_chunks);

    // Upload shared src arenas once.
    let mut src_gpu = Vec::new();
    for p in 0..N_PAL {
        assert_eq!(
            src_arenas_cpu[p].len(),
            arena_bytes(src_fmt, PAL_DIM, num_chunks)
        );
        src_gpu.push(cuda_dev.memcpy_stod(&src_arenas_cpu[p])?);
    }
    let src_ptrs_shared: [u64; N_PAL] =
        std::array::from_fn(|p| src_gpu[p].device_ptr(&stream).0 as u64);

    // Per-format mid and out arenas.
    let mut mid_gpu: Vec<Vec<cudarc::driver::CudaSlice<u8>>> = Vec::new();
    let mut out_gpu: Vec<Vec<cudarc::driver::CudaSlice<u8>>> = Vec::new();
    for &(mid_fmt, _, _) in ALL_QUANT_FMTS {
        let mid_sz = arena_bytes(mid_fmt, PAL_DIM, num_chunks);
        let mut mid_row = Vec::new();
        let mut out_row = Vec::new();
        for _ in 0..N_PAL {
            mid_row.push(cuda_dev.memcpy_stod(&vec![0u8; mid_sz])?);
            out_row.push(cuda_dev.memcpy_stod(&vec![0u8; r16_sz])?);
        }
        mid_gpu.push(mid_row);
        out_gpu.push(out_row);
    }

    // Compute per-format ptr arrays.
    let mid_ptrs: Vec<[u64; N_PAL]> = (0..n_fmts)
        .map(|fi| std::array::from_fn(|p| mid_gpu[fi][p].device_ptr(&stream).0 as u64))
        .collect();
    let out_ptrs: Vec<[u64; N_PAL]> = (0..n_fmts)
        .map(|fi| std::array::from_fn(|p| out_gpu[fi][p].device_ptr(&stream).0 as u64))
        .collect();

    // Build one PalHeadDesc per format for each pass then issue 2 batched calls.
    let src_gdtype = fmt_code_to_ggml_dtype(src_fmt);
    let mut p1_descs: Vec<PalHeadDesc> = Vec::new();
    let mut p2_descs: Vec<PalHeadDesc> = Vec::new();
    for (fi, &(mid_fmt, _, _)) in ALL_QUANT_FMTS.iter().enumerate() {
        let mid_gdtype = fmt_code_to_ggml_dtype(mid_fmt);
        let mid_gdtypes = [mid_gdtype; N_PAL];
        p1_descs.push(PalHeadDesc {
            k_src_arena_ptrs: src_ptrs_shared,
            v_src_arena_ptrs: src_ptrs_shared,
            k_src_fmts: [src_gdtype; N_PAL],
            v_src_fmts: [src_gdtype; N_PAL],
            k_src_pal_map: ident,
            v_src_pal_map: ident,
            k_src_scales: [1.0f32; N_PAL],
            v_src_scales: [1.0f32; N_PAL],
            k_dst_arena_ptrs: mid_ptrs[fi],
            v_dst_arena_ptrs: mid_ptrs[fi],
            k_dst_fmts: mid_gdtypes,
            v_dst_fmts: mid_gdtypes,
            k_dst_pal_map: ident,
            v_dst_pal_map: ident,
            k_dst_scales: [1.0f32; N_PAL],
            v_dst_scales: [1.0f32; N_PAL],
        });
        p2_descs.push(PalHeadDesc {
            k_src_arena_ptrs: mid_ptrs[fi],
            v_src_arena_ptrs: mid_ptrs[fi],
            k_src_fmts: mid_gdtypes,
            v_src_fmts: mid_gdtypes,
            k_src_pal_map: ident,
            v_src_pal_map: ident,
            k_src_scales: [1.0f32; N_PAL],
            v_src_scales: [1.0f32; N_PAL],
            k_dst_arena_ptrs: out_ptrs[fi],
            v_dst_arena_ptrs: out_ptrs[fi],
            k_dst_fmts: [GgmlDType::R16; N_PAL],
            v_dst_fmts: [GgmlDType::R16; N_PAL],
            k_dst_pal_map: ident,
            v_dst_pal_map: ident,
            k_dst_scales: [1.0f32; N_PAL],
            v_dst_scales: [1.0f32; N_PAL],
        });
    }

    // Two kernel calls covering all N_FMTS heads in parallel.
    quantize_palette4_convert_buffered(
        &p1_descs,
        n_fmts,
        1,
        num_chunks,
        &PinnedStager::new(cuda_dev).begin_generation(),
        &stream,
    )?;
    dev.synchronize()?;
    quantize_palette4_convert_buffered(
        &p2_descs,
        n_fmts,
        1,
        num_chunks,
        &PinnedStager::new(cuda_dev).begin_generation(),
        &stream,
    )?;
    dev.synchronize()?;

    let mut results = Vec::new();
    for fi in 0..n_fmts {
        let mut fmt_result = Vec::new();
        for p in 0..N_PAL {
            let data = cuda_dev.memcpy_dtov(&out_gpu[fi][p])?;
            fmt_result.push(read_r16_arena(&data, PAL_DIM, num_chunks));
        }
        results.push(fmt_result);
    }
    Ok(results)
}

#[test]
fn palette4_convert_f16_to_all_quant_formats() -> Result<()> {
    // F16 → every quant format → R16 round-trip, all 17 formats in 2 kernel launches.
    let dev = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let cuda_dev = match &dev {
        Device::Cuda(d) => d,
        _ => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();
    let num_chunks = 1usize;

    // Values in [1.0, 1.032]: stable through all quantizers including 2-bit.
    let src_cpu: Vec<Vec<u8>> = (0..N_PAL)
        .map(|p| fill_f16_channel_arena(PAL_DIM, num_chunks, 1.0 + p as f32 * 0.05))
        .collect();
    let f16_ref: Vec<Vec<Vec<Vec<f32>>>> = (0..N_PAL)
        .map(|p| read_f16_channel_arena(&src_cpu[p], PAL_DIM, num_chunks))
        .collect();

    let results = multi_format_sweep(cuda_dev, &dev, FMT_F16, &src_cpu, num_chunks)?;

    for (fi, &(_, name, tol)) in ALL_QUANT_FMTS.iter().enumerate() {
        if tol >= 999.0 {
            continue;
        } // Q0: just verifying no crash
        for p in 0..N_PAL {
            for ld in 0..PAL_DIM {
                for t in 0..CHUNK_SIZE {
                    let expected = f16_ref[p][ld][0][t];
                    let got = results[fi][p][ld][0][t];
                    if tol >= 1.0 {
                        assert_eq!(got.signum(), expected.signum(),
                            "F16→{name}→R16: sign pal={p} ld={ld} t={t} expected={expected} got={got}");
                    } else {
                        let rel_err = (got - expected).abs() / (expected.abs().max(1e-6));
                        assert!(rel_err <= tol,
                            "F16→{name}→R16: rel_err={rel_err:.4} > {tol} pal={p} ld={ld} t={t} expected={expected} got={got}");
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn palette4_convert_r16_to_all_quant_formats() -> Result<()> {
    // R16 → every quant format → R16 round-trip, all formats in 2 kernel launches.
    let dev = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let cuda_dev = match &dev {
        Device::Cuda(d) => d,
        _ => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();
    let num_chunks = 1usize;

    let src_cpu: Vec<Vec<u8>> = (0..N_PAL)
        .map(|p| fill_r16_arena_small(PAL_DIM, num_chunks, 1.0 + p as f32 * 0.05))
        .collect();
    let r16_ref: Vec<Vec<Vec<Vec<f32>>>> = (0..N_PAL)
        .map(|p| read_r16_arena(&src_cpu[p], PAL_DIM, num_chunks))
        .collect();

    let results = multi_format_sweep(cuda_dev, &dev, FMT_R16, &src_cpu, num_chunks)?;

    for (fi, &(_, name, tol)) in ALL_QUANT_FMTS.iter().enumerate() {
        if tol >= 999.0 {
            continue;
        }
        for p in 0..N_PAL {
            for ld in 0..PAL_DIM {
                for t in 0..CHUNK_SIZE {
                    let expected = r16_ref[p][ld][0][t];
                    let got = results[fi][p][ld][0][t];
                    if tol >= 1.0 {
                        assert_eq!(got.signum(), expected.signum(),
                            "R16→{name}→R16: sign pal={p} ld={ld} t={t} expected={expected} got={got}");
                    } else {
                        let rel_err = (got - expected).abs() / (expected.abs().max(1e-6));
                        assert!(rel_err <= tol,
                            "R16→{name}→R16: rel_err={rel_err:.4} > {tol} pal={p} ld={ld} t={t} expected={expected} got={got}");
                    }
                }
            }
        }
    }
    Ok(())
}

/// fill_r16_arena variant with smaller increments so values stay in [base, base+~0.064]
fn fill_r16_arena_small(num_dims: usize, num_chunks: usize, base_val: f32) -> Vec<u8> {
    let total_blocks = num_dims * num_chunks;
    let mut data = vec![0u8; total_blocks * R16_BLOCK_BYTES];
    for ld in 0..num_dims {
        for c in 0..num_chunks {
            let blk_off = (ld * num_chunks + c) * R16_BLOCK_BYTES;
            for t in 0..CHUNK_SIZE {
                let val = base_val + (c * CHUNK_SIZE + t) as f32 * 0.001 + ld as f32 * 0.0001;
                let h = f16::from_f32(val);
                let off = blk_off + t * 2;
                data[off..off + 2].copy_from_slice(&h.to_le_bytes());
            }
        }
    }
    data
}

// =============================================================================
// FLOAT SOURCE/DESTINATION TYPE TESTS
// =============================================================================

#[test]
fn palette4_convert_f32_source_to_r16() -> Result<()> {
    // F32 channel-oriented src → R16 token-oriented dst.
    let dev = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let cuda_dev = match &dev {
        Device::Cuda(d) => d,
        _ => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();
    let num_chunks = 1usize;
    let r16_sz = arena_bytes(FMT_R16, PAL_DIM, num_chunks);
    let ident = identity_pal_map();

    let mut src_gpu = Vec::new();
    let mut dst_gpu = Vec::new();
    let mut src_cpu = Vec::new();
    for p in 0..N_PAL {
        let data = fill_f32_channel_arena(PAL_DIM, num_chunks, 1.0 + p as f32 * 0.5);
        src_gpu.push(cuda_dev.memcpy_stod(&data)?);
        dst_gpu.push(cuda_dev.memcpy_stod(&vec![0u8; r16_sz])?);
        src_cpu.push(data);
    }

    run_kernel_pass(
        &stream,
        &dev,
        &src_gpu,
        &dst_gpu,
        [FMT_F32; N_PAL],
        [FMT_R16; N_PAL],
        &ident,
        &ident,
        num_chunks,
        true,
    )?;

    for p in 0..N_PAL {
        let dst_data = cuda_dev.memcpy_dtov(&dst_gpu[p])?;
        let dst_vals = read_r16_arena(&dst_data, PAL_DIM, num_chunks);
        let src_vals = read_f32_channel_arena(&src_cpu[p], PAL_DIM, num_chunks);
        for ld in 0..PAL_DIM {
            for c in 0..num_chunks {
                for t in 0..CHUNK_SIZE {
                    let expected = f16::from_f32(src_vals[ld][c][t]).to_f32();
                    let got = dst_vals[ld][c][t];
                    // F32→f16 staging→R16: at most 1 f16 rounding step
                    let err = (got - expected).abs();
                    assert!(
                        err < 1e-3,
                        "F32→R16: pal={p} ld={ld} c={c} t={t} expected={expected} got={got}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn palette4_convert_bf16_source_to_r16() -> Result<()> {
    // BF16 channel-oriented src → R16.
    let dev = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let cuda_dev = match &dev {
        Device::Cuda(d) => d,
        _ => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();
    let num_chunks = 1usize;
    let r16_sz = arena_bytes(FMT_R16, PAL_DIM, num_chunks);
    let ident = identity_pal_map();

    let mut src_gpu = Vec::new();
    let mut dst_gpu = Vec::new();
    let mut src_cpu = Vec::new();
    for p in 0..N_PAL {
        let data = fill_bf16_channel_arena(PAL_DIM, num_chunks, 1.0 + p as f32 * 0.5);
        src_gpu.push(cuda_dev.memcpy_stod(&data)?);
        dst_gpu.push(cuda_dev.memcpy_stod(&vec![0u8; r16_sz])?);
        src_cpu.push(data);
    }

    run_kernel_pass(
        &stream,
        &dev,
        &src_gpu,
        &dst_gpu,
        [FMT_BF16; N_PAL],
        [FMT_R16; N_PAL],
        &ident,
        &ident,
        num_chunks,
        true,
    )?;

    for p in 0..N_PAL {
        let dst_data = cuda_dev.memcpy_dtov(&dst_gpu[p])?;
        let dst_vals = read_r16_arena(&dst_data, PAL_DIM, num_chunks);
        for ld in 0..PAL_DIM {
            for c in 0..num_chunks {
                for t in 0..CHUNK_SIZE {
                    let off = (c * CHUNK_SIZE * PAL_DIM + t * PAL_DIM + ld) * 2;
                    let src_bf = bf16::from_le_bytes([src_cpu[p][off], src_cpu[p][off + 1]]);
                    // BF16 → f32 → f16 staging → R16: tolerance for bf16→f16 precision loss
                    let expected_f32 = src_bf.to_f32();
                    let got = dst_vals[ld][c][t];
                    let rel_err = (got - expected_f32).abs() / (expected_f32.abs().max(1e-6));
                    assert!(
                        rel_err < 0.01,
                        "BF16→R16: pal={p} ld={ld} c={c} t={t} expected={expected_f32} got={got}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn palette4_convert_r16_to_f32_dst() -> Result<()> {
    // R16 token-oriented src → F32 channel-oriented dst.
    let dev = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let cuda_dev = match &dev {
        Device::Cuda(d) => d,
        _ => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();
    let num_chunks = 1usize;
    let _r16_sz = arena_bytes(FMT_R16, PAL_DIM, num_chunks);
    let f32_sz = arena_bytes(FMT_F32, PAL_DIM, num_chunks);
    let ident = identity_pal_map();

    let mut src_gpu = Vec::new();
    let mut dst_gpu = Vec::new();
    let mut src_cpu = Vec::new();
    for p in 0..N_PAL {
        let data = fill_r16_arena_small(PAL_DIM, num_chunks, 1.0 + p as f32 * 0.1);
        src_gpu.push(cuda_dev.memcpy_stod(&data)?);
        dst_gpu.push(cuda_dev.memcpy_stod(&vec![0u8; f32_sz])?);
        src_cpu.push(read_r16_arena(&data, PAL_DIM, num_chunks));
    }

    run_kernel_pass(
        &stream,
        &dev,
        &src_gpu,
        &dst_gpu,
        [FMT_R16; N_PAL],
        [FMT_F32; N_PAL],
        &ident,
        &ident,
        num_chunks,
        true,
    )?;

    for p in 0..N_PAL {
        let dst_data = cuda_dev.memcpy_dtov(&dst_gpu[p])?;
        let dst_vals = read_f32_channel_arena(&dst_data, PAL_DIM, num_chunks);
        for ld in 0..PAL_DIM {
            for c in 0..num_chunks {
                for t in 0..CHUNK_SIZE {
                    let expected = src_cpu[p][ld][c][t];
                    let got = dst_vals[ld][c][t];
                    // R16 loads f16 values; R16→f16 staging→F32 has f16 precision loss
                    let err = (got - expected).abs() / (expected.abs().max(1e-6));
                    assert!(
                        err < 1e-3, // f16→f32 is lossless up to f16 precision
                        "R16→F32: pal={p} ld={ld} c={c} t={t} expected={expected} got={got}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn palette4_convert_r16_to_bf16_dst() -> Result<()> {
    // R16 → BF16 channel-oriented dst.
    let dev = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let cuda_dev = match &dev {
        Device::Cuda(d) => d,
        _ => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();
    let num_chunks = 1usize;
    let _r16_sz = arena_bytes(FMT_R16, PAL_DIM, num_chunks);
    let bf16_sz = arena_bytes(FMT_BF16, PAL_DIM, num_chunks);
    let ident = identity_pal_map();

    let mut src_gpu = Vec::new();
    let mut dst_gpu = Vec::new();
    let mut src_cpu = Vec::new();
    for p in 0..N_PAL {
        let data = fill_r16_arena_small(PAL_DIM, num_chunks, 1.0 + p as f32 * 0.1);
        src_gpu.push(cuda_dev.memcpy_stod(&data)?);
        dst_gpu.push(cuda_dev.memcpy_stod(&vec![0u8; bf16_sz])?);
        src_cpu.push(read_r16_arena(&data, PAL_DIM, num_chunks));
    }

    run_kernel_pass(
        &stream,
        &dev,
        &src_gpu,
        &dst_gpu,
        [FMT_R16; N_PAL],
        [FMT_BF16; N_PAL],
        &ident,
        &ident,
        num_chunks,
        true,
    )?;

    for p in 0..N_PAL {
        let dst_data = cuda_dev.memcpy_dtov(&dst_gpu[p])?;
        // Read BF16 channel arena manually
        for ld in 0..PAL_DIM {
            for c in 0..num_chunks {
                for t in 0..CHUNK_SIZE {
                    let off = (c * CHUNK_SIZE * PAL_DIM + t * PAL_DIM + ld) * 2;
                    let got_bf = bf16::from_le_bytes([dst_data[off], dst_data[off + 1]]);
                    let got = got_bf.to_f32();
                    let expected = src_cpu[p][ld][c][t];
                    // R16 f16 → f16 staging → BF16: f16→bf16 precision loss ~3.9e-3
                    let rel_err = (got - expected).abs() / (expected.abs().max(1e-6));
                    assert!(
                        rel_err < 0.01,
                        "R16→BF16: pal={p} ld={ld} c={c} t={t} expected={expected} got={got}"
                    );
                }
            }
        }
    }
    Ok(())
}

// =============================================================================
// V-CHANNEL (is_k=false) TEST
// =============================================================================

#[test]
fn palette4_convert_v_channel_identity_roundtrip() -> Result<()> {
    // Test is_k=false: V arena roundtrip R16→R16 with identity pal_map.
    let dev = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let cuda_dev = match &dev {
        Device::Cuda(d) => d,
        _ => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();
    let num_chunks = 1usize;
    let arena_sz = PAL_DIM * num_chunks * R16_BLOCK_BYTES;
    let ident = identity_pal_map();

    let mut src_gpu = Vec::new();
    let mut dst_gpu = Vec::new();
    let mut src_cpu = Vec::new();
    for p in 0..N_PAL {
        let data = fill_r16_arena(PAL_DIM, num_chunks, (p as f32 + 1.0) * 200.0);
        src_gpu.push(cuda_dev.memcpy_stod(&data)?);
        dst_gpu.push(cuda_dev.memcpy_stod(&vec![0u8; arena_sz])?);
        src_cpu.push(data);
    }

    // Build KvHead with V pointers set (K ptrs zeroed, V ptrs point at arenas)
    let src_v_ptrs: [u64; N_PAL] = std::array::from_fn(|p| src_gpu[p].device_ptr(&stream).0 as u64);
    let dst_v_ptrs: [u64; N_PAL] = std::array::from_fn(|p| dst_gpu[p].device_ptr(&stream).0 as u64);

    use candle_core::quantized::cuda::{quantize_palette4_convert_buffered, PalHeadDesc};
    quantize_palette4_convert_buffered(
        &[PalHeadDesc {
            k_src_arena_ptrs: src_v_ptrs,
            v_src_arena_ptrs: src_v_ptrs,
            k_src_fmts: [GgmlDType::R16; N_PAL],
            v_src_fmts: [GgmlDType::R16; N_PAL],
            k_src_pal_map: ident,
            v_src_pal_map: ident,
            k_src_scales: [1.0f32; N_PAL],
            v_src_scales: [1.0f32; N_PAL],
            k_dst_arena_ptrs: dst_v_ptrs,
            v_dst_arena_ptrs: dst_v_ptrs,
            k_dst_fmts: [GgmlDType::R16; N_PAL],
            v_dst_fmts: [GgmlDType::R16; N_PAL],
            k_dst_pal_map: ident,
            v_dst_pal_map: ident,
            k_dst_scales: [1.0f32; N_PAL],
            v_dst_scales: [1.0f32; N_PAL],
        }],
        1,
        1,
        num_chunks,
        &PinnedStager::new(cuda_dev).begin_generation(),
        &stream,
    )?;
    dev.synchronize()?;

    for p in 0..N_PAL {
        let dst_data = cuda_dev.memcpy_dtov(&dst_gpu[p])?;
        let dst_vals = read_r16_arena(&dst_data, PAL_DIM, num_chunks);
        let src_vals = read_r16_arena(&src_cpu[p], PAL_DIM, num_chunks);
        for ld in 0..PAL_DIM {
            for c in 0..num_chunks {
                for t in 0..CHUNK_SIZE {
                    assert_eq!(
                        dst_vals[ld][c][t], src_vals[ld][c][t],
                        "V-channel roundtrip: pal={p} ld={ld} c={c} t={t}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn palette4_convert_v_channel_reversed_pal_map() -> Result<()> {
    // Test is_k=false with a non-trivial pal_map: reversed→uniform.
    let dev = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let cuda_dev = match &dev {
        Device::Cuda(d) => d,
        _ => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();
    let num_chunks = 1usize;
    let arena_sz = PAL_DIM * num_chunks * R16_BLOCK_BYTES;
    let ident = identity_pal_map();
    let rev = reversed_block_pal_map();

    let mut src_gpu = Vec::new();
    let mut dst_gpu = Vec::new();
    let mut src_cpu = Vec::new();
    for p in 0..N_PAL {
        let data = fill_r16_arena(PAL_DIM, num_chunks, (p as f32 + 1.0) * 300.0);
        src_gpu.push(cuda_dev.memcpy_stod(&data)?);
        dst_gpu.push(cuda_dev.memcpy_stod(&vec![0u8; arena_sz])?);
        src_cpu.push(data);
    }

    let src_v_ptrs: [u64; N_PAL] = std::array::from_fn(|p| src_gpu[p].device_ptr(&stream).0 as u64);
    let dst_v_ptrs: [u64; N_PAL] = std::array::from_fn(|p| dst_gpu[p].device_ptr(&stream).0 as u64);

    use candle_core::quantized::cuda::{quantize_palette4_convert_buffered, PalHeadDesc};
    // K and V both use the rev→ident transform; K output is redundant but valid.
    quantize_palette4_convert_buffered(
        &[PalHeadDesc {
            k_src_arena_ptrs: src_v_ptrs,
            v_src_arena_ptrs: src_v_ptrs,
            k_src_fmts: [GgmlDType::R16; N_PAL],
            v_src_fmts: [GgmlDType::R16; N_PAL],
            k_src_pal_map: rev,
            v_src_pal_map: rev,
            k_src_scales: [1.0f32; N_PAL],
            v_src_scales: [1.0f32; N_PAL],
            k_dst_arena_ptrs: dst_v_ptrs,
            v_dst_arena_ptrs: dst_v_ptrs,
            k_dst_fmts: [GgmlDType::R16; N_PAL],
            v_dst_fmts: [GgmlDType::R16; N_PAL],
            k_dst_pal_map: ident,
            v_dst_pal_map: ident,
            k_dst_scales: [1.0f32; N_PAL],
            v_dst_scales: [1.0f32; N_PAL],
        }],
        1,
        1,
        num_chunks,
        &PinnedStager::new(cuda_dev).begin_generation(),
        &stream,
    )?;
    dev.synchronize()?;

    // reversed→identity: dst_pal[p] == src_pal[3-p]
    for dst_p in 0..N_PAL {
        let src_p = 3 - dst_p;
        let dst_data = cuda_dev.memcpy_dtov(&dst_gpu[dst_p])?;
        let dst_vals = read_r16_arena(&dst_data, PAL_DIM, num_chunks);
        let src_vals = read_r16_arena(&src_cpu[src_p], PAL_DIM, num_chunks);
        for ld in 0..PAL_DIM {
            for c in 0..num_chunks {
                for t in 0..CHUNK_SIZE {
                    assert_eq!(
                        dst_vals[ld][c][t], src_vals[ld][c][t],
                        "V reversed→uniform: dst_p={dst_p} src_p={src_p} ld={ld} c={c} t={t}"
                    );
                }
            }
        }
    }
    Ok(())
}

// =============================================================================
// ADDITIONAL COVERAGE TESTS
// =============================================================================

#[test]
fn palette4_convert_quant_multi_chunk() -> Result<()> {
    // R16 → Q8_0 → R16 with num_chunks=4: exercises the per-chunk quant encode/decode
    // loop for c in 0..num_chunks. Verifies block-index arithmetic is correct for c>0.
    let dev = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let cuda_dev = match &dev {
        Device::Cuda(d) => d,
        _ => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();
    let num_chunks = 4usize;
    let ident = identity_pal_map();
    let r16_sz = arena_bytes(FMT_R16, PAL_DIM, num_chunks);
    let q8_sz = arena_bytes(FMT_Q8_0, PAL_DIM, num_chunks);

    let mut src_gpu = Vec::new();
    let mut mid_gpu = Vec::new();
    let mut dst_gpu = Vec::new();
    let mut src_cpu = Vec::new();
    for p in 0..N_PAL {
        // Values in [1.0, 1.16]: survive Q8_0 quantisation.
        let data = fill_r16_arena_small(PAL_DIM, num_chunks, 1.0 + p as f32 * 0.04);
        src_gpu.push(cuda_dev.memcpy_stod(&data)?);
        mid_gpu.push(cuda_dev.memcpy_stod(&vec![0u8; q8_sz])?);
        dst_gpu.push(cuda_dev.memcpy_stod(&vec![0u8; r16_sz])?);
        src_cpu.push(read_r16_arena(&data, PAL_DIM, num_chunks));
    }

    // Pass 1: R16 → Q8_0
    run_kernel_pass(
        &stream,
        &dev,
        &src_gpu,
        &mid_gpu,
        [FMT_R16; N_PAL],
        [FMT_Q8_0; N_PAL],
        &ident,
        &ident,
        num_chunks,
        true,
    )?;
    // Pass 2: Q8_0 → R16
    run_kernel_pass(
        &stream,
        &dev,
        &mid_gpu,
        &dst_gpu,
        [FMT_Q8_0; N_PAL],
        [FMT_R16; N_PAL],
        &ident,
        &ident,
        num_chunks,
        true,
    )?;

    for p in 0..N_PAL {
        let dst_data = cuda_dev.memcpy_dtov(&dst_gpu[p])?;
        let dst_vals = read_r16_arena(&dst_data, PAL_DIM, num_chunks);
        for ld in 0..PAL_DIM {
            for c in 0..num_chunks {
                for t in 0..CHUNK_SIZE {
                    let expected = src_cpu[p][ld][c][t];
                    let got = dst_vals[ld][c][t];
                    let rel_err = (got - expected).abs() / (expected.abs().max(1e-6));
                    assert!(rel_err < 0.01,
                        "quant multi-chunk: pal={p} ld={ld} c={c} t={t} expected={expected} got={got} rel_err={rel_err:.4}");
                }
            }
        }
    }
    Ok(())
}

#[test]
fn palette4_convert_quant_nonidentity_pal_map() -> Result<()> {
    // Exercises the xlat table with a quant format:
    //   pass 1: src R16 (identity_pal) → dst Q8_0 (reversed_pal)
    //   pass 2: src Q8_0 (reversed_pal) → dst R16 (identity_pal)
    // Net effect: R16 roundtrip through Q8_0 with pal_map going out and back.
    // Values should be approximately preserved (Q8_0 tolerance).
    let dev = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let cuda_dev = match &dev {
        Device::Cuda(d) => d,
        _ => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();
    let num_chunks = 2usize;
    let ident = identity_pal_map();
    let rev = reversed_block_pal_map();
    let r16_sz = arena_bytes(FMT_R16, PAL_DIM, num_chunks);
    let q8_sz = arena_bytes(FMT_Q8_0, PAL_DIM, num_chunks);

    let mut src_gpu = Vec::new();
    let mut mid_gpu = Vec::new();
    let mut dst_gpu = Vec::new();
    let mut src_cpu = Vec::new();
    for p in 0..N_PAL {
        let data = fill_r16_arena_small(PAL_DIM, num_chunks, 1.0 + p as f32 * 0.04);
        src_gpu.push(cuda_dev.memcpy_stod(&data)?);
        mid_gpu.push(cuda_dev.memcpy_stod(&vec![0u8; q8_sz])?);
        dst_gpu.push(cuda_dev.memcpy_stod(&vec![0u8; r16_sz])?);
        src_cpu.push(read_r16_arena(&data, PAL_DIM, num_chunks));
    }

    // Pass 1: identity src → reversed dst (Q8_0)
    run_kernel_pass(
        &stream,
        &dev,
        &src_gpu,
        &mid_gpu,
        [FMT_R16; N_PAL],
        [FMT_Q8_0; N_PAL],
        &ident,
        &rev,
        num_chunks,
        true,
    )?;
    // Pass 2: reversed src → identity dst (R16)
    run_kernel_pass(
        &stream,
        &dev,
        &mid_gpu,
        &dst_gpu,
        [FMT_Q8_0; N_PAL],
        [FMT_R16; N_PAL],
        &rev,
        &ident,
        num_chunks,
        true,
    )?;

    // After two pal_map inversions the data should be back in its original pal slots.
    for p in 0..N_PAL {
        let dst_data = cuda_dev.memcpy_dtov(&dst_gpu[p])?;
        let dst_vals = read_r16_arena(&dst_data, PAL_DIM, num_chunks);
        for ld in 0..PAL_DIM {
            for c in 0..num_chunks {
                for t in 0..CHUNK_SIZE {
                    let expected = src_cpu[p][ld][c][t];
                    let got = dst_vals[ld][c][t];
                    let rel_err = (got - expected).abs() / (expected.abs().max(1e-6));
                    assert!(rel_err < 0.01,
                        "quant pal_map roundtrip: pal={p} ld={ld} c={c} t={t} expected={expected} got={got} rel_err={rel_err:.4}");
                }
            }
        }
    }
    Ok(())
}

#[test]
fn palette4_convert_v_channel_quant() -> Result<()> {
    // is_k=false path with a quantized arena format.
    //   pass 1: V R16 → V Q8_0 (identity pal_map)
    //   pass 2: V Q8_0 → V R16
    // Verifies that the IS_K=false branch reads/writes V ptrs and invokes quant correctly.
    let dev = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let cuda_dev = match &dev {
        Device::Cuda(d) => d,
        _ => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();
    let num_chunks = 2usize;
    let ident = identity_pal_map();
    let r16_sz = arena_bytes(FMT_R16, PAL_DIM, num_chunks);
    let q8_sz = arena_bytes(FMT_Q8_0, PAL_DIM, num_chunks);

    let mut src_gpu = Vec::new();
    let mut mid_gpu = Vec::new();
    let mut dst_gpu = Vec::new();
    let mut src_cpu = Vec::new();
    for p in 0..N_PAL {
        let data = fill_r16_arena_small(PAL_DIM, num_chunks, 1.0 + p as f32 * 0.04);
        src_gpu.push(cuda_dev.memcpy_stod(&data)?);
        mid_gpu.push(cuda_dev.memcpy_stod(&vec![0u8; q8_sz])?);
        dst_gpu.push(cuda_dev.memcpy_stod(&vec![0u8; r16_sz])?);
        src_cpu.push(read_r16_arena(&data, PAL_DIM, num_chunks));
    }

    // Build KvHead with V ptrs for V-channel (K=V same arenas).
    use candle_core::quantized::cuda::{quantize_palette4_convert_buffered, PalHeadDesc};
    let src_v_ptrs: [u64; N_PAL] = std::array::from_fn(|p| src_gpu[p].device_ptr(&stream).0 as u64);
    let mid_v_ptrs: [u64; N_PAL] = std::array::from_fn(|p| mid_gpu[p].device_ptr(&stream).0 as u64);
    let dst_v_ptrs: [u64; N_PAL] = std::array::from_fn(|p| dst_gpu[p].device_ptr(&stream).0 as u64);

    // Pass 1: R16 → Q8_0
    quantize_palette4_convert_buffered(
        &[PalHeadDesc {
            k_src_arena_ptrs: src_v_ptrs,
            v_src_arena_ptrs: src_v_ptrs,
            k_src_fmts: [GgmlDType::R16; N_PAL],
            v_src_fmts: [GgmlDType::R16; N_PAL],
            k_src_pal_map: ident,
            v_src_pal_map: ident,
            k_src_scales: [1.0f32; N_PAL],
            v_src_scales: [1.0f32; N_PAL],
            k_dst_arena_ptrs: mid_v_ptrs,
            v_dst_arena_ptrs: mid_v_ptrs,
            k_dst_fmts: [GgmlDType::Q8_0; N_PAL],
            v_dst_fmts: [GgmlDType::Q8_0; N_PAL],
            k_dst_pal_map: ident,
            v_dst_pal_map: ident,
            k_dst_scales: [1.0f32; N_PAL],
            v_dst_scales: [1.0f32; N_PAL],
        }],
        1,
        1,
        num_chunks,
        &PinnedStager::new(cuda_dev).begin_generation(),
        &stream,
    )?;
    dev.synchronize()?;
    // Pass 2: Q8_0 → R16
    quantize_palette4_convert_buffered(
        &[PalHeadDesc {
            k_src_arena_ptrs: mid_v_ptrs,
            v_src_arena_ptrs: mid_v_ptrs,
            k_src_fmts: [GgmlDType::Q8_0; N_PAL],
            v_src_fmts: [GgmlDType::Q8_0; N_PAL],
            k_src_pal_map: ident,
            v_src_pal_map: ident,
            k_src_scales: [1.0f32; N_PAL],
            v_src_scales: [1.0f32; N_PAL],
            k_dst_arena_ptrs: dst_v_ptrs,
            v_dst_arena_ptrs: dst_v_ptrs,
            k_dst_fmts: [GgmlDType::R16; N_PAL],
            v_dst_fmts: [GgmlDType::R16; N_PAL],
            k_dst_pal_map: ident,
            v_dst_pal_map: ident,
            k_dst_scales: [1.0f32; N_PAL],
            v_dst_scales: [1.0f32; N_PAL],
        }],
        1,
        1,
        num_chunks,
        &PinnedStager::new(cuda_dev).begin_generation(),
        &stream,
    )?;
    dev.synchronize()?;

    for p in 0..N_PAL {
        let dst_data = cuda_dev.memcpy_dtov(&dst_gpu[p])?;
        let dst_vals = read_r16_arena(&dst_data, PAL_DIM, num_chunks);
        for ld in 0..PAL_DIM {
            for c in 0..num_chunks {
                for t in 0..CHUNK_SIZE {
                    let expected = src_cpu[p][ld][c][t];
                    let got = dst_vals[ld][c][t];
                    let rel_err = (got - expected).abs() / (expected.abs().max(1e-6));
                    assert!(rel_err < 0.01,
                        "V-channel quant: pal={p} ld={ld} c={c} t={t} expected={expected} got={got} rel_err={rel_err:.4}");
                }
            }
        }
    }
    Ok(())
}

#[test]
fn palette4_convert_multi_layer_quant() -> Result<()> {
    // 2 layers × 2 heads with Q8_0 format: exercises multi-block grid with quant arenas.
    let dev = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let cuda_dev = match &dev {
        Device::Cuda(d) => d,
        _ => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();
    let num_chunks = 1usize;
    let num_layers = 2usize;
    let num_kv_heads = 2usize;
    let total_jobs = num_layers * num_kv_heads;
    let ident = identity_pal_map();
    let r16_sz = arena_bytes(FMT_R16, PAL_DIM, num_chunks);
    let q8_sz = arena_bytes(FMT_Q8_0, PAL_DIM, num_chunks);

    // Allocate src/mid/dst arena slices per (job, pal).
    let mut src_arenas: Vec<Vec<cudarc::driver::CudaSlice<u8>>> = Vec::new();
    let mut mid_arenas: Vec<Vec<cudarc::driver::CudaSlice<u8>>> = Vec::new();
    let mut dst_arenas: Vec<Vec<cudarc::driver::CudaSlice<u8>>> = Vec::new();
    let mut src_cpu = Vec::new();

    for job in 0..total_jobs {
        let mut job_src = Vec::new();
        let mut job_mid = Vec::new();
        let mut job_dst = Vec::new();
        let mut job_ref = Vec::new();
        for p in 0..N_PAL {
            let data =
                fill_r16_arena_small(PAL_DIM, num_chunks, 1.0 + (job * N_PAL + p) as f32 * 0.01);
            job_src.push(cuda_dev.memcpy_stod(&data)?);
            job_mid.push(cuda_dev.memcpy_stod(&vec![0u8; q8_sz])?);
            job_dst.push(cuda_dev.memcpy_stod(&vec![0u8; r16_sz])?);
            job_ref.push(read_r16_arena(&data, PAL_DIM, num_chunks));
        }
        src_arenas.push(job_src);
        mid_arenas.push(job_mid);
        dst_arenas.push(job_dst);
        src_cpu.push(job_ref);
    }

    use candle_core::quantized::cuda::{quantize_palette4_convert_buffered, PalHeadDesc};
    let mut p1_descs: Vec<PalHeadDesc> = Vec::new();
    let mut p2_descs: Vec<PalHeadDesc> = Vec::new();
    for job in 0..total_jobs {
        let src_ptrs: [u64; N_PAL] =
            std::array::from_fn(|p| src_arenas[job][p].device_ptr(&stream).0 as u64);
        let mid_ptrs: [u64; N_PAL] =
            std::array::from_fn(|p| mid_arenas[job][p].device_ptr(&stream).0 as u64);
        let dst_ptrs: [u64; N_PAL] =
            std::array::from_fn(|p| dst_arenas[job][p].device_ptr(&stream).0 as u64);
        p1_descs.push(PalHeadDesc {
            k_src_arena_ptrs: src_ptrs,
            v_src_arena_ptrs: src_ptrs,
            k_src_fmts: [GgmlDType::R16; N_PAL],
            v_src_fmts: [GgmlDType::R16; N_PAL],
            k_src_pal_map: ident,
            v_src_pal_map: ident,
            k_src_scales: [1.0f32; N_PAL],
            v_src_scales: [1.0f32; N_PAL],
            k_dst_arena_ptrs: mid_ptrs,
            v_dst_arena_ptrs: mid_ptrs,
            k_dst_fmts: [GgmlDType::Q8_0; N_PAL],
            v_dst_fmts: [GgmlDType::Q8_0; N_PAL],
            k_dst_pal_map: ident,
            v_dst_pal_map: ident,
            k_dst_scales: [1.0f32; N_PAL],
            v_dst_scales: [1.0f32; N_PAL],
        });
        p2_descs.push(PalHeadDesc {
            k_src_arena_ptrs: mid_ptrs,
            v_src_arena_ptrs: mid_ptrs,
            k_src_fmts: [GgmlDType::Q8_0; N_PAL],
            v_src_fmts: [GgmlDType::Q8_0; N_PAL],
            k_src_pal_map: ident,
            v_src_pal_map: ident,
            k_src_scales: [1.0f32; N_PAL],
            v_src_scales: [1.0f32; N_PAL],
            k_dst_arena_ptrs: dst_ptrs,
            v_dst_arena_ptrs: dst_ptrs,
            k_dst_fmts: [GgmlDType::R16; N_PAL],
            v_dst_fmts: [GgmlDType::R16; N_PAL],
            k_dst_pal_map: ident,
            v_dst_pal_map: ident,
            k_dst_scales: [1.0f32; N_PAL],
            v_dst_scales: [1.0f32; N_PAL],
        });
    }
    quantize_palette4_convert_buffered(
        &p1_descs,
        num_kv_heads,
        num_layers,
        num_chunks,
        &PinnedStager::new(cuda_dev).begin_generation(),
        &stream,
    )?;
    dev.synchronize()?;
    quantize_palette4_convert_buffered(
        &p2_descs,
        num_kv_heads,
        num_layers,
        num_chunks,
        &PinnedStager::new(cuda_dev).begin_generation(),
        &stream,
    )?;
    dev.synchronize()?;

    for job in 0..total_jobs {
        for p in 0..N_PAL {
            let dst_data = cuda_dev.memcpy_dtov(&dst_arenas[job][p])?;
            let dst_vals = read_r16_arena(&dst_data, PAL_DIM, num_chunks);
            for ld in 0..PAL_DIM {
                for c in 0..num_chunks {
                    for t in 0..CHUNK_SIZE {
                        let expected = src_cpu[job][p][ld][c][t];
                        let got = dst_vals[ld][c][t];
                        let rel_err = (got - expected).abs() / (expected.abs().max(1e-6));
                        assert!(rel_err < 0.01,
                            "multi_layer_quant: job={job} pal={p} ld={ld} c={c} t={t} expected={expected} got={got} rel_err={rel_err:.4}");
                    }
                }
            }
        }
    }
    Ok(())
}

// ============================================================================
// Tests for the buffered API: ggml_dtype_to_arena_fmt_code, identity_pal_map_128,
// build_kvhead_bytes_raw, quantize_palette4_convert_buffered,
// and quantize_palette4_convert_identity.
// ============================================================================

/// Spot-check that ggml_dtype_to_arena_fmt_code returns known codes.
#[test]
fn buffered_api_arena_fmt_codes() -> Result<()> {
    use candle_core::quantized::cuda::ggml_dtype_to_arena_fmt_code;
    // Codes match GgmlDType discriminants after format-code migration
    assert_eq!(ggml_dtype_to_arena_fmt_code(GgmlDType::F32)?, 0);
    assert_eq!(ggml_dtype_to_arena_fmt_code(GgmlDType::F16)?, 1);
    assert_eq!(ggml_dtype_to_arena_fmt_code(GgmlDType::BF16)?, 2);
    assert_eq!(ggml_dtype_to_arena_fmt_code(GgmlDType::R16)?, 3);
    assert_eq!(ggml_dtype_to_arena_fmt_code(GgmlDType::Q8_0)?, 7);
    assert_eq!(ggml_dtype_to_arena_fmt_code(GgmlDType::Q4_0)?, 15);
    assert_eq!(ggml_dtype_to_arena_fmt_code(GgmlDType::Q2_0)?, 22);
    assert_eq!(ggml_dtype_to_arena_fmt_code(GgmlDType::Q3_1)?, 20);
    Ok(())
}

/// Verify identity_pal_map_128 bit-packing.
/// For each dimension d (0..128), the 2-bit field at position d should equal d/32.
#[test]
fn buffered_api_identity_pal_map() -> Result<()> {
    use candle_core::quantized::cuda::identity_pal_map_128;
    let map = identity_pal_map_128();
    assert_eq!(map.len(), 32);
    for d in 0..128usize {
        let byte_idx = d / 4;
        let bit_shift = 2 * (d % 4);
        let got = (map[byte_idx] >> bit_shift) & 0x3;
        let expected = (d / 32) as u8;
        assert_eq!(
            got, expected,
            "identity_pal_map: d={d} byte={byte_idx} shift={bit_shift} got={got} expected={expected}"
        );
    }
    Ok(())
}

/// Verify build_kvhead_bytes_raw writes arena ptrs and fmt codes at the correct offsets.
/// This is a pure CPU struct-layout test — no GPU needed.
#[test]
fn buffered_api_build_kvhead_bytes_raw_layout() -> Result<()> {
    use candle_core::quantized::cuda::{
        build_kvhead_bytes_raw, ggml_dtype_to_arena_fmt_code, identity_pal_map_128,
    };

    let k_ptrs: [u64; 4] = [0x1000_0000, 0x2000_0000, 0x3000_0000, 0x4000_0000];
    let v_ptrs: [u64; 4] = [0x5000_0000, 0x6000_0000, 0x7000_0000, 0x8000_0000];
    let k_fmts = [GgmlDType::R16; 4];
    let v_fmts = [GgmlDType::Q8_0; 4];
    let id = identity_pal_map_128();

    let unit_scales = [1.0f32; 4];
    let bytes = build_kvhead_bytes_raw(
        &k_ptrs,
        &v_ptrs,
        &k_fmts,
        &v_fmts,
        &id,
        &id,
        &unit_scales,
        &unit_scales,
    )?;
    assert_eq!(
        bytes.len(),
        168,
        "KvHead must be 168 bytes (HD/2 + 104 for HD=128, f32 scales)"
    );

    // Pal maps should be identity
    assert_eq!(&bytes[0..32], &id[..], "k_pal_map mismatch");
    assert_eq!(&bytes[32..64], &id[..], "v_pal_map mismatch");

    // k_ptr[p] at offset 64 + p*8
    for p in 0..4usize {
        let off = 64 + p * 8;
        let got = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
        assert_eq!(got, k_ptrs[p], "k_ptr[{p}] mismatch");
    }

    // v_ptr[p] at offset 96 + p*8
    for p in 0..4usize {
        let off = 96 + p * 8;
        let got = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
        assert_eq!(got, v_ptrs[p], "v_ptr[{p}] mismatch");
    }

    // k_fmt[p] at offset 128 + p, v_fmt[p] at offset 132 + p
    let r16_code = ggml_dtype_to_arena_fmt_code(GgmlDType::R16)?;
    let q8_code = ggml_dtype_to_arena_fmt_code(GgmlDType::Q8_0)?;
    for p in 0..4usize {
        assert_eq!(bytes[128 + p], r16_code, "k_fmt[{p}] mismatch");
        assert_eq!(bytes[132 + p], q8_code, "v_fmt[{p}] mismatch");
    }

    Ok(())
}

/// build_kvhead_bytes_raw with a custom pal_map: verify the map bytes are written verbatim.
#[test]
fn buffered_api_build_kvhead_bytes_raw_custom_pal_map() -> Result<()> {
    use candle_core::quantized::cuda::build_kvhead_bytes_raw;

    let k_ptrs = [0u64; 4];
    let v_ptrs = [0u64; 4];
    let fmts = [GgmlDType::F16; 4];

    // All dims to palette 2 (0b10 packed × 4 per byte = 0xAA)
    let mut custom_map = [0u8; 32];
    for b in custom_map.iter_mut() {
        *b = 0xAA;
    }

    let unit_scales = [1.0f32; 4];
    let bytes = build_kvhead_bytes_raw(
        &k_ptrs,
        &v_ptrs,
        &fmts,
        &fmts,
        &custom_map,
        &custom_map,
        &unit_scales,
        &unit_scales,
    )?;
    assert_eq!(&bytes[0..32], &custom_map[..], "k_pal_map not written");
    assert_eq!(&bytes[32..64], &custom_map[..], "v_pal_map not written");
    Ok(())
}

/// quantize_palette4_convert_identity: R16 → R16 roundtrip.
/// Exercises the full stack: PalHeadDesc construction, KvHead serialisation,
/// GPU upload, both kernel launches.
#[test]
fn buffered_api_identity_r16_roundtrip() -> Result<()> {
    let dev = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let cuda_dev = match &dev {
        Device::Cuda(d) => d,
        _ => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();

    let num_chunks = 2usize;
    let num_layers = 1usize;
    let num_kv_heads = 1usize;

    let src_data: Vec<Vec<u8>> = (0..N_PAL)
        .map(|p| fill_r16_arena(PAL_DIM, num_chunks, p as f32 * 10000.0))
        .collect();
    let arena_size = PAL_DIM * num_chunks * R16_BLOCK_BYTES;

    let src_gpu: Vec<_> = src_data
        .iter()
        .map(|d| cuda_dev.memcpy_stod(d).unwrap())
        .collect();
    let dst_gpu: Vec<_> = (0..N_PAL)
        .map(|_| cuda_dev.memcpy_stod(&vec![0u8; arena_size]).unwrap())
        .collect();

    let k_src_ptrs: [u64; 4] = std::array::from_fn(|p| src_gpu[p].device_ptr(&stream).0 as u64);
    let k_dst_ptrs: [u64; 4] = std::array::from_fn(|p| dst_gpu[p].device_ptr(&stream).0 as u64);

    use candle_core::quantized::cuda::{
        identity_pal_map_128, quantize_palette4_convert_buffered, PalHeadDesc,
    };
    let id = identity_pal_map_128();
    let descs = vec![PalHeadDesc {
        k_src_arena_ptrs: k_src_ptrs,
        v_src_arena_ptrs: k_src_ptrs, // V src same arenas
        k_src_fmts: [GgmlDType::R16; N_PAL],
        v_src_fmts: [GgmlDType::R16; N_PAL],
        k_src_pal_map: id,
        v_src_pal_map: id,
        k_src_scales: [1.0f32; N_PAL],
        v_src_scales: [1.0f32; N_PAL],
        k_dst_arena_ptrs: k_dst_ptrs,
        v_dst_arena_ptrs: k_dst_ptrs, // V dst same arenas
        k_dst_fmts: [GgmlDType::R16; N_PAL],
        v_dst_fmts: [GgmlDType::R16; N_PAL],
        k_dst_pal_map: id,
        v_dst_pal_map: id,
        k_dst_scales: [1.0f32; N_PAL],
        v_dst_scales: [1.0f32; N_PAL],
    }];
    let generation = PinnedStager::new(cuda_dev).begin_generation();
    quantize_palette4_convert_buffered(
        &descs,
        num_kv_heads,
        num_layers,
        num_chunks,
        &generation,
        &stream,
    )?;
    dev.synchronize()?;

    for p in 0..N_PAL {
        let dst_data = cuda_dev.memcpy_dtov(&dst_gpu[p])?;
        let src_vals = read_r16_arena(&src_data[p], PAL_DIM, num_chunks);
        let dst_vals = read_r16_arena(&dst_data, PAL_DIM, num_chunks);
        for ld in 0..PAL_DIM {
            for c in 0..num_chunks {
                for t in 0..CHUNK_SIZE {
                    assert_eq!(
                        dst_vals[ld][c][t], src_vals[ld][c][t],
                        "identity_r16_roundtrip: pal={p} ld={ld} c={c} t={t}"
                    );
                }
            }
        }
    }

    drop(src_gpu);
    drop(dst_gpu);
    Ok(())
}

/// quantize_palette4_convert_buffered: R16 → Q8_0 → R16 via explicit PalHeadDesc.
/// Verifies the full buffer-construction path including separate src/dst KvHead structs.
#[test]
fn buffered_api_r16_to_q8_roundtrip() -> Result<()> {
    use candle_core::quantized::cuda::{identity_pal_map_128, PalHeadDesc};

    let dev = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let cuda_dev = match &dev {
        Device::Cuda(d) => d,
        _ => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();

    let num_chunks = 1usize;
    let num_layers = 1usize;
    let num_kv_heads = 1usize;

    // Q8_0 scale = max_val / 127.  For values base..base+31001, relative error at the minimum
    // is ≤ scale/2 / base.  With base=10000+p*2000 all palettes stay within f16 range (<65504).
    let src_data: Vec<Vec<u8>> = (0..N_PAL)
        .map(|p| fill_r16_arena(PAL_DIM, num_chunks, 10000.0 + p as f32 * 2000.0))
        .collect();
    let q8_bytes = arena_bytes(FMT_Q8_0, PAL_DIM, num_chunks);
    let r16_bytes = PAL_DIM * num_chunks * R16_BLOCK_BYTES;

    let src_r16: Vec<_> = src_data
        .iter()
        .map(|d| cuda_dev.memcpy_stod(d).unwrap())
        .collect();
    let mid_q8: Vec<_> = (0..N_PAL)
        .map(|_| cuda_dev.memcpy_stod(&vec![0u8; q8_bytes]).unwrap())
        .collect();
    let dst_r16: Vec<_> = (0..N_PAL)
        .map(|_| cuda_dev.memcpy_stod(&vec![0u8; r16_bytes]).unwrap())
        .collect();

    let src_p: [u64; 4] = std::array::from_fn(|p| src_r16[p].device_ptr(&stream).0 as u64);
    let mid_p: [u64; 4] = std::array::from_fn(|p| mid_q8[p].device_ptr(&stream).0 as u64);
    let dst_p: [u64; 4] = std::array::from_fn(|p| dst_r16[p].device_ptr(&stream).0 as u64);

    let r16_fmts = [GgmlDType::R16; N_PAL];
    let q8_fmts = [GgmlDType::Q8_0; N_PAL];

    // Pass 1: R16 → Q8_0
    let id = identity_pal_map_128();
    candle_core::quantized::cuda::quantize_palette4_convert_buffered(
        &[PalHeadDesc {
            k_src_arena_ptrs: src_p,
            v_src_arena_ptrs: src_p,
            k_src_fmts: r16_fmts,
            v_src_fmts: r16_fmts,
            k_src_pal_map: id,
            v_src_pal_map: id,
            k_src_scales: [1.0f32; N_PAL],
            v_src_scales: [1.0f32; N_PAL],
            k_dst_arena_ptrs: mid_p,
            v_dst_arena_ptrs: mid_p,
            k_dst_fmts: q8_fmts,
            v_dst_fmts: q8_fmts,
            k_dst_pal_map: id,
            v_dst_pal_map: id,
            k_dst_scales: [1.0f32; N_PAL],
            v_dst_scales: [1.0f32; N_PAL],
        }],
        num_kv_heads,
        num_layers,
        num_chunks,
        &PinnedStager::new(cuda_dev).begin_generation(),
        &stream,
    )?;
    dev.synchronize()?;

    // Pass 2: Q8_0 → R16
    candle_core::quantized::cuda::quantize_palette4_convert_buffered(
        &[PalHeadDesc {
            k_src_arena_ptrs: mid_p,
            v_src_arena_ptrs: mid_p,
            k_src_fmts: q8_fmts,
            v_src_fmts: q8_fmts,
            k_src_pal_map: id,
            v_src_pal_map: id,
            k_src_scales: [1.0f32; N_PAL],
            v_src_scales: [1.0f32; N_PAL],
            k_dst_arena_ptrs: dst_p,
            v_dst_arena_ptrs: dst_p,
            k_dst_fmts: r16_fmts,
            v_dst_fmts: r16_fmts,
            k_dst_pal_map: id,
            v_dst_pal_map: id,
            k_dst_scales: [1.0f32; N_PAL],
            v_dst_scales: [1.0f32; N_PAL],
        }],
        num_kv_heads,
        num_layers,
        num_chunks,
        &PinnedStager::new(cuda_dev).begin_generation(),
        &stream,
    )?;
    dev.synchronize()?;

    for p in 0..N_PAL {
        let dst_data = cuda_dev.memcpy_dtov(&dst_r16[p])?;
        let src_vals = read_r16_arena(&src_data[p], PAL_DIM, num_chunks);
        let dst_vals = read_r16_arena(&dst_data, PAL_DIM, num_chunks);
        for ld in 0..PAL_DIM {
            for c in 0..num_chunks {
                for t in 0..CHUNK_SIZE {
                    let s = src_vals[ld][c][t];
                    let d = dst_vals[ld][c][t];
                    let rel = (d - s).abs() / s.abs().max(1e-6);
                    assert!(
                        rel < 0.02,
                        "buffered_r16_to_q8: pal={p} ld={ld} c={c} t={t} src={s} dst={d} rel={rel:.4}"
                    );
                }
            }
        }
    }

    drop(src_r16);
    drop(mid_q8);
    drop(dst_r16);
    Ok(())
}

/// quantize_palette4_convert_identity: 2 layers × 2 heads, R16 → R16 roundtrip.
/// Exercises the multi-layer/multi-head path through the identity shim.
#[test]
fn buffered_api_identity_multi_layer_multi_head() -> Result<()> {
    let dev = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let cuda_dev = match &dev {
        Device::Cuda(d) => d,
        _ => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();

    let num_chunks = 1usize;
    let num_layers = 2usize;
    let num_kv_heads = 2usize;
    let total_jobs = num_layers * num_kv_heads; // 4
    let arena_size = PAL_DIM * num_chunks * R16_BLOCK_BYTES;

    let mut src_cpu: Vec<Vec<Vec<u8>>> = Vec::new();
    let mut src_gpu_arenas: Vec<Vec<_>> = Vec::new();
    let mut dst_gpu_arenas: Vec<Vec<_>> = Vec::new();
    let mut k_src_all: Vec<[u64; 4]> = Vec::new();
    let mut k_dst_all: Vec<[u64; 4]> = Vec::new();

    for job in 0..total_jobs {
        let pals: Vec<Vec<u8>> = (0..N_PAL)
            .map(|p| fill_r16_arena(PAL_DIM, num_chunks, (job * 100 + p * 10) as f32 + 1.0))
            .collect();
        let sg: Vec<_> = pals
            .iter()
            .map(|d| cuda_dev.memcpy_stod(d).unwrap())
            .collect();
        let dg: Vec<_> = (0..N_PAL)
            .map(|_| cuda_dev.memcpy_stod(&vec![0u8; arena_size]).unwrap())
            .collect();
        let sp: [u64; 4] = std::array::from_fn(|p| sg[p].device_ptr(&stream).0 as u64);
        let dp: [u64; 4] = std::array::from_fn(|p| dg[p].device_ptr(&stream).0 as u64);
        k_src_all.push(sp);
        k_dst_all.push(dp);
        src_cpu.push(pals);
        src_gpu_arenas.push(sg);
        dst_gpu_arenas.push(dg);
    }

    use candle_core::quantized::cuda::{
        identity_pal_map_128, quantize_palette4_convert_buffered, PalHeadDesc,
    };
    let id = identity_pal_map_128();
    let descs: Vec<PalHeadDesc> = (0..total_jobs)
        .map(|i| PalHeadDesc {
            k_src_arena_ptrs: k_src_all[i],
            v_src_arena_ptrs: k_src_all[i], // V shares same arenas
            k_src_fmts: [GgmlDType::R16; N_PAL],
            v_src_fmts: [GgmlDType::R16; N_PAL],
            k_src_pal_map: id,
            v_src_pal_map: id,
            k_src_scales: [1.0f32; N_PAL],
            v_src_scales: [1.0f32; N_PAL],
            k_dst_arena_ptrs: k_dst_all[i],
            v_dst_arena_ptrs: k_dst_all[i],
            k_dst_fmts: [GgmlDType::R16; N_PAL],
            v_dst_fmts: [GgmlDType::R16; N_PAL],
            k_dst_pal_map: id,
            v_dst_pal_map: id,
            k_dst_scales: [1.0f32; N_PAL],
            v_dst_scales: [1.0f32; N_PAL],
        })
        .collect();
    let generation = PinnedStager::new(cuda_dev).begin_generation();
    quantize_palette4_convert_buffered(
        &descs,
        num_kv_heads,
        num_layers,
        num_chunks,
        &generation,
        &stream,
    )?;
    dev.synchronize()?;

    for job in 0..total_jobs {
        for p in 0..N_PAL {
            let dst_data = cuda_dev.memcpy_dtov(&dst_gpu_arenas[job][p])?;
            let src_vals = read_r16_arena(&src_cpu[job][p], PAL_DIM, num_chunks);
            let dst_vals = read_r16_arena(&dst_data, PAL_DIM, num_chunks);
            for ld in 0..PAL_DIM {
                for c in 0..num_chunks {
                    for t in 0..CHUNK_SIZE {
                        assert_eq!(
                            dst_vals[ld][c][t], src_vals[ld][c][t],
                            "identity_multi: job={job} pal={p} ld={ld} c={c} t={t}"
                        );
                    }
                }
            }
        }
    }

    drop(src_gpu_arenas);
    drop(dst_gpu_arenas);
    Ok(())
}

/// Performance benchmark: measures R16→{Q8_0, Q4_0, Q5_0, Q2_0} conversion throughput.
///
/// For each of 4 quant types, we run batches of calls until ~2 seconds of GPU work
/// has elapsed, repeating 4 rounds.  Reports throughput as tokens/second (where one
/// "token" is one CHUNK_SIZE-token slot across all 4 palettes × all head/layer entries).
///
/// Run with:
///   cargo test --package candle-core --features cuda --test palette4_convert_gpu_test \
///     palette4_convert_throughput_bench -- --ignored --nocapture
#[test]
#[ignore]
fn palette4_convert_throughput_bench() -> Result<()> {
    use candle_core::quantized::cuda::{identity_pal_map_128, PalHeadDesc};
    use std::time::{Duration, Instant};

    let dev = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => {
            eprintln!("No CUDA device available, skipping throughput bench.");
            return Ok(());
        }
    };
    let cuda_dev = match &dev {
        Device::Cuda(d) => d,
        _ => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();

    // Configuration: large enough to give meaningful GPU work per call.
    let num_layers: usize = 32;
    let num_kv_heads: usize = 8;
    let num_chunks: usize = 64; // 64 × 32 = 2048 tokens in flight per call

    let total_heads = num_layers * num_kv_heads; // 256 heads
                                                 // Tokens processed per kernel call: heads × palettes × chunks × chunk_size
    let tokens_per_call = total_heads * N_PAL * num_chunks * CHUNK_SIZE;

    let target_duration = Duration::from_secs_f64(2.0);

    // All quant types (src always R16 → dst quant, then back to R16 for full round).
    let quant_configs: &[(&str, GgmlDType, u8)] = &[
        ("Q4_0", GgmlDType::Q4_0, FMT_Q4_0),
        ("Q4_1", GgmlDType::Q4_1, FMT_Q4_1),
        ("Q5_0", GgmlDType::Q5_0, FMT_Q5_0),
        ("Q5_1", GgmlDType::Q5_1, FMT_Q5_1),
        ("Q8_0", GgmlDType::Q8_0, FMT_Q8_0),
        ("Q8_1", GgmlDType::Q8_1, FMT_Q8_1),
        ("Q4_KS", GgmlDType::Q4_KS, FMT_Q4_KS),
        ("Q8_KS", GgmlDType::Q8_KS, FMT_Q8_KS),
        ("Q2_0", GgmlDType::Q2_0, FMT_Q2_0),
        ("Q3_0", GgmlDType::Q3_0, FMT_Q3_0),
        ("Q0", GgmlDType::Q0, FMT_Q0),
        ("Q1_S", GgmlDType::Q1_S, FMT_Q1_S),
        ("Q2_S", GgmlDType::Q2_S, FMT_Q2_S),
        ("Q2_A", GgmlDType::Q2_A, FMT_Q2_A),
        ("Q2_1", GgmlDType::Q2_1, FMT_Q2_1),
        ("Q3_1", GgmlDType::Q3_1, FMT_Q3_1),
        ("Q0_V", GgmlDType::Q0_V, FMT_Q0_V),
        ("Q1_A", GgmlDType::Q1_A, FMT_Q1_A),
        ("Q0_X", GgmlDType::Q0_X, FMT_Q0_X),
        ("Q0_M2", GgmlDType::Q0_M2, FMT_Q0_M2),
        ("Q0_M4", GgmlDType::Q0_M4, FMT_Q0_M4),
    ];

    let r16_bytes_per_pal = PAL_DIM * num_chunks * R16_BLOCK_BYTES;

    // Pre-allocate GPU arenas once and reuse across all benchmarks.
    // src R16, mid quant (one per quant type), dst R16.
    let src_r16_arenas: Vec<_> = (0..total_heads)
        .flat_map(|job| {
            (0..N_PAL).map(move |p| {
                let cpu = fill_r16_arena(
                    PAL_DIM,
                    num_chunks,
                    10000.0 + (job * N_PAL + p) as f32 * 100.0,
                );
                cpu
            })
        })
        .collect();

    let src_gpu: Vec<_> = src_r16_arenas
        .iter()
        .map(|d| cuda_dev.memcpy_stod(d).unwrap())
        .collect();

    let src_ptrs: Vec<[u64; N_PAL]> = (0..total_heads)
        .map(|job| std::array::from_fn(|p| src_gpu[job * N_PAL + p].device_ptr(&stream).0 as u64))
        .collect();

    let id = identity_pal_map_128();
    let r16_fmts = [GgmlDType::R16; N_PAL];

    println!(
        "\n=== palette4_convert throughput benchmark ===\n\
         Config: {num_layers} layers × {num_kv_heads} heads, {num_chunks} chunks/head\n\
         Tokens per call: {tokens_per_call}  (~2s per quant type)\n"
    );
    println!("{:<10}  {:>14}  {:>14}", "FMT", "calls/s", "Mtok/s");
    println!("{}", "-".repeat(42));

    for (name, ggml_dtype, fmt_code) in quant_configs {
        let q_bytes_per_pal = arena_bytes(*fmt_code, PAL_DIM, num_chunks);

        let mid_gpu: Vec<_> = (0..total_heads * N_PAL)
            .map(|_| cuda_dev.memcpy_stod(&vec![0u8; q_bytes_per_pal]).unwrap())
            .collect();
        let mid_ptrs: Vec<[u64; N_PAL]> = (0..total_heads)
            .map(|job| {
                std::array::from_fn(|p| mid_gpu[job * N_PAL + p].device_ptr(&stream).0 as u64)
            })
            .collect();

        let dst_gpu: Vec<_> = (0..total_heads * N_PAL)
            .map(|_| cuda_dev.memcpy_stod(&vec![0u8; r16_bytes_per_pal]).unwrap())
            .collect();
        let dst_ptrs: Vec<[u64; N_PAL]> = (0..total_heads)
            .map(|job| {
                std::array::from_fn(|p| dst_gpu[job * N_PAL + p].device_ptr(&stream).0 as u64)
            })
            .collect();

        let q_fmts = [*ggml_dtype; N_PAL];

        // Build desc slices once.
        let descs_fwd: Vec<PalHeadDesc> = (0..total_heads)
            .map(|job| PalHeadDesc {
                k_src_arena_ptrs: src_ptrs[job],
                v_src_arena_ptrs: src_ptrs[job],
                k_src_fmts: r16_fmts,
                v_src_fmts: r16_fmts,
                k_src_pal_map: id,
                v_src_pal_map: id,
                k_src_scales: [1.0f32; N_PAL],
                v_src_scales: [1.0f32; N_PAL],
                k_dst_arena_ptrs: mid_ptrs[job],
                v_dst_arena_ptrs: mid_ptrs[job],
                k_dst_fmts: q_fmts,
                v_dst_fmts: q_fmts,
                k_dst_pal_map: id,
                v_dst_pal_map: id,
                k_dst_scales: [1.0f32; N_PAL],
                v_dst_scales: [1.0f32; N_PAL],
            })
            .collect();

        let descs_bwd: Vec<PalHeadDesc> = (0..total_heads)
            .map(|job| PalHeadDesc {
                k_src_arena_ptrs: mid_ptrs[job],
                v_src_arena_ptrs: mid_ptrs[job],
                k_src_fmts: q_fmts,
                v_src_fmts: q_fmts,
                k_src_pal_map: id,
                v_src_pal_map: id,
                k_src_scales: [1.0f32; N_PAL],
                v_src_scales: [1.0f32; N_PAL],
                k_dst_arena_ptrs: dst_ptrs[job],
                v_dst_arena_ptrs: dst_ptrs[job],
                k_dst_fmts: r16_fmts,
                v_dst_fmts: r16_fmts,
                k_dst_pal_map: id,
                v_dst_pal_map: id,
                k_dst_scales: [1.0f32; N_PAL],
                v_dst_scales: [1.0f32; N_PAL],
            })
            .collect();

        // Warm-up: one pair of calls before timing.
        let stager = PinnedStager::new(cuda_dev);
        let generation = stager.begin_generation();
        candle_core::quantized::cuda::quantize_palette4_convert_buffered(
            &descs_fwd,
            num_kv_heads,
            num_layers,
            num_chunks,
            &generation,
            &stream,
        )?;
        candle_core::quantized::cuda::quantize_palette4_convert_buffered(
            &descs_bwd,
            num_kv_heads,
            num_layers,
            num_chunks,
            &generation,
            &stream,
        )?;
        dev.synchronize()?;

        let mut call_pairs: u64 = 0;
        let t0 = Instant::now();
        while t0.elapsed() < target_duration {
            candle_core::quantized::cuda::quantize_palette4_convert_buffered(
                &descs_fwd,
                num_kv_heads,
                num_layers,
                num_chunks,
                &generation,
                &stream,
            )?;
            candle_core::quantized::cuda::quantize_palette4_convert_buffered(
                &descs_bwd,
                num_kv_heads,
                num_layers,
                num_chunks,
                &generation,
                &stream,
            )?;
            call_pairs += 1;
        }
        stager.flush()?;
        dev.synchronize()?;
        let elapsed = t0.elapsed().as_secs_f64();

        // Each pair = 2 calls (fwd + bwd), count individual calls for throughput.
        let total_calls = call_pairs * 2;
        let calls_per_sec = total_calls as f64 / elapsed;
        let mtok_per_sec = (total_calls as f64 * tokens_per_call as f64) / elapsed / 1e6;

        println!(
            "{:<10}  {:>14.1}  {:>14.2}",
            name, calls_per_sec, mtok_per_sec
        );

        drop(mid_gpu);
        drop(dst_gpu);
    }

    drop(src_gpu);
    println!("=== done ===");
    Ok(())
}
