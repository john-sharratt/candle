//! Checks how a launch is cut and staged, and the int8 fragment loaders that the
//! decode QK dot uses.
//!
//! A launch stages exactly its bulk sequences and nothing for a verify window, and
//! a deep launch is cut into row groups and key windows. The fragment test checks
//! the int8 MMA loaders against an exact reference, because a lane mistake there
//! would corrupt every dot silently. A flash-attention smoke test confirms that
//! the BF16 prefill path returns finite output.

use super::*;

#[cfg(all(test, feature = "cuda"))]
pub(super) mod tests {
    #![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

    use candle_nn::kv_cache::PrefillLaunchBounds;

    /// Serialize the GPU tests: the split-KV launcher's grow-on-demand
    /// partial pool is a function-local static sized for the production
    /// single-scheduler-thread model — concurrent test launches interleave
    /// its free/realloc. So does the region pool every `KvCache` here draws
    /// from, which is why the lock is crate-wide rather than module-local.
    use crate::models::gpu_test_lock::gpu_serial;

    use candle::{Device, Result};

    /// A launch stages exactly its bulk sequences, whole: a verify window
    /// beside them — or alone — stages nothing and reads the arena, and the
    /// stage is as deep as the deepest bulk context.
    #[test]
    fn a_launch_stages_its_bulk_sequences_whole_and_nothing_else() {
        use super::KvStagePlan;
        assert_eq!(
            KvStagePlan::new(&[5], &[4196]),
            KvStagePlan {
                positions: 0,
                max_kv_len: 0
            },
            "a verify window stages nothing"
        );
        let ragged = KvStagePlan::new(&[300, 5, 256], &[102, 50_000, 2001]);
        assert_eq!(
            ragged,
            KvStagePlan {
                positions: 402 + 2257,
                max_kv_len: 2257
            }
        );
        // 5,318 rows: K 1,361,408 B; scales 85,088 B padded to 85,248; V
        // 2,722,816 B.
        assert_eq!(ragged.bytes(2, 256), 1_361_408 + 85_248 + 2_722_816);
    }

    /// The launch's cut on Flash-Next attention (24 heads over 2 at head_dim
    /// 256): an 8K chunk at 295K of history runs two row groups of ten key
    /// windows each, out of one 152,862,720 B carve; beside a verify window it
    /// cuts the same, and the verify window stages nothing.
    #[test]
    fn a_deep_launch_is_cut_into_groups_and_windows() {
        use super::PrefillLaunchCut;
        let cut = PrefillLaunchCut::new(
            &[8192, 5],
            &[295_000, 4096],
            24,
            2,
            256,
            PrefillLaunchBounds::PRODUCTION,
        );
        assert_eq!(cut.group_chunks, vec![10, 10]);
        assert_eq!(cut.max_kv_len, 303_192);
        assert_eq!(cut.layout.stage_positions, 32_768);
        assert_eq!(cut.layout.group_blocks, 2048);
        assert_eq!(cut.layout.bytes(), 152_862_720);
        // An 8K prompt from zero fits one window: nothing carries, so it is
        // one launch over the whole grid.
        let fresh =
            PrefillLaunchCut::new(&[8192], &[0], 24, 2, 256, PrefillLaunchBounds::PRODUCTION);
        assert_eq!(fresh.group_chunks, vec![1]);
        assert_eq!(fresh.layout.group_blocks, 4096);
        assert_eq!(fresh.layout.carry_tokens, 0);
        assert_eq!(fresh.layout.bytes(), 8192 * 2 * 784);
    }

    #[test]
    #[cfg(feature = "flash-attn")]
    fn paged_prefill_uses_flash_attn_kernel_smoke() -> Result<()> {
        // This forces a multi-token BF16 prefill so the flash-attn paged-prefill path is eligible.
        let device = Device::new_cuda(0)?;

        let b_sz = 1usize;
        let seq_len = 2usize;
        let n_head = 1usize;
        let n_kv_head = 1usize;
        let head_dim = 64usize;

        // Shapes: q (B, H, T, D), k/v (B, H_kv, T, D)
        let q = Tensor::randn(0f32, 1f32, (b_sz, n_head, seq_len, head_dim), &device)?
            .to_dtype(DType::BF16)?;
        let k = Tensor::randn(0f32, 1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
            .to_dtype(DType::BF16)?
            .contiguous()?;
        let v = Tensor::randn(0f32, 1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
            .to_dtype(DType::BF16)?
            .contiguous()?;

        let mut cache0 = KvCache::new(2, 512);
        cache0.force_dtype(DType::BF16);
        let offsets = [0usize];
        let mut caches: [&mut KvCache; 1] = [&mut cache0];

        let generation = PinnedStager::new(device.as_cuda_device()?).begin_generation();

        let out = paged_prefill_uniform(
            &mut caches,
            &offsets,
            &q,
            &k,
            &v,
            b_sz,
            seq_len,
            n_head,
            n_kv_head,
            head_dim,
            None,
            &make_test_rope(head_dim, &device)?,
            false, // rope_interleaved
            &generation,
        )
        .expect("expected paged prefill to be applicable");

        assert_eq!(out.len(), 1);
        let y = &out[0];
        assert_eq!(y.dims(), &[1, n_head, seq_len, head_dim]);
        // Output dtype matches input dtype (BF16 in, BF16 out)
        assert_eq!(y.dtype(), DType::BF16);

        // Basic sanity: output should be finite.
        let max_abs = y
            .to_dtype(DType::F32)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_vec0::<f32>()?;
        assert!(max_abs.is_finite());

        Ok(())
    }

    /// Regression test for the int8 m16n8k32 MMA fragment loaders.
    ///
    /// Feeds known int8 `A` (16×32) and `B` (8×32) through the exact loaders the
    /// decode QK dot uses (`load_a_frag_m16k32` / `load_b_frag_n8k32` /
    /// `mma_int8_m16n8k32`) and asserts the int32 `C = A·Bᵀ` matches a trivial
    /// CPU reference — exactly, no tolerance. Guards against the
    /// `load_b_frag_n8k32` lane-decomposition bug that silently corrupted every
    /// int8 QK dot, and any future MMA-fragment regression. Fast (one warp, no
    /// model), so it runs on every `cargo test`.
    #[test]
    fn int8_mma_m16n8k32_fragment_layout() -> Result<()> {
        let _gpu = gpu_serial();
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        let device = Device::new_cuda(0)?;
        let cuda = device.as_cuda_device()?;
        let stream = cuda.cuda_stream();

        // Deterministic int8 inputs spanning a range of magnitudes and signs.
        let mut a = vec![0i8; 16 * 32];
        let mut b = vec![0i8; 8 * 32];
        for (i, x) in a.iter_mut().enumerate() {
            *x = ((i * 31 + 7) % 17) as i8 - 8;
        }
        for (i, x) in b.iter_mut().enumerate() {
            *x = ((i * 13 + 5) % 19) as i8 - 9;
        }
        // Reference: C[m][n] = Σ_k A[m][k]·B[n][k] (int32, exact).
        let mut c_ref = vec![0i32; 16 * 8];
        for m in 0..16 {
            for n in 0..8 {
                let mut s = 0i32;
                for k in 0..32 {
                    s += a[m * 32 + k] as i32 * b[n * 32 + k] as i32;
                }
                c_ref[m * 8 + n] = s;
            }
        }

        let a_slice = cuda.memcpy_stod(&a)?;
        let b_slice = cuda.memcpy_stod(&b)?;
        let c_slice = cuda.alloc_zeros::<i32>(16 * 8)?;
        unsafe {
            let (a_ptr, _ga) = a_slice.device_ptr(&stream);
            let (b_ptr, _gb) = b_slice.device_ptr(&stream);
            let (c_ptr, _gc) = c_slice.device_ptr(&stream);
            candle_kernels::paged_decode::mma_int8_m16n8k32_test(
                a_ptr as *const i8,
                b_ptr as *const i8,
                c_ptr as *mut i32,
                stream.cu_stream() as *mut core::ffi::c_void,
            );
        }
        device.synchronize()?;
        let c_gpu = cuda.memcpy_dtov(&c_slice)?;
        assert_eq!(
            c_gpu, c_ref,
            "int8 m16n8k32 MMA fragment layout regressed (load_a/load_b/mma)"
        );
        Ok(())
    }
}
