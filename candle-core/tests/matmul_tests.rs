use candle_core::{test_device, DType, Device, IndexOp, Result, Tensor};

fn matmul(device: &Device) -> Result<()> {
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let a = Tensor::from_slice(&data, (2, 2), device)?;
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = Tensor::from_slice(&data, (2, 2), device)?;

    let c = a.matmul(&b)?;
    assert_eq!(c.to_vec2::<f32>()?, &[[7.0f32, 10.0], [15.0, 22.0]]);

    let data = vec![1.0f32, 2.0];
    let a = Tensor::from_slice(&data, (2, 1), device)?;
    let data = vec![3.0f32, 4.0];
    let b = Tensor::from_slice(&data, (1, 2), device)?;
    let c = a.matmul(&b)?;
    assert_eq!(c.to_vec2::<f32>()?, &[&[3.0, 4.0], &[6.0, 8.0]]);

    let data: Vec<_> = (0..6).map(|i| i as f32).collect();
    let a = Tensor::from_slice(&data, (2, 3), device)?;
    let data: Vec<_> = (0..6).map(|i| (i + 2) as f32).collect();
    let b = Tensor::from_slice(&data, (3, 2), device)?;
    let c = a.matmul(&b)?;
    assert_eq!(c.to_vec2::<f32>()?, &[&[16., 19.], &[52., 64.]]);

    let data: Vec<_> = (0..12).map(|i| i as f32).collect();
    let a = Tensor::from_slice(&data, (2, 2, 3), device)?;
    let data: Vec<_> = (0..12).map(|i| (i + 2) as f32).collect();
    let b = Tensor::from_slice(&data, (2, 3, 2), device)?;
    let expected = [[[16., 19.], [52., 64.]], [[214., 235.], [304., 334.]]];

    let c = a.matmul(&b)?;
    assert_eq!(c.to_vec3::<f32>()?, &expected);

    // Also perform the matmul on contiguous transposed versions.
    let a_tt = a.t()?.contiguous()?.t()?;
    assert!(!a_tt.is_contiguous());
    assert_eq!(a.dims(), a_tt.dims());
    assert_eq!(a_tt.stride(), &[6, 1, 2]);

    let b_tt = b.t()?.contiguous()?.t()?;
    assert!(!b_tt.is_contiguous());
    assert_eq!(b.dims(), b_tt.dims());
    assert_eq!(b_tt.stride(), &[6, 1, 3]);

    assert_eq!(a_tt.matmul(&b)?.to_vec3::<f32>()?, &expected);
    assert_eq!(a.matmul(&b_tt)?.to_vec3::<f32>()?, &expected);
    assert_eq!(a_tt.matmul(&b_tt)?.to_vec3::<f32>()?, &expected);
    Ok(())
}

fn matmul_bf16(device: &Device) -> Result<()> {
    if !device.supports_bf16() {
        return Ok(());
    }
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let a = Tensor::from_slice(&data, (2, 2), device)?.to_dtype(DType::BF16)?;
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = Tensor::from_slice(&data, (2, 2), device)?.to_dtype(DType::BF16)?;

    let c = a.matmul(&b)?.to_dtype(DType::F32)?;
    assert_eq!(c.to_vec2::<f32>()?, &[[7.0f32, 10.0], [15.0, 22.0]]);
    Ok(())
}

fn broadcast_matmul(device: &Device) -> Result<()> {
    let lhs = Tensor::randn(0f32, 1f32, (3, 1, 4, 5), device)?;
    let rhs = Tensor::randn(0f32, 1f32, (6, 5, 2), device)?;
    let out = lhs.broadcast_matmul(&rhs)?;
    assert_eq!(out.dims(), &[3, 6, 4, 2]);
    for idx1 in 0..3 {
        for idx2 in 0..6 {
            let out = out.i((idx1, idx2))?;
            let lhs = lhs.i((idx1, 0))?;
            let rhs = rhs.i(idx2)?;
            let out2 = lhs.matmul(&rhs);
            let sum_diff2 = (out - out2)?.sqr()?.sum_all()?;
            // With cuda, we see errors of up to ~1e-12.
            assert!(sum_diff2.to_vec0::<f32>()? < 1e-6)
        }
    }
    Ok(())
}

#[test]
fn tensor_dot() -> Result<()> {
    let lhs = Tensor::new(&[1., 2., 3.], &Device::Cpu)?;
    let rhs = Tensor::new(&[4., 5., 6.], &Device::Cpu)?;
    let expected = Tensor::new(32., &Device::Cpu)?;
    let dot_ret = lhs.dot(&rhs)?;
    candle_core::test_utils::assert_tensor_eq(&dot_ret, &expected)?;
    Ok(())
}

#[test]
fn tensor_mv() -> Result<()> {
    let mat = Tensor::new(&[[1., 2., 3.], [4., 5., 6.]], &Device::Cpu)?;
    let vec = Tensor::new(&[1., 1., 1.], &Device::Cpu)?;
    let expected = Tensor::new(&[6., 15.], &Device::Cpu)?;
    let mv_ret = mat.mv(&vec)?;
    candle_core::test_utils::assert_tensor_eq(&mv_ret, &expected)?;
    Ok(())
}

// https://github.com/huggingface/candle/issues/1948
fn squeeze_mm(device: &Device) -> Result<()> {
    let seq_len = 8_usize;
    let a = Tensor::zeros((1, seq_len, 16), DType::F32, device)?;
    let x = a.i((.., seq_len - 1, ..))?;
    let w = Tensor::zeros((32, 16), DType::F32, device)?.t()?;
    let x = x.matmul(&w)?;
    assert_eq!(x.dims(), &[1, 32]);
    Ok(())
}

// https://github.com/huggingface/candle/issues/1992
fn mm_layout(device: &Device) -> Result<()> {
    let a = Tensor::arange(0f32, 16f32, device)?.reshape((1, 1, 4, 4))?;
    let b = Tensor::arange(0f32, 8f32, device)?.reshape((1, 1, 4, 2))?;
    let mm1 = a.matmul(&b)?;
    // Forces the layout to be:
    // shape: [1, 1, 4, 2], stride: [8, 2, 2, 1], start_offset: 0
    // This is still a contiguous matrix but matmul checks are only the two last dimensions have
    // non 1 sizes but matmul check may be reluctant to handle it.
    let b = b.transpose(1, 2)?.force_contiguous()?.transpose(1, 2)?;
    let mm2 = a.matmul(&b)?;
    let diff = (mm1 - mm2)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert_eq!(diff, 0.);
    Ok(())
}

/// A GEMM with a zero extent must not reach the BLAS library.
///
/// cuBLAS derives its launch grid from the problem shape, so a zero dimension
/// asks it to launch `(0,1,1)` — which the CUDA runtime refuses with
/// `cudaErrorInvalidConfiguration` and leaves as sticky state for whatever calls
/// `cudaGetLastError` next. compute-sanitizer counted 41 of these in one short
/// daemon run, planted by calls that had no work to do.
///
/// Both shapes below are reachable in ordinary operation, not just in tests: a
/// wave quantum with no rows produces the first, an empty group in a grouped
/// GEMM the second.
///
/// Registered through `test_device!` so the CPU backend has to agree — the two
/// must produce the same tensor for a shape neither of them multiplies.
fn degenerate_matmul(device: &Device) -> Result<()> {
    // No output elements: `[0,4] @ [4,3]` is a `[0,3]` tensor and there is
    // nothing to compute.
    let a = Tensor::zeros((0usize, 4usize), DType::F32, device)?;
    let b = Tensor::zeros((4usize, 3usize), DType::F32, device)?;
    let c = a.matmul(&b)?;
    assert_eq!(c.dims(), &[0, 3]);
    assert_eq!(c.elem_count(), 0);

    // **`k == 0` is the case that must not return uninitialised memory.** Every
    // output element is an empty sum, so the answer is a materialised zero — the
    // value cuBLAS itself would have written with `beta = 0`, and the reason the
    // early return allocates zeroed rather than uninitialised.
    let a = Tensor::zeros((2usize, 0usize), DType::F32, device)?;
    let b = Tensor::zeros((0usize, 3usize), DType::F32, device)?;
    let c = a.matmul(&b)?;
    assert_eq!(c.dims(), &[2, 3]);
    assert_eq!(c.to_vec2::<f32>()?, &[[0.0f32, 0.0, 0.0], [0.0, 0.0, 0.0]]);

    // Batched, so the guard is exercised through the strided-batched path too.
    let a = Tensor::zeros((2usize, 3usize, 0usize), DType::F32, device)?;
    let b = Tensor::zeros((2usize, 0usize, 5usize), DType::F32, device)?;
    let c = a.matmul(&b)?;
    assert_eq!(c.dims(), &[2, 3, 5]);
    assert_eq!(c.sum_all()?.to_vec0::<f32>()?, 0.0);
    Ok(())
}

test_device!(matmul, matmul_cpu, matmul_gpu, matmul_metal);
test_device!(
    degenerate_matmul,
    degenerate_matmul_cpu,
    degenerate_matmul_gpu,
    degenerate_matmul_metal
);
test_device!(
    matmul_bf16,
    matmul_bf16_cpu,
    matmul_bf16_gpu,
    matmul_bf16_metal
);
test_device!(
    broadcast_matmul,
    broadcast_matmul_cpu,
    broadcast_matmul_gpu,
    broadcast_matmul_metal
);
test_device!(squeeze_mm, squeeze_mm_cpu, squeeze_mm_gpu, squeeze_mm_metal);
test_device!(mm_layout, mm_layout_cpu, mm_layout_gpu, mm_layout_metal);

/// **A 4-D matmul whose batch dims are both 1, against a transposed operand.**
///
/// This is the shape Stable Diffusion's VAE self-attention runs at: one image,
/// one head, every latent position attending to every other. The operands are
/// `(1, 1, hw, c)` and the transpose of the same, so the right-hand side is
/// non-contiguous *and* every batch dim is 1 — the "squeezed layout" that
/// `gemm_config` has dedicated match arms for, and that the stale TODO in
/// `stable_diffusion::attention` still refers to a workaround for.
///
/// `squeeze_mm` and `mm_layout` above cover 3-D and small cases. Nothing covered
/// a 4-D batch of ones at a width where the batch stride is large enough to be
/// mis-selected, which is where a wrong stride reads the same block repeatedly
/// and paints a periodic pattern rather than failing.
fn attention_shaped_mm(device: &Device) -> Result<()> {
    for &(hw, c) in &[(256usize, 64usize), (1024, 128), (4096, 64)] {
        let mk = |n: usize, seed: u64| -> Result<Tensor> {
            let v: Vec<f32> = (0..n)
                .map(|i| {
                    (((i as u64 * 6364136223846793005).wrapping_add(seed) >> 33) % 1000) as f32
                        / 500.0
                        - 1.0
                })
                .collect();
            Tensor::from_vec(v, n, device)
        };
        let q = mk(hw * c, 1)?.reshape((1, 1, hw, c))?;
        let k = mk(hw * c, 7)?.reshape((1, 1, hw, c))?;

        // What the attention block does: matmul straight against a transpose.
        let got = q.matmul(&k.t()?)?;
        // The same product with the transpose materialised first, which no
        // backend can shortcut.
        let want = q.matmul(&k.t()?.contiguous()?)?;

        let (a, b) = (
            got.flatten_all()?.to_vec1::<f32>()?,
            want.flatten_all()?.to_vec1::<f32>()?,
        );
        let worst = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(
            worst < 1e-3,
            "matmul (1,1,{hw},{c}) x transposed differs from the contiguous form by {worst} — \
             the batch stride was chosen wrongly for a squeezed layout, so the product repeats \
             one block instead of walking the operand"
        );
    }
    Ok(())
}

test_device!(
    attention_shaped_mm,
    attention_shaped_mm_cpu,
    attention_shaped_mm_gpu,
    attention_shaped_mm_metal
);
