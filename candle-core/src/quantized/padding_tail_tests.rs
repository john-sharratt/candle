//! **A weight placed from the host has a zero matmul padding tail.**
//!
//! The q-matmul kernels read past a weight's payload into its
//! `MATRIX_ROW_PADDING` tail and multiply what they find by the activation's
//! zero padding. Zero times a NaN scale is NaN, so a tail left as the allocator
//! found it turned a 12-row Q2_K matmul's whole result NaN once earlier
//! allocations had left non-finite bytes there (the `dmmv_q8_*` tests, then
//! `qmm_b_cuda`). The tail is checked here as raw bytes, on memory that held
//! 0xFF the moment before it was claimed.

use super::*;
use crate::quantized::k_quants::BlockQ8_0;
use crate::quantized::GgmlType;
use crate::{Device, Storage, Tensor};

/// The CUDA device, or `None` with no card to test on.
fn cuda() -> Option<CudaDevice> {
    match Device::new_cuda(0) {
        Ok(Device::Cuda(d)) => Some(d),
        _ => None,
    }
}

/// Fill and release buffers around `bytes` with 0xFF, so the next claims of
/// that size land on non-zero memory.
fn poison(dev: &CudaDevice, bytes: usize) -> Result<()> {
    let ones = vec![0xFFu8; bytes + 256];
    for len in bytes.saturating_sub(64)..=bytes + 64 {
        let mut buf = unsafe { dev.alloc::<u8>(len)? };
        dev.memcpy_htod(&ones[..len], &mut buf)?;
        drop(buf);
    }
    Ok(())
}

/// Every byte past `len` in `inner`.
fn tail(dev: &CudaDevice, inner: &CudaSlice<u8>, len: usize) -> Result<Vec<u8>> {
    Ok(dev.memcpy_dtov(inner)?[len..].to_vec())
}

#[test]
fn a_weight_quantized_from_the_host_has_a_zero_tail() -> Result<()> {
    let Some(dev) = cuda() else { return Ok(()) };
    let el = 6 * 256;
    let padded = padded_storage_bytes(el, GgmlDType::Q2_K);
    let src = Tensor::arange(0f32, el as f32, &Device::Cuda(dev.clone()))?;
    let (src, _) = src.storage_and_layout();
    let Storage::Cuda(src) = &*src else {
        unreachable!("a tensor made on the CUDA device")
    };
    let mut q = QCudaStorage::zeros(&dev, el, GgmlDType::Q2_K)?;
    poison(&dev, padded)?;
    q.quantize(src)?;
    let len = q.data.len;
    assert_eq!(
        tail(&dev, &q.data.inner, len)?,
        vec![0u8; q.data.inner.len() - len]
    );
    Ok(())
}

#[test]
fn a_weight_loaded_from_the_host_has_a_zero_tail() -> Result<()> {
    let Some(dev) = cuda() else { return Ok(()) };
    let blocks = vec![BlockQ8_0::zeros(); 48];
    let payload = core::mem::size_of_val(blocks.as_slice());
    poison(&dev, padded_storage_bytes(48 * 32, GgmlDType::Q8_0))?;
    let QStorage::Cuda(q) = load_quantized(&dev, &blocks)? else {
        unreachable!("loaded onto the CUDA device")
    };
    assert_eq!(q.data.len, payload);
    assert_eq!(
        tail(&dev, &q.data.inner, payload)?,
        vec![0u8; q.data.inner.len() - payload]
    );
    Ok(())
}
