//! Tensors on the reservation's held ground: long-lived buffers, and logits
//! rows kept past the forward that produced them.
//!
//! A buffer sized once and reused across forwards — a verify cohort's capture
//! rows, a draft head's scratch — was an ordinary `Tensor::empty`, a pool
//! allocation outside the reservation for as long as it lived. A logits row
//! read after its forward (a prefill's continuation, a closing tail's next-token
//! row, a sliced prefill's slices) has to be copied off the forward's wave span
//! first, since the next forward cannot open while that span is held; the copy
//! was `to_owned_tensor`, one vocabulary-wide pool allocation per row, and a
//! batch of rows was then joined with `Tensor::cat` into a second one.
//!
//! Here both land in held ground ([`Held`]). The block is anchored to the
//! tensor viewing it and goes back to the ground when the last view drops.

use std::sync::Arc;

use candle::cuda_backend::cudarc::driver::result::memset_d8_async;
use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::cuda_backend::{Held, WrapErr};
use candle::{DType, Device, LeaseAnchor, Result, Shape, Tensor};

/// An uninitialised tensor of `shape` on held ground; off CUDA, `Tensor::empty`.
///
/// Every element must be written before it is read, as for `Tensor::empty`.
///
/// This and the helpers below track their caller, so the ground's ledger names
/// the code that holds the block rather than this module.
#[track_caller]
pub fn held_empty(shape: impl Into<Shape>, dtype: DType, device: &Device) -> Result<Tensor> {
    let shape = shape.into();
    let Device::Cuda(cuda) = device else {
        return Tensor::empty(shape, dtype, device);
    };
    let bytes = shape.elem_count() * dtype.size_in_bytes();
    // SAFETY: the caller writes every element before reading it.
    let held = unsafe { Held::<u8>::alloc(cuda, bytes)? };
    let ptr = held.device_ptr(&cuda.cuda_stream()).0;
    // SAFETY: the range is `bytes` of held ground the anchor keeps for as long
    // as this tensor or any view of it exists.
    unsafe {
        Tensor::from_anchored_cuda_ptr(ptr, dtype, shape, device, LeaseAnchor::new(Arc::new(held)))
    }
}

/// A zeroed tensor of `shape` on held ground; off CUDA, `Tensor::zeros`.
///
/// For a buffer whose zero is read before it is written — a count table kept
/// zero between dispatches, an atomic accumulator — and only those (hot-path
/// invariant 6).
#[track_caller]
pub fn held_zeros(shape: impl Into<Shape>, dtype: DType, device: &Device) -> Result<Tensor> {
    let shape = shape.into();
    let Device::Cuda(cuda) = device else {
        return Tensor::zeros(shape, dtype, device);
    };
    let bytes = shape.elem_count() * dtype.size_in_bytes();
    let t = held_empty(shape, dtype, device)?;
    if bytes > 0 {
        let (storage, layout) = t.storage_and_layout();
        let candle::Storage::Cuda(c) = &*storage else {
            candle::bail!("held zeros: a CUDA tensor without CUDA storage");
        };
        let stream = cuda.cuda_stream();
        let at =
            c.slice.device_ptr(&stream) + (layout.start_offset() * dtype.size_in_bytes()) as u64;
        // SAFETY: `at` names the tensor's own `bytes`, which nothing has read.
        unsafe { memset_d8_async(at, 0, bytes, stream.cu_stream()) }.w()?;
    }
    Ok(t)
}

/// `rows` — each a `[1, vocab]` or `[vocab]` logits row — stacked into one
/// `[rows.len(), vocab]` block of held ground, in order.
#[track_caller]
pub fn hold_rows(rows: &[Tensor]) -> Result<Tensor> {
    let Some(first) = rows.first() else {
        candle::bail!("held rows: nothing to hold");
    };
    let vocab = first.elem_count();
    let dtype = first.dtype();
    for (i, r) in rows.iter().enumerate() {
        if r.elem_count() != vocab || r.dtype() != dtype {
            candle::bail!(
                "held rows: row {i} is {:?} {:?}, the first {vocab} {dtype:?}",
                r.dims(),
                r.dtype()
            );
        }
    }
    let block = held_empty((rows.len(), vocab), dtype, first.device())?;
    for (i, row) in rows.iter().enumerate() {
        block.slice_set(&row.reshape((1, vocab))?, 0, i)?;
    }
    Ok(block)
}

/// One row on held ground, as `[1, vocab]`.
#[track_caller]
pub fn hold_row(row: &Tensor) -> Result<Tensor> {
    hold_rows(std::slice::from_ref(row))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rows are stacked in order, whatever shape each arrived in.
    #[test]
    fn rows_stack_in_order() -> Result<()> {
        let a = Tensor::new(&[1f32, 2., 3.], &Device::Cpu)?;
        let b = Tensor::new(&[[4f32, 5., 6.]], &Device::Cpu)?;
        let block = hold_rows(&[a, b])?;
        assert_eq!(
            block.to_vec2::<f32>()?,
            vec![vec![1., 2., 3.], vec![4., 5., 6.]]
        );
        Ok(())
    }

    /// Rows of different widths are refused rather than misaligned.
    #[test]
    fn rows_of_different_widths_are_refused() -> Result<()> {
        let a = Tensor::new(&[1f32, 2., 3.], &Device::Cpu)?;
        let b = Tensor::new(&[1f32, 2.], &Device::Cpu)?;
        assert!(hold_rows(&[a, b]).is_err());
        Ok(())
    }

    /// On CUDA the block holds the rows' values exactly, and a held buffer
    /// written through `slice_set` reads back what was written.
    #[test]
    fn held_tensors_read_back_on_cuda() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let a = Tensor::new(&[[1f32, 2., 3.]], &device)?;
        let b = Tensor::new(&[[4f32, 5., 6.]], &device)?;
        let block = hold_rows(&[a, b])?;
        assert_eq!(
            block.to_vec2::<f32>()?,
            vec![vec![1., 2., 3.], vec![4., 5., 6.]]
        );
        let buf = held_empty((2, 2), DType::F32, &device)?;
        buf.slice_set(&Tensor::new(&[[7f32, 8.], [9., 10.]], &device)?, 0, 0)?;
        assert_eq!(buf.to_vec2::<f32>()?, vec![vec![7., 8.], vec![9., 10.]]);
        // Zeroed even where the ground last held something else.
        drop(buf);
        let zeros = held_zeros((2, 2), DType::F32, &device)?;
        assert_eq!(zeros.to_vec2::<f32>()?, vec![vec![0.; 2]; 2]);
        Ok(())
    }
}
