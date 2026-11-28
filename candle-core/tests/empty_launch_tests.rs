//! Tests that an op on an empty tensor launches nothing on the card.
//!
//! A launcher that sizes its grid by its elements launches a zero-block grid
//! for none of them, which is `cudaErrorInvalidConfiguration`. Most launchers
//! never read their launch's result, so the error waits on the thread for the
//! next reader: the next layer's `moe_bucketize`, which refuses to run and
//! stops the expert pipeline. An ONNX `Slice` over an empty range is one
//! `index_select` of no indices, and BiRefNet's backbone makes one on every
//! cutout (WhiteWorld, 2026-11-28).

#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Result, Tensor};

/// The card, or `None` where there is none to test on.
fn card() -> Option<Device> {
    match Device::new_cuda(0) {
        Ok(d) => Some(d),
        Err(_) => {
            eprintln!("skipping: CUDA device required");
            None
        }
    }
}

/// **The index family, the elementwise maps, the pools and the resize launch
/// nothing for no elements**, and leave no error for whoever reads next.
#[test]
fn an_empty_op_leaves_no_launch_error_pending() -> Result<()> {
    let Some(device) = card() else {
        return Ok(());
    };
    // Whatever an earlier test on this thread left behind is not this test's.
    let _ = device.take_launch_error();

    let data = Tensor::arange(0f32, 24.0, &device)?.reshape((4, 6))?;
    let none = Tensor::arange_step(2i64, 2, 1, &device)?;
    assert_eq!(none.dims(), &[0]);
    assert_eq!(data.index_select(&none, 0)?.dims(), &[0, 6]);
    assert_eq!(data.index_select(&none, 1)?.dims(), &[4, 0]);

    let empty = Tensor::zeros((0, 6), DType::F32, &device)?;
    let ids = Tensor::zeros((0, 6), DType::U32, &device)?;
    assert_eq!(empty.gather(&ids, 1)?.dims(), &[0, 6]);
    assert_eq!((&empty + &empty)?.dims(), &[0, 6]);
    assert_eq!(empty.sqrt()?.dims(), &[0, 6]);
    assert_eq!(empty.ge(&empty)?.dims(), &[0, 6]);
    let mask = empty.ge(&empty)?;
    assert_eq!(mask.where_cond(&empty, &empty)?.dims(), &[0, 6]);
    assert_eq!(empty.sum_keepdim(1)?.dims(), &[0, 1]);
    // Over an empty axis every sum is zero, written though nothing is read.
    let across = empty.sum_keepdim(0)?;
    assert_eq!(across.dims(), &[1, 6]);
    assert_eq!(across.flatten_all()?.to_vec1::<f32>()?, vec![0.0; 6]);
    assert_eq!(empty.affine(2.0, 1.0)?.dims(), &[0, 6]);

    let images = Tensor::zeros((0, 3, 8, 8), DType::F32, &device)?;
    assert_eq!(images.max_pool2d(2)?.dims(), &[0, 3, 4, 4]);
    assert_eq!(images.avg_pool2d(2)?.dims(), &[0, 3, 4, 4]);
    assert_eq!(images.upsample_nearest2d(16, 16)?.dims(), &[0, 3, 16, 16]);

    device.synchronize()?;
    assert_eq!(
        device.take_launch_error(),
        None,
        "an op on an empty tensor left a CUDA launch error pending on the thread"
    );
    Ok(())
}

/// **A pending error is taken once, and cleared**; the CPU has none to take.
#[test]
fn a_pending_launch_error_is_taken_and_cleared() -> Result<()> {
    let Some(device) = card() else {
        return Ok(());
    };
    let _ = device.take_launch_error();
    assert_eq!(device.take_launch_error(), None);
    assert_eq!(Device::Cpu.take_launch_error(), None);
    Ok(())
}
