//! The synchronous check inside a wave capture.
//!
//! An armed capture runs inside the forward, and the forward records its
//! launches as a wave capture. The check has to see the value the recorded
//! launch produced — not refuse its fence because the stream it holds is the
//! capture stream, and not read the slot before the segment that fills it has
//! run.
//!
//! A binary of its own: a capture records whatever this process issues on the
//! device, so the parallel tests of `tensor_assert_tests` would land inside it.
//! Within it, the tests take turns on [`WAVE`].

#![cfg(all(feature = "cuda", feature = "tensor-assert"))]

use std::sync::{Mutex, MutexGuard};

use candle_core::tensor_assert::check_now;
use candle_core::{Device, Result, Tensor};

/// A device holds one wave capture at a time, and a capture records whatever
/// the process issues: the tests here take turns.
static WAVE: Mutex<()> = Mutex::new(());

fn wave_lock() -> MutexGuard<'static, ()> {
    WAVE.lock().unwrap_or_else(|p| p.into_inner())
}

/// The overwrite guard reads a copy's destination address before the copy.
/// Read through the compute stream while the thread records, that read is
/// refused by the driver and the refusal kills the capture — every later call
/// of the wave fails with `STREAM_CAPTURE_UNSUPPORTED`. Guarded copies inside a
/// recording wave must leave the wave intact and land in order.
#[test]
fn guarded_copies_inside_a_wave_capture_keep_the_capture_alive() -> Result<()> {
    let _wave = wave_lock();
    let d = Device::new_cuda(0)?;
    let Device::Cuda(cu) = &d else {
        unreachable!("asked for a CUDA device")
    };
    let mut dst = cu.alloc_zeros::<f32>(4)?;
    let mut copy = cu.alloc_zeros::<f32>(4)?;
    let capture = cu.begin_wave_capture()?;
    cu.record_launches()?;
    cu.memcpy_htod(&[1.0f32, 2.0, 3.0, 4.0], &mut dst)?;
    cu.memcpy_dtod(&dst, &mut copy)?;
    let doubled = Tensor::from_vec(vec![5.0f32, 6.0], (2,), &d)?.affine(2.0, 0.0)?;
    capture.finish()?;

    assert_eq!(cu.memcpy_dtov(&copy)?, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(doubled.to_vec1::<f32>()?, vec![10.0, 12.0]);
    Ok(())
}

#[test]
fn a_synchronous_check_inside_a_wave_capture_sees_the_recorded_launch() -> Result<()> {
    let _wave = wave_lock();
    let d = Device::new_cuda(0)?;
    let Device::Cuda(cu) = &d else {
        unreachable!("asked for a CUDA device")
    };
    let x = Tensor::from_vec(vec![1.0f32, f32::NAN, 3.0, f32::INFINITY], (4,), &d)?;
    let finite = Tensor::from_vec(vec![1.0f32, 2.0], (2,), &d)?;
    let capture = cu.begin_wave_capture()?;
    cu.record_launches()?;
    let y = x.affine(2.0, 1.0)?;
    let mut seen = None;
    let bad = check_now(&y, "capture::bad", |f| seen = Some(f.clone()));
    let clean = finite.affine(2.0, 1.0)?;
    let clean_bad = check_now(&clean, "capture::clean", |_| {});
    capture.finish()?;

    assert!(bad, "the recorded launch's NaN must be seen by the check");
    let f = seen.expect("a bad check reports through its callback");
    assert_eq!((f.nan, f.inf, f.elems), (1, 1, 4));
    assert_eq!((f.min, f.max), (Some(3.0), Some(7.0)));
    assert!(
        !clean_bad,
        "a finite tensor checked inside a capture is not bad"
    );
    Ok(())
}
