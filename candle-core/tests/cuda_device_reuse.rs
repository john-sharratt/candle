//! `Device::new_cuda(n)` must be cheap to call repeatedly for the same ordinal,
//! and must not quietly change what the caller draws.
//!
//! Callers treat a device handle as a value: helpers take `&Device`, tests build
//! one per case, and the KV cache's own documentation states that its
//! device-side state is *process-global, one region pool per device*. All of
//! that assumes asking for ordinal 0 twice is free.
//!
//! It was not. The context was never the problem — `CudaContext::new` retains
//! the driver's *primary* context, so every call for an ordinal already shared
//! one. It was the memoised state hanging off the handle: compiled modules and
//! the upload caches, whose entries hold a `ManuallyDrop<CudaSlice>` that a
//! dropped device never gives back. A process that asked often enough ran the
//! card down until something could not allocate, and the next request failed
//! `CUBLAS_STATUS_NOT_INITIALIZED` — which reads as "CUDA is broken" rather than
//! "you have made too many of these". In `candle-nn` that surfaced as three GPU
//! tests failing only when run after ~220 others, passing in isolation, and
//! passing under any subset: a count threshold, not a bad test.
//!
//! The first test below is that threshold, well overshot. It doubles as the
//! evidence that neither the cuBLAS handle nor the curand generator was the
//! leak — it builds 512 of each — which is what allows both to stay per-handle,
//! as the second test requires.

#[cfg(feature = "cuda")]
#[test]
fn repeated_new_cuda_returns_the_same_device() {
    use candle_core::Device;

    if !candle_core::utils::cuda_is_available() {
        return;
    }
    let first = Device::new_cuda(0).expect("first device");

    // Well past the point the old path fell over, and each handle is dropped
    // before the next is taken — the leak was not about holding them.
    for i in 0..512 {
        let d = Device::new_cuda(0).unwrap_or_else(|e| panic!("device {i} of 512 failed: {e}"));
        assert!(
            d.same_device(&first),
            "device {i} is a different device from the first — asking for ordinal 0 \
             twice must not build two of them"
        );
    }
}

/// A new handle draws from a freshly seeded generator, cached or not.
///
/// `candle` guarantees this — `set_seed` replaces the generator rather than
/// re-seeding it, precisely so that the numbers repeat — and a fixture built
/// with `Tensor::randn` is only a fixture because of it.
///
/// Sharing the cached handle's generator instead breaks it in the least visible
/// way available: nothing fails at the call, and instead what a caller draws
/// becomes a function of who drew before it. It cost two `candle-transformers`
/// tests, both of which passed alone and failed in the suite.
#[cfg(feature = "cuda")]
#[test]
fn each_handle_draws_the_same_numbers() {
    use candle_core::{Device, Tensor};

    if !candle_core::utils::cuda_is_available() {
        return;
    }

    let draw = || -> Vec<f32> {
        let d = Device::new_cuda(0).expect("device");
        Tensor::randn(0f32, 1f32, 8, &d)
            .expect("randn")
            .to_vec1::<f32>()
            .expect("readback")
    };

    let first = draw();
    // Drawn between the two, so a shared generator would have advanced past
    // whatever `first` saw and `third` could not match it.
    let _intervening = draw();
    let third = draw();

    assert_eq!(
        first, third,
        "two handles on ordinal 0 drew different numbers, so the cached device is \
         handing out an advancing generator instead of a freshly seeded one"
    );
}
