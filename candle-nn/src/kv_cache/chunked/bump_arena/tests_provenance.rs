//! Tests that an op reading a wave-backed operand allocates from the same arena.
//!
//! The type system bounds the lifetime of a wave-backed result but cannot see
//! which bytes back it, so these tests assert on the cursor as well as on the
//! backing. A pool-backed operand stays on the pool, and a ticket from a closed
//! generation resolves to nothing, so a stale lease allocates from the pool
//! rather than from ground the span now uses for something else.

use super::*;

#[cfg(all(test, feature = "cuda"))]
pub(super) mod provenance_tests {
    use super::super::super::gpu_test_lock::gpu_serial as serial;

    use super::{begin_wave, LayerPhase};

    use candle::cuda_backend::wave_provenance::LeaseOrigin;

    use candle::{DType, Device, Result, Storage, Tensor};

    pub(in super::super) fn cuda() -> Option<Device> {
        match Device::cuda_if_available(0) {
            Ok(d @ Device::Cuda(_)) => Some(d),
            _ => None,
        }
    }

    /// The backing a tensor's storage carries.
    pub(in super::super) fn backing_of(t: &Tensor) -> candle::cuda_backend::Backing {
        let (storage, _) = t.storage_and_layout();
        match &*storage {
            Storage::Cuda(c) => c.backing,
            _ => panic!("expected CUDA storage"),
        }
    }

    /// **The rule.** An op reading a wave-backed operand allocates its output
    /// from the same generation — not from the pool.
    ///
    /// This is what makes the `'w` on the result true rather than merely
    /// permitted, so it is worth pinning directly: the type system cannot check
    /// that the *bytes* came from the arena, only that the lifetime is not
    /// widened. Asserting on the cursor as well as the backing catches a ticket
    /// that resolves to the right arena but never actually carves from it.
    #[test]
    fn an_op_inherits_its_operands_arena() -> Result<()> {
        let _gpu = serial();
        let Some(dev) = cuda() else { return Ok(()) };
        let Device::Cuda(cd) = &dev else {
            return Ok(());
        };
        let stream = cd.cuda_stream();

        // A real allocation to point the lease at, so the cast reads live bytes.
        let owner = Tensor::zeros((256,), DType::F32, &dev)?;

        let wave = begin_wave(&stream, LayerPhase::Attention)?;
        let before = {
            let inner = wave.inner.lock().unwrap();
            inner.cursor
        };

        let range = wave.alloc(256 * 4, 256)?;
        let operand = unsafe {
            Tensor::from_leased_cuda_ptr(
                range.ptr,
                DType::F32,
                (256,),
                &dev,
                LeaseOrigin::Wave(wave.ticket()),
            )?
        };
        let cursor_after_operand = {
            let inner = wave.inner.lock().unwrap();
            inner.cursor
        };

        let cast = operand.to_dtype(DType::BF16)?;

        match backing_of(&cast) {
            candle::cuda_backend::Backing::Lease(LeaseOrigin::Wave(t)) => {
                assert_eq!(t, wave.ticket(), "inherited the wrong generation");
            }
            other => panic!("cast fell back to the pool: {other:?}"),
        }
        let cursor_after_cast = {
            let inner = wave.inner.lock().unwrap();
            inner.cursor
        };
        assert!(
            cursor_after_cast > cursor_after_operand,
            "the cast claimed no bytes from the arena ({cursor_after_operand} -> {cursor_after_cast})"
        );
        assert!(cursor_after_operand > before);

        drop(cast);
        drop(operand);
        drop(wave);
        // The owner must survive every one of those drops: a wave-backed result
        // that reached `cuMemFreeAsync` would take the pool with it.
        assert_eq!(owner.to_vec1::<f32>()?.len(), 256);
        Ok(())
    }

    /// A pool-backed operand stays on the pool. The rule inherits an arena; it
    /// does not invent one, so an ordinary tensor is untouched by any of this
    /// even while a generation happens to be open.
    #[test]
    fn a_pool_operand_does_not_acquire_a_wave() -> Result<()> {
        let _gpu = serial();
        let Some(dev) = cuda() else { return Ok(()) };
        let Device::Cuda(cd) = &dev else {
            return Ok(());
        };
        let wave = begin_wave(&cd.cuda_stream(), LayerPhase::Ffn)?;
        let ordinary = Tensor::zeros((128,), DType::F32, &dev)?;
        let cast = ordinary.to_dtype(DType::BF16)?;
        assert!(
            matches!(backing_of(&cast), candle::cuda_backend::Backing::Owned),
            "a pool operand must not pick up a wave it never came from"
        );
        drop(wave);
        Ok(())
    }

    /// A ticket from a closed generation resolves to nothing, so an op holding
    /// a stale lease allocates from the pool instead of carving from whatever
    /// occupies that span now. `LiveTensor<'w>` makes this unreachable in safe
    /// code; the backstop exists for the `unsafe` raw-pointer constructors.
    #[test]
    fn a_stale_ticket_does_not_resolve() -> Result<()> {
        let _gpu = serial();
        let Some(dev) = cuda() else { return Ok(()) };
        let Device::Cuda(cd) = &dev else {
            return Ok(());
        };
        let stream = cd.cuda_stream();

        let stale = {
            let wave = begin_wave(&stream, LayerPhase::Attention)?;
            let t = wave.ticket();
            let _r = wave.alloc(1024, 256)?;
            t
        };
        assert_eq!(
            candle::cuda_backend::wave_provenance::wave_alloc(stale, 1024, 256),
            super::WaveCarve::Closed,
            "a ticket outlived its generation and still carved from the span"
        );
        let _ = dev;
        Ok(())
    }
}
