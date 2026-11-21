//! Tests for nested guest generations and for stage results carried upward.
//!
//! These tests pin the two properties that let a guest's stages compose. A
//! nested generation releases only to the mark it opened at, and a stage's
//! result is carried at the top of the span so the enclosing generation owns it
//! and the bottom cannot carve into it. A chain of stages keeps two carried
//! results however long it runs. Each test closes the process-wide guest arena
//! before it returns, so a failed assertion cannot leave later tests reading
//! freed ground.

#[cfg(test)]
pub(super) mod tests {

    /// One guest-arena test at a time. The registry is per device and process
    /// wide, so two tests opening an arena on the same card replace each
    /// other's, and each reads the other's cursor.
    #[cfg(feature = "cuda")]
    pub(in super::super) fn guest_arena_test() -> std::sync::MutexGuard<'static, ()> {
        use std::sync::Mutex;
        static ONE: Mutex<()> = Mutex::new(());
        ONE.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// **A nested generation is a mark/release scope; only the outermost
    /// rewinds.**
    ///
    /// This is the property that makes staging composable. Before it, `live`
    /// was merely a count and an inner generation moved nothing when it
    /// dropped, so a decoder that staged its resnets inside a per-tile
    /// generation accumulated every resnet's intermediates anyway — the arena
    /// filled, the first failed carve produced an owned tensor, and everything
    /// downstream of it fell to the pool. The assertions below are the whole
    /// contract: the inner drop returns the cursor to where that generation
    /// opened, and the outer drop returns it to zero.
    #[test]
    #[cfg(feature = "cuda")]
    fn a_nested_generation_releases_to_its_mark() {
        use crate::kv_cache::{
            begin_guest, close_guest_arena, guest_domain_stats, open_guest_arena,
        };
        use candle::cuda_backend::cudarc::driver::{CudaStream, DevicePtr};
        use candle::Device;
        use std::sync::Arc;

        let _one = guest_arena_test();
        let Ok(device) = Device::new_cuda(0) else {
            return;
        };
        let cuda = device.as_cuda_device().unwrap();
        let stream = cuda.cuda_stream();
        let ordinal = stream.context().ordinal();

        // Real ground for the arena to stand on, so a carve inside it addresses
        // memory this test owns for the duration.
        let ground = unsafe { stream.alloc::<u8>(1 << 20).unwrap() };
        let (base, _sync) = ground.device_ptr(&stream);

        // **Closed on the way out however this test leaves.** The registry is
        // process-global and `ground` dies at the end of this scope, so an
        // assertion that fires before a trailing `close_guest_arena` would leave
        // every later test in this binary looking at an arena over freed memory.
        // That is not hypothetical — it is exactly how a nested-release bug in
        // the wave path presented, as unrelated tests failing with an
        // unspecified launch failure.
        struct CloseOnDrop(Arc<CudaStream>);
        impl Drop for CloseOnDrop {
            fn drop(&mut self) {
                close_guest_arena(&self.0);
            }
        }
        open_guest_arena(&stream, base, 1 << 20);
        let _close = CloseOnDrop(stream.clone());

        let outer = begin_guest(&stream).unwrap();
        outer.alloc(4096, 256).unwrap();
        let mark = guest_domain_stats(ordinal).unwrap().0;
        assert!(mark >= 4096, "the outer carve did not move the cursor");

        {
            let inner = begin_guest(&stream).unwrap();
            inner.alloc(8192, 256).unwrap();
            let within = guest_domain_stats(ordinal).unwrap().0;
            assert!(within >= mark + 8192, "the inner carve shared the cursor");
        }
        assert_eq!(
            guest_domain_stats(ordinal).unwrap().0,
            mark,
            "a nested generation must release to the cursor it opened at"
        );

        // The outer scope is still usable after the release, and reuses the
        // bytes the inner one gave back.
        outer.alloc(8192, 256).unwrap();
        assert!(guest_domain_stats(ordinal).unwrap().0 >= mark + 8192);

        drop(outer);
        assert_eq!(
            guest_domain_stats(ordinal).unwrap().0,
            0,
            "the outermost generation must rewind the arena"
        );
    }

    /// **A stage's result is carried in the generation around it, never the
    /// pool, and the bottom cannot carve into it.**
    ///
    /// The result outlives its stage — the stage releases to its mark the
    /// moment it closes — so it is carried at the top of the span, and the
    /// enclosing generation owns it. The numbers must survive the release, a
    /// stage nested in a stage must compose, and while a carry stands the
    /// cursor must refuse the bytes it holds.
    #[test]
    #[cfg(feature = "cuda")]
    fn a_stage_result_is_carried_in_its_enclosing_generation() {
        use crate::kv_cache::{
            begin_guest, close_guest_arena, guest_domain_stats, guest_stage, open_guest_arena,
        };
        use candle::cuda_backend::cudarc::driver::{CudaStream, DevicePtr};
        use candle::{Device, Tensor};
        use std::sync::Arc;

        let _one = guest_arena_test();
        let Ok(device) = Device::new_cuda(0) else {
            return;
        };
        let cuda = device.as_cuda_device().unwrap();
        let stream = cuda.cuda_stream();
        let ordinal = stream.context().ordinal();
        let capacity = 1usize << 20;
        let ground = unsafe { stream.alloc::<u8>(capacity).unwrap() };
        let (base, _sync) = ground.device_ptr(&stream);
        struct CloseOnDrop(Arc<CudaStream>);
        impl Drop for CloseOnDrop {
            fn drop(&mut self) {
                close_guest_arena(&self.0);
            }
        }
        open_guest_arena(&stream, base, capacity);
        let _close = CloseOnDrop(stream.clone());

        let outer = begin_guest(&stream).unwrap();
        let x = Tensor::arange(0f32, 1024., &device).unwrap();
        let y = guest_stage(&x, |x| x.affine(2., 1.)).unwrap();
        assert_eq!(
            guest_domain_stats(ordinal).unwrap().0,
            0,
            "the stage did not release to its mark"
        );
        let z = guest_stage(&y, |y| guest_stage(y, |t| t.affine(1., 1.))?.affine(1., 0.)).unwrap();
        let want: Vec<f32> = (0..1024).map(|i| 2. * i as f32 + 2.).collect();
        assert_eq!(
            z.to_vec1::<f32>().unwrap(),
            want,
            "a carried result lost its numbers"
        );
        let y_want: Vec<f32> = (0..1024).map(|i| 2. * i as f32 + 1.).collect();
        assert_eq!(
            y.to_vec1::<f32>().unwrap(),
            y_want,
            "a later stage wrote over an earlier one's carried result"
        );

        // Two results of 4 KiB stand at the top; the bottom cannot reach them.
        assert!(
            outer.alloc(capacity - 4096, 256).is_err(),
            "the cursor carved into a standing carry"
        );
        outer.alloc(capacity - 3 * 4096, 256).unwrap();

        drop(outer);
        assert_eq!(guest_domain_stats(ordinal).unwrap().0, 0);
    }

    /// **A chain carries two results however long it is**, and each stage reads
    /// the one before it. Ten stages of 4 KiB leave 8 KiB carried, not 40 — the
    /// bottom can still carve all but that — and the numbers are the fold's.
    #[test]
    #[cfg(feature = "cuda")]
    fn a_chain_carries_two_results_however_long_it_is() {
        use crate::kv_cache::{
            begin_guest, close_guest_arena, guest_chain, guest_domain_stats, open_guest_arena,
        };
        use candle::cuda_backend::cudarc::driver::{CudaStream, DevicePtr};
        use candle::{Device, Tensor};
        use std::sync::Arc;

        let _one = guest_arena_test();
        let Ok(device) = Device::new_cuda(0) else {
            return;
        };
        let cuda = device.as_cuda_device().unwrap();
        let stream = cuda.cuda_stream();
        let ordinal = stream.context().ordinal();
        let capacity = 1usize << 20;
        let ground = unsafe { stream.alloc::<u8>(capacity).unwrap() };
        let (base, _sync) = ground.device_ptr(&stream);
        struct CloseOnDrop(Arc<CudaStream>);
        impl Drop for CloseOnDrop {
            fn drop(&mut self) {
                close_guest_arena(&self.0);
            }
        }
        open_guest_arena(&stream, base, capacity);
        let _close = CloseOnDrop(stream.clone());

        let outer = begin_guest(&stream).unwrap();
        let x = Tensor::arange(0f32, 1024., &device).unwrap();
        let steps: Vec<f64> = (1..=10).map(f64::from).collect();
        let y = guest_chain(&x, &steps, |h, k| h.affine(1., *k)).unwrap();
        assert_eq!(
            guest_domain_stats(ordinal).unwrap().0,
            0,
            "a stage did not release"
        );
        let want: Vec<f32> = (0..1024).map(|i| i as f32 + 55.).collect();
        assert_eq!(
            y.to_vec1::<f32>().unwrap(),
            want,
            "the chain is not the fold"
        );
        outer.alloc(capacity - 3 * 4096, 256).unwrap();
        drop(outer);
        assert_eq!(guest_domain_stats(ordinal).unwrap().0, 0);
    }
}
