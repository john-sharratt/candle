//! A derived, int8-quantised copy of the corpus entries the decode kernel reads
//! at every step.
//!
//! Each finalised entry is roped once at its fixed position and quantised per
//! band, so the decode reads four times fewer bytes than the f32 gallery rows
//! and does not rotate the corpus on every query. The canonical corpus stays
//! f32, pre-RoPE and position-free; this cache is rebuilt or extended as the
//! gallery grows. The rope bands are kept before rotation, and the kernel
//! rotates them from the stored positions at read time.

use super::*;

/// Persistent roped int8 corpus cache: every finalized entry roped ONCE at
/// its fixed group-start position and per-band int8-quantized, then read by
/// each decode step — 4× smaller than the f32 gallery rows that dominated
/// decode traffic, and the per-query RoPE disappears. The canonical corpus
/// stays f32/pre-RoPE/position-free (§C); this is a derived, caller-owned
/// cache (one per layer, rebuilt/extended as the gallery grows).
#[cfg(feature = "cuda")]
pub struct CorpusCache {
    /// `[g, NOPE_DIM]` u8 (int8 bits): the NOPE bands, position-free component-amax
    /// int8. The decode kernel dequantizes with `nope_scale`.
    pub nope_i8: Tensor,
    /// `[g, NOPE_BANDS]` f32 per-nope-band amax scale.
    pub nope_scale: Tensor,
    /// `[g, ROPE_DIM]` bf16: the ROPE bands, PRE-rotation (float, matches the
    /// window ring). The decode kernel rotates them at read time from `comp_pos`.
    pub rope_bf: Tensor,
    /// `[g]` u32 assembled position per entry — rope-at-load input for decode.
    pub comp_pos: Tensor,
    pub(super) len: usize,
}

#[cfg(feature = "cuda")]
impl CorpusCache {
    /// Build the cache for the whole corpus (entries `[0, g)`). Callers with
    /// no corpus pass dummy tensors whose SHAPE may be one row while the
    /// storage is empty — the pre-pass launches only over rows that both the
    /// data and the positions actually back (an unbacked row's device pointer
    /// is null); unbuilt rows are never read (their gids are never selected).
    pub fn build(comp: &Tensor, comp_pos: &Tensor) -> Result<Self> {
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::Storage;
        // Clamp by ACTUAL storage length, not shape: dummy corpora carry a
        // one-row shape over empty storage (whose device pointer is null).
        let storage_rows = |t: &Tensor, per_row: usize| -> Result<usize> {
            let (storage, _) = t.storage_and_layout();
            Ok(match &*storage {
                Storage::Cuda(c) => {
                    c.as_cuda_slice::<f32>().map(|s| s.len()).unwrap_or(0) / per_row
                }
                _ => 0,
            })
        };
        let g = comp.dim(0)?.min(storage_rows(comp, HEAD_DIM)?).min({
            let (storage, _) = comp_pos.storage_and_layout();
            match &*storage {
                Storage::Cuda(c) => c.as_cuda_slice::<u32>().map(|s| s.len()).unwrap_or(0),
                _ => 0,
            }
        });
        let dev = match comp.device() {
            candle::Device::Cuda(d) => d.clone(),
            _ => candle::bail!("CorpusCache requires CUDA tensors"),
        };
        let stream = dev.cuda_stream();
        let nope_i8 = Tensor::zeros((g.max(1), NOPE_DIM), DType::U8, comp.device())?;
        let nope_scale = Tensor::zeros((g.max(1), NOPE_BANDS), DType::F32, comp.device())?;
        let rope_bf = Tensor::zeros((g.max(1), ROPE_DIM), DType::BF16, comp.device())?;
        macro_rules! p {
            ($t:expr, $ty:ty) => {{
                let (storage, layout) = $t.storage_and_layout();
                match &*storage {
                    Storage::Cuda(c) => {
                        let (p, _g) = c.as_cuda_slice::<$ty>()?.device_ptr(&stream);
                        p + (layout.start_offset() * std::mem::size_of::<$ty>()) as u64
                    }
                    _ => candle::bail!("expected CUDA storage"),
                }
            }};
        }
        if g > 0 {
            let comp_p = p!(comp, f32);
            let ni8_p = p!(&nope_i8, u8);
            let nsc_p = p!(&nope_scale, f32);
            let rbf_p = p!(&rope_bf, half::bf16);
            unsafe {
                candle_kernels::paged_latent::run_latent_build_corpus_cache(
                    comp_p as *const f32,
                    ni8_p as *mut u8,
                    nsc_p as *mut f32,
                    rbf_p as *mut core::ffi::c_void,
                    0,
                    g as i32,
                    stream.cu_stream() as *mut core::ffi::c_void,
                );
            }
        }
        // Retain the positions (narrowed to the built rows) — the decode kernel
        // rotates the rope bands from them at read time.
        let comp_pos = comp_pos.narrow(0, 0, g.max(1))?.contiguous()?;
        Ok(Self {
            nope_i8,
            nope_scale,
            rope_bf,
            comp_pos,
            len: g,
        })
    }

    /// Hold a pre-built two-region cache gathered from the gallery (no rebuild).
    /// `comp_pos` are the entries' assembled positions; `len` is the count of
    /// real entries — pass it explicitly rather than deriving from the row shape,
    /// because the empty cache carries a one-row placeholder shape over zero
    /// entries (`len = 0`), and any `is_empty()`/`len()` caller must see that.
    #[cfg(feature = "cuda")]
    pub fn from_gathered(
        nope_i8: Tensor,
        nope_scale: Tensor,
        rope_bf: Tensor,
        comp_pos: Tensor,
        len: usize,
    ) -> Result<Self> {
        Ok(Self {
            nope_i8,
            nope_scale,
            rope_bf,
            comp_pos,
            len,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}
