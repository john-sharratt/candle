//! Wave-scoped buffers for the inference loop.
//!
//! An attention output is the archetypal wave intermediate: written by one
//! kernel, consumed by `o_proj`, dead immediately after. Inside the inference
//! loop it comes from the wave's transient half — one cursor bump, no allocator
//! traffic, and the whole half reclaimed when the guard drops
//! (`docs/archived/arena_unification.md` §3.6). On the decode path that replaces an
//! alloc/free pair per layer per forward.
//!
//! The guard bounding these is **layer-scoped**, spanning attention ->
//! `o_proj`. That is deliberate and was learned the hard way: a guard held for
//! the whole forward keeps every layer's output live at once, so consumption
//! grows with depth instead of staying at one layer's working set. Halves
//! alternate per layer, which puts a full layer of same-stream work between one
//! layer's reads and the next reuse of that half.
//!
//! Outside the inference loop — kernel tests, replay harnesses, the `decode_ab`
//! and `prefill_ab` fixtures — there is no generation to bound the lifetime, so
//! the caller allocates and owns its buffer as before. The absence is real
//! state, not a mode: [`wave_alloc`] reports whether a wave is actually in
//! flight.
//!
//! # Scope
//!
//! *Our* kernels take preallocated leased buffers, because each has an
//! allocation site to redirect. Interior op outputs — the temporaries candle's
//! own ops allocate, including the inter-layer hidden state, which is only ever
//! the result of a residual add — have no such site and stay on the pool
//! remnant. Redirecting those would need an allocator scope that captures
//! `device.alloc` for the extent of a wave, so that every interior output lands
//! in the wave plan with no call-site changes. No such scope exists: the
//! remnant is bounded by `scratch_margin` and sits outside the reservation,
//! where it costs address space rather than KV capacity.
//!
//! The MoE combine target is here too, via [`wave_zeros`]. It is *returned*
//! from the expert forward, so nothing inside the MoE code bounds it — the
//! bound comes from one level up, where the layer opens a generation around
//! `ffn_forward` and the residual add that consumes the result. That is the
//! same layer scoping the attention path uses, applied to the layer's other
//! half.

use candle::cuda_backend::cudarc::driver::result::memset_d8_async;
use candle::cuda_backend::cudarc::driver::{
    CudaSlice, CudaStream, DevicePtr, DeviceRepr, SyncOnDrop,
};
use candle::cuda_backend::CudaDType;
use candle::{CudaDevice, CudaStorage, DType, Device, Result, Shape, Tensor};
use candle_nn::kv_cache::wave_alloc;

/// Alignment for every wave buffer.
///
/// Matches what `cudaMalloc` guarantees, so a leased buffer is as aligned as
/// the owned one it replaces for every vectorised access the kernels make.
const WAVE_ALIGN: usize = 256;

/// Where a kernel writes its output.
pub(crate) enum KernelOutput<T> {
    /// A range of the in-flight wave's half. The wave owns it; this does not.
    ///
    /// `elems`, not bytes — `BumpRange::len` is bytes, and the two are the same
    /// number only for `u8` outputs.
    Leased { ptr: u64, elems: usize },
    /// This op's own allocation, freed when the storage drops.
    Owned(CudaSlice<T>),
}

impl<T: CudaDType + DeviceRepr> KernelOutput<T> {
    /// Reserve room for `elem_count` elements of `T`.
    pub(crate) fn new(dev: &CudaDevice, elem_count: usize) -> Result<Self> {
        let bytes = elem_count * std::mem::size_of::<T>();
        match wave_alloc(&dev.cuda_stream(), bytes, WAVE_ALIGN)? {
            Some(range) => Ok(Self::Leased {
                ptr: range.ptr,
                elems: elem_count,
            }),
            // SAFETY: the storage is written by the kernel launched at the call
            // site before anything reads it, exactly as when it was allocated
            // inline there.
            None => Ok(Self::Owned(unsafe { dev.alloc::<T>(elem_count)? })),
        }
    }

    /// The destination address, plus the stream guard the owned arm needs.
    ///
    /// A leased range needs none: the half is not handed out again until a
    /// whole layer's work has been issued on the same stream, which
    /// subsumes the per-slice dependency `device_ptr` records. Callers hold the
    /// returned guard across the launch, as they did the one
    /// `CudaSlice::device_ptr` gave them.
    pub(crate) fn device_ptr<'a>(
        &'a self,
        stream: &'a CudaStream,
    ) -> (u64, Option<SyncOnDrop<'a>>) {
        match self {
            Self::Leased { ptr, .. } => (*ptr, None),
            Self::Owned(slice) => {
                let (ptr, guard) = slice.device_ptr(stream);
                (ptr, Some(guard))
            }
        }
    }

    /// Hand the output to candle as storage.
    pub(crate) fn into_storage(self, dev: CudaDevice) -> CudaStorage {
        match self {
            // SAFETY: `ptr` is `elems` elements of `T` in the current half,
            // kept live by the layer's generation guard — which spans this
            // kernel through `o_proj`, and so outlives every tensor derived
            // from this storage.
            Self::Leased { ptr, elems } => unsafe {
                CudaStorage::wrap_leased_ptr::<T>(ptr, elems, dev)
            },
            Self::Owned(slice) => CudaStorage::wrap_cuda_slice(slice, dev),
        }
    }
}

/// A zeroed tensor on the in-flight wave's half, or an ordinary one when no
/// wave is in flight.
///
/// For accumulators, where the caller needs the buffer to *start* at zero — the
/// MoE combine target is scattered into, not overwritten, so it cannot take a
/// wave range as-is. The fill is `memset` on the device's stream, which is the
/// same work `Tensor::zeros` does; what it replaces is the allocate/free pair
/// around it, one per MoE layer per forward.
pub(crate) fn wave_zeros<S: Into<Shape>>(
    shape: S,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let shape = shape.into();
    let Device::Cuda(cuda) = device else {
        return Tensor::zeros(shape, dtype, device);
    };
    let stream = cuda.cuda_stream();
    let bytes = shape.elem_count() * dtype.size_in_bytes();
    let Some(range) = wave_alloc(&stream, bytes, WAVE_ALIGN)? else {
        return Tensor::zeros(shape, dtype, device);
    };
    // SAFETY: `range` is `bytes` of the current half, held by the caller's
    // generation guard, and nothing else addresses it within this generation.
    unsafe { memset_d8_async(range.ptr, 0, bytes, stream.cu_stream()) }
        .map_err(|e| candle::Error::Msg(format!("zeroing a wave buffer: {e}")))?;
    // SAFETY: as above — and the tensor never outlives the guard, because the
    // layer drops it after the residual add that consumes this buffer.
    unsafe { Tensor::from_leased_cuda_ptr(range.ptr, dtype, shape, device) }
}
