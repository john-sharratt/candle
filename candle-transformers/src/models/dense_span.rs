//! Loading a model's dense weights into the device reservation.
//!
//! Two calls bracket a checkpoint load, and every quantized model's loader makes
//! them:
//!
//! ```ignore
//! dense_span::open_for_load(device, content)?;   // before the first tensor
//! …                                             // weights land in the span
//! dense_span::close_load(device)?;               // before anything else claims
//! ```
//!
//! # Why the bracket exists
//!
//! The span used to be sized from the VRAM left *after* the weights were
//! resident, so it was measured against the hole they had already taken from the
//! CUDA pool and could not, even in principle, contain them. [`open_for_load`]
//! inverts that: it claims the reservation first, from the whole card, and the
//! weights are then carved out of it as they arrive
//! ([`QMatMul`](crate::models::quantized_matmul::QMatMul) asks
//! `claim_dense` for each repacked twin). Measured on Qwen3.6-35B-A3B: 1,827 MiB
//! of model moved inside the reservation and the pool's post-load footprint fell
//! from 1,952 MiB to 32 MiB.
//!
//! Nothing predicts the model's size. The dense block grows to whatever the
//! checkpoint needs and the KV region count is derived from the remainder at
//! [`close_load`].
//!
//! # Why it is a shared module rather than a few lines per loader
//!
//! There are eight of these loaders. The bracket is three facts —
//! the ordering, the headroom bound, and the freeze — and a copy in each loader
//! is eight places for one of the three to be dropped or to drift. A loader that
//! forgets [`close_load`] still works (the first region claim freezes the block
//! as a backstop) which is exactly what makes the omission invisible, so the
//! call is worth having one definition of.

use candle::quantized::gguf_file;
use candle::{Device, Result};

/// Claim the device reservation before a checkpoint's tensors are read.
///
/// Returns `true` when this call claimed it, meaning this load's weights become
/// span tenants. `false` means a reservation already existed — a second model in
/// the same process — and the load proceeds through the CUDA pool exactly as it
/// did before any of this. Neither is an error, and a caller that does not care
/// which happened can ignore the answer.
///
/// A no-op off CUDA.
pub fn open_for_load(device: &Device, content: &gguf_file::Content) -> Result<bool> {
    #[cfg(feature = "cuda")]
    {
        if matches!(device, Device::Cuda(_)) {
            let headroom = peak_repack_scratch(content);
            let claimed = candle_nn::kv_cache::ensure_reservation(device, headroom)?;
            if claimed {
                tracing::info!(
                    target: "candle_transformers::dense_span",
                    headroom_mib = headroom >> 20,
                    "device reservation claimed before load; the repack peak stays with the pool"
                );
            } else {
                tracing::info!(
                    target: "candle_transformers::dense_span",
                    "a reservation already exists — this model's weights load through the \
                     CUDA pool, as every model did before the span could hold them"
                );
            }
            return Ok(claimed);
        }
    }
    #[cfg(not(feature = "cuda"))]
    let _ = content;
    let _ = device;
    Ok(false)
}

/// End the load phase: lock the dense block and hand the rest of the span to the
/// runtime.
///
/// Must run before anything claims a KV region or installs an expert zone — both
/// measure their addresses from the dense block's right edge, and it has to be
/// final by then. Returns the bytes the model actually took, which is a
/// measurement rather than a prediction.
///
/// A no-op off CUDA, and harmless when [`open_for_load`] returned `false`: the
/// block is empty, so freezing it locks nothing.
pub fn close_load(device: &Device) -> Result<usize> {
    #[cfg(feature = "cuda")]
    {
        if let Device::Cuda(cuda) = device {
            let dense = candle_nn::kv_cache::freeze_dense(&cuda.cuda_stream())?;
            tracing::info!(
                target: "candle_transformers::dense_span",
                dense_mib = dense >> 20,
                "load phase closed; the dense block is locked and the remainder is runtime ground"
            );
            return Ok(dense);
        }
    }
    let _ = device;
    Ok(0)
}

/// Device scratch the largest single weight's repack needs, in bytes.
///
/// **An arithmetic bound, not an estimate.** `QCudaStorage::repack_ko`
/// dequantizes a whole tensor to F32 before quantizing it into its KO twin —
/// `let f32_storage = self.dequantize(nrows * ncols)?` — so the transient is
/// `nrows × ncols × 4` regardless of how few bits the source or the twin use. On
/// Qwen3.6-35B that is `output.weight` at `[248320, 2048]`: ~1,940 MiB of
/// scratch to produce a few hundred MiB of weight.
///
/// That scratch comes from the CUDA pool, so the span gives up exactly this much
/// and no more. Read from the GGUF header, so it costs nothing and is known
/// before a byte of tensor data is touched — which is what lets the span be
/// claimed first.
///
/// Two-dimensional tensors only. Expert banks are 3-D and are repacked
/// per-expert by the expert cache; norms are 1-D and are never repacked. The max
/// over 2-D covers both the weights that repack and the embedding, which does
/// not repack but is read to the device at the same size — so the bound holds
/// for whichever is larger without having to model the difference.
///
/// Goes to zero once the repack is chunked: the peak becomes one chunk, and this
/// ground returns to the KV side.
pub fn peak_repack_scratch(content: &gguf_file::Content) -> usize {
    content
        .tensor_infos
        .values()
        .filter_map(|info| match info.shape.dims() {
            [rows, cols] => Some(rows * cols),
            _ => None,
        })
        .max()
        .unwrap_or(0)
        * std::mem::size_of::<f32>()
}
