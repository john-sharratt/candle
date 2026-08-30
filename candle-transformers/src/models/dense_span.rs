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
            let headroom = peak_load_pool_bytes(content);
            let claimed = candle_nn::kv_cache::ensure_reservation(device, headroom)?;
            if claimed {
                tracing::info!(
                    target: "candle_transformers::dense_span",
                    headroom_mib = headroom >> 20,
                    "device reservation claimed before load; the largest source tensor and the \
                     repack's bounded f32 band stay with the pool"
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
    let dense = freeze(device)?;
    reclaim_headroom(device)?;
    Ok(dense)
}

/// Lock the dense block's right edge. Every weight-side address is measured from
/// it, so nothing may size a zone until this has run.
///
/// Split from [`reclaim_headroom`] because the two have different deadlines: the
/// freeze must happen *before* anything carves a zone, and the reclaim must
/// happen *after* the last repack that draws on the CUDA pool. For a resident
/// model there is nothing in between and [`close_load`] does both; a streamed one
/// builds its layer pack in the gap.
pub fn freeze(device: &Device) -> Result<usize> {
    #[cfg(feature = "cuda")]
    {
        if let Device::Cuda(cuda) = device {
            let dense = candle_nn::kv_cache::freeze_dense(&cuda.cuda_stream())?;
            tracing::info!(
                target: "candle_transformers::dense_span",
                dense_mib = dense >> 20,
                "dense block locked; every weight-side address is now fixed"
            );
            return Ok(dense);
        }
    }
    let _ = device;
    Ok(0)
}

/// Give the load headroom back to the span.
///
/// **Only once every repack that draws on the CUDA pool has finished.**
/// `open_for_load` conceded this ground so the pool could hold the largest source
/// tensor and the repack's bands ([`peak_load_pool_bytes`]); taking it back while
/// a repack still needs it leaves that pass with nothing but the governor's
/// cushion.
///
/// That is not hypothetical — it was the ordering in-tree. `close_load` reclaimed
/// here and was then called *before* `build_layers`, which repacks a whole layer
/// at a time through `WeightResidency::Pool` to build the layer pack. Both
/// `layer_stream::build`'s header and `docs/qwen38_layer_streaming.md` §12.2 say
/// `peak_load_pool_bytes` is what reserves room for exactly that pass, so the
/// invariant was stated in two places and violated in one.
///
/// Growing the span moves its right edge and the weight side is placed downward
/// from it, so this must still precede anything that carves a zone.
/// `reclaim_load_headroom` re-checks that for itself.
pub fn reclaim_headroom(device: &Device) -> Result<usize> {
    #[cfg(feature = "cuda")]
    {
        if let Device::Cuda(cuda) = device {
            let reclaimed = candle_nn::kv_cache::reclaim_load_headroom(&cuda.cuda_stream())?;
            tracing::info!(
                target: "candle_transformers::dense_span",
                reclaimed_mib = reclaimed >> 20,
                "load headroom returned to the span; the remainder is runtime ground"
            );
            return Ok(reclaimed);
        }
    }
    let _ = device;
    Ok(0)
}

/// CUDA-pool room the load needs, in bytes: the largest single tensor's **quantized** size,
/// plus the repack's two bands.
///
/// **An arithmetic bound, not an estimate**, read from the GGUF header — so it costs nothing
/// and is known before a byte of tensor data is touched, which is what lets the span be claimed
/// first. The load reads one tensor at a time and drops it, so the pool's peak is one source
/// tensor — `elems / block_size × type_size` — and the bands the repack holds beside it.
///
/// # This used to be the f32 expansion, and that was the largest claim on the card
///
/// `repack_ko` composes dequantize-then-requantize through a buffer, and that buffer is the
/// whole tensor in f32 — `nrows × ncols × 4` however few bits the source and twin use. On the
/// 27B's `[248320, 5120]` head it was **4,850 MiB** to produce a 993 MiB twin, and the span
/// conceded exactly that much. Not for the load: *permanently*. `cuMemAddressReserve` sizes the
/// span once and cannot grow, and the weight zone and KV regions are carved from the span — so
/// a buffer that lived for one tensor during load held a third of the card until the process
/// exited.
///
/// The repack now runs a row band at a time (see `QCudaStorage::repack_ko_into`), so the
/// intermediate is `REPACK_BAND_BYTES` whatever the tensor's size and what remains is the
/// source tensor plus that constant. On that same head: **995 MiB instead of 4,850**, returning
/// ~3.8 GiB to the span. Host-mapping the intermediate was tried first and does not work at
/// these sizes — the band stays in VRAM.
///
/// # Why it is not zero
///
/// Two reasons, and both terms are load-bearing. The source arrives through `dev.alloc` — the
/// CUDA pool — before anything repacks it; and the bands are device memory, live at the same
/// time as that source. Sizing this at zero, or dropping the band term as "small", leaves the
/// pool nothing but the governor's cushion and OOMs on the first large repack. Getting to zero
/// means fusing the dequant into `quantize_ko` and dequantizing straight from the mapped GGUF,
/// so neither the source nor an intermediate reaches VRAM; that is a further change to the
/// loader and the kernels, not to this bound.
///
/// Two-dimensional tensors only. Expert banks are 3-D and are repacked per-expert by the expert
/// cache; norms are 1-D and are never repacked. The max over 2-D covers both the weights that
/// repack and the embedding, which does not repack but is read to the device at the same size.
pub fn peak_load_pool_bytes(content: &gguf_file::Content) -> usize {
    let largest_source = content
        .tensor_infos
        .values()
        .filter_map(|info| match info.shape.dims() {
            [rows, cols] => {
                let elems = rows * cols;
                let d = info.ggml_dtype;
                Some(elems / d.block_size() * d.type_size())
            }
            _ => None,
        })
        .max()
        .unwrap_or(0);
    // The repack's f32 band and the KO band it fills are live at the same time as the source
    // they are repacking, so they add rather than overlap. The KO band is the smaller of the
    // two by construction (it is the quantized form of the same rows), so one band's worth of
    // allowance on top covers both.
    largest_source + 2 * candle::quantized::cuda::REPACK_BAND_BYTES
}
