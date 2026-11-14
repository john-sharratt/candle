//! Loading a model's dense weights into the device reservation.
//!
//! Two calls bracket a checkpoint load, and every quantized model's loader makes
//! them:
//!
//! ```ignore
//! let _load = dense_span::open_for_load(device, content)?;   // before the first tensor
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

#[cfg(feature = "cuda")]
use candle::cuda_backend::LoadWindow;
use candle::quantized::ggml_file::qtensor_from_ggml;
use candle::quantized::gguf_file::TensorInfo;
#[cfg(feature = "cuda")]
use candle::quantized::ko_quant::ko_tileable;
use candle::quantized::{gguf_file, GgmlDType, Int8Mode, QTensor};
use candle::{Device, Result, Tensor};

use crate::models::quantized_matmul::QMatMul;

/// A model load in progress: open from [`open_for_load`] until the loader that
/// holds it returns.
///
/// It stands the device's [`LoadWindow`], so the load's own allocations — the
/// ones [`transient`] does not serve from the load's scratch, and the layer
/// pack's build after [`close_load`] — go to the driver, and a driver allocation
/// made outside every load is the `driver-alloc-panic` feature's to refuse. Held by the loader for its whole
/// body (`let _load = …`), so everything that builds the model is inside it.
#[must_use = "the load window closes when this drops: bind it for the loader's whole body"]
pub struct Load {
    #[cfg(feature = "cuda")]
    _window: Option<LoadWindow>,
}

/// Claim the device reservation before a checkpoint's tensors are read, and open
/// the load's window.
///
/// The reservation is claimed when none exists yet, and this load's weights then
/// become span tenants. A reservation that already exists — a second model in the
/// same process — leaves the load to the driver, as before the span could hold
/// it. Neither is an error.
///
/// A no-op off CUDA.
pub fn open_for_load(device: &Device, content: &gguf_file::Content) -> Result<Load> {
    open_for_load_sized(device, peak_load_pool_bytes(content.tensor_infos.values()))
}

/// [`open_for_load`] with the pool headroom already computed — for a load that reads some of
/// its tensors from outside `content` (see `qwen35::TensorOverrides::effective_infos`), whose
/// largest source is therefore not the one `content`'s header names.
pub fn open_for_load_sized(device: &Device, headroom: usize) -> Result<Load> {
    #[cfg(feature = "cuda")]
    {
        if let Device::Cuda(cuda) = device {
            let window = LoadWindow::open(cuda.cuda_stream().context().ordinal());
            let claimed = candle_nn::kv_cache::ensure_reservation(device, headroom)?;
            if claimed {
                tracing::info!(
                    target: "candle_transformers::dense_span",
                    headroom_mib = headroom >> 20,
                    "device reservation claimed before load; the largest source tensor and the \
                     repack's bounded f32 band are served from the load's scratch at the span's \
                     top until close_load"
                );
            } else {
                tracing::info!(
                    target: "candle_transformers::dense_span",
                    "a reservation already exists — this model's weights load through the \
                     driver, as every model did before the span could hold them"
                );
            }
            return Ok(Load {
                _window: Some(window),
            });
        }
    }
    #[cfg(not(feature = "cuda"))]
    let _ = headroom;
    let _ = device;
    Ok(Load {
        #[cfg(feature = "cuda")]
        _window: None,
    })
}

/// Open a load window without touching the reservation — for the construction a
/// model's runtime wrapper does once over a model already loaded (its RoPE
/// tables, its fixed buffers), which is part of loading it.
pub fn open_window(device: &Device) -> Load {
    #[cfg(feature = "cuda")]
    {
        if let Device::Cuda(cuda) = device {
            return Load {
                _window: Some(LoadWindow::open(cuda.cuda_stream().context().ordinal())),
            };
        }
    }
    let _ = device;
    Load {
        #[cfg(feature = "cuda")]
        _window: None,
    }
}

/// Move a weight that is used as stored — not repacked — into the dense block.
///
/// A repacked weight's twin is born in the dense block ([`QMatMul`] asks
/// `claim_dense` for it), but a weight read as it was loaded stayed where
/// loading put it: the CUDA pool. The token embedding is one, and on
/// Flash-Next it is 644 MiB of Q8_0 — most of what the pool still held after
/// load, outside the reservation for the life of the process.
///
/// The copy is claimed with the matmul kernels' row padding included and the
/// block's tail is zero (the dense block is never reused), so the placed tensor
/// reads exactly as the loaded one did. When the block declines — no
/// reservation, a second model in the process — the weight stays where it was.
///
/// [`QMatMul`]: crate::models::quantized_matmul::QMatMul
pub fn place_in_dense(device: &Device, weight: QTensor) -> Result<QTensor> {
    #[cfg(feature = "cuda")]
    {
        use candle::cuda_backend::cudarc::driver::result::memcpy_dtod_async;
        use candle::cuda_backend::WrapErr;
        use candle::quantized::cuda::{padded_storage_bytes, view_repacked};
        use candle::quantized::QStorage;

        let (Device::Cuda(cuda), QStorage::Cuda(src)) = (device, weight.storage()) else {
            return Ok(weight);
        };
        let dtype = weight.dtype();
        let payload = src.storage_size_in_bytes();
        let extent = padded_storage_bytes(weight.shape().elem_count(), dtype);
        let stream = cuda.cuda_stream();
        let at = match candle_nn::kv_cache::claim_dense(&stream, extent) {
            Ok(at) => at,
            Err(e) => {
                tracing::debug!(
                    target: "candle_transformers::dense_span",
                    extent,
                    "dense block declined a stored weight ({e}); it stays in the CUDA pool"
                );
                return Ok(weight);
            }
        };
        // SAFETY: `at` names `extent` ≥ `payload` bytes of the dense block, claimed
        // for this weight alone; the source is the loaded weight's own payload,
        // alive until the copy below has been queued ahead of its free.
        unsafe { memcpy_dtod_async(at, src.data_ptr(), payload, stream.cu_stream()) }.w()?;
        // SAFETY: as above — the dense block is never released, so the view's
        // memory outlives every user of it.
        let storage = unsafe { view_repacked(cuda, at, payload, extent, dtype)? };
        QTensor::new(storage, weight.shape().clone())
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = device;
        Ok(weight)
    }
}

/// A weight used as stored, copied from its bytes on the host straight into the
/// dense block — the [`place_in_dense`] of a tensor that was never on the card.
///
/// Read through the driver and then copied, a stored weight stands on the card
/// twice for a moment, and the load headroom has to hold the transient copy: on
/// Flash-Next the token embedding, 644 MiB, the largest tensor the load reads.
/// The headroom stays outside the span for the life of the process
/// (`docs/driver_model.md` §4), so that moment cost the expert zone 644 MiB for
/// good. From the host the weight costs its dense claim and nothing beside it.
///
/// `bytes` is the tensor as GGUF lays it out — whole blocks, row-major — and
/// `dims` its candle shape. When the block declines — no reservation, a second
/// model in the process — the weight is built where any loaded tensor is.
pub fn place_host_in_dense(
    device: &Device,
    bytes: &[u8],
    dims: &[usize],
    dtype: GgmlDType,
) -> Result<QTensor> {
    let elems: usize = dims.iter().product();
    let payload = elems / dtype.block_size() * dtype.type_size();
    if bytes.len() != payload {
        candle::bail!(
            "dense placement: {} bytes for a {dims:?} {dtype:?} tensor of {payload}",
            bytes.len()
        );
    }
    #[cfg(feature = "cuda")]
    {
        use candle::cuda_backend::cudarc::driver::result::memcpy_htod_async;
        use candle::cuda_backend::WrapErr;
        use candle::quantized::cuda::{padded_storage_bytes, view_repacked};

        if let Device::Cuda(cuda) = device {
            let extent = padded_storage_bytes(elems, dtype);
            let stream = cuda.cuda_stream();
            match candle_nn::kv_cache::claim_dense(&stream, extent) {
                Ok(at) => {
                    // SAFETY: `at` names `extent` ≥ `payload` bytes of the dense
                    // block, claimed for this weight alone; `bytes` is borrowed
                    // until the synchronise below has seen the copy finish.
                    unsafe { memcpy_htod_async(at, bytes, stream.cu_stream()) }.w()?;
                    stream.synchronize().w()?;
                    // SAFETY: the dense block is never released, so the view's
                    // memory outlives every user of it.
                    let storage = unsafe { view_repacked(cuda, at, payload, extent, dtype)? };
                    return QTensor::new(storage, dims);
                }
                Err(e) => tracing::debug!(
                    target: "candle_transformers::dense_span",
                    extent,
                    "dense block declined a stored weight ({e}); it is built on the driver's ground"
                ),
            }
        }
    }
    qtensor_from_ggml(dtype, bytes, dims.to_vec(), device)
}

/// Run `f` with its device allocations served from the load's scratch at the
/// span's top (`load_scratch`), for a read whose result is a twin in the dense
/// block: the source `f` reads whole, and the repack's bands, are gone when it
/// returns. A twin the dense block declines would be left standing in the
/// scratch, and `close_load` refuses it there. A no-op off CUDA.
pub fn transient<R>(f: impl FnOnce() -> R) -> R {
    #[cfg(feature = "cuda")]
    {
        candle::cuda_backend::load_scratch::transient(f)
    }
    #[cfg(not(feature = "cuda"))]
    {
        f()
    }
}

/// Whether a load builds this tensor's KO twin from its bytes on the host, a
/// band at a time, so the tensor never stands on the card whole.
///
/// The conditions the banded route itself needs, asked before a byte is read
/// so the load headroom can leave the tensor out: an int8 mode (at `Off` there
/// is no twin to build), a 2-D shape the KO matmul tiles, and a source the
/// banded repack reads. Off CUDA nothing repacks, so nothing bands.
pub fn bands_from_host(info: &TensorInfo, mode: Int8Mode) -> bool {
    #[cfg(feature = "cuda")]
    if let [nrows, ncols] = info.shape.dims() {
        return mode.is_int8()
            && ko_tileable(*nrows, *ncols)
            && candle::quantized::cuda::repackable_to_ko(info.ggml_dtype);
    }
    let _ = (info, mode);
    false
}

/// A projection's KO twin, repacked from its bytes on the host a band at a time
/// into the dense block (`cuda::repack_ko_from_host`).
///
/// The device holds a staging band, the f32 band and the twin, never the source
/// — the route the qwen35 loader takes for every mapped projection, here for a
/// load that reads its file by offset. `Ok(None)` when [`bands_from_host`]
/// refuses the tensor: the caller reads it the ordinary way, and the load
/// headroom has counted it.
pub fn twin_from_host(
    device: &Device,
    bytes: &[u8],
    info: &TensorInfo,
    mode: Int8Mode,
) -> Result<Option<QMatMul>> {
    if !bands_from_host(info, mode) {
        return Ok(None);
    }
    #[cfg(feature = "cuda")]
    {
        use candle::quantized::cuda::repack_ko_from_host;
        use candle::quantized::QStorage;

        use crate::models::quantized_matmul::dense_destination_for;

        let Device::Cuda(cuda) = device else {
            return Ok(None);
        };
        let shape = info.shape.clone();
        let ko = info.ggml_dtype.to_ko(mode)?;
        let dst = dense_destination_for(device, &shape, ko);
        let twin = repack_ko_from_host(cuda, bytes, &shape, info.ggml_dtype, ko, dst)?;
        let qt = QTensor::new(QStorage::Cuda(twin), shape)?;
        Ok(Some(QMatMul::from_qtensor_view(qt, mode)?))
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (device, bytes);
        Ok(None)
    }
}

/// Move a plain tensor a model keeps for its whole life — a norm's gain, a
/// gate, a scale — into the dense block.
///
/// The loader builds it where any tensor is built, the driver's ground, and
/// there it stayed: on Flash-Next 298 such tensors, 91 MiB, held outside the
/// reservation for the life of the process (the `driver-alloc-panic` census of a
/// load's survivors). Copied into a dense claim, the driver's copy is a load
/// transient and goes when the caller drops it. When the block declines — no
/// reservation, a second model in the process — the tensor stays where it was.
pub fn place_tensor_in_dense(device: &Device, t: Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        use candle::cuda_backend::cudarc::driver::result::memcpy_dtod_async;
        use candle::cuda_backend::wave_provenance::LeaseOrigin;
        use candle::cuda_backend::WrapErr;

        let Device::Cuda(cuda) = device else {
            return Ok(t);
        };
        let t = t.contiguous()?;
        let bytes = t.elem_count() * t.dtype().size_in_bytes();
        if bytes == 0 {
            return Ok(t);
        }
        let stream = cuda.cuda_stream();
        let at = match candle_nn::kv_cache::claim_dense(&stream, bytes) {
            Ok(at) => at,
            Err(e) => {
                tracing::debug!(
                    target: "candle_transformers::dense_span",
                    bytes,
                    "dense block declined a tensor ({e}); it stays on the driver's ground"
                );
                return Ok(t);
            }
        };
        let src = {
            let (storage, layout) = t.storage_and_layout();
            let candle::Storage::Cuda(c) = &*storage else {
                candle::bail!("dense placement: a CUDA tensor without CUDA storage");
            };
            c.slice.device_ptr(&stream) + (layout.start_offset() * t.dtype().size_in_bytes()) as u64
        };
        // SAFETY: `at` names `bytes` of the dense block, claimed for this tensor
        // alone; `src` is the tensor's own contiguous payload, which `t` keeps
        // alive until the copy below has been queued ahead of its free.
        unsafe { memcpy_dtod_async(at, src, bytes, stream.cu_stream()) }.w()?;
        // SAFETY: the dense block is never released, so the view outlives every
        // user of it; nothing frees a `Foreign` lease.
        unsafe {
            Tensor::from_leased_cuda_ptr(
                at,
                t.dtype(),
                t.dims().to_vec(),
                device,
                LeaseOrigin::Foreign,
            )
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = device;
        Ok(t)
    }
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
    freeze(device)
}

/// Lock the dense block's right edge. Every weight-side address is measured from
/// it, so nothing may size a zone until this has run.
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
/// repack and a stored weight read to the device whole, which does not repack but stands there
/// at the same size.
///
/// Takes the tensors rather than the header so a load that substitutes some of them from
/// another file bounds what it will actually read — and so a load leaves out the tensors it
/// reads on the host ([`bands_from_host`], [`place_host_in_dense`]), which never stand on the
/// card whole. That matters for good: the span never grows, so whatever this bound names stays
/// outside it for the life of the process (`docs/driver_model.md` §4).
pub fn peak_load_pool_bytes<'a>(
    infos: impl IntoIterator<Item = &'a gguf_file::TensorInfo>,
) -> usize {
    let largest_source = infos
        .into_iter()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn info(dtype: GgmlDType, dims: &[usize]) -> TensorInfo {
        TensorInfo {
            ggml_dtype: dtype,
            shape: dims.into(),
            offset: 0,
        }
    }

    /// Off the card a stored weight is built from its bytes as any loaded one
    /// is, and reads back exactly.
    #[test]
    fn a_stored_weight_is_built_from_its_host_bytes() -> Result<()> {
        let bytes: Vec<u8> = [1.0f32, -2.5, 0.25]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let t = place_host_in_dense(&Device::Cpu, &bytes, &[3], GgmlDType::F32)?;
        assert_eq!(
            t.dequantize(&Device::Cpu)?.to_vec1::<f32>()?,
            vec![1.0, -2.5, 0.25]
        );
        Ok(())
    }

    /// Bytes that are not the tensor's whole payload are refused, never read
    /// short or past.
    #[test]
    fn a_short_payload_is_refused() {
        let err = place_host_in_dense(&Device::Cpu, &[0u8; 11], &[3], GgmlDType::F32)
            .expect_err("11 bytes are not three f32s");
        assert!(err.to_string().contains("11 bytes"), "{err}");
    }

    /// Nothing bands without an int8 twin to build, or a 2-D shape the KO
    /// matmul tiles.
    #[test]
    fn nothing_bands_without_a_twin_or_a_tile() {
        let tiled = info(GgmlDType::Q8_0, &[64, 256]);
        assert!(!bands_from_host(&tiled, Int8Mode::Off));
        assert!(!bands_from_host(
            &info(GgmlDType::Q8_0, &[16, 256]),
            Int8Mode::Performance
        ));
        assert!(!bands_from_host(
            &info(GgmlDType::Q8_0, &[64, 64]),
            Int8Mode::Performance
        ));
        assert!(!bands_from_host(
            &info(GgmlDType::F32, &[2048]),
            Int8Mode::Precision
        ));
    }

    /// A tiled 2-D source the banded repack reads bands under either int8 mode,
    /// a float one included.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_tiled_source_bands_under_an_int8_mode() {
        for dtype in [GgmlDType::Q8_0, GgmlDType::Q6_K, GgmlDType::BF16] {
            let tiled = info(dtype, &[64, 256]);
            assert!(bands_from_host(&tiled, Int8Mode::Performance), "{dtype:?}");
            assert!(bands_from_host(&tiled, Int8Mode::Precision), "{dtype:?}");
        }
    }
}
