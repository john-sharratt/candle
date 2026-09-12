//! A GGUF checkpoint placed in guest ground, int8 twins and all.
//!
//! The quantized counterpart of [`super::varground`]. Both exist for the same
//! reason: a guest's weights cannot come from the CUDA pool. The engine's span
//! is one reservation holding nearly the whole card, and a guest drain does not
//! return it to the driver — it takes back *regions inside it* that the KV side
//! hands over ([`GuestGround::claim`]). So the pool has only whatever the span
//! left over, which is nowhere near a diffusion transformer, and a loader that
//! allocates from it fails on a card with 17 GiB free by its own accounting.
//!
//! # Why not `quantized_var_builder::VarBuilder`
//!
//! Because it allocates. `VarBuilder::from_gguf` walks the header and calls
//! `Content::tensor` for every name, each of which is a pool allocation — the
//! whole checkpoint, before the model has said which tensors it wants. This
//! places the same tensors at addresses the guest owns and then hands the model
//! a `VarBuilder` built over the results, so [`z_image::quantized_model`] does
//! not know the difference.
//!
//! # The int8 twin is built into ground, not copied there
//!
//! A projection the q8a128 matmul can tile is not placed as its file bytes at
//! all. It is uploaded to a **pool** scratch one tensor at a time, repacked
//! straight into its ground address by `repack_ko_into`, and the scratch
//! dropped — so ground holds the twin alone and the pool never holds more than
//! one source tensor plus the repack's 48 MiB band. Placing the source in ground
//! *and* the twin would need the checkpoint twice over, which on a 7 GiB model is
//! the difference between fitting and not.
//!
//! Everything else — norms, the learned pad embeddings, and any projection too
//! narrow to tile — is placed as its file bytes. Those go down with
//! [`MATRIX_ROW_PADDING`] of zeroed tail, because the dequantising matmul reads
//! into that padding and a leased view does not carry any of its own.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use candle::quantized::cuda::{ko_repacked_bytes, QCudaStorage, MATRIX_ROW_PADDING};
use candle::quantized::ko_quant::ko_tileable;
use candle::quantized::{GgmlDType, Int8Mode, QStorage, QTensor};
use candle::{Device, Shape};
use candle_transformers::quantized_var_builder::VarBuilder;

use super::checkpoint;
use super::ground::GuestGround;

/// Alignment every placement takes.
///
/// 256 rather than the 512 of [`MATRIX_ROW_PADDING`]: this aligns the *start* of
/// a tensor, which the kernels want at a texture-friendly boundary, while the
/// padding is about what may be read past the *end*. They are different
/// requirements and conflating them wastes a quarter of a gigabyte over a
/// four-hundred-tensor checkpoint.
const PLACE_ALIGN: usize = 256;

/// Read the GGUF at `path` and place every tensor in `ground`.
///
/// `mode` decides whether a tileable projection becomes its KO twin or is placed
/// as its file bytes; `Int8Mode::Off` places everything verbatim, which is what a
/// card without the int8 MMA gets.
///
/// **One barrier at the end, not one per tensor.** Every copy below is issued
/// against the stream without waiting, exactly as `GroundVars::with` does, so the
/// transfers pipeline; the sync before returning is what makes the whole set
/// ordered before any read.
/// `skip` names tensors the model will never ask for, so they are not placed.
/// Ground is the KV side's, taken for the length of a drain, and a tensor
/// nothing reads is that much of the engine's working set evicted for nothing —
/// a text encoder's embedding table is 414 MiB of a 4 GiB checkpoint.
pub fn place_gguf(
    path: &Path,
    device: &Device,
    ground: &Arc<Mutex<GuestGround>>,
    mode: Int8Mode,
    skip: &[&str],
) -> Result<VarBuilder, String> {
    let header = checkpoint::gguf_header(path).map_err(|e| format!("{path:?}: {e}"))?;
    let payload = checkpoint::payload(path).map_err(|e| format!("{path:?}: {e}"))?;
    let Device::Cuda(cuda) = device else {
        return Err("guest weights: the reservation is a CUDA allocation".into());
    };

    let mut placed: HashMap<String, Arc<QTensor>> =
        HashMap::with_capacity(header.tensor_infos.len());
    let mut twins = 0usize;
    for (name, info) in header.tensor_infos.iter() {
        if skip.contains(&name.as_str()) {
            continue;
        }
        let shape = Shape::from(info.shape.dims().to_vec());
        let dtype = info.ggml_dtype;
        let start = header.tensor_data_offset as usize + info.offset as usize;
        let len = shape.elem_count() / dtype.block_size() * dtype.type_size();
        let raw = payload.get(start..start + len).ok_or_else(|| {
            format!("{name}: the header describes bytes past the end of {path:?}")
        })?;

        let twin = mode.is_int8()
            && match shape.dims() {
                [nrows, ncols] => ko_tileable(*nrows, *ncols),
                _ => false,
            };
        // Named before it is placed, not after: a placement that faults takes
        // the process with it, and the last line written is then the only
        // evidence of which tensor did it.
        tracing::trace!(
            target: "candle_conversation::guest",
            tensor = %name, shape = ?shape.dims(), ?dtype, twin,
            "placing"
        );
        let qt = if twin {
            twins += 1;
            place_ko_twin(cuda, ground, raw, &shape, dtype, mode)
        } else {
            place_verbatim(cuda, ground, raw, &shape, dtype)
        }
        .map_err(|e| format!("{name} {:?}: {e}", shape.dims()))?;
        placed.insert(name.clone(), Arc::new(qt));
    }

    device
        .synchronize()
        .map_err(|e| format!("guest weights: the load barrier: {e}"))?;
    tracing::debug!(
        target: "candle_conversation::guest",
        tensors = placed.len(),
        int8_twins = twins,
        "placed a gguf in guest ground"
    );
    Ok(VarBuilder::from_tensors(placed, device.clone()))
}

/// Upload one tensor, repack it into ground as its KO twin, drop the source.
fn place_ko_twin(
    cuda: &candle::CudaDevice,
    ground: &Arc<Mutex<GuestGround>>,
    raw: &[u8],
    shape: &Shape,
    dtype: GgmlDType,
    mode: Int8Mode,
) -> candle::Result<QTensor> {
    let ko = dtype.to_ko(mode)?;
    let bytes = ko_repacked_bytes(shape, ko)?;
    let at = {
        let mut g = ground.lock().unwrap();
        g.place(bytes, PLACE_ALIGN)
            .map_err(|e| candle::Error::Msg(format!("placing a {ko:?} twin: {e}")))?
    };

    // The scratch is the *source* quant, on the pool, for one tensor. `zeros`
    // sizes it with the row padding the dequantise half of the repack reads
    // into, and zeroes it — so copying only the file's bytes leaves a defined
    // tail rather than whatever the pool last held.
    let mut scratch = QCudaStorage::zeros(cuda, shape.elem_count(), dtype)?;
    let stream = cuda.cuda_stream();
    // SAFETY: `zeros` allocated at least `raw.len()` bytes at this address (it
    // adds row padding on top), and the slice is `forget`ed rather than dropped
    // so it never frees storage the `QCudaStorage` owns.
    let mut dst = unsafe { stream.upgrade_device_ptr::<u8>(scratch.data_ptr_mut(), raw.len()) };
    stream
        .memcpy_htod(raw, &mut dst.slice_mut(..raw.len()))
        .map_err(candle::Error::wrap)?;
    std::mem::forget(dst);

    // The routing seed, as `place_verbatim` — and this is the path that matters
    // most, because the int8 twin is what the image transformer's thirty-four
    // blocks are made of. Stamping only the verbatim path would leave the
    // largest model in the guest allocating from the pool.
    let twin = scratch.repack_ko_into(
        shape,
        ko,
        Some((
            at.ptr,
            super::varground::guest_origin_on(cuda.cuda_stream().context().ordinal()),
        )),
    )?;
    QTensor::new(QStorage::Cuda(twin), shape.clone())
}

/// Place a tensor's file bytes in ground and view them.
fn place_verbatim(
    cuda: &candle::CudaDevice,
    ground: &Arc<Mutex<GuestGround>>,
    raw: &[u8],
    shape: &Shape,
    dtype: GgmlDType,
) -> candle::Result<QTensor> {
    // **Room for the padding the matmul reads past the end.** A leased view
    // carries none of its own (`QCudaStorage::from_leased_device_ptr` says so),
    // and the dequantising matmul reads there — so the tail is carved and zeroed
    // here or the kernel reads the next tensor. Only the narrow projections
    // actually reach a matmul this way, but norms cost nothing to be safe about.
    let padded =
        (shape.elem_count() + MATRIX_ROW_PADDING).div_ceil(dtype.block_size()) * dtype.type_size();
    let at = {
        let mut g = ground.lock().unwrap();
        g.place(padded, PLACE_ALIGN)
            .map_err(|e| candle::Error::Msg(format!("placing {dtype:?}: {e}")))?
    };
    let stream = cuda.cuda_stream();
    // SAFETY: `at.ptr` names `padded` bytes of ground this guest holds, and the
    // slice is `forget`ed below rather than dropped, so it never tries to free an
    // address the pool did not allocate.
    let mut dst = unsafe { stream.upgrade_device_ptr::<u8>(at.ptr, padded) };
    stream
        .memcpy_htod(raw, &mut dst.slice_mut(..raw.len()))
        .map_err(candle::Error::wrap)?;
    if padded > raw.len() {
        stream
            .memset_zeros(&mut dst.slice_mut(raw.len()..padded))
            .map_err(candle::Error::wrap)?;
    }
    std::mem::forget(dst);

    // SAFETY: the bytes are in flight to a range this guest owns, the ground
    // outlives every tensor handed out (the drain drops the model before the
    // ground), and nothing else writes the range.
    let storage = unsafe {
        QCudaStorage::from_leased_device_ptr(
            at.ptr,
            shape.elem_count(),
            dtype,
            cuda,
            // The routing seed, not `Foreign` — see `varground::guest_origin`.
            // A quantized weight is read by exactly the ops whose outputs this
            // is meant to carve, so leaving it foreign would route the dense
            // half of a model and not the quantized half.
            super::varground::guest_origin_on(cuda.cuda_stream().context().ordinal()),
        )?
    };
    QTensor::new(QStorage::Cuda(storage), shape.clone())
}

/// Ground a GGUF needs, before any of it is claimed.
///
/// Read from the header rather than from the file's length: a tileable
/// projection becomes its KO twin, which is a different size from its file bytes
/// — Q6_K → Q6_KO is the same width but not the same layout, and a float source
/// becomes Q8_KO and *shrinks*. Guessing from the file would under-claim, and a
/// short claim is discovered after the engine has already been evicted for it.
pub fn ground_bytes(path: &Path, mode: Int8Mode, skip: &[&str]) -> Result<usize, String> {
    let header = checkpoint::gguf_header(path).map_err(|e| format!("{path:?}: {e}"))?;
    let mut total = 0usize;
    for (name, info) in header.tensor_infos.iter() {
        if skip.contains(&name.as_str()) {
            continue;
        }
        let shape = Shape::from(info.shape.dims().to_vec());
        let dtype = info.ggml_dtype;
        let twin = mode.is_int8()
            && match shape.dims() {
                [nrows, ncols] => ko_tileable(*nrows, *ncols),
                _ => false,
            };
        let bytes = if twin {
            let ko = dtype
                .to_ko(mode)
                .map_err(|e| format!("{name}: no KO twin: {e}"))?;
            ko_repacked_bytes(&shape, ko).map_err(|e| format!("{name}: {e}"))?
        } else {
            (shape.elem_count() + MATRIX_ROW_PADDING).div_ceil(dtype.block_size())
                * dtype.type_size()
        };
        // Each tensor starts on its own alignment boundary, so the claim has to
        // carry the rounding as well as the bytes — four hundred tensors of
        // up-to-255 wasted bytes is not much, but a claim that is short by any
        // amount fails at the last tensor, after everything has been evicted.
        total += bytes.next_multiple_of(PLACE_ALIGN);
    }
    Ok(total)
}
