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
use std::time::Instant;

use candle::quantized::cuda::{ko_repacked_bytes, QCudaStorage, MATRIX_ROW_PADDING};
use candle::quantized::ko_quant::ko_tileable;
use candle::quantized::{GgmlDType, Int8Mode, QStorage, QTensor};
use candle::{Device, Shape};
use candle_transformers::quantized_var_builder::VarBuilder;
use cudarc::driver::result::{memcpy_htod_async, memset_d8_async};

use super::checkpoint;
use super::ground::GuestGround;
use super::staging::Staging;
use super::windowed::{window_bytes, Windowed};

/// The most of a checkpoint read off the disk at once: 16 direct reads of
/// 4 MiB in flight four times over, and a host buffer the engine's RAM does not
/// notice for the length of a load. A tensor larger than this is a window of its
/// own.
const WINDOW: u64 = 256 << 20;

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
/// **Read in file order, a window at a time** ([`Windowed`]): the tensors are
/// sorted by where they sit and gathered into windows of up to [`WINDOW`]
/// ([`plan_windows`]), and each window comes off the disk unbuffered, straight
/// into a pinned buffer ([`Staging`]).
///
/// **The disk and the card overlap.** Every copy out of a window is one async
/// DMA and every repack an async launch, so placing a window issues its work
/// and returns; the next window is then read into the other buffer while the
/// card drains this one. A buffer is refilled only after the event recorded
/// past its copies, and the sync before returning orders the whole set before
/// any read. Copied out of pageable memory instead, every tensor's copy
/// synchronised the stream, so each waited out every kernel before it and the
/// disk and the card took turns: on BlackWorld (2026-10-27) the transformer's
/// 7.2 GB placed in 2.19 s and the encoder's in 1.14 s, each about 3.3 GB/s.
///
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
    let began = Instant::now();
    let header = checkpoint::gguf_header(path).map_err(|e| format!("{path:?}: {e}"))?;
    let mut file = Windowed::open(path).map_err(|e| format!("{path:?}: {e}"))?;
    let Device::Cuda(cuda) = device else {
        return Err("guest weights: the reservation is a CUDA allocation".into());
    };

    let mut wanted: Vec<(&String, Shape, GgmlDType, u64, u64)> = header
        .tensor_infos
        .iter()
        .filter(|(name, _)| !skip.contains(&name.as_str()))
        .map(|(name, info)| {
            let shape = Shape::from(info.shape.dims().to_vec());
            let dtype = info.ggml_dtype;
            let start = header.tensor_data_offset + info.offset;
            let len = shape.elem_count() / dtype.block_size() * dtype.type_size();
            (name, shape, dtype, start, start + len as u64)
        })
        .collect();
    wanted.sort_by_key(|&(_, _, _, start, _)| start);
    let ranges: Vec<(u64, u64)> = wanted.iter().map(|t| (t.3, t.4)).collect();
    let windows = plan_windows(&ranges, WINDOW);
    let largest = windows
        .iter()
        .map(|w| window_bytes(w.from, w.to))
        .max()
        .unwrap_or(0);
    let mut staging = Staging::new(largest).map_err(|e| format!("guest weights: staging: {e}"))?;
    let staged_ms = began.elapsed().as_millis() as u64;
    let stream = cuda.cuda_stream();

    let mut placed: HashMap<String, Arc<QTensor>> = HashMap::with_capacity(wanted.len());
    let mut twins = 0usize;
    let mut read = std::time::Duration::ZERO;
    let mut bytes = 0u64;
    for w in &windows {
        let (slot, buf) = staging
            .acquire()
            .map_err(|e| format!("guest weights: the staging barrier: {e}"))?;
        let t_read = Instant::now();
        let landed = file
            .read_into(w.from, w.to, buf)
            .map_err(|e| format!("{path:?}: reading bytes {}..{}: {e}", w.from, w.to))?;
        read += t_read.elapsed();
        bytes += w.to - w.from;
        let window = &buf[landed];
        let (from, at, until) = (w.from, w.first, w.end);

        for (name, shape, dtype, start, end) in &wanted[at..until] {
            let (shape, dtype) = (shape, *dtype);
            let raw = &window[(start - from) as usize..(end - from) as usize];

            let twin = mode.is_int8()
                && match shape.dims() {
                    [nrows, ncols] => ko_tileable(*nrows, *ncols),
                    _ => false,
                };
            // Named before it is placed, not after: a placement that faults
            // takes the process with it, and the last line written is then the
            // only evidence of which tensor did it.
            tracing::trace!(
                target: "candle_conversation::guest",
                tensor = %name, shape = ?shape.dims(), ?dtype, twin,
                "placing"
            );
            let qt = if twin {
                twins += 1;
                place_ko_twin(cuda, ground, raw, shape, dtype, mode)
            } else {
                place_verbatim(cuda, ground, raw, shape, dtype)
            }
            .map_err(|e| format!("{name} {:?}: {e}", shape.dims()))?;
            placed.insert((*name).clone(), Arc::new(qt));
        }

        // The buffer is refilled two windows on, once these copies have landed.
        staging
            .issued(slot, &stream)
            .map_err(|e| format!("guest weights: the staging event: {e}"))?;
    }
    device
        .synchronize()
        .map_err(|e| format!("guest weights: the load barrier: {e}"))?;
    drop(staging);
    // At info: the load is most of a draw's own cost, and whether a slow one is
    // the disk or the card is the first thing it has to answer.
    tracing::info!(
        target: "candle_conversation::guest",
        path = %path.display(),
        tensors = placed.len(),
        int8_twins = twins,
        mib = bytes >> 20,
        windows = windows.len(),
        staged_ms,
        read_ms = read.as_millis() as u64,
        total_ms = began.elapsed().as_millis() as u64,
        "placed a gguf in guest ground"
    );
    Ok(VarBuilder::from_tensors(placed, device.clone()))
}

/// One window of a checkpoint: the file bytes `from..to`, holding the sorted
/// tensors `first..end`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window {
    pub from: u64,
    pub to: u64,
    pub first: usize,
    pub end: usize,
}

/// The windows sorted tensor `ranges` (`start..end` in the file) are read in:
/// each holds every tensor that ends within `cap` of its first one's start, and
/// always that first one, however large it is.
pub fn plan_windows(ranges: &[(u64, u64)], cap: u64) -> Vec<Window> {
    let mut out = Vec::new();
    let mut at = 0usize;
    while at < ranges.len() {
        let from = ranges[at].0;
        let mut end = at + 1;
        while end < ranges.len() && ranges[end].1 - from <= cap {
            end += 1;
        }
        let to = ranges[at..end].iter().map(|r| r.1).max().unwrap_or(from);
        out.push(Window {
            from,
            to,
            first: at,
            end,
        });
        at = end;
    }
    out
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
    // adds row padding on top) on this stream, and `raw` is a pinned staging
    // buffer `place_gguf` refills only after the event past this copy.
    unsafe { memcpy_htod_async(scratch.data_ptr_mut(), raw, stream.cu_stream()) }
        .map_err(candle::Error::wrap)?;

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
    // SAFETY: `at.ptr` names `padded` bytes of ground this guest holds, and
    // `raw` is a pinned staging buffer `place_gguf` refills only after the
    // event past this copy.
    unsafe {
        memcpy_htod_async(at.ptr, raw, stream.cu_stream()).map_err(candle::Error::wrap)?;
        if padded > raw.len() {
            memset_d8_async(
                at.ptr + raw.len() as u64,
                0,
                padded - raw.len(),
                stream.cu_stream(),
            )
            .map_err(candle::Error::wrap)?;
        }
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    /// **Windows follow the file**: tensors that end within the cap of the
    /// window's first start share it, the next starts a window of its own, and
    /// a tensor wider than the cap is a window by itself.
    #[test]
    fn windows_gather_tensors_up_to_the_cap() {
        let ranges = [(0, 100), (100, 250), (250, 300), (300, 900), (900, 950)];
        assert_eq!(
            plan_windows(&ranges, 300),
            vec![
                Window {
                    from: 0,
                    to: 300,
                    first: 0,
                    end: 3
                },
                Window {
                    from: 300,
                    to: 900,
                    first: 3,
                    end: 4
                },
                Window {
                    from: 900,
                    to: 950,
                    first: 4,
                    end: 5
                },
            ]
        );
        assert_eq!(plan_windows(&[], 300), Vec::<Window>::new());
    }
}
