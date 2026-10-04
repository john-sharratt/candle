//! Slot images: one expert's three projections at fixed aligned offsets.
//!
//! The same image is a VRAM weight-zone slot, a warm-tier slot, a pad slot and
//! a pack record, so moving an expert between any two of them is one copy with
//! nothing rearranged, and the live table names a projection in any of them by
//! the image's base plus a fixed offset.

use super::compute::QMatMul;
use super::pack::RecordLayout;
use super::pinned::LayerGeometry;
use super::types::{ExpertSlot, MmapExpertRef};
use candle::Result;
use cudarc::driver::CudaStream;
use std::sync::Arc;

/// Alignment every projection's base within a slot is rounded up to — what the
/// tensor-core paths require of an operand.
const PROJECTION_ALIGN: usize = 256;

/// Byte offsets of gate / up / down within one expert slot, and the aligned
/// total.
///
/// A slot is one range of the weight zone holding all three projections, so the
/// zone can stay an array of equal-sized units — the property that makes "the
/// rightmost free spot" a single index and a relocation a memcpy. The three sit
/// at aligned offsets inside it.
///
/// The total is what [`slot_bytes_for`] maxes over layers, so a slot always
/// holds the widest layer's three projections and any layer's fit in any slot.
pub(crate) fn slot_offsets(geom: &LayerGeometry) -> (usize, usize, usize, usize) {
    let align = |n: usize| n.div_ceil(PROJECTION_ALIGN) * PROJECTION_ALIGN;
    let up = align(geom.gate_repacked_size);
    let down = up + align(geom.up_repacked_size);
    let total = down + align(geom.down_repacked_size);
    (0, up, down, total)
}

/// Bytes one zone slot must be to hold any layer's expert.
pub(crate) fn slot_bytes_for(geoms: &[LayerGeometry]) -> usize {
    geoms.iter().map(|g| slot_offsets(g).3).max().unwrap_or(0)
}

/// Bytes of one 32-row tile of the largest projection of any layer — what an
/// expert GEMM worker copies into its VRAM scratch slot per item. A KO
/// projection is `[K block][N/8 row chunks]` of equal chunks, so a 32-row tile
/// is `size · 32 / N`.
pub(crate) fn row_tile_bytes_for(geoms: &[LayerGeometry]) -> usize {
    geoms
        .iter()
        .flat_map(|g| {
            [
                (g.gate_repacked_size, g.gate_shape[0]),
                (g.up_repacked_size, g.up_shape[0]),
                (g.down_repacked_size, g.down_shape[0]),
            ]
        })
        .map(|(bytes, nrows)| bytes * 32 / nrows.max(1))
        .max()
        .unwrap_or(0)
}

/// Wrap an already-populated slot at `slot_base` as an `ExpertSlot`, without
/// moving any bytes.
///
/// The three `QMatMul`s hold device pointers, so a slot whose bytes arrived by
/// a copy — a promotion, a relocation — gets its storages built over its
/// address after the bytes are in place.
///
/// # Safety
///
/// `slot_base` must name a slot the zone has handed out, already holding (or
/// about to hold, before anything reads the views) this layer's three
/// projections at [`slot_offsets`].
pub(crate) unsafe fn build_slot_view(
    geom: &LayerGeometry,
    cuda_dev: &candle::CudaDevice,
    slot_base: u64,
) -> Result<ExpertSlot> {
    let (gate_off, up_off, down_off, _) = slot_offsets(geom);
    let view = |off: usize, bytes: usize, dtype, shape: &Vec<usize>| -> Result<_> {
        // Extent == payload: an expert slot is cut to exactly its repacked
        // bytes, and its KO twin's int8 kernel reads exactly those.
        let storage = candle::quantized::view_repacked(
            cuda_dev,
            slot_base + off as u64,
            bytes,
            bytes,
            dtype,
        )?;
        candle::quantized::QTensor::new(storage, shape.clone())
    };
    let gate_qt = view(
        gate_off,
        geom.gate_repacked_size,
        geom.gate_dtype,
        &geom.gate_shape,
    )?;
    let up_qt = view(up_off, geom.up_repacked_size, geom.up_dtype, &geom.up_shape)?;
    let down_qt = view(
        down_off,
        geom.down_repacked_size,
        geom.down_dtype,
        &geom.down_shape,
    )?;
    Ok(ExpertSlot {
        gate_proj: QMatMul::from_qtensor_repacked(gate_qt)?,
        up_proj: QMatMul::from_qtensor_repacked(up_qt)?,
        down_proj: QMatMul::from_qtensor_repacked(down_qt)?,
    })
}

/// Build an `ExpertSlot` from already-repacked host bytes, uploading them into
/// the weight-zone slot at `slot_base` on the device's default stream.
///
/// # Safety
///
/// `slot_base` must name a slot the zone has handed out and not reclaimed, of at
/// least `slot_offsets(geom).3` bytes.
pub(crate) unsafe fn build_slot_from_repacked_with_device(
    gate_bytes: &[u8],
    up_bytes: &[u8],
    down_bytes: &[u8],
    geom: &LayerGeometry,
    cuda_dev: &candle::CudaDevice,
    slot_base: u64,
) -> Result<ExpertSlot> {
    let (gate_off, up_off, down_off, _) = slot_offsets(geom);
    let stream = cuda_dev.cuda_stream();
    let gate_storage = candle::quantized::load_repacked_into(
        cuda_dev,
        &stream,
        slot_base + gate_off as u64,
        gate_bytes,
        geom.gate_dtype,
    )?;
    let up_storage = candle::quantized::load_repacked_into(
        cuda_dev,
        &stream,
        slot_base + up_off as u64,
        up_bytes,
        geom.up_dtype,
    )?;
    let down_storage = candle::quantized::load_repacked_into(
        cuda_dev,
        &stream,
        slot_base + down_off as u64,
        down_bytes,
        geom.down_dtype,
    )?;
    let gate_qt = candle::quantized::QTensor::new(gate_storage, geom.gate_shape.clone())?;
    let up_qt = candle::quantized::QTensor::new(up_storage, geom.up_shape.clone())?;
    let down_qt = candle::quantized::QTensor::new(down_storage, geom.down_shape.clone())?;
    Ok(ExpertSlot {
        gate_proj: QMatMul::from_qtensor_repacked(gate_qt)?,
        up_proj: QMatMul::from_qtensor_repacked(up_qt)?,
        down_proj: QMatMul::from_qtensor_repacked(down_qt)?,
    })
}

/// Bytes of a slot image from the gate's start to the down projection's end —
/// what one copy of the image moves.
pub(crate) fn image_extent(layout: RecordLayout) -> usize {
    layout.down.offset + layout.down.bytes
}

/// Upload one pack **record** into the weight-zone slot at `slot_base` on
/// `stream` — one copy, padding and all, since the record is the slot image —
/// and build the slot's views over it.
///
/// # Safety
///
/// As [`build_slot_from_repacked_with_device`]; `record` must stay unwritten
/// until the copy on `stream` has landed.
pub(crate) unsafe fn build_slot_from_record_on_stream(
    record: &[u8],
    layout: RecordLayout,
    geom: &LayerGeometry,
    cuda_dev: &candle::CudaDevice,
    stream: &Arc<CudaStream>,
    slot_base: u64,
) -> Result<ExpertSlot> {
    let (gate_off, up_off, down_off, _) = slot_offsets(geom);
    if (layout.gate.offset, layout.up.offset, layout.down.offset) != (gate_off, up_off, down_off) {
        candle::bail!(
            "expert record layout ({}, {}, {}) is not the slot's ({gate_off}, {up_off}, \
             {down_off}); the record cannot be copied into the slot whole",
            layout.gate.offset,
            layout.up.offset,
            layout.down.offset,
        )
    }
    cudarc::driver::result::memcpy_htod_async(
        slot_base,
        &record[..image_extent(layout)],
        stream.cu_stream(),
    )
    .map_err(candle::Error::wrap)?;
    build_slot_view(geom, cuda_dev, slot_base)
}

/// Lay one expert's three projections into a record buffer at their spans.
///
/// The gaps between projections are alignment padding the kernels never read;
/// they are zeroed so a record is a deterministic function of its expert, which
/// is what lets the pack file be compared byte for byte between builds.
pub(crate) fn write_record(
    dst: &mut [u8],
    layout: RecordLayout,
    gate: &[u8],
    up: &[u8],
    down: &[u8],
) {
    dst.fill(0);
    for (span, src) in [(layout.gate, gate), (layout.up, up), (layout.down, down)] {
        dst[span.offset..span.offset + src.len()].copy_from_slice(src);
    }
}

/// GPU-repack one expert's three projections from GGML to K/128 format.
///
/// Returns `(gate_bytes, up_bytes, down_bytes)` as host `Vec<u8>`.
pub(crate) fn repack_expert_projections(
    mmap: &[u8],
    r: &MmapExpertRef,
    geom: &LayerGeometry,
    cuda_dev: &candle::CudaDevice,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let gate_ggml = &mmap[r.gate_offset..r.gate_offset + r.gate_len];
    let up_ggml = &mmap[r.up_offset..r.up_offset + r.up_len];
    let down_ggml = &mmap[r.down_offset..r.down_offset + r.down_len];

    // `r.*_dtype` is the compact GGUF source; `geom.*_dtype` is the target the
    // tiers cache (the KO twin). This repacks Q4_K→KO once per expert.
    let gate_repacked = candle::quantized::repack_to_host(
        cuda_dev,
        gate_ggml,
        geom.gate_shape[0],
        geom.gate_shape[1],
        r.gate_dtype,
        geom.gate_dtype,
    )?;
    let up_repacked = candle::quantized::repack_to_host(
        cuda_dev,
        up_ggml,
        geom.up_shape[0],
        geom.up_shape[1],
        r.up_dtype,
        geom.up_dtype,
    )?;
    let down_repacked = candle::quantized::repack_to_host(
        cuda_dev,
        down_ggml,
        geom.down_shape[0],
        geom.down_shape[1],
        r.down_dtype,
        geom.down_dtype,
    )?;
    Ok((gate_repacked, up_repacked, down_repacked))
}
