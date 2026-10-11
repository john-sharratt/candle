//! The expert section of a model pack: built once from a checkpoint, opened on
//! every load.
//!
//! The build is the only place a checkpoint's expert bytes are read. It repacks
//! every expert of every MoE layer into the kernel-ready record the slots hold
//! and streams the records into the model pack (`crate::models::model_pack`).
//! A load then takes the cache's whole geometry from the section's header —
//! shapes, source and repacked dtypes, sizes, offsets — and never sees the
//! checkpoint at all.

use super::pack::{pairs_in, ExpertPack, LayerSpans, PackHeader, PackWriter, ProjectionSpan};
use super::pinned::{layer_geometries, LayerGeometry};
use super::slot_image::{repack_expert_projections, slot_bytes_for, slot_offsets};
use super::types::MmapExpertRef;
use crate::models::repack_fingerprint::{pair_fingerprint, prints_for};
use candle::direct_io::round_up_sector;
use candle::quantized::Int8Mode;
use candle::{CudaDevice, Result};
use std::io::Write;
use std::path::Path;

/// The section header for a checkpoint whose MoE layers are `host_refs`, block
/// `blocks[i]` for layer `i`, repacked for `int8mode` by this build.
///
/// Known before a byte is repacked — the geometry is a function of shapes and
/// target dtypes, and the fingerprints of a reference matrix — so a model pack
/// can place everything after this section before writing anything.
pub(crate) fn section_header(
    host_refs: &[Vec<MmapExpertRef>],
    blocks: &[u32],
    int8mode: Int8Mode,
    device: &CudaDevice,
) -> Result<PackHeader> {
    if int8mode == Int8Mode::Off {
        candle::bail!(
            "expert section: Int8Mode::Off has no expert kernel — routed models are packed for \
             Precision or Performance"
        );
    }
    if blocks.len() != host_refs.len() {
        candle::bail!(
            "expert section: {} block indices for {} MoE layers",
            blocks.len(),
            host_refs.len()
        );
    }
    let experts_per_layer = host_refs.first().map_or(0, |l| l.len());
    if host_refs.iter().any(|l| l.len() != experts_per_layer) {
        candle::bail!("expert section: MoE layers disagree on their expert count");
    }
    let geoms = layer_geometries(host_refs, int8mode)?;
    let slot_bytes = slot_bytes_for(&geoms);
    let mut layers = Vec::with_capacity(geoms.len());
    for ((g, refs), &block) in geoms.iter().zip(host_refs).zip(blocks) {
        let r = &refs[0];
        let (gate, up, down, _) = slot_offsets(g);
        let span =
            |offset: usize, bytes: usize, dtype, src_dtype, shape: &[usize]| ProjectionSpan {
                offset: offset as u32,
                bytes: bytes as u32,
                dtype,
                src_dtype,
                rows: shape[0] as u32,
                cols: shape[1] as u32,
            };
        layers.push(LayerSpans {
            block,
            gate: span(
                gate,
                g.gate_repacked_size,
                g.gate_dtype,
                r.gate_dtype,
                &g.gate_shape,
            ),
            up: span(up, g.up_repacked_size, g.up_dtype, r.up_dtype, &g.up_shape),
            down: span(
                down,
                g.down_repacked_size,
                g.down_dtype,
                r.down_dtype,
                &g.down_shape,
            ),
        });
    }
    let pairs = prints_for(device, &pairs_in(&layers));
    Ok(PackHeader {
        num_layers: layers.len() as u32,
        experts_per_layer: experts_per_layer as u32,
        slot_bytes: slot_bytes as u32,
        stride: round_up_sector(slot_bytes) as u64,
        int8_mode: int8mode as u32,
        layers,
        pairs,
    })
}

/// Repack every expert of every layer out of the checkpoint `mmap` and stream
/// the section into `out`. Returns the bytes written.
///
/// `progress` is called per expert against the total.
pub(crate) fn write_section<W: Write + ?Sized>(
    out: &mut W,
    header: PackHeader,
    mmap: &[u8],
    host_refs: &[Vec<MmapExpertRef>],
    int8mode: Int8Mode,
    device: &CudaDevice,
    progress: Option<&dyn Fn(usize, usize)>,
) -> Result<u64> {
    let geoms = layer_geometries(host_refs, int8mode)?;
    let total = header.total_experts();
    let per_layer = header.experts_per_layer as usize;
    let t0 = std::time::Instant::now();
    let mut w = PackWriter::new(out, header)?;
    for (layer, (geom, refs)) in geoms.iter().zip(host_refs).enumerate() {
        for (expert, r) in refs.iter().enumerate() {
            // A repack that fails leaves an expert with no valid record, and a
            // record of zeroes reads back as a plausible expert. There is no
            // partial answer here: the section is authoritative or it is nothing.
            let (gate, up, down) = repack_expert_projections(mmap, r, geom, device)?;
            w.write_expert(layer, expert, &gate, &up, &down)?;
            if let Some(cb) = progress {
                cb(layer * per_layer + expert + 1, total);
            }
        }
    }
    let len = w.finish()?;
    tracing::info!(
        target: "candle_transformers::expert_lre",
        layers = geoms.len(),
        experts = total,
        gib = len as f64 / (1u64 << 30) as f64,
        secs = t0.elapsed().as_secs_f64(),
        "expert section written"
    );
    Ok(len)
}

/// The cache's per-layer geometry, read off a section header.
///
/// Each layer's offsets are checked against the slot layout this build lays a
/// geometry out with: a section written by a build whose layout has since
/// changed is refused here, by comparing offsets, rather than trusted to a
/// version number someone had to remember to bump.
pub(crate) fn geometries_of(header: &PackHeader) -> Result<Vec<LayerGeometry>> {
    let mut geoms = Vec::with_capacity(header.layers.len());
    for (i, l) in header.layers.iter().enumerate() {
        let shape = |p: &ProjectionSpan| vec![p.rows as usize, p.cols as usize];
        let g = LayerGeometry {
            gate_shape: shape(&l.gate),
            gate_dtype: l.gate.dtype,
            gate_repacked_size: l.gate.bytes as usize,
            up_shape: shape(&l.up),
            up_dtype: l.up.dtype,
            up_repacked_size: l.up.bytes as usize,
            down_shape: shape(&l.down),
            down_dtype: l.down.dtype,
            down_repacked_size: l.down.bytes as usize,
            total_repacked_size: (l.gate.bytes + l.up.bytes + l.down.bytes) as usize,
        };
        let (gate, up, down, _) = slot_offsets(&g);
        let recorded = (
            l.gate.offset as usize,
            l.up.offset as usize,
            l.down.offset as usize,
        );
        if (gate, up, down) != recorded {
            candle::bail!(
                "expert section layer {i}: projections at {recorded:?}, this build lays them at \
                 {:?} — the slot layout changed since the pack was built",
                (gate, up, down)
            );
        }
        geoms.push(g);
    }
    if slot_bytes_for(&geoms) != header.slot_bytes as usize {
        candle::bail!(
            "expert section: slot is {} bytes, this build's layout makes it {}",
            header.slot_bytes,
            slot_bytes_for(&geoms)
        );
    }
    Ok(geoms)
}

/// Open the expert section at `base` of the model pack `path`, checking every
/// repack pair it holds against this build's.
pub(crate) fn open_section(path: &Path, base: u64, device: &CudaDevice) -> Result<ExpertPack> {
    ExpertPack::open_section(path, base, |src, dtype| {
        pair_fingerprint(device, src, dtype)
    })
}
