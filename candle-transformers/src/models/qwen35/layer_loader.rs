//! Standing up the layer cache for a dense checkpoint.
//!
//! The layer-streaming counterpart of [`super::expert_loader`], and it runs in
//! the same place in the load: after every resident tensor is down and the span
//! knows what it holds, so the zone is carved from a measured remainder rather
//! than a prediction.
//!
//! ## The order, and why it is this order
//!
//! ```text
//! images_from_gguf      geometry from the HEADER — no weight is read
//! open or build pack    the streaming pass, only when the pack is missing
//! carve the zone        slots from the ground the dense weights left
//! residues              a few hundred KB per layer, no repack
//! LayerCache::new       fills the warm tier, builds the pinned views
//! upload the pinned     the one part with no record anywhere
//! ```
//!
//! The first step is what makes the rest possible: `ko_repacked_bytes` is a
//! function of a shape and a target dtype, both of which are in the tensor
//! table, so the pack's `slot_bytes` — the max over images — is known before a
//! single weight is touched. Without that the pack build would need one pass to
//! measure and another to write.

use std::path::Path;
use std::sync::{Arc, Mutex};

use candle::quantized::{gguf_file, Int8Mode};
use candle::{Device, Result};

use super::config::Qwen35Config;
use super::layer_store::{sell_ground, LayerStore, StreamedLayers};
use super::quantized_weights::{
    load_layer, narrow_resident_twin, streaming_twin, Loader, QuantLayer, ResidentResidue,
};
use crate::models::expert_lre::handle::warm_slots_for;
use crate::models::layer_stream::assemble::assemble_layer;
use crate::models::layer_stream::build::images_from_gguf;
use crate::models::layer_stream::cache::{pack_path_for, SlotAssembler, STAGING_SLOTS};
use crate::models::layer_stream::descriptor::LayerImage;
use crate::models::layer_stream::pack::{header_for, LayerPack, PackIdentity, PackWriter};
use crate::models::layer_stream::view::StreamedLayer;
use crate::models::layer_stream::zone::{plan_zone, ZonePlan};
use crate::models::layer_stream::{slot_bytes_for_layers, LayerCache, LoadedLayer};
use crate::models::quantized_matmul::WeightResidency;

/// The assembled cache a streamed dense model reads its layers from.
pub type QwenLayerCache = LayerCache<QuantLayer, LayerAssembler>;

/// Turns a slot's views into the `QuantLayer` the forward reads.
///
/// A named type rather than a closure because it is spelled in
/// [`QuantModel`](super::quantized_weights::QuantModel)'s field type, and a
/// closure has none that can be written down.
pub struct LayerAssembler {
    /// Shared with [`StreamedLayers`], which answers the residue-only consumers
    /// out of the same vector rather than a second copy of it.
    residues: Arc<Vec<ResidentResidue>>,
    images: Vec<LayerImage>,
}

impl SlotAssembler<QuantLayer> for LayerAssembler {
    fn assemble(&self, view: StreamedLayer, layer: usize) -> Result<QuantLayer> {
        let img = self.images.get(layer).ok_or_else(|| {
            candle::Error::Msg(format!("layer assembler: no image for layer {layer}"))
        })?;
        let residue = self.residues.get(layer).ok_or_else(|| {
            candle::Error::Msg(format!("layer assembler: no residue for layer {layer}"))
        })?;
        assemble_layer(view, residue, img.kind, img.ffn)
    }
}

/// How many warm slots to ask the host for.
///
/// **Measured, not asked-and-stepped-down.** The obvious version wants every
/// streamable layer and leans on `WarmPool::new` stepping down until
/// `cuMemAllocHost` accepts. That converges, and to the wrong number:
/// availability counts droppable page cache — the checkpoint's own mmap reads as
/// available — so the allocator says yes to a tier that then leaves the OS
/// paging everything else, and the step-down cannot tell "the machine is full"
/// from "the machine will regret this". On the 27B it asks for 16 GB of
/// page-locked memory on a 32 GB box.
///
/// So the host is asked properly, through the same three ceilings the expert
/// warm tier uses (`expert_lre::handle::warm_slots_for`): what the machine is
/// big enough for, what it had free at launch, and how much may be page-locked
/// at all. A dense model and a routed one ask the identical question of the
/// identical machine, and a checkpoint is one or the other — so they share the
/// arithmetic rather than growing a second copy of it.
///
/// **The staging ring is deducted, not measured.** The probe reads
/// `host_pinned_bytes()`, and it necessarily runs before `LayerCache::new` pins
/// the cold-read ring — so the ring is invisible to exactly the ceiling it has
/// to fit under. Left uncharged it is ~1.16 GiB of the 27B's host budget spent
/// twice: `cuMemAllocHost` still says yes (availability counts droppable page
/// cache, the failure this function's whole design is aimed at) and the process
/// ends up over-pinned, leaving the OS paging everything else. Both tiers are
/// whole records of the same stride, so the correction is exact in slot units.
fn warm_budget(slot_bytes: usize, num_layers: usize, pinned: usize) -> usize {
    warm_slots_for(slot_bytes, num_layers.saturating_sub(pinned)).saturating_sub(STAGING_SLOTS)
}

/// Build the streamed layer store for a dense checkpoint.
///
/// `residues` were read by `load_quantized_model` inside the load window, so
/// they are span tenants like the rest of the resident model rather than pool
/// allocations made after the dense block was frozen.
#[allow(clippy::too_many_arguments)]
pub fn build_layer_cache<R: std::io::Read + std::io::Seek>(
    content: &gguf_file::Content,
    reader: &mut R,
    device: &Device,
    cfg: &Qwen35Config,
    mode: Int8Mode,
    gguf_path: &Path,
    gguf_identity: PackIdentity,
    pinned_layers: usize,
    residues: Arc<Vec<ResidentResidue>>,
) -> Result<LayerStore> {
    let Device::Cuda(cuda) = device else {
        candle::bail!("qwen35: the layer cache is a CUDA-only path");
    };
    if residues.len() != cfg.num_layers {
        candle::bail!(
            "qwen35 layer cache: {} residues for a {}-layer trunk",
            residues.len(),
            cfg.num_layers
        );
    }

    // ── Geometry, from the header alone ──
    // The narrowing schedule is an input to the geometry, not just to the bytes: a slot's size
    // is derived from the twin width, so a pack built without it would describe records the
    // narrowed load does not fill. Decided here, from the same tight-VRAM predicate the resident
    // weights use, so the two halves of one model are narrowed on one condition.
    let stream_narrow = narrow_resident_twin(device, cfg, content).map(|_| cfg.num_layers);
    let narrow = |name: &str| stream_narrow.and_then(|n| streaming_twin(name, n));
    let images = images_from_gguf(content, &cfg.layer_kinds, mode, &narrow)?;
    let slot_bytes = slot_bytes_for_layers(&images);
    let pinned = pinned_layers.min(cfg.num_layers);

    // ── The cold tier ──
    let path = pack_path_for(gguf_path, mode, stream_narrow);
    let pack = match LayerPack::open(&path, gguf_identity, &images, pinned) {
        Ok(p) => p,
        Err(e) => {
            tracing::info!(
                target: "candle_transformers::qwen35",
                path = %path.display(),
                "layer pack absent or stale ({e}); building it"
            );
            build_pack(
                content,
                reader,
                device,
                cfg,
                mode,
                &path,
                gguf_identity,
                &images,
                pinned,
                slot_bytes,
                stream_narrow,
            )?;
            LayerPack::open(&path, gguf_identity, &images, pinned)?
        }
    };

    // ── The hot tier ──
    let mut g = Loader::new(content, reader, device, mode, WeightResidency::Pool);
    g.set_stream_narrow(stream_narrow);
    let plan = carve_zone(cuda, &images, pinned)?;
    let assembler = LayerAssembler {
        residues: residues.clone(),
        images: images.clone(),
    };
    let cache = LayerCache::new(
        cuda,
        images,
        mode,
        pack,
        &plan,
        warm_budget(slot_bytes, cfg.num_layers, pinned),
        assembler,
    )?;

    // The pinned head is the one part of the model with no record in any tier —
    // it is never loaded and never evicted — so its bytes go into slots
    // `0..pinned` here, straight from the checkpoint. `LayerCache::new` has
    // already built the views over those addresses and is waiting for them.
    upload_pinned(&mut g, cfg, mode, cuda, &cache, pinned, slot_bytes)?;

    // Fill the rest of the zone now rather than inside the first forward — the
    // bytes move either way and this is where the wait belongs.
    let mut cache = cache;
    cache.warm_start()?;

    // **Open the shop.** A KV arena claim or a transient-tier placement that
    // runs out of ground buys more here, at the price of layer residency,
    // instead of refusing — the dense counterpart of what `expert_loader` does
    // for a routed checkpoint.
    //
    // This is a **process-global hook, not a call on the model**, and that is
    // why it is easy to miss: `BatchedModelCore::request_kv_ground` is a
    // different caller reaching the same seller, and wiring only that one leaves
    // `region_pool::buy_ground` answering zero. Measured on the 27B with the
    // trait method wired and this absent: the first four-context wave died on
    // "wave transient tier needs 939524096 B below the weight floor … the weight
    // side could not concede them", with 6 GiB of droppable layer slots sitting
    // right there.
    //
    // `Weak`, so the static registry does not outlive the model that owns the
    // cache.
    let cache = Arc::new(Mutex::new(cache));
    let seller = Arc::downgrade(&cache);
    let candle::DeviceLocation::Cuda { gpu_id } = device.location() else {
        candle::bail!("qwen35: the layer cache is a CUDA-only path")
    };
    candle_nn::kv_cache::set_ground_broker(gpu_id, move |regions| {
        seller.upgrade().map_or(0, |c| sell_ground(&c, regions))
    });

    Ok(LayerStore::Streamed(StreamedLayers::new(cache, residues)))
}

/// Place the pinned head's repacked bytes into its slots.
///
/// Loaded one layer at a time through the same `load_layer` the pack build
/// uses, repacked to the CUDA pool, copied slot-ward, and dropped — so the peak
/// is one layer even here, and the pool's ground is reused by the next.
///
/// # The whole slot is written, zeros included
///
/// Assembled host-side into one `slot_bytes` buffer and sent in a single H2D,
/// rather than a copy per projection. That is not only fewer transfers: the
/// gaps a per-projection copy would leave are **read**. The GGML matmul kernels
/// address `MATRIX_ROW_PADDING` elements past the end of every row, and
/// `QCudaStorage::zeros` exists precisely so that read is a defined zero — so a
/// slot whose padding held whatever the zone last had there would multiply
/// activations by stale weights. Every streamed layer already gets this for
/// free, because `PackWriter::write_layer` zeroes each record before placing
/// the payloads; the pinned head is the one path that does not go through the
/// pack, and it has to make the same guarantee itself.
fn upload_pinned<R: std::io::Read + std::io::Seek>(
    g: &mut Loader<'_, R>,
    cfg: &Qwen35Config,
    mode: Int8Mode,
    cuda: &candle::CudaDevice,
    cache: &QwenLayerCache,
    pinned: usize,
    slot_bytes: usize,
) -> Result<()> {
    let stream = cuda.cuda_stream();
    let mut slot = vec![0u8; slot_bytes];
    for li in 0..pinned {
        let layer = load_layer(g, cfg, li, mode, &mut 0)?.resolve_dense()?;
        let image = cache.image(li)?;
        let loaded = loaded_layer(&layer, image)?;
        let bufs = loaded.read_back(&stream)?;
        slot.fill(0);
        for (p, b) in image.placements.iter().zip(&bufs) {
            if p.bytes != b.len() {
                candle::bail!(
                    "layer stream: pinned L{li} {:?} read back {} B against the image's {} B",
                    p.role,
                    b.len(),
                    p.bytes
                );
            }
            slot[p.offset..p.offset + b.len()].copy_from_slice(b);
        }
        let base = cache.slot_base_of(li)?;
        // A homed slot is exactly its own image wide — the zone packs them
        // densely, so the staging buffer's `slot_bytes` (the max over every
        // layer) overruns any slot whose image is smaller than that max. Copy
        // the image, not the buffer.
        // SAFETY: `base` names `image.total` of a slot the zone handed this
        // cache and has not reclaimed. Synchronous on the compute stream, at
        // load, with nothing else in flight.
        let res = unsafe {
            candle::cuda_backend::cudarc::driver::sys::cuMemcpyHtoD_v2(
                base,
                slot.as_ptr() as *const _,
                image.total,
            )
        };
        if res != candle::cuda_backend::cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            candle::bail!("layer stream: pinned L{li} H2D failed: {res:?}");
        }
    }
    Ok(())
}

/// The streaming pack build: one layer at a time, repacked to the pool.
#[allow(clippy::too_many_arguments)]
fn build_pack<R: std::io::Read + std::io::Seek>(
    content: &gguf_file::Content,
    reader: &mut R,
    device: &Device,
    cfg: &Qwen35Config,
    mode: Int8Mode,
    path: &Path,
    identity: PackIdentity,
    images: &[LayerImage],
    pinned: usize,
    slot_bytes: usize,
    stream_narrow: Option<usize>,
) -> Result<()> {
    let Device::Cuda(cuda) = device else {
        candle::bail!("qwen35: the layer pack build is a CUDA-only path");
    };
    let stream = cuda.cuda_stream();
    let mut w = PackWriter::create(path, header_for(images, identity, pinned, slot_bytes))?;
    // Pool, not span: each layer here is materialised only to be read back and
    // dropped, and the dense block never frees. See [`WeightResidency`].
    let mut g = Loader::new(content, reader, device, mode, WeightResidency::Pool);
    g.set_stream_narrow(stream_narrow);

    for (li, image) in images.iter().enumerate().take(cfg.num_layers).skip(pinned) {
        // Loaded, read back, and dropped before the next one is touched — the
        // peak is one layer, which is what lets a model larger than the card be
        // packed at all.
        let layer = load_layer(&mut g, cfg, li, mode, &mut 0)?.resolve_dense()?;
        let loaded = loaded_layer(&layer, image)?;
        let bufs = loaded.read_back(&stream)?;
        let refs: Vec<&[u8]> = bufs.iter().map(|b| b.as_slice()).collect();
        w.write_layer(li, &refs)?;
        drop(loaded);
        drop(layer);
    }
    let published = w.finish()?;
    tracing::info!(
        target: "candle_transformers::qwen35",
        path = %published.display(),
        layers = cfg.num_layers - pinned,
        slot_mib = slot_bytes >> 20,
        "layer pack built"
    );
    Ok(())
}

/// Borrow a loaded layer's streamable projections in the image's order.
fn loaded_layer<'a>(layer: &'a QuantLayer, image: &LayerImage) -> Result<LoadedLayer<'a>> {
    let mut projections = Vec::with_capacity(image.placements.len());
    for p in &image.placements {
        projections.push((p.role, layer.streamed_projection(p.role)?));
    }
    Ok(LoadedLayer {
        kind: image.kind,
        ffn: image.ffn,
        projections,
    })
}

/// Carve the weight zone into layer slots and return their base addresses.
///
/// Sized from the ground the dense weights left, exactly as the expert zone is,
/// and capped at the model's depth: a zone with more slots than layers is ground
/// that can never hold anything.
fn carve_zone(cuda: &candle::CudaDevice, images: &[LayerImage], pinned: usize) -> Result<ZonePlan> {
    use candle_nn::kv_cache::{initial_weight_bytes, set_weight_floor, span_end};

    let stream = cuda.cuda_stream();
    let end = span_end(&stream)?;
    let opening = initial_weight_bytes(&stream)?;
    let num_layers = images.len();
    // **The floor is the pinned head plus one streaming cell, and the planner
    // owns it.** A zone that can hold the pinned head and nothing else loads
    // perfectly and then dies on the first forward with "L2 is absent and no
    // slot can hold it", because there is nowhere to put the layer the wave is
    // standing on. `plan_zone` refuses that case by construction; here it only
    // has to be reported against the span, which is the thing actually wrong.
    let plan = plan_zone(images, pinned, end, opening).map_err(|e| {
        candle::Error::Msg(format!(
            "{e} — the span leaves {} MiB after the dense residue and the KV side's \
             opening reserve",
            opening >> 20
        ))
    })?;
    let kv_regions = set_weight_floor(&stream, plan.floor)?;
    let used = plan.used_bytes(end);
    let homed = plan.resident();
    // On the arena-stats channel too: a test binary installs no tracing subscriber, so the
    // line below is invisible exactly where the partition is being measured. This is the last
    // piece of the breakdown — `[reclaim]` prints the span with its dense block and no zone,
    // and this is what the zone then takes out of it.
    if std::env::var("KV_ARENA_STATS").is_ok() {
        let cell = plan.floating.map_or(0, |f| f.bytes);
        eprintln!(
            "[zone] {homed} of {num_layers} layers resident in {} MiB (mean {} MiB, dense) \
             + {} MiB cell; {} stream; {} MiB left unclaimed; {kv_regions} regions to KV",
            used >> 20,
            (used - cell).checked_div(homed).unwrap_or(0) >> 20,
            cell >> 20,
            plan.missing.len(),
            opening.saturating_sub(used) >> 20,
        );
    }
    tracing::info!(
        target: "candle_transformers::qwen35",
        homed,
        layers = num_layers,
        zone_mib = used >> 20,
        streamed = plan.missing.len(),
        whole = plan.is_whole(),
        kv_regions,
        "qwen35 layer zone opened against the span"
    );
    Ok(plan)
}
