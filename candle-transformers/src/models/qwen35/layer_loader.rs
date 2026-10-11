//! Standing up the layer cache for a dense model pack.
//!
//! The layer-streaming counterpart of [`super::expert_loader`], and it runs in
//! the same place in the load: after every resident tensor is down and the span
//! knows what it holds, so the zone is carved from a measured remainder rather
//! than a prediction.
//!
//! ## The order, and why it is this order
//!
//! ```text
//! open the section      the pack's layer section, checked against this build
//! images_of             geometry from the section's HEADER — no weight is read
//! carve the zone        slots from the ground the dense weights left
//! LayerCache::new       fills the warm tier and the pinned head, from the pack
//! warm_start            fills the rest of the zone
//! ```
//!
//! The pack was built (`pack_build`) by loading every trunk layer through the
//! same `load_layer` a resident load runs, so a record is exactly what this
//! loader would have produced — including the pinned head, which is filled from
//! its record like any other layer.

use std::sync::{Arc, Mutex};

use candle::{Device, Result};

use super::config::Qwen35Config;
use super::layer_store::{sell_ground, LayerStore, StreamedLayers};
use super::quantized_weights::{narrow_resident_twin, QuantLayer, ResidentResidue};
use crate::models::delta_net::LayerKind;
use crate::models::expert_lre::handle::warm_slots_for;
use crate::models::expert_lre::PINNED_LAYERS;
use crate::models::layer_stream::assemble::assemble_layer;
use crate::models::layer_stream::cache::{SlotAssembler, STAGING_SLOTS};
use crate::models::layer_stream::descriptor::{LayerImage, MixKind};
use crate::models::layer_stream::pack::images_of;
use crate::models::layer_stream::section::open_layer_section;
use crate::models::layer_stream::view::StreamedLayer;
use crate::models::layer_stream::zone::{plan_zone, ZonePlan};
use crate::models::layer_stream::{slot_bytes_for_layers, LayerCache, LoadedLayer};
use crate::models::model_pack::ModelPack;

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
/// availability counts droppable page cache — the pack's own mapping reads as
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

/// Build the streamed layer store for a dense model pack.
///
/// `residues` were read by `load_quantized_model` inside the load window, so
/// they are span tenants like the rest of the resident model rather than pool
/// allocations made after the dense block was frozen.
pub fn build_layer_cache(
    pack: &ModelPack,
    device: &Device,
    cfg: &Qwen35Config,
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
    let Some(at) = pack.layers else {
        candle::bail!("qwen35: a dense model's pack has no layer section");
    };
    // The pack was built for one narrowing; this card must want the same, or
    // the resident weights and the streamed ones were narrowed on different
    // conditions. The resolver picks the pack by this rule, so a mismatch is a
    // pack opened by path for a card it was not built for.
    let want = narrow_resident_twin(device, cfg, pack.checkpoint_bytes).map(|_| cfg.num_layers);
    if pack.narrow != want {
        candle::bail!(
            "qwen35: the pack streams its layers narrowed for {:?}, this card wants {want:?} — \
             it was built for a card of another size",
            pack.narrow
        );
    }

    // ── The cold tier, and the geometry from its header ──
    let section = open_layer_section(&pack.path, at.offset, cuda)?;
    let images = images_of(section.header())?;
    if images.len() != cfg.num_layers {
        candle::bail!(
            "qwen35: the layer section holds {} layers, the trunk has {}",
            images.len(),
            cfg.num_layers
        );
    }
    for (li, (img, kind)) in images.iter().zip(&cfg.layer_kinds).enumerate() {
        let want = match kind {
            LayerKind::DeltaNet => MixKind::DeltaNet,
            LayerKind::Attention => MixKind::Attention,
        };
        if img.kind != want {
            candle::bail!(
                "qwen35: layer {li} is {:?} in the pack and {want:?} in the model",
                img.kind
            );
        }
    }
    let slot_bytes = slot_bytes_for_layers(&images);
    let pinned = PINNED_LAYERS.min(cfg.num_layers);

    // ── The hot tier ──
    let plan = carve_zone(cuda, &images, pinned)?;
    let assembler = LayerAssembler {
        residues: residues.clone(),
        images: images.clone(),
    };
    let mut cache = LayerCache::new(
        cuda,
        images,
        pack.int8_mode,
        section,
        &plan,
        pinned,
        warm_budget(slot_bytes, cfg.num_layers, pinned),
        assembler,
    )?;

    // Fill the rest of the zone now rather than inside the first forward — the
    // bytes move either way and this is where the wait belongs.
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

/// Borrow a loaded layer's streamable projections in the image's order — what
/// the pack build reads back into a record.
pub(crate) fn loaded_layer<'a>(
    layer: &'a QuantLayer,
    image: &LayerImage,
) -> Result<LoadedLayer<'a>> {
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
