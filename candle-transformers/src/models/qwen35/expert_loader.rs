//! Standing up the expert cache for a routed model pack.
//!
//! The caller's half of [`super::quantized_weights::load_quantized_model`]:
//! the cache is sized from a live measurement of what the dense weights left
//! behind, so it cannot be built inside the dense loop
//! (`docs/archived/elastic_vram_partition.md` §4). The order is fixed —
//!
//! 1. dense weights resident (the loop in `quantized_weights`);
//! 2. measure the span, carve the weight zone, place the boundary;
//! 3. fill the cache into the zone that measurement produced.
//!
//! Nothing here is Qwen3.5-specific: the expert section carries its own
//! geometry, and the lineage's frozen `ffn_{gate,up,down}_exps` schema is shared
//! with qwen4exp (`docs/archived/qwen38_flash_next.md` §12.2 pins it as "exactly
//! as qwen35moe"), so each model hands over its own expert count and layer
//! range.

use candle::{Device, Result};
use std::sync::Arc;

use super::config::Qwen35Config;
use crate::models::expert_lre::{minimum_resident_slots, ExpertCache, ExpertCacheSetup};
use crate::models::model_pack::ModelPack;

/// Build the expert cache for a routed pack, against the span the dense
/// weights left.
///
/// Returns `None` for a dense model, which is not an error — the 9B has no
/// experts and wants no cache.
#[cfg(feature = "cuda")]
pub fn build_expert_cache(
    pack: &ModelPack,
    cfg: &Qwen35Config,
    device: &Device,
) -> Result<Option<Arc<ExpertCache>>> {
    let Some(moe) = cfg.moe else {
        return Ok(None);
    };
    build_expert_cache_for(
        pack,
        moe.n_experts,
        moe.n_experts_used,
        cfg.num_layers + cfg.num_mtp_layers,
        device,
        // The qwen35 loader carries no progress hook of its own.
        None,
        // Its embedding is copied into pinned host memory at load, and every
        // other tensor of the mapping is paged on demand — none of it is served
        // by a cache of the model's own.
        0,
    )
}

/// [`build_expert_cache`] with the geometry passed directly. Everything below
/// the expert section (the span measurement, the elastic boundary, the zone
/// floor, the ground broker) is the engine's.
///
/// **The MTP draft head's layer is included**, and it must be. The head is
/// `blk.{num_layers}`, and on a routed checkpoint it carries a full expert set
/// of its own — so `n_layers_total` counts it. The position of a layer in the
/// section IS the `moe_layer_idx` its router keys the cache on, and the section
/// is written in block order — trunk first, then the head — which is the order
/// `load_quantized_model` assigns those indices in.
///
/// `offloaded_bytes` is [`ExpertCacheSetup::offloaded_bytes`]: mapped bytes a
/// bounded cache of the model's own serves, or the device holds, instead of the
/// page cache.
#[cfg(feature = "cuda")]
pub fn build_expert_cache_for(
    pack: &ModelPack,
    n_expert: usize,
    n_expert_used: usize,
    n_layers_total: usize,
    device: &Device,
    progress: Option<&dyn Fn(usize, usize)>,
    offloaded_bytes: u64,
) -> Result<Option<Arc<ExpertCache>>> {
    use crate::models::expert_lre::section::{geometries_of, open_section};
    use crate::models::expert_lre::slot_bytes_for;
    use candle_nn::kv_cache::{
        initial_weight_bytes, set_weight_floor, span_end, weight_capacity_bytes, WeightZone,
    };

    let Some(at) = pack.experts else {
        candle::bail!("qwen35: the model routes but its pack has no expert section");
    };
    let Device::Cuda(cuda_dev) = device else {
        candle::bail!("qwen35: the expert cache is a CUDA-only path");
    };
    let section = open_section(&pack.path, at.offset, cuda_dev)?;
    let header = section.header();
    // The section's layers must be the model's MoE layers, in block order —
    // `0..n_layers_total`, every one routed — or a router's `moe_layer_idx`
    // names another layer's experts.
    let blocks: Vec<u32> = header.layers.iter().map(|l| l.block).collect();
    let want: Vec<u32> = (0..n_layers_total as u32).collect();
    if blocks != want {
        candle::bail!(
            "qwen35: the expert section holds blocks {blocks:?}, the model routes {want:?}"
        );
    }
    if header.experts_per_layer as usize != n_expert {
        candle::bail!(
            "qwen35: the expert section holds {} experts per layer, the metadata declares \
             {n_expert}",
            header.experts_per_layer
        );
    }
    let geoms = geometries_of(header)?;
    let total_experts = geoms.len() * n_expert;
    let stream = cuda_dev.cuda_stream();

    // Slot size comes from the *repacked* geometry: a slot holds one expert's
    // three projections at aligned offsets, and it is what the zone is carved
    // into, so it must be the figure the upload actually writes.
    let slot_bytes = slot_bytes_for(&geoms);
    // Two different numbers: where the boundary starts, and how far it may
    // ever go. The zone opens at `initial` — sized to leave the KV side its
    // measured cold-boot peak — and may grow to `limit` once the KV side has
    // shown what it actually uses.
    let slots_in = |bytes: usize| {
        bytes
            .checked_div(slot_bytes)
            .map_or(0, |n| n.min(total_experts))
    };
    let measured = slots_in(initial_weight_bytes(&stream)?);
    let limit = slots_in(weight_capacity_bytes(&stream)?);
    let floor = minimum_resident_slots(n_expert);
    // `minimum_resident_slots` is what the zone may never retract *below*: the
    // pinned head layers plus a full working layer. The zone cannot be given a
    // retraction floor above the ground it actually opened with, and raising
    // the opening size to meet it instead moves the elastic boundary and
    // starves the KV side (that is not hypothetical: it OOMs Qwen3-30B at load).
    //
    // On this family the measurement can come back at *zero* — every one of
    // the 35B's 40 layers routes, so the dense weights plus the KV side's
    // cold-boot peak can leave no ground at all — and a cache of nothing cannot
    // serve a layer. Opening at the floor takes that ground from the KV side,
    // which the elastic boundary renegotiates once the KV side has shown what
    // it really uses; opening below it is not a slower engine but a stopped one.
    let capacity = measured.max(floor).min(total_experts);
    let zone_floor = floor.min(capacity);
    // No bail here, deliberately: `capacity` is below the floor only when the
    // model has fewer experts in total than the floor prices — the
    // all-resident case, which is legal. The real check is in
    // `ExpertCache::new`, on the path every MoE loader takes.
    debug_assert!(capacity >= floor.min(total_experts));
    let zone = WeightZone::new(span_end(&stream)?, slot_bytes, capacity, limit, zone_floor);
    // Place the boundary; everything left of it belongs to the KV side.
    let kv_regions = set_weight_floor(&stream, zone.frontier_for_capacity())?;
    tracing::info!(
        target: "candle_transformers::qwen35",
        moe_layers = geoms.len(),
        experts_per_layer = n_expert,
        slots = capacity,
        zone_floor,
        max_slots = limit,
        floor_slots = floor,
        slot_bytes,
        kv_regions,
        "qwen35 expert cache opened against the span"
    );

    let cache = ExpertCache::new(ExpertCacheSetup {
        pack: section,
        zone,
        device,
        experts_per_layer: n_expert,
        experts_used: n_expert_used,
        progress,
        int8mode: pack.int8_mode,
        mapped_bytes: pack.gguf_len(),
        offloaded_bytes,
    })?;
    let cache = Arc::new(cache);
    // Open the shop: a KV arena claim that runs out of ground can now buy
    // more at the price of expert residency, rather than refusing. `Weak` so
    // the static registry does not outlive the model that owns the cache.
    let seller = Arc::downgrade(&cache);
    let candle::DeviceLocation::Cuda { gpu_id } = device.location() else {
        candle::bail!("qwen35: expert cache on a non-CUDA device")
    };
    candle_nn::kv_cache::set_ground_broker(gpu_id, move |regions| {
        seller.upgrade().map_or(0, |c| c.request_kv_ground(regions))
    });
    Ok(Some(cache))
}
