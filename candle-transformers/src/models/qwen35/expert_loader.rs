//! Standing up the expert cache for a MoE checkpoint.
//!
//! The caller's half of [`super::quantized_weights::load_quantized_model`]:
//! the cache is sized from a live measurement of what the dense weights left
//! behind, so it cannot be built inside the dense loop
//! (`docs/elastic_vram_partition.md` §4). The order is fixed —
//!
//! 1. dense weights resident (the loop in `quantized_weights`);
//! 2. measure the span, carve the weight zone, place the boundary;
//! 3. fill the cache into the zone that measurement produced.
//!
//! Nothing here is Qwen3.5-specific except the tensor names, which are the
//! frozen `ffn_{gate,up,down}_exps` schema. The expert machinery — the
//! cache, its tiers, the GPU-native/host dispatch fork — is the engine's and
//! is used exactly as Qwen3-MoE uses it.

use candle::quantized::gguf_file::Content;
use candle::{Device, Result};
use std::sync::Arc;

use super::config::Qwen35Config;
use crate::models::expert_lre::{
    minimum_resident_slots, ExpertCache, ExpertCacheSetup, MmapExpertRef,
};

/// Per-expert mmap references for every MoE layer, in layer order.
///
/// The checkpoint stores each projection as one **3-D merged** tensor
/// `[n_expert, out, in]`, so an expert's bytes are a fixed-size stride into
/// it — no per-expert tensor lookup, and the arithmetic is the same for
/// every layer. A checkpoint that split experts into 2-D per-expert tensors
/// would need the other branch; these do not, and are refused explicitly
/// rather than silently mis-read.
pub fn expert_host_refs(content: &Content, cfg: &Qwen35Config) -> Result<Vec<Vec<MmapExpertRef>>> {
    let moe = cfg
        .moe
        .ok_or_else(|| candle::Error::Msg("expert_host_refs: model declares no experts".into()))?;
    let n_expert = moe.n_experts;
    let mut all = Vec::new();

    for li in 0..cfg.num_layers {
        let p = format!("blk.{li}");
        let names = [
            format!("{p}.ffn_gate_exps.weight"),
            format!("{p}.ffn_up_exps.weight"),
            format!("{p}.ffn_down_exps.weight"),
        ];
        let infos: Vec<_> = names.iter().map(|n| content.tensor_infos.get(n)).collect();
        let (gate, up, down) = match (infos[0], infos[1], infos[2]) {
            (Some(g), Some(u), Some(d)) => (g, u, d),
            // Not a MoE layer at all — a mixed stack is legal, and its dense
            // layers simply contribute no experts.
            (None, None, None) => continue,
            _ => candle::bail!(
                "blk.{li} has only some of the merged expert tensors — a partial \
                 expert set cannot be indexed"
            ),
        };

        // `[n_expert, out, in]`: the leading dim must be the expert count, or
        // the stride below walks into the wrong expert.
        for (name, info) in names.iter().zip([gate, up, down]) {
            let dims = info.shape.dims();
            if dims.len() != 3 {
                candle::bail!(
                    "{name} is {}-D — this loader reads the merged \
                     [n_expert, out, in] form only",
                    dims.len()
                );
            }
            if dims[0] != n_expert {
                candle::bail!(
                    "{name} holds {} experts but the metadata declares {n_expert}",
                    dims[0]
                );
            }
        }

        let stride_of = |info: &candle::quantized::gguf_file::TensorInfo| -> usize {
            let elems: usize = info.shape.dims()[1..].iter().product();
            elems / info.ggml_dtype.block_size() * info.ggml_dtype.type_size()
        };
        let (gs, us, ds) = (stride_of(gate), stride_of(up), stride_of(down));
        let base = |info: &candle::quantized::gguf_file::TensorInfo| -> usize {
            (content.tensor_data_offset + info.offset) as usize
        };
        let (gb, ub, db) = (base(gate), base(up), base(down));

        let mut layer = Vec::with_capacity(n_expert);
        for e in 0..n_expert {
            layer.push(MmapExpertRef {
                gate_offset: gb + e * gs,
                gate_len: gs,
                up_offset: ub + e * us,
                up_len: us,
                down_offset: db + e * ds,
                down_len: ds,
                gate_shape: gate.shape.dims()[1..].to_vec(),
                up_shape: up.shape.dims()[1..].to_vec(),
                down_shape: down.shape.dims()[1..].to_vec(),
                gate_dtype: gate.ggml_dtype,
                up_dtype: up.ggml_dtype,
                down_dtype: down.ggml_dtype,
            });
        }
        all.push(layer);
    }
    Ok(all)
}

/// Build the expert cache for a MoE checkpoint, against the span the dense
/// weights left.
///
/// Returns `None` for a dense model (no expert tensors), which is not an
/// error — the 9B has no experts and wants no cache.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn build_expert_cache(
    content: &Content,
    cfg: &Qwen35Config,
    device: &Device,
    gguf_path: &std::path::Path,
    mmap: Arc<memmap2::Mmap>,
    int8mode: candle::quantized::Int8Mode,
    expert_pack_dir: Option<&std::path::Path>,
) -> Result<Option<Arc<ExpertCache>>> {
    use crate::models::expert_lre::{layer_geometries, slot_bytes_for};
    use candle_nn::kv_cache::{
        initial_weight_bytes, set_weight_floor, span_end, weight_capacity_bytes, WeightZone,
    };

    let Some(moe) = cfg.moe else {
        return Ok(None);
    };
    let host_refs = expert_host_refs(content, cfg)?;
    if host_refs.is_empty() {
        return Ok(None);
    }
    let n_expert = moe.n_experts;
    let total_experts = host_refs.len() * n_expert;

    let Device::Cuda(cuda_dev) = device else {
        candle::bail!("qwen35: the expert cache is a CUDA-only path");
    };
    let stream = cuda_dev.cuda_stream();

    // Slot size comes from the *repacked* geometry, not the raw GGML lengths:
    // a slot holds one expert's three projections at aligned offsets, and it
    // is what the zone is carved into, so it must be the figure the upload
    // actually writes.
    let geoms = layer_geometries(&host_refs, int8mode)?;
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
    // `minimum_resident_slots` is what the zone may never retract *below*, and
    // it is priced for a cache that pins three head layers. A model with 256
    // experts in every layer wants 769 slots on that basis, which a 16 GB card
    // does not have once the KV side has taken its cold-boot peak — but the
    // zone cannot be given a retraction floor above the ground it actually
    // opened with, and raising the opening size to meet it instead moves the
    // elastic boundary and starves the KV side (that is not hypothetical: it
    // OOMs Qwen3-30B at load).
    //
    // So the floor is clamped to what was measured, and the deadlock it used
    // to guard against is handled where it belongs — `affordable_pinned_layers`
    // derives the pinned count from the capacity the cache really has, so the
    // pinned set is always at least one layer's routed set short of the whole
    // cache and the eviction scan always has candidates.
    // On this family the measurement can come back at *zero* — every one of the
    // 35B's 40 layers routes, so the dense weights plus the KV side's cold-boot
    // peak can leave no ground at all — and a cache of nothing cannot serve a
    // layer. Opening at the floor takes that ground from the KV side, which the
    // elastic boundary renegotiates once the KV side has shown what it really
    // uses; opening below it is not a slower engine but a stopped one.
    //
    // The clamp is deliberately here and not in `minimum_resident_slots`:
    // that function feeds the *boundary placement* of every MoE model, and
    // moving it starves the KV side of a model that was fine (Qwen3-30B OOMs
    // at load).
    let capacity = measured.max(floor).min(total_experts);
    let zone_floor = floor.min(capacity);
    // What no amount of pinning arithmetic can rescue: a cache too small to
    // hold the routed set of the one layer it is executing.
    if capacity < n_expert + 1 {
        candle::bail!(
            "qwen35: this device affords {capacity} expert slots but a single MoE \
             layer can route to all {n_expert} of its experts at once — there is \
             no wave narrow enough to fit, and no eviction order that helps"
        );
    }
    let zone = WeightZone::new(span_end(&stream)?, slot_bytes, capacity, limit, zone_floor);
    // Place the boundary; everything left of it belongs to the KV side. The
    // region count is re-derived from the zone rather than assumed.
    let kv_regions = set_weight_floor(&stream, zone.frontier_for_capacity())?;
    tracing::info!(
        target: "candle_transformers::qwen35",
        moe_layers = host_refs.len(),
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
        mmap,
        host_refs,
        zone,
        device,
        experts_per_layer: n_expert,
        gguf_path,
        expert_pack_dir,
        progress: None,
        int8mode,
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

// The 35B-pinned stride audit for the merged expert tensors lives with the
// model it pins: `models/quantized_qwen35_moe.rs`.
