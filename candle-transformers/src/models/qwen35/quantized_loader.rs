//! Loading a production hybrid from its model pack.
//!
//! [`quantized_weights`](super::quantized_weights) maps checkpoint tensors onto
//! weights; this is the surrounding orchestration — opening the pack, and
//! standing the expert cache or the layer cache up against the span the dense
//! weights leave behind.
//!
//! The order is fixed and structural (`docs/archived/elastic_vram_partition.md` §4):
//! dense weights resident → measure the span, carve the weight zone, place the
//! boundary → fill the cache → graft it onto the layers that route. The
//! measurement is only meaningful at that one point, which is why
//! `load_quantized_model` takes the cache builder as a callback and calls it
//! there rather than trusting a caller to sequence it.
//!
//! The pack is the whole model: a draft head shipped as a sidecar, a gate
//! donor's recurrent path and any tensor override were folded in when it was
//! built (`pack_build`), so a load opens one file. The pinned-checkpoint gates
//! live with the models they pin: `models/quantized_qwen35.rs` (dense 0.8B /
//! 9B) and `models/quantized_qwen35_moe.rs` (35B-A3B).

use std::io::Cursor;
use std::path::Path;

use candle::{Device, Result};

use super::embedding::EmbeddingTable;
use super::expert_loader::build_expert_cache;
use super::layer_loader::build_layer_cache;
use super::quantized_weights::{load_quantized_model, LoadInputs, QuantModel};
use crate::models::batched_model::ensure_vram_governor;
use crate::models::model_pack::ModelPack;

/// Load a hybrid of this lineage from its model pack.
///
/// Arch and dense-vs-routed are detected from the pack's GGUF part, and the
/// numeric mode is the one the pack was built for. The per-model entry points
/// (`quantized_qwen35::from_pack` and its siblings) wrap this with the identity
/// checks and construct the scheduler-facing [`super::batched::HybridBatched`]
/// around it with the model's own derived KV threshold factors — which is why
/// this returns the bare [`QuantModel`] rather than the wrapper.
pub fn load_hybrid_pack(path: &Path, device: &Device) -> Result<QuantModel> {
    // Before anything allocates: the KV region span is sized from the
    // governor's balloon-measured capacity, and without one it falls back to
    // the 3 GiB governor-less test constant — which silently caps the whole
    // partition (measured on the 3.6-35B: a 3,024 MiB span left the expert
    // zone a 529-slot / 1.0 GiB ceiling on a 16 GB card).
    ensure_vram_governor(device);

    let pack = ModelPack::open(path)?;
    let int8mode = pack.int8_mode;
    // Feed the host-RAM budget: reserving the mapping means warm-KV growth can
    // never push weight pages out of RAM. The mapping is the GGUF part alone;
    // the sections are read with direct I/O.
    candle::vram::set_weights_mmap(pack.gguf_len());

    tracing::info!(
        target: "candle_transformers::qwen35",
        ?int8mode,
        pack = ?path,
        "loading qwen35 model pack"
    );

    // **Deliberately not host-registered.** Pinning the mapping would buy
    // full-DMA H2D out of it, but what lives here is the dense tensors and the
    // embedding table, gathered a few rows at a time. Registering would lock
    // the whole part non-pageable for the process lifetime, competing directly
    // with the warm tiers — which are what keep expert and layer loads off the
    // disk.
    //
    // The embedding is the one dense tensor read per token rather than per
    // forward, so it is bound to host-mapped memory here and the GPU gathers
    // its rows from device-side ids. `None` falls back to the F32 host table
    // inside the load.
    let host_embed = EmbeddingTable::widest_host_mapped(&[(&pack.content, &*pack.mmap)]);
    let mut reader = Cursor::new(&pack.mmap[..]);
    let inputs = LoadInputs {
        host_embed,
        mtp_src: None,
        gate_src: None,
        overrides: None,
        // The mapping, so a large projection is repacked from it a band at a
        // time rather than uploaded whole first.
        map: Some(&pack.mmap[..]),
        build_experts: |_content: &_, cfg: &_| build_expert_cache(&pack, cfg, device),
        // **Every dense checkpoint streams its layers** —
        // `docs/archived/qwen38_layer_streaming.md` §7. A model that fits is
        // the degenerate case of the same mechanism: capacity covers the trunk,
        // nothing is ever evicted, and no byte moves after load. A routed
        // checkpoint is filtered out inside `load_quantized_model`, where the
        // config is already parsed.
        build_layers: Some(|_content: &_, cfg: &_, residues| {
            build_layer_cache(&pack, device, cfg, residues)
        }),
    };
    load_quantized_model(&pack.content, &mut reader, device, int8mode, inputs)
}
