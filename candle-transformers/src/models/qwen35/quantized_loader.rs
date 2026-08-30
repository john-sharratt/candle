//! Loading a production hybrid from a GGUF file.
//!
//! [`quantized_weights`](super::quantized_weights) maps checkpoint tensors onto
//! weights; this is the surrounding orchestration — mapping the file, choosing
//! the numeric mode, and standing the expert cache up against the span the
//! dense weights leave behind.
//!
//! The order is fixed and structural (`docs/elastic_vram_partition.md` §4):
//! dense weights resident → measure the span, carve the weight zone, place the
//! boundary → fill the cache → graft it onto the layers that route. The
//! measurement is only meaningful at that one point, which is why
//! `load_quantized_model` takes the cache builder as a callback and calls it
//! there rather than trusting a caller to sequence it.
//!
//! The pinned-checkpoint gates live with the models they pin:
//! `models/quantized_qwen35.rs` (dense 0.8B / 9B) and
//! `models/quantized_qwen35_moe.rs` (35B-A3B).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle::quantized::{gguf_file::Content, Int8Mode};
use candle::{Device, Result};

use super::config::Qwen35Config;
use super::embedding::EmbeddingTable;
use super::expert_loader::build_expert_cache;
use super::layer_loader::build_layer_cache;
use super::layer_store::LayerStore;
use super::quantized_weights::{load_quantized_model, LoadInputs, QuantModel};
use crate::models::batched_model::ensure_vram_governor;
use crate::models::expert_lre::pack::repack_fingerprint;
use crate::models::expert_lre::{ExpertCache, PINNED_LAYERS};
use crate::models::layer_stream::PackIdentity;

/// Load-time knobs.
#[derive(Debug, Default, Clone)]
pub struct Qwen35LoadOptions {
    /// Numeric path for every projection. `None` picks int8 on an int8-MMA
    /// GPU whose headroom allows it, sized by the checkpoint's length.
    pub int8mode: Option<Int8Mode>,
    /// Where the authoritative expert pack file lives.
    ///
    /// `None` makes the pack **ephemeral**: it is written under the system temp
    /// directory and unlinked the moment its writer publishes it, so nothing
    /// reuses it and every load repacks every expert. That is the right default
    /// for a test, and the wrong one for anything that restarts — name a
    /// directory (the checkpoint's own is the usual answer, so the pack is
    /// shared by every workspace on that model) to turn the repack into a read.
    pub expert_pack_dir: Option<PathBuf>,
    /// A GGUF holding the NextN / MTP draft head, when the checkpoint does not.
    ///
    /// Two conventions are in the wild. Unsloth embeds the head's tensors in the
    /// main file, and nothing needs naming here. ggml-org ships them as a
    /// sidecar — `mtp-Qwen3.8-27B-Q4_0.gguf` beside `Qwen3.8-27B-Q4_K_M.gguf` —
    /// whose main file declares no `nextn_predict_layers` at all, so without
    /// this the model loads perfectly and simply never drafts.
    ///
    /// That is worth a knob rather than a guess at a filename: what the head
    /// costs is VRAM the layer zone would otherwise hold, and on a card where
    /// the model already streams that is a trade the caller should make
    /// explicitly.
    pub mtp_path: Option<PathBuf>,
}

/// Load a hybrid checkpoint of this lineage.
///
/// Arch and dense-vs-routed are detected from the GGUF. The per-model entry
/// points (`quantized_qwen35::from_gguf_path` and its siblings) wrap this
/// with the identity checks and construct the scheduler-facing
/// [`super::batched::HybridBatched`] around it with the model's own derived
/// KV threshold factors — which is why this returns the bare [`QuantModel`]
/// rather than the wrapper.
pub fn load_hybrid_gguf(
    file_path: &Path,
    device: &Device,
    options: Qwen35LoadOptions,
) -> Result<QuantModel> {
    use memmap2::MmapOptions;

    // Before anything allocates: the KV region span is sized from the
    // governor's balloon-measured capacity, and without one it falls back to
    // the 3 GiB governor-less test constant — which silently caps the whole
    // partition (measured on the 3.6-35B: a 3,024 MiB span left the expert
    // zone a 529-slot / 1.0 GiB ceiling on a 16 GB card, pinning the VRAM
    // expert hit rate under 17% and pushing 30–40% of expert loads to the
    // NVMe pack).
    ensure_vram_governor(device);

    let int8mode = match options.int8mode {
        Some(m) => m,
        None => {
            let model_bytes = std::fs::metadata(file_path)
                .map(|m| m.len() as usize)
                .unwrap_or(0);
            Int8Mode::auto_sized(device, model_bytes)
        }
    };

    let file = std::fs::File::open(file_path)?;
    let mmap = unsafe {
        MmapOptions::new()
            .map(&file)
            .map_err(|e| candle::Error::Msg(format!("qwen35: failed to mmap {file_path:?}: {e}")))?
    };
    let mmap = Arc::new(mmap);
    // Feed the host-RAM budget: reserving the full mapping means warm-KV growth
    // can never push weight pages out of RAM.
    candle::vram::set_weights_mmap(mmap.len() as u64);

    // **Deliberately not host-registered.** Pinning the mapping would buy
    // full-DMA H2D out of it, but the experts move to the pack file at startup
    // and what stays live here is the dense tensors and the embedding table,
    // gathered a few rows at a time. Registering would lock the whole file
    // non-pageable for the process lifetime, competing directly with the warm
    // expert tier — which is the thing that keeps expert loads off the disk.

    let mut cursor = std::io::Cursor::new(&mmap[..]);
    let content = Content::read(&mut cursor)?;

    tracing::info!(
        target: "candle_transformers::qwen35",
        ?int8mode,
        file = ?file_path,
        "loading qwen35 checkpoint"
    );

    // Weights are read *through* the mapping into device or host buffers rather
    // than aliased out of it, so the mapping itself only has to outlive the
    // load — except on a routed checkpoint, where the expert cache keeps its own
    // `Arc` and streams from it for the life of the model.
    // The draft head's own file, when the checkpoint keeps the head outside
    // itself. Mapped here so its `Content` and bytes outlive the load, exactly
    // as the checkpoint's do.
    let mtp_mmap = match options.mtp_path.as_deref() {
        None => None,
        Some(p) => {
            let f = std::fs::File::open(p)?;
            let m = unsafe {
                MmapOptions::new().map(&f).map_err(|e| {
                    candle::Error::Msg(format!("qwen35: failed to mmap MTP head {p:?}: {e}"))
                })?
            };
            let mut c = std::io::Cursor::new(&m[..]);
            let content = Content::read(&mut c)?;
            tracing::info!(
                target: "candle_transformers::qwen35",
                file = ?p,
                "loading the MTP draft head from a sidecar"
            );
            Some((content, m))
        }
    };
    let mtp_src = mtp_mmap.as_ref().map(|(c, m)| (c, &m[..]));

    // The embedding is the one dense tensor read per token rather than per forward, so it is
    // bound to host-mapped memory here — where the mappings are — and the GPU gathers its rows
    // from device-side ids. `None` falls back to the F32 host table inside the load.
    //
    // Decided after the sidecar is mapped, because it is a candidate: the table costs host RAM
    // and no VRAM, so the wider of the two copies is taken. See `widest_host_mapped`.
    let mut embed_sources: Vec<(&Content, &memmap2::Mmap)> = vec![(&content, &mmap)];
    if let Some((c, m)) = mtp_mmap.as_ref() {
        embed_sources.push((c, m));
    }
    let host_embed = EmbeddingTable::widest_host_mapped(&embed_sources);
    drop(embed_sources);

    let mut reader = std::io::Cursor::new(&mmap[..]);
    let pack_dir = options.expert_pack_dir.clone();
    let mmap_for_cache = mmap.clone();
    let mmap_for_layers = mmap.clone();
    let inputs = LoadInputs {
        host_embed,
        mtp_src,
        build_experts: |content: &Content,
                        cfg: &Qwen35Config|
         -> Result<Option<Arc<ExpertCache>>> {
            build_expert_cache(
                content,
                cfg,
                device,
                file_path,
                mmap_for_cache,
                int8mode,
                pack_dir.as_deref(),
            )
        },
        // **Every dense checkpoint streams its layers** —
        // `docs/qwen38_layer_streaming.md` §7. A model that fits is the
        // degenerate case of the same mechanism: capacity covers the trunk,
        // nothing is ever evicted, and no byte moves after load. A routed
        // checkpoint is filtered out inside `load_quantized_model`, where the
        // config is already parsed.
        //
        // Its own cursor over the same mapping, so the pack build and the
        // pinned-head upload read the checkpoint without contending for the
        // loader's reader.
        build_layers: Some(
            |content: &Content, cfg: &Qwen35Config, residues| -> Result<LayerStore> {
                let Device::Cuda(cuda) = device else {
                    candle::bail!("qwen35: layer streaming is a CUDA-only path")
                };
                let mut reader = std::io::Cursor::new(&mmap_for_layers[..]);
                // The **same** fingerprint the expert pack uses: both hold
                // weights repacked by identical code, so a change that
                // invalidates one must invalidate the other.
                let identity =
                    PackIdentity::of(&mmap_for_layers[..], int8mode, repack_fingerprint(cuda));
                build_layer_cache(
                    content,
                    &mut reader,
                    device,
                    cfg,
                    int8mode,
                    file_path,
                    identity,
                    PINNED_LAYERS,
                    residues,
                )
            },
        ),
    };
    load_quantized_model(&content, &mut reader, device, int8mode, inputs)
}
