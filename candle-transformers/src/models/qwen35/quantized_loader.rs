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

use super::embedding::EmbeddingTable;
use super::expert_loader::build_expert_cache;
use super::quantized_weights::{load_quantized_model, QuantModel};
use crate::models::batched_model::ensure_vram_governor;
use crate::models::expert_lre::ExpertCache;

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
    // The embedding is the one dense tensor read per token rather than per
    // forward, so it is bound to host-mapped memory here — where the mapping
    // is — and the GPU gathers its rows from device-side ids. `None` falls back
    // to the F32 host table inside the load.
    let host_embed = EmbeddingTable::host_mapped(&content, &mmap);

    let mut reader = std::io::Cursor::new(&mmap[..]);
    let pack_dir = options.expert_pack_dir.clone();
    let mmap_for_cache = mmap.clone();
    load_quantized_model(
        &content,
        &mut reader,
        device,
        int8mode,
        host_embed,
        |content, cfg| -> Result<Option<Arc<ExpertCache>>> {
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
    )
}
