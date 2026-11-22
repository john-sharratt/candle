//! The inputs a model load takes beyond its checkpoint, bundled so each absence
//! has a name.
//!
//! A load needs its reader and device plus a few optional sources: a mapping
//! that lets projections be repacked a band at a time, a donor for the recurrent
//! gates, a sidecar for the draft head, and tensors to read from another file. An
//! override is exact: a tensor it names comes from that file, and the load fails
//! if one is never read. Absent sources are typed as `None` so each call reads
//! by name.

use super::*;

/// The `build_layers` type of a load that keeps every layer resident.
///
/// `None` carries no type of its own, and `load_quantized_model` is generic over
/// the builder, so a caller that does not stream needs *some* concrete type to
/// spell the absence with: `None::<ResidentLayers>`.
pub type ResidentLayers =
    fn(&gguf_file::Content, &Qwen35Config, Arc<Vec<ResidentResidue>>) -> Result<LayerStore>;

/// The `build_experts` type of a load with no expert cache — a dense checkpoint.
///
/// The same absence-needs-a-type problem [`ResidentLayers`] solves, one argument
/// over: `build_experts` is not an `Option`, so a dense load has to spell the
/// no-op, and a bare `|_, _| Ok(None)` per call site is one anonymous type per
/// call site — which is what stops [`LoadInputs::resident`] from naming a single
/// shared value.
pub type NoExperts = fn(&gguf_file::Content, &Qwen35Config) -> Result<Option<Arc<ExpertCache>>>;

/// What a load is given beyond the checkpoint, the reader and the device.
///
/// Four inputs that vary independently, passed as one value. Positionally they
/// were four arguments, and every one of them is an absence at a test call site
/// — `None, None, |_, _| Ok(None), None::<ResidentLayers>` names nothing, so a
/// reader had to count commas against the signature to find out which absence
/// was which. Now those call sites read [`LoadInputs::resident`].
pub struct LoadInputs<'a, F, L> {
    /// The embedding table, already pinned on the host by the caller. `None`
    /// dequantizes it to a plain host tensor instead — see [`EmbeddingTable`].
    pub host_embed: Option<HostEmbedding>,
    /// The NextN draft head's own GGUF and its mapped bytes, for a checkpoint
    /// whose head ships as a sidecar rather than embedded in the trunk file.
    /// `None` reads the head from `content` if it declares one, and leaves the
    /// model without a drafter if it does not.
    pub mtp_src: Option<(&'a gguf_file::Content, &'a [u8])>,
    /// The trunk checkpoint's own mapped bytes.
    ///
    /// **What lets a projection be repacked without ever existing on the device.** With it,
    /// [`Loader::proj`] reads a band at a time straight from the mapping; without it the
    /// tensor is uploaded whole first, which for a `[248320, 4096]` BF16 head is 1,940 MiB
    /// that exists only to be read once and dropped — and is what OOMed a 24 GiB card with
    /// 14 GiB free.
    ///
    /// `None` is the ordinary answer for a caller that has a reader and no mapping (a test
    /// fixture, a sidecar read through a cursor); the load simply takes the older route.
    pub map: Option<&'a [u8]>,
    /// A checkpoint to take the DeltaNet recurrent gates from, when this one quantized them.
    ///
    /// **A repair for a specific, common defect and not a general weight-mixing facility.**
    /// `ssm_alpha`/`ssm_beta` must be F32 (see [`check_recurrent_precision`]); a conversion
    /// whose quant rules do not know this architecture stores them like any other 2-D weight
    /// and the model then generates incoherent text from its first token. The precision is
    /// gone from that file, but the *base* checkpoint the fine-tune was built from still has
    /// it — and these are the tensors a fine-tune has least reason to have moved.
    ///
    /// Only the gates are taken, only when the primary's are not F32, and only when the
    /// shapes agree. Everything that makes the fine-tune what it is — attention, the FFN, the
    /// head — is the primary's throughout.
    pub gate_src: Option<(&'a gguf_file::Content, &'a [u8])>,
    /// Tensors to read from another checkpoint instead of this one — see
    /// [`TensorOverrides`].
    ///
    /// **The general facility `gate_src` is not.** Where the donor repairs a known defect and
    /// decides for itself which tensors need it, an override is the caller naming exact tensors
    /// and exact files. The load refuses one it never read, so an override is either on the
    /// card or the load fails.
    pub overrides: Option<&'a TensorOverrides<'a>>,
    /// Builds the expert cache at the one point the span means what it says —
    /// see this module's [`load_quantized_model`] header.
    pub build_experts: F,
    /// Builds the layer store, and by being `Some` at all decides that the
    /// trunk loop does not run. See [`load_quantized_model`].
    pub build_layers: Option<L>,
}

impl LoadInputs<'_, NoExperts, ResidentLayers> {
    /// A dense checkpoint loaded whole: no expert cache, no sidecar head, every
    /// layer resident in the dense block.
    ///
    /// What a test harness wants — it dequantizes projections to build a
    /// reference, which needs the weights in hand rather than a slot's view of
    /// them — and never what production wants, since production streams every
    /// dense checkpoint.
    pub fn resident() -> Self {
        Self {
            host_embed: None,
            mtp_src: None,
            // A harness passes a reader, not a mapping, so the host-banded path is simply not
            // available to it — and does not need to be: a reference load is not the one
            // competing for the card.
            map: None,
            // A reference load reads a checkpoint that already satisfies the precision rule,
            // so there is nothing to repair and no second file to repair it from.
            gate_src: None,
            overrides: None,
            build_experts: |_, _| Ok(None),
            build_layers: None,
        }
    }
}

/// The driver's ground this load needs — [`dense_span::peak_load_pool_bytes`]
/// over every tensor it reads to the device whole, each as the load will read it
/// (an override's own entry where one replaces the checkpoint's).
///
/// The span never grows, so this figure stays outside it for the life of the
/// process (`docs/driver_model.md` §4): on a routed model every MiB here is a MiB
/// of expert slots. Two tensors therefore stay out of it, each because the card
/// never holds it whole:
///
/// - the token embedding, which lives on the host (`EmbeddingTable`), host-mapped
///   or read there ([`Loader::host`]) — 515 MiB on the 35B, its largest tensor;
/// - the output head, when its twin is repacked from the mapping a band at a time
///   (`banded`: a mapping and a CUDA device, and [`dense_span::bands_from_host`]
///   for the tensor) — 398 MiB on the 35B. With tied embeddings the head is the
///   embedding read a second time, through the projection route, so it is left
///   out under the same rule and counted otherwise.
///
/// Bounded with them the 35B's headroom was 611 MiB; without them it is its
/// widest `attn_qkv` (17 MiB) and the repack's two bands, 113 MiB.
pub(super) fn load_headroom_bytes(
    content: &gguf_file::Content,
    overrides: Option<&TensorOverrides<'_>>,
    banded: bool,
    mode: Int8Mode,
) -> usize {
    const EMBEDDING: &str = "token_embd.weight";
    let head = if content.tensor_infos.contains_key("output.weight") {
        "output.weight"
    } else {
        EMBEDDING
    };
    dense_span::peak_load_pool_bytes(content.tensor_infos.iter().filter_map(|(name, info)| {
        let name = name.as_str();
        let info = overrides
            .and_then(|o| o.effective_info(content, name))
            .unwrap_or(info);
        let on_host = if name == head {
            banded && dense_span::bands_from_host(info, mode)
        } else {
            name == EMBEDDING
        };
        (!on_host).then_some(info)
    }))
}
