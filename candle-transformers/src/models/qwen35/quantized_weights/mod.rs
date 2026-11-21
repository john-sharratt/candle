//! Production weights for the hybrid stack, loaded by role.
//!
//! Projections are quantized and repacked for the numeric mode, norms stay
//! re-materialisable so they can emit the activation type, and the recurrence's
//! constants are kept in F32. The trunk is resident or streamed through a layer
//! store. On a card that streams, the output head and the draft head are narrowed.
//!
//! - `layer`: a trunk layer's weights and its residue.
//! - `layer_load`: reading a trunk layer and the MTP head from a checkpoint.
//! - `load_inputs`: the optional inputs a load takes beyond its checkpoint.
//! - `loader`: the per-tensor reader and its choice of source.
//! - `narrowing`: the schedule for a streamed trunk.
//! - `recurrent`: the recurrent gates, their precision guard and donor repair.
//! - `tests_headroom`: device headroom accounting.
//! - `tests_narrowing`: the narrowing schedule's table.
//! - `tests_recurrent_precision`: the gate guard and the donor rules.

use candle::quantized::{gguf_file, GgmlDType, Int8Mode, QTensor};
use candle::{Device, Result, Tensor};
use std::io::{Read, Seek};
#[cfg(feature = "cuda")]
use std::sync::Arc;

use super::config::Qwen35Config;
use super::embedding::EmbeddingTable;
use super::layer_store::LayerStore;
use super::mtp::{MtpHead, MtpInput};
#[cfg(feature = "cuda")]
use super::quantized_moe::Qwen35MoeBlock;
use super::tensor_override::TensorOverrides;
use crate::models::delta_net::{LayerKind, QuantDeltaNetWeights};
use crate::models::dense_span;
#[cfg(feature = "cuda")]
use crate::models::expert_lre::ExpertCache;
use crate::models::host_embedding::HostEmbedding;
use crate::models::layer_stream::LayerTensor;
use crate::models::model_pack::compose::checkpoint_bytes_of;
use crate::models::quantized_matmul::{QMatMul, WeightResidency};
use crate::models::quantized_mlp::QuantizedMlp;
#[cfg(feature = "cuda")]
use crate::models::quantized_qwen3_moe::SparseMoeBlock;
use crate::quantized_nn::RmsNorm;

/// The loaded production model.
///
/// `embed` is deliberately **off the card**: a `vocab × hidden` table is 4 GB at
/// the 9B's geometry for one row read per token, the worst VRAM-per-access ratio
/// in the model. Where it lives instead, and why the production path no longer
/// pays a GPU→CPU touch for it, is [`EmbeddingTable`]. The tied LM head is a
/// separate device-side [`QMatMul`] over the same checkpoint tensor, because
/// that one *is* a matmul and reads its whole weight every call.
pub struct QuantModel {
    pub cfg: Qwen35Config,
    /// The token-embedding table, off the card either way — see
    /// [`EmbeddingTable`] for what the two residencies cost.
    pub embed: EmbeddingTable,
    /// The trunk's layers, resident or streamed — see [`LayerStore`].
    ///
    /// Not a `Vec`: on a dense checkpoint the layers are tenants of the weight
    /// zone's slots and only a working set of them exists at any moment, so
    /// there is no vector to index. Everything that wants a layer's *weights*
    /// goes through [`LayerStore::ensure`], and everything that wants only its
    /// norms or the recurrence's constants goes through
    /// [`LayerStore::residue`], which is resident either way.
    pub layers: LayerStore,
    /// The NextN / MTP draft head, when the conversion kept it.
    ///
    /// `None` on a plain GGUF — which is not a property of the architecture but
    /// of the file: Qwen3.5/3.6 are trained with multi-token prediction and
    /// ship the head, and unsloth publishes a parallel `…-MTP-GGUF` repo per
    /// model that keeps it. With `None` the lineage drafts through the
    /// weightless fallback instead. See [`super::mtp`].
    pub mtp: Option<MtpHead>,
    /// The shared expert cache, `None` on a dense checkpoint.
    ///
    /// Also reachable through any MoE layer's router, but held here as well
    /// because the model-level answers the scheduler asks — how much weight
    /// ground can be ceded, what the resident footprint is, the pipeline's
    /// telemetry — are about the cache as a whole and must not depend on the
    /// stack having a MoE layer at a particular index.
    #[cfg(feature = "cuda")]
    pub experts: Option<Arc<ExpertCache>>,
    /// The head's norm — a fused producer feeding `lm_head`, so [`RmsNorm`]
    /// rather than a bare gain, exactly like the per-layer norms.
    pub final_norm: RmsNorm,
    pub lm_head: QMatMul,
    pub device: Device,
    /// VRAM the DENSE tensors hold — everything above except the paged experts.
    ///
    /// Kept because a footprint that is only ever handed to the governor is a
    /// footprint no report can read back. `resident_weight_bytes` is documented
    /// as "fixed base + time-varying resident experts" and could answer only the
    /// second half without this, so the whole-card decomposition subtracted the
    /// experts from themselves and reported the dense half as ZERO — several
    /// GiB of VRAM invisible to every consumer, including the budget that is
    /// supposed to size the reservation around it.
    pub dense_bytes: usize,
}

pub fn load_quantized_model<R, F, L>(
    content: &gguf_file::Content,
    reader: &mut R,
    device: &Device,
    mode: Int8Mode,
    inputs: LoadInputs<'_, F, L>,
) -> Result<QuantModel>
where
    R: Read + Seek,
    F: FnOnce(&gguf_file::Content, &Qwen35Config) -> Result<Option<Arc<ExpertCache>>>,
    L: FnOnce(&gguf_file::Content, &Qwen35Config, Arc<Vec<ResidentResidue>>) -> Result<LayerStore>,
{
    let LoadInputs {
        host_embed,
        mtp_src,
        map,
        gate_src,
        overrides,
        build_experts,
        build_layers,
    } = inputs;

    // A checkpoint that quantized its recurrent gates is refused unless a donor was supplied
    // to repair them. Decided before anything is read, so an unusable checkpoint costs a
    // header rather than a full load.
    let repairing = match (undersized_gates(content).is_empty(), gate_src) {
        (true, _) => false,
        (false, None) => return Err(check_recurrent_precision(content).unwrap_err()),
        // Headers only — the tensors themselves are read lazily, per loader, by
        // `Loader::donated`.
        (false, Some((donor, _))) => {
            // Validated here, once, rather than at each of the many places a gate is read:
            // every loader below asks `Loader::donated` per tensor and would otherwise
            // rediscover the same donor mismatch a dozen times over. A mismatch is fatal, so
            // finding it before any weight is read is also what makes it cheap to report.
            let n = donor_gates(content, donor)?.len();
            // Loud, because the model being run is not the one on disk: its recurrence comes
            // from the base checkpoint. Worth a line in every log that will ever be read back
            // while wondering why this fine-tune behaves as it does.
            tracing::warn!(
                target: "candle_transformers::qwen35",
                tensors = n,
                "this checkpoint quantized its DeltaNet recurrent path; reading it from the \
                 base checkpoint instead. Attention, the FFN and the head are the \
                 checkpoint's own."
            );
            true
        }
    };
    let gate_src = repairing.then_some(gate_src).flatten();

    let arch = super::loader::detect_arch(content);
    let mut cfg = Qwen35Config::from_gguf_metadata(&arch, &content.metadata)?;

    // **Claim the span before a single weight is read** — see `dense_span`. The
    // reservation used to be created lazily by the expert cache, which runs at
    // the *end* of this function, so it was sized from the VRAM left after the
    // weights had already been taken from the pool and could not contain them.
    // Sized from what this load will read: an override from another file can be a wider
    // tensor than the entry it replaces, and the pool has to hold the largest source.
    let banded = map.is_some() && device.is_cuda();
    let headroom = load_headroom_bytes(content, overrides, banded, mode);
    // A plain `[rows, hidden]` residual, which the pool cushion covers: no runtime reserve.
    let _load = dense_span::open_for_load_sized(device, headroom)?;

    // A routed checkpoint's layers stay resident however the caller asked — see
    // this function's header. Resolved here, where the config is known, so the
    // caller does not have to parse it a second time to find out.
    let build_layers = build_layers.filter(|_| cfg.moe.is_none());
    // Whether this load narrows its two permanently-resident weights. A hardware-and-model
    // fact, derived here where both halves are known, rather than a knob: see
    // [`narrow_resident_twin`].
    let narrow_resident = narrow_resident_twin(device, &cfg, checkpoint_bytes_of(content));

    let mut g = Loader::new(content, reader, device, mode, WeightResidency::Span)
        .with_map(map)
        .with_gate_src(gate_src)
        .with_overrides(overrides);

    // The embedding table, off the card under either residency (see
    // [`EmbeddingTable`]). Host-mapped when the caller could pin it, which is
    // strictly better and therefore not a choice made here.
    let embed = match host_embed {
        Some(h) => EmbeddingTable::HostMapped(h),
        None => {
            // Read and widened on the host: the table never touches the card,
            // so the load headroom leaves it out (`load_headroom_bytes`).
            let table = g.host("token_embd.weight")?.dequantize(&Device::Cpu)?;
            EmbeddingTable::Host(table)
        }
    };
    let final_norm = g.norm("output_norm.weight", cfg.rms_norm_eps)?;
    // **The output head is narrowed on a card that streams.** It is one of only two weights
    // that stay resident whatever the trunk does, it is often published *wider* than the trunk
    // (bartowski ships the 27B's at Q6_K against a Q3_K body), and every byte it holds is a
    // layer slot the zone does not have — which is bandwidth on every forward, not VRAM once.
    // Its own precision is the cheapest to give up: it decides a token from a 248k-way argmax,
    // where the trunk's projections decide the residual stream the whole rest of the forward
    // reads. `narrow_resident` is `None` on a card that holds the model, so nothing changes
    // there. See `Loader::narrow`.
    g.set_narrow(narrow_resident);
    let lm_head = if g.has("output.weight") {
        g.proj("output.weight")?
    } else {
        g.proj("token_embd.weight")?
    };
    g.set_narrow(None);

    // MoE layers are numbered among themselves — the expert cache keys on
    // `(moe_layer_idx, expert)`, not on the trunk layer index, so a stack
    // that mixes dense and MoE layers still indexes the cache densely.
    #[cfg(feature = "cuda")]
    let mut moe_layer_idx_next = 0usize;

    // The trunk, in whichever of its two forms this load takes.
    //
    // Streamed: the projections do NOT come into the dense block — that is the
    // whole point — so only each layer's residue is read, and it is read *here*,
    // inside the load window, so it lands in the span with the rest of the
    // resident model rather than in the pool after the block is frozen.
    let mut pending_layers = Vec::new();
    let mut residues = Vec::new();
    if build_layers.is_some() {
        residues.reserve_exact(cfg.num_layers);
        for li in 0..cfg.num_layers {
            residues.push(load_residue(&mut g, &cfg, li)?);
        }
    } else {
        pending_layers.reserve_exact(cfg.num_layers);
        // The trunk's narrowing schedule applies only here — to the layers themselves. It is
        // set for the loop and cleared after, so nothing loaded outside it (the head, the draft
        // block, the residues) can pick up a per-layer rule meant for a slot tenant.
        g.set_stream_narrow(narrow_resident.map(|_| cfg.num_layers));
        for li in 0..cfg.num_layers {
            pending_layers.push(load_layer(
                &mut g,
                &cfg,
                li,
                mode,
                #[cfg(feature = "cuda")]
                &mut moe_layer_idx_next,
            )?);
        }
        g.set_stream_narrow(None);
    }
    let residues = std::sync::Arc::new(residues);

    // ── Every dense tensor is resident. NOW the span means what it says. ──
    //
    // Record the dense footprint with the governor first. `kv_floor` is
    // `abs + pct × (C − weights)`, so leaving it unrecorded computes the floor
    // against the whole card as though the model were free — inflating the
    // floor and taking the difference straight out of the expert budget, which
    // is the scarcest thing on a tight card. Set, not add: this is the whole
    // dense footprint, and adding it twice drives `C − weights` to zero.
    #[cfg(feature = "cuda")]
    if let candle::DeviceLocation::Cuda { gpu_id } = device.location() {
        if let Some(gov) = candle::vram::get(gpu_id) {
            gov.set_class(candle::vram::AllocClass::Weights, g.device_bytes as u64);
        }
        tracing::info!(
            target: "candle_transformers::qwen35",
            dense_mib = g.device_bytes >> 20,
            layers = cfg.num_layers,
            "qwen35 dense weights resident"
        );
    }
    // ── The NextN / MTP draft head, when the conversion kept it ──
    //
    // `nextn_predict_layers` blocks sit past the trunk, and the trunk loop
    // above stops before them because `cfg.num_layers` already excludes them.
    // The head is structurally a trunk attention layer (see `super::mtp`), so
    // it loads through the same two helpers.
    //
    // Absent on a plain GGUF conversion, which drops the MTP tensors — that is
    // the common case and is not an error; the model simply has no drafter and
    // decodes a token at a time. Present-but-malformed IS an error.
    //
    // # The head may live in a **second file**
    //
    // Two conventions are in the wild and this reads both. Unsloth embeds the
    // NextN tensors in the main GGUF, so `cfg.num_mtp_layers` is non-zero and
    // `blk.{num_layers}.*` resolves against the checkpoint already open. ggml-org
    // splits them into a sidecar — `mtp-Qwen3.8-27B-Q4_0.gguf` beside
    // `Qwen3.8-27B-Q4_K_M.gguf` — whose main file declares **no**
    // `nextn_predict_layers` at all, so a loader that only looked there would
    // decode the 27B a token at a time and report nothing amiss.
    //
    // The sidecar's tensors are named exactly as the embedded convention names
    // them (`blk.64.nextn.eh_proj.weight` and the rest of a trunk attention
    // block), so the only difference is which reader they come from. It also
    // carries its own `token_embd`/`output`, which are ignored: the head shares
    // the target's, which is what makes it cost one block rather than a second
    // model.
    let mut mtp_cursor = mtp_src.map(|(c, bytes)| (c, std::io::Cursor::new(bytes)));
    let (mtp_layers, mut head_loader) = match mtp_cursor {
        Some((content2, ref mut cur)) => {
            let n = Qwen35Config::mtp_layers_in(&content2.metadata, &arch);
            // **The config has to learn the head exists**, and it cannot learn
            // it from the checkpoint: `num_mtp_layers` was parsed from the main
            // file, which under the sidecar convention says nothing about a
            // head. Everything downstream gates on this field —
            // `engine::mtp_kv_layer` returns `None` without it, so the head gets
            // no paged KV layer to write, `session_kv_layers` sizes the cache
            // without one, and `wave_kv_range` never covers the head's pass.
            //
            // The head then drafts into nothing and **every proposal is
            // rejected**: measured on the 27B as `draft budget 2` with
            // `1.00 accepted/step`, which reads as a drafter that is merely bad
            // rather than one with nowhere to write.
            //
            // `num_layers` is *not* adjusted. `from_gguf_metadata` subtracts the
            // head from `block_count` because the embedded convention counts it
            // there; a sidecar's head was never in the main file's count, so the
            // trunk is already the right size.
            // **A named sidecar that declares no head is an error, not a
            // fallback.** The caller went to the trouble of pointing at a file;
            // reading zero out of it means the arch string does not match or the
            // conversion dropped the key, and silently continuing produces
            // exactly the failure the knob exists to remove — a model that loads
            // clean, drafts nothing, and reports `draft budget 0`.
            if n == 0 {
                candle::bail!(
                    "qwen35: the MTP sidecar declares no {arch}.nextn_predict_layers, so it \
                     carries no draft head this loader can use. Either it is not an MTP \
                     conversion or it is for a different architecture."
                );
            }
            cfg.num_mtp_layers = n;
            (
                n,
                Some(Loader::new(
                    content2,
                    cur,
                    device,
                    mode,
                    WeightResidency::Span,
                )),
            )
        }
        None => (cfg.num_mtp_layers, None),
    };
    // The head reads from the sidecar when there is one and from the checkpoint
    // otherwise. Two reader *types*, so this dispatches rather than unifying
    // them — one transcription of the tensor names either way.
    // **The drafter is narrowed for the same reason, and it is the safer of the two.** The MTP
    // block is a full layer's worth of permanently-resident weight earning a 1-in-65 share of
    // the forward, so on a streaming card it is the most expensive residency in the model. And
    // speculation is *lossless*: the trunk verifies every proposal, so a coarser drafter moves
    // the acceptance rate and can never move a token. Precision here buys throughput, not
    // correctness, which is exactly the thing worth trading for layer slots.
    let mtp = match (mtp_layers, head_loader.as_mut()) {
        (0, _) => None,
        (n, Some(h)) => {
            h.set_narrow(narrow_resident);
            let head = load_mtp_head(
                h,
                &cfg,
                n,
                mode,
                #[cfg(feature = "cuda")]
                &mut moe_layer_idx_next,
            )?;
            h.set_narrow(None);
            Some(head)
        }
        (n, None) => {
            g.set_narrow(narrow_resident);
            let head = load_mtp_head(
                &mut g,
                &cfg,
                n,
                mode,
                #[cfg(feature = "cuda")]
                &mut moe_layer_idx_next,
            )?;
            g.set_narrow(None);
            Some(head)
        }
    };

    // **The load phase ends here, headroom included, and it cannot end later.**
    //
    // Every dense tensor is placed; the block's right edge is final. It has to
    // close before `build_experts` — the expert cache sizes its zone from the
    // span's free ground, which is not knowable while the block can still grow.
    // The load headroom stays outside the span for good, so it still covers
    // `build_layers`, which repacks a whole layer at a time through
    // `WeightResidency::Pool`.
    dense_span::close_load(device)?;

    let experts = build_experts(content, &cfg)?;
    let (n_used, norm_topk) = cfg
        .moe
        .map_or((0, false), |m| (m.n_experts_used, m.norm_topk_prob));
    // The layer store, built at the same point and for the same reason as the
    // expert cache: its zone is carved from the span's *measured* remainder,
    // which only means anything now that the dense block is frozen.
    let layers = match build_layers {
        Some(build) => build(content, &cfg, residues)?,
        None => LayerStore::Resident(
            pending_layers
                .into_iter()
                .map(|l| {
                    l.resolve(experts.as_ref(), n_used, norm_topk)
                        .map(std::sync::Arc::new)
                })
                .collect::<Result<Vec<_>>>()?,
        ),
    };
    // The head resolves against the same cache, after it — see `PendingMtp`.
    let mtp = mtp
        .map(|m| m.resolve(experts.as_ref(), n_used, norm_topk))
        .transpose()?;

    // Every read is done — the trunk, the head, the expert cache and the layer store. An
    // override nothing took is a tensor that reached the model from the checkpoint's own copy.
    if let Some(o) = overrides {
        o.ensure_all_read()?;
    }

    Ok(QuantModel {
        cfg,
        embed,
        layers,
        mtp,
        #[cfg(feature = "cuda")]
        experts,
        final_norm,
        lm_head,
        device: device.clone(),
        dense_bytes: g.device_bytes,
    })
}

mod layer;
mod layer_load;
mod load_inputs;
mod loader;
mod narrowing;
mod recurrent;
#[cfg(test)]
mod tests_headroom;
#[cfg(test)]
mod tests_narrowing;
#[cfg(test)]
mod tests_recurrent_precision;

pub use layer::*;
pub use layer_load::*;
pub use load_inputs::*;
pub use loader::*;
pub(crate) use narrowing::*;
pub use recurrent::*;
