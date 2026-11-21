//! Checking the recurrent gates of a checkpoint, and repairing them from a donor
//! when they are stored too coarsely.
//!
//! The gates `ssm_alpha` and `ssm_beta` produce the per-step gates of the
//! recurrence, and their error compounds along the sequence. A checkpoint storing
//! them below eight bits is refused unless a donor supplies the same tensors. A
//! repair takes the whole recurrent path from the donor, for the tensors whose
//! format the checkpoint changed, so the recurrence comes from one source. Each
//! donor tensor must exist at eight bits or better and at the same shape, and a
//! failure names the tensor.

use super::*;

/// Load the production model from a GGUF onto `device`.
///
/// `mode` selects the numeric path for every projection uniformly: an int8
/// mode makes each one a KO twin so the q8a128 activations the fused producers
/// emit always meet a KO weight, and [`Int8Mode::Off`] keeps the standard
/// dequant-weight path.
///
/// # Why the cache arrives as a callback
///
/// The expert cache's capacity is `(span − reserve) / slot_bytes`, and the span
/// is *measured*, not declared (`docs/archived/elastic_vram_partition.md` §4). That
/// measurement is only meaningful once every dense tensor is resident: taken
/// before, it reads the model's own weights as free ground and hands the expert
/// side room the KV side is about to need. Qwen3-MoE hit exactly this and its
/// loader carries the fixed ordering as a comment.
///
/// So the ordering is *structural* here rather than a caller's obligation: this
/// function loads every dense tensor, then calls `build_experts` — at the one
/// point where the span means what it says — then grafts the cache onto the
/// layers that route. A dense checkpoint passes a closure returning `Ok(None)`.
///
/// # `build_layers`, and why it is an `Option` rather than a returned `Option`
///
/// The layer cache has the same ordering constraint for the same reason — its
/// zone is carved from the span's measured remainder — but it also decides
/// something the expert cache does not: **whether the trunk loop runs at all**.
/// A streamed checkpoint must not load its layers into the dense block, because
/// not loading them there is the entire point. That decision has to be made
/// before the loop, and a closure whose answer arrives after it cannot make it.
///
/// So `Some(f)` *is* the decision: the trunk loop is skipped, the per-layer
/// residues are read in its place (a few hundred KB against ~240 MB a layer),
/// and `f` builds the store once the span is closed. `None` loads every layer
/// resident.
///
/// One case overrides it. A **routed** checkpoint keeps its layers resident
/// whatever the caller asked, because its weight is in the experts — which
/// `expert_lre` already streams — and its per-layer projections are a rounding
/// error beside them. That is not the fits/does-not-fit branch
/// `docs/archived/qwen38_layer_streaming.md` §7 forbids: every *dense* checkpoint
/// streams, and "it fits" is the degenerate case where nothing is ever evicted.
/// The DeltaNet gate projections, which a conversion may not quantize.
///
/// `ssm_alpha` and `ssm_beta` produce the per-step gates of the recurrence. Their error does
/// not stay where it is made: the state carries it to the next token and the next, so an
/// error small enough to be invisible in a feed-forward weight compounds along the sequence.
/// This is the same reasoning that keeps `ssm_a`, the `dt` bias and the per-head norm gain in
/// F32 (module docs above, design doc §8), and the checkpoints themselves declare it —
/// `mamba_ssm_dtype: float32`.
pub(super) const RECURRENT_GATES: [&str; 2] = ["ssm_alpha.weight", "ssm_beta.weight"];

/// The whole recurrent path: the gates above plus the output projection they feed.
///
/// **What a repair restores, which is wider than what it is diagnosed by.** The F32 rule is
/// specific to the gates, and `ssm_out` breaks no rule by being quantized — the reference
/// conversion stores it at `Q8_0`. But the three are one subsystem, and a recurrence assembled
/// half from the base and half from a fine-tune that stores its half differently is a
/// combination nobody has measured. When a donor is used at all, it supplies every tensor on
/// this path whose format the primary changed, so the recurrence comes from one source.
pub(crate) const RECURRENT_PATH: [&str; 3] =
    ["ssm_alpha.weight", "ssm_beta.weight", "ssm_out.weight"];

/// Refuse a checkpoint whose recurrent gates were quantized.
///
/// # Why this is a load failure and not a warning
///
/// The model loads, reports nothing, and generates word salad or degenerate repetition from
/// its first token — with weights that measure 0.999 cosine against a checkpoint that works,
/// because the defect is a ~1–2% error in the one place the architecture cannot absorb one.
/// Everything about the failure points somewhere else: the tokenizer is identical, the token
/// table is identical, the prompt is identical, and the batched gates pass. Three third-party
/// Qwen3.5 conversions were each debugged for hours from that starting point.
///
/// llama.cpp's own quantizer excludes these tensors, so a checkpoint built with it is fine and
/// never reaches this message. A conversion with custom quant rules that does not know this
/// architecture will quantize them like any other 2-D weight, and the file gives no other sign.
///
/// The message names the tensor and the format because the remedy is a different conversion,
/// not a different setting — nothing at run time can restore precision the file does not carry.
pub(super) fn check_recurrent_precision(content: &gguf_file::Content) -> Result<()> {
    let bad = undersized_gates(content);
    if bad.is_empty() {
        return Ok(());
    }
    let (first, dtype) = &bad[0];
    candle::bail!(
        "this checkpoint stores the DeltaNet recurrent gates below 8-bit precision — {} of them, \
         starting with `{first}` at {dtype:?}. The recurrence accumulates their error at \
         every token, so the model loads cleanly and then generates incoherent text from \
         its first token, with no other symptom. The precision is not recoverable from this \
         file: use a conversion that keeps `ssm_alpha`/`ssm_beta` at F32 or Q8_0 (llama.cpp's \
         own quantizer excludes them, and the checkpoint's `mamba_ssm_dtype` declares \
         float32), or supply the base checkpoint as a gate donor so its gates are read from \
         it instead.",
        bad.len(),
    )
}

/// Every recurrent gate this checkpoint stores below F32, sorted by name.
///
/// Sorted so the tensor a message names is stable across runs — `tensor_infos` is a hash map
/// and would otherwise offer a different one each time.
pub fn undersized_gates(content: &gguf_file::Content) -> Vec<(String, GgmlDType)> {
    let mut bad: Vec<(String, GgmlDType)> = content
        .tensor_infos
        .iter()
        .filter(|(name, _)| RECURRENT_GATES.iter().any(|g| name.ends_with(g)))
        .filter(|(_, info)| !gate_precision_suffices(info.ggml_dtype))
        .map(|(name, info)| (name.clone(), info.ggml_dtype))
        .collect();
    bad.sort_by(|a, b| a.0.cmp(&b.0));
    bad
}

/// Whether a format carries enough precision for a recurrent gate.
///
/// **Eight bits, not F32** — and the difference is a checkpoint that would otherwise be
/// refused for working perfectly. The rule was first written as "must be F32", generalising
/// from one working conversion (`QWEN35_9B`, which keeps them F32) and three broken ones
/// (`Q6_K` and `BF16`). The pinned **0.8B** is the fourth data point and it contradicts that:
/// it stores its gates at `Q8_0` and has been the lineage's proving ground for as long as the
/// gate has existed. A guard that refuses a model known to work is worse than no guard, and
/// this one did until `test_parallel_batched_forwarding_0_8b` said so.
///
/// So the line is drawn where the evidence puts it. `Q8_0`'s per-32 block scale gives a gate
/// weight eight mantissa bits about its own local magnitude, which is enough; `Q6_K` at 6.5
/// bits per weight is not, and the recurrence compounds the difference over every token.
/// Float formats clear it by construction.
pub(super) fn gate_precision_suffices(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::F32
            | GgmlDType::F16
            | GgmlDType::BF16
            | GgmlDType::Q8_0
            | GgmlDType::Q8_1
            | GgmlDType::Q8_K
    )
}

/// The recurrent-path tensors a donor will supply, checked but **not read**.
///
/// Names only. The tensors themselves are read lazily by [`Loader::donated`], once per loader
/// that asks for one — reading them here as well would put every gate on the card a second
/// time (~400 MiB across the trunk) purely to count them and drop them.
///
/// # What is checked, and why each one
///
/// Taking a weight from another file is only sound when it is the *same* weight, so every
/// gate is admitted on its own evidence:
///
/// * the donor has a tensor of that name — a differently-named architecture supplies nothing;
/// * at 8-bit or better — a donor with the same defect repairs nothing, and quietly accepting
///   its `Q6_K` would leave the model broken in precisely the way this exists to prevent;
/// * of the same shape — a donor for a different model size would otherwise load and be
///   wrong, and shape is what catches it before a single tensor is read.
///
/// A gate failing any of these fails the load, naming it. Partial repair is deliberately not
/// offered: a stack with some layers' gates restored and others still quantized is a model
/// nobody has measured, and it would degrade in a way that reads as ordinary bad prose
/// rather than as a fault.
pub(crate) fn donor_gates(
    content: &gguf_file::Content,
    donor: &gguf_file::Content,
) -> Result<Vec<String>> {
    let mut out = Vec::new();
    // **Repair only what is broken.** A checkpoint whose gates are F32 is correct as it
    // stands, and a donor differing from it anywhere on the recurrent path is then just a
    // different conversion — not something to import. Without this a healthy primary paired
    // with any donor would have its recurrence quietly replaced.
    if undersized_gates(content).is_empty() {
        return Ok(out);
    }
    // Every recurrent-path tensor whose format this checkpoint changed from the donor's —
    // which for a fine-tune and its base means every one its conversion quantized differently.
    // Sorted, so a failure names the same tensor on every run.
    let mut path: Vec<(&String, &gguf_file::TensorInfo)> = content
        .tensor_infos
        .iter()
        .filter(|(n, _)| RECURRENT_PATH.iter().any(|r| n.ends_with(r)))
        .filter(|(n, i)| {
            donor
                .tensor_infos
                .get(*n)
                .is_none_or(|d| d.ggml_dtype != i.ggml_dtype)
        })
        .collect();
    path.sort_by(|a, b| a.0.cmp(b.0));

    for (name, want) in path {
        let name = name.clone();
        let Some(have) = donor.tensor_infos.get(&name) else {
            candle::bail!(
                "the gate donor has no `{name}`, so it cannot supply this checkpoint's \
                 quantized recurrent gates"
            )
        };
        // The precision rule is the *gates'*, and only theirs. `ssm_out` is legitimately
        // quantized in the stock conversion (`Q8_0` in `QWEN35_9B`, `Q6_K` elsewhere), so applying
        // it to the output projection would refuse valid donors; what `ssm_out` must not be is
        // quantized the way the primary quantized it, which the filter above established.
        let is_gate = RECURRENT_GATES.iter().any(|g| name.ends_with(g));
        if is_gate && !gate_precision_suffices(have.ggml_dtype) {
            candle::bail!(
                "the gate donor stores `{name}` at {:?}, which is below 8-bit — it carries \
                 the same defect as the checkpoint it is meant to repair",
                have.ggml_dtype
            )
        }
        if have.shape.dims() != want.shape.dims() {
            candle::bail!(
                "the gate donor's `{name}` is {:?} where this checkpoint's is {:?} — these are \
                 different models, not a checkpoint and its base",
                have.shape.dims(),
                want.shape.dims()
            )
        }
        out.push(name);
    }
    Ok(out)
}
