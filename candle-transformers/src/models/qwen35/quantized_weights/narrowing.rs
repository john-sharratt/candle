//! The quantization schedule a streamed trunk is loaded under.
//!
//! A streamed trunk's in-projections and FFN down-projection step down one width,
//! because their KO twin is what crosses PCIe on every forward. The recurrent
//! gates and the output projection keep their checkpoint precision, since their
//! error compounds and nothing downstream corrects it, and the first and last
//! blocks keep their carve-outs. The schedule applies only when the checkpoint
//! does not fit with room to serve from.

use super::*;

/// Trunk blocks held at their checkpoint width at each end of the stack.
///
/// The first and last layers are where a quantization schedule's own carve-outs live — this
/// checkpoint puts `ffn_down` at Q5_K on blocks 0-3 and nowhere else — and they are where the
/// literature puts the outlier ("super") weights. Five covers at least four DeltaNet layers at
/// each end under the lineage's 3:1 interleave.
pub(super) const NARROW_STREAM_ENDS: usize = 5;

/// The twin a **streamed trunk tensor** takes when the card cannot hold the model.
///
/// `None` leaves the mode's own choice, which is every tensor not named here and every load
/// that is not narrowing. Applied through [`Loader::twin_for`], which only ever *narrows*: a
/// target wider than what the mode picked is ignored, so naming one width per role cannot
/// accidentally inflate a block the checkpoint deliberately kept small.
///
/// # Why these three and not the rest
///
/// The bytes that matter are the KO twin's, not the source quant's — the twin is what crosses
/// PCIe on every forward. Sorted by what that yields on the 27B, the trunk's tensors above
/// `Q3_KO` are `ffn_down` (638 MiB), `attn_qkv` (300), `ssm_out` (450), `attn_q` (60). Two are
/// deliberately absent:
///
/// * **`ssm_out` is not narrowed at all.** It is the DeltaNet output projection — the most
///   sensitive tensor in this architecture, and unrescuable, because it has no preceding norm
///   for a calibrated repack to correct against. The checkpoint's own schedule agrees only
///   partly (front and back at Q8_0, the middle already at Q3_K), which is itself worth
///   knowing: the middle third is already the model's weakest point and this must not deepen
///   it.
/// * **`attn_output` likewise** — an out-projection, same argument, and only 60 MiB.
///
/// What is left cuts **in**-projections and the FFN's down-projection, and leaves the recurrent
/// path's own weights (`ssm_*`) at their checkpoint precision entirely.
pub(crate) fn streaming_twin(name: &str, num_layers: usize) -> Option<GgmlDType> {
    let rest = name.strip_prefix("blk.")?;
    let (idx, role) = rest.split_once('.')?;
    let li: usize = idx.parse().ok()?;
    // The draft block sits at `num_layers` and is resident, not streamed — `narrow_resident`
    // governs it, and reaching it here would apply a schedule meant for the trunk.
    if li >= num_layers {
        return None;
    }
    let interior = li >= NARROW_STREAM_ENDS && li + NARROW_STREAM_ENDS < num_layers;
    match role {
        // Blocks 0-3 carry the checkpoint's Q5_K carve-out; leave it standing.
        "ffn_down.weight" => (li >= 4).then_some(GgmlDType::Q3_KO),
        // One step (Q5_KO → Q4_KO), interior only. In-projections are norm-preceded, which is
        // what makes them the defensible half of the attention weights to cut.
        "attn_qkv.weight" => interior.then_some(GgmlDType::Q4_KO),
        // One step (Q6_KO → Q5_KO). The blocks holding Q6_K are llama.cpp's positional
        // `use_more_bits` bump, not a sensitivity finding — `attn_v` shows the identical
        // pattern — so a single step off it is cheap.
        "attn_q.weight" => Some(GgmlDType::Q5_KO),
        _ => None,
    }
}

/// The twin the two permanently-resident weights take when the card cannot hold the model.
///
/// `Q3_KO` — the narrowest twin with a real 3-bit grid. `Q2_KO` exists and is narrower still,
/// but its four levels floor at `rel_l2 ≈ 0.325`, which is a different thing from a repack.
pub(super) const NARROW_TWIN: GgmlDType = GgmlDType::Q3_KO;

/// Whether this load should narrow the output head and the draft block to [`NARROW_TWIN`].
///
/// **Both halves of the condition are load-bearing, and neither is a preference.**
///
/// *Dense*, because a routed checkpoint's weight is in its experts, which `expert_lre` already
/// streams — its resident set is small and narrowing the head buys it almost nothing while
/// costing the same precision. Only a dense model pays for residency in layer slots.
///
/// *Tight*, meaning the checkpoint does not fit in the card with room to serve from. When it
/// does fit, every layer is resident, no byte crosses PCIe after load, and narrowing would be
/// giving up precision for memory nobody needs. The comparison is against `checkpoint_bytes` —
/// everything the checkpoint's tensors weigh, which the model pack records at build time — plus
/// the KV side's opening reserve.
///
/// Returns `None` when either half fails, and `None` is the whole of the "off" path — there is
/// no second code path here, only a twin that is or is not overridden.
pub(crate) fn narrow_resident_twin(
    device: &Device,
    cfg: &Qwen35Config,
    checkpoint_bytes: u64,
) -> Option<GgmlDType> {
    if cfg.moe.is_some() {
        return None;
    }
    let Device::Cuda(_) = device else {
        return None;
    };
    let total = candle::quantized::get_vram_info().ok()?.1;
    // "Tight" is: the weights plus the KV side's opening reserve do not both fit. A model that
    // clears this has slots for every layer and nothing to gain here.
    let needed = checkpoint_bytes as usize + candle_nn::kv_cache::INITIAL_KV_RESERVE;
    (needed > total).then_some(NARROW_TWIN)
}
