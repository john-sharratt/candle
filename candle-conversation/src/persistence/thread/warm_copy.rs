//! The copy of hot turns into warm host copies, and the policy a conversation's
//! turns are quantized under.
//!
//! `effective_turn_policy` resolves the policy from the engine's base and the
//! conversation's override. An override can change the compression level or pin
//! K and V to fixed formats, but it never introduces quantization on an engine
//! that has none. `copy_group_hot_to_warm` makes one batched device-to-host copy
//! for every layer of a group, and the copy changes nothing on the device because
//! the hot copies are already in their stored formats. A refused warm arena is a
//! normal deferral: the group stays hot and is copied on a later pass. Any other
//! failure is a fault and is reported.

use super::*;

/// Resolve the effective policy a conversation's turns are quantized under —
/// what the gap quantizes them with (`scheduler/turn_seal.rs`), and what tells
/// this thread whether a native turn is still the gap's to finish.
///
/// `base` is the engine-wide turn policy. `cc` is the conversation's
/// per-conversation override (from [`ConvCompression`]): when present it may
/// replace the compression level, drop the global K-format override so K is
/// adaptively quantized like V (`disable_k_override`), and/or pin K and V to
/// fixed uniform formats (`force_k` / `force_v`), bypassing adaptive selection.
/// Returns `None` (no quantization) only when the engine itself has no
/// policy — a per-conversation override never *introduces* quantization on
/// an otherwise-float engine.
pub(crate) fn effective_turn_policy(
    base: Option<&CompressionPolicy>,
    cc: Option<ConvCompression>,
) -> Option<CompressionPolicy> {
    match (base, cc) {
        // Lossless capture: no quantize, so turns keep native R16/F16.
        (_, Some(c)) if c.lossless => None,
        (Some(b), Some(c)) => {
            let mut p = *b;
            if let Some(level) = c.level {
                p.compression_level = level;
            }
            if c.disable_k_override {
                p.override_k_quant = None;
            }
            // Forced formats win last: they pin the uniform dst format and
            // re-enable the K override path even when `disable_k_override` set it.
            if let Some(k) = c.force_k {
                p.override_k_quant = Some(k);
            }
            if let Some(v) = c.force_v {
                p.override_v_quant = Some(v);
            }
            Some(p)
        }
        (Some(b), None) => Some(*b),
        (None, _) => None,
    }
}

/// Whether a hot→warm failure is the wave-in-flight deferral rather than a fault.
///
/// A copy allocates its CPU warm copies as it goes, and some of those need an
/// arena that does not exist yet. Creating one moves the arena frontier a
/// running wave's transient tier was placed against, which is why
/// `enter_arena_window` refuses it.
///
/// The refusal is **normal operation**, not an error: the group stays hot
/// and consistent and the next pass finds the gap between forwards. It reaches
/// the same `break 'work false` a genuine fault does, so the only thing that has
/// to differ is what gets said about it — a warning with a CUDA breadcrumb reads
/// as a fault, and this is the engine doing what it was built to do.
///
/// **Both refusals a forward causes.** One is the arena window, refused while
/// a forward is open. The other is a claim refused because the free regions sit
/// above the tier the last forward placed, which stands until the scheduler's
/// next gap takes it down — the same wait, said by the region pool instead.
/// Read as a fault, it was logged as one fifty times in ten minutes, and as a
/// device out-of-memory it would have told the governor the card was starving.
pub(super) fn is_wave_deferral(err: &candle::Error) -> bool {
    err.to_string()
        .contains(candle_nn::kv_cache::KV_ARENA_MID_WAVE)
        || is_tier_standing_over_free(err)
}

/// Install one copied group's `(idx, new_warm)` pairs on their residences,
/// each only while the residence still holds the hot content in `copied` it
/// was made from. Returns the warm bytes and the count installed.
pub(super) fn install_hot_to_warm(
    conversation: &Conversation,
    copied: &HashMap<ResidenceIndex, Arc<HotKv>>,
    installs: Vec<(ResidenceIndex, Vec<SealedSequence>)>,
) -> (u64, usize) {
    if installs.is_empty() {
        return (0, 0);
    }
    let mut total_bytes = 0u64;
    let mut count = 0usize;
    let view = conversation.read();
    for (idx, warm) in installs {
        let Some(from) = copied.get(&idx) else {
            continue;
        };
        let bytes: u64 = warm
            .iter()
            .flat_map(|s| s.chunks.iter())
            .map(|c| c.byte_size)
            .sum();
        if !view.install_copied(idx, from, warm) {
            tracing::debug!(
                target: "candle_conversation::persistence::tier",
                residence = idx.0,
                "hot → warm dropped: the residence moved on during the copy"
            );
            continue;
        }
        tracing::trace!(
            target: "candle_conversation::persistence::tier",
            residence = idx.0,
            bytes,
            "cached hot → warm"
        );
        total_bytes = total_bytes.saturating_add(bytes);
        count += 1;
    }
    (total_bytes, count)
}

/// Copy one group's hot residences to the warm tier: one format-preserving
/// DtoH of every layer's hot sequences into CPU warm copies. The hot copies
/// are already in their stored formats — a turn's seal quantized it in the
/// gap (`docs/kv_quantize_in_the_gap.md` §5) — so the copy changes nothing on
/// the device. Each residence whose every layer copied is appended to
/// `installs` as `(idx, new_warm)`.
///
/// Returns whether a forward turned the group away — a warm arena it needed
/// could not be created mid-wave — rather than anything failing. The group is
/// intact and hot; the caller runs it again once the next gap has made those
/// arenas.
///
/// **The copy writes nothing a compaction moves**, so the flight settles
/// before it: the group's chunks are in `held`, which a pass steps around.
pub(super) fn copy_group_hot_to_warm(
    backings: &[ChunkedKvBacking],
    device: &Device,
    copy_stream: &Arc<CudaStream>,
    group: &[(ResidenceIndex, Vec<SealedSequence>)],
    flight: &mut MigrateFlight,
    pinned_scratch: &mut Option<PinnedBuf>,
    n_layers: usize,
    installs: &mut Vec<(ResidenceIndex, Vec<SealedSequence>)>,
    copy_ms: &mut u64,
) -> bool {
    if group.is_empty() {
        return false;
    }
    flight.settle();
    let mut warm_per: Vec<Vec<SealedSequence>> = (0..group.len())
        .map(|_| Vec::with_capacity(n_layers))
        .collect();
    let mut wave_deferred = false;
    let success = 'work: {
        // ONE batched hot → warm DtoH across every layer: a single gather, DtoH
        // and sync for all layers.
        let t_c = std::time::Instant::now();
        let per_layer_refs: Vec<Vec<&SealedSequence>> = (0..n_layers)
            .map(|layer| group.iter().map(|(_, hot)| &hot[layer]).collect())
            .collect();
        let backing_refs: Vec<&ChunkedKvBacking> = backings.iter().collect();
        let warm_result = ChunkedKvBacking::migrate_sealed_layers_to_cpu_batch(
            &backing_refs,
            device,
            copy_stream,
            pinned_scratch,
            &per_layer_refs,
        );
        *copy_ms += t_c.elapsed().as_millis() as u64;
        drop(per_layer_refs);
        let warm_per_layer = match warm_result {
            Ok(v) => v,
            Err(e) if is_wave_deferral(&e) => {
                tracing::debug!(
                    "cache: hot→warm DtoH deferred — a wave holds the partition and the CPU \
                     warm copy needs a new arena; retrying at the next gap"
                );
                wave_deferred = true;
                break 'work false;
            }
            // **A failure here is a fault, not a batch too big for the staging
            // span.** The batched copy stages in band-sized groups that each fit the
            // span (`staging_groups` over every layer's bands), so no hot set — not
            // a 128K turn's 84 MB layer, not a burst of lossless-R16 turns — can
            // exceed it. What is left is a CUDA or host-allocation error: reported,
            // and the group stays hot and consistent for the next pass.
            Err(e) => {
                tracing::warn!(
                    "cache: hot→warm batched DtoH failed: {e} — the group stays hot and \
                     consistent and is retried next pass (last CUDA kernel on this \
                     thread: {})",
                    candle::last_cuda_kernel_launch()
                );
                signal_vram_starvation(device, &e);
                break 'work false;
            }
        };
        // Distribute warm (per layer, in layer order) into per-residence order.
        for (layer, layer_warm) in warm_per_layer.into_iter().enumerate() {
            if layer_warm.len() != group.len() {
                tracing::warn!(
                    "cache: hot→warm layer {layer} mismatch — {} hot vs {} warm",
                    group.len(),
                    layer_warm.len()
                );
                break 'work false;
            }
            for (i, w) in layer_warm.into_iter().enumerate() {
                warm_per[i].push(w);
            }
        }
        true
    };
    if success {
        for ((idx, _), warm) in group.iter().zip(warm_per) {
            if warm.len() == n_layers {
                installs.push((*idx, warm));
            }
        }
    }
    wave_deferred
}

/// Whether a hot copy still holds a native chunk with tokens in it — a turn
/// the gap has yet to quantize.
pub(super) fn holds_native(hot: &HotKv) -> bool {
    hot.iter()
        .any(|s| s.chunks.iter().any(|c| c.token_count > 0 && c.is_native()))
}
