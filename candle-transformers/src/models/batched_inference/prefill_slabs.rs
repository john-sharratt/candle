//! Packs pure-prefill sequences into token-bounded slabs, each as wide as
//! compute saturation allows and no wider.
//!
//! The packer closes a slab at the width cap, and the slack cap lets only the
//! final slab run up to a quarter wider. The slack is there because a wave's
//! fixed sweep cost is paid per slab whatever its width, while tokens past the
//! cap cost the same as the ones before. Packing is greedy over whole sequences,
//! which are never split, so a sequence longer than the cap travels alone.

/// Compute-side ceiling on the tokens one prefill forward carries.
///
/// This is a *throughput* limit, not a memory one: beyond roughly this width the
/// prefill kernels are compute-bound, so a wider forward costs the same per token
/// and slicing above it costs only the per-slab fixed wave overhead. A multiple
/// of 32, for kernel utilisation. (16384 was tried against the [1,4,8,16,1]
/// sweep and measured as a no-op there — that path's waves are shaped by the
/// scheduler's turn admission, not this slicer — so the value stays at the
/// established compute-saturation point until the pure-prefill ingest path is
/// measured wider.)
///
/// **It reserves nothing.** The transient tier is
/// `WAVE_ATTN_BYTES + WAVE_FFN_BYTES + MIGRATION_STAGING_CAP_BYTES`, carved once
/// at first use whatever this value is. Raising it does not cost VRAM; it permits
/// a wider wave, and whether that wave *fits* is the separate question
/// [`ManagedBatchedModel::prefill_width_cap`] asks the wave plan (the KV-admission
/// cap and the activation-pool cushion still bound the real wave). The narrower
/// of the two wins, so this can lead the span rather than having to trail it.
pub(crate) const MAX_PREFILL_TOKENS: usize = 8192;

/// The width cap plus 25% slack — the ceiling a single prefill slab may
/// actually reach. The asymmetry is measured: a wave's FIXED cost (~2.37 s —
/// the full sweep of every layer's launches and routed-expert fetches) is paid per slab
/// regardless of width, while the compute-saturation cap is soft — tokens 25%
/// past it cost the same ~0.87 ms each as the ones before. So a 128-token
/// straggler slab after an 8192 slab spends a whole fixed sweep on 1.5% of the
/// tokens (~25% extra wall), where absorbing it into one 8320-token wave costs
/// per-token rate only.
pub fn prefill_slack_cap(width_cap: usize) -> usize {
    width_cap + width_cap / 4
}

/// Pack pure-prefill sequences into token-bounded slabs, returned as
/// `start..end` index ranges over `lens`.
///
/// Greedy whole-sequence packing against `width_cap`, with the
/// [`prefill_slack_cap`] tail rule: at the point a slab would close, if
/// EVERYTHING still unpacked fits within the slack ceiling it is absorbed into
/// this final slab instead of becoming one or more remainder waves. Only the
/// final slab may overshoot, so ordinary slabs still respect the cap; a single
/// sequence longer than the cap always travels alone and uncut (this packer
/// never splits inside a sequence).
pub(crate) fn pack_prefill_slabs(lens: &[usize], width_cap: usize) -> Vec<(usize, usize)> {
    let slack_cap = prefill_slack_cap(width_cap);
    let mut slabs = Vec::new();
    let mut start = 0usize;
    while start < lens.len() {
        let mut toks = 0usize;
        let mut end = start;
        while end < lens.len() {
            let l = lens[end];
            if end > start && toks + l > width_cap {
                let tail: usize = lens[end..].iter().sum();
                if toks + tail <= slack_cap {
                    end = lens.len();
                }
                break;
            }
            toks += l;
            end += 1;
        }
        slabs.push((start, end));
        start = end;
    }
    slabs
}
