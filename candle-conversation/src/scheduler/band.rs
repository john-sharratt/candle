//! A conversation's priority band, as the scheduler ranks it.
//!
//! A conversation's band is its target layer's `decode_priority`: the dialogue
//! is `High`, the ingest layers `Low`. Bands share the device rather than take
//! turns with it (`docs/continuous_fair_waves.md` §3): every band's decode rows
//! ride every forward, and a band's prefill creeps at the airtime ratio of the
//! decodes that outrank it — and a `High` prefill, beside a `High` decode, is
//! paced in tokens. Nothing is held out of a forward for its band.
//!
//! Tenants are equal: the band alone ranks, whichever tenant the work belongs
//! to (`docs/shared_model_host.md` §3.6).

use crate::projection::DecodePriority;

/// A band's rank: higher outranks lower.
pub(super) fn rank(p: DecodePriority) -> usize {
    match p {
        DecodePriority::Low => 0,
        DecodePriority::Normal => 1,
        DecodePriority::High => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use DecodePriority::{High, Low, Normal};

    #[test]
    fn the_bands_rank_low_normal_high() {
        assert!(rank(Low) < rank(Normal));
        assert!(rank(Normal) < rank(High));
    }
}
