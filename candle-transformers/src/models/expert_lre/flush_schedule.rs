//! Which MoE invocations end the graph segment they are recorded into.
//!
//! Every invocation's host protocol polls the device for its bucketize's
//! summary, so the launches must reach the device; inside a wave capture that
//! means launching the recording segment. It does not mean launching it at
//! every invocation.
//!
//! **Why segments grow, not one per layer.** Each boundary is a graph launch the
//! device starts only behind the one before it, plus host driver calls; at one
//! per MoE layer that is ~50 a decode step. The boundaries exist so the GPU is
//! not left idle while the host records ahead of it (`docs/decode_graphs.md`
//! §3.3) — which matters only at the start of a wave. The forward thread
//! records a Flash-Next trunk layer in roughly a third of the time the GPU takes
//! to execute it, so once the first two one-invocation segments are running the
//! host's lead grows with every layer, and each segment may be twice the one
//! before it without the GPU waiting on its recording: segments of 1, 1, 2, 4,
//! 8, 16, then [`MAX_UNFLUSHED_INVOCATIONS`] invocations, the wave's own finish
//! launching the tail.
//!
//! **Keyed on the invocation's place in its wave, not on the segment.** The
//! flushes land after the same MoE layers in every wave — 1, 2, 4, 8, 16, 32,
//! then every 32nd — whatever else cut the wave, and whether its segments run
//! recorded or eagerly. Keyed on the hub's segment ordinal, an eager cut
//! anywhere earlier in a wave moved every flush after it, and a segment running
//! eagerly (where the dispatch flushed at every invocation) cut the wave at
//! every MoE layer: the same layers fell into different segments from one wave
//! to the next, each a topology of its own whose graph was never reused — on
//! `.5` (2026-10-17) half of 4,915 new topologies began with a different kernel
//! from the executable they were compared against.

/// The most invocations a segment holds. The dispatch keeps this at half its
/// summary ring, so the ring hold — which waits for the readers of the ticket a
/// ring's length back — never waits on a bucketize that has not been launched
/// to write it.
pub(crate) const MAX_UNFLUSHED_INVOCATIONS: usize = 32;

/// Whether the wave's `n`th MoE invocation (counting from one) ends its
/// segment: the 1st, 2nd, 4th, 8th, 16th and 32nd, then every 32nd — segments
/// of 1, 1, 2, 4, 8, 16, then [`MAX_UNFLUSHED_INVOCATIONS`] invocations.
pub(crate) fn ends_segment(n: usize) -> bool {
    n > 0
        && if n <= MAX_UNFLUSHED_INVOCATIONS {
            n.is_power_of_two()
        } else {
            n.is_multiple_of(MAX_UNFLUSHED_INVOCATIONS)
        }
}

/// The MoE invocations issued so far in the current wave.
#[derive(Debug, Default)]
pub(crate) struct SegmentFill {
    /// The wave counted — `None` before the first.
    wave: Option<u64>,
    count: usize,
}

impl SegmentFill {
    /// Count one invocation issued in wave `wave`, and answer whether it ends
    /// the segment it is in.
    pub(crate) fn record(&mut self, wave: u64) -> bool {
        if self.wave != Some(wave) {
            self.wave = Some(wave);
            self.count = 0;
        }
        self.count += 1;
        ends_segment(self.count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Segments double to the cap, then hold the cap: the 1st, 2nd, 4th, 8th,
    /// 16th and 32nd invocations end one, then every 32nd.
    #[test]
    fn segments_double_to_the_cap() {
        let ends: Vec<usize> = (1..=128).filter(|&n| ends_segment(n)).collect();
        assert_eq!(ends, vec![1, 2, 4, 8, 16, 32, 64, 96, 128]);
        assert!(!ends_segment(0));
    }

    /// **The flushes land after the same layers in every wave**, however the
    /// wave is cut otherwise: a 48-layer pass flushes after invocations 0, 1,
    /// 3, 7, 15 and 31 (counting from zero), and the last 16 are left to the
    /// wave's finish — in this wave and the next.
    #[test]
    fn every_wave_flushes_after_the_same_layers() {
        let mut fill = SegmentFill::default();
        for wave in 1..=3 {
            let flushed: Vec<usize> = (0..48).filter(|_| fill.record(wave)).collect();
            assert_eq!(flushed, vec![0, 1, 3, 7, 15, 31], "wave {wave}");
        }
    }

    /// A new wave starts its count over, whatever the last one reached.
    #[test]
    fn the_count_restarts_with_the_wave() {
        let mut fill = SegmentFill::default();
        for _ in 0..5 {
            fill.record(1);
        }
        assert!(
            fill.record(2),
            "the next wave's first invocation ends a segment"
        );
        assert!(fill.record(2));
        assert!(!fill.record(2));
        assert!(fill.record(2));
    }
}
