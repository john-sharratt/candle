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
//! The schedule is keyed on the hub's segment ordinal, so it restarts with every
//! wave — the draft walk's steps, the verify — and a segment some other host
//! interaction cut short counts from its own start. Cuts made before a wave's
//! first MoE invocation advance the ordinal too, so that wave's first MoE
//! segment starts further up the ramp: the GPU is already running the work
//! those cuts launched, which is what the ramp's short first segments are for.

/// The most invocations a segment holds. The dispatch keeps this at half its
/// summary ring, so the ring hold — which waits for the readers of the ticket a
/// ring's length back — never waits on a bucketize that has not been launched
/// to write it.
pub(crate) const MAX_UNFLUSHED_INVOCATIONS: usize = 32;

/// MoE invocations the segment at `ordinal` within its wave holds before it is
/// launched: 1, 1, 2, 4, 8, 16, 32, 32, ….
pub(crate) fn segment_invocations(ordinal: usize) -> usize {
    if ordinal <= 1 {
        1
    } else {
        1usize
            .checked_shl((ordinal - 1) as u32)
            .unwrap_or(usize::MAX)
            .min(MAX_UNFLUSHED_INVOCATIONS)
    }
}

/// The invocations recorded into the current segment.
#[derive(Debug, Default)]
pub(crate) struct SegmentFill {
    /// The segment counted, as `(wave, ordinal)` — `None` before the first.
    segment: Option<(u64, usize)>,
    count: usize,
}

impl SegmentFill {
    /// Count one invocation recorded into segment `ordinal` of wave `wave`, and
    /// answer whether it ends the segment.
    pub(crate) fn record(&mut self, wave: u64, ordinal: usize) -> bool {
        if self.segment != Some((wave, ordinal)) {
            self.segment = Some((wave, ordinal));
            self.count = 0;
        }
        self.count += 1;
        self.count >= segment_invocations(ordinal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segments_double_to_the_cap() {
        let sizes: Vec<usize> = (0..9).map(segment_invocations).collect();
        assert_eq!(sizes, vec![1, 1, 2, 4, 8, 16, 32, 32, 32]);
        assert_eq!(segment_invocations(200), 32);
    }

    /// A 48-layer pass recorded from segment 0, the ordinal advancing each time
    /// a flush launches the segment: the flushes land after invocations 0, 1,
    /// 3, 7, 15 and 31, and the last 16 are left to the wave's finish.
    #[test]
    fn a_trunk_pass_flushes_at_each_doubling() {
        let mut fill = SegmentFill::default();
        let mut ordinal = 0;
        let mut flushed = Vec::new();
        for i in 0..48 {
            if fill.record(1, ordinal) {
                flushed.push(i);
                ordinal += 1;
            }
        }
        assert_eq!(flushed, vec![0, 1, 3, 7, 15, 31]);
    }

    /// A new wave starts over at one invocation, whatever the last one reached;
    /// and a segment some other cut began counts from its own start.
    #[test]
    fn the_count_restarts_with_the_wave_and_the_segment() {
        let mut fill = SegmentFill::default();
        assert!(!fill.record(1, 4));
        assert!(!fill.record(1, 4));
        // The wave closes mid-segment; the next wave's first invocation flushes.
        assert!(fill.record(2, 0));
        // Segment 3 holds four; an eager cut moves recording to segment 4 after
        // two of them, which then holds eight from its own start.
        assert!(!fill.record(2, 3));
        assert!(!fill.record(2, 3));
        for _ in 0..7 {
            assert!(!fill.record(2, 4));
        }
        assert!(fill.record(2, 4));
    }
}
