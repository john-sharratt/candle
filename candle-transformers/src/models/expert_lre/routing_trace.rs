//! The routing the pipeline thread has served, kept for the residency
//! references and written out for offline replay.
//!
//! Every routed invocation's row and distinct experts, in order, from the
//! start of the previous reporting interval: the previous interval warms the
//! replay (the zone was warm when the current one was measured), and the
//! current one is what [`RoutingTrace::ceiling`] and [`RoutingTrace::lru`]
//! score (`belady`). The permanently resident layers are always hits; they are
//! counted, not replayed.
//!
//! Each expert carries the tokens that routed to it and whether a decode row
//! did — what a policy weighs a routing by — so a written trace
//! ([`RoutingTrace::write`]) can be replayed under any policy offline.
//!
//! **The file**, little-endian: `u32 invocations`, `u32 mark` (the first
//! invocation of the current interval), `u32 pinned_requests`, `u32 capacity`
//! (the zone's evictable slots when it was written), then per
//! invocation `u16 row`, `u16 n`, and `n` × (`u16 expert`, `u16 tokens |
//! decode << 15`).

use super::belady::{belady, lru, Invocation};
use std::io::Write;
use std::path::Path;

/// Invocations kept at most; past it the oldest half is dropped. ~150 bytes
/// an invocation, so ~60 MB at the bound.
const MAX_INVOCATIONS: usize = 400_000;

/// The decode flag over a routed expert's token count, in the file.
const DECODE_BIT: u16 = 1 << 15;

pub(crate) struct RoutingTrace {
    invocations: Vec<Invocation>,
    /// Per invocation, per expert: tokens routed to it, with [`DECODE_BIT`]
    /// when a decode row was among them.
    weights: Vec<Vec<u16>>,
    /// Where the current interval starts.
    mark: usize,
    /// Requests of the permanently resident layers in the current interval.
    pinned_requests: usize,
}

impl RoutingTrace {
    pub(crate) fn new() -> Self {
        Self {
            invocations: Vec::new(),
            weights: Vec::new(),
            mark: 0,
            pinned_requests: 0,
        }
    }

    /// An invocation of `row` routed `experts` (distinct, each with its token
    /// count and whether a decode row routed it); `pinned` when the row's
    /// experts are permanently resident.
    pub(crate) fn push(&mut self, row: usize, experts: &[(usize, u32, bool)], pinned: bool) {
        if pinned {
            self.pinned_requests += experts.len();
            return;
        }
        if self.invocations.len() == MAX_INVOCATIONS {
            let drop = MAX_INVOCATIONS / 2;
            self.invocations.drain(..drop);
            self.weights.drain(..drop);
            self.mark = self.mark.saturating_sub(drop);
        }
        self.invocations.push((
            row as u16,
            experts.iter().map(|&(e, _, _)| e as u16).collect(),
        ));
        self.weights.push(
            experts
                .iter()
                .map(|&(_, tokens, decode)| {
                    tokens.min((DECODE_BIT - 1) as u32) as u16 | if decode { DECODE_BIT } else { 0 }
                })
                .collect(),
        );
    }

    /// A new interval starts: the one just ended becomes the warm-up.
    pub(crate) fn reset(&mut self) {
        self.invocations.drain(..self.mark);
        self.weights.drain(..self.mark);
        self.mark = self.invocations.len();
        self.pinned_requests = 0;
    }

    /// Belady's hit rate over the current interval, in percent, at
    /// `capacity` evictable slots — the permanently resident layers' requests
    /// counted as hits, as the measured rate counts them. `None` when the
    /// interval routed nothing.
    pub(crate) fn ceiling(&self, capacity: usize) -> Option<f64> {
        self.rate(belady(&self.invocations, capacity, self.mark))
    }

    /// LRU's hit rate on the same terms as [`Self::ceiling`] — the recency
    /// reference the zone's policy sits above or below.
    pub(crate) fn lru(&self, capacity: usize) -> Option<f64> {
        self.rate(lru(&self.invocations, capacity, self.mark))
    }

    fn rate(&self, (hits, misses): (usize, usize)) -> Option<f64> {
        let total = hits + misses + self.pinned_requests;
        (total > 0).then(|| 100.0 * (hits + self.pinned_requests) as f64 / total as f64)
    }

    /// The trace in the file format of the module docs, at `capacity`
    /// evictable slots.
    pub(crate) fn encode(&self, capacity: usize) -> Vec<u8> {
        let mut out = Vec::new();
        for v in [
            self.invocations.len(),
            self.mark,
            self.pinned_requests,
            capacity,
        ] {
            out.extend_from_slice(&(v as u32).to_le_bytes());
        }
        for ((row, experts), weights) in self.invocations.iter().zip(&self.weights) {
            out.extend_from_slice(&row.to_le_bytes());
            out.extend_from_slice(&(experts.len() as u16).to_le_bytes());
            for (e, w) in experts.iter().zip(weights) {
                out.extend_from_slice(&e.to_le_bytes());
                out.extend_from_slice(&w.to_le_bytes());
            }
        }
        out
    }

    /// Write the trace to `path` ([`Self::encode`]), creating its directory.
    pub(crate) fn write(&self, path: &Path, capacity: usize) -> std::io::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::File::create(path)?.write_all(&self.encode(capacity))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn routed(experts: &[usize]) -> Vec<(usize, u32, bool)> {
        experts.iter().map(|&e| (e, 1, true)).collect()
    }

    /// The current interval is scored with the previous one as warm-up, and
    /// pinned requests count as hits.
    #[test]
    fn the_ceiling_scores_the_current_interval_warmed_by_the_last() {
        let mut t = RoutingTrace::new();
        t.push(2, &routed(&[1, 2]), false);
        assert_eq!(t.ceiling(2), Some(0.0), "two cold misses");
        t.reset();
        t.push(2, &routed(&[1, 2]), false);
        t.push(0, &routed(&[4, 5]), true);
        // Both hits after the warm-up, plus two pinned hits: 4 of 4.
        assert_eq!(t.ceiling(2), Some(100.0));
        // One slot: one of the two misses again — 3 of 4.
        assert_eq!(t.ceiling(1), Some(75.0));
        // LRU at one slot: 2 evicted 1 in the warm-up, so 1 misses and then
        // evicts 2 — both miss, 2 pinned of 4.
        assert_eq!(t.lru(1), Some(50.0));
        assert_eq!(t.lru(2), Some(100.0));
        t.reset();
        assert_eq!(t.ceiling(2), None, "nothing routed since");
        t.push(2, &routed(&[1]), false);
        assert_eq!(t.ceiling(2), Some(100.0), "warmed by the interval before");
    }

    /// Reset drops everything before the interval that just ended.
    #[test]
    fn a_reset_keeps_one_interval_of_warm_up() {
        let mut t = RoutingTrace::new();
        t.push(2, &routed(&[1]), false);
        t.reset();
        t.push(3, &routed(&[1]), false);
        t.reset();
        t.push(2, &routed(&[1]), false);
        assert_eq!(
            t.ceiling(4),
            Some(0.0),
            "row 2's first use is two intervals back"
        );
    }

    /// The file: the header, then each invocation's row, count and experts
    /// with their token counts and decode bit — raw bytes.
    #[test]
    fn the_trace_encodes_to_its_file_format() {
        let mut t = RoutingTrace::new();
        t.push(3, &[(7, 2, true), (9, 1, false)], false);
        t.reset();
        t.push(0, &routed(&[1]), true);
        t.push(4, &[(258, 40_000, false)], false);
        assert_eq!(
            t.encode(4_100),
            vec![
                2, 0, 0, 0, // invocations
                1, 0, 0, 0, // mark
                1, 0, 0, 0, // pinned requests
                0x04, 0x10, 0, 0, // capacity
                3, 0, 2, 0, // row 3, two experts
                7, 0, 2, 0x80, // expert 7: 2 tokens, decode
                9, 0, 1, 0, // expert 9: 1 token
                4, 0, 1, 0, // row 4, one expert
                2, 1, 0xff, 0x7f, // expert 258: the count saturates below the decode bit
            ]
        );
    }
}
