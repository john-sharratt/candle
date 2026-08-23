//! Where a hybrid stack's KV actually lives.
//!
//! On a uniform transformer every layer owns a KV cache, so "layer index"
//! and "KV index" are the same number and nothing has to say which it means.
//! A 3:1 hybrid breaks that: the 9B has 32 transformer layers and only 8 of
//! them attend, so the two indices diverge by 4×.
//!
//! The engine already keeps the distinction — `session.num_layers()` is the
//! count of per-layer KV chunk sets (it is what sizes chunk headers and what
//! `recover_section_cold_refs` reads a sequence back with), while
//! `model.num_layers()` is transformer depth (it is what bounds
//! `forward_wave`'s layer range). This type is the map between them, so that
//! the difference is stated once instead of being re-derived, differently,
//! at each site that needs it.
//!
//! Allocating KV per *attention* layer rather than per transformer layer is
//! not only a 4× memory saving: admission prices a wave's KV by the
//! per-layer row cost times the layer count, so counting all 32 would refuse
//! four times more prefill than the cache can actually hold.

use super::types::LayerKind;

/// The transformer-layer ↔ KV-layer correspondence for one hybrid stack.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvLayerMap {
    /// Per transformer layer: `Some(kv_index)` for an attention layer,
    /// `None` for a DeltaNet layer (whose state is recurrent, not paged).
    to_kv: Vec<Option<usize>>,
    /// Per KV index: the transformer layer that owns it.
    to_layer: Vec<usize>,
}

impl KvLayerMap {
    /// Build the map from a layer schedule.
    pub fn new(kinds: &[LayerKind]) -> Self {
        let mut to_kv = Vec::with_capacity(kinds.len());
        let mut to_layer = Vec::new();
        for (li, kind) in kinds.iter().enumerate() {
            match kind {
                LayerKind::Attention => {
                    to_kv.push(Some(to_layer.len()));
                    to_layer.push(li);
                }
                LayerKind::DeltaNet => to_kv.push(None),
            }
        }
        Self { to_kv, to_layer }
    }

    /// Transformer depth.
    pub fn num_layers(&self) -> usize {
        self.to_kv.len()
    }

    /// How many KV backings this stack needs — the session's layer count.
    pub fn num_kv_layers(&self) -> usize {
        self.to_layer.len()
    }

    /// The KV backing index a transformer layer writes to, or `None` when the
    /// layer is recurrent and writes no KV at all.
    pub fn kv_index(&self, layer: usize) -> Option<usize> {
        self.to_kv.get(layer).copied().flatten()
    }

    /// The transformer layer that owns a KV index.
    pub fn layer_of_kv(&self, kv: usize) -> Option<usize> {
        self.to_layer.get(kv).copied()
    }

    /// Every attention layer, in order.
    pub fn attention_layers(&self) -> &[usize] {
        &self.to_layer
    }

    /// The KV-index half-open range covering the attention layers inside the
    /// transformer-layer window `[start, end)`.
    ///
    /// KV indices are handed out in layer order, so the attention layers of
    /// any contiguous layer window occupy a contiguous KV window — which is
    /// what lets a re-entrant wave admit and claim per layer range without
    /// enumerating layers. An empty window (a range containing no attention
    /// layers) returns an empty range rather than a reversed one.
    pub fn kv_range(&self, start: usize, end: usize) -> (usize, usize) {
        let count_before = |bound: usize| self.to_layer.partition_point(|&l| l < bound);
        let lo = count_before(start.min(end));
        let hi = count_before(end.max(start));
        (lo, hi.max(lo))
    }

    /// The attention layer at or below `layer`, falling back to the first one
    /// when `layer` sits below every attention layer.
    ///
    /// Provenance capture reads Q out of an attention layer, so the
    /// depth-fraction indices the scheduler asks for (roughly 15%, 50%, 85%
    /// of the stack) have to be moved to a layer that actually has a Q to
    /// capture. Snapping *down* keeps the requested depth an upper bound,
    /// which matters because the deepest index is derived as `n - 1` and must
    /// not walk off the end.
    pub fn snap_to_attention(&self, layer: usize) -> Option<usize> {
        if self.to_layer.is_empty() {
            return None;
        }
        match self.to_layer.binary_search(&layer) {
            Ok(_) => Some(layer),
            // `i` is where `layer` would insert, so `i - 1` is the last
            // attention layer below it; `i == 0` means there is none.
            Err(i) => Some(self.to_layer[i.saturating_sub(1)]),
        }
    }
}

// The schedules under test come from `qwen35::config`, which lives in the
// CUDA-gated hybrid lineage; the map itself is pure and builds either way.
#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    fn schedule(n: usize, interval: usize) -> Vec<LayerKind> {
        crate::models::qwen35::config::Qwen35Config::schedule_from_interval(n, interval)
    }

    #[test]
    fn maps_the_nine_b_schedule() {
        // 32 layers, attention on every 4th → 8 KV layers at 3,7,…,31.
        let m = KvLayerMap::new(&schedule(32, 4));
        assert_eq!(m.num_layers(), 32);
        assert_eq!(m.num_kv_layers(), 8);
        assert_eq!(
            m.attention_layers(),
            &[3, 7, 11, 15, 19, 23, 27, 31],
            "attention sits at (i+1) % 4 == 0"
        );
        assert_eq!(m.kv_index(3), Some(0));
        assert_eq!(m.kv_index(31), Some(7));
        assert_eq!(m.kv_index(0), None);
        assert_eq!(m.kv_index(30), None);
        // Round-trip every KV index.
        for kv in 0..m.num_kv_layers() {
            let li = m.layer_of_kv(kv).unwrap();
            assert_eq!(m.kv_index(li), Some(kv));
        }
        assert_eq!(m.layer_of_kv(8), None);
        assert_eq!(m.kv_index(32), None, "out of range is not a panic");
    }

    /// A re-entrant wave admits KV for a layer window; on a hybrid that has
    /// to become the KV window, or admit claims the wrong layers' chunks.
    #[test]
    fn kv_ranges_track_layer_windows() {
        let m = KvLayerMap::new(&schedule(32, 4)); // attention at 3,7,…,31
        assert_eq!(m.kv_range(0, 32), (0, 8), "the whole stack");
        assert_eq!(m.kv_range(0, 4), (0, 1), "layer 3 only");
        assert_eq!(m.kv_range(4, 8), (1, 2), "layer 7 only");
        assert_eq!(m.kv_range(0, 3), (0, 0), "no attention layer yet");
        assert_eq!(m.kv_range(0, 0), (0, 0), "empty window");
        assert_eq!(m.kv_range(28, 32), (7, 8), "the last block");
        assert_eq!(m.kv_range(32, 32), (8, 8), "past the end");
        // Windows tile: consecutive ranges must abut with no gap or overlap.
        let mut prev = 0;
        for w in 0..8 {
            let (lo, hi) = m.kv_range(w * 4, (w + 1) * 4);
            assert_eq!(lo, prev, "window {w} does not abut the previous");
            prev = hi;
        }
        assert_eq!(prev, m.num_kv_layers());
    }

    #[test]
    fn provenance_indices_snap_down_onto_attention_layers() {
        let m = KvLayerMap::new(&schedule(32, 4));
        // The scheduler's depth fractions for a 32-layer stack (the `.min`
        // the default applies is inert at this depth, so it is left off).
        let (syn, sem, prag) = (32 * 15 / 100, 32 / 2, 32 * 85 / 100);
        assert_eq!((syn, sem, prag), (4, 16, 27));
        assert_eq!(m.snap_to_attention(syn), Some(3));
        assert_eq!(m.snap_to_attention(sem), Some(15));
        assert_eq!(m.snap_to_attention(prag), Some(27), "27 already attends");
        // An exact hit stays put; the deepest layer is an attention layer.
        assert_eq!(m.snap_to_attention(31), Some(31));
        // Below every attention layer there is nothing to snap down to, so
        // the shallowest attention layer stands in.
        assert_eq!(m.snap_to_attention(0), Some(3));
        assert_eq!(m.snap_to_attention(2), Some(3));
    }

    #[test]
    fn an_all_attention_stack_is_the_identity() {
        let m = KvLayerMap::new(&[LayerKind::Attention; 4]);
        assert_eq!(m.num_kv_layers(), 4);
        for i in 0..4 {
            assert_eq!(m.kv_index(i), Some(i));
            assert_eq!(m.snap_to_attention(i), Some(i));
        }
    }

    #[test]
    fn an_all_recurrent_stack_has_no_kv() {
        let m = KvLayerMap::new(&[LayerKind::DeltaNet; 3]);
        assert_eq!(m.num_kv_layers(), 0);
        assert_eq!(m.kv_index(1), None);
        assert_eq!(
            m.snap_to_attention(1),
            None,
            "nothing to capture Q from — the caller must handle this rather \
             than be handed a layer that does not attend"
        );
    }

    #[test]
    fn a_leading_attention_schedule_maps_from_zero() {
        // `(i + 1) % 1 == 0` makes every layer attend; interval 2 puts the
        // first attention layer at index 1, so index 0 has no KV.
        let m = KvLayerMap::new(&schedule(6, 2));
        assert_eq!(m.attention_layers(), &[1, 3, 5]);
        assert_eq!(m.kv_index(0), None);
        assert_eq!(m.kv_index(1), Some(0));
        assert_eq!(m.snap_to_attention(0), Some(1));
    }
}
