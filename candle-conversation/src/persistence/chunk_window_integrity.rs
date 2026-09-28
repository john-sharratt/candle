//! Generic per-layer KV chunk window-geometry integrity check.
//!
//! A sequence's K/V is stored one chunk list per layer backing. Every layer
//! must describe the **same window** — `(offset within the chunk, valid
//! token count)` — at the **same chunk index**: attention reads a chunk's
//! tokens by that pair once per layer, and if two layers disagree the
//! shared position math attends to the wrong tokens on whichever layer is
//! out of step, with nothing raising a fault anywhere near the mistake.
//!
//! [`first_divergent_chunk`] is the pure comparison at the heart of that
//! invariant, factored out so a caller can ask the same question of a data
//! shape neither of the two existing, independent checkers reads:
//!
//! - `candle-transformers`' `assert_sealed_layers_aligned` asks it (its own
//!   copy of the same comparison, not this function) of a fresh live
//!   snapshot at seal time, before the K/V ever reaches disk.
//! - `candle-nn`'s `ChunkedKvBacking::first_window_divergence` asks it (also
//!   its own copy, over a different tuple type) of live GPU-resident
//!   backings at decode time, as a repair diagnostic.
//! - [`crate::persistence::resume::read_persisted_section_windows`] asks
//!   *this* module's [`first_divergent_chunk`] of a section's **persisted**
//!   chunks, read back from the redo log — the shape neither of the above
//!   two covers, because neither reads data as it would come back from disk
//!   with no live model attached. Consolidating the other two onto this
//!   function is future cleanup, not done here — they are not touched by
//!   the reactive repair this module exists for.
//!
//! This module holds only the comparison. Reading a particular caller's
//! data into the `&[Vec<ChunkWindow>]` shape it expects is that caller's
//! concern, not this one's — which is what keeps this function reusable by
//! a future caller (e.g. a boot-time integrity pass, or the two live
//! detectors above, should they ever be consolidated onto it) without it
//! needing to know anything about turns, sections, or the redo log.

/// One chunk's window geometry: `(offset within the physical chunk, valid
/// tokens from it)`. The part of a per-layer chunk record that every layer
/// backing the same sequence must agree on.
pub type ChunkWindow = (u16, u16);

/// The first point at which two layers describing the same sequence
/// disagree about a chunk's window geometry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowDivergence {
    /// Index of the first chunk where any layer disagrees with layer 0 —
    /// earliest rather than every disagreement, because a partially-applied
    /// per-layer operation diverges from the point it stopped, so the first
    /// index *is* the boundary and the layer split at it *is* the diagnosis.
    pub chunk_index: usize,
    /// Every layer's window at `chunk_index`, in layer order. `None` marks a
    /// layer with no chunk at that index at all (a short layer), which is a
    /// divergence in its own right and not merely a different window.
    pub per_layer: Vec<Option<ChunkWindow>>,
}

impl WindowDivergence {
    /// Group layers by what they hold at the divergent chunk and render a
    /// compact one-line summary — the same legend the two live-data
    /// detectors this module's `first_divergent_chunk` was factored out of
    /// use, because this is the same fault, reported from a different data
    /// source. Suitable as a tombstone record's `reason` or a log field.
    pub fn describe(&self) -> String {
        let mut groups: Vec<(Option<ChunkWindow>, Vec<usize>)> = Vec::new();
        for (layer, window) in self.per_layer.iter().enumerate() {
            match groups.iter_mut().find(|(held, _)| held == window) {
                Some((_, layers)) => layers.push(layer),
                None => groups.push((*window, vec![layer])),
            }
        }
        let split = groups
            .iter()
            .map(|(window, layers)| {
                let held = match window {
                    Some((offset, token_count)) => {
                        format!("(offset {offset}, {token_count} tokens)")
                    }
                    None => "no such chunk".to_string(),
                };
                format!("{held} on layer(s) {layers:?}")
            })
            .collect::<Vec<_>>()
            .join("; ");
        format!("chunk {}: {split}", self.chunk_index)
    }
}

/// Compare per-chunk `(offset, token_count)` windows across layers and
/// return the first point of divergence, if any.
///
/// `per_layer[l]` is layer `l`'s ordered chunk-window list. Every layer is
/// compared against layer 0; any two layers that disagree have the same
/// first divergent index (agreement is transitive up to the point one of
/// them breaks it), so comparing against a fixed layer rather than every
/// pair is sufficient and cheaper. A layer holding fewer chunks than
/// another counts as a divergence at the first index it lacks — a missing
/// chunk is not "no opinion", it is a different window (none) from a layer
/// that has one.
///
/// Returns `None` when `per_layer` has fewer than two layers (nothing to
/// compare) or when every layer agrees chunk-for-chunk over their full
/// length.
pub fn first_divergent_chunk(per_layer: &[Vec<ChunkWindow>]) -> Option<WindowDivergence> {
    if per_layer.len() < 2 {
        return None;
    }
    let max_len = per_layer.iter().map(Vec::len).max().unwrap_or(0);
    for chunk_index in 0..max_len {
        let base = per_layer[0].get(chunk_index).copied();
        let disagrees = per_layer[1..]
            .iter()
            .any(|layer| layer.get(chunk_index).copied() != base);
        if disagrees {
            let per_layer_at = per_layer
                .iter()
                .map(|layer| layer.get(chunk_index).copied())
                .collect();
            return Some(WindowDivergence {
                chunk_index,
                per_layer: per_layer_at,
            });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every layer holding the same windows is not a divergence — the
    /// overwhelmingly common case, and the one this must not flag.
    #[test]
    fn aligned_layers_report_no_divergence() {
        let uniform: Vec<ChunkWindow> = vec![(0, 32), (0, 32), (0, 17)];
        let per_layer: Vec<Vec<ChunkWindow>> = (0..13).map(|_| uniform.clone()).collect();
        assert_eq!(first_divergent_chunk(&per_layer), None);
    }

    /// A contiguous prefix of layers holds a chunk the rest never got —
    /// the mid-sweep-death signature. Reported at the first differing
    /// index, with both sides and the exact layer split.
    #[test]
    fn one_layer_diverges_early() {
        let ahead: Vec<ChunkWindow> = vec![(0, 32), (0, 32), (0, 32)];
        let behind: Vec<ChunkWindow> = vec![(0, 32), (0, 32), (0, 0)];
        let per_layer: Vec<Vec<ChunkWindow>> = (0..13)
            .map(|li| {
                if li < 3 {
                    ahead.clone()
                } else {
                    behind.clone()
                }
            })
            .collect();

        let divergence = first_divergent_chunk(&per_layer).expect("must report a divergence");
        assert_eq!(divergence.chunk_index, 2);
        assert_eq!(
            divergence.per_layer,
            (0..13)
                .map(|li| Some(if li < 3 { (0, 32) } else { (0, 0) }))
                .collect::<Vec<_>>()
        );
        let msg = divergence.describe();
        assert!(msg.contains("chunk 2"), "{msg}");
        assert!(
            msg.contains("(offset 0, 32 tokens) on layer(s) [0, 1, 2]"),
            "{msg}"
        );
        assert!(
            msg.contains("(offset 0, 0 tokens) on layer(s) [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]"),
            "{msg}"
        );
    }

    /// A same-length, same-token-count chunk whose OFFSET differs is still a
    /// divergence — the window is `(offset, token_count)`, not token_count
    /// alone, so an offset-only mismatch must not read as agreement.
    #[test]
    fn offset_mismatch_is_a_divergence_even_when_token_count_agrees() {
        let a: Vec<ChunkWindow> = vec![(0, 32)];
        let b: Vec<ChunkWindow> = vec![(1, 32)];
        let per_layer = vec![a, b];

        let divergence = first_divergent_chunk(&per_layer).expect("offset differs — must report");
        assert_eq!(divergence.chunk_index, 0);
        assert_eq!(divergence.per_layer, vec![Some((0, 32)), Some((1, 32))]);
    }

    /// A layer with fewer chunks than another is caught at the first index
    /// it lacks, distinct from a chunk it holds with a different window.
    #[test]
    fn a_short_layer_diverges_at_the_missing_index() {
        let full: Vec<ChunkWindow> = vec![(0, 32), (0, 32)];
        let short: Vec<ChunkWindow> = vec![(0, 32)];
        let per_layer = vec![full.clone(), full, short];

        let divergence = first_divergent_chunk(&per_layer).expect("short layer — must report");
        assert_eq!(divergence.chunk_index, 1);
        assert_eq!(
            divergence.per_layer,
            vec![Some((0, 32)), Some((0, 32)), None]
        );
        assert!(divergence.describe().contains("no such chunk"));
    }

    /// No layers at all is not a divergence — there is nothing to compare.
    #[test]
    fn empty_input_reports_no_divergence() {
        assert_eq!(first_divergent_chunk(&[]), None);
    }

    /// One layer alone is not a divergence either — divergence is a
    /// cross-layer disagreement, and there is no second layer to disagree.
    #[test]
    fn single_layer_input_reports_no_divergence() {
        let per_layer: Vec<Vec<ChunkWindow>> = vec![vec![(0, 32), (0, 17)]];
        assert_eq!(first_divergent_chunk(&per_layer), None);
    }
}
