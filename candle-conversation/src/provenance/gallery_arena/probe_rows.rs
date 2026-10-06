//! Each probe token's kernel result over one paged index, kept for the next
//! scan of the same index.
//!
//! The paged kernel writes one row per probe token — the best case and its
//! vote, per group, per segment — and a token's row depends on nothing but
//! that token and the gallery the index addresses. A decode's reprojections
//! rescan the same index with probes that overlap almost entirely: the query
//! head is the same every time and the trailing window has moved by the few
//! tokens decoded since. Scoring only the tokens no earlier scan of this index
//! scored is the same scan, row for row, at a fraction of the kernel time —
//! which is most of a reprojection's.
//!
//! The rows belong to the index: an index is reused only while every turn it
//! addresses holds the run it recorded and its segments fingerprint the same,
//! so a row is never read against a gallery other than the one that produced
//! it. Group weights are applied by the tally, after the rows, so one row
//! serves every weighting.

use std::collections::HashMap;

/// One probe token's kernel output: `n_groups × n_segments` cases and votes,
/// group-major, exactly as the kernel lays a token's slice of its output.
struct Row {
    case: Box<[i32]>,
    vote: Box<[f32]>,
}

/// The rows kept for one index, keyed by the probe token's words.
pub(super) struct ProbeRows {
    rows: HashMap<Box<[u64]>, Row>,
    /// Values per row (`n_groups × n_segments`).
    width: usize,
    /// Rows held at most; a scan that would pass it starts the cache over.
    cap: usize,
}

/// The host bytes one index's rows may hold. A row is 8 bytes per segment per
/// group, so for the ~1,000-file `code_reading` group this is about 2,800
/// tokens — several turns' worth of probes.
const ROW_BYTES: usize = 64 << 20;

impl ProbeRows {
    /// An empty cache for rows of `width` values.
    pub(super) fn new(width: usize) -> Self {
        Self {
            rows: HashMap::new(),
            width,
            cap: (ROW_BYTES / (width.max(1) * 8)).max(1),
        }
    }

    /// The tokens of `probe` (`wpt` words each) a launch has to score, as their
    /// first positions in `probe` — each distinct token with no row, once.
    ///
    /// When their rows would take the cache past its budget it starts over
    /// first, and every distinct token of `probe` is to be scored: the rows the
    /// probe already had are dropped here, before the launch, never between
    /// the launch and [`Self::assemble`].
    pub(super) fn plan(&mut self, probe: &[u64], wpt: usize) -> Vec<usize> {
        let missing = self.missing(probe, wpt);
        if self.rows.len() + missing.len() <= self.cap {
            return missing;
        }
        self.rows.clear();
        self.missing(probe, wpt)
    }

    fn missing(&self, probe: &[u64], wpt: usize) -> Vec<usize> {
        let mut seen: HashMap<&[u64], ()> = HashMap::new();
        probe
            .chunks_exact(wpt)
            .enumerate()
            .filter(|(_, words)| !self.rows.contains_key(*words))
            .filter(|(_, words)| seen.insert(words, ()).is_none())
            .map(|(t, _)| t)
            .collect()
    }

    /// Keep the rows a launch produced for `tokens` (their words, in launch
    /// order) — `case` and `vote` are that launch's whole output.
    pub(super) fn insert(&mut self, tokens: &[&[u64]], case: &[i32], vote: &[f32]) {
        let w = self.width;
        for (i, words) in tokens.iter().enumerate() {
            self.rows.insert(
                (*words).into(),
                Row {
                    case: case[i * w..(i + 1) * w].into(),
                    vote: vote[i * w..(i + 1) * w].into(),
                },
            );
        }
    }

    /// The full kernel output for `probe`, every token's row in order — what a
    /// launch over the whole probe would have written. `None` when a token has
    /// no row, which [`Self::plan`] followed by [`Self::insert`] rules out.
    pub(super) fn assemble(&self, probe: &[u64], wpt: usize) -> Option<(Vec<i32>, Vec<f32>)> {
        let n = probe.len() / wpt;
        let mut case = Vec::with_capacity(n * self.width);
        let mut vote = Vec::with_capacity(n * self.width);
        for words in probe.chunks_exact(wpt) {
            let row = self.rows.get(words)?;
            case.extend_from_slice(&row.case);
            vote.extend_from_slice(&row.vote);
        }
        Some((case, vote))
    }
}

#[cfg(test)]
mod tests {
    use super::ProbeRows;

    /// Two-word tokens, rows of three values.
    const WPT: usize = 2;

    fn rows_for(tokens: &[[u64; WPT]], seed: i32) -> (Vec<i32>, Vec<f32>) {
        let case = (0..tokens.len() as i32 * 3).map(|v| v + seed).collect();
        let vote = (0..tokens.len() * 3)
            .map(|v| v as f32 + seed as f32 * 0.5)
            .collect();
        (case, vote)
    }

    /// **A probe is assembled from rows in its own token order**, each token's
    /// row exactly as its launch wrote it — and only the tokens no scan has
    /// scored are asked for, each once.
    #[test]
    fn a_probe_is_assembled_from_the_rows_of_its_tokens() {
        let mut cache = ProbeRows::new(3);
        let first = [[1, 1], [2, 2]];
        let (case, vote) = rows_for(&first, 10);
        cache.insert(&[&first[0], &first[1]], &case, &vote);

        let probe: Vec<u64> = [[2, 2], [3, 3], [1, 1], [3, 3]].concat();
        assert_eq!(cache.plan(&probe, WPT), vec![1]);
        cache.insert(&[&[3, 3]], &[100, 101, 102], &[7.0, 8.0, 9.0]);
        assert_eq!(cache.plan(&probe, WPT), Vec::<usize>::new());

        let (case, vote) = cache.assemble(&probe, WPT).unwrap();
        assert_eq!(
            case,
            vec![13, 14, 15, 100, 101, 102, 10, 11, 12, 100, 101, 102]
        );
        assert_eq!(
            vote,
            vec![8.0, 9.0, 10.0, 7.0, 8.0, 9.0, 5.0, 6.0, 7.0, 7.0, 8.0, 9.0]
        );
    }

    /// A token with no row assembles nothing rather than a short output.
    #[test]
    fn a_token_without_a_row_assembles_nothing() {
        let cache = ProbeRows::new(3);
        assert!(cache.assemble(&[4, 4], WPT).is_none());
        assert!(cache.assemble(&[], WPT).unwrap().0.is_empty());
    }

    /// **The cache starts over rather than outgrow its budget — before the
    /// launch.** A probe whose new tokens do not fit has EVERY token scored,
    /// the ones it already had a row for included, so its output still
    /// assembles once the launch's rows are in.
    #[test]
    fn a_full_cache_starts_over_and_scores_the_whole_probe() {
        let mut cache = ProbeRows::new(3);
        cache.cap = 2;
        let a = [[1, 1], [2, 2]];
        let (case, vote) = rows_for(&a, 0);
        cache.insert(&[&a[0], &a[1]], &case, &vote);
        let probe: Vec<u64> = [[1, 1], [9, 9]].concat();
        assert_eq!(cache.plan(&probe, WPT), vec![0, 1]);
        let (case, vote) = rows_for(&[[1, 1], [9, 9]], 50);
        cache.insert(&[&[1, 1], &[9, 9]], &case, &vote);
        assert_eq!(
            cache.assemble(&probe, WPT).unwrap().0,
            vec![50, 51, 52, 53, 54, 55]
        );
    }
}
