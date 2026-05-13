//! Binary Directional Provenance scanner.
//!
//! Computes per-turn relevance scores by comparing a slice of probe Q
//! signatures against every stored chunk-group signature in the
//! [`ProvenanceFile`].  Produces five aggregation statistics per (turn,
//! depth) in a single pass — the projection engine picks whichever statistic
//! its layer schema asks for.
//!
//! # Algorithm
//!
//! ```text
//!  Input:
//!    probe[depth]  : Vec<TokenSignature>     (user-message Q sigs, per depth)
//!    corpus        : Vec<(group, idx, &[SigEntry])>   (turn → its chunks)
//!
//!  For each (group, idx) in corpus, sorted by byte_offset:
//!    For each depth in [syn, sem, prag]:
//!      For each (probe_tok, corpus_tok) pair:
//!        agreement = popcount(XNOR(probe_tok, corpus_tok))    // 0..=128
//!      Aggregate:
//!        max         = max(agreements)
//!        sum         = Σ agreements
//!        mean        = sum / count_pairs
//!        top_k_mean  = mean of the K highest agreements (K configurable)
//!        count       = number of pairs with agreement >= hit_threshold
//!  Store: (group, idx) → PerDepthScores
//! ```
//!
//! # Performance notes
//!
//! - Entries are sorted by `byte_offset` before scanning so the OS prefetcher
//!   can pull pages sequentially from the mmap.
//! - The mmap receives `MADV_SEQUENTIAL` via `with_mmap` to bias readahead.
//! - The scanner's score map is **persistent** for the conversation's
//!   lifetime — `scan` calls `clear` and re-inserts, reusing allocated
//!   bucket capacity.
//!
//! # Multi-chunk turns
//!
//! A turn that spans multiple 32-token blocks produces multiple [`SigEntry`]
//! records.  The scanner processes each entry separately and aggregates into
//! the same per-turn stat record — an `agreement` against any chunk of the
//! turn contributes to that turn's per-turn `max` / `sum` / `count` / etc.
//!
//! See `BdpScanner::scan` for the entry point.

use ahash::AHashMap;

use super::store::{
    depth_byte_range, DEPTH_PRAGMATIC, DEPTH_SEMANTIC, DEPTH_SYNTACTIC, ENTRY_BYTES_PER_TOKEN,
    NUM_DEPTHS,
};
use super::{ProvenanceFile, SigEntry, TokenSignature};
use crate::projection::{PerDepthScores, SectionId, TimelineId, TurnIndex, TurnScores};

// ── Tunables ──────────────────────────────────────────────────────────────────

/// Hamming-agreement threshold for the `count` metric.  Random baseline is
/// 64; useful directional signal starts around 80–90.  90 is conservative.
pub const DEFAULT_HIT_THRESHOLD: u32 = 90;

/// Default `K` for the `top_k_mean` metric.  Matches the projection schema's
/// `score_formula_k` default.
pub const DEFAULT_TOP_K: usize = 8;

// ── Aggregator ────────────────────────────────────────────────────────────────

/// Per-(turn, depth) running aggregator.  Folds in `(probe, corpus)`
/// agreement values one at a time, then collapses to [`TurnScores`] at the
/// end of the turn's chunk run.
struct Aggregator {
    max: u32,
    sum: u64,
    count_pairs: u64,
    count_hits: u64,
    /// Min-heap-of-size-K via a sorted ascending Vec: index 0 is the
    /// smallest of the current top-K.  When a new value beats it, pop and
    /// insert sorted.
    top_k: Vec<u32>,
    top_k_capacity: usize,
}

impl Aggregator {
    fn new(top_k_capacity: usize) -> Self {
        Self {
            max: 0,
            sum: 0,
            count_pairs: 0,
            count_hits: 0,
            top_k: Vec::with_capacity(top_k_capacity),
            top_k_capacity,
        }
    }

    #[inline]
    fn observe(&mut self, agreement: u32, hit_threshold: u32) {
        if agreement > self.max {
            self.max = agreement;
        }
        self.sum += agreement as u64;
        self.count_pairs += 1;
        if agreement >= hit_threshold {
            self.count_hits += 1;
        }

        // Maintain a sorted-ascending top-K.  O(K) insert; K is small (~8).
        if self.top_k.len() < self.top_k_capacity {
            let pos = self.top_k.partition_point(|&v| v <= agreement);
            self.top_k.insert(pos, agreement);
        } else if let Some(&min) = self.top_k.first() {
            if agreement > min {
                self.top_k.remove(0);
                let pos = self.top_k.partition_point(|&v| v <= agreement);
                self.top_k.insert(pos, agreement);
            }
        }
    }

    fn finish(self) -> TurnScores {
        let pairs = self.count_pairs.max(1) as f32;
        let mean = self.sum as f32 / pairs;
        let k = self.top_k.len().max(1) as f32;
        let top_k_sum: u64 = self.top_k.iter().map(|&v| v as u64).sum();
        let top_k_mean = top_k_sum as f32 / k;
        TurnScores {
            max: self.max as f32,
            sum: self.sum as f32,
            mean,
            top_k_mean,
            count: self.count_hits as f32,
        }
    }
}

// ── Scanner ───────────────────────────────────────────────────────────────────

/// Persistent per-conversation BDP scanner.
///
/// Holds an `ahash::AHashMap` keyed by `(TimelineId, TurnIndex)`.  Each
/// [`Self::scan`] call refreshes that map: the existing keys are cleared,
/// then repopulated from a fresh scan of the supplied corpus entries against
/// the supplied probe signatures.  Reusing the map's internal storage
/// across scans saves on allocator churn.
#[derive(Default)]
pub struct BdpScanner {
    scores: AHashMap<(TimelineId, TurnIndex), PerDepthScores>,
    section_scores: AHashMap<SectionId, PerDepthScores>,
    hit_threshold: u32,
    top_k: usize,
}

impl BdpScanner {
    /// Create a scanner with default tunables (`hit_threshold = 90`,
    /// `top_k = 8`).
    pub fn new() -> Self {
        Self {
            scores: AHashMap::new(),
            section_scores: AHashMap::new(),
            hit_threshold: DEFAULT_HIT_THRESHOLD,
            top_k: DEFAULT_TOP_K,
        }
    }

    /// Configure the agreement threshold for the `count` metric.
    pub fn with_hit_threshold(mut self, threshold: u32) -> Self {
        self.hit_threshold = threshold;
        self
    }

    /// Configure `K` for the `top_k_mean` metric.
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k.max(1);
        self
    }

    /// Read the score map produced by the most recent `scan`.
    pub fn scores(&self) -> &AHashMap<(TimelineId, TurnIndex), PerDepthScores> {
        &self.scores
    }

    /// Read the section-score map produced by the most recent
    /// [`Self::scan_sections`].
    pub fn section_scores(&self) -> &AHashMap<SectionId, PerDepthScores> {
        &self.section_scores
    }

    /// Clear all scores without releasing the map's allocated capacity.
    pub fn clear(&mut self) {
        self.scores.clear();
        self.section_scores.clear();
    }

    /// Scan all `corpus` entries against the supplied per-depth probe
    /// signatures.  The map is cleared first; each turn touched by the scan
    /// produces a fresh [`PerDepthScores`].
    ///
    /// `corpus` is `Vec<(TimelineId, TurnIndex, Vec<SigEntry>)>` — a
    /// single turn may contribute multiple entries (one per sealed
    /// 32-token chunk), and all of them aggregate into the same
    /// per-turn stat record.
    pub fn scan(
        &mut self,
        provenance: &ProvenanceFile,
        probe_syn: &[TokenSignature],
        probe_sem: &[TokenSignature],
        probe_prag: &[TokenSignature],
        corpus: &[(TimelineId, TurnIndex, Vec<SigEntry>)],
    ) -> crate::Result<()> {
        self.scores.clear();
        if corpus.is_empty() {
            return Ok(());
        }

        // Build a (byte_offset → corpus index, entry index) ordering so we
        // touch the mmap in offset order regardless of the caller's input
        // order.  This maximises sequential page access.
        let mut order: Vec<(u64, usize, usize)> = corpus
            .iter()
            .enumerate()
            .flat_map(|(i, (_, _, entries))| {
                entries
                    .iter()
                    .enumerate()
                    .map(move |(j, e)| (e.byte_offset, i, j))
            })
            .collect();
        order.sort_unstable_by_key(|&(off, _, _)| off);

        // Pre-allocate aggregators in corpus order so we can index them
        // directly during the offset-ordered walk.
        let mut aggs: Vec<[Aggregator; 3]> = (0..corpus.len())
            .map(|_| {
                [
                    Aggregator::new(self.top_k),
                    Aggregator::new(self.top_k),
                    Aggregator::new(self.top_k),
                ]
            })
            .collect();

        provenance.with_mmap(|mmap| {
            let file_len = mmap.len();
            for &(_, ci, ei) in &order {
                let entry = corpus[ci].2[ei];
                if entry.token_count == 0 {
                    continue;
                }
                let n = entry.token_count as usize;
                let offset = entry.byte_offset as usize;
                let total = n * ENTRY_BYTES_PER_TOKEN;
                if offset + total > file_len {
                    continue; // truncated or corrupt
                }
                let chunk = &mmap[offset..offset + total];
                let depth_slices: [&[u8]; NUM_DEPTHS] = [
                    &chunk[depth_byte_range(DEPTH_SYNTACTIC, n)],
                    &chunk[depth_byte_range(DEPTH_SEMANTIC, n)],
                    &chunk[depth_byte_range(DEPTH_PRAGMATIC, n)],
                ];
                let probes: [&[TokenSignature]; NUM_DEPTHS] = [probe_syn, probe_sem, probe_prag];

                for (di, (data, probe)) in depth_slices.iter().zip(probes.iter()).enumerate() {
                    accumulate_depth(data, n, probe, &mut aggs[ci][di], self.hit_threshold);
                }
            }
        })?;

        // Finalise: collapse aggregators into the score map.
        for (i, (timeline, idx, _)) in corpus.iter().enumerate() {
            let [syn_a, sem_a, prag_a] = aggs.remove(0);
            let _ = i; // placement is index-stable since we always remove(0)
            self.scores.insert(
                (*timeline, *idx),
                PerDepthScores {
                    syn: syn_a.finish(),
                    sem: sem_a.finish(),
                    prag: prag_a.finish(),
                },
            );
        }

        Ok(())
    }

    /// Section-keyed sibling of [`Self::scan`].
    ///
    /// Same algorithm — sort by `byte_offset` for sequential mmap
    /// access, accumulate per-(section, depth) Hamming agreements
    /// against the probes, finalise into [`PerDepthScores`].  The only
    /// difference is the corpus shape: keys are
    /// [`crate::projection::SectionId`] not `(TimelineId, TurnIndex)`.
    /// Results land in [`Self::section_scores`].
    ///
    /// Section scoring runs on the same probe as a turn scan; callers
    /// typically issue both to score the full corpus (turns + sections)
    /// against the same query in one round trip.
    pub fn scan_sections(
        &mut self,
        provenance: &ProvenanceFile,
        probe_syn: &[TokenSignature],
        probe_sem: &[TokenSignature],
        probe_prag: &[TokenSignature],
        corpus: &[(SectionId, Vec<SigEntry>)],
    ) -> crate::Result<()> {
        self.section_scores.clear();
        if corpus.is_empty() {
            return Ok(());
        }

        let mut order: Vec<(u64, usize, usize)> = corpus
            .iter()
            .enumerate()
            .flat_map(|(i, (_, entries))| {
                entries
                    .iter()
                    .enumerate()
                    .map(move |(j, e)| (e.byte_offset, i, j))
            })
            .collect();
        order.sort_unstable_by_key(|&(off, _, _)| off);

        let mut aggs: Vec<[Aggregator; 3]> = (0..corpus.len())
            .map(|_| {
                [
                    Aggregator::new(self.top_k),
                    Aggregator::new(self.top_k),
                    Aggregator::new(self.top_k),
                ]
            })
            .collect();

        provenance.with_mmap(|mmap| {
            let file_len = mmap.len();
            for &(_, ci, ei) in &order {
                let entry = corpus[ci].1[ei];
                if entry.token_count == 0 {
                    continue;
                }
                let n = entry.token_count as usize;
                let offset = entry.byte_offset as usize;
                let total = n * ENTRY_BYTES_PER_TOKEN;
                if offset + total > file_len {
                    continue;
                }
                let chunk = &mmap[offset..offset + total];
                let depth_slices: [&[u8]; NUM_DEPTHS] = [
                    &chunk[depth_byte_range(DEPTH_SYNTACTIC, n)],
                    &chunk[depth_byte_range(DEPTH_SEMANTIC, n)],
                    &chunk[depth_byte_range(DEPTH_PRAGMATIC, n)],
                ];
                let probes: [&[TokenSignature]; NUM_DEPTHS] =
                    [probe_syn, probe_sem, probe_prag];

                for (di, (data, probe)) in
                    depth_slices.iter().zip(probes.iter()).enumerate()
                {
                    accumulate_depth(data, n, probe, &mut aggs[ci][di], self.hit_threshold);
                }
            }
        })?;

        for (i, (section_id, _)) in corpus.iter().enumerate() {
            let [syn_a, sem_a, prag_a] = aggs.remove(0);
            let _ = i;
            self.section_scores.insert(
                *section_id,
                PerDepthScores {
                    syn: syn_a.finish(),
                    sem: sem_a.finish(),
                    prag: prag_a.finish(),
                },
            );
        }

        Ok(())
    }
}

/// Walk all `(probe_tok, corpus_tok)` pairs for one (turn, depth) chunk and
/// fold each agreement into `agg`.
#[inline]
fn accumulate_depth(
    data: &[u8],
    n: usize,
    probe: &[TokenSignature],
    agg: &mut Aggregator,
    hit_threshold: u32,
) {
    if probe.is_empty() || n == 0 {
        return;
    }
    for ci in 0..n {
        let c_bytes = &data[ci * TokenSignature::BYTE_LEN..(ci + 1) * TokenSignature::BYTE_LEN];
        for p in probe {
            let agreement = popcount_xnor(c_bytes, p.as_bytes());
            agg.observe(agreement, hit_threshold);
        }
    }
}

/// Hamming agreement between two `TokenSignature::BYTE_LEN`-byte signatures:
/// `popcount(XNOR(a, b))`, in the range `0..=128`.
#[inline]
fn popcount_xnor(a: &[u8], b: &[u8; TokenSignature::BYTE_LEN]) -> u32 {
    debug_assert_eq!(a.len(), TokenSignature::BYTE_LEN);
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (!(x ^ y)).count_ones())
        .sum()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::{TimelineId, TurnIndex};

    /// Test fixture: synthesise a deterministic [`TimelineId`].  The
    /// scanner doesn't care which timeline a corpus entry belongs to —
    /// it just keys per-(timeline, idx).
    fn timeline_id() -> TimelineId {
        TimelineId::for_test(1)
    }

    fn sig_with_first_byte(b: u8) -> TokenSignature {
        let mut bytes = [0u8; 16];
        bytes[0] = b;
        TokenSignature::from_bytes(&bytes)
    }

    #[test]
    fn aggregator_basic_stats() {
        let mut agg = Aggregator::new(3);
        for &v in &[10u32, 50, 90, 100, 128, 60] {
            agg.observe(v, 95);
        }
        let s = agg.finish();
        assert_eq!(s.max, 128.0);
        assert_eq!(s.sum, (10 + 50 + 90 + 100 + 128 + 60) as f32);
        assert_eq!(s.mean, s.sum / 6.0);
        // Top-3: 128, 100, 90 → mean = 318/3 = 106
        assert_eq!(s.top_k_mean, 318.0 / 3.0);
        // Hits >= 95: 100 and 128
        assert_eq!(s.count, 2.0);
    }

    #[test]
    fn aggregator_top_k_smaller_than_observations() {
        let mut agg = Aggregator::new(2);
        for &v in &[10u32, 50, 90, 100, 128, 60] {
            agg.observe(v, 95);
        }
        let s = agg.finish();
        // Top-2: 128, 100 → mean = 114
        assert_eq!(s.top_k_mean, 114.0);
    }

    #[test]
    fn aggregator_top_k_capacity_one() {
        let mut agg = Aggregator::new(1);
        for &v in &[5u32, 200, 50] {
            agg.observe(v, 100);
        }
        let s = agg.finish();
        assert_eq!(s.top_k_mean, 200.0);
    }

    #[test]
    fn popcount_identical_signatures_is_128() {
        let s = sig_with_first_byte(0xAB);
        let bytes = s.as_bytes();
        assert_eq!(popcount_xnor(bytes, bytes), 128);
    }

    #[test]
    fn popcount_complement_signatures_is_0() {
        let mut a_bytes = [0u8; 16];
        let mut b_bytes = [0u8; 16];
        for i in 0..16 {
            a_bytes[i] = 0xFF;
            b_bytes[i] = 0x00;
        }
        assert_eq!(popcount_xnor(&a_bytes, &b_bytes), 0);
    }

    #[test]
    fn scanner_empty_corpus_returns_empty_scores() {
        let provenance = ProvenanceFile::new().unwrap();
        let mut scanner = BdpScanner::new();
        scanner.scan(&provenance, &[], &[], &[], &[]).unwrap();
        assert!(scanner.scores().is_empty());
    }

    #[test]
    fn scanner_clears_between_calls() {
        let provenance = ProvenanceFile::new().unwrap();
        let probe = sig_with_first_byte(0x11);
        let corpus_sig = sig_with_first_byte(0x11);
        let entry = provenance
            .append(&[corpus_sig], &[corpus_sig], &[corpus_sig])
            .unwrap();
        let t = timeline_id();
        let i = TurnIndex(0);

        let mut scanner = BdpScanner::new();
        scanner
            .scan(
                &provenance,
                &[probe],
                &[probe],
                &[probe],
                &[(t, i, vec![entry])],
            )
            .unwrap();
        assert_eq!(scanner.scores().len(), 1);

        // Empty corpus on the next scan should clear the map.
        scanner.scan(&provenance, &[probe], &[probe], &[probe], &[]).unwrap();
        assert!(scanner.scores().is_empty());
    }

    #[test]
    fn scanner_identical_probe_and_corpus_yields_max_128() {
        let provenance = ProvenanceFile::new().unwrap();
        let s = sig_with_first_byte(0x33);
        let entry = provenance.append(&[s, s, s], &[s, s, s], &[s, s, s]).unwrap();
        let t = timeline_id();
        let i = TurnIndex(0);

        let mut scanner = BdpScanner::new().with_hit_threshold(120);
        scanner
            .scan(
                &provenance,
                &[s],
                &[s],
                &[s],
                &[(t, i, vec![entry])],
            )
            .unwrap();

        let scores = scanner.scores().get(&(t, i)).unwrap();
        // 3 corpus tokens × 1 probe token = 3 pairs, each agreement = 128
        assert_eq!(scores.syn.max, 128.0);
        assert_eq!(scores.sem.max, 128.0);
        assert_eq!(scores.prag.max, 128.0);
        assert_eq!(scores.syn.sum, 384.0);
        assert_eq!(scores.syn.mean, 128.0);
        assert_eq!(scores.syn.top_k_mean, 128.0);
        assert_eq!(scores.syn.count, 3.0);
    }

    #[test]
    fn scanner_multi_turn_aggregates_separately() {
        let provenance = ProvenanceFile::new().unwrap();
        let s_match = sig_with_first_byte(0x55);
        let s_mismatch = {
            let mut b = [0u8; 16];
            for i in 0..16 {
                b[i] = !s_match.as_bytes()[i];
            }
            TokenSignature::from_bytes(&b)
        };
        let entry_match = provenance
            .append(&[s_match], &[s_match], &[s_match])
            .unwrap();
        let entry_mismatch = provenance
            .append(&[s_mismatch], &[s_mismatch], &[s_mismatch])
            .unwrap();

        let t = timeline_id();
        let i_match = TurnIndex(0);
        let i_mismatch = TurnIndex(1);

        let mut scanner = BdpScanner::new();
        scanner
            .scan(
                &provenance,
                &[s_match],
                &[s_match],
                &[s_match],
                &[
                    (t, i_match, vec![entry_match]),
                    (t, i_mismatch, vec![entry_mismatch]),
                ],
            )
            .unwrap();

        let m = scanner.scores().get(&(t, i_match)).unwrap();
        let mm = scanner.scores().get(&(t, i_mismatch)).unwrap();
        assert_eq!(m.syn.max, 128.0);
        assert_eq!(mm.syn.max, 0.0);
    }
}
