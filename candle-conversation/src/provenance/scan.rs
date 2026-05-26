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
use crate::projection::{PerDepthScores, SectionId, TurnKey, TurnScores};

// ── Tunables ──────────────────────────────────────────────────────────────────

/// Hamming-agreement threshold for the `count` metric.  Random baseline is
/// 64; useful directional signal starts around 80–90.  90 is conservative.
pub const DEFAULT_HIT_THRESHOLD: u32 = 90;

/// Expected XOR-popcount agreement between two independent 128-bit
/// signatures — half the 128 bits agree by chance.  The `pertok_excess`
/// metric scores agreement *relative to* this baseline so a pure-noise pair
/// contributes ~0 rather than ~64.
pub const AGREEMENT_BASELINE: f32 = 64.0;

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
    /// Span tracking: one bool per probe token position.  `true` when any
    /// corpus token produced an above-threshold agreement with that probe
    /// token.  Lazily allocated on first hit (saves the alloc when no
    /// above-threshold pairs occur at all).
    probe_hits: Vec<bool>,
    /// Per-probe-token best (max) agreement across all corpus tokens — the
    /// graded, threshold-free counterpart to `probe_hits`.  Drives the
    /// `pertok_excess` metric.  Sized to the probe length on first
    /// `accumulate_depth` call; accumulates the max across multi-chunk turns.
    probe_best: Vec<u32>,
    /// α exponent for the span score (default 2.0).
    span_alpha: f32,
}

impl Aggregator {
    fn new(top_k_capacity: usize, span_alpha: f32) -> Self {
        Self {
            max: 0,
            sum: 0,
            count_pairs: 0,
            count_hits: 0,
            top_k: Vec::with_capacity(top_k_capacity),
            top_k_capacity,
            probe_hits: Vec::new(),
            probe_best: Vec::new(),
            span_alpha,
        }
    }

    /// Ensure `probe_hits` is sized for `probe_len` tokens.  Called lazily
    /// on the first above-threshold hit so we never allocate for cold turns.
    #[inline]
    fn ensure_probe_hits(&mut self, probe_len: usize) {
        if self.probe_hits.is_empty() {
            self.probe_hits.resize(probe_len, false);
        }
    }

    /// Ensure `probe_best` is sized for `probe_len` tokens.  Called once per
    /// `accumulate_depth` — unlike `probe_hits` it tracks every pair, not
    /// just above-threshold ones, so it is sized eagerly.
    #[inline]
    fn ensure_probe_best(&mut self, probe_len: usize) {
        if self.probe_best.len() < probe_len {
            self.probe_best.resize(probe_len, 0);
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

        // Span: Σ L^α over consecutive runs of hit probe positions.
        let span: f32 = self.probe_hits
            .split(|&h| !h)
            .filter(|run| !run.is_empty())
            .map(|run| (run.len() as f32).powf(self.span_alpha))
            .sum();

        // PerTokenExcess: Σ over probe tokens of max(0, best_agreement − 64).
        // Recentered (noise → ~0) and per-probe-token (one promiscuous token
        // cannot inflate it), threshold-free so weak sub-90 signal survives.
        let pertok_excess: f32 = self
            .probe_best
            .iter()
            .map(|&a| (a as f32 - AGREEMENT_BASELINE).max(0.0))
            .sum();

        TurnScores {
            max: self.max as f32,
            sum: self.sum as f32,
            mean,
            top_k_mean,
            count: self.count_hits as f32,
            span,
            pertok_excess,
        }
    }
}

// ── Hit record ────────────────────────────────────────────────────────────────

/// One above-threshold (probe, corpus) pair recorded during `scan_sections`
/// when [`BdpScanner::with_record_hits`] is enabled.
///
/// `depth` is 0 = syntactic, 1 = semantic, 2 = pragmatic.
#[derive(Debug, Clone)]
pub struct TokenHit {
    pub probe_tok: u16,
    pub corpus_tok: u16,
    pub agreement: u32,
    pub depth: u8,
}

// ── Scanner ───────────────────────────────────────────────────────────────────

/// Persistent per-conversation BDP scanner.
///
/// Holds an `ahash::AHashMap` keyed by [`TurnKey`].  Each
/// [`Self::scan`] call refreshes that map: the existing keys are cleared,
/// then repopulated from a fresh scan of the supplied corpus entries against
/// the supplied probe signatures.  Reusing the map's internal storage
/// across scans saves on allocator churn.
/// Default α exponent for the span score.
pub const DEFAULT_SPAN_ALPHA: f32 = 2.0;

#[derive(Default)]
pub struct BdpScanner {
    scores: AHashMap<TurnKey, PerDepthScores>,
    section_scores: AHashMap<SectionId, PerDepthScores>,
    section_hit_log: AHashMap<SectionId, Vec<TokenHit>>,
    hit_threshold: u32,
    top_k: usize,
    span_alpha: f32,
    record_hits: bool,
}

impl BdpScanner {
    /// Create a scanner with default tunables (`hit_threshold = 90`,
    /// `top_k = 8`, `span_alpha = 2.0`).
    pub fn new() -> Self {
        Self {
            scores: AHashMap::new(),
            section_scores: AHashMap::new(),
            section_hit_log: AHashMap::new(),
            hit_threshold: DEFAULT_HIT_THRESHOLD,
            top_k: DEFAULT_TOP_K,
            span_alpha: DEFAULT_SPAN_ALPHA,
            record_hits: false,
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

    /// Configure the α exponent for span scoring (default 2.0).
    /// α=1.0 gives linear span (same as count); α=2.0 rewards long runs
    /// quadratically; α>2.0 amplifies sustained attention further.
    pub fn with_span_alpha(mut self, alpha: f32) -> Self {
        self.span_alpha = alpha.max(1.0);
        self
    }

    /// Enable per-hit recording for [`Self::scan_sections`].
    ///
    /// When `true`, every (probe, corpus) pair whose agreement meets
    /// [`Self::hit_threshold`] is appended to [`Self::section_hit_log`].
    /// Off by default — recording adds O(hits) allocation to each scan.
    pub fn with_record_hits(mut self, record: bool) -> Self {
        self.record_hits = record;
        self
    }

    /// Read the score map produced by the most recent `scan`.
    pub fn scores(&self) -> &AHashMap<TurnKey, PerDepthScores> {
        &self.scores
    }

    /// Read the section-score map produced by the most recent
    /// [`Self::scan_sections`].
    pub fn section_scores(&self) -> &AHashMap<SectionId, PerDepthScores> {
        &self.section_scores
    }

    /// Materialize the scanner's accumulated turn + section scores into a
    /// fresh [`crate::substrate::ProjectionScores`] suitable for
    /// [`crate::projection::resolver::Conversation::read_scored`]. The
    /// scanner retains its own copy; this clone-out is for callers that
    /// want a self-contained, owned scores value.
    pub fn to_projection_scores(&self) -> crate::substrate::ProjectionScores {
        let mut out = crate::substrate::ProjectionScores::new();
        for (&key, scores) in &self.scores {
            out.set_turn(key.timeline, key.index, *scores);
        }
        for (&section_id, scores) in &self.section_scores {
            out.set_section(section_id, *scores);
        }
        out
    }

    /// Read the per-section hit log populated by the most recent
    /// [`Self::scan_sections`] when [`Self::with_record_hits`] is enabled.
    pub fn section_hit_log(&self) -> &AHashMap<SectionId, Vec<TokenHit>> {
        &self.section_hit_log
    }

    /// Clear all scores without releasing the map's allocated capacity.
    pub fn clear(&mut self) {
        self.scores.clear();
        self.section_scores.clear();
        self.section_hit_log.clear();
    }

    /// Scan all `corpus` entries against the supplied per-depth probe
    /// signatures.  The map is cleared first; each turn touched by the scan
    /// produces a fresh [`PerDepthScores`].
    ///
    /// `corpus` is `&[(TurnKey, Vec<SigEntry>)]` — a single turn may
    /// contribute multiple entries (one per sealed 32-token chunk), and
    /// all of them aggregate into the same per-turn stat record.
    pub fn scan(
        &mut self,
        provenance: &ProvenanceFile,
        probe_syn: &[TokenSignature],
        probe_sem: &[TokenSignature],
        probe_prag: &[TokenSignature],
        corpus: &[(TurnKey, Vec<SigEntry>)],
    ) -> crate::Result<()> {
        self.scores.clear();
        if corpus.is_empty() {
            return Ok(());
        }

        let entries_per_item: Vec<&[SigEntry]> =
            corpus.iter().map(|(_, e)| e.as_slice()).collect();
        let (aggs, _) = scan_core(
            provenance,
            &entries_per_item,
            probe_syn,
            probe_sem,
            probe_prag,
            self.top_k,
            self.span_alpha,
            self.hit_threshold,
            false,
        )?;

        for ((key, _), [syn_a, sem_a, prag_a]) in corpus.iter().zip(aggs) {
            self.scores.insert(
                *key,
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
    /// [`crate::projection::SectionId`] not [`TurnKey`].
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
        self.section_hit_log.clear();
        if corpus.is_empty() {
            return Ok(());
        }

        let entries_per_item: Vec<&[SigEntry]> =
            corpus.iter().map(|(_, e)| e.as_slice()).collect();
        let (aggs, raw_hits) = scan_core(
            provenance,
            &entries_per_item,
            probe_syn,
            probe_sem,
            probe_prag,
            self.top_k,
            self.span_alpha,
            self.hit_threshold,
            self.record_hits,
        )?;

        for ((section_id, _), [syn_a, sem_a, prag_a]) in corpus.iter().zip(aggs) {
            self.section_scores.insert(
                *section_id,
                PerDepthScores {
                    syn: syn_a.finish(),
                    sem: sem_a.finish(),
                    prag: prag_a.finish(),
                },
            );
        }

        if let Some(hits) = raw_hits {
            for ((section_id, _), section_hits) in corpus.iter().zip(hits) {
                self.section_hit_log.insert(*section_id, section_hits);
            }
        }

        Ok(())
    }
}

/// Core mmap-walk shared by [`BdpScanner::scan`] and [`BdpScanner::scan_sections`].
///
/// `entries_per_item[i]` is the slice of [`SigEntry`] records for corpus item `i`.
/// Entries are sorted by byte offset before the walk so the OS prefetcher reads
/// the mmap sequentially.
///
/// Returns one `[Aggregator; 3]` per item (in corpus order) and, when
/// `record_hits` is `true`, one `Vec<TokenHit>` per item.
fn scan_core(
    provenance: &ProvenanceFile,
    entries_per_item: &[&[SigEntry]],
    probe_syn: &[TokenSignature],
    probe_sem: &[TokenSignature],
    probe_prag: &[TokenSignature],
    top_k: usize,
    span_alpha: f32,
    hit_threshold: u32,
    record_hits: bool,
) -> crate::Result<(Vec<[Aggregator; 3]>, Option<Vec<Vec<TokenHit>>>)> {
    let n_items = entries_per_item.len();

    let mut order: Vec<(u64, usize, usize)> = entries_per_item
        .iter()
        .enumerate()
        .flat_map(|(i, entries)| {
            entries
                .iter()
                .enumerate()
                .map(move |(j, e)| (e.byte_offset, i, j))
        })
        .collect();
    order.sort_unstable_by_key(|&(off, _, _)| off);

    let mut aggs: Vec<[Aggregator; 3]> = (0..n_items)
        .map(|_| [
            Aggregator::new(top_k, span_alpha),
            Aggregator::new(top_k, span_alpha),
            Aggregator::new(top_k, span_alpha),
        ])
        .collect();

    let mut raw_hits: Option<Vec<Vec<TokenHit>>> = if record_hits {
        Some((0..n_items).map(|_| Vec::new()).collect())
    } else {
        None
    };

    provenance.with_mmap(|mmap| {
        let file_len = mmap.len();
        for &(_, ci, ei) in &order {
            let entry = entries_per_item[ci][ei];
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
            let probes: [&[TokenSignature]; NUM_DEPTHS] = [probe_syn, probe_sem, probe_prag];

            for (di, (data, probe)) in depth_slices.iter().zip(probes.iter()).enumerate() {
                accumulate_depth(data, n, probe, &mut aggs[ci][di], hit_threshold);

                if let Some(ref mut hits) = raw_hits {
                    for ct in 0..n {
                        let c_bytes = &data
                            [ct * TokenSignature::BYTE_LEN..(ct + 1) * TokenSignature::BYTE_LEN];
                        for (pt, p) in probe.iter().enumerate() {
                            let agreement = popcount_xnor(c_bytes, p.as_bytes());
                            if agreement >= hit_threshold {
                                hits[ci].push(TokenHit {
                                    probe_tok: pt as u16,
                                    corpus_tok: ct as u16,
                                    agreement,
                                    depth: di as u8,
                                });
                            }
                        }
                    }
                }
            }
        }
    })?;

    Ok((aggs, raw_hits))
}

/// Walk all `(probe_tok, corpus_tok)` pairs for one (turn, depth) chunk and
/// fold each agreement into `agg`.  Dispatches to the AVX2 fast path at
/// runtime when the CPU supports it; falls back to scalar otherwise.
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
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: feature detected at runtime immediately above.
        unsafe {
            accumulate_depth_avx2(data, n, probe, agg, hit_threshold);
        }
        return;
    }
    accumulate_depth_scalar(data, n, probe, agg, hit_threshold);
}

/// Scalar fallback: outer loop over corpus tokens, inner over probes.
fn accumulate_depth_scalar(
    data: &[u8],
    n: usize,
    probe: &[TokenSignature],
    agg: &mut Aggregator,
    hit_threshold: u32,
) {
    agg.ensure_probe_best(probe.len());
    for ci in 0..n {
        let c_bytes = &data[ci * TokenSignature::BYTE_LEN..(ci + 1) * TokenSignature::BYTE_LEN];
        for (pi, p) in probe.iter().enumerate() {
            let agreement = popcount_xnor(c_bytes, p.as_bytes());
            agg.observe(agreement, hit_threshold);
            if agreement > agg.probe_best[pi] {
                agg.probe_best[pi] = agreement;
            }
            if agreement >= hit_threshold {
                agg.ensure_probe_hits(probe.len());
                agg.probe_hits[pi] = true;
            }
        }
    }
}

/// AVX2 fast path: broadcasts each probe signature to a 256-bit register,
/// then processes two corpus tokens (32 bytes) per SIMD iteration using the
/// standard nibble-table popcount trick (`vpshufb` + `vpsadbw`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn accumulate_depth_avx2(
    data: &[u8],
    n: usize,
    probe: &[TokenSignature],
    agg: &mut Aggregator,
    hit_threshold: u32,
) {
    use std::arch::x86_64::*;

    const B: usize = TokenSignature::BYTE_LEN; // 16

    // Nibble → popcount lookup, duplicated for both 128-bit halves of a YMM.
    let lut = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    );
    let lo_mask = _mm256_set1_epi8(0x0F_u8 as i8);
    let zero = _mm256_setzero_si256();

    agg.ensure_probe_best(probe.len());
    for (pi, p) in probe.iter().enumerate() {
        // Broadcast 16-byte probe → 32-byte YMM (same value in both halves).
        let probe128 = _mm_loadu_si128(p.as_bytes().as_ptr() as *const __m128i);
        let probe256 = _mm256_broadcastsi128_si256(probe128);

        let mut ci = 0usize;
        let mut probe_hit = false;
        let mut best_pi = 0u32;
        // Two corpus tokens (32 bytes) per AVX2 iteration.
        while ci + 2 <= n {
            let corpus256 =
                _mm256_loadu_si256(data.as_ptr().add(ci * B) as *const __m256i);

            // Per-byte Hamming bits via nibble popcount of XOR.
            let xor_v = _mm256_xor_si256(corpus256, probe256);
            let lo = _mm256_and_si256(xor_v, lo_mask);
            let hi = _mm256_and_si256(_mm256_srli_epi16(xor_v, 4), lo_mask);
            let pc = _mm256_add_epi8(
                _mm256_shuffle_epi8(lut, lo),
                _mm256_shuffle_epi8(lut, hi),
            );

            // vpsadbw sums bytes in 64-bit blocks → 4 × u64 partial sums.
            let sad = _mm256_sad_epu8(pc, zero);
            let lo128 = _mm256_castsi256_si128(sad);
            let hi128 = _mm256_extracti128_si256(sad, 1);

            // Token ci+0: Hamming = bytes 0..15 = sad lane0 + lane1.
            let hdist0 = (_mm_extract_epi64(lo128, 0) as u64
                + _mm_extract_epi64(lo128, 1) as u64) as u32;
            // Token ci+1: Hamming = bytes 16..31 = sad lane2 + lane3.
            let hdist1 = (_mm_extract_epi64(hi128, 0) as u64
                + _mm_extract_epi64(hi128, 1) as u64) as u32;

            let ag0 = 128 - hdist0;
            let ag1 = 128 - hdist1;
            agg.observe(ag0, hit_threshold);
            agg.observe(ag1, hit_threshold);
            best_pi = best_pi.max(ag0).max(ag1);
            if ag0 >= hit_threshold || ag1 >= hit_threshold {
                probe_hit = true;
            }
            ci += 2;
        }

        // Scalar tail for odd n.
        if ci < n {
            let c = &data[ci * B..(ci + 1) * B];
            let ag = popcount_xnor(c, p.as_bytes());
            agg.observe(ag, hit_threshold);
            best_pi = best_pi.max(ag);
            if ag >= hit_threshold {
                probe_hit = true;
            }
        }

        if best_pi > agg.probe_best[pi] {
            agg.probe_best[pi] = best_pi;
        }
        if probe_hit {
            agg.ensure_probe_hits(probe.len());
            agg.probe_hits[pi] = true;
        }
    }
}

/// Hamming agreement between two `TokenSignature::BYTE_LEN`-byte signatures:
/// `popcount(XNOR(a, b))`, in the range `0..=128`.
/// Uses u128 XOR + NOT + count_ones → 2 POPCNT instructions on x86-64.
#[inline]
fn popcount_xnor(a: &[u8], b: &[u8; TokenSignature::BYTE_LEN]) -> u32 {
    debug_assert_eq!(a.len(), TokenSignature::BYTE_LEN);
    let a_arr: &[u8; TokenSignature::BYTE_LEN] = a.try_into().expect("BYTE_LEN == 16");
    let a128 = u128::from_le_bytes(*a_arr);
    let b128 = u128::from_le_bytes(*b);
    (!(a128 ^ b128)).count_ones()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::{TimelineId, TurnIndex, TurnKey, TurnScores};

    /// Test fixture: synthesise a deterministic [`TimelineId`].  The
    /// scanner doesn't care which timeline a corpus entry belongs to —
    /// it just keys per-[`TurnKey`].
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
        let mut agg = Aggregator::new(3, 2.0);
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
        let mut agg = Aggregator::new(2, 2.0);
        for &v in &[10u32, 50, 90, 100, 128, 60] {
            agg.observe(v, 95);
        }
        let s = agg.finish();
        // Top-2: 128, 100 → mean = 114
        assert_eq!(s.top_k_mean, 114.0);
    }

    #[test]
    fn aggregator_top_k_capacity_one() {
        let mut agg = Aggregator::new(1, 2.0);
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
        let key = TurnKey::new(t, i);

        let mut scanner = BdpScanner::new();
        scanner
            .scan(
                &provenance,
                &[probe],
                &[probe],
                &[probe],
                &[(key, vec![entry])],
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
        let key = TurnKey::new(t, i);

        let mut scanner = BdpScanner::new().with_hit_threshold(120);
        scanner
            .scan(
                &provenance,
                &[s],
                &[s],
                &[s],
                &[(key, vec![entry])],
            )
            .unwrap();

        let scores = scanner.scores().get(&key).unwrap();
        // 3 corpus tokens × 1 probe token = 3 pairs, each agreement = 128
        assert_eq!(scores.syn.max, 128.0);
        assert_eq!(scores.sem.max, 128.0);
        assert_eq!(scores.prag.max, 128.0);
        assert_eq!(scores.syn.sum, 384.0);
        assert_eq!(scores.syn.mean, 128.0);
        assert_eq!(scores.syn.top_k_mean, 128.0);
        assert_eq!(scores.syn.count, 3.0);
    }

    // ── AVX2 fast-path tests ──────────────────────────────────────────────────

    fn sigs_to_bytes(sigs: &[TokenSignature]) -> Vec<u8> {
        sigs.iter().flat_map(|s| s.as_bytes().iter().copied()).collect()
    }

    fn run_scalar_agg(data: &[u8], n: usize, probe: &[TokenSignature], thr: u32) -> TurnScores {
        let mut agg = Aggregator::new(8, 2.0);
        accumulate_depth_scalar(data, n, probe, &mut agg, thr);
        agg.finish()
    }

    #[cfg(target_arch = "x86_64")]
    fn run_avx2_agg(data: &[u8], n: usize, probe: &[TokenSignature], thr: u32) -> Option<TurnScores> {
        if !is_x86_feature_detected!("avx2") {
            return None;
        }
        let mut agg = Aggregator::new(8, 2.0);
        unsafe { accumulate_depth_avx2(data, n, probe, &mut agg, thr); }
        Some(agg.finish())
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn run_avx2_agg(_: &[u8], _: usize, _: &[TokenSignature], _: u32) -> Option<TurnScores> {
        None
    }

    #[test]
    fn avx2_identical_signatures_agreement_128() {
        let sig = sig_with_first_byte(0xAB);
        let corpus = [sig, sig];
        let data = sigs_to_bytes(&corpus);
        let s = run_scalar_agg(&data, 2, &[sig], 90);
        assert_eq!(s.max, 128.0);
        if let Some(a) = run_avx2_agg(&data, 2, &[sig], 90) {
            assert_eq!(a.max, s.max);
            assert_eq!(a.sum, s.sum);
            assert_eq!(a.count, s.count);
        }
    }

    #[test]
    fn avx2_complement_signatures_agreement_0() {
        let probe = {
            let mut b = [0xFFu8; 16];
            b[0] = 0xFF;
            TokenSignature::from_bytes(&b)
        };
        let corpus_sig = TokenSignature::from_bytes(&[0x00u8; 16]);
        let corpus = [corpus_sig, corpus_sig];
        let data = sigs_to_bytes(&corpus);
        let s = run_scalar_agg(&data, 2, &[probe], 10);
        assert_eq!(s.max, 0.0);
        if let Some(a) = run_avx2_agg(&data, 2, &[probe], 10) {
            assert_eq!(a.max, s.max);
            assert_eq!(a.sum, s.sum);
            assert_eq!(a.count, s.count);
        }
    }

    #[test]
    fn avx2_matches_scalar_n1_scalar_tail_only() {
        // n=1: no full AVX2 pairs, only the scalar tail fires.
        let probe = sig_with_first_byte(0x55);
        let corpus = [sig_with_first_byte(0x55)];
        let data = sigs_to_bytes(&corpus);
        let s = run_scalar_agg(&data, 1, &[probe], 90);
        if let Some(a) = run_avx2_agg(&data, 1, &[probe], 90) {
            assert_eq!(a.max, s.max);
            assert_eq!(a.sum, s.sum);
        }
    }

    #[test]
    fn avx2_matches_scalar_n3_pair_plus_tail() {
        // n=3: one full AVX2 pair + one scalar tail.
        let probe = sig_with_first_byte(0x33);
        let c0 = sig_with_first_byte(0x33); // full match
        let c1 = sig_with_first_byte(0x00); // partial
        let c2 = sig_with_first_byte(0xFF); // partial
        let corpus = [c0, c1, c2];
        let data = sigs_to_bytes(&corpus);
        let s = run_scalar_agg(&data, 3, &[probe], 64);
        if let Some(a) = run_avx2_agg(&data, 3, &[probe], 64) {
            assert_eq!(a.max, s.max);
            assert_eq!(a.sum, s.sum);
            assert_eq!(a.count, s.count);
        }
    }

    #[test]
    fn avx2_matches_scalar_multi_probe() {
        // Multiple probe tokens — exercises the outer probe loop in AVX2 path.
        let p0 = sig_with_first_byte(0xAA);
        let p1 = sig_with_first_byte(0x55);
        let c0 = sig_with_first_byte(0xAA);
        let c1 = sig_with_first_byte(0x55);
        let c2 = sig_with_first_byte(0xFF);
        let c3 = sig_with_first_byte(0x00);
        let corpus = [c0, c1, c2, c3];
        let data = sigs_to_bytes(&corpus);
        let probe = [p0, p1];
        let s = run_scalar_agg(&data, 4, &probe, 64);
        if let Some(a) = run_avx2_agg(&data, 4, &probe, 64) {
            assert_eq!(a.max, s.max);
            assert_eq!(a.sum, s.sum);
            assert_eq!(a.count, s.count);
        }
    }

    #[test]
    fn avx2_matches_scalar_n4_exact_two_pairs() {
        // n=4: exactly two AVX2 pairs, no tail.
        let probe = sig_with_first_byte(0xCC);
        let corpus: Vec<TokenSignature> = (0u8..4)
            .map(|i| sig_with_first_byte(0xCC ^ (i * 0x11)))
            .collect();
        let data = sigs_to_bytes(&corpus);
        let s = run_scalar_agg(&data, 4, &[probe], 80);
        if let Some(a) = run_avx2_agg(&data, 4, &[probe], 80) {
            assert_eq!(a.max, s.max);
            assert_eq!(a.sum, s.sum);
            assert_eq!(a.count, s.count);
        }
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
        let k_match = TurnKey::new(t, TurnIndex(0));
        let k_mismatch = TurnKey::new(t, TurnIndex(1));

        let mut scanner = BdpScanner::new();
        scanner
            .scan(
                &provenance,
                &[s_match],
                &[s_match],
                &[s_match],
                &[
                    (k_match, vec![entry_match]),
                    (k_mismatch, vec![entry_mismatch]),
                ],
            )
            .unwrap();

        let m = scanner.scores().get(&k_match).unwrap();
        let mm = scanner.scores().get(&k_mismatch).unwrap();
        assert_eq!(m.syn.max, 128.0);
        assert_eq!(mm.syn.max, 0.0);
    }
}
