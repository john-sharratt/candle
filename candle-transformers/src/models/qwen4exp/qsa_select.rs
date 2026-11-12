//! QSA selection — the one definition of *which cells a query attends*.
//!
//! Both consumers read this module: the CPU oracle ([`super::qsa`]) turns the
//! selection into an additive `0 / −inf` mask row, and the engine hands it to
//! the paged attention kernels as a packed entry list. Stating it once is what
//! makes "the engine matches the oracle" a property of one function rather
//! than of two implementations that happen to agree.
//!
//! # The selection, from `build_qsa_top_k` (`docs/qwen38_flash_next.md` §12.5)
//!
//! A query at absolute position `qpos`, over an index cache of `ratio`-cell
//! blocks:
//!
//! - Its **tail** — cells `tail_start ..= qpos` where
//!   `tail_start = (qpos+1)/ratio·ratio` — is always attended. That is the
//!   query's own incomplete block, `(qpos+1) mod ratio` cells of it.
//! - Every **complete block below the tail** is a candidate, carrying the
//!   indexer's score. Blocks are ranked by `(score desc, block asc)` and
//!   spent, whole, against the remaining budget; the block the budget runs out
//!   inside contributes its **lowest** cells.
//! - The budget is `top_k + ratio − 1` cells in total (2051 for the released
//!   checkpoint), tail included.
//! - A query with no more visible cells than the budget attends all of them —
//!   [`RowSelection::Dense`], the identity, and the reason a ≤2051-position
//!   context needs no indexer at all.
//!
//! Cells share a score with their block, and the reference breaks score ties
//! by ascending cell index, so ranking cells and ranking blocks give the same
//! set — see [`selection_entries`]'s implementation note.
//!
//! # Stratified selection
//!
//! The candidates may be cut into windows ([`Strata`]), each ranking its own
//! blocks — plus the system prompt's and, as candidates, the recent span's —
//! and spending the whole budget on them; the query attends the union. One
//! window over every candidate is the selection above, exactly
//! (`docs/qsa_stratified_selection.md`).
//!
//! # The packing
//!
//! One entry is one **run of cells at the bottom of a block**, which is all
//! the selection can produce: `(block << 2) | (cells − 1)`, ascending by
//! block. A whole block is `cells == ratio`. The two-bit cell field is why
//! [`MAX_RATIO`] is 4 — the released checkpoint's `indexer_compress_ratio`,
//! and the widest the packing admits without a second word per entry.

use std::collections::BTreeMap;
use std::ops::Range;

use crate::models::selection_strata::{Recent, StrataTokens};

/// Bits of an entry reserved for the cell count.
const CELL_BITS: u32 = 2;
/// The largest compression ratio the entry packing expresses.
pub const MAX_RATIO: usize = 1 << CELL_BITS;
/// `sel_cnt` marking a row that attends every visible cell (no restriction).
///
/// The kernels test this before touching the entry list, so a dense row costs
/// one comparison rather than a search.
pub const DENSE_ROW: u32 = u32::MAX;

/// Pack a run of `cells` cells at the bottom of `block`.
#[inline]
pub fn pack_entry(block: usize, cells: usize) -> u32 {
    debug_assert!((1..=MAX_RATIO).contains(&cells));
    ((block as u32) << CELL_BITS) | (cells as u32 - 1)
}

/// The block an entry names.
#[inline]
pub fn entry_block(entry: u32) -> usize {
    (entry >> CELL_BITS) as usize
}

/// How many of the block's lowest cells the entry selects.
#[inline]
pub fn entry_cells(entry: u32) -> usize {
    (entry & ((1 << CELL_BITS) - 1)) as usize + 1
}

/// The attended-cell budget: `top_k` positions plus the tail's `ratio − 1`.
#[inline]
pub fn selected_width(top_k: usize, ratio: usize) -> usize {
    top_k + ratio - 1
}

/// The most candidate blocks one query's selection can **keep** — the kernel's
/// `keep`, which bounds how much of the ranking it must materialise.
///
/// See [`max_entries`] for why the bound is taken over the tail width rather
/// than read off `top_k` directly.
#[inline]
pub fn max_keep(top_k: usize, ratio: usize) -> usize {
    let width = selected_width(top_k, ratio);
    // `n_tail == 0` leaves the whole width as candidate budget, so `keep` is
    // `full + [rem > 0]` = `ceil(width / ratio)`.
    let no_tail = width.div_ceil(ratio);
    if ratio >= 2 {
        // A tail takes `n_tail` off the budget, so the widest case is the
        // *narrowest* tail, `n_tail == 1` — which only exists at `ratio >= 2`.
        no_tail.max((width - 1) / ratio + 1)
    } else {
        no_tail
    }
}

/// The most entries one query's selection can hold.
///
/// The budget buys `budget/ratio` whole blocks, at most one partial block where
/// it runs out, and the tail — and a query short enough to attend everything
/// visible reaches the same bound from the other side (it is
/// [`RowSelection::Dense`] before it can exceed it).
///
/// **The bound is over the tail width, not `top_k/ratio + 2`.** That older form
/// read the whole-block count off `top_k` as though the budget were `top_k`,
/// but the budget is `selected_width(top_k, ratio) − n_tail`, and the two
/// disagree whenever `top_k % ratio >= 3`: at `top_k = 2047, ratio = 4` a query
/// with `n_tail == 1` produces 514 entries against a bound of 513. This sizes
/// the device row stride (`indexer`), while the selection kernel writes its
/// entry count unbounded — so the overrun landed in the next query's row and
/// its `cnt` recorded the over-count, leaving the search reading a neighbour's
/// blocks as its own. Silent wrong attention, no fault. The CPU reference
/// pushes into a growable `Vec`, so no oracle test could see it, and the
/// shipped 2048/4 divides exactly and reaches the same 514 either way.
#[inline]
pub fn max_entries(top_k: usize, ratio: usize) -> usize {
    // One more than `max_keep`'s tail case: the tail block gets its own entry.
    let width = selected_width(top_k, ratio);
    let no_tail = width.div_ceil(ratio);
    if ratio >= 2 {
        no_tail.max((width - 1) / ratio + 2)
    } else {
        no_tail
    }
}

/// Whether the selection kernel can run a budget of `top_k` positions on a
/// checkpoint of `context` tokens whose attention layers select at `ratios`,
/// when the kernel keeps at most `kernel_keep` blocks a row.
///
/// A budget whose [`max_keep`] passes the kernel's ceiling fails the first wave
/// that selects — every wave past the budget, mid-decode — so it is refused
/// where it is set. A budget at or past the context is the one wide budget that
/// runs: no row ever has more visible cells than it, so nothing selects and the
/// read is dense through the same code.
pub fn budget_fits_kernel(
    top_k: usize,
    ratios: &[usize],
    context: usize,
    kernel_keep: usize,
) -> Result<(), String> {
    if top_k >= context {
        return Ok(());
    }
    match ratios
        .iter()
        .copied()
        .filter(|&r| r > 0)
        .find(|&r| max_keep(top_k, r) > kernel_keep)
    {
        None => Ok(()),
        Some(ratio) => Err(format!(
            "a QSA selection budget of {top_k} position(s) keeps up to {} blocks a row at \
             ratio {ratio}, past the selection kernel's {kernel_keep}: set one the kernel can \
             run, or one of at least the {context}-token context, where nothing selects",
            max_keep(top_k, ratio)
        )),
    }
}

/// What [`selection_entries`] decided for one query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowSelection {
    /// Every visible cell is attended — the mask is the causal mask alone.
    Dense,
    /// `n` entries were written, ascending by block.
    Entries(usize),
}

/// How a query's candidate blocks are divided before they are ranked — the
/// stratified selection (`docs/qsa_stratified_selection.md`).
///
/// The candidates are cut into windows of `window_blocks` blocks walking
/// forward from block 0, and **each window spends the whole budget on its own
/// blocks**: a block competes only with the blocks of its window, plus the two
/// spans every window also ranks — the conversation's system prompt and, as a
/// candidate, the recent span nearest the query. The attended set is the union
/// of every window's choice, the forced recent span, and the query's own tail.
///
/// Why: the budget is a fixed count, so the share of a deep context it can
/// reach shrinks with depth — 512 of ~73,700 blocks at 294K tokens. A short
/// turn of the conversation then has to outrank every near-duplicate in the
/// whole context to be seen at all. A window bounds the competition to what the
/// checkpoint was trained to choose among.
///
/// [`Strata::WHOLE`] — one window spanning every candidate, no recent span — is
/// exactly the checkpoint's selection, not an approximation of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Strata {
    /// Candidate blocks per window, counted from block 0. `0` is one window
    /// spanning every candidate.
    pub window_blocks: usize,
    /// How many of the candidate blocks nearest the query form the recent span.
    pub recent_blocks: usize,
    /// How the recent span enters the selection.
    pub recent: Recent,
    /// The most blocks one row may pin — what the row's width allows for
    /// beside the budget. A row whose pinned ranges cover more is trimmed to
    /// it ([`PinRow::new`]).
    pub pin_blocks: usize,
}

impl Strata {
    /// One window over every candidate, no recent span, nothing pinned: the
    /// checkpoint's selection.
    pub const WHOLE: Self = Self {
        window_blocks: 0,
        recent_blocks: 0,
        recent: Recent::Candidate,
        pin_blocks: 0,
    };

    /// `tokens`, a strata stated in positions, in blocks of `ratio` positions.
    /// A window or span that does not divide into whole blocks rounds up, so it
    /// never covers less than was asked for. The pinned bound allows twice
    /// the positions' blocks — a short block per sealed section — and one
    /// straddling block per range.
    pub fn from_tokens(tokens: StrataTokens, ratio: usize) -> Self {
        Self {
            window_blocks: tokens.window.div_ceil(ratio),
            recent_blocks: tokens.recent.div_ceil(ratio),
            recent: tokens.mode,
            pin_blocks: match tokens.pinned {
                0 => 0,
                // Twice the positions' blocks: every pinned section is its own
                // sealed page, and each page can end on a short block, so a
                // run of sections of at least `ratio` positions covers at
                // most twice its positions' blocks — a journal of many short
                // entries fits without its front being trimmed.
                p => 2 * p.div_ceil(ratio) + MAX_PINS,
            },
        }
    }

    /// Blocks per window over `cand` candidates.
    pub fn window_width(&self, cand: usize) -> usize {
        if self.window_blocks == 0 {
            cand.max(1)
        } else {
            self.window_blocks
        }
    }

    /// Windows `cand` candidates are cut into — at least one.
    pub fn windows(&self, cand: usize) -> usize {
        cand.div_ceil(self.window_width(cand)).max(1)
    }

    /// Blocks attended outside every window's ranking.
    pub fn forced(&self, cand: usize) -> usize {
        match self.recent {
            Recent::Forced => self.recent_blocks.min(cand),
            Recent::Candidate => 0,
        }
    }
}

/// The most entries one query's selection can hold under `strata` over `cand`
/// candidate blocks — what a selection table's row stride is sized from.
///
/// Each window keeps at most [`max_keep`] blocks, the forced span and the
/// pinned ranges add their own, and no row can name more blocks than it has
/// candidates; the tail takes one more. Under [`Strata::WHOLE`] with at least
/// [`max_keep`] candidates this is [`max_entries`] — the checkpoint's row,
/// unchanged.
pub fn max_entries_for(top_k: usize, ratio: usize, cand: usize, strata: &Strata) -> usize {
    let blocks =
        strata.windows(cand) * max_keep(top_k, ratio) + strata.forced(cand) + strata.pin_blocks;
    blocks.min(cand) + 1
}

/// The most entries one query's windows choose before the union reduces
/// repeats — what the selection kernel's sort buffer is sized from.
///
/// [`max_entries_for`] bounds what survives the union. Before it, a block
/// chosen by several windows (a prompt block, or a recent candidate) appears
/// once per window that chose it, so the count is bounded by the sum of every
/// window's keep, not by the row's candidates. The pinned blocks are sorted in
/// with them; the forced span and the tail sit above every window and never
/// enter the sort.
pub fn max_gathered_for(top_k: usize, ratio: usize, cand: usize, strata: &Strata) -> usize {
    strata.windows(cand) * max_keep(top_k, ratio) + strata.pin_blocks
}

/// The most pinned ranges one row carries — the kernel's per-row table
/// (`QSA_MAX_PINS` in `qsa_select.cuh`).
pub const MAX_PINS: usize = 4;

/// One row's pinned block ranges as the selection kernel reads them: up to
/// [`MAX_PINS`] disjoint `[lo, hi)` pairs, unused ones empty.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PinRow(pub [(u32, u32); MAX_PINS]);

impl PinRow {
    /// No pins.
    pub const EMPTY: Self = Self([(0, 0); MAX_PINS]);

    /// `ranges` (block ranges) as a row of at most `max_blocks` pinned blocks:
    /// sorted, merged where they touch, coalesced across the smallest gaps
    /// down to [`MAX_PINS`] ranges ([`coalesce_ranges`]), then trimmed from
    /// the front of the last range while the row pins more than `max_blocks`.
    ///
    /// Never refused, because it is built inside a forward, where a refusal
    /// would fail the wave. The last range holds the newest pinned content
    /// before the query — a journal's entries run oldest first — so trimming
    /// its front gives up the oldest of it first; a range trimmed away leaves
    /// the one before it to trim.
    pub fn new(ranges: &[Range<usize>], max_blocks: usize) -> Self {
        let mut ranges = coalesce_ranges(ranges, MAX_PINS);
        let mut total: usize = ranges.iter().map(|r| r.len()).sum();
        while total > max_blocks {
            let Some(last) = ranges.last_mut() else {
                break;
            };
            let cut = (total - max_blocks).min(last.len());
            last.start += cut;
            total -= cut;
            if last.start >= last.end {
                ranges.pop();
            }
        }
        let mut row = Self::EMPTY;
        for (slot, r) in row.0.iter_mut().zip(&ranges) {
            *slot = (r.start as u32, r.end as u32);
        }
        row
    }

    /// The row's ranges, empty ones left out.
    pub fn ranges(&self) -> Vec<Range<usize>> {
        self.0
            .iter()
            .filter(|(lo, hi)| hi > lo)
            .map(|&(lo, hi)| lo as usize..hi as usize)
            .collect()
    }

    /// Blocks the row pins.
    pub fn blocks(&self) -> usize {
        self.0
            .iter()
            .map(|&(lo, hi)| hi.saturating_sub(lo) as usize)
            .sum()
    }
}

/// `ranges` merged ([`merge_ranges`]) and then, while more than `max` remain,
/// the two nearest joined across the gap between them — the fewest unpinned
/// positions attended to fit the table. A pinned span split only by the glue
/// between a collection's members comes back whole.
pub fn coalesce_ranges(ranges: &[Range<usize>], max: usize) -> Vec<Range<usize>> {
    let mut out = merge_ranges(ranges);
    while out.len() > max.max(1) {
        let at = (0..out.len() - 1)
            .min_by_key(|&i| out[i + 1].start - out[i].end)
            .expect("two ranges at least");
        out[at].end = out[at + 1].end;
        out.remove(at + 1);
    }
    out
}

/// `ranges` sorted by start, empty ones dropped, overlapping or touching ones
/// merged.
pub fn merge_ranges(ranges: &[Range<usize>]) -> Vec<Range<usize>> {
    let mut sorted: Vec<Range<usize>> = ranges.iter().filter(|r| !r.is_empty()).cloned().collect();
    sorted.sort_by_key(|r| r.start);
    let mut out: Vec<Range<usize>> = Vec::with_capacity(sorted.len());
    for r in sorted {
        match out.last_mut() {
            Some(last) if r.start <= last.end => last.end = last.end.max(r.end),
            _ => out.push(r),
        }
    }
    out
}

/// Whether block `b` is pinned by one of `pins` (block ranges).
#[inline]
pub fn is_pinned(pins: &[Range<usize>], b: usize) -> bool {
    pins.iter().any(|p| p.contains(&b))
}

/// `pins` (block ranges) as a row of `cand` candidates whose forced span
/// starts at `forced_lo` sees them: clipped below the forced span, empty
/// ranges dropped. A pinned block inside the forced span is attended there.
pub fn row_pins(pins: &[Range<usize>], forced_lo: usize) -> Vec<Range<usize>> {
    pins.iter()
        .map(|p| p.start.min(forced_lo)..p.end.min(forced_lo))
        .filter(|p| !p.is_empty())
        .collect()
}

/// The candidate blocks window `w` ranks, ascending: its own blocks, the
/// prompt's, and — as candidates — the recent span's, less the forced span
/// and the pinned blocks.
fn window_pool(
    strata: &Strata,
    cand: usize,
    prompt_blocks: usize,
    pins: &[Range<usize>],
    w: usize,
) -> Vec<u32> {
    let width = strata.window_width(cand);
    let (lo, hi) = (w * width, ((w + 1) * width).min(cand));
    let forced_lo = cand - strata.forced(cand);
    let recent_lo = cand - strata.recent_blocks.min(cand);
    let prompt = prompt_blocks.min(cand);
    let recent = match strata.recent {
        Recent::Candidate => recent_lo..cand,
        Recent::Forced => cand..cand,
    };
    let mut pool: Vec<u32> = (lo..hi)
        .chain(0..prompt)
        .chain(recent)
        .filter(|&b| b < forced_lo && !is_pinned(pins, b))
        .map(|b| b as u32)
        .collect();
    pool.sort_unstable();
    pool.dedup();
    pool
}

/// Build one query's selection under `strata`, ascending by block, into `out`.
///
/// `scores[b]` is block `b`'s indexer score; only blocks below the query's
/// tail are read, so `scores` may be shorter than the sequence's block count
/// as long as it covers `((qpos+1)/ratio·ratio)/ratio` entries.
/// `prompt_blocks` is how many leading blocks hold the conversation's system
/// prompt — ranked in every window. `pins` are the sequence's pinned block
/// ranges: attended whole, outside every window's ranking, so they take no
/// candidate's seat (`docs/hit_driven_projection.md` §5). `out` is cleared
/// first and grows to at most [`max_entries_for`].
///
/// # Implementation note — why ranking blocks is ranking cells
///
/// The reference ranks *cells* by `(score desc, cell asc)`, and a block's
/// cells all carry its score. Cells of a lower-indexed block therefore precede
/// cells of a higher-indexed block of equal score, and a block's own cells
/// stay in index order, so the cell ranking is exactly the block ranking with
/// each block expanded in place. Cutting the cell ranking at the budget cuts
/// the block ranking at `budget/ratio` whole blocks plus the low
/// `budget mod ratio` cells of the next.
///
/// Each window makes that cut over its own pool. A block two windows both
/// choose — a prompt block, or a recent one ranked as a candidate — is attended
/// once, at the wider of the two cuts.
#[allow(clippy::too_many_arguments)]
pub fn selection_entries(
    scores: &[f32],
    qpos: usize,
    ratio: usize,
    top_k: usize,
    strata: &Strata,
    prompt_blocks: usize,
    pins: &[Range<usize>],
    out: &mut Vec<u32>,
) -> RowSelection {
    debug_assert!((1..=MAX_RATIO).contains(&ratio));
    out.clear();
    let width = selected_width(top_k, ratio);
    let visible = qpos + 1;
    if visible <= width {
        return RowSelection::Dense;
    }

    let tail_start = visible / ratio * ratio;
    let n_tail = visible - tail_start;
    // `visible > width` ⇒ `tail_start > width − n_tail`, so the candidate
    // blocks always hold at least the budget the tail leaves.
    let budget = width - n_tail;
    let n_cand = tail_start / ratio;
    let full = budget / ratio;
    let rem = budget % ratio;

    let by_rank = |a: &u32, b: &u32| {
        let (sa, sb) = (scores[*a as usize], scores[*b as usize]);
        sb.partial_cmp(&sa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(b))
    };
    // Cells attended per chosen block; a block chosen by several windows keeps
    // its widest cut.
    let mut cells: BTreeMap<u32, usize> = BTreeMap::new();
    let mut choose = |b: u32, n: usize| {
        cells.entry(b).and_modify(|c| *c = (*c).max(n)).or_insert(n);
    };
    let pins = row_pins(pins, n_cand - strata.forced(n_cand));
    for w in 0..strata.windows(n_cand) {
        // Rank the window's pool by (score desc, block asc). The cut needs the
        // first `full + 1` in that order, so a partial sort of the prefix is
        // enough.
        let mut order = window_pool(strata, n_cand, prompt_blocks, &pins, w);
        let n_pool = order.len();
        let keep = (full + usize::from(rem > 0)).min(n_pool);
        if keep < n_pool {
            order.select_nth_unstable_by(keep, by_rank);
            order.truncate(keep + 1);
        }
        order.sort_unstable_by(by_rank);
        for &b in order.iter().take(full) {
            choose(b, ratio);
        }
        if rem > 0 {
            if let Some(&b) = order.get(full) {
                choose(b, rem);
            }
        }
    }
    for b in pins.iter().flat_map(|p| p.clone()) {
        choose(b as u32, ratio);
    }
    for b in n_cand - strata.forced(n_cand)..n_cand {
        choose(b as u32, ratio);
    }

    // Ascending already: the chosen blocks are all below the tail's block.
    out.extend(cells.iter().map(|(&b, &n)| pack_entry(b as usize, n)));
    if n_tail > 0 {
        out.push(pack_entry(n_cand, n_tail));
    }
    RowSelection::Entries(out.len())
}

/// Whether `pos` is attended, given a row's ascending entry list.
///
/// The predicate the kernels implement: binary-search the entry naming
/// `pos / ratio`, then test `pos mod ratio` against its cell count.
pub fn selects(entries: &[u32], pos: usize, ratio: usize) -> bool {
    let (block, cell) = (pos / ratio, pos % ratio);
    match entries.binary_search_by_key(&block, |&e| entry_block(e)) {
        Ok(i) => cell < entry_cells(entries[i]),
        Err(_) => false,
    }
}

// Pinned ranges are lists of ranges, and a list of one is the common case.
#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use super::*;

    /// `max_entries` and `max_keep` must bound what `selection_entries`
    /// actually emits, for **every** `(top_k, ratio, qpos)` — not just the ones
    /// the shipped checkpoint uses.
    ///
    /// This is the test the old `top_k / ratio + 2` bound never had to face. It
    /// was checked against 2048/4, which divides exactly and agrees with the
    /// true bound; the whole failure lived at `top_k % ratio >= 3`, four values
    /// away. A device row is sized from these, so an over-emission by one is a
    /// write into the next query's selection.
    #[test]
    fn the_entry_bounds_hold_for_every_geometry() {
        for ratio in 1..=MAX_RATIO {
            for top_k in 1..=64usize {
                let width = selected_width(top_k, ratio);
                let bound = max_entries(top_k, ratio);
                let keep_bound = max_keep(top_k, ratio);
                // Past `width` the row is no longer dense, which is where the
                // budget arithmetic starts; go well beyond so every `n_tail`
                // residue is exercised at several block counts.
                for qpos in 0..(width + 4 * ratio + 8) {
                    let n_blocks = (qpos + 1).div_ceil(ratio).max(1);
                    // Descending scores, so the ranking is a non-trivial
                    // permutation rather than the identity.
                    let scores: Vec<f32> = (0..n_blocks).map(|b| (n_blocks - b) as f32).collect();
                    let mut out = Vec::new();
                    if let RowSelection::Entries(n) = selection_entries(
                        &scores,
                        qpos,
                        ratio,
                        top_k,
                        &Strata::WHOLE,
                        0,
                        &[],
                        &mut out,
                    ) {
                        assert_eq!(n, out.len());
                        assert!(
                            n <= bound,
                            "top_k={top_k} ratio={ratio} qpos={qpos}: emitted {n} entries \
                             against max_entries {bound}"
                        );
                        // Whole-block entries are exactly the kept candidates;
                        // a partial and the tail are the two extras.
                        let whole = out.iter().filter(|&&e| entry_cells(e) == ratio).count();
                        assert!(
                            whole <= keep_bound,
                            "top_k={top_k} ratio={ratio} qpos={qpos}: kept {whole} blocks \
                             against max_keep {keep_bound}"
                        );
                    }
                }
            }
        }
    }

    /// The exact case the old bound got wrong, pinned so it cannot regress
    /// quietly: `2047 % 4 == 3` with a one-cell tail emits 514 entries, which
    /// `top_k / ratio + 2` bounded at 513.
    #[test]
    fn a_tail_of_one_is_the_widest_selection() {
        assert_eq!(max_entries(2047, 4), 514);
        // The shipped geometry divides exactly and is unchanged by the fix.
        assert_eq!(max_entries(2048, 4), 514);

        let n_blocks = 4096 / 4;
        let scores: Vec<f32> = (0..n_blocks).map(|b| (n_blocks - b) as f32).collect();
        let mut out = Vec::new();
        // qpos 4096 ⇒ visible 4097, tail_start 4096, n_tail 1.
        let sel = selection_entries(&scores, 4096, 4, 2047, &Strata::WHOLE, 0, &[], &mut out);
        assert_eq!(sel, RowSelection::Entries(514));
        assert!(out.len() <= max_entries(2047, 4));
    }

    /// The reference cell ranking, written out longhand exactly as §12.5
    /// states it — the thing [`selection_entries`] claims to be a compact
    /// form of.
    fn reference_cells(scores: &[f32], qpos: usize, ratio: usize, top_k: usize) -> Vec<usize> {
        let width = selected_width(top_k, ratio);
        let tail_start = (qpos + 1) / ratio * ratio;
        let mut order: Vec<(f32, usize)> = (0..=qpos)
            .map(|j| {
                let s = if j >= tail_start {
                    1e9f32
                } else {
                    scores[j / ratio]
                };
                (s, j)
            })
            .collect();
        order.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        let mut sel: Vec<usize> = order.into_iter().take(width).map(|(_, j)| j).collect();
        sel.sort_unstable();
        sel
    }

    fn expand(entries: &[u32], ratio: usize) -> Vec<usize> {
        let mut cells: Vec<usize> = entries
            .iter()
            .flat_map(|&e| {
                let b = entry_block(e);
                (0..entry_cells(e)).map(move |c| b * ratio + c)
            })
            .collect();
        cells.sort_unstable();
        cells
    }

    fn lcg_scores(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 40) as f32 / 64.0
            })
            .collect()
    }

    #[test]
    fn packing_round_trips() {
        for block in [0usize, 1, 7, 512, 262_143] {
            for cells in 1..=MAX_RATIO {
                let e = pack_entry(block, cells);
                assert_eq!(entry_block(e), block);
                assert_eq!(entry_cells(e), cells);
            }
        }
    }

    #[test]
    fn entries_expand_to_the_reference_cell_set() {
        // Every phase of `qpos` modulo the ratio, at a depth where the budget
        // bites — the partial-block cut lands differently at each.
        let (ratio, top_k) = (4usize, 8usize);
        let scores = lcg_scores(64, 11);
        let mut out = Vec::new();
        for qpos in 11..64 {
            let sel = selection_entries(
                &scores,
                qpos,
                ratio,
                top_k,
                &Strata::WHOLE,
                0,
                &[],
                &mut out,
            );
            let want = reference_cells(&scores, qpos, ratio, top_k);
            match sel {
                RowSelection::Dense => {
                    assert_eq!(want, (0..=qpos).collect::<Vec<_>>(), "qpos {qpos} dense");
                }
                RowSelection::Entries(n) => {
                    assert_eq!(n, out.len());
                    assert_eq!(expand(&out, ratio), want, "qpos {qpos}");
                    assert!(out.len() <= max_entries(top_k, ratio), "qpos {qpos} count");
                }
            }
        }
    }

    #[test]
    fn ratio_two_and_three_agree_with_the_reference() {
        for ratio in [2usize, 3] {
            let top_k = 6;
            let scores = lcg_scores(64, 23 + ratio as u64);
            let mut out = Vec::new();
            for qpos in 0..64 {
                let sel = selection_entries(
                    &scores,
                    qpos,
                    ratio,
                    top_k,
                    &Strata::WHOLE,
                    0,
                    &[],
                    &mut out,
                );
                let want = reference_cells(&scores, qpos, ratio, top_k);
                match sel {
                    RowSelection::Dense => {
                        assert_eq!(want, (0..=qpos).collect::<Vec<_>>(), "r{ratio} q{qpos}")
                    }
                    RowSelection::Entries(_) => {
                        assert_eq!(expand(&out, ratio), want, "r{ratio} q{qpos}")
                    }
                }
            }
        }
    }

    #[test]
    fn ties_are_broken_by_ascending_block() {
        // Every candidate scores the same: the reference keeps the LOWEST
        // cells, so the selection is a prefix of the blocks.
        let (ratio, top_k) = (4usize, 8usize);
        let scores = vec![1.0f32; 64];
        let mut out = Vec::new();
        let qpos = 50; // tail = 48..=50, budget = 11 − 3 = 8 cells = 2 blocks
        assert_eq!(
            selection_entries(
                &scores,
                qpos,
                ratio,
                top_k,
                &Strata::WHOLE,
                0,
                &[],
                &mut out
            ),
            RowSelection::Entries(3)
        );
        assert_eq!(
            expand(&out, ratio),
            vec![0, 1, 2, 3, 4, 5, 6, 7, 48, 49, 50]
        );
    }

    #[test]
    fn dense_below_the_budget_and_selective_above_it() {
        let (ratio, top_k) = (4usize, 2048usize);
        let width = selected_width(top_k, ratio); // 2051
        let scores = lcg_scores(2048, 7);
        let mut out = Vec::new();
        assert_eq!(
            selection_entries(
                &scores,
                width - 1,
                ratio,
                top_k,
                &Strata::WHOLE,
                0,
                &[],
                &mut out
            ),
            RowSelection::Dense
        );
        let sel = selection_entries(
            &scores,
            width,
            ratio,
            top_k,
            &Strata::WHOLE,
            0,
            &[],
            &mut out,
        );
        let RowSelection::Entries(n) = sel else {
            panic!("position {width} must engage selection");
        };
        assert_eq!(n, out.len());
        assert_eq!(expand(&out, ratio).len(), width);
        assert!(n <= max_entries(top_k, ratio));
    }

    #[test]
    fn membership_matches_the_expanded_set() {
        let (ratio, top_k) = (4usize, 8usize);
        let scores = lcg_scores(64, 31);
        let mut out = Vec::new();
        for qpos in [13usize, 22, 47, 63] {
            let RowSelection::Entries(_) = selection_entries(
                &scores,
                qpos,
                ratio,
                top_k,
                &Strata::WHOLE,
                0,
                &[],
                &mut out,
            ) else {
                continue;
            };
            let cells = expand(&out, ratio);
            for pos in 0..=qpos {
                assert_eq!(
                    selects(&out, pos, ratio),
                    cells.contains(&pos),
                    "qpos {qpos} pos {pos}"
                );
            }
        }
    }

    /// A budget is accepted only when the kernel can select with it, or when it
    /// is wide enough that nothing ever selects. At ratio 4 and the kernel's
    /// 768 survivors the last budget that runs is 3,069 positions (width 3,072,
    /// 768 blocks); 3,070 needs 769, and 49,152 needs 12,289.
    #[test]
    fn a_budget_the_selection_kernel_cannot_run_is_refused() {
        let ratios = [0usize, 4, 4];
        let context = 262_144usize;
        assert_eq!(budget_fits_kernel(2048, &ratios, context, 768), Ok(()));
        assert_eq!(budget_fits_kernel(3069, &ratios, context, 768), Ok(()));
        assert_eq!(
            budget_fits_kernel(3070, &ratios, context, 768),
            Err(
                "a QSA selection budget of 3070 position(s) keeps up to 769 blocks a row at \
                 ratio 4, past the selection kernel's 768: set one the kernel can run, or one \
                 of at least the 262144-token context, where nothing selects"
                    .to_string()
            )
        );
        assert_eq!(
            budget_fits_kernel(49_152, &ratios, context, 768),
            Err(
                "a QSA selection budget of 49152 position(s) keeps up to 12289 blocks a row at \
                 ratio 4, past the selection kernel's 768: set one the kernel can run, or one \
                 of at least the 262144-token context, where nothing selects"
                    .to_string()
            )
        );
        // At or past the context nothing selects, however wide.
        assert_eq!(budget_fits_kernel(context, &ratios, context, 768), Ok(()));
        assert_eq!(budget_fits_kernel(1 << 20, &ratios, context, 768), Ok(()));
    }

    // —— Stratified selection ————————————————————————————————————————————————
    //
    // One geometry throughout: ratio 4, top_k 8 (width 11), a query at 63 —
    // visible 64, no tail, 16 candidate blocks, a budget of 11 cells = 2 whole
    // blocks and the low 3 cells of a third, spent once per window.

    const R: usize = 4;
    const K: usize = 8;
    const Q: usize = 63;

    /// Window 0 (blocks 0–7) outscores window 1 (blocks 8–15) everywhere.
    const SKEWED: [f32; 16] = [
        9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 1.5, 0.5, 1.7, 0.2, 0.9, 1.1, 0.3,
    ];

    fn strata(window_blocks: usize, recent_blocks: usize, recent: Recent) -> Strata {
        Strata {
            window_blocks,
            recent_blocks,
            recent,
            pin_blocks: 0,
        }
    }

    fn select(scores: &[f32], strata: &Strata, prompt_blocks: usize) -> Vec<u32> {
        select_pinned(scores, strata, prompt_blocks, &[])
    }

    fn select_pinned(
        scores: &[f32],
        strata: &Strata,
        prompt_blocks: usize,
        pins: &[Range<usize>],
    ) -> Vec<u32> {
        let mut out = Vec::new();
        let sel = selection_entries(scores, Q, R, K, strata, prompt_blocks, pins, &mut out);
        assert_eq!(sel, RowSelection::Entries(out.len()));
        out
    }

    /// **A pinned range is attended whole and takes no candidate's seat**:
    /// the budget still buys the same number of ranked blocks, chosen from
    /// what is not pinned, and the pinned blocks are added beside them.
    #[test]
    fn a_pinned_range_is_attended_beside_the_budget() {
        // Unpinned, the top of the skew: blocks 0, 1 and the low 3 cells of 2.
        assert_eq!(
            select(&SKEWED, &Strata::WHOLE, 0),
            entries(&[(0, 4), (1, 4), (2, 3)])
        );
        // Pin the two weakest blocks of window 1 (12 and 15 score lowest):
        // the ranked part is unchanged and the pins ride on top.
        assert_eq!(
            select_pinned(&SKEWED, &Strata::WHOLE, 0, &[12..13, 15..16]),
            entries(&[(0, 4), (1, 4), (2, 3), (12, 4), (15, 4)])
        );
        // Pin the strongest two: the budget moves down the ranking to 2, 3
        // and the low cells of 4, and 0 and 1 are attended whole.
        assert_eq!(
            select_pinned(&SKEWED, &Strata::WHOLE, 0, &[0..2]),
            entries(&[(0, 4), (1, 4), (2, 4), (3, 4), (4, 3)])
        );
    }

    /// Pins reach every window, are never ranked in one, and stop below the
    /// forced span, which attends its own blocks.
    #[test]
    fn pins_hold_under_strata_and_stop_at_the_forced_span() {
        let two = strata(8, 0, Recent::Candidate);
        // Each window spends 2 whole blocks and 3 cells. Window 1 ranks
        // 11 (1.7), 9 (1.5), 14 (1.1), 8 (1.0): unpinned it keeps 11 and 9
        // whole and 14's low cells.
        let plain = select(&SKEWED, &two, 0);
        let pinned = select_pinned(&SKEWED, &two, 0, &[11..12]);
        assert!(plain.contains(&pack_entry(11, 4)));
        assert!(pinned.contains(&pack_entry(11, 4)));
        // Window 1 now ranks without 11: 9, 14 whole and the low cells of 8.
        assert!(pinned.contains(&pack_entry(9, 4)));
        assert!(pinned.contains(&pack_entry(14, 4)));
        assert!(pinned.contains(&pack_entry(8, 3)));
        // A pin reaching into a forced recent span is clipped there: the
        // forced span attends those blocks itself, once.
        let forced = strata(0, 2, Recent::Forced);
        let got = select_pinned(&SKEWED, &forced, 0, &[13..16]);
        let blocks: Vec<usize> = got.iter().map(|&e| entry_block(e)).collect();
        assert_eq!(blocks.iter().filter(|&&b| b == 14).count(), 1);
        assert_eq!(blocks.iter().filter(|&&b| b == 15).count(), 1);
        assert!(got.contains(&pack_entry(13, 4)));
    }

    /// The pinned bound is part of the row's: twice the positions' blocks, for
    /// the short block each sealed section can end on, plus one straddling
    /// block per range — and both the entry and gathered bounds carry them.
    #[test]
    fn the_pinned_bound_widens_the_row() {
        let s = Strata::from_tokens(StrataTokens::WHOLE.with_pinned(1024), 4);
        assert_eq!(s.pin_blocks, 512 + MAX_PINS);
        assert_eq!(max_entries_for(2048, 4, 73_728, &s), 514 + 512 + MAX_PINS);
        assert_eq!(max_gathered_for(2048, 4, 73_728, &s), 513 + 512 + MAX_PINS);
        assert_eq!(Strata::from_tokens(StrataTokens::WHOLE, 4).pin_blocks, 0);
    }

    /// The definition's pin table is the kernel's.
    #[cfg(feature = "cuda")]
    #[test]
    fn the_pin_table_is_the_kernels() {
        assert_eq!(MAX_PINS, candle_kernels::simple::qsa_topk::MAX_PINS);
    }

    /// A pin row is the ranges sorted and merged where they touch, padded
    /// with empty pairs.
    #[test]
    fn a_pin_row_merges_and_pads() {
        let row = PinRow::new(&[9..12, 0..3, 2..5, 12..14, 20..20], 100);
        assert_eq!(row.0, [(0, 5), (9, 14), (0, 0), (0, 0)]);
        assert_eq!(row.ranges(), vec![0..5, 9..14]);
        assert_eq!(row.blocks(), 10);
        assert_eq!(PinRow::new(&[], 100), PinRow::EMPTY);
    }

    /// **More ranges than the table holds coalesce across the smallest gaps**:
    /// a persona and a journal whose entries are split by one-block glue come
    /// back as the persona and one journal range.
    #[test]
    fn ranges_past_the_table_coalesce_across_the_smallest_gaps() {
        let journal = [20..25, 26..30, 31..34, 35..40];
        let mut ranges = vec![0..8];
        ranges.extend(journal.iter().cloned());
        let row = PinRow::new(&ranges, 100);
        assert_eq!(row.ranges(), vec![0..8, 20..30, 31..34, 35..40]);
        assert_eq!(coalesce_ranges(&ranges, 2), vec![0..8, 20..40]);
        assert_eq!(coalesce_ranges(&ranges, 1), vec![0..40]);
    }

    /// **A row past its bound gives up its newest range's front first** —
    /// the oldest of what the last pinned span holds — and never fails.
    #[test]
    fn a_row_past_its_bound_trims_the_last_ranges_front() {
        let row = PinRow::new(&[0..8, 20..40], 18);
        assert_eq!(row.ranges(), vec![0..8, 30..40]);
        let row = PinRow::new(&[0..8, 20..24], 6);
        assert_eq!(
            row.ranges(),
            vec![2..8],
            "the last range gone, then the one before"
        );
        assert_eq!(PinRow::new(&[0..8], 0), PinRow::EMPTY);
    }

    fn entries(blocks: &[(usize, usize)]) -> Vec<u32> {
        blocks.iter().map(|&(b, c)| pack_entry(b, c)).collect()
    }

    /// The whole-context cut takes the top of the skew and nothing from the
    /// lower window; two windows each spend the full budget on their own.
    #[test]
    fn each_window_spends_the_whole_budget_on_its_own_blocks() {
        assert_eq!(
            select(&SKEWED, &Strata::WHOLE, 0),
            entries(&[(0, 4), (1, 4), (2, 3)])
        );
        assert_eq!(
            select(&SKEWED, &strata(8, 0, Recent::Candidate), 0),
            entries(&[(0, 4), (1, 4), (2, 3), (9, 4), (11, 4), (14, 3)])
        );
    }

    /// The prompt is ranked in every window: block 0 tops window 1 as well,
    /// and is attended once.
    #[test]
    fn the_prompt_competes_in_every_window_and_is_attended_once() {
        assert_eq!(
            select(&SKEWED, &strata(8, 0, Recent::Candidate), 1),
            entries(&[(0, 4), (1, 4), (2, 3), (9, 3), (11, 4)])
        );
    }

    /// A recent candidate is ranked beside window 0's blocks too, and wins a
    /// seat there that it would also have won in its own window.
    #[test]
    fn a_recent_candidate_is_ranked_in_every_window() {
        let mut scores = SKEWED;
        scores[15] = 8.5;
        assert_eq!(
            select(&scores, &Strata::WHOLE, 0),
            entries(&[(0, 4), (1, 3), (15, 4)])
        );
        assert_eq!(
            select(&scores, &strata(8, 2, Recent::Candidate), 0),
            entries(&[(0, 4), (1, 3), (9, 3), (11, 4), (15, 4)])
        );
    }

    /// Forced recent blocks are attended whole however they score, and no
    /// window spends budget on them: window 1 ranks only blocks 8–13.
    #[test]
    fn a_forced_recent_span_is_attended_whole_and_ranked_nowhere() {
        assert_eq!(
            select(&SKEWED, &strata(8, 2, Recent::Forced), 0),
            entries(&[
                (0, 4),
                (1, 4),
                (2, 3),
                (8, 3),
                (9, 4),
                (11, 4),
                (14, 4),
                (15, 4)
            ])
        );
        // One window, forced: the span is still added and still unranked.
        assert_eq!(
            select(&SKEWED, &strata(0, 2, Recent::Forced), 0),
            entries(&[(0, 4), (1, 4), (2, 3), (14, 4), (15, 4)])
        );
    }

    /// Block 14 is the third pick of its own window — a partial cut — and the
    /// first of window 0's; attended once, it keeps the whole block.
    #[test]
    fn a_block_whole_in_one_window_and_partial_in_another_stays_whole() {
        let scores: [f32; 16] = [
            1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 9.0, 8.0, 0.1, 0.1, 0.1, 0.1, 7.0, 0.2,
        ];
        assert_eq!(
            select(&scores, &strata(8, 2, Recent::Candidate), 0),
            entries(&[(0, 4), (1, 3), (8, 4), (9, 4), (14, 4)])
        );
    }

    /// A window as wide as the candidates, or wider, is one window — the
    /// checkpoint's selection, with or without a prompt and a recent candidate
    /// span, both of which it already ranks.
    #[test]
    fn one_window_is_the_checkpoints_selection() {
        let mut out = Vec::new();
        let mut want = Vec::new();
        for seed in 0..32u64 {
            let scores = lcg_scores(64, 0x5157 + seed);
            for qpos in 11..256usize {
                let n_cand = (qpos + 1) / R;
                selection_entries(&scores, qpos, R, K, &Strata::WHOLE, 0, &[], &mut want);
                for (window, prompt, recent) in [(n_cand, 0, 0), (n_cand + 7, 3, 5), (0, 9, 64)] {
                    let s = strata(window, recent, Recent::Candidate);
                    selection_entries(&scores, qpos, R, K, &s, prompt, &[], &mut out);
                    assert_eq!(out, want, "seed {seed} qpos {qpos} window {window}");
                }
            }
        }
    }

    /// [`max_entries_for`] bounds every stratified row, for every geometry,
    /// window, span and mode tried — the row stride is sized from it.
    #[test]
    fn the_stratified_entry_bound_holds() {
        let mut out = Vec::new();
        for ratio in 1..=MAX_RATIO {
            for top_k in [1usize, 3, 8, 13] {
                let width = selected_width(top_k, ratio);
                for qpos in width..width + 160 {
                    let n_cand = (qpos + 1) / ratio;
                    let scores = lcg_scores(n_cand.max(1), (qpos * 7 + top_k) as u64);
                    for window in [0usize, 1, 3, 8] {
                        for recent_blocks in [0usize, 2, 9] {
                            for recent in [Recent::Candidate, Recent::Forced] {
                                for (prompt, pins) in [
                                    (0usize, vec![]),
                                    (2, vec![0..1]),
                                    (20, vec![0..2, 5..9]),
                                    (0, vec![3..40]),
                                ] {
                                    let mut s = strata(window, recent_blocks, recent);
                                    s.pin_blocks =
                                        pins.iter().map(|p: &Range<usize>| p.len()).sum();
                                    let sel = selection_entries(
                                        &scores, qpos, ratio, top_k, &s, prompt, &pins, &mut out,
                                    );
                                    let RowSelection::Entries(n) = sel else {
                                        continue;
                                    };
                                    let bound = max_entries_for(top_k, ratio, n_cand, &s);
                                    assert!(
                                        n <= bound,
                                        "ratio {ratio} top_k {top_k} qpos {qpos} {s:?} prompt \
                                         {prompt}: {n} entries against {bound}"
                                    );
                                    assert!(
                                        out.windows(2)
                                            .all(|w| entry_block(w[0]) < entry_block(w[1])),
                                        "entries must be strictly ascending by block"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// The row the daemon sizes for at 294,912 tokens (73,728 candidate
    /// blocks): the checkpoint's 514, and the stratified rows at 128K-token
    /// windows (32,768 blocks — three windows) with an 8,192-token recent span.
    #[test]
    fn the_strides_at_depth() {
        let cand = 73_728;
        assert_eq!(max_entries_for(2048, 4, cand, &Strata::WHOLE), 514);
        assert_eq!(
            max_entries_for(2048, 4, cand, &strata(32_768, 2048, Recent::Candidate)),
            3 * 513 + 1
        );
        assert_eq!(
            max_entries_for(2048, 4, cand, &strata(32_768, 2048, Recent::Forced)),
            3 * 513 + 2048 + 1
        );
    }

    /// The kernel's union buffer at the same depth, and at 1M tokens (262,144
    /// blocks — eight windows), where repeats are counted before the union
    /// removes them. Both round up well inside the kernel's 16,384.
    #[test]
    fn the_gathered_bounds_at_depth() {
        let forced = strata(32_768, 2048, Recent::Forced);
        assert_eq!(max_gathered_for(2048, 4, 73_728, &Strata::WHOLE), 513);
        assert_eq!(max_gathered_for(2048, 4, 73_728, &forced), 3 * 513);
        assert_eq!(max_gathered_for(2048, 4, 262_144, &forced), 8 * 513);
        // A tiny row whose windows all rank the same prompt gathers past its
        // candidates; the union is what brings it back under them.
        let narrow = strata(2, 0, Recent::Candidate);
        assert_eq!(max_gathered_for(8, 4, 6, &narrow), 3 * 3);
        assert_eq!(max_entries_for(8, 4, 6, &narrow), 6 + 1);
    }

    /// Positions become blocks at the ratio, rounding up so a window or span
    /// never covers less than was asked for; a zero window stays zero.
    #[test]
    fn strata_tokens_become_blocks_at_the_ratio() {
        let deployed = StrataTokens {
            window: 131_072,
            recent: 8192,
            mode: Recent::Forced,
            pinned: 0,
        };
        assert_eq!(
            Strata::from_tokens(deployed, 4),
            strata(32_768, 2048, Recent::Forced)
        );
        assert_eq!(
            Strata::from_tokens(StrataTokens::DEFAULT, 4),
            strata(65_536, 2048, Recent::Candidate)
        );
        let ragged = StrataTokens {
            window: 10,
            recent: 5,
            mode: Recent::Candidate,
            pinned: 0,
        };
        assert_eq!(
            Strata::from_tokens(ragged, 4),
            strata(3, 2, Recent::Candidate)
        );
        assert_eq!(Strata::from_tokens(StrataTokens::WHOLE, 4), Strata::WHOLE);
    }
}
