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
//! # The packing
//!
//! One entry is one **run of cells at the bottom of a block**, which is all
//! the selection can produce: `(block << 2) | (cells − 1)`, ascending by
//! block. A whole block is `cells == ratio`. The two-bit cell field is why
//! [`MAX_RATIO`] is 4 — the released checkpoint's `indexer_compress_ratio`,
//! and the widest the packing admits without a second word per entry.

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

/// Build one query's selection, ascending by block, into `out`.
///
/// `scores[b]` is block `b`'s indexer score; only blocks below the query's
/// tail are read, so `scores` may be shorter than the sequence's block count
/// as long as it covers `((qpos+1)/ratio·ratio)/ratio` entries. `out` is
/// cleared first and grows to at most [`max_entries`].
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
pub fn selection_entries(
    scores: &[f32],
    qpos: usize,
    ratio: usize,
    top_k: usize,
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

    // Rank the candidates by (score desc, block asc). The cut needs the first
    // `full + 1` in that order, so a partial sort of the prefix is enough.
    let mut order: Vec<u32> = (0..n_cand as u32).collect();
    let keep = (full + usize::from(rem > 0)).min(n_cand);
    let by_rank = |a: &u32, b: &u32| {
        let (sa, sb) = (scores[*a as usize], scores[*b as usize]);
        sb.partial_cmp(&sa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(b))
    };
    if keep < n_cand {
        order.select_nth_unstable_by(keep, by_rank);
        order.truncate(keep + 1);
    }
    order.sort_unstable_by(by_rank);

    for &b in order.iter().take(full) {
        out.push(pack_entry(b as usize, ratio));
    }
    if rem > 0 {
        if let Some(&b) = order.get(full) {
            out.push(pack_entry(b as usize, rem));
        }
    }
    if n_tail > 0 {
        out.push(pack_entry(tail_start / ratio, n_tail));
    }
    out.sort_unstable();
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

#[cfg(test)]
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
                    if let RowSelection::Entries(n) =
                        selection_entries(&scores, qpos, ratio, top_k, &mut out)
                    {
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
        let sel = selection_entries(&scores, 4096, 4, 2047, &mut out);
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
            let sel = selection_entries(&scores, qpos, ratio, top_k, &mut out);
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
                let sel = selection_entries(&scores, qpos, ratio, top_k, &mut out);
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
            selection_entries(&scores, qpos, ratio, top_k, &mut out),
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
            selection_entries(&scores, width - 1, ratio, top_k, &mut out),
            RowSelection::Dense
        );
        let sel = selection_entries(&scores, width, ratio, top_k, &mut out);
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
            let RowSelection::Entries(_) = selection_entries(&scores, qpos, ratio, top_k, &mut out)
            else {
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
}
