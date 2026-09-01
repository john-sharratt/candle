//! Lay several turns into ONE prefill grid so each can be sealed to its own
//! block range.
//!
//! # Why
//!
//! Calibration builds a provenance exemplar per case by prefilling it and
//! sealing the result, which captures the turn's per-token `sign(Q)` window.
//! One case per forward puts the model deep in the launch-overhead regime:
//! measured 353 tokens across 2.4 sequences in a 1,095 ms forward — 263 t/s
//! against a >1000 t/s target — while a backlog of over a thousand tokens sat
//! queued. The forward costs about the same whether it carries 350 tokens or
//! 3,500, so the fix is to put many cases in one.
//!
//! # The block-alignment rule, which is the whole of this module
//!
//! A seal captures a **block** range, and the gallery window it gathers is the
//! `sign(Q)` of every token in those blocks. Blocks are [`CHUNK_SIZE`] tokens.
//! So if two cases share a block, the first case's window contains up to
//! `CHUNK_SIZE - 1` tokens belonging to the second — a question about file
//! permissions carrying the tail of a question about timezones, scored as
//! though it were one exemplar. Nothing catches that: the corpus is simply,
//! quietly worse.
//!
//! Hence: **every case starts on a block boundary**, which means every case's
//! length is rounded up to a multiple of `CHUNK_SIZE` with padding tokens.
//!
//! # Where the padding goes
//!
//! Into the **assistant** half, never the user half. These cases are
//! question exemplars — a user turn followed by an empty assistant turn — and
//! the region a live probe resembles is the user's. Padding the assistant body
//! leaves the user span byte-identical to what a lone prefill would produce, so
//! an analysis narrowed to the user region (`substrate_inspect --probe-phase
//! user`) cannot see the padding at all, and a whole-window scan sees it only as
//! a short inert tail.
//!
//! Padding is applied as repeated token **ids**, not text: the grid is prefilled
//! from token ids directly, so appending `k` copies of one id adds exactly `k`
//! tokens. Padding with characters would not — BPE merges runs of whitespace, so
//! `k` newline characters is not `k` tokens and the alignment would drift.

use candle_nn::CHUNK_SIZE;

/// One case to place in a stuffed grid.
///
/// `tokens` is the case's own complete grid — exactly what a lone prefill of
/// this case would lay down — and the two content bounds index into it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaseGrid {
    /// The case's tokens, in order.
    pub tokens: Vec<u32>,
    /// End of the user content within `tokens`, exclusive.
    pub user_content_end: u32,
    /// Start of the assistant content within `tokens`.
    pub assistant_content_start: u32,
}

/// Where one case landed in the stuffed grid.
///
/// `token_start` and the block range are relative to the GRID; the seal rebases
/// them onto the slot, which opens with the projection's system prompt (see
/// `Scheduler::perform_carved_turn_seals`).
///
/// The content bounds stay **relative to the case**, because that is how a turn
/// records them and how `turn_layout::phase_span_of` reads them back — a span
/// indexes the turn's own signature window, not the grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CarvedRegion {
    /// First token of this case in the grid. Always block-aligned.
    pub token_start: usize,
    /// The case's real tokens, excluding padding.
    pub token_len: usize,
    /// Padding tokens appended after the real ones. Always `< CHUNK_SIZE`.
    pub pad: usize,
    /// First block of the seal range.
    pub block_from: usize,
    /// One past the last block of the seal range.
    pub block_to: usize,
    /// End of the user content, relative to `token_start`.
    pub user_content_end: u32,
    /// Start of the assistant content, relative to `token_start`.
    pub assistant_content_start: u32,
}

impl CarvedRegion {
    /// The case's real tokens within the grid, excluding padding.
    pub fn token_range(&self) -> std::ops::Range<usize> {
        self.token_start..self.token_start + self.token_len
    }
}

/// A stuffed grid: the tokens to prefill, and where each case landed in it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StuffedGrid {
    /// The tokens to prefill, in one forward.
    pub tokens: Vec<u32>,
    /// One region per input case, in the same order.
    pub regions: Vec<CarvedRegion>,
}

/// Padding needed to round `len` up to a whole number of blocks.
///
/// Zero when `len` is already a multiple — the modulo on the outside is what
/// makes that true, and without it an exactly-aligned case is handed a whole
/// wasted block of padding.
fn pad_to_block(len: usize) -> usize {
    (CHUNK_SIZE - len % CHUNK_SIZE) % CHUNK_SIZE
}

/// Lay `cases` end to end, each starting on a block boundary.
///
/// `pad_token` fills the gap between a case's real tokens and its next block
/// boundary; it should be an inert id (a newline or the dialect's turn
/// terminator), and it lands in the assistant half — see the module docs.
///
/// Cases with no tokens are **skipped entirely** rather than given a region: a
/// zero-token case would otherwise claim an empty block range, and a seal over
/// an empty range persists a turn with no K/V and an empty gallery window. The
/// returned `regions` therefore lines up with the *non-empty* inputs; callers
/// pairing regions back to cases must filter the same way, which
/// [`plan_stuffed_grid_with_indices`] does for them.
pub fn plan_stuffed_grid(cases: &[CaseGrid], pad_token: u32) -> StuffedGrid {
    let (grid, _) = plan_stuffed_grid_with_indices(cases, pad_token);
    grid
}

/// As [`plan_stuffed_grid`], also returning each region's index in `cases`.
///
/// The extra vector exists because empty cases are dropped, so `regions[i]` is
/// not necessarily `cases[i]`. Losing that correspondence would tag an exemplar
/// with the wrong tool — the single worst failure this path can produce, since
/// it corrupts the corpus in a way that looks like a routing bug forever after.
pub fn plan_stuffed_grid_with_indices(
    cases: &[CaseGrid],
    pad_token: u32,
) -> (StuffedGrid, Vec<usize>) {
    let mut tokens: Vec<u32> = Vec::new();
    let mut regions: Vec<CarvedRegion> = Vec::new();
    let mut sources: Vec<usize> = Vec::new();

    for (i, case) in cases.iter().enumerate() {
        if case.tokens.is_empty() {
            continue;
        }
        let token_start = tokens.len();
        debug_assert_eq!(
            token_start % CHUNK_SIZE,
            0,
            "every case starts on a block boundary by construction",
        );
        let token_len = case.tokens.len();
        let pad = pad_to_block(token_len);

        tokens.extend_from_slice(&case.tokens);
        tokens.extend(std::iter::repeat_n(pad_token, pad));

        // Clamp the content bounds into the case and keep them ordered. A
        // tokenizer that merges across the user/assistant join can otherwise
        // report an assistant start below the user end, which would invert the
        // span and make the phase lens read a backwards range.
        let user_content_end = (case.user_content_end as usize).min(token_len) as u32;
        let assistant_content_start = (case.assistant_content_start as usize)
            .min(token_len)
            .max(user_content_end as usize) as u32;

        regions.push(CarvedRegion {
            token_start,
            token_len,
            pad,
            block_from: token_start / CHUNK_SIZE,
            block_to: (token_start + token_len + pad) / CHUNK_SIZE,
            user_content_end,
            assistant_content_start,
        });
        sources.push(i);
    }

    (StuffedGrid { tokens, regions }, sources)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A case of `len` tokens whose user half is the first `user` of them.
    fn case(len: usize, user: u32) -> CaseGrid {
        CaseGrid {
            // Distinct ids per case so a mis-sliced region is visible as the
            // wrong VALUES, not merely the wrong length.
            tokens: (0..len as u32).map(|i| 1000 + i).collect(),
            user_content_end: user,
            assistant_content_start: user,
        }
    }

    /// Ids unique per case, so a region that reads its neighbour's tokens is
    /// caught by value rather than by length.
    fn tagged_case(tag: u32, len: usize, user: u32) -> CaseGrid {
        CaseGrid {
            tokens: (0..len as u32).map(|i| tag * 10_000 + i).collect(),
            user_content_end: user,
            assistant_content_start: user,
        }
    }

    #[test]
    fn padding_rounds_up_and_is_zero_when_already_aligned() {
        assert_eq!(pad_to_block(0), 0);
        assert_eq!(pad_to_block(1), CHUNK_SIZE - 1);
        assert_eq!(pad_to_block(CHUNK_SIZE - 1), 1);
        // **The off-by-one that matters.** `CHUNK_SIZE - len % CHUNK_SIZE`
        // without the outer modulo returns a whole wasted block here.
        assert_eq!(pad_to_block(CHUNK_SIZE), 0);
        assert_eq!(pad_to_block(CHUNK_SIZE * 3), 0);
        assert_eq!(pad_to_block(CHUNK_SIZE * 3 + 1), CHUNK_SIZE - 1);
    }

    #[test]
    fn a_single_case_starts_at_zero_and_is_padded_to_a_whole_block() {
        let g = plan_stuffed_grid(&[case(40, 20)], 7);
        assert_eq!(g.regions.len(), 1);
        let r = g.regions[0];
        assert_eq!(r.token_start, 0);
        assert_eq!(r.token_len, 40);
        assert_eq!(r.pad, CHUNK_SIZE * 2 - 40);
        assert_eq!((r.block_from, r.block_to), (0, 2));
        assert_eq!(g.tokens.len(), CHUNK_SIZE * 2);
        // The real tokens are intact and the pad is exactly the tail.
        assert_eq!(&g.tokens[..40], &case(40, 20).tokens[..]);
        assert!(g.tokens[40..].iter().all(|&t| t == 7));
    }

    /// **The invariant the whole module exists for.** Regions must partition
    /// the grid's blocks: no block in two windows, none left out. A shared
    /// block means one exemplar's `sign(Q)` window carries another's tokens.
    #[test]
    fn regions_partition_the_blocks_with_no_gap_and_no_overlap() {
        let cases = vec![
            case(1, 1),
            case(CHUNK_SIZE, 10),
            case(CHUNK_SIZE + 1, 10),
            case(100, 50),
            case(CHUNK_SIZE * 4, 3),
        ];
        let g = plan_stuffed_grid(&cases, 0);

        let mut expected_next_block = 0usize;
        let mut covered: Vec<usize> = Vec::new();
        for r in &g.regions {
            assert_eq!(
                r.block_from, expected_next_block,
                "each region must begin where the previous one ended",
            );
            assert!(r.block_to > r.block_from, "no region may be empty");
            covered.extend(r.block_from..r.block_to);
            expected_next_block = r.block_to;
        }
        let total_blocks = g.tokens.len() / CHUNK_SIZE;
        assert_eq!(
            expected_next_block, total_blocks,
            "the grid is fully carved"
        );
        assert_eq!(
            covered,
            (0..total_blocks).collect::<Vec<_>>(),
            "every block exactly once, in order",
        );
    }

    /// Every region must be block-aligned and its block range must agree with
    /// its token range. These are two derivations of the same fact and the seal
    /// trusts the blocks, so a disagreement seals the wrong tokens.
    #[test]
    fn block_range_and_token_range_agree_for_every_region() {
        let cases = vec![
            case(1, 1),
            case(31, 5),
            case(32, 5),
            case(33, 5),
            case(77, 9),
        ];
        let g = plan_stuffed_grid(&cases, 0);
        for r in &g.regions {
            assert_eq!(r.token_start % CHUNK_SIZE, 0, "region start is aligned");
            assert_eq!(r.block_from, r.token_start / CHUNK_SIZE);
            let padded = r.token_len + r.pad;
            assert_eq!(padded % CHUNK_SIZE, 0, "region is whole");
            assert_eq!(
                r.block_to - r.block_from,
                padded / CHUNK_SIZE,
                "block count must equal the padded token count in blocks",
            );
            assert!(r.pad < CHUNK_SIZE, "padding never adds a whole block");
        }
    }

    /// A region must read back its OWN tokens. Slicing the grid by a region's
    /// token range has to return exactly the case that produced it — the check
    /// that catches an off-by-one that a length-only assertion would pass.
    #[test]
    fn slicing_the_grid_by_a_region_returns_that_cases_own_tokens() {
        let cases = vec![
            tagged_case(1, 5, 2),
            tagged_case(2, CHUNK_SIZE, 8),
            tagged_case(3, 70, 40),
            tagged_case(4, 33, 1),
        ];
        let g = plan_stuffed_grid(&cases, 999);
        assert_eq!(g.regions.len(), cases.len());
        for (r, c) in g.regions.iter().zip(&cases) {
            assert_eq!(
                &g.tokens[r.token_range()],
                &c.tokens[..],
                "region {r:?} must slice back to its own case",
            );
            // And the padding that follows it is padding, not the neighbour.
            let pad_span = r.token_start + r.token_len..r.token_start + r.token_len + r.pad;
            assert!(
                g.tokens[pad_span].iter().all(|&t| t == 999),
                "the gap after a case must be padding, never the next case",
            );
        }
    }

    /// The seal reads BLOCKS, so the block range must contain every real token
    /// of its case and no real token of any other. This is the property stated
    /// in the block granularity the seal actually uses.
    #[test]
    fn a_regions_blocks_contain_all_of_its_own_tokens_and_no_others() {
        let cases = vec![
            tagged_case(1, 40, 20),
            tagged_case(2, 10, 5),
            tagged_case(3, CHUNK_SIZE * 2, 12),
        ];
        let g = plan_stuffed_grid(&cases, 999);
        for (i, r) in g.regions.iter().enumerate() {
            let block_span = r.block_from * CHUNK_SIZE..r.block_to * CHUNK_SIZE;
            // Contains all of its own.
            assert!(block_span.start <= r.token_start);
            assert!(block_span.end >= r.token_start + r.token_len);
            // And nothing real from a neighbour: every token in the block span
            // is either this case's or padding.
            let tag = (i as u32 + 1) * 10_000;
            for &t in &g.tokens[block_span] {
                assert!(
                    t == 999 || (tag..tag + 10_000).contains(&t),
                    "block range of region {i} contains a foreign token {t}",
                );
            }
        }
    }

    /// An empty case claims no blocks at all. A zero-length region would seal an
    /// empty range — a turn with no K/V and an empty gallery window — and would
    /// also break the partition, since `block_to == block_from`.
    #[test]
    fn empty_cases_are_dropped_and_do_not_shift_the_others() {
        let cases = vec![
            tagged_case(1, 10, 5),
            CaseGrid {
                tokens: Vec::new(),
                user_content_end: 0,
                assistant_content_start: 0,
            },
            tagged_case(3, 20, 5),
        ];
        let (g, sources) = plan_stuffed_grid_with_indices(&cases, 0);
        assert_eq!(g.regions.len(), 2, "the empty case claims no region");
        assert_eq!(
            sources,
            vec![0, 2],
            "regions must name the cases they came from, or an exemplar gets \
             tagged with the wrong tool",
        );
        assert_eq!(&g.tokens[g.regions[0].token_range()], &cases[0].tokens[..]);
        assert_eq!(&g.tokens[g.regions[1].token_range()], &cases[2].tokens[..]);
    }

    #[test]
    fn no_cases_yields_an_empty_grid() {
        let g = plan_stuffed_grid(&[], 0);
        assert!(g.tokens.is_empty());
        assert!(g.regions.is_empty());
    }

    /// Content bounds stay relative to the case, never rebased into the grid:
    /// a turn records them against its own token run, and `phase_span_of` reads
    /// them back against the turn's own signature window.
    #[test]
    fn content_bounds_are_relative_to_the_case_not_the_grid() {
        let cases = vec![tagged_case(40, 40, 15), tagged_case(50, 50, 22)];
        let g = plan_stuffed_grid(&cases, 0);
        assert_eq!(g.regions[0].user_content_end, 15);
        assert_eq!(g.regions[1].user_content_end, 22);
        // The second case starts well into the grid, and its bound did NOT
        // absorb that offset.
        assert!(g.regions[1].token_start >= CHUNK_SIZE);
        assert!((g.regions[1].user_content_end as usize) < g.regions[1].token_len);
    }

    /// Bounds past the end of a case are clamped, and an assistant start below
    /// the user end is lifted to meet it — an inverted span would make the
    /// phase lens read a backwards range.
    #[test]
    fn out_of_range_and_inverted_content_bounds_are_repaired() {
        let cases = vec![
            CaseGrid {
                tokens: vec![1; 10],
                user_content_end: 99,
                assistant_content_start: 99,
            },
            CaseGrid {
                tokens: vec![2; 10],
                user_content_end: 8,
                assistant_content_start: 3,
            },
        ];
        let g = plan_stuffed_grid(&cases, 0);
        assert_eq!(g.regions[0].user_content_end, 10, "clamped to the case");
        assert_eq!(g.regions[0].assistant_content_start, 10);
        assert_eq!(g.regions[1].user_content_end, 8);
        assert_eq!(
            g.regions[1].assistant_content_start, 8,
            "an assistant start below the user end is lifted, never inverted",
        );
    }

    /// The grid is exactly the concatenation of the padded regions — no token
    /// unaccounted for at either end.
    #[test]
    fn the_grid_length_is_the_sum_of_the_padded_regions() {
        let cases = vec![case(7, 1), case(64, 1), case(65, 1), case(1, 1)];
        let g = plan_stuffed_grid(&cases, 0);
        let summed: usize = g.regions.iter().map(|r| r.token_len + r.pad).sum();
        assert_eq!(summed, g.tokens.len());
        assert_eq!(g.tokens.len() % CHUNK_SIZE, 0, "the grid is whole blocks");
    }

    /// A stuffed group must lay down exactly what the same cases would produce
    /// one at a time — the real-token content is identical and only the
    /// alignment padding is new. If this drifts, exemplars built by stuffing
    /// stop being comparable to exemplars built the old way.
    #[test]
    fn stuffing_preserves_each_cases_tokens_exactly() {
        let cases = vec![tagged_case(1, 13, 6), tagged_case(2, 91, 44)];
        let g = plan_stuffed_grid(&cases, 0);
        for (r, c) in g.regions.iter().zip(&cases) {
            let got: Vec<u32> = g.tokens[r.token_range()].to_vec();
            assert_eq!(got, c.tokens, "a stuffed case must be byte-identical");
            assert_eq!(r.token_len, c.tokens.len());
        }
    }

    /// **The cross-module property, and the one most likely to rot.** A region's
    /// `sign(Q)` window is its BLOCK span — real tokens *plus* padding — while
    /// its layout spans are measured against the real tokens alone. The two must
    /// agree, or the phase lens reads the wrong region of the wrong window.
    ///
    /// Specifically: the user span must land inside the real tokens, and every
    /// padding token must fall outside every phase span. That is what makes the
    /// claim in this module's docs true — that a lensed scan cannot see the
    /// padding at all.
    #[test]
    fn phase_spans_land_on_real_tokens_and_never_on_padding() {
        use crate::normalization::Phase;
        use crate::turn_layout::{phase_span_of, TurnLayout};

        // A case whose length is deliberately NOT a multiple of the block size,
        // so it carries real padding.
        let head_len = 3u32;
        let case_len = 45usize;
        let user_end = 30u32;
        let assistant_start = 34u32;
        let trailing = 2u32;
        let cases = vec![CaseGrid {
            tokens: (0..case_len as u32).collect(),
            user_content_end: user_end,
            assistant_content_start: assistant_start,
        }];
        let g = plan_stuffed_grid(&cases, 4242);
        let r = g.regions[0];
        assert!(r.pad > 0, "the fixture must actually exercise padding");

        let layout = TurnLayout::from_flat_grid_with_tail(
            head_len,
            r.user_content_end,
            r.assistant_content_start,
            r.token_len as u32,
            2,
            2,
            trailing,
            "q".to_string(),
            Some(String::new()),
            false,
        );

        let user = phase_span_of(&layout.segments, Phase::User)
            .expect("a question exemplar has a user span");
        assert_eq!(
            user,
            head_len as usize..user_end as usize,
            "the user span is the question's own tokens, past the baked opener",
        );

        // The window the gallery actually holds is the PADDED span, because the
        // seal captures whole blocks. Every phase span must sit inside the real
        // tokens of that window, leaving the padded tail untouched.
        let window_len = r.token_len + r.pad;
        for phase in [Phase::User, Phase::Thinking, Phase::Response] {
            let Some(span) = phase_span_of(&layout.segments, phase) else {
                continue;
            };
            assert!(
                span.end <= r.token_len,
                "{phase:?} span {span:?} runs into the padding — the window is \
                 {window_len} tokens but only the first {} are real",
                r.token_len,
            );
        }
    }

    /// Scale check: a realistic group carves cleanly and the padding overhead
    /// stays small enough that the batching win is not eaten by it.
    #[test]
    fn a_realistic_group_carves_cleanly_with_modest_padding() {
        // 16 question cases of the length the corpus actually produces.
        let lens = [
            48, 61, 55, 72, 39, 88, 51, 64, 43, 77, 58, 69, 45, 82, 53, 66,
        ];
        let cases: Vec<CaseGrid> = lens
            .iter()
            .enumerate()
            .map(|(i, &l)| tagged_case(i as u32 + 1, l, (l / 2) as u32))
            .collect();
        let g = plan_stuffed_grid(&cases, 0);

        assert_eq!(g.regions.len(), 16);
        let real: usize = g.regions.iter().map(|r| r.token_len).sum();
        let padding: usize = g.regions.iter().map(|r| r.pad).sum();
        assert_eq!(real + padding, g.tokens.len());
        assert!(
            padding * 100 / real < 35,
            "padding overhead {padding} on {real} real tokens is too high to be \
             worth the batching",
        );
        // Still a partition, at scale.
        let mut next = 0;
        for r in &g.regions {
            assert_eq!(r.block_from, next);
            next = r.block_to;
        }
        assert_eq!(next, g.tokens.len() / CHUNK_SIZE);
    }
}
