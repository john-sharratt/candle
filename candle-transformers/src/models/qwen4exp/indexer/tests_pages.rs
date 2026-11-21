//! Tests that the ragged page walk resolves positions exactly, and that cuts,
//! splices and compaction keep it so.
//!
//! A position resolves to the same block whether the pages are uniform, ragged,
//! or spliced, so dropping or changing one page renumbers what follows it by
//! construction. Compaction is tested to leave what each buffer holds unchanged.
//! A page may not overlap the one before it.

use super::*;

// Pinned ranges are lists of ranges, and a list of one is the common case.
#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
pub(super) mod tests {
    use super::*;

    use super::super::tests_support::tests::{
        assert_same_walk, caged, cuda, lcg, placed_page, width_page, zero_page,
    };

    /// **A compaction pass moves buffers, never their contents.** Three caches
    /// across several key pages, the middle one dropped to leave holes, plus a
    /// cache holding a resident page shared with a fork of it, then a pass: every
    /// surviving cache reads back exactly the keys, open rows and page rows it
    /// held.
    #[test]
    fn compaction_moves_index_buffers_without_changing_them() -> Result<()> {
        let Some(device) = cuda() else {
            return Ok(());
        };
        let d = 128usize;
        let build = |n: usize, seed: u64| -> Result<(IndexCache, Q8Keys, Vec<f32>)> {
            let keys = Q8Keys::quantize(&lcg(n * d, seed, 1.0), d);
            let open = lcg(3 * d, seed ^ 0x55, 1.0);
            let cache = IndexCache::from_rows(&keys, &open, d, &device)?;
            Ok((cache, keys, open))
        };
        let (mut a, a_keys, a_open) = build(2 * PAGE_BLOCKS + 7, 1)?;
        let hole = build(3 * PAGE_BLOCKS, 2)?;
        let (mut c, c_keys, c_open) = build(PAGE_BLOCKS + 1, 3)?;
        let page_rows = lcg((PAGE_BLOCKS + 9) * d, 4, 1.0);
        let mut p = IndexCache::new(d, &device)?;
        p.push_page(placed_page(&page_rows, 2, d, &device)?, 0, 4)?;
        let mut p_fork = p.fork()?;
        drop(hole);

        let report = compact_index_caches(
            &mut [&mut a, &mut c, &mut p, &mut p_fork],
            &[],
            d,
            &device,
            0,
        )?;
        assert!(report.moved <= report.planned);
        for (cache, keys, open) in [(&a, &a_keys, &a_open), (&c, &c_keys, &c_open)] {
            assert_eq!(&cache.live_rows_host()?, keys);
            assert_eq!(&cache.open_rows_host()?, open);
        }
        for cache in [&p, &p_fork] {
            let (page, _) = cache.page_at(0).expect("the page survives the pass");
            assert_eq!(page.host_rows()?, Q8Keys::quantize(&page_rows, d));
        }
        Ok(())
    }

    /// **A cut keeps the pages below it whole and drops the rest.** Three pages
    /// of 8 rows at ratio 4, the last row covering 2 tokens — 30 tokens each,
    /// abutting at 0, 30 and 60. A cut at 60 keeps two and hands back the third,
    /// and the tail resumes at the cut; a cut inside a page is refused.
    #[test]
    fn a_cut_keeps_whole_pages_below_it() -> Result<()> {
        let Some(device) = cuda() else {
            return Ok(());
        };
        let d = 128usize;
        let mut c = IndexCache::new(d, &device)?;
        for seed in 1..=3 {
            let page = placed_page(&lcg(8 * d, seed, 1.0), 2, d, &device)?;
            let base = c.next_base();
            c.push_page(page, base, 4)?;
        }
        assert_eq!(c.next_base(), 90);

        assert_eq!(c.cut_floor(0), 0);
        assert_eq!(c.cut_floor(3), 0, "3 is inside the page at 0..30");
        assert_eq!(c.cut_floor(30), 30);
        assert_eq!(c.cut_floor(45), 30, "45 is inside the page at 30..60");
        assert_eq!(c.cut_floor(90), 90);
        assert_eq!(c.cut_floor(95), 95, "past the pages, in the tail");

        let dropped = c.truncate_to(60)?;
        assert_eq!(dropped.len(), 1);
        assert_eq!(c.page_count(), 2);
        assert_eq!(c.page_row_span(), 16);
        assert_eq!(c.next_base(), 60);

        assert!(
            c.truncate_to(45).is_err(),
            "45 is inside the page at 30..60"
        );
        assert_eq!(c.page_count(), 2, "a refused cut changes nothing");

        let dropped = c.truncate_to(0)?;
        assert_eq!(dropped.len(), 2);
        assert_eq!(c.page_count(), 0);
        assert_eq!(c.next_base(), 0);
        Ok(())
    }

    /// **An unindexed span moves nothing but where the next page opens.**
    ///
    /// This is the property the whole placement design buys. Positions used to
    /// be the running sum of the page widths, so a piece injected without its
    /// rows did not merely leave a span unindexed — it dragged every later page
    /// earlier by its own width, and every query past it resolved through a
    /// block describing different tokens. Measured live as 328 tokens of section
    /// K/V against zero indexed tokens on all thirteen layers.
    #[test]
    fn an_unindexed_span_leaves_later_pages_where_their_kv_is() {
        const RATIO: usize = 4;
        // 40 tokens of K/V with no rows, then a real 20-token page.
        let mut c = caged(&[], RATIO);
        c.skip_to(40).expect("skip");
        let rows = 20usize.div_ceil(RATIO);
        let page = zero_page(rows, 20 - (rows - 1) * RATIO, 4, &Device::Cpu);
        let base = c.next_base();
        c.push_page(page, base, RATIO).expect("page after the span");

        assert_eq!(
            c.indexed_tokens(RATIO),
            60,
            "the skipped 40 tokens are accounted for even though nothing indexed \
             them — without that the cache claims 20 while the K/V holds 60"
        );
        assert_eq!(
            c.block_start(0, RATIO),
            40,
            "the first REAL row begins where its K/V does, past the span"
        );
        // The span itself offers nothing: a position inside it resolves to the
        // row count before it, which is zero.
        assert_eq!(
            c.candidates_at(20, RATIO),
            0,
            "a query inside the span has no candidates — nothing indexed it"
        );
        // And one inside the real page resolves through THAT page.
        assert_eq!(
            c.candidates_at(43, RATIO),
            1,
            "a query four tokens into the page sees the page's first row"
        );
    }

    /// Moving the tail adds no rows and no candidates — only distance.
    #[test]
    fn skipping_adds_tokens_but_no_rows() {
        const RATIO: usize = 4;
        let mut c = caged(&[12], RATIO);
        let rows_before = c.page_row_span();
        let tokens_before = c.page_token_span();
        c.skip_to(tokens_before + 17).expect("skip");
        assert_eq!(
            c.page_row_span(),
            rows_before,
            "an unindexed span contributes no candidate rows"
        );
        assert_eq!(
            c.page_token_span(),
            tokens_before + 17,
            "but it does move the position the next page opens at"
        );
        // Standing still is a no-op, so a caller may declare unconditionally.
        c.skip_to(tokens_before + 17).expect("no-op skip");
        assert_eq!(c.page_token_span(), tokens_before + 17);
    }

    /// Moving the tail obeys the same ordering rule as a page: injected prefix
    /// state must precede anything the sequence forwarded, or the live tail's
    /// own rows would sit before content that came earlier.
    #[test]
    fn a_skip_is_refused_after_live_rows() {
        const RATIO: usize = 4;
        let mut c = caged(&[8], RATIO);
        c.n_blocks = 1;
        assert!(
            c.skip_to(c.next_base() + 4).is_err(),
            "moving the tail after live rows must be refused, exactly as a page is"
        );
    }

    /// **Pages ascend and may not overlap.** A hole is a span nobody indexed; an
    /// overlap is two rows claiming one token, which no placement can mean.
    #[test]
    fn a_page_may_not_overlap_the_one_before_it() {
        const RATIO: usize = 4;
        let mut c = caged(&[12], RATIO);
        let page = zero_page(2, RATIO, 4, &Device::Cpu);
        assert!(
            c.push_page(page, 4, RATIO).is_err(),
            "a page placed inside the previous page's span was accepted"
        );
    }

    /// **The ragged-page contract, which every thinking-span cut relies on.**
    ///
    /// A page records the tokens its last row covers, and the scorer walks those
    /// widths instead of dividing by `ratio`. These pin that a position resolves
    /// to the same block whether the pages are uniform, ragged, or a mix — the
    /// property that makes dropping a whole page renumber everything downstream
    /// by construction.
    #[test]
    fn ragged_pages_resolve_positions_by_walking_widths() {
        let ratio = 4;
        // Uniform pages: the walk must agree with the plain division.
        let c = caged(&[8, 8], ratio);
        assert_eq!(c.indexed_tokens(ratio), 16);
        for pos in 0..16 {
            assert_eq!(
                c.candidates_at(pos, ratio),
                (pos + 1) / ratio,
                "uniform pages must match the uniform formula at {pos}"
            );
        }

        // Ragged: 5 tokens (rows [4,1]) then 6 (rows [4,2]).
        let c = caged(&[5, 6], ratio);
        assert_eq!(c.indexed_tokens(ratio), 11);
        // Inside page 0: its short last row only counts once the position
        // reaches the page's full span.
        assert_eq!(c.candidates_at(0, ratio), 0);
        assert_eq!(c.candidates_at(3, ratio), 1);
        assert_eq!(
            c.candidates_at(4, ratio),
            2,
            "page 0 complete at its 5th token"
        );
        // Inside page 1, which starts at token 5.
        assert_eq!(c.candidates_at(8, ratio), 3);
        assert_eq!(c.candidates_at(10, ratio), 4, "page 1 complete at token 11");
    }

    /// A cut leaves a SHORT page, and the pages either side of it must still
    /// resolve exactly — this is the shape a thinking turn seals
    /// (`[prefill][reasoning][answer]`, none of them a multiple of `ratio`).
    #[test]
    fn a_short_page_between_two_others_keeps_the_walk_exact() {
        let ratio = 4;
        let c = caged(&[7, 3, 9], ratio);
        assert_eq!(
            c.indexed_tokens(ratio),
            19,
            "the cache spans its pages exactly"
        );

        // Every position must map to a block whose start is at or below it, and
        // the count must be monotone — the two properties the scorer needs.
        let mut prev = 0;
        for pos in 0..19 {
            let n = c.candidates_at(pos, ratio);
            assert!(n >= prev, "candidate count went backwards at {pos}");
            prev = n;
            if n > 0 {
                assert!(
                    c.block_start(n - 1, ratio) <= pos,
                    "block {} starts after the position that counts it",
                    n - 1
                );
            }
        }
        // A page boundary is exactly where the previous page's rows all count.
        assert_eq!(c.candidates_at(6, ratio), 2, "page 0 = rows [4,3]");
        assert_eq!(c.candidates_at(9, ratio), 3, "page 1 = one 3-token row");
    }

    /// **Dropping the reasoning page renumbers by construction.** The whole
    /// design rests on this: a cache built from `[pre][answer]` must resolve
    /// positions exactly as one built from `[pre][reasoning][answer]` does for
    /// the tokens that survive — because the projection compacts the K/V the
    /// same way.
    #[test]
    fn dropping_a_page_compacts_the_walk() {
        let ratio = 4;
        let with = caged(&[7, 3, 9], ratio);
        let without = caged(&[7, 9], ratio);

        assert_eq!(with.indexed_tokens(ratio), 19);
        assert_eq!(
            without.indexed_tokens(ratio),
            16,
            "the 3-token page is gone"
        );

        // The retained pages keep their own row counts, so every position in the
        // compacted cache sees exactly the rows of `[pre] ++ [answer]`.
        let pre_rows = with.candidates_at(6, ratio);
        assert_eq!(without.candidates_at(6, ratio), pre_rows);
        // The answer's last token: all rows of both retained pages.
        assert_eq!(without.candidates_at(15, ratio), without.page_row_span());
    }

    /// **A splice is a build of the same pages.** Replacing the middle page of
    /// `[7 | 3 | 9]` with a 5-token one resolves every position exactly as a
    /// cache built `[7 | 5 | 9]` does: the page after the splice moved with it,
    /// and nothing before it did.
    #[test]
    fn a_page_spliced_into_the_middle_walks_as_if_built_there() {
        const RATIO: usize = 4;
        let mut c = caged(&[7, 3, 9], RATIO);
        let taken = c
            .splice_pages(7, 10, vec![width_page(5, RATIO)], RATIO)
            .expect("splice");
        assert_eq!(taken.len(), 1, "the 3-token page came out");
        assert_same_walk(&c, &caged(&[7, 5, 9], RATIO), RATIO);
    }

    /// Evicting, inserting at the front and inserting at the end are the same
    /// operation, and each walks as the pages built in that order do.
    #[test]
    fn a_splice_evicts_and_inserts_anywhere() {
        const RATIO: usize = 4;
        let mut c = caged(&[7, 3, 9], RATIO);
        c.splice_pages(7, 10, Vec::new(), RATIO).expect("evict");
        assert_same_walk(&c, &caged(&[7, 9], RATIO), RATIO);

        c.splice_pages(0, 0, vec![width_page(6, RATIO)], RATIO)
            .expect("insert at the front");
        assert_same_walk(&c, &caged(&[6, 7, 9], RATIO), RATIO);

        c.splice_pages(22, 22, vec![width_page(2, RATIO)], RATIO)
            .expect("insert at the end");
        assert_same_walk(&c, &caged(&[6, 7, 9, 2], RATIO), RATIO);
    }

    /// A span inserted with no rows moves the pages after it and indexes
    /// nothing — the splice's version of [`IndexCache::skip_to`].
    #[test]
    fn a_spliced_gap_moves_what_follows_and_indexes_nothing() {
        const RATIO: usize = 4;
        let mut c = caged(&[7, 9], RATIO);
        c.splice_pages(7, 7, vec![SplicePiece::Gap(5)], RATIO)
            .expect("gap");
        assert_eq!(c.next_base(), 21);
        assert_eq!(c.page_row_span(), caged(&[7, 9], RATIO).page_row_span());
        let first_of_second = caged(&[7], RATIO).page_row_span();
        assert_eq!(
            c.block_start(first_of_second, RATIO),
            12,
            "the second page starts past the gap, where its K/V does"
        );
    }

    /// A splice that would cut a page, reach past the placed tokens, or follow
    /// live rows is refused, and changes nothing.
    #[test]
    fn a_splice_off_a_page_boundary_is_refused() {
        const RATIO: usize = 4;
        let mut c = caged(&[7, 3, 9], RATIO);
        assert!(
            c.splice_pages(3, 10, Vec::new(), RATIO).is_err(),
            "3 is inside 0..7"
        );
        assert!(
            c.splice_pages(7, 12, Vec::new(), RATIO).is_err(),
            "12 is inside 10..19"
        );
        assert!(
            c.splice_pages(19, 25, Vec::new(), RATIO).is_err(),
            "past the tail"
        );
        assert_same_walk(&c, &caged(&[7, 3, 9], RATIO), RATIO);

        c.n_blocks = 1;
        assert!(
            c.splice_pages(7, 10, Vec::new(), RATIO).is_err(),
            "a splice after live rows was accepted"
        );
    }

    /// The prompt end moves with the pages after a splice before it, and is
    /// left alone by one after it.
    #[test]
    fn the_prompt_end_moves_with_the_pages_it_follows() {
        const RATIO: usize = 4;
        let mut c = caged(&[7, 3, 9], RATIO);
        c.set_prompt_end(10);
        c.set_pins(&[8..12]);
        c.splice_pages(0, 7, vec![width_page(2, RATIO)], RATIO)
            .expect("splice before the prompt end");
        assert_eq!(c.prompt_end(), 5);
        assert_eq!(c.pins(), &[3..7], "a pin past the splice moves with it");
        c.splice_pages(5, 14, Vec::new(), RATIO)
            .expect("splice after it");
        assert_eq!(c.prompt_end(), 5);
        assert_eq!(
            c.pins(),
            &[3..5],
            "a pin end inside the splice closes at it"
        );

        // A pin that starts inside a splice and ends past it keeps only its
        // part past the splice — never the content put in its place.
        let mut c = caged(&[7, 3, 9], RATIO);
        c.set_pins(&[8..15]);
        c.splice_pages(7, 10, vec![width_page(30, RATIO)], RATIO)
            .expect("splice the middle page");
        // [7, 10) became [7, 37): positions from 10 on move by +27.
        assert_eq!(c.pins(), &[37..42]);

        // A pin exactly over the spliced span goes; one ending at the
        // splice's start or starting at its end keeps clear of the new
        // content.
        let mut c = caged(&[7, 3, 9], RATIO);
        c.set_pins(&[0..7, 7..10, 10..19]);
        c.splice_pages(7, 10, vec![width_page(30, RATIO)], RATIO)
            .expect("splice the middle page");
        assert_eq!(c.pins(), &[0..7, 37..46]);
    }

    /// An empty cache, a single page, and a page narrower than `ratio` — the
    /// degenerate shapes a suppressed turn or a one-token answer produces.
    #[test]
    fn degenerate_page_shapes_are_expressible() {
        let ratio = 4;
        let empty = caged(&[], ratio);
        assert_eq!(empty.indexed_tokens(ratio), 0);
        assert_eq!(empty.candidates_at(0, ratio), 0);

        // A single 1-token page — a collapsed `<think></think>`.
        let one = caged(&[1], ratio);
        assert_eq!(one.indexed_tokens(ratio), 1);
        assert_eq!(one.candidates_at(0, ratio), 1, "its only row is complete");

        // Exactly one full block.
        let full = caged(&[4], ratio);
        assert_eq!(full.indexed_tokens(ratio), 4);
        assert_eq!(full.candidates_at(2, ratio), 0);
        assert_eq!(full.candidates_at(3, ratio), 1);
    }

    /// The push guard, from the CUT's side: `close_tail_into_page` resets the
    /// tail before pushing precisely because a page may not land after live
    /// rows. This pins both halves — refused with a tail, accepted without —
    /// so the reset in the cut cannot be dropped as redundant.
    #[test]
    fn a_page_is_refused_after_live_rows_and_accepted_once_the_tail_is_reset() {
        let ratio = 4;
        let d = 4usize;
        let rows = Q8Keys::quantize(&vec![0f32; 2 * d], d);

        let mut forwarded = IndexCache::from_rows(&rows, &[], d, &Device::Cpu).unwrap();
        assert_eq!(forwarded.live_blocks(), 2);
        let base = forwarded.next_base();
        assert!(
            forwarded
                .push_page(zero_page(1, 1, d, &Device::Cpu), base, ratio)
                .is_err(),
            "a page landing after live rows would sit at the wrong positions"
        );

        // A cache whose tail is empty — what the cut leaves behind — accepts it.
        let mut c = caged(&[8], ratio);
        let base = c.next_base();
        assert!(c
            .push_page(zero_page(1, 1, d, &Device::Cpu), base, ratio)
            .is_ok());
        assert_eq!(c.indexed_tokens(ratio), 9);
    }

    /// `IndexCache` keeps its pins as positions and moves them with the page
    /// layout: merged when set, clipped by a cut, shifted by a splice past
    /// them, closed by one inside them, and dropped by a reset.
    #[test]
    fn pins_travel_with_the_cache() -> Result<()> {
        // Fifteen complete blocks at ratio 4: 60 positions.
        let mut c = IndexCache::from_rows(&Q8Keys::quantize(&[0.0; 30], 2), &[], 2, &Device::Cpu)?;
        c.set_pins(&[40..60, 0..10, 8..20]);
        assert_eq!(c.pins(), &[0..20, 40..60]);
        // Uniform blocks of 4: [0, 20) is blocks 0..5; [40, 60) blocks 10..15.
        assert_eq!(c.pin_row(4, 100).ranges(), vec![0..5, 10..15]);
        // Held to a bound, the last range gives up its front first.
        assert_eq!(c.pin_row(4, 7).ranges(), vec![0..5, 13..15]);
        let fork = c.fork()?;
        assert_eq!(fork.pins(), c.pins());
        c.truncate_to(50)?;
        assert_eq!(c.pins(), &[0..20, 40..50]);
        c.truncate_to(10)?;
        assert_eq!(c.pins(), &[0..10]);
        c.reset();
        assert!(c.pins().is_empty());
        Ok(())
    }
}
