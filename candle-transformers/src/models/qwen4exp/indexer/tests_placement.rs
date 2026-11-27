//! Tests that an injected page selects what the sequence that forwarded its
//! tokens selects.
//!
//! A page is scored where it is placed, not where it was made. The test forwards
//! the same tokens at two offsets and checks that an injected page selects as the
//! forwarded one does. Forks share their parent's pages, and a cache holding only
//! injected pages still forks.

use super::*;

// Pinned ranges are lists of ranges, and a list of one is the common case.
#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
pub(super) mod tests {
    use super::*;

    use super::super::tests_support::tests::{cuda, lcg, placed_page, zero_page, Rig};

    /// **A cache populated only by INJECTION still forks.**
    ///
    /// The three carried classes are populated by different things: the
    /// recurrent store and the PLE state come from running a wave, the index
    /// also from a projection installing pages into a slot that has never
    /// decoded a token. `fork_recurrent` used to return early when the parent
    /// had no recurrent store — true reasoning for the recurrence, false for the
    /// index, and it took all three out together.
    ///
    /// That is exactly the base conversation's shape: an Arc-injected prefix of
    /// sections, never decoded. Every conversation forked from it got the base's
    /// K/V and none of the index describing it, which is silent until the prefix
    /// passes the QSA identity threshold and then fails every first turn.
    #[test]
    fn a_cache_holding_only_injected_pages_forks_them() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let rig = Rig::new(device.clone(), 128)?;
        let ratio = rig.ratio;

        // Injected only: pages pushed, nothing ever appended — no `n_blocks`,
        // no open rows, which is what "has not run a wave" looks like here.
        let mut injected = IndexCache::new(rig.cfg.head_dim, &device)?;
        let mut at = 0usize;
        for &w in &[16usize, 12, 20] {
            let rows = w.div_ceil(ratio);
            let page = zero_page(rows, w - (rows - 1) * ratio, rig.cfg.head_dim, &device);
            injected.push_page(page, at, ratio)?;
            at += w;
        }
        assert_eq!(injected.live_blocks(), 0, "the fixture forwarded something");
        assert_eq!(injected.indexed_tokens(ratio), 48);

        let child = injected.fork()?;
        assert_eq!(
            child.indexed_tokens(ratio),
            injected.indexed_tokens(ratio),
            "a fork of an injection-only cache lost its pages — the child would \
             hold the parent's borrowed K/V and no index describing it"
        );
        assert_eq!(child.page_count(), injected.page_count());
        Ok(())
    }

    /// **Placing pages together is placing them one by one.**
    ///
    /// A projection places every layer's page of a piece in one launch; a single
    /// page is placed alone. The rows each page holds — and so what it selects —
    /// must not depend on which pages shared its launch.
    #[test]
    fn pages_placed_together_select_the_same_as_pages_placed_alone() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let rig = Rig::new(device.clone(), 256)?;
        let (ratio, d) = (rig.ratio, rig.cfg.head_dim);
        let widths = [12usize, 8, 20, 4];

        let mut made = Vec::new();
        for (i, &w) in widths.iter().enumerate() {
            let rows = w.div_ceil(ratio);
            made.push((lcg(rows * d, 0xA0 + i as u64, 1.0), w - (rows - 1) * ratio));
        }
        let stored: Vec<(Q8Keys, usize)> = made
            .iter()
            .map(|(r, l)| (Q8Keys::quantize(r, d), *l))
            .collect();
        let host: Vec<(&Q8Keys, usize)> = stored.iter().map(|(k, l)| (k, *l)).collect();
        let together = ResidentPage::place_host(&host, d, &device)?;
        let alone = made
            .iter()
            .map(|(r, l)| placed_page(r, *l, d, &device))
            .collect::<Result<Vec<_>>>()?;

        let mut batched = IndexCache::new(d, &device)?;
        let mut single = IndexCache::new(d, &device)?;
        let mut at = 0usize;
        for ((a, b), &w) in together.into_iter().zip(alone).zip(widths.iter()) {
            batched.push_page(a, at, ratio)?;
            single.push_page(b, at, ratio)?;
            at += w;
        }
        let qpos: Vec<usize> = (0..at).collect();
        assert_eq!(
            rig.select(&mut batched, &qpos)?,
            rig.select(&mut single, &qpos)?,
            "the same pages selected differently depending on which launch placed them"
        );
        Ok(())
    }

    /// **A fork shares its parent's pages and selects what the parent selects.**
    ///
    /// Every view carve forks — a `repo_map` ingest carves one per directory —
    /// so a fork must score with no step of its own between the carve and the
    /// forward: it holds the parent's resident pages, not copies of them.
    #[test]
    fn a_forked_cache_shares_its_parents_pages_and_selects_the_same() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let rig = Rig::new(device.clone(), 256)?;
        let ratio = rig.ratio;

        // A parent holding one closed page plus a live tail — the shape a view
        // is carved from mid-conversation.
        let mut parent = IndexCache::new(rig.cfg.head_dim, &device)?;
        rig.append(&mut parent, 0, 40)?;
        parent.close_tail_into_page(&rig.w, ratio, rig.eps)?;
        rig.append(&mut parent, 40, 24)?;
        assert!(parent.has_pages(), "the fixture carved no page to inherit");

        let mut child = parent.fork()?;
        assert_eq!(child.page_count(), parent.page_count());
        for i in 0..parent.page_count() {
            let (a, _) = parent.page_at(i).expect("parent page");
            let (b, _) = child.page_at(i).expect("child page");
            assert!(
                Arc::ptr_eq(a, b),
                "page {i} of the fork is a copy — every carve would duplicate the \
                 parent's whole index"
            );
        }

        let qpos: Vec<usize> = (0..64).step_by(7).collect();
        assert_eq!(
            rig.select(&mut child, &qpos)?,
            rig.select(&mut parent, &qpos)?,
            "a fork selected differently than the parent it was carved from"
        );
        Ok(())
    }

    /// **The gate the placement design exists for: a page scores where it is
    /// PUT, not where it was made.**
    ///
    /// Forward the same tokens at two different offsets. Seal the first into a
    /// position-free page and inject it at the second's offset; it must select
    /// exactly what the sequence that actually forwarded those tokens there
    /// selects.
    ///
    /// A page scored at the wrong relative distance is invisible below the
    /// identity threshold (`selection_engages`) and, above it, a plausible
    /// score for the wrong block — so this is gated directly.
    #[test]
    fn a_sealed_page_selects_the_same_wherever_it_is_placed() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let rig = Rig::new(device.clone(), 512)?;
        let ratio = rig.ratio;
        // A whole number of blocks, so the pooling is identical on both sides —
        // a short trailing block is a different summary by design, and this
        // gate is about position, not about raggedness.
        let (made_at, placed_at, tokens) = (64usize, 320usize, 48usize);

        // The page, made at one offset. Its rows are un-rotated, so they carry
        // no trace of it.
        let mut origin = IndexCache::new(rig.cfg.head_dim, &device)?;
        origin.skip_to(made_at)?;
        rig.append(&mut origin, 0, tokens)?;
        let page = ResidentPage::place_host(
            &[(&origin.live_rows_host()?, ratio)],
            rig.cfg.head_dim,
            &device,
        )?
        .pop()
        .expect("one page placed");

        // Injected at a different offset.
        let mut injected = IndexCache::new(rig.cfg.head_dim, &device)?;
        injected.skip_to(placed_at)?;
        injected.push_page(page, placed_at, ratio)?;

        // What that offset's own tokens would have produced.
        let mut forwarded = IndexCache::new(rig.cfg.head_dim, &device)?;
        forwarded.skip_to(placed_at)?;
        rig.append(&mut forwarded, 0, tokens)?;

        assert_eq!(
            injected.indexed_tokens(ratio),
            forwarded.indexed_tokens(ratio),
            "the two caches do not even cover the same span"
        );
        let qpos: Vec<usize> = (placed_at..placed_at + tokens).step_by(3).collect();
        assert_eq!(
            rig.select(&mut injected, &qpos)?,
            rig.select(&mut forwarded, &qpos)?,
            "an injected page selected differently than the same tokens forwarded at \
             that position — its rows are being read at the wrong positions"
        );
        Ok(())
    }

    /// **Closing a page mid-sequence changes nothing.** The live decode path: a
    /// unit boundary lifts the tail into a page that is placed exactly where it
    /// already sat, so every row keeps its position and the selection is
    /// untouched.
    ///
    /// Block-aligned deliberately — `close_tail_into_page` flushes a SHORT
    /// trailing block, which is a different summary on purpose, so a boundary
    /// off a block would be testing raggedness rather than position.
    #[test]
    fn closing_a_page_at_a_block_boundary_leaves_the_selection_alone() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let rig = Rig::new(device.clone(), 512)?;
        let ratio = rig.ratio;
        let (cut, total) = (40usize, 96usize);

        let mut whole = IndexCache::new(rig.cfg.head_dim, &device)?;
        rig.append(&mut whole, 0, total)?;

        let mut split = IndexCache::new(rig.cfg.head_dim, &device)?;
        rig.append(&mut split, 0, cut)?;
        let closed = split.close_tail_into_page(&rig.w, ratio, rig.eps)?;
        assert_eq!(closed, cut, "the cut did not close the tokens it was given");
        rig.append(&mut split, cut, total - cut)?;

        assert_eq!(split.indexed_tokens(ratio), whole.indexed_tokens(ratio));
        let qpos: Vec<usize> = (0..total).step_by(5).collect();
        assert_eq!(
            rig.select(&mut split, &qpos)?,
            rig.select(&mut whole, &qpos)?,
            "closing a page at a block boundary moved the selection"
        );
        Ok(())
    }

    /// **A hole after forwarded tokens closes them into a page and opens the
    /// tail past it** — a system prompt prefilled live, crossing a reserved
    /// range (`docs/placed_section_layout.md` §3.1). 40 tokens forwarded, a
    /// 24-position hole, 16 more: the forwarded 40 are one page of 40 tokens,
    /// the tail opens at 64, and the index reaches position 80 — the page, the
    /// hole and the 16. `skip_to` alone refuses the same hole.
    #[test]
    fn a_hole_after_forwarded_tokens_closes_them_into_a_page() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let rig = Rig::new(device.clone(), 256)?;
        let ratio = rig.ratio;

        let mut refused = IndexCache::new(rig.cfg.head_dim, &device)?;
        rig.append(&mut refused, 0, 40)?;
        assert!(
            refused.skip_to(64).is_err(),
            "skip_to over live rows is what the hole replaces"
        );

        let mut c = IndexCache::new(rig.cfg.head_dim, &device)?;
        rig.append(&mut c, 0, 40)?;
        c.hole(&rig.w, ratio, rig.eps, 24)?;
        assert_eq!(c.page_count(), 1);
        assert_eq!(c.page_at(0).map(|(_, tokens)| tokens), Some(40));
        assert_eq!(c.live_blocks(), 0);
        assert_eq!(c.next_base(), 64);
        rig.append(&mut c, 40, 16)?;
        assert_eq!(c.indexed_tokens(ratio), 80);
        Ok(())
    }

    /// **A page's rows are visible to exactly the queries they sit below.**
    ///
    /// The arithmetic that replaces `(pos + 1) / ratio` once a sequence holds a
    /// prefix it did not forward. Swept over three ragged pages and every
    /// position across them, because the widths only disagree with the uniform
    /// formula *after* the first boundary — a fixture with one page, or with
    /// block-aligned ones, agrees with it everywhere and proves nothing.
    #[test]
    fn candidates_walk_the_pages_widths() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let head_dim = 16usize;
        let ratio = 4usize;
        // Section lengths that are deliberately NOT multiples of the ratio, so
        // every boundary lands mid-block — which is what a section does, since
        // injection advances the sequence by its real token count.
        let sections = [13usize, 7, 26];
        let mut cache = IndexCache::new(head_dim, &device)?;
        let mut pos = 0usize;
        for &t in &sections {
            let rows = t.div_ceil(ratio);
            let page = zero_page(rows, t - (rows - 1) * ratio, head_dim, &device);
            cache.push_page(page, pos, ratio)?;
            pos += t;
        }
        let total: usize = sections.iter().sum();
        assert_eq!(cache.len(ratio), total, "pages do not span their tokens");
        assert_eq!(
            cache.page_row_span(),
            sections.iter().map(|t| t.div_ceil(ratio)).sum::<usize>()
        );

        // Monotone, never past what exists, and exactly complete at the end.
        let mut last = 0usize;
        for p in 0..total {
            let c = cache.candidates_at(p, ratio);
            assert!(c >= last, "candidates went backwards at {p}: {last} -> {c}");
            assert!(
                c <= cache.page_row_span(),
                "position {p} sees {c} rows but only {} exist",
                cache.page_row_span()
            );
            last = c;
        }
        assert_eq!(
            cache.candidates_at(total - 1, ratio),
            cache.page_row_span(),
            "the last token must see every page row — a page's short final row \
             is still wholly below it"
        );
        Ok(())
    }

    /// **The live tail counts from where the pages end, not from zero.**
    ///
    /// The join is the part that cannot be got right by accident: rows appended
    /// after an injected prefix pool from the tail's own first position, so
    /// their visibility is `page_rows + (pos − span) / ratio`. Off by one page
    /// and every forwarded token addresses a block belonging to the prefix.
    #[test]
    fn the_live_tail_counts_from_the_pages_end() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let rig = Rig::new(device.clone(), 128)?;
        let mut cache = IndexCache::new(rig.cfg.head_dim, &device)?;
        // One ragged page of 13 tokens => 4 rows, last covering 1.
        let page_tokens = 13usize;
        let page_rows = page_tokens.div_ceil(rig.ratio);
        cache.push_page(
            zero_page(
                page_rows,
                page_tokens - (page_rows - 1) * rig.ratio,
                rig.cfg.head_dim,
                &device,
            ),
            0,
            rig.ratio,
        )?;

        // Forward 9 more tokens on top: 2 whole rows and 1 carried.
        rig.append(&mut cache, 0, 9)?;
        assert_eq!(cache.len(rig.ratio), page_tokens + 9);
        assert_eq!(
            cache.candidates_at(page_tokens - 1, rig.ratio),
            page_rows,
            "the last page token must see the whole page and none of the tail"
        );
        assert_eq!(
            cache.candidates_at(page_tokens + 3, rig.ratio),
            page_rows + 1,
            "four tail tokens complete exactly one tail row"
        );
        assert_eq!(
            cache.candidates_at(page_tokens + 8, rig.ratio),
            page_rows + 2,
            "nine tail tokens complete two, with one carried"
        );
        Ok(())
    }

    /// **A page arriving after live rows is refused.**
    ///
    /// Pages describe a prefix. One pushed after a forward would claim
    /// positions the live tail already holds, and every later query would
    /// address the wrong block — silently, because the counts would still add
    /// up.
    #[test]
    fn a_page_after_live_rows_is_refused() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let rig = Rig::new(device.clone(), 32)?;
        let mut cache = IndexCache::new(rig.cfg.head_dim, &device)?;
        rig.append(&mut cache, 0, 8)?;
        let page = zero_page(2, rig.ratio, rig.cfg.head_dim, &device);
        let base = cache.next_base();
        let err = cache.push_page(page, base, rig.ratio).unwrap_err();
        assert!(err.to_string().contains("must precede"), "{err}");
        Ok(())
    }

    /// **An open block wider than one can be is refused, not truncated.**
    ///
    /// `n_open` is always `< ratio ≤ MAX_RATIO` by construction, so a record
    /// declaring more is corrupt. Installing it would leave rows in the raw
    /// buffer that the next append neither pools nor overwrites.
    #[test]
    fn an_oversized_open_block_is_refused() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let head_dim = 16usize;
        let rows = Q8Keys::quantize(&vec![0f32; 4 * head_dim], head_dim);
        let open = vec![0f32; (MAX_RATIO + 1) * head_dim];
        let err = IndexCache::from_rows(&rows, &open, head_dim, &device).unwrap_err();
        assert!(err.to_string().contains("open rows"), "{err}");
        Ok(())
    }
}
