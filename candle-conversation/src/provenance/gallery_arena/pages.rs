//! The group-major page layout and the token-major → group-major transpose.
//!
//! A page holds [`PAGE_TOKENS`] tokens laid out **group-major** —
//! `[group][token][word]` — so the scan kernel, which fixes a layer-group per
//! block and reads consecutive tokens' `gw`-word signatures, gets coalesced
//! loads (see `docs/archived/paged_gallery_arena.md` §4.2). A [`WideQSig`]'s `words` are
//! already group-contiguous within a token (the 4 heads of a group are adjacent
//! and head-major), so the transpose is a pure regroup, no per-word shuffle.

use super::super::WideQSig;

/// Tokens per page. Mirrors the KV `CHUNK_SIZE`.
pub const PAGE_TOKENS: usize = 32;

/// Words per page for a signature width `wpt` (= `PAGE_TOKENS * wpt`). Independent
/// of the group split — group-major only reorders the same words.
#[inline]
pub fn page_u64(wpt: usize) -> usize {
    PAGE_TOKENS * wpt
}

/// Number of pages a turn of `n_tokens` occupies (last page partial).
#[inline]
pub fn pages_for(n_tokens: usize) -> usize {
    n_tokens.div_ceil(PAGE_TOKENS)
}

/// Transpose a turn's token-major folded sigs into group-major pages, laid end to
/// end in one buffer with page `p` starting at word `p * stride_words` — the slot
/// stride, so a run of address-contiguous slots is one contiguous slice of it.
/// Within a page, token `t` lands at group-major offset
/// `g*(PAGE_TOKENS*gw) + (t % PAGE_TOKENS)*gw`. The unused tail of the last page,
/// and any stride beyond a page's `page_u64(wpt)` words, stay zero (never
/// addressed by a scan).
pub fn transpose_to_pages(
    sigs: &[WideQSig],
    wpt: usize,
    n_groups: usize,
    stride_words: usize,
) -> Vec<u64> {
    assert!(
        stride_words >= page_u64(wpt),
        "a {stride_words}-word slot cannot hold a {}-word page",
        page_u64(wpt)
    );
    let gw = wpt / n_groups;
    let mut pages = vec![0u64; pages_for(sigs.len()) * stride_words];
    for (t, sig) in sigs.iter().enumerate() {
        let base = (t / PAGE_TOKENS) * stride_words;
        let in_pg = t % PAGE_TOKENS;
        for g in 0..n_groups {
            let s = g * gw;
            // Defensive: a well-formed folded sig is exactly `wpt` wide; a short
            // one leaves that group's slot zero rather than panicking.
            if s + gw <= sig.words.len() {
                let dst = base + g * (PAGE_TOKENS * gw) + in_pg * gw;
                pages[dst..dst + gw].copy_from_slice(&sig.words[s..s + gw]);
            }
        }
    }
    pages
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sig(fill: u64) -> WideQSig {
        WideQSig {
            n_heads: 12,
            words: (0..24).map(|w| fill.wrapping_add(w as u64)).collect(),
        }
    }

    #[test]
    fn one_full_page_group_major_layout() {
        // 32 tokens, wpt=24, 3 groups, gw=8 → one full page of 768 words.
        let sigs: Vec<WideQSig> = (0..32).map(|t| sig((t as u64) << 32)).collect();
        let pages = transpose_to_pages(&sigs, 24, 3, 768);
        assert_eq!(pages.len(), 768);
        // Group g, token t, word w must equal sigs[t].words[g*8 + w].
        for t in 0..32 {
            for g in 0..3 {
                for w in 0..8 {
                    let got = pages[g * (32 * 8) + t * 8 + w];
                    let want = sigs[t].words[g * 8 + w];
                    assert_eq!(got, want, "mismatch at token {t} group {g} word {w}");
                }
            }
        }
    }

    #[test]
    fn partial_last_page_zero_tail() {
        // 33 tokens → 2 pages; page 1 holds token 32 only, rest zero.
        let sigs: Vec<WideQSig> = (0..33).map(|t| sig(0xA000 + t as u64)).collect();
        let pages = transpose_to_pages(&sigs, 24, 3, 768);
        assert_eq!(pages.len(), 2 * 768);
        let page1 = &pages[768..];
        // Token 32 → page 1, in_pg 0, so its words sit at each group's base.
        for g in 0..3 {
            for w in 0..8 {
                assert_eq!(page1[g * 256 + w], sigs[32].words[g * 8 + w]);
            }
        }
        // in_pg 1..32 of page 1 are zero (unused tail).
        for in_pg in 1..32 {
            for g in 0..3 {
                for w in 0..8 {
                    assert_eq!(page1[g * 256 + in_pg * 8 + w], 0);
                }
            }
        }
    }

    /// A slot wider than a page starts each page at the slot stride, and the
    /// words between a page's end and the next page's start stay zero.
    #[test]
    fn pages_sit_at_the_slot_stride() {
        let sigs: Vec<WideQSig> = (0..64).map(|t| sig(0xB000 + t as u64)).collect();
        let pages = transpose_to_pages(&sigs, 24, 3, 800);
        assert_eq!(pages.len(), 2 * 800);
        let narrow = transpose_to_pages(&sigs, 24, 3, 768);
        assert_eq!(
            &pages[..768],
            &narrow[..768],
            "page 0 is laid out unchanged"
        );
        assert_eq!(
            &pages[800..1568],
            &narrow[768..],
            "page 1 starts at the stride"
        );
        assert!(
            pages[768..800].iter().all(|&w| w == 0),
            "the gap between pages is zero"
        );
    }

    #[test]
    fn page_sizing_helpers() {
        assert_eq!(page_u64(24), 768);
        assert_eq!(pages_for(0), 0);
        assert_eq!(pages_for(1), 1);
        assert_eq!(pages_for(32), 1);
        assert_eq!(pages_for(33), 2);
        assert_eq!(pages_for(64), 2);
    }
}
