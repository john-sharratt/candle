//! The group-major page layout and the token-major → group-major transpose.
//!
//! A page holds [`PAGE_TOKENS`] tokens of the fold groups its turn keeps
//! resident, laid out **group-major** — `[group][token][word]`, the groups packed
//! in group order — so the scan kernel, which fixes a layer-group per block and
//! reads consecutive tokens' `gw`-word signatures, gets coalesced loads (see
//! `docs/archived/paged_gallery_arena.md` §4.2). A group nothing weighs is not
//! in the page at all (`docs/gallery_residency.md` §4). A [`WideQSig`]'s `words`
//! are already group-contiguous within a token (the 4 heads of a group are
//! adjacent and head-major), so the transpose is a pure regroup, no per-word
//! shuffle.

use super::super::WideQSig;
use super::dedup::KeepMask;
use super::groups::GroupSet;

/// Tokens per page. Mirrors the KV `CHUNK_SIZE`.
pub const PAGE_TOKENS: usize = 32;

/// Words per page holding `groups`, `gw` words a token each.
#[inline]
pub fn page_words(gw: usize, groups: GroupSet) -> usize {
    PAGE_TOKENS * gw * groups.len()
}

/// Number of pages a turn of `n_tokens` occupies (last page partial).
#[inline]
pub fn pages_for(n_tokens: usize) -> usize {
    n_tokens.div_ceil(PAGE_TOKENS)
}

/// Transpose the tokens `mask` keeps of a turn's token-major folded sigs into
/// group-major pages of `groups`, packed in order — the `t`-th kept token at
/// place `t` — and laid end to end in one buffer with page `p` starting at word
/// `p * stride_words` — the slot stride, so a run of address-contiguous slots
/// is one contiguous slice of it. Within a page, kept token `t`'s group `g`
/// lands at `place(g)*(PAGE_TOKENS*gw) + (t % PAGE_TOKENS)*gw`, `place` its
/// rank in `groups`. The unused tail of the last page, and any stride beyond a
/// page's [`page_words`], stay zero (never addressed by a scan).
pub fn transpose_to_pages(
    sigs: &[WideQSig],
    mask: &KeepMask,
    gw: usize,
    groups: GroupSet,
    stride_words: usize,
) -> Vec<u64> {
    assert!(
        stride_words >= page_words(gw, groups),
        "a {stride_words}-word slot cannot hold a {}-word page",
        page_words(gw, groups)
    );
    debug_assert_eq!(mask.len(), sigs.len(), "a mask of another turn");
    let mut pages = vec![0u64; pages_for(mask.kept()) * stride_words];
    let kept = sigs
        .iter()
        .enumerate()
        .filter(|&(t, _)| mask.keeps(t))
        .map(|(_, sig)| sig);
    for (t, sig) in kept.enumerate() {
        let base = (t / PAGE_TOKENS) * stride_words;
        let in_pg = t % PAGE_TOKENS;
        for (place, g) in groups.iter().enumerate() {
            let s = g * gw;
            // Defensive: a well-formed folded sig is exactly `wpt` wide; a short
            // one leaves that group's slot zero rather than panicking.
            if s + gw <= sig.words.len() {
                let dst = base + place * (PAGE_TOKENS * gw) + in_pg * gw;
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

    /// Every token of `sigs` paged.
    fn transpose_all(sigs: &[WideQSig], gw: usize, groups: GroupSet, stride: usize) -> Vec<u64> {
        transpose_to_pages(sigs, &KeepMask::all(sigs.len()), gw, groups, stride)
    }

    #[test]
    fn one_full_page_group_major_layout() {
        // 32 tokens, gw=8, all 3 groups → one full page of 768 words.
        let sigs: Vec<WideQSig> = (0..32).map(|t| sig((t as u64) << 32)).collect();
        let pages = transpose_all(&sigs, 8, GroupSet::all(3), 768);
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

    /// **A page holds only its groups, packed in group order.** Groups 1 and 2
    /// of 32 tokens are a 512-word page — group 1's strip first, group 2's
    /// second — and group 0's words are nowhere in it.
    #[test]
    fn a_page_packs_only_its_groups() {
        let sigs: Vec<WideQSig> = (0..32).map(|t| sig((t as u64) << 32)).collect();
        let groups = GroupSet::default().with(1).with(2);
        assert_eq!(page_words(8, groups), 512);
        let pages = transpose_all(&sigs, 8, groups, 512);
        assert_eq!(pages.len(), 512);
        for t in 0..32 {
            for w in 0..8 {
                assert_eq!(pages[t * 8 + w], sigs[t].words[8 + w], "group 1, token {t}");
                assert_eq!(
                    pages[256 + t * 8 + w],
                    sigs[t].words[16 + w],
                    "group 2, token {t}"
                );
            }
        }
        let one = transpose_all(&sigs, 8, GroupSet::default().with(2), 256);
        assert_eq!(
            one,
            pages[256..].to_vec(),
            "group 2 alone is group 2's strip"
        );
    }

    #[test]
    fn partial_last_page_zero_tail() {
        // 33 tokens → 2 pages; page 1 holds token 32 only, rest zero.
        let sigs: Vec<WideQSig> = (0..33).map(|t| sig(0xA000 + t as u64)).collect();
        let pages = transpose_all(&sigs, 8, GroupSet::all(3), 768);
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
        let all = GroupSet::all(3);
        let pages = transpose_all(&sigs, 8, all, 800);
        assert_eq!(pages.len(), 2 * 800);
        let narrow = transpose_all(&sigs, 8, all, 768);
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

    /// **A mask's kept tokens are packed in order**: of 40 tokens, the even
    /// ones — 20, one page — with kept token `k` (source token `2k`) at place
    /// `k`.
    #[test]
    fn kept_tokens_pack_in_order() {
        let sigs: Vec<WideQSig> = (0..40).map(|t| sig(0xC000 + t as u64)).collect();
        let keep: Vec<bool> = (0..40).map(|t| t % 2 == 0).collect();
        let pages = transpose_to_pages(
            &sigs,
            &KeepMask::from_bools(&keep),
            8,
            GroupSet::all(3),
            768,
        );
        assert_eq!(pages.len(), 768, "20 kept tokens are one page");
        for k in 0..20 {
            for g in 0..3 {
                for w in 0..8 {
                    assert_eq!(
                        pages[g * 256 + k * 8 + w],
                        0xC000 + 2 * k as u64 + (g * 8 + w) as u64
                    );
                }
            }
        }
        assert!(
            pages[20 * 8..32 * 8].iter().all(|&w| w == 0),
            "the tail is zero"
        );
    }

    #[test]
    fn page_sizing_helpers() {
        assert_eq!(page_words(8, GroupSet::all(3)), 768);
        assert_eq!(page_words(8, GroupSet::default().with(2)), 256);
        assert_eq!(pages_for(0), 0);
        assert_eq!(pages_for(1), 1);
        assert_eq!(pages_for(32), 1);
        assert_eq!(pages_for(33), 2);
        assert_eq!(pages_for(64), 2);
    }
}
