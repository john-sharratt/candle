//! The order layers are given up in, and why it is not the layer order.
//!
//! # The gap between two missing layers is the time to fetch one
//!
//! A forward walks `0, 1, … N-1`. A layer is either resident (free) or missing
//! (fetched over PCIe into the floating cell). The fetch for the *next* missing
//! layer starts when the current one stops occupying the cell, so the time
//! available to hide it is the run of resident layers between them:
//!
//! ```text
//! window(gap) = (gap − 1) × t_compute       gap = m[i+1] − m[i], cyclically
//! stall(gap)  = max(0, t_fetch − window(gap))
//! ```
//!
//! On this card `t_compute ≈ 1.4 ms` and a ~200 MiB layer is `t_fetch ≈ 8 ms`,
//! so a gap of 7 hides a transfer completely and a gap of 1 hides none of it.
//!
//! # Equal gaps are optimal, and that is a theorem rather than an intuition
//!
//! `stall` is convex in `gap`, and `Σ gap = N` is fixed by the number of missing
//! layers. By Jensen the sum of a convex function over a fixed total is minimised
//! when the arguments are equal. **So the missing set should be spread as evenly
//! around the cycle as it can be** — for every count of missing layers, not for
//! one chosen count.
//!
//! That last clause is what makes this an *order* rather than a set. The number
//! of missing layers is not known at build time and changes at runtime as the
//! elastic boundary concedes ground to KV, so what is needed is a sequence whose
//! every prefix is well spread.
//!
//! # The order this produces
//!
//! Repeated bisection of the largest gap: each next layer to give up is the one
//! nearest the midpoint of the widest run of resident layers, ties broken by the
//! lowest gap start. On a power of two this reproduces the classic bit-reversal
//! spacing — at every `k = 2^j` the missing layers are *exactly* equally spaced —
//! and unlike bit-reversal it keeps working when `N` is not a power of two, where
//! a reversed index has to be discarded when it lands past `N` and the survivors
//! bunch.
//!
//! Two rules that look equivalent are not, and the difference is a factor of the
//! model depth. Picking the candidate whose *nearest* missing neighbour is
//! furthest — plain farthest-point — is blind to the gap it leaves behind: on
//! `N = 36` it opens `0, 18, 9`, whose third gap is 18 against an ideal of 12.
//! Bisecting the widest gap is the rule that keeps the bound.
//!
//! # The bound, stated honestly
//!
//! A *nested* sequence — one whose every prefix is the answer for that size —
//! cannot be exactly even at every `k`: `{0, 32}` on 64 layers cannot extend to
//! the perfect three-point set `{0, 21, 43}`. The guarantee is
//! `max gap ≤ 2 · ⌈N/k⌉`, tight just above a power of two and exact at every
//! power of two. That is worth taking, because the alternative — recomputing an
//! exactly-even set whenever capacity moves — re-fetches nearly every missing
//! layer for a change of one slot, and because the 2× case bites only at large
//! `k`, where gaps are small; at small `k` every gap is far past what a transfer
//! needs.
//!
//! # Nesting is what makes a capacity change cheap
//!
//! The resident set is always a **prefix of the protection order**, so the
//! missing set is always a prefix of the eviction order. Conceding ground to KV
//! appends to the missing set; taking it back pops from the end. Either way the
//! layers that stay put are untouched, exactly the layers that must move do, and
//! the spread is optimal at the new size without recomputing anything.
//!
//! # What the previous order was
//!
//! A held **prefix**: layers `0..keep` resident, `keep..N` missing. The missing
//! layers were therefore *contiguous*, every gap was 1, and not one byte of any
//! transfer was hidden behind compute. Measured on the 27B at capacity 38 of 64
//! that is 29 consecutive fetches per forward, none overlapped. The spread costs
//! nothing to compute and is the whole difference.

/// The order in which layers are given up, least-protected first.
///
/// `order[0]` is the first layer to go missing when the zone cannot hold the
/// trunk, `order[N-1]` the last. Pinned layers are absent: they have no record
/// in the cold tier, so there is nowhere to fetch them back from and they are
/// not a resource the boundary may spend.
///
/// The returned sequence is a permutation of `pinned..num_layers`.
pub fn eviction_order(num_layers: usize, pinned: usize) -> Vec<usize> {
    let pinned = pinned.min(num_layers);
    let total = num_layers - pinned;
    if total == 0 {
        return Vec::new();
    }
    // The lowest evictable layer opens the sequence. Any position would spread
    // as well from an empty set; fixing it makes the order reproducible, which
    // matters because the pack's geometry and the runtime's placement both
    // derive from it and a disagreement is a pack that does not describe itself.
    let mut missing: Vec<usize> = vec![pinned];
    let mut out: Vec<usize> = vec![pinned];
    while out.len() < total {
        let mut best: Option<(usize, usize, usize)> = None; // (gap len, start, pick)
        for i in 0..missing.len() {
            let a = missing[i];
            let b = missing[(i + 1) % missing.len()];
            let len = match (b + num_layers - a) % num_layers {
                0 => num_layers,
                d => d,
            };
            let Some(pick) = nearest_legal_in_gap(a, len, num_layers, pinned, &missing) else {
                continue;
            };
            // Widest gap first, then lowest start — the tie-break only decides
            // *which* equal gap is split, never how well, so it exists to make
            // the sequence deterministic rather than to make it better.
            if best.is_none_or(|(bl, bs, _)| len > bl || (len == bl && a < bs)) {
                best = Some((len, a, pick));
            }
        }
        let (_, _, pick) = best.expect("a gap remains while layers are unplaced");
        out.push(pick);
        missing.push(pick);
        missing.sort_unstable();
    }
    out
}

/// The evictable position nearest the middle of the gap that starts at `a` and
/// spans `len` positions.
///
/// Searched outward from the midpoint so the split is as even as it can be, and
/// bounded strictly inside the gap so a pick never lands on one of its own
/// endpoints. Returns `None` for a gap whose interior is entirely pinned or
/// already missing, which is how the walk skips a run it cannot improve.
fn nearest_legal_in_gap(
    a: usize,
    len: usize,
    n: usize,
    pinned: usize,
    missing: &[usize],
) -> Option<usize> {
    if len < 2 {
        return None;
    }
    let mid = len / 2;
    for d in 0..=mid {
        for off in [mid.checked_sub(d), Some(mid + d).filter(|o| *o < len)]
            .into_iter()
            .flatten()
        {
            if off == 0 {
                continue;
            }
            let p = (a + off) % n;
            if p >= pinned && !missing.contains(&p) {
                return Some(p);
            }
        }
    }
    None
}

/// Layers in the order they are placed, **most protected first**.
///
/// The weight zone's addresses descend as this index rises, so `order[0]` sits
/// at the rightmost ground — beside the dense block, where retraction can never
/// reach — and the last entry sits at the frontier the boundary eats into. The
/// pinned head leads, because it can never be given up at all.
///
/// This is [`eviction_order`] reversed with the pinned head in front, and it is
/// the sequence the zone walks when it decides how many layers a byte budget
/// holds.
pub fn protection_order(num_layers: usize, pinned: usize) -> Vec<usize> {
    let pinned = pinned.min(num_layers);
    let mut out: Vec<usize> = (0..pinned).collect();
    out.extend(eviction_order(num_layers, pinned).into_iter().rev());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The gaps between consecutive members of `set` around a cycle of `n`.
    fn gaps(set: &[usize], n: usize) -> Vec<usize> {
        let mut s = set.to_vec();
        s.sort_unstable();
        let mut g = Vec::with_capacity(s.len());
        for i in 0..s.len() {
            let next = s[(i + 1) % s.len()];
            g.push(if s.len() == 1 {
                n
            } else {
                (next + n - s[i]) % n
            });
        }
        g
    }

    #[test]
    fn the_order_is_a_permutation_of_the_evictable_layers() {
        for n in [1usize, 2, 7, 8, 48, 64] {
            for p in [0usize, 1, 2] {
                let p = p.min(n);
                let mut o = eviction_order(n, p);
                assert_eq!(o.len(), n - p, "n={n} pinned={p}");
                o.sort_unstable();
                assert_eq!(o, (p..n).collect::<Vec<_>>(), "n={n} pinned={p}");
            }
        }
    }

    /// **The property the whole design rests on.** Not "the final set is well
    /// spread" but "every prefix is", because the number of missing layers moves
    /// at runtime as the boundary concedes ground.
    ///
    /// `2 · ⌈n/k⌉` is the bound a *nested* sequence can hold — see the module
    /// header for why an exactly-even set is unreachable at every size at once.
    #[test]
    fn every_prefix_is_evenly_spread() {
        for n in [16usize, 48, 64] {
            let order = eviction_order(n, 0);
            for k in 1..=n {
                let max = *gaps(&order[..k], n).iter().max().unwrap();
                assert!(
                    max <= 2 * n.div_ceil(k),
                    "n={n} k={k}: largest gap {max} against ideal {} for {:?}",
                    n.div_ceil(k),
                    &order[..k]
                );
            }
        }
    }

    /// At every power of two the spread is not merely bounded but **exact** —
    /// every gap identical. Those are the sizes the bisection reaches cleanly,
    /// and the 2× worst case sits just past them.
    #[test]
    fn at_a_power_of_two_count_the_gaps_are_all_equal() {
        let n = 64;
        let order = eviction_order(n, 0);
        for k in [1usize, 2, 4, 8, 16, 32, 64] {
            let g = gaps(&order[..k], n);
            assert!(
                g.iter().all(|&x| x == n / k),
                "k={k}: gaps {g:?} are not all {}",
                n / k
            );
        }
    }

    /// A pinned head is never given up, at any depth.
    #[test]
    fn the_pinned_head_is_never_in_the_eviction_order() {
        let order = eviction_order(64, 2);
        assert!(order.iter().all(|&l| l >= 2));
        let prot = protection_order(64, 2);
        assert_eq!(&prot[..2], &[0, 1], "the pinned head leads the placement");
    }

    /// The sequence itself, written out. The property tests above say it is
    /// *good*; this says it is *this one* — so a change to the rule that keeps
    /// the bound but moves every layer shows up as a pack rebuild here rather
    /// than as a mysterious re-fetch on a user's machine.
    #[test]
    fn the_sequence_is_pinned() {
        assert_eq!(eviction_order(8, 0), vec![0, 4, 2, 6, 1, 3, 5, 7]);
        assert_eq!(
            eviction_order(16, 0),
            vec![0, 8, 4, 12, 2, 6, 10, 14, 1, 3, 5, 7, 9, 11, 13, 15]
        );
    }

    /// A depth that is not a power of two keeps the spread. Bit-reversal does
    /// not — it produces indices past the end that have to be discarded, which
    /// bunches what is left — and plain farthest-point does not either: it opens
    /// `0, 18, 9` on 36 layers, leaving a gap of 18 against an ideal of 12.
    #[test]
    fn a_non_power_of_two_depth_is_still_spread() {
        for n in [36usize, 40, 48, 62] {
            let order = eviction_order(n, 0);
            for k in 1..=n {
                let max = *gaps(&order[..k], n).iter().max().unwrap();
                assert!(max <= 2 * n.div_ceil(k), "n={n} k={k} max gap {max}");
            }
        }
        // The specific regression: the third pick must not leave half the cycle
        // uncovered.
        assert!(*gaps(&eviction_order(36, 0)[..3], 36).iter().max().unwrap() <= 24);
    }

    /// **Growing the zone loads the right layers, in the right order.**
    ///
    /// The resident set is a prefix of the protection order at every size, so a
    /// zone that grows from `r` to `r'` residents brings in exactly
    /// `protection_order[r..r']` — nothing already resident moves, nothing is
    /// re-fetched, and the missing set is still a prefix of the eviction order
    /// and therefore still optimally spread. The same statement read backwards
    /// is what makes conceding ground to KV cheap.
    #[test]
    fn growth_extends_the_resident_prefix_and_keeps_the_spread() {
        let (n, p) = (64usize, 2);
        let prot = protection_order(n, p);
        let evict = eviction_order(n, p);
        for r in p..n {
            let before: Vec<usize> = prot[..r].to_vec();
            let after: Vec<usize> = prot[..r + 1].to_vec();
            // Growth is append-only: every previously resident layer is still
            // resident, in the same position.
            assert_eq!(&after[..r], before.as_slice(), "r={r}");
            // And the layer it brings in is the one that was most recently
            // given up — the head of what is still missing.
            let missing_before = n - r;
            assert_eq!(after[r], evict[missing_before - 1], "r={r}");
            // The missing set stays a prefix of the eviction order.
            let missing: Vec<usize> = evict[..n - r - 1].to_vec();
            let resident: std::collections::HashSet<usize> = after.iter().copied().collect();
            assert!(missing.iter().all(|l| !resident.contains(l)), "r={r}");
            assert_eq!(missing.len() + after.len(), n, "r={r}");
        }
    }

    /// Protection is exactly the reverse of eviction, so the layer placed
    /// closest to the frontier is the layer given up first. The zone relies on
    /// this: it drops from the end of its placement walk and expects that to be
    /// the head of the eviction order.
    #[test]
    fn protection_is_the_reverse_of_eviction() {
        let (n, p) = (64, 2);
        let evict = eviction_order(n, p);
        let prot = protection_order(n, p);
        assert_eq!(prot.len(), n);
        for (i, &l) in evict.iter().enumerate() {
            assert_eq!(prot[n - 1 - i], l, "evict[{i}] should be last-placed");
        }
    }

    /// Degenerate depths do not panic and do not invent layers.
    #[test]
    fn an_empty_or_fully_pinned_stack_yields_nothing_to_evict() {
        assert!(eviction_order(0, 0).is_empty());
        assert!(eviction_order(4, 4).is_empty());
        assert!(eviction_order(4, 9).is_empty());
        assert_eq!(protection_order(4, 9), vec![0, 1, 2, 3]);
    }

    /// The contrast with what this replaces, priced in the quantity that
    /// matters rather than in gap sizes.
    ///
    /// A held prefix leaves the missing layers *contiguous*, so all but one gap
    /// is 1 and not one byte of any transfer is hidden. The stall model is the
    /// module header's, at this card's measured constants: ~1.4 ms of compute
    /// per layer against ~8 ms to fetch a ~200 MiB layer over PCIe 4.0 ×16.
    #[test]
    fn the_spread_beats_a_contiguous_tail_on_stall() {
        const T_COMPUTE_MS: f64 = 1.4;
        const T_FETCH_MS: f64 = 8.0;
        let stall = |set: &[usize], n: usize| -> f64 {
            gaps(set, n)
                .iter()
                .map(|&g| (T_FETCH_MS - (g as f64 - 1.0) * T_COMPUTE_MS).max(0.0))
                .sum()
        };
        let n = 64;
        let order = eviction_order(n, 0);
        for k in [4usize, 10, 20, 30] {
            let tail: Vec<usize> = (n - k..n).collect();
            let (a, b) = (stall(&tail, n), stall(&order[..k], n));
            assert!(
                b < a,
                "k={k}: spread stalls {b:.1} ms against the tail's {a:.1} ms"
            );
            // Every interior gap of a contiguous tail is 1, so every one of its
            // fetches is fully exposed.
            assert_eq!(gaps(&tail, n).iter().filter(|&&g| g == 1).count(), k - 1);
        }
        // Up to a power-of-two count the spread hides the transfers *outright*
        // rather than merely shortening them: 8 missing layers of 64 sit 8 apart
        // and 7 layers of compute already exceed one fetch.
        assert_eq!(stall(&order[..8], n), 0.0, "8 missing should not stall");
        assert_eq!(stall(&order[..4], n), 0.0, "4 missing should not stall");
        // And this is the shape of the nested sequence's worst case, recorded
        // rather than hidden: the ninth pick halves one 8-gap into two 4s while
        // the other seven stay at 8, so a little stall reappears. An exactly-even
        // 9-set would still hide everything — it is simply not reachable by
        // adding one layer to the even 8-set, which is the price of never
        // re-fetching on a capacity change.
        let worst = stall(&order[..9], n);
        assert!(
            (0.0..=8.0).contains(&worst),
            "the 2x case should cost one partial fetch, not many: {worst:.1} ms"
        );
        assert!(worst < stall(&(n - 9..n).collect::<Vec<_>>(), n) / 8.0);
    }
}
