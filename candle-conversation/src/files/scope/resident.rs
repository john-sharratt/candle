//! The sections a range keeps hot beside it (`docs/placed_section_layout.md`
//! §3.5, §3.7b): its residency, twice the range.
//!
//! A section resident and out of scope can enter at the next step with no
//! PCIe or redo-log wait, and its index rows are what the shadow scan rates it
//! by. The placed sections are resident by being placed and take their share
//! of the budget first; the rest goes to the strongest unplaced sections,
//! those the shadow scan read since the last boundary ahead of those it did
//! not — the index's evidence and provenance's ranking, fused — each that
//! fits in what is left.

use super::range_scope::RangeScope;
use super::step::Candidate;
use crate::files::key::SectionKey;

/// How many ranges' worth of sections are resident.
pub const RESIDENT_RANGES: usize = 2;

/// The unplaced sections of `ranked` — strongest first — `scope` keeps
/// resident, in the order they were chosen: those `read` names first, each
/// group in rank order, each that fits in [`RESIDENT_RANGES`] ranges less the
/// placed sections' tokens.
pub fn resident(
    scope: &RangeScope,
    ranked: &[Candidate],
    read: impl Fn(SectionKey) -> bool,
) -> Vec<Candidate> {
    let budget = scope.positions.saturating_mul(RESIDENT_RANGES);
    let placed: usize = scope.sections().map(|e| e.tokens).sum();
    let mut left = budget.saturating_sub(placed);
    let (first, rest): (Vec<&Candidate>, Vec<&Candidate>) = ranked
        .iter()
        .filter(|c| !scope.holds(c.key))
        .partition(|c| read(c.key));
    let mut out = Vec::new();
    for c in first.into_iter().chain(rest) {
        if c.tokens > left {
            continue;
        }
        left -= c.tokens;
        out.push(*c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(key: u64, tokens: usize) -> Candidate {
        Candidate {
            key: SectionKey(key),
            tokens,
        }
    }

    /// **Residency is twice the range less what is placed, the read sections
    /// first, each that fits**: a 50-range holds a (30), so 70 of 100 are
    /// left; d, read, goes first (40), then b (20); c (25) does not fit the 10
    /// left and is passed over for e (10). a itself is placed, not resident.
    #[test]
    fn residency_is_twice_the_range_the_read_sections_first() {
        let mut s = RangeScope::new(50);
        s.enter(SectionKey(1), 30, false);
        let ranked = [c(1, 30), c(2, 20), c(3, 25), c(4, 40), c(5, 10)];
        let got = resident(&s, &ranked, |k| k.0 == 4);
        assert_eq!(got, vec![c(4, 40), c(2, 20), c(5, 10)]);
    }

    /// **An empty range keeps nothing resident.**
    #[test]
    fn an_empty_range_keeps_nothing_resident() {
        let s = RangeScope::new(0);
        assert!(resident(&s, &[c(1, 1)], |_| true).is_empty());
    }
}
