//! The answerer's side of a catch-up (§7.2): it speaks only for what it covers
//! completely. For each origin it lists the ranges of its applied set above
//! what the requester has applied through; a gap in its own set ends a range,
//! so it never lists past a hole.

use applied_set::{AppliedSet, OriginApplied};

/// The most `oseq`s one `HEADERS` lists, held or gone: about 360 KiB of
/// headers, so no single frame holds the catch-up lane for long.
pub const HEADERS_PER_FRAME: u64 = 8192;

/// The ranges of `mine` above `since` that it covers completely.
pub fn complete_ranges(mine: &OriginApplied, since: u64) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    if mine.through() > since {
        out.push((since + 1, mine.through()));
    }
    for (s, e) in mine.ranges_above() {
        if e > since {
            out.push((s.max(since + 1), e));
        }
    }
    out
}

/// Every `(origin, from, to)` a `HEADERS` lists in answer to `since`, each
/// spanning at most [`HEADERS_PER_FRAME`] `oseq`s, in origin order.
pub fn answer_ranges(mine: &AppliedSet, since: &AppliedSet) -> Vec<(u64, u64, u64)> {
    let mut out = Vec::new();
    for (origin, applied) in mine.origins() {
        for (from, to) in complete_ranges(applied, since.through(origin)) {
            let mut at = from;
            while at <= to {
                let end = to.min(at + HEADERS_PER_FRAME - 1);
                out.push((origin, at, end));
                at = end + 1;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A hole in the answerer's own set ends a range: nothing past it is
    /// claimed, and the covered range above it is listed on its own.
    #[test]
    fn a_hole_ends_a_range() {
        let mine = OriginApplied::from_parts(10, [(14, 20)]);
        assert_eq!(complete_ranges(&mine, 0), vec![(1, 10), (14, 20)]);
        assert_eq!(complete_ranges(&mine, 7), vec![(8, 10), (14, 20)]);
        assert_eq!(complete_ranges(&mine, 10), vec![(14, 20)]);
        assert_eq!(complete_ranges(&mine, 15), vec![(16, 20)]);
        assert_eq!(complete_ranges(&mine, 20), vec![]);
    }

    /// Every origin above the requester's `since`, split to the per-frame
    /// bound; an origin the requester has never seen is listed from one.
    #[test]
    fn ranges_are_listed_per_origin_and_split() {
        let mut mine = AppliedSet::default();
        mine.insert_range(4, 1, 20_000);
        mine.insert_range(9, 1, 3);
        let mut since = AppliedSet::default();
        since.insert_range(4, 1, 5);
        since.insert_range(9, 1, 3);
        assert_eq!(
            answer_ranges(&mine, &since),
            vec![(4, 6, 8197), (4, 8198, 16389), (4, 16390, 20_000)]
        );
        assert_eq!(answer_ranges(&mine, &AppliedSet::default())[3], (9, 1, 3));
    }
}
