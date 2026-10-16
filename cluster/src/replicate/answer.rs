//! The answerer's side of a catch-up (§7.2): it speaks only for what it covers
//! completely. For each origin it lists the ranges of its applied set above
//! what the requester has applied through; a gap in its own set ends a range,
//! so it never lists past a hole.
//!
//! **An answer is bounded** to [`FRAMES_PER_ANSWER`] `HEADERS`. The requester
//! fetches what they list and asks again from where it now stands, so a member
//! far behind catches up an answer at a time. Unbounded, one answer to a member
//! lacking millions of records was a thousand frames queued at once, far past
//! what a gateway holds for one member: frames were dropped, the requester went
//! quiet waiting for them and asked again, and every new answer queued behind
//! the old ones it would ignore.

use applied_set::{AppliedSet, OriginApplied};

/// The most `oseq`s one `HEADERS` lists, held or gone: about 360 KiB of
/// headers, so no single frame holds the catch-up lane for long.
pub const HEADERS_PER_FRAME: u64 = 8192;

/// The most `HEADERS` one answer sends: about 6 MiB of headers, well inside
/// what a gateway queues for one member's catch-up lane (`CATCHUP_CAP`).
pub const FRAMES_PER_ANSWER: usize = 16;

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

/// The `(origin, from, to)` each `HEADERS` of one answer to `since` lists,
/// each spanning at most [`HEADERS_PER_FRAME`] `oseq`s, in origin order, and
/// at most [`FRAMES_PER_ANSWER`] of them.
pub fn answer_ranges(mine: &AppliedSet, since: &AppliedSet) -> Vec<(u64, u64, u64)> {
    let mut out = Vec::new();
    for (origin, applied) in mine.origins() {
        for (from, to) in complete_ranges(applied, since.through(origin)) {
            let mut at = from;
            while at <= to {
                if out.len() == FRAMES_PER_ANSWER {
                    return out;
                }
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

    /// An answer stops at its frame bound, mid-origin; asked again from where
    /// the requester then stands, the next answer starts where it stopped.
    #[test]
    fn an_answer_is_bounded_and_the_next_one_resumes() {
        let n = HEADERS_PER_FRAME;
        let mut mine = AppliedSet::default();
        mine.insert_range(4, 1, n * 20);
        mine.insert_range(9, 1, 3);
        let first = answer_ranges(&mine, &AppliedSet::default());
        assert_eq!(first.len(), FRAMES_PER_ANSWER);
        assert_eq!(first[0], (4, 1, n));
        assert_eq!(first[15], (4, 15 * n + 1, 16 * n));

        let mut since = AppliedSet::default();
        since.insert_range(4, 1, 16 * n);
        let next = answer_ranges(&mine, &since);
        assert_eq!(
            next,
            vec![
                (4, 16 * n + 1, 17 * n),
                (4, 17 * n + 1, 18 * n),
                (4, 18 * n + 1, 19 * n),
                (4, 19 * n + 1, 20 * n),
                (9, 1, 3),
            ]
        );
    }
}
