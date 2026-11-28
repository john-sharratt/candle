//! A ranged collection filled by tokens (`docs/placed_section_layout.md`
//! §3.5): its mandatory members first, then what its selection chose, by
//! score, until the next does not fit the range or the range holds the most
//! it declares — so the range is a fixed size and its count is what fits. One
//! member always stays, whatever its size: the selection's minimum.

use std::collections::HashSet;

use crate::projection::SectionId;

/// Of `chosen` — `(member, score, tokens)` — the members a range of `room`
/// positions holds after `mandatory` tokens: by descending score (ties by
/// id), each while it fits, stopping at the first that does not or at
/// `most`; the strongest always.
pub(crate) fn fit_by_tokens(
    chosen: &[(SectionId, f32, usize)],
    room: usize,
    mandatory: usize,
    most: Option<usize>,
) -> HashSet<SectionId> {
    let mut ranked: Vec<&(SectionId, f32, usize)> = chosen.iter().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.raw().cmp(&b.0.raw())));
    ranked.truncate(most.unwrap_or(usize::MAX));
    let mut left = room.saturating_sub(mandatory);
    let mut out = HashSet::new();
    for (i, &&(id, _, tokens)) in ranked.iter().enumerate() {
        if tokens > left {
            if i == 0 {
                out.insert(id);
            }
            break;
        }
        left -= tokens;
        out.insert(id);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(raw: u32) -> SectionId {
        SectionId::new(raw)
    }

    /// **By score while each fits, stopping at the first that does not**:
    /// in 1,000 positions less 200 mandatory, b (900, 300) and a (800, 400)
    /// fit; c (700, 200) does not fit the 100 left, and d (600, 50), though
    /// it would, is not reached.
    #[test]
    fn members_fill_by_score_until_the_next_does_not_fit() {
        let chosen = [
            (id(1), 800.0, 400),
            (id(2), 900.0, 300),
            (id(3), 700.0, 200),
            (id(4), 600.0, 50),
        ];
        let got = fit_by_tokens(&chosen, 1_000, 200, None);
        assert_eq!(got, HashSet::from([id(1), id(2)]));
    }

    /// **The strongest always stays**, though it does not fit.
    #[test]
    fn the_strongest_member_always_stays() {
        let chosen = [(id(1), 800.0, 900), (id(2), 700.0, 10)];
        assert_eq!(fit_by_tokens(&chosen, 500, 0, None), HashSet::from([id(1)]));
        assert!(fit_by_tokens(&[], 500, 0, None).is_empty());
    }

    /// **A range's bound stops it however much room is left**: all three fit
    /// 1,000 positions, and at most two are b (900) and a (800).
    #[test]
    fn a_bound_stops_the_fill_with_room_left() {
        let chosen = [
            (id(1), 800.0, 100),
            (id(2), 900.0, 100),
            (id(3), 700.0, 100),
        ];
        assert_eq!(
            fit_by_tokens(&chosen, 1_000, 0, Some(2)),
            HashSet::from([id(1), id(2)])
        );
        assert_eq!(
            fit_by_tokens(&chosen, 1_000, 0, Some(1)),
            HashSet::from([id(2)])
        );
    }
}
