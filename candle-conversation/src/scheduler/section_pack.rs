//! Which section ingests one forward carries.
//!
//! Tokens are one cost of a section, and not the one that ran out. On a model
//! that carries per-sequence recurrent state, a section that has not started
//! claims a whole store the moment it enters a forward — 72 slots of 3.27 MB on
//! Flash-Next, 235 MB — however few tokens it brings. Packed by tokens alone, a
//! mind's boot sent ~50 short sections into one forward on a 24 GB card: 12.5 GB
//! of stores claimed in a single step, the weight side conceded down to its
//! floor, and the span still could not place the next store.
//!
//! Sections never pass through admission (`admit_ground`'s header records why),
//! so this is where their store is priced: a forward takes no more fresh
//! sections than the ground above the weight floor pays for, one store each. A
//! section already under way holds its store, costs nothing more to carry, and
//! is never held back for one that has not started.

/// One section waiting to advance.
#[derive(Debug, Clone, Copy)]
pub(super) struct Candidate {
    /// Tokens it has left to ingest.
    pub remaining: usize,
    /// Nothing of it has run yet, so entering a forward claims its store.
    pub fresh: bool,
}

/// Fresh sections the ground pays for: `room` above the weight floor at one
/// `store` each. Unbounded on a model that carries no store.
pub(super) fn fresh_allowance(room: u64, store: u64) -> usize {
    if store == 0 {
        return usize::MAX;
    }
    usize::try_from(room / store).unwrap_or(usize::MAX)
}

/// The sections one forward carries, as `(candidate index, tokens advanced)` in
/// candidate order.
///
/// * Each advances by `min(remaining, cap)`.
/// * Packing stops at the first section that would take the forward past `cap`
///   tokens.
/// * A fresh section past `fresh_allowed` is passed over, not stopped at: a
///   started section behind it already holds its store and still rides.
/// * The forward is never empty while a candidate exists — the first rides
///   whatever the allowance, so ingest always progresses and every seal hands
///   its store back.
pub(super) fn pack(
    candidates: &[Candidate],
    cap: usize,
    fresh_allowed: usize,
) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut tokens = 0usize;
    let mut fresh_left = fresh_allowed;
    for (i, c) in candidates.iter().enumerate() {
        let advance = c.remaining.min(cap);
        if !out.is_empty() && tokens + advance > cap {
            break;
        }
        if c.fresh {
            if fresh_left == 0 {
                continue;
            }
            fresh_left -= 1;
        }
        tokens += advance;
        out.push((i, advance));
    }
    if out.is_empty() {
        if let Some(c) = candidates.first() {
            out.push((0, c.remaining.min(cap)));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh(remaining: usize) -> Candidate {
        Candidate {
            remaining,
            fresh: true,
        }
    }

    fn started(remaining: usize) -> Candidate {
        Candidate {
            remaining,
            fresh: false,
        }
    }

    /// The boot that failed: fifty short sections fit one forward's tokens, and
    /// the ground pays for three stores. Three ride, not fifty.
    #[test]
    fn short_fresh_sections_are_bounded_by_the_stores_the_ground_pays_for() {
        let many = vec![fresh(20); 50];
        let room = 3 * 246_022_144 + 1_000;
        let allowed = fresh_allowance(room, 246_022_144);
        assert_eq!(allowed, 3);
        assert_eq!(pack(&many, 8_192, allowed), vec![(0, 20), (1, 20), (2, 20)]);
    }

    /// A started section holds its store already: it rides past fresh ones the
    /// allowance turns away.
    #[test]
    fn started_sections_ride_past_fresh_ones_the_ground_cannot_pay_for() {
        let cands = [fresh(10), started(30), fresh(10), started(40)];
        assert_eq!(pack(&cands, 8_192, 0), vec![(1, 30), (3, 40)]);
        assert_eq!(pack(&cands, 8_192, 1), vec![(0, 10), (1, 30), (3, 40)]);
    }

    /// With no ground at all a forward still carries one section, so the ingest
    /// moves and its seal returns ground.
    #[test]
    fn a_forward_is_never_empty_while_a_section_waits() {
        assert_eq!(pack(&[fresh(500), fresh(9)], 8_192, 0), vec![(0, 500)]);
        assert!(pack(&[], 8_192, 0).is_empty());
    }

    /// The token budget still closes the forward, as before: a section that
    /// would take it past `cap` ends the packing, and a long one advances by
    /// `cap` alone.
    #[test]
    fn the_token_budget_still_closes_the_forward() {
        let cands = [started(5_000), started(4_000), started(10)];
        assert_eq!(pack(&cands, 8_192, usize::MAX), vec![(0, 5_000)]);
        assert_eq!(pack(&[fresh(20_000)], 8_192, 4), vec![(0, 8_192)]);
    }

    /// A model with no recurrent state prices nothing, so tokens alone decide.
    #[test]
    fn a_model_without_a_store_is_bounded_by_tokens_alone() {
        assert_eq!(fresh_allowance(0, 0), usize::MAX);
        assert_eq!(
            pack(&vec![fresh(20); 50], 8_192, fresh_allowance(0, 0)).len(),
            50
        );
    }
}
