//! Prefill gangs: long background prompts prefilled together, so they reach
//! their first tokens in the same forward and decode side by side
//! (`docs/prefill_gangs.md`).
//!
//! A decode step costs nearly the same at one row as at many — the step is the
//! expert stream, not the rows — so long prompts that finish one at a time
//! decode one row wide and pay a whole step per token. A gang finishes together
//! and shares its steps.

use super::admission::fill_in_order;
use super::CHUNK_SIZE;

/// The widest gang, up to `longs` members, whose decode step `step(n)` costs no
/// more than twice a lone decode's: each member then decodes at worst half as
/// fast as it would alone, while each step carries `n` rows. `1` — no gang —
/// when a lone step has no learned cost; `0` with nothing to gang.
pub(super) fn gang_width(longs: usize, step: impl Fn(usize) -> f64) -> usize {
    if longs == 0 {
        return 0;
    }
    let lone = step(1);
    if !(lone > 0.0 && lone.is_finite()) {
        return 1;
    }
    (1..=longs)
        .take_while(|&n| step(n) <= 2.0 * lone)
        .last()
        .unwrap_or(1)
}

/// The gang formed from `longs` — each long prompt's `(seq_id, band rank)`, in
/// arrival order — `width` wide: the highest band's prompts, first come first.
/// Empty when that is fewer than two, since a gang of one is no gang.
pub(super) fn form_gang(longs: &[(usize, usize)], width: usize) -> Vec<usize> {
    let Some(top) = longs.iter().map(|&(_, rank)| rank).max() else {
        return Vec::new();
    };
    let gang: Vec<usize> = longs
        .iter()
        .filter(|&&(_, rank)| rank == top)
        .map(|&(seq, _)| seq)
        .take(width)
        .collect();
    if gang.len() < 2 {
        return Vec::new();
    }
    gang
}

/// What a group of `budget` rows leaves its prompts beside `section_rows` of
/// live sections: the rest, and never less than a chunk, so a queue of sections
/// cannot hold a prompt out of the group that carries them.
pub(super) fn prompt_budget(budget: usize, section_rows: usize) -> usize {
    budget
        .saturating_sub(section_rows)
        .max(CHUNK_SIZE.min(budget))
}

/// The rows each of `remaining` advances in one forward of `budget` rows, the
/// members flagged `in_gang` being a gang.
///
/// With no gang member the budget is filled in order
/// ([`fill_in_order`]). Otherwise the members before the first gang member —
/// the shorts — are filled in order from the budget less the gang's reserve
/// (the lesser of its remainder and half the budget); the gang shares what they
/// left in proportion to each member's remainder ([`proportional`]); and the
/// members after it are filled in order from what is left, each only while it
/// gets its `least` rows.
pub(super) fn gang_shares(
    remaining: &[usize],
    in_gang: &[bool],
    budget: usize,
    least: usize,
) -> Vec<usize> {
    let Some(first) = in_gang.iter().position(|&g| g) else {
        return fill_in_order(remaining, budget, least);
    };
    let mut out = vec![0; remaining.len()];
    let gang: Vec<usize> = (first..remaining.len()).filter(|&i| in_gang[i]).collect();
    let gang_remaining: Vec<usize> = gang.iter().map(|&i| remaining[i]).collect();
    let reserve = gang_remaining.iter().sum::<usize>().min(budget / 2);
    let mut left = budget;
    if first > 0 {
        let ahead = fill_in_order(&remaining[..first], budget.saturating_sub(reserve), least);
        out[..first].copy_from_slice(&ahead);
        left = left.saturating_sub(ahead.iter().sum());
    }
    let shares = proportional(&gang_remaining, left, least);
    for (&i, &s) in gang.iter().zip(&shares) {
        out[i] = s;
    }
    left = left.saturating_sub(shares.iter().sum());
    for i in (first..remaining.len()).filter(|&i| !in_gang[i]) {
        let take = remaining[i].min(left);
        if take == 0 || take < remaining[i].min(least) {
            break;
        }
        out[i] = take;
        left -= take;
    }
    out
}

/// `left` rows shared in proportion to `remaining`, so every member reaches its
/// end in the same forward; whole when they all fit. A member whose share would
/// fall under its `least` rows (or its whole remainder, if smaller) drops out
/// from the last, and waits for the next forward.
pub(super) fn proportional(remaining: &[usize], left: usize, least: usize) -> Vec<usize> {
    let mut out = vec![0; remaining.len()];
    if left == 0 {
        return out;
    }
    for k in (1..=remaining.len()).rev() {
        let prefix = &remaining[..k];
        let total: u128 = prefix.iter().map(|&r| r as u128).sum();
        if total == 0 {
            return out;
        }
        let shares: Vec<usize> = prefix
            .iter()
            .map(|&r| ((r as u128 * left as u128 / total) as usize).min(r))
            .collect();
        if k == 1 || shares.iter().zip(prefix).all(|(&s, &r)| s >= r.min(least)) {
            out[..k].copy_from_slice(&shares);
            return out;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A gang is as wide as its step stays within twice a lone one.**
    #[test]
    fn a_gang_widens_while_its_step_stays_within_twice_a_lone_one() {
        let rising = |n: usize| 10.0 + n as f64;
        assert_eq!(gang_width(13, rising), 12);
        assert_eq!(gang_width(5, rising), 5);
        assert_eq!(gang_width(13, |n| 10.0 * n as f64), 2);
        assert_eq!(gang_width(13, |_| 7.0), 13);
    }

    /// Unlearned — a lone step that costs nothing or is not a number — is no
    /// gang, and nothing waiting is nothing to gang.
    #[test]
    fn an_unlearned_step_forms_no_gang() {
        assert_eq!(gang_width(8, |_| 0.0), 1);
        assert_eq!(gang_width(8, |_| f64::NAN), 1);
        assert_eq!(gang_width(0, |n| n as f64), 0);
    }

    /// **The highest band's long prompts, first come first.**
    #[test]
    fn a_gang_is_the_top_bands_longs_in_arrival_order() {
        let longs = [(7, 0), (3, 1), (9, 0), (4, 1), (5, 1)];
        assert_eq!(form_gang(&longs, 2), vec![3, 4]);
        assert_eq!(form_gang(&longs, 8), vec![3, 4, 5]);
        assert_eq!(form_gang(&[(7, 0), (9, 0)], 4), vec![7, 9]);
    }

    /// A gang of one is no gang.
    #[test]
    fn a_gang_of_one_is_none() {
        assert_eq!(form_gang(&[(3, 1), (4, 0)], 4), Vec::<usize>::new());
        assert_eq!(form_gang(&[(3, 1), (4, 1)], 1), Vec::<usize>::new());
        assert_eq!(form_gang(&[], 4), Vec::<usize>::new());
    }

    /// **With no gang, the budget is filled in order**, as the interactive
    /// regime fills it.
    #[test]
    fn without_a_gang_the_budget_fills_in_order() {
        assert_eq!(
            gang_shares(&[100, 2000, 2000], &[false; 3], 1000, 32),
            vec![100, 900, 0]
        );
    }

    /// **A gang shares in proportion to its remainders**, so 14k, 14k and 7k
    /// pages against 1,024 rows advance 409, 409 and 204 and stay in step.
    #[test]
    fn a_gang_shares_in_proportion_to_its_remainders() {
        assert_eq!(
            gang_shares(&[14_000, 14_000, 7_000], &[true; 3], 1024, 32),
            vec![409, 409, 204]
        );
    }

    /// **Shorts ride first, from what the gang's reserve leaves**: two shorts
    /// take 500 rows whole and the gang the 524 left; a long outside the gang
    /// waits.
    #[test]
    fn live_sections_keep_their_rows_and_prompts_keep_a_chunk() {
        assert_eq!(prompt_budget(3072, 0), 3072);
        assert_eq!(prompt_budget(3072, 149), 2923);
        assert_eq!(prompt_budget(3072, 3072), CHUNK_SIZE);
        assert_eq!(prompt_budget(3072, 9000), CHUNK_SIZE);
        assert_eq!(prompt_budget(16, 149), 16, "a budget under a chunk is all");
    }

    #[test]
    fn shorts_ride_ahead_of_the_gang_and_the_rest_behind() {
        assert_eq!(
            gang_shares(
                &[300, 200, 14_000, 14_000, 9_000],
                &[false, false, true, true, false],
                1024,
                32
            ),
            vec![300, 200, 262, 262, 0]
        );
    }

    /// **A burst of shorts takes no more than half**: the gang keeps its 512.
    #[test]
    fn a_burst_of_shorts_leaves_the_gang_half_the_budget() {
        assert_eq!(
            gang_shares(
                &[400, 400, 400, 14_000, 14_000],
                &[false, false, false, true, true],
                1024,
                32
            ),
            vec![400, 112, 0, 256, 256]
        );
    }

    /// A gang finishing leaves what it does not need to the prompts behind it.
    #[test]
    fn a_finishing_gang_leaves_the_rest_to_the_prompts_behind() {
        assert_eq!(
            gang_shares(&[100, 150, 5_000], &[true, true, false], 1024, 32),
            vec![100, 150, 774]
        );
    }

    /// A gang's reserve is only what it has left: two short gang members
    /// reserve 250 rows, and the shorts ahead take the rest.
    #[test]
    fn a_gang_reserves_only_its_remainder() {
        assert_eq!(
            gang_shares(&[900, 100, 150], &[false, true, true], 1024, 32),
            vec![774, 100, 150]
        );
    }

    /// Whole when everything fits; the tail drops out when a share would fall
    /// under a chunk; one member takes all there is.
    #[test]
    fn proportional_shares_drop_the_tail_under_a_chunk() {
        assert_eq!(proportional(&[100, 200], 1024, 32), vec![100, 200]);
        assert_eq!(proportional(&[1000, 1000, 1000], 64, 32), vec![32, 32, 0]);
        assert_eq!(proportional(&[1000], 64, 32), vec![64]);
        assert_eq!(proportional(&[1000, 1000], 0, 32), vec![0, 0]);
    }
}
