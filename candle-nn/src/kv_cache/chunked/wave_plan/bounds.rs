//! Bounds the width searches so an unspendable budget ends them rather than
//! overflowing the cost arithmetic.
//!
//! The doubling in the width searches climbs until a width stops fitting,
//! which assumes some width exceeds the budget. A caller that prices against
//! ground the weight side might concede can hand in a budget no wave could
//! spend, and the search would then run until the row count overflowed. A
//! million rows is far past the widest wave the engine composes (8,192
//! tokens), so stopping there answers "wider than anything you can run"
//! before the overflow. Bisection rather than division keeps the search exact,
//! because every cost is a staircase of `div_ceil` steps.

/// Rows the width searches will consider before giving up on the budget.
///
/// The doubling in [`WavePlan::max_rows_for_tier`] and
/// [`WavePlan::max_rows_within`] climbs until a width does **not** fit, which
/// assumes a budget some width exceeds. A caller that prices against ground the
/// weight side might concede can hand in a budget no wave could spend, and then
/// the search runs until the row count overflows the cost arithmetic and panics
/// — a pure function faulting on a large argument, far from the caller that
/// chose it.
///
/// A million rows is four orders of magnitude past the widest wave this engine
/// composes (`MAX_PREFILL_TOKENS` is 8,192) and prices to tens of gigabytes, so
/// stopping here answers "wider than anything you can run" without ever
/// reaching the overflow.
pub(super) const ROW_SEARCH_CEILING: usize = 1 << 20;

/// The most rows whose `cost` stays within `budget`, for a `cost` that is
/// non-decreasing in rows; `0` when not even one row fits.
///
/// Bisected rather than divided: the `div_ceil` steps make every cost here a
/// staircase, so dividing the budget by a per-row average lands inside a step
/// and over-admits.
pub(super) fn max_rows_within_cost(budget: usize, cost: impl Fn(usize) -> usize) -> usize {
    if cost(1) > budget {
        return 0;
    }
    let mut lo = 1usize;
    let mut hi = 2usize;
    while hi < ROW_SEARCH_CEILING && cost(hi) <= budget {
        lo = hi;
        hi = hi.saturating_mul(2);
    }
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        if cost(mid) <= budget {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}
