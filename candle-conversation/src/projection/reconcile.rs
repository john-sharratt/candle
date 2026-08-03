//! Budget reconciliation — CSS-flexbox-style proportional distribution.
//!
//! # The flexbox model
//!
//! Within a parent's budget, each child receives:
//!
//! ```text
//!   ideal_share = (child.priority / sum_of_unsaturated_priorities) * remaining
//!   allocated   = clamp(ideal_share, child.min, child.max)
//! ```
//!
//! After the first pass, summed allocations may not equal the parent's
//! budget — clamping to `max` leaves leftover. Leftover is redistributed by
//! re-running the share calculation against the remaining unsaturated
//! children. This loops until either:
//!
//! - All children are saturated (at their `max`)
//! - The unallocated remainder is `<= EPSILON_TOKENS`
//! - `MAX_ITERATIONS` is reached (safety cap)
//!
//! # Visual: how excess redistributes
//!
//! ```text
//!  Budget = 1000, three siblings with equal priority:
//!
//!   A: priority 1, max 100        ┃    Pass 1: A = min(333, 100) = 100  → SATURATED, freed 233
//!   B: priority 1, no max         ┃             B = 333
//!   C: priority 1, no max         ┃             C = 333
//!                                 ┃
//!                                 ┃    Pass 2: redistribute 233 over {B, C}
//!                                 ┃             B += 116                → 449
//!                                 ┃             C += 116                → 449
//!                                 ┃
//!                                 ┃    Final: A=100, B=449, C=449 (sum 998, freed 2 ≤ EPSILON)
//! ```
//!
//! # Dynamic shortfall
//!
//! When the sum of declared `min_percent`s exceeds the parent's budget
//! (statically infeasible mins are caught at construction; this handles the
//! dynamic case where percents resolve to absolute tokens that don't fit),
//! the distributor proportionally shrinks all mins to fit the available
//! budget. No error is raised — projection always produces a result.
//!
//! # Natural-consumption capping (used by [`super::project`])
//!
//! When the projection orchestrator distributes the turn budget across
//! layers and groups, it sets each `FlexItem`'s `max_tokens` to the group's
//! **natural consumption** (the token count of the unbounded-selected
//! turns). This makes a single flexbox pass equivalent to the doc's
//! iterative redistribution: a group that under-consumes saturates at its
//! natural cap, and the freed budget redistributes to other groups in the
//! same `flexbox_distribute` call.

use super::schema::Budget;

/// Cap on flex iterations to guarantee termination.
pub const MAX_ITERATIONS: usize = 8;

/// Convergence threshold — if unallocated budget falls to or below this many
/// tokens, redistribution stops. Tokens are integral, so 1 is effectively
/// "no rounding dust left".
pub const EPSILON_TOKENS: usize = 1;

// ── Core flex distribution ────────────────────────────────────────────────────

/// One node participating in a flexbox distribution pass.
///
/// Constructed via [`FlexItem::from_budget`] from a [`Budget`] declaration
/// plus the parent's resolved token budget — percentages turn into absolute
/// token counts at this point.
#[derive(Debug, Clone)]
pub struct FlexItem {
    /// Relative weight for proportional allocation. Must be > 0.
    pub priority: f32,
    /// Absolute token floor (resolved from `min_percent` × parent_budget).
    pub min_tokens: usize,
    /// Absolute token ceiling. `None` = uncapped. The projection orchestrator
    /// usually sets this to the group's natural consumption so freed budget
    /// auto-redistributes.
    pub max_tokens: Option<usize>,
}

impl FlexItem {
    /// Resolve a [`Budget`] (with percentage min/max) against an absolute
    /// `parent_budget` token count.
    ///
    /// `min_percent → floor()` (don't reserve more than declared)
    /// `max_percent → ceil()`  (don't cap below declared)
    pub fn from_budget(budget: &Budget, parent_budget: usize) -> Self {
        let min_tokens = budget
            .min_percent
            .map(|p| ((p / 100.0) * parent_budget as f32).floor() as usize)
            .unwrap_or(0);
        let max_tokens = budget
            .max_percent
            .map(|p| ((p / 100.0) * parent_budget as f32).ceil() as usize);
        FlexItem {
            priority: budget.priority,
            min_tokens,
            max_tokens,
        }
    }

    /// [`Self::from_budget`] under Concept B adaptivity
    /// (`docs/provenance_adaptive_projection.md` §4): the node's attention
    /// `mass` scales its flexbox priority
    /// (`priority × (1 + gain × mass/1000)`), and the adaptive `max_percent` —
    /// when declared — becomes the outer ceiling rail. The declared floor is
    /// untouched: zero mass rests at the static allocation, it never sinks a
    /// node below its `min_percent`.
    pub fn from_budget_with_mass(budget: &Budget, parent_budget: usize, mass: f32) -> Self {
        let mut item = Self::from_budget(budget, parent_budget);
        if let Some(ad) = &budget.adaptive {
            item.priority = ad.effective_priority(budget.priority, mass, 1000.0);
            if let Some(mp) = ad.max_percent {
                let rail = ((mp / 100.0) * parent_budget as f32).ceil() as usize;
                item.max_tokens = Some(item.max_tokens.map_or(rail, |m| m.max(rail)));
            }
        }
        item
    }
}

/// Distribute `budget` tokens across `items` proportionally to their
/// priority, respecting declared min/max bounds.
///
/// # Algorithm
///
/// 1. **Seed each item at its `min_tokens`** (with proportional shrink if
///    the sum of mins exceeds `budget`).
/// 2. **Iteratively distribute the remainder** to unsaturated items
///    proportionally to their priority. An item saturates when it hits
///    its `max_tokens` cap; saturated items leave the redistribution pool.
/// 3. **Stop** when remaining budget ≤ [`EPSILON_TOKENS`], when all items
///    are saturated, or when [`MAX_ITERATIONS`] is reached.
///
/// # Returns
///
/// One allocation per item, in the **same order as `items`**. Allocations
/// sum to at most `budget` (may be less when all items saturate before the
/// budget is exhausted; the leftover is "released" and the caller decides
/// what to do with it).
///
/// # Dynamic shortfall
///
/// When `sum(min_tokens) > budget`, mins are proportionally shrunk so they
/// fit (the floor is honoured "as much as possible", but not beyond the
/// available pool). No error.
pub fn flexbox_distribute(items: &[FlexItem], budget: usize) -> Vec<usize> {
    if items.is_empty() {
        return vec![];
    }

    let n = items.len();
    let mut allocations = vec![0usize; n];
    let mut saturated = vec![false; n]; // at max cap

    // Handle dynamic shortfall: scale mins proportionally if sum > budget.
    let total_min: usize = items.iter().map(|i| i.min_tokens).sum();
    let effective_mins: Vec<usize> = if total_min > budget {
        let scale = budget as f32 / total_min as f32;
        items
            .iter()
            .map(|i| (i.min_tokens as f32 * scale).floor() as usize)
            .collect()
    } else {
        items.iter().map(|i| i.min_tokens).collect()
    };

    // Seed allocations at their effective minimums.
    for (i, m) in effective_mins.iter().enumerate() {
        allocations[i] = *m;
        if let Some(max) = items[i].max_tokens {
            if *m >= max {
                saturated[i] = true;
                allocations[i] = (*m).min(max);
            }
        }
    }

    let mut remaining = budget.saturating_sub(allocations.iter().sum::<usize>());

    for _ in 0..MAX_ITERATIONS {
        if remaining <= EPSILON_TOKENS {
            break;
        }

        let unsaturated: Vec<usize> = (0..n).filter(|&i| !saturated[i]).collect();
        if unsaturated.is_empty() {
            break;
        }

        let total_priority: f32 = unsaturated.iter().map(|&i| items[i].priority).sum();
        if total_priority <= 0.0 {
            break;
        }

        let mut newly_freed = 0usize;

        for &i in &unsaturated {
            let share = ((items[i].priority / total_priority) * remaining as f32).floor() as usize;
            let candidate = allocations[i] + share;
            if let Some(max) = items[i].max_tokens {
                if candidate >= max {
                    newly_freed += candidate - max;
                    allocations[i] = max;
                    saturated[i] = true;
                } else {
                    allocations[i] = candidate;
                }
            } else {
                allocations[i] = candidate;
            }
        }

        let consumed: usize = allocations.iter().sum();
        remaining = budget.saturating_sub(consumed);

        // Any rounding dust that couldn't be assigned is irrelevant if ≤ EPSILON.
        if remaining <= EPSILON_TOKENS && newly_freed == 0 {
            break;
        }
    }

    allocations
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(priority: f32, min: usize, max: Option<usize>) -> FlexItem {
        FlexItem {
            priority,
            min_tokens: min,
            max_tokens: max,
        }
    }

    #[test]
    fn even_split() {
        let items = vec![item(1.0, 0, None), item(1.0, 0, None)];
        let allocs = flexbox_distribute(&items, 100);
        assert_eq!(allocs[0], allocs[1]);
        assert!(allocs.iter().sum::<usize>() <= 100);
    }

    #[test]
    fn weighted_split() {
        let items = vec![item(1.0, 0, None), item(3.0, 0, None)];
        let allocs = flexbox_distribute(&items, 100);
        // item[1] should get ~3x item[0]
        assert!(allocs[1] >= allocs[0] * 2);
    }

    #[test]
    fn min_respected() {
        let items = vec![item(1.0, 30, None), item(1.0, 0, None)];
        let allocs = flexbox_distribute(&items, 100);
        assert!(allocs[0] >= 30);
    }

    #[test]
    fn max_saturates_and_releases() {
        // item[0] capped at 20; remaining 80 should go to item[1]
        let items = vec![item(1.0, 0, Some(20)), item(1.0, 0, None)];
        let allocs = flexbox_distribute(&items, 100);
        assert_eq!(allocs[0], 20);
        assert!(allocs[1] > 50); // released budget reaches item[1]
    }

    #[test]
    fn shortfall_proportional_shrink() {
        // Total min = 150, budget = 100 → proportional shrink
        let items = vec![item(1.0, 100, None), item(1.0, 50, None)];
        let allocs = flexbox_distribute(&items, 100);
        let total: usize = allocs.iter().sum();
        assert!(total <= 100);
        // item[0] should get ~2/3, item[1] ~1/3
        assert!(allocs[0] > allocs[1]);
    }

    #[test]
    fn empty_input() {
        let allocs = flexbox_distribute(&[], 100);
        assert!(allocs.is_empty());
    }

    #[test]
    fn zero_budget() {
        let items = vec![item(1.0, 0, None), item(1.0, 0, None)];
        let allocs = flexbox_distribute(&items, 0);
        assert_eq!(allocs, vec![0, 0]);
    }

    #[test]
    fn single_item_gets_full_budget() {
        let items = vec![item(1.0, 0, None)];
        let allocs = flexbox_distribute(&items, 500);
        assert_eq!(allocs[0], 500);
    }
}
