//! Selection rule implementation — which turns survive into the projection.
//!
//! # Two-phase selection
//!
//! Selection is run **twice** during a projection (see [`super::project::run`]):
//!
//! ```text
//!   ┌─────────────────────────────────────────────────────────────────┐
//!   │ Phase 1: UNBOUNDED selection                                    │
//!   │   apply_selection(rule, threshold, turns, budget=None, ...)     │
//!   │                                                                  │
//!   │   Determines each group's NATURAL consumption — the set of      │
//!   │   turns the rule would pick if budget were unlimited. Used to   │
//!   │   compute the per-group score and to cap flexbox allocation.    │
//!   └─────────────────────────────────────────────────────────────────┘
//!                                ▼
//!   ┌─────────────────────────────────────────────────────────────────┐
//!   │ flexbox_distribute → per-group token budget                     │
//!   └─────────────────────────────────────────────────────────────────┘
//!                                ▼
//!   ┌─────────────────────────────────────────────────────────────────┐
//!   │ Phase 2: BOUNDED selection                                      │
//!   │   apply_selection(rule, threshold, turns, budget=Some(n), ...)  │
//!   │                                                                  │
//!   │   Trims the natural set to fit the allocated token budget.     │
//!   │   Trim policy varies per rule (lowest-score, drop, historical-  │
//!   │   only).                                                        │
//!   └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Selection vs emission ordering
//!
//! Selection orders **by score** (highest scoring deserves the window).
//! Emission orders **by insertion** (the LLM reads sequentially and
//! reordering by score destroys dialogue coherence). The split is deliberate:
//! all selection rules in this module return their result sorted by
//! [`TurnKey`] regardless of the score-based order they were chosen in.
//!
//! # Sequence rule
//!
//! ```text
//!   group.turns = [t0  t1  t2  t3  t4  t5  t6  t7]   (insertion order)
//!                  ──────────────────  ─────────────
//!                       historical          recent
//!                                          (last `recent` turns,
//!                                           inviolate — no threshold,
//!                                           never trimmed by budget)
//!
//!   selection: (top-k by score from historical) + (all of recent)
//!   emission:  insertion order
//! ```

use super::ids::{TimelineId, TurnKey};
use super::schema::SelectionRule;
#[cfg(test)]
use super::schema::LINEAGE_BASE;

/// What one step nearer in the ancestry adds to a lineage turn's rank — far
/// below any score difference that means anything, enough to order ties.
pub(super) const LINEAGE_RANK_STEP: f32 = 1e-3;

/// Apply a selection rule to a group's turns.
///
/// # Arguments
///
/// - `rule` — the [`SelectionRule`] declared for the group
/// - `threshold` — turns with `score < threshold` are filtered out as
///   ineligible (except `Sequence`'s recent-N which bypasses the gate)
/// - `turns` — all candidate turns, **in insertion order** (ascending
///   `TurnKey`), each as `(TurnKey, score)`
/// - `budget_tokens`:
///   - `None` — unbounded phase-1 pass (pick natural set)
///   - `Some(n)` — bounded phase-2 pass; trim to fit `n` tokens per the
///     rule's trim policy
/// - `token_counts` — closure mapping `TurnKey → token count`
///
/// # Trim policy under budget
///
/// | Rule              | When `total > budget`                                |
/// |-------------------|------------------------------------------------------|
/// | `AlwaysVisible`   | Drop lowest-scored turns until total fits           |
/// | `TopK`            | Trim from lowest-scored end of the top-k set        |
/// | `Single`          | Drop entirely if the chosen turn doesn't fit        |
/// | `Sequence`    | Trim historical (lowest-scored first); recent is inviolate |
///
/// # Return
///
/// Always in **insertion order** (ascending `TurnKey`), even though
/// internal selection ordered by score.
pub fn apply_selection(
    rule: &SelectionRule,
    threshold: f32,
    turns: &[(TurnKey, f32)],
    budget_tokens: Option<usize>,
    token_counts: &dyn Fn(TurnKey) -> usize,
    own_timeline: Option<TimelineId>,
) -> Vec<TurnKey> {
    match rule {
        SelectionRule::AlwaysVisible => {
            select_always_visible(threshold, turns, budget_tokens, token_counts)
        }
        SelectionRule::TopK { k } => {
            select_top_k(*k, threshold, turns, budget_tokens, token_counts)
        }
        SelectionRule::Single => select_single(threshold, turns, budget_tokens, token_counts),
        // `Named` selects a collection member by name; turn groups have no
        // member names, so nothing survives. (Collections resolve `Named` in
        // `project::select_collection_sections`, not here.)
        SelectionRule::Named { .. } | SelectionRule::Offered { .. } => Vec::new(),
        // A working-set group is filled from the target's working set, whole
        // conversations sized by its own budget (`project::working_set_picks`);
        // it never reaches a per-turn rule.
        SelectionRule::WorkingSet { .. } => Vec::new(),
        SelectionRule::Sequence {
            recent,
            historical_top_k,
            lineage_base,
        } => select_conversation(
            *recent,
            *historical_top_k,
            *lineage_base,
            threshold,
            turns,
            budget_tokens,
            token_counts,
            own_timeline,
        ),
    }
}

// ── AlwaysVisible ─────────────────────────────────────────────────────────────

fn select_always_visible(
    threshold: f32,
    turns: &[(TurnKey, f32)],
    budget_tokens: Option<usize>,
    token_counts: &dyn Fn(TurnKey) -> usize,
) -> Vec<TurnKey> {
    // Filter by threshold, keep insertion order.
    let mut eligible: Vec<(TurnKey, f32)> = turns
        .iter()
        .filter(|(_, s)| *s >= threshold)
        .copied()
        .collect();

    if let Some(budget) = budget_tokens {
        trim_to_budget_low_score_first(&mut eligible, budget, token_counts);
    }

    eligible.iter().map(|(idx, _)| *idx).collect()
}

// ── TopK ──────────────────────────────────────────────────────────────────────

fn select_top_k(
    k: usize,
    threshold: f32,
    turns: &[(TurnKey, f32)],
    budget_tokens: Option<usize>,
    token_counts: &dyn Fn(TurnKey) -> usize,
) -> Vec<TurnKey> {
    // **No evidence, no seat.** A zero score means no probe has ever matched
    // the turn — the pre-belief path (fork/startup) scores everything 0 — and
    // ranking all-zero candidates is decided by the tie-break alone: ascending
    // key, i.e. ingestion order. That seeded the same arbitrary
    // earliest-ingested repo_map folders into every conversation ever forked,
    // which the model then adopted as its own dialogue history (proven with a
    // conversation whose only content was "hello"). Content that must appear
    // without evidence has two sanctioned homes — the group's `default:`
    // re-injection and the `AlwaysVisible` rule. An evidence-ranked rule
    // admits evidence, whatever its configured threshold.
    let mut eligible: Vec<(TurnKey, f32)> = turns
        .iter()
        .filter(|(_, s)| *s >= threshold && *s > 0.0)
        .copied()
        .collect();

    // Sort descending by score; ties broken by ascending TurnKey (earlier wins).
    eligible.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    eligible.truncate(k);

    if let Some(budget) = budget_tokens {
        trim_to_budget_low_score_first(&mut eligible, budget, token_counts);
    }

    // Emit in insertion order.
    eligible.sort_by_key(|(idx, _)| *idx);
    eligible.iter().map(|(idx, _)| *idx).collect()
}

// ── Single ────────────────────────────────────────────────────────────────────

fn select_single(
    threshold: f32,
    turns: &[(TurnKey, f32)],
    budget_tokens: Option<usize>,
    token_counts: &dyn Fn(TurnKey) -> usize,
) -> Vec<TurnKey> {
    // Same zero-score exclusion as `select_top_k`, same reasoning: `Single` is
    // an evidence-ranked rule, and with no evidence its winner is an artifact
    // of the tie-break, not a selection.
    let best = turns
        .iter()
        .filter(|(_, s)| *s >= threshold && *s > 0.0)
        .max_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.0.cmp(&a.0)) // tie: lower TurnKey wins
        });

    match best {
        None => vec![],
        Some((idx, _)) => {
            if let Some(budget) = budget_tokens {
                if token_counts(*idx) > budget {
                    return vec![]; // single-turn overflow → drop
                }
            }
            vec![*idx]
        }
    }
}

// ── Sequence ──────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn select_conversation(
    recent: usize,
    historical_top_k: usize,
    lineage_base: f32,
    threshold: f32,
    turns: &[(TurnKey, f32)],
    budget_tokens: Option<usize>,
    token_counts: &dyn Fn(TurnKey) -> usize,
    own_timeline: Option<TimelineId>,
) -> Vec<TurnKey> {
    // **Inherited turns are scored from a base, all the way up the ancestry**
    // (`docs/hit_driven_projection.md` §8.2).
    //
    // A turn on a timeline other than this conversation's own is here because
    // the lineage put it here (`Substrate::inherited_chain` walks `forked_from`
    // recursively, so this covers every ancestor, not just the immediate
    // parent). It is never ranked below `lineage_base`, so it does not age out
    // with the oldest history — `*s > 0.0` would drop an unscored turn
    // outright, and a conversation would forget the documents it was founded
    // on part-way through. But it is not exempt: when the window runs short it
    // is trimmed with the history, lowest first, and a turn the model keeps
    // reading — whose score its hits have lifted — outlasts one it does not.
    //
    // The count is bounded by `Substrate::INHERITED_CHAIN_TOKEN_CAP`.
    let (inherited, own): (Vec<(TurnKey, f32)>, Vec<(TurnKey, f32)>) = match own_timeline {
        Some(own_tl) => turns.iter().partition(|(key, _)| key.timeline != own_tl),
        None => (Vec::new(), turns.to_vec()),
    };
    // Ties at the base would trim by key, last first — the nearest ancestor
    // and every document's tail. A whisker per ancestor, nearest highest,
    // makes the trim take the oldest ancestor first and whole, as the chain's
    // cap does (`Substrate::inherited_chain`). The turns arrive in the chain's
    // order, oldest ancestor first, so an ancestor's rank is the order it
    // first appears in — never an assumption about how its id was minted.
    let mut ancestors: Vec<TimelineId> = Vec::new();
    for (k, _) in &inherited {
        if !ancestors.contains(&k.timeline) {
            ancestors.push(k.timeline);
        }
    }
    let inherited: Vec<(TurnKey, f32)> = inherited
        .into_iter()
        .map(|(k, s)| {
            let rank = ancestors.iter().position(|&t| t == k.timeline).unwrap_or(0) as f32;
            (k, s.max(lineage_base) + rank * LINEAGE_RANK_STEP)
        })
        .collect();
    let turns: &[(TurnKey, f32)] = &own;

    // Split: the last `recent` turns hold their seats regardless of score.
    let split_at = turns.len().saturating_sub(recent);
    let (older, recent_turns) = turns.split_at(split_at);

    // **Recent within the window, newest first, the newest always.** A seat
    // earned by recency still costs tokens, and a window that recency may
    // overrun is no window. On WhiteWorld (2026-10-27) a Maker's turns ran
    // ~15.6k tokens each — whole drafts and the table's verdicts on them — so
    // `recent: 16` under a 16k window projected ~170k tokens, 16 GB of hot K/V,
    // and image draws were refused beside it. The newest turn stays whatever it
    // costs: it is the exchange the conversation is in. The recent window is
    // charged first; the lineage competes with the history for what is left.
    let inviolate: Vec<(TurnKey, f32)> = match budget_tokens {
        Some(budget) => {
            let mut spent: usize = 0;
            let mut kept: Vec<(TurnKey, f32)> = Vec::new();
            for (i, &(key, score)) in recent_turns.iter().enumerate().rev() {
                let cost = token_counts(key);
                let newest = i + 1 == recent_turns.len();
                if !newest && spent + cost > budget {
                    break;
                }
                spent += cost;
                kept.push((key, score));
            }
            kept.reverse();
            kept
        }
        None => recent_turns.to_vec(),
    };

    // Historical: threshold-filtered, top-k by score — with the same zero-score
    // exclusion as `select_top_k`. The recent window below earns its seats by
    // recency, which *is* its evidence; the historical seats are evidence-ranked
    // like any TopK, and an all-zero field decided by the ascending-key
    // tie-break would seed the earliest-ingested turns into every fork.
    let mut historical: Vec<(TurnKey, f32)> = older
        .iter()
        .filter(|(_, s)| *s >= threshold && *s > 0.0)
        .copied()
        .collect();
    historical.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    historical.truncate(historical_top_k);
    // The lineage joins the history beside the top-k, never counted against
    // it: it is the conversation's founding, not a recall.
    historical.extend(inherited);

    // Budget pass: trim the history and the lineage (lowest-scored first) to
    // what the recent window leaves.
    if let Some(budget) = budget_tokens {
        let kept_tokens: usize = inviolate.iter().map(|(idx, _)| token_counts(*idx)).sum();
        let remaining = budget.saturating_sub(kept_tokens);
        trim_to_budget_low_score_first(&mut historical, remaining, token_counts);
    }

    // Combine and emit in insertion order.
    let mut selected: Vec<TurnKey> = historical
        .iter()
        .map(|(idx, _)| *idx)
        .chain(inviolate.iter().map(|(idx, _)| *idx))
        .collect();
    selected.sort();
    selected
}

// ── Budget trimmer ────────────────────────────────────────────────────────────

/// Remove turns (lowest-scored first) from `eligible` until the total token
/// count fits within `budget`. `eligible` must already be sorted by score desc.
pub(super) fn trim_to_budget_low_score_first(
    eligible: &mut Vec<(TurnKey, f32)>,
    budget: usize,
    token_counts: &dyn Fn(TurnKey) -> usize,
) {
    // Sort descending score so we can pop from the back (lowest score).
    eligible.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });

    loop {
        let total: usize = eligible.iter().map(|(idx, _)| token_counts(*idx)).sum();
        if total <= budget || eligible.is_empty() {
            break;
        }
        eligible.pop(); // remove lowest-scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::ids::{TimelineId, TurnIndex};

    /// All selection-rule tests operate within ONE conversation, so every key
    /// shares a timeline and orders by index — the single-timeline case the
    /// rules are defined against.
    fn t(n: u32) -> TurnKey {
        TurnKey::new(TimelineId::for_test(1), TurnIndex(n))
    }
    fn tc(_: TurnKey) -> usize {
        10
    } // uniform 10 tokens each

    /// The conversation being projected.
    fn own() -> TimelineId {
        TimelineId::for_test(1)
    }

    /// A turn on an ANCESTOR's timeline — inherited through `forked_from`.
    fn anc(tl: u64, n: u32) -> TurnKey {
        TurnKey::new(TimelineId::for_test(tl), TurnIndex(n))
    }

    /// An inherited turn survives even with a zero score and no seat in the
    /// recent window — the case that silently dropped the priming chain once a
    /// conversation grew past `recent`.
    #[test]
    fn an_inherited_turn_outranks_the_recent_window_and_the_score_filter() {
        let turns = vec![
            (anc(9, 0), 0.0),
            (t(0), 0.9),
            (t(1), 0.9),
            (t(2), 0.9),
            (t(3), 0.9),
        ];
        let r = apply_selection(
            &SelectionRule::Sequence {
                recent: 2,
                historical_top_k: 0,
                lineage_base: LINEAGE_BASE,
            },
            0.0,
            &turns,
            None,
            &tc,
            Some(own()),
        );
        assert!(
            r.contains(&anc(9, 0)),
            "a zero-scored ancestor turn is structural, not a retrieval candidate: {r:?}",
        );
        assert!(r.contains(&t(2)) && r.contains(&t(3)), "recent window kept");
    }

    /// Recursive: every ancestor in the lineage is ranked from the base, not
    /// just the nearest one.
    #[test]
    fn every_ancestor_in_the_lineage_is_kept_from_the_base() {
        let turns = vec![
            (anc(7, 0), 0.0),
            (anc(8, 0), 0.0),
            (anc(9, 0), 0.0),
            (t(0), 0.9),
            (t(1), 0.9),
        ];
        let r = apply_selection(
            &SelectionRule::Sequence {
                recent: 1,
                historical_top_k: 0,
                lineage_base: LINEAGE_BASE,
            },
            0.0,
            &turns,
            None,
            &tc,
            Some(own()),
        );
        for a in [anc(7, 0), anc(8, 0), anc(9, 0)] {
            assert!(r.contains(&a), "ancestor {a:?} dropped from {r:?}");
        }
    }

    /// **The lineage gives way only to what outscores its base**: with room
    /// for one more turn beside the recent window, a history turn whose score
    /// (lifted by the model's own hits) passes the base takes the seat, and
    /// the lineage keeps it against anything below.
    #[test]
    fn the_lineage_gives_way_only_to_a_stronger_turn() {
        let rule = SelectionRule::Sequence {
            recent: 1,
            historical_top_k: 8,
            lineage_base: LINEAGE_BASE,
        };
        let turns = vec![(anc(9, 0), 0.0), (t(0), 2_400.0), (t(1), 0.9)];
        let r = apply_selection(&rule, 0.0, &turns, Some(20), &tc, Some(own()));
        assert_eq!(
            r,
            vec![t(0), t(1)],
            "the stronger turn takes the seat: {r:?}"
        );
        let turns = vec![(anc(9, 0), 0.0), (t(0), 600.0), (t(1), 0.9)];
        let r = apply_selection(&rule, 0.0, &turns, Some(20), &tc, Some(own()));
        assert!(r.contains(&anc(9, 0)) && !r.contains(&t(0)), "{r:?}");
    }

    /// **A tight budget gives up the oldest ancestor first, and whole**: three
    /// unscored ancestors of two turns each, room for two of them beside the
    /// recent turn, keep the two nearest.
    #[test]
    fn a_tight_budget_drops_the_oldest_ancestor_whole() {
        let turns = vec![
            (anc(7, 0), 0.0),
            (anc(7, 1), 0.0),
            (anc(8, 0), 0.0),
            (anc(8, 1), 0.0),
            (anc(9, 0), 0.0),
            (anc(9, 1), 0.0),
            (t(0), 0.9),
        ];
        let rule = SelectionRule::Sequence {
            recent: 1,
            historical_top_k: 0,
            lineage_base: LINEAGE_BASE,
        };
        let r = apply_selection(&rule, 0.0, &turns, Some(50), &tc, Some(own()));
        assert_eq!(
            r,
            vec![t(0), anc(8, 0), anc(8, 1), anc(9, 0), anc(9, 1)],
            "{r:?}"
        );
    }

    /// The budget trims the history below the base before the lineage.
    #[test]
    fn a_tight_budget_trims_history_before_touching_the_lineage() {
        let turns = vec![(anc(9, 0), 0.0), (t(0), 0.4), (t(1), 0.5), (t(2), 0.9)];
        let r = apply_selection(
            &SelectionRule::Sequence {
                recent: 1,
                historical_top_k: 8,
                lineage_base: LINEAGE_BASE,
            },
            0.0,
            &turns,
            // Room for the ancestor and the recent turn only.
            Some(20),
            &tc,
            Some(own()),
        );
        assert!(r.contains(&anc(9, 0)), "lineage kept under budget: {r:?}");
        assert!(r.contains(&t(2)), "recent window kept under budget: {r:?}");
        assert!(!r.contains(&t(0)), "historical trimmed first: {r:?}");
    }

    /// The recent window fits the budget newest-first, and the newest turn is
    /// kept even when it alone overruns it.
    #[test]
    fn the_recent_window_fits_the_budget_newest_first() {
        let turns = vec![(t(0), 0.9), (t(1), 0.9), (t(2), 0.9), (t(3), 0.9)];
        let rule = SelectionRule::Sequence {
            recent: 4,
            historical_top_k: 0,
            lineage_base: LINEAGE_BASE,
        };
        // Room for two of the 10-token turns.
        let r = apply_selection(&rule, 0.0, &turns, Some(25), &tc, Some(own()));
        assert_eq!(r, vec![t(2), t(3)]);
        let r = apply_selection(&rule, 0.0, &turns, Some(5), &tc, Some(own()));
        assert_eq!(r, vec![t(3)], "the newest turn is the exchange itself");
        let r = apply_selection(&rule, 0.0, &turns, None, &tc, Some(own()));
        assert_eq!(r.len(), 4, "no budget, no trim");
    }

    /// With no target timeline every turn is treated as the conversation's
    /// own — the single-timeline behaviour the other rules are defined against.
    #[test]
    fn without_a_target_timeline_nothing_is_treated_as_inherited() {
        let turns = vec![(t(0), 0.0), (t(1), 0.9), (t(2), 0.9)];
        let r = apply_selection(
            &SelectionRule::Sequence {
                recent: 1,
                historical_top_k: 0,
                lineage_base: LINEAGE_BASE,
            },
            0.0,
            &turns,
            None,
            &tc,
            None,
        );
        assert_eq!(r, vec![t(2)], "only the recent window survives: {r:?}");
    }

    #[test]
    fn always_visible_all_pass() {
        let turns = vec![(t(0), 0.5), (t(1), 0.8), (t(2), 0.3)];
        let r = apply_selection(&SelectionRule::AlwaysVisible, 0.0, &turns, None, &tc, None);
        assert_eq!(r, vec![t(0), t(1), t(2)]);
    }

    #[test]
    fn always_visible_threshold_filters() {
        let turns = vec![(t(0), 0.5), (t(1), 0.8), (t(2), 0.1)];
        let r = apply_selection(&SelectionRule::AlwaysVisible, 0.3, &turns, None, &tc, None);
        assert_eq!(r, vec![t(0), t(1)]);
    }

    #[test]
    fn always_visible_budget_trims_lowest_score() {
        let turns = vec![(t(0), 0.9), (t(1), 0.5), (t(2), 0.3)];
        // 3 * 10 = 30; budget 20 → drop lowest-scored (t(2))
        let r = apply_selection(
            &SelectionRule::AlwaysVisible,
            0.0,
            &turns,
            Some(20),
            &tc,
            None,
        );
        assert_eq!(r, vec![t(0), t(1)]);
    }

    /// **Evidence-ranked rules never seat zero-score members**, whatever the
    /// configured threshold. With every candidate at 0 (the pre-belief path:
    /// fork and startup, before any probe exists) the "ranking" is the
    /// tie-break alone — ingestion order — which is how the same arbitrary
    /// repo_map folders ended up in every conversation's history. Must-show
    /// content goes through `default:` or `AlwaysVisible`, not through an
    /// evidence rule with no evidence.
    #[test]
    fn zero_score_is_never_selectable_by_evidence_rules() {
        let turns = vec![(t(0), 0.0), (t(1), 0.0), (t(2), 0.0)];
        let r = apply_selection(&SelectionRule::TopK { k: 3 }, 0.0, &turns, None, &tc, None);
        assert!(
            r.is_empty(),
            "TopK over all-zero scores must select nothing"
        );
        let r = apply_selection(&SelectionRule::Single, 0.0, &turns, None, &tc, None);
        assert!(
            r.is_empty(),
            "Single over all-zero scores must select nothing"
        );

        // Mixed: only the evidenced member seats.
        let turns = vec![(t(0), 0.0), (t(1), 0.4), (t(2), 0.0)];
        let r = apply_selection(&SelectionRule::TopK { k: 3 }, 0.0, &turns, None, &tc, None);
        assert_eq!(r, vec![t(1)]);

        // Sequence: the historical seats are evidence-ranked too — all-zero
        // older turns must select nothing beyond the recent window...
        let turns = vec![(t(0), 0.0), (t(1), 0.0), (t(2), 0.0), (t(3), 0.0)];
        let rule = SelectionRule::Sequence {
            recent: 1,
            historical_top_k: 2,
            lineage_base: LINEAGE_BASE,
        };
        let r = apply_selection(&rule, 0.0, &turns, None, &tc, None);
        assert_eq!(
            r,
            vec![t(3)],
            "only the recent window seats; recency is its evidence, zero-score history has none"
        );

        // ...while an evidenced older turn still earns a historical seat.
        let turns = vec![(t(0), 0.0), (t(1), 0.4), (t(2), 0.0), (t(3), 0.0)];
        let r = apply_selection(&rule, 0.0, &turns, None, &tc, None);
        assert_eq!(r, vec![t(1), t(3)]);
    }

    #[test]
    fn top_k_basic() {
        let turns = vec![(t(0), 0.3), (t(1), 0.9), (t(2), 0.6)];
        let r = apply_selection(&SelectionRule::TopK { k: 2 }, 0.0, &turns, None, &tc, None);
        // top 2: t(1)=0.9, t(2)=0.6 → emission insertion order
        assert_eq!(r, vec![t(1), t(2)]);
    }

    #[test]
    fn top_k_ties_earlier_wins() {
        let turns = vec![(t(0), 0.7), (t(1), 0.7), (t(2), 0.7)];
        let r = apply_selection(&SelectionRule::TopK { k: 2 }, 0.0, &turns, None, &tc, None);
        assert_eq!(r, vec![t(0), t(1)]);
    }

    #[test]
    fn top_k_threshold_gate() {
        let turns = vec![(t(0), 0.9), (t(1), 0.1), (t(2), 0.8)];
        let r = apply_selection(&SelectionRule::TopK { k: 3 }, 0.5, &turns, None, &tc, None);
        assert_eq!(r, vec![t(0), t(2)]);
    }

    #[test]
    fn top_k_budget_trims_low_end() {
        let turns = vec![(t(0), 0.9), (t(1), 0.7), (t(2), 0.5)];
        // top 3, budget 20 → keep t(0)+t(1)
        let r = apply_selection(
            &SelectionRule::TopK { k: 3 },
            0.0,
            &turns,
            Some(20),
            &tc,
            None,
        );
        assert_eq!(r, vec![t(0), t(1)]);
    }

    #[test]
    fn single_picks_highest() {
        let turns = vec![(t(0), 0.3), (t(1), 0.9), (t(2), 0.6)];
        let r = apply_selection(&SelectionRule::Single, 0.0, &turns, None, &tc, None);
        assert_eq!(r, vec![t(1)]);
    }

    #[test]
    fn single_tie_lower_index_wins() {
        let turns = vec![(t(0), 0.7), (t(1), 0.7)];
        let r = apply_selection(&SelectionRule::Single, 0.0, &turns, None, &tc, None);
        assert_eq!(r, vec![t(0)]);
    }

    #[test]
    fn single_budget_overflow_drops() {
        let turns = vec![(t(0), 0.9)];
        let tc_big = |_: TurnKey| 100usize;
        let r = apply_selection(&SelectionRule::Single, 0.0, &turns, Some(50), &tc_big, None);
        assert!(r.is_empty());
    }

    #[test]
    fn single_threshold_all_below() {
        let turns = vec![(t(0), 0.3), (t(1), 0.1)];
        let r = apply_selection(&SelectionRule::Single, 0.5, &turns, None, &tc, None);
        assert!(r.is_empty());
    }

    #[test]
    fn conversation_recent_inviolate() {
        let turns = vec![
            (t(0), 0.01), // old, low score — below threshold
            (t(1), 0.9),  // old, high score
            (t(2), 0.01), // recent, low score — still included
            (t(3), 0.01), // recent
        ];
        let rule = SelectionRule::Sequence {
            recent: 2,
            historical_top_k: 1,
            lineage_base: LINEAGE_BASE,
        };
        let r = apply_selection(&rule, 0.5, &turns, None, &tc, None);
        // t(0) below threshold, t(1) top-1 historical, t(2)+t(3) inviolate
        assert_eq!(r, vec![t(1), t(2), t(3)]);
    }

    #[test]
    fn conversation_budget_trims_historical_not_recent() {
        let turns = vec![
            (t(0), 0.9),  // old, high score
            (t(1), 0.8),  // old
            (t(2), 0.01), // recent
            (t(3), 0.01), // recent
        ];
        let rule = SelectionRule::Sequence {
            recent: 2,
            historical_top_k: 2,
            lineage_base: LINEAGE_BASE,
        };
        // 4 * 10 = 40; budget 30; inviolate = 20; remaining for historical = 10
        // historical: t(0) 10 tokens fits, t(1) would be 20 total → drop t(1)
        let r = apply_selection(&rule, 0.0, &turns, Some(30), &tc, None);
        assert_eq!(r, vec![t(0), t(2), t(3)]);
    }

    #[test]
    fn conversation_recent_past_the_budget_keeps_only_the_newest() {
        // A recent window that alone exceeds the budget keeps its newest turn.
        let turns = vec![(t(0), 0.9), (t(1), 0.5)];
        let rule = SelectionRule::Sequence {
            recent: 2,
            historical_top_k: 1,
            lineage_base: LINEAGE_BASE,
        };
        let r = apply_selection(&rule, 0.0, &turns, Some(5), &tc, None);
        assert_eq!(r, vec![t(1)]);
    }

    #[test]
    fn conversation_emission_insertion_order() {
        let turns = vec![
            (t(0), 0.7), // old historical
            (t(1), 0.3), // old historical (lower score)
            (t(2), 0.1), // recent
            (t(3), 0.9), // recent
        ];
        let rule = SelectionRule::Sequence {
            recent: 2,
            historical_top_k: 2,
            lineage_base: LINEAGE_BASE,
        };
        let r = apply_selection(&rule, 0.0, &turns, None, &tc, None);
        // Must be insertion order: t(0), t(1), t(2), t(3)
        assert_eq!(r, vec![t(0), t(1), t(2), t(3)]);
    }
}
