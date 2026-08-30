//! Behavioural tests for the VRAM partition — `docs/vram_partition_behavioural_tests.md`.
//!
//! **No device, no checkpoint, no forward pass.** Every planner these exercise is
//! pure arithmetic over small integers, which is what lets the whole catalogue
//! run in parallel in well under a second and therefore run on every change.
//!
//! # What these assert that the per-function tests do not
//!
//! The unit tests beside each planner assert that one call returns the right
//! number. Every partition defect found during development was instead an
//! interaction — a shrink path and a grow path that were each defensible alone,
//! a hysteresis floor that could not be reached by the grant sized to clear it —
//! and it was silent, because nothing errors when the engine merely runs slowly.
//!
//! So these are stated over *sweeps and trajectories*: a property that must hold
//! at **every** residency count, or across a whole sequence of budgets, rather
//! than at one point someone thought to check.

use candle_transformers::models::layer_stream::{
    eviction_order, layer_image, plan_zone, protection_order, warm_membership, FfnForm, LayerImage,
    LayerTensor, MixKind, Projection, PROJECTION_ALIGN,
};

use candle::quantized::GgmlDType;

// ── The cost model (`order.rs`) ─────────────────────────────────────────────
//
// A forward walks `0..N`. A missing layer is fetched into the floating cell, and
// the time available to hide that fetch is the run of resident layers before it:
//
//   window(gap) = (gap − 1) × T_COMPUTE
//   stall(gap)  = max(0, T_FETCH − window(gap))
//
// Measured on the 4090 Mobile for a ~200 MiB layer. Integers in microseconds, so
// the arithmetic is exact and the assertions have no tolerance to tune.

/// Per-layer compute, µs.
const T_COMPUTE: i64 = 1_400;
/// Per-layer PCIe fetch, µs.
const T_FETCH: i64 = 8_000;

/// Cyclic gaps between consecutive missing layers, in walk order.
fn gaps(missing: &[usize], n: usize) -> Vec<usize> {
    let mut m = missing.to_vec();
    m.sort_unstable();
    if m.is_empty() {
        return vec![];
    }
    let mut out = Vec::with_capacity(m.len());
    for w in m.windows(2) {
        out.push(w[1] - w[0]);
    }
    // The wrap: from the last missing layer round to the first.
    out.push(n - m[m.len() - 1] + m[0]);
    out
}

/// Total stall, µs, for one sweep with this missing set.
fn stall_of(missing: &[usize], n: usize) -> i64 {
    gaps(missing, n)
        .into_iter()
        .map(|g| (T_FETCH - (g as i64 - 1) * T_COMPUTE).max(0))
        .sum()
}

/// The stall an ideal (perfectly equal-gap) spread of `k` misses would pay.
///
/// A lower bound rather than an achievable plan when `k` does not divide `n` —
/// which is the point: a bound that cannot be beaten is what a real order should
/// be measured against.
fn stall_lower_bound(k: usize, n: usize) -> i64 {
    if k == 0 {
        return 0;
    }
    // Jensen: with Σgap = n fixed and stall convex, equal gaps minimise the sum.
    let g = n as f64 / k as f64;
    let per = (T_FETCH as f64 - (g - 1.0) * T_COMPUTE as f64).max(0.0);
    (per * k as f64).round() as i64
}

// ── Model shapes ────────────────────────────────────────────────────────────

/// A layer image whose `total` is `units × PROJECTION_ALIGN`.
///
/// Built through the real `layer_image`, so the placement rules — alignment,
/// ordering, the padding tail — are the production ones rather than a fixture's
/// idea of them.
fn image(units: usize) -> LayerImage {
    let roles = [
        LayerTensor::Wqkv,
        LayerTensor::Wz,
        LayerTensor::WOut,
        LayerTensor::FfnGateUp,
        LayerTensor::FfnDown,
    ];
    let rest = units - (roles.len() - 1);
    let p: Vec<Projection> = roles
        .iter()
        .enumerate()
        .map(|(i, &role)| Projection {
            role,
            shape: [32, 128],
            dtype: GgmlDType::Q4_KO,
            payload: PROJECTION_ALIGN * if i == 0 { rest } else { 1 },
            extent: PROJECTION_ALIGN * if i == 0 { rest } else { 1 },
        })
        .collect();
    layer_image(MixKind::DeltaNet, FfnForm::Fused, &p).unwrap()
}

/// A uniform stack: `n` layers of `units` each.
fn uniform(n: usize, units: usize) -> Vec<LayerImage> {
    (0..n).map(|_| image(units)).collect()
}

/// A ragged stack, alternating two sizes — the real 27B is not uniform, and a
/// planner that only ever sees equal images cannot be trusted with one that is
/// not (dense packing exists precisely because they differ).
fn ragged(n: usize, a: usize, b: usize) -> Vec<LayerImage> {
    (0..n)
        .map(|i| image(if i % 3 == 0 { a } else { b }))
        .collect()
}

/// The shapes every sweep runs over, named so a failure says which.
fn shapes() -> Vec<(&'static str, Vec<LayerImage>, usize)> {
    vec![
        ("uniform-64x8", uniform(64, 8), 2),
        ("uniform-48x8", uniform(48, 8), 2),
        ("uniform-32x12", uniform(32, 12), 2),
        ("uniform-16x6", uniform(16, 6), 1),
        ("ragged-64", ragged(64, 10, 8), 2),
        ("ragged-47", ragged(47, 9, 7), 2),
        ("uniform-8x6", uniform(8, 6), 1),
        ("uniform-100x8", uniform(100, 8), 3),
    ]
}

// ═══════════════════════════════════════════════════════════════════════════
// Family D — dense residency and PCIe latency hiding
// ═══════════════════════════════════════════════════════════════════════════

/// **D3 — max gap ≤ 2⌈N/k⌉ at every prefix.**
///
/// Asserted across every shape and every count, not at a chosen k. The order's
/// whole reason to exist is that the number of missing layers is not known at
/// build time — it moves at runtime as the boundary concedes — so a spread that
/// is good at one count and bunched at another is no spread at all.
#[test]
fn d3_every_prefix_is_evenly_spread_at_every_count() {
    for (name, imgs, pinned) in shapes() {
        let n = imgs.len();
        let order = eviction_order(n, pinned);
        for k in 1..=order.len() {
            let missing = &order[..k];
            let bound = 2 * n.div_ceil(k);
            let worst = gaps(missing, n).into_iter().max().unwrap_or(0);
            assert!(
                worst <= bound,
                "{name}: {k} missing of {n} left a gap of {worst}, bound {bound}"
            );
        }
    }
}

/// **D2 — total stall is within a small factor of the equal-gap optimum**, at
/// every residency count.
///
/// The bound is a *lower* bound (perfectly equal gaps, which an integer layout
/// cannot always achieve), so a small constant factor is the honest assertion.
/// What it catches is an order that is merely "spread-ish" — one whose stall
/// drifts away from optimal as k grows.
#[test]
fn d2_stall_stays_near_the_equal_gap_optimum() {
    for (name, imgs, pinned) in shapes() {
        let n = imgs.len();
        let order = eviction_order(n, pinned);
        for k in 1..=order.len() {
            let got = stall_of(&order[..k], n);
            let best = stall_lower_bound(k, n);
            // Slack for the integer layout: a gap of ⌊n/k⌋ against a real ⌈n/k⌉
            // costs one compute step per miss.
            let allowed = best + (k as i64) * T_COMPUTE;
            assert!(
                got <= allowed,
                "{name}: {k} missing of {n} stalls {got}µs, optimum {best}µs, allowed {allowed}µs"
            );
        }
    }
}

/// **D4 — stall is monotone non-increasing in residency.**
///
/// One more resident layer must never make the sweep slower. This is the
/// property a "good at some counts" order fails: it catches a prefix whose k+1
/// spread is worse than its k, which no single-count assertion can see.
#[test]
fn d4_one_more_resident_layer_never_costs_more_stall() {
    for (name, imgs, pinned) in shapes() {
        let n = imgs.len();
        let order = eviction_order(n, pinned);
        // Walking k downward is residency going *up*.
        for k in (1..order.len()).rev() {
            let more_missing = stall_of(&order[..k + 1], n);
            let fewer_missing = stall_of(&order[..k], n);
            assert!(
                fewer_missing <= more_missing,
                "{name}: going from {} to {k} missing of {n} raised stall {more_missing} → {fewer_missing}",
                k + 1
            );
        }
    }
}

/// **D5 — the spread beats a contiguous held prefix**, which is the natural
/// implementation and measurably the worst arrangement.
///
/// Kept as a regression guard: "hold the first k layers" is what anyone writes
/// first, and it concentrates every miss into one run where no transfer is
/// hidden at all.
#[test]
fn d5_the_spread_beats_a_contiguous_tail() {
    for (name, imgs, pinned) in shapes() {
        let n = imgs.len();
        let order = eviction_order(n, pinned);
        for k in 2..order.len() {
            let spread = stall_of(&order[..k], n);
            // The contiguous alternative: give up the last k evictable layers.
            let tail: Vec<usize> = (n - k..n).collect();
            let contiguous = stall_of(&tail, n);
            assert!(
                spread <= contiguous,
                "{name}: at {k} missing of {n} the spread ({spread}µs) lost to a contiguous tail ({contiguous}µs)"
            );
        }
    }
}

/// **D6 — growth extends the resident prefix.**
///
/// The layers taken back are the ones most recently given up, so growth restores
/// the spread the concession cost rather than filling the new ground with
/// whatever faults next. Stated as: the missing set at k+1 contains the missing
/// set at k.
#[test]
fn d6_growth_takes_back_the_layer_most_recently_given_up() {
    for (name, imgs, pinned) in shapes() {
        let n = imgs.len();
        let order = eviction_order(n, pinned);
        for k in 1..order.len() {
            let smaller: std::collections::BTreeSet<_> = order[..k].iter().copied().collect();
            let larger: std::collections::BTreeSet<_> = order[..k + 1].iter().copied().collect();
            assert!(
                smaller.is_subset(&larger),
                "{name}: the {k}-missing set is not a subset of the {}-missing set",
                k + 1
            );
        }
    }
}

/// **D1 — a model that fits is fully resident and never streams.**
///
/// The streaming machinery must cost nothing when unused: no floating cell, no
/// missing layers, nothing to fetch.
#[test]
fn d1_a_model_that_fits_streams_nothing() {
    for (name, imgs, pinned) in shapes() {
        let total: usize = imgs.iter().map(|i| i.total).sum();
        let end = 1 << 40;
        // Budget generously past the sum: whole residency must be reachable, and
        // reachable *without* also reserving a cell nothing will use.
        let plan = plan_zone(&imgs, pinned, end, total * 2).unwrap();
        assert_eq!(
            plan.resident(),
            imgs.len(),
            "{name}: a budget of twice the model left {} of {} layers homed",
            plan.resident(),
            imgs.len()
        );
        assert!(plan.is_whole(), "{name}: whole residency still kept a cell");
        assert!(
            plan.missing.is_empty(),
            "{name}: whole residency has misses"
        );
        assert_eq!(stall_of(&plan.missing, imgs.len()), 0);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Family A — span integrity, over the zone planner
// ═══════════════════════════════════════════════════════════════════════════

/// **A1/A2/A4 — placements are disjoint, descend from the top, and stay inside
/// the budget**, at every budget from the floor to whole residency.
///
/// A sweep rather than a sample: the planner packs densely at natural sizes, so
/// an off-by-one in the remainder shows up only at the budgets where a layer
/// *just* fits.
#[test]
fn a1_placements_are_disjoint_and_within_the_zone_at_every_budget() {
    let end: u64 = 1 << 40;
    for (name, imgs, pinned) in shapes() {
        let total: usize = imgs.iter().map(|i| i.total).sum();
        let widest = imgs.iter().map(|i| i.total).max().unwrap();
        let mut budget = imgs[..pinned].iter().map(|i| i.total).sum::<usize>() + widest;
        while budget <= total + widest {
            let Ok(plan) = plan_zone(&imgs, pinned, end, budget) else {
                budget += widest / 4 + 1;
                continue;
            };
            let mut spans: Vec<(u64, u64)> = plan
                .homes
                .iter()
                .flatten()
                .map(|p| (p.base, p.base + p.bytes as u64))
                .collect();
            if let Some(c) = plan.floating {
                spans.push((c.base, c.base + c.bytes as u64));
            }
            spans.sort_unstable();
            for w in spans.windows(2) {
                assert!(
                    w[0].1 <= w[1].0,
                    "{name} @ {budget}: placements overlap {:?} / {:?}",
                    w[0],
                    w[1]
                );
            }
            for (lo, hi) in &spans {
                assert!(
                    *lo >= plan.floor,
                    "{name} @ {budget}: placement below floor"
                );
                assert!(*hi <= end, "{name} @ {budget}: placement past the zone top");
            }
            assert!(
                plan.used_bytes(end) <= budget,
                "{name} @ {budget}: used {} bytes",
                plan.used_bytes(end)
            );
            budget += widest / 4 + 1;
        }
    }
}

/// **A9 — the zone never plans below its floor**, and says so rather than
/// producing a layout that cannot run.
///
/// The floor is the pinned head plus one streaming cell: below it the model
/// cannot execute at all, as opposed to merely executing slowly, and the planner
/// must refuse rather than return something.
#[test]
fn a9_a_budget_under_the_floor_is_refused_not_rounded() {
    let end: u64 = 1 << 40;
    for (name, imgs, pinned) in shapes() {
        let head: usize = imgs[..pinned].iter().map(|i| i.total).sum();
        let widest = imgs.iter().map(|i| i.total).max().unwrap();
        let floor = head + widest;
        assert!(
            plan_zone(&imgs, pinned, end, floor.saturating_sub(1)).is_err(),
            "{name}: a budget one byte under the floor was accepted"
        );
        assert!(
            plan_zone(&imgs, pinned, end, floor).is_ok(),
            "{name}: a budget exactly at the floor was refused"
        );
    }
}

/// **The pinned head is resident at every budget the planner accepts**, and sits
/// furthest from the frontier.
///
/// It is the one part of the model with no record in any tier — never loaded,
/// never evicted — so a plan that moves or drops it has nothing to fall back on.
#[test]
fn a6_the_pinned_head_is_resident_and_furthest_from_the_frontier() {
    let end: u64 = 1 << 40;
    for (name, imgs, pinned) in shapes() {
        let total: usize = imgs.iter().map(|i| i.total).sum();
        let widest = imgs.iter().map(|i| i.total).max().unwrap();
        let mut budget = imgs[..pinned].iter().map(|i| i.total).sum::<usize>() + widest;
        while budget <= total {
            if let Ok(plan) = plan_zone(&imgs, pinned, end, budget) {
                for l in 0..pinned {
                    assert!(
                        plan.homes[l].is_some(),
                        "{name} @ {budget}: pinned layer {l} has no home"
                    );
                }
                // Furthest from the frontier == highest address.
                let head_low = (0..pinned)
                    .map(|l| plan.homes[l].unwrap().base)
                    .min()
                    .unwrap();
                let others = plan
                    .homes
                    .iter()
                    .enumerate()
                    .filter(|(l, _)| *l >= pinned)
                    .filter_map(|(_, h)| h.map(|p| p.base))
                    .max();
                if let Some(o) = others {
                    assert!(
                        head_low > o,
                        "{name} @ {budget}: a streamable layer sits above the pinned head"
                    );
                }
            }
            budget += widest / 3 + 1;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Family C — no ratchet, over the planner's monotonicity
// ═══════════════════════════════════════════════════════════════════════════

/// **C1 (planner half) — a larger budget is a superset and never loses a layer.**
///
/// The ratchet's shape at this level: if growing the budget could drop a layer
/// that a smaller budget held, then a rise-and-fall in KV demand would leave the
/// zone worse than it started even with a perfect boundary policy above it.
#[test]
fn c1_a_larger_budget_never_loses_a_layer() {
    let end: u64 = 1 << 40;
    for (name, imgs, pinned) in shapes() {
        let total: usize = imgs.iter().map(|i| i.total).sum();
        let widest = imgs.iter().map(|i| i.total).max().unwrap();
        let head: usize = imgs[..pinned].iter().map(|i| i.total).sum();
        let step = widest / 5 + 1;
        let mut prev: Option<std::collections::BTreeSet<usize>> = None;
        let mut budget = head + widest;
        while budget <= total + widest {
            if let Ok(plan) = plan_zone(&imgs, pinned, end, budget) {
                let held: std::collections::BTreeSet<usize> = plan
                    .homes
                    .iter()
                    .enumerate()
                    .filter_map(|(l, h)| h.map(|_| l))
                    .collect();
                if let Some(p) = &prev {
                    assert!(
                        p.is_subset(&held),
                        "{name}: budget {budget} dropped layers a smaller budget held: {:?}",
                        p.difference(&held).collect::<Vec<_>>()
                    );
                }
                prev = Some(held);
            }
            budget += step;
        }
    }
}

/// **C2 (planner half) — residency is monotone non-decreasing in budget.**
#[test]
fn c2_residency_never_falls_as_the_budget_grows() {
    let end: u64 = 1 << 40;
    for (name, imgs, pinned) in shapes() {
        let total: usize = imgs.iter().map(|i| i.total).sum();
        let widest = imgs.iter().map(|i| i.total).max().unwrap();
        let head: usize = imgs[..pinned].iter().map(|i| i.total).sum();
        let mut last = 0usize;
        let mut budget = head + widest;
        while budget <= total + widest {
            if let Ok(plan) = plan_zone(&imgs, pinned, end, budget) {
                assert!(
                    plan.resident() >= last,
                    "{name}: residency fell from {last} at budget {budget}"
                );
                last = plan.resident();
            }
            budget += widest / 7 + 1;
        }
    }
}

/// **Dense packing is never worse than equal cells.**
///
/// The whole reason a slot is its own image's width rather than the maximum:
/// charging every layer the widest layer's size wasted a measured 18% on the
/// 27B. This asserts the property that motivated it, at every budget.
#[test]
fn dense_packing_holds_at_least_as_many_layers_as_equal_cells() {
    let end: u64 = 1 << 40;
    for (name, imgs, pinned) in shapes() {
        let widest = imgs.iter().map(|i| i.total).max().unwrap();
        let total: usize = imgs.iter().map(|i| i.total).sum();
        let head: usize = imgs[..pinned].iter().map(|i| i.total).sum();
        let mut budget = head + widest;
        while budget <= total {
            if let Ok(plan) = plan_zone(&imgs, pinned, end, budget) {
                // Like for like: an equal-cell zone spends one of its cells on
                // the floating slot too, so its *homed* count is one less than
                // the cells the budget buys. Comparing homes against cells would
                // charge dense packing for a slot equal cells also pay.
                let equal_cell_homes = (budget / widest).saturating_sub(1);
                assert!(
                    plan.resident() >= equal_cell_homes.min(imgs.len()),
                    "{name} @ {budget}: dense packing homed {} where equal cells home {equal_cell_homes}",
                    plan.resident()
                );
            }
            budget += widest / 6 + 1;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// The warm tier
// ═══════════════════════════════════════════════════════════════════════════

/// **The warm tier's membership is drawn so it stays correct as the zone grows.**
///
/// It is filled once and never redrawn, which is only sound if the layers it
/// holds are streamed under *every* prefix the tier can reach. A membership
/// drawn against a moving prefix is stale the moment the boundary moves.
#[test]
fn warm_membership_is_a_prefix_of_the_eviction_order_at_every_size() {
    for (name, imgs, pinned) in shapes() {
        let n = imgs.len();
        let order = eviction_order(n, pinned);
        for slots in 0..=order.len() {
            let members = warm_membership(n, pinned, slots);
            assert_eq!(members.len(), slots, "{name}: wrong membership size");
            let want: std::collections::BTreeSet<_> = order[..slots].iter().copied().collect();
            let got: std::collections::BTreeSet<_> = members.iter().copied().collect();
            assert_eq!(
                want, got,
                "{name} @ {slots} slots: membership is not the eviction order's prefix"
            );
            assert!(
                members.windows(2).all(|w| w[0] < w[1]),
                "{name}: membership is not ascending"
            );
        }
    }
}

/// **Protection is the exact reverse of eviction**, so the two orders can never
/// disagree about which layer is safest.
#[test]
fn protection_is_the_reverse_of_eviction_over_every_shape() {
    for (name, imgs, pinned) in shapes() {
        let n = imgs.len();
        let evict = eviction_order(n, pinned);
        let protect = protection_order(n, pinned);
        assert_eq!(
            protect.len(),
            n,
            "{name}: protection must cover every layer"
        );
        assert_eq!(
            &protect[..pinned],
            &(0..pinned).collect::<Vec<_>>()[..],
            "{name}: the pinned head must lead the protection order"
        );
        let tail: Vec<usize> = protect[pinned..].iter().rev().copied().collect();
        assert_eq!(tail, evict, "{name}: protection is not eviction reversed");
    }
}
