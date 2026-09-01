//! Unit tests for the normalization module — std-only, deterministic.

// The raw-score comparison is asserted alongside the normalized one so the
// before/after pair reads as a single claim, even though the raw half is
// constant.
#![allow(clippy::assertions_on_constants)]

use super::{ChildKey, NormConfig, NormalizationCache, ScopeKey};

fn approx(got: f32, want: f32, tol: f32) {
    assert!(
        (got - want).abs() <= tol,
        "expected ~{want}, got {got} (tol {tol})"
    );
}

/// Value normalize assigned to `k`.
fn val(v: &[(ChildKey, f32)], k: &ChildKey) -> f32 {
    v.iter().find(|(c, _)| c == k).expect("child present").1
}

/// Observe as a piece of evidence never seen before.
///
/// `observe` is idempotent per SOURCE, so a test about the fold itself has to
/// present distinct evidence each time — otherwise it is exercising the dedup
/// instead of the thing it means to. Tests about idempotency call `observe`
/// directly with a fixed source.
fn obs(cache: &mut NormalizationCache, scope: &ScopeKey, raw: &[(ChildKey, f32)]) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    cache.observe(scope, NEXT.fetch_add(1, Ordering::Relaxed), raw);
}

#[test]
fn hit_level_asymmetric_ewma() {
    // Default: alpha_up 0.30, alpha_dn 0.02, prior 400.
    // Distinct evidence per step — re-folding the same source is a no-op by
    // design, see `re_observing_the_same_evidence_changes_nothing`.
    let scope = ScopeKey::turn_group(1, 1);
    let cc = ChildKey::named("cc");
    let mut cache = NormalizationCache::default();

    // Rise fast: 400 + 0.30*(1000-400) = 580.
    obs(&mut cache, &scope, &[(cc.clone(), 1000.0)]);
    approx(cache.level_of(&scope, &cc).unwrap(), 580.0, 0.01);

    // Still rising, so still fast: 580 + 0.30*(900-580) = 676.
    obs(&mut cache, &scope, &[(cc.clone(), 900.0)]);
    approx(cache.level_of(&scope, &cc).unwrap(), 676.0, 0.01);

    // Decay slow: 676 + 0.02*(0-676) = 662.48.
    obs(&mut cache, &scope, &[(cc.clone(), 0.0)]);
    approx(cache.level_of(&scope, &cc).unwrap(), 662.48, 0.01);

    // **The zero moved the level but not the count.** The level is "how loud is
    // this child here", which a zero legitimately drags down; the count is "how
    // much has this scope taught me about this child", which a zero says nothing
    // about. Counting it would let a child the subdivision never saw score read
    // as warm — see `HitLevel::observe`.
    assert_eq!(cache.count_of(&scope, &cc), Some(2));
}

/// Re-observing evidence a scope has already folded changes nothing.
///
/// This is what lets the levels be rebuilt on EVERY load, from empty, against a
/// substrate already on disk, while the same scopes keep learning from live
/// traffic afterwards. Without it the two paths fight: a replay drags every level
/// toward whatever it re-feeds, so the levels a load reproduces differ from the
/// ones originally learned and the ranking they drive drifts on nothing but a
/// restart. It is also the hot-path fast case — one lookup and return.
#[test]
fn re_observing_the_same_evidence_changes_nothing() {
    const TURN: u64 = 7;
    let scope = ScopeKey::turn_group(1, 1);
    let cc = ChildKey::named("cc");
    let mut cache = NormalizationCache::default();

    cache.observe(&scope, TURN, &[(cc.clone(), 1000.0)]);
    let level = cache.level_of(&scope, &cc).unwrap();
    assert_eq!(cache.count_of(&scope, &cc), Some(1));

    // Replay the same turn many times over — including with a different score,
    // which must also be ignored: the turn has already been folded.
    for _ in 0..25 {
        cache.observe(&scope, TURN, &[(cc.clone(), 1000.0)]);
        cache.observe(&scope, TURN, &[(cc.clone(), 10.0)]);
    }
    approx(cache.level_of(&scope, &cc).unwrap(), level, 0.0);
    assert_eq!(
        cache.count_of(&scope, &cc),
        Some(1),
        "re-folding the same evidence must not count as new evidence",
    );

    // Different evidence still teaches — learning is not switched off.
    cache.observe(&scope, TURN + 1, &[(cc.clone(), 500.0)]);
    assert_ne!(cache.level_of(&scope, &cc).unwrap(), level);
    assert_eq!(cache.count_of(&scope, &cc), Some(2));
}

#[test]
fn normalize_puts_a_hit_at_the_scale() {
    let scope = ScopeKey::turn_group(1, 1);
    let cc = ChildKey::named("cc");
    let mut cache = NormalizationCache::default();

    // Warm cc to a known level (400 -> 580).
    obs(&mut cache, &scope, &[(cc.clone(), 1000.0)]);
    let level = cache.level_of(&scope, &cc).unwrap(); // 580

    // A probe AT the hit level normalizes to ~scale (1000). Single child ⇒ floor
    // = its own level, so denom = level exactly.
    let out = cache.normalize(&scope, &[(cc.clone(), level)]);
    approx(val(&out, &cc), 1000.0, 0.5);

    // A 2× hit locks on at ~2000; a half hit lands at ~500.
    approx(
        val(&cache.normalize(&scope, &[(cc.clone(), 2.0 * level)]), &cc),
        2000.0,
        1.0,
    );
    approx(
        val(&cache.normalize(&scope, &[(cc.clone(), 0.5 * level)]), &cc),
        500.0,
        0.5,
    );
}

#[test]
fn cold_child_normalizes_against_prior() {
    // A scope never observed: every child divides by the prior (400), floored at
    // floor_min (50). So a probe at the prior → scale; half the prior → scale/2.
    let cache = NormalizationCache::default();
    let scope = ScopeKey::turn_group(2, 2);
    let x = ChildKey::turn(0);

    approx(
        val(&cache.normalize(&scope, &[(x.clone(), 400.0)]), &x),
        1000.0,
        0.5,
    );
    approx(
        val(&cache.normalize(&scope, &[(x.clone(), 200.0)]), &x),
        500.0,
        0.5,
    );
}

#[test]
fn floor_min_caps_amplification_of_a_quiet_child() {
    // A tiny hit level would blow a partial match up; floor_min is the hard cap.
    // prior 10 < floor_min 100, so a cold child's denom is floor_min, not 10.
    let cfg = NormConfig {
        hit_prior: 10.0,
        floor_min: 100.0,
        ..Default::default()
    };
    let cache = NormalizationCache::new(cfg);
    let scope = ScopeKey::turn_group(1, 1);
    let x = ChildKey::turn(0);

    // With the floor: 1000 * 300 / 100 = 3000. Without it would be 1000*300/10 = 30000.
    let got = val(&cache.normalize(&scope, &[(x.clone(), 300.0)]), &x);
    approx(got, 3000.0, 0.5);
    assert!(
        got < 30000.0,
        "floor_min must cap the tiny-denominator blow-up"
    );
}

#[test]
fn scopes_are_isolated() {
    let a = ScopeKey::turn_group(1, 1);
    let b = ScopeKey::collection(2, "b");
    let x = ChildKey::turn(0);
    let mut cache = NormalizationCache::default();

    obs(&mut cache, &a, &[(x.clone(), 1000.0)]); // -> 580
    obs(&mut cache, &b, &[(x.clone(), 0.0)]); // 400 + 0.02*(0-400) = 392

    approx(cache.level_of(&a, &x).unwrap(), 580.0, 0.01);
    approx(cache.level_of(&b, &x).unwrap(), 392.0, 0.01);
}

#[test]
fn observe_retains_the_groups_sibling_timeline_scopes() {
    // Same group, two timelines. A belief group like `code_reading` has MANY
    // simultaneously-active timelines (one per file), each with its own learned
    // hit levels — observing one timeline must NOT evict the group's others.
    // (An earlier eviction-on-observe wiped every file's scope but the last,
    // degenerating normalization to a flat multiple of the raw score.)
    let old = ScopeKey::turn_group(7, 100);
    let new = ScopeKey::turn_group(7, 200);
    let other = ScopeKey::turn_group(9, 100);
    let c = ChildKey::turn(0);
    let mut cache = NormalizationCache::default();

    obs(&mut cache, &old, &[(c.clone(), 1000.0)]);
    obs(&mut cache, &other, &[(c.clone(), 1000.0)]);
    assert!(cache.level_of(&old, &c).is_some());

    obs(&mut cache, &new, &[(c.clone(), 1000.0)]);
    assert!(cache.level_of(&new, &c).is_some());
    assert!(
        cache.level_of(&old, &c).is_some(),
        "sibling-timeline scope must be retained (one scope per active file)"
    );
    assert!(
        cache.level_of(&other, &c).is_some(),
        "a different group's scope must be retained"
    );
}

/// The core promise: a generic loud child beats a specific one on RAW score, but
/// after learning hit levels from traffic, the specific child wins its own query.
#[test]
fn normalization_flips_generic_vs_specific() {
    let scope = ScopeKey::turn_group(1, 1);
    let root = ChildKey::named("root"); // loud: ~600 on every probe
    let cc = ChildKey::named("cc"); // specific: 500 on cc-probes, 40 otherwise
    let mut cache = NormalizationCache::default();

    // Warm-up traffic: mostly unrelated probes (cc quiet), two cc-probes.
    let other = [(root.clone(), 600.0), (cc.clone(), 40.0)];
    let cc_probe = [(root.clone(), 600.0), (cc.clone(), 500.0)];
    for probe in [&other, &other, &cc_probe, &other, &cc_probe, &other, &other] {
        obs(&mut cache, &scope, probe);
    }

    // root learned a high hit level (it's always loud); cc a middling one.
    assert!(cache.level_of(&scope, &root).unwrap() > cache.level_of(&scope, &cc).unwrap());

    // On a cc-probe, RAW ranks root (600) above cc (500)...
    assert!(600.0 > 500.0);
    // ...but NORMALIZED flips it — the specific target wins.
    let out = cache.normalize(&scope, &cc_probe);
    let (root_n, cc_n) = (val(&out, &root), val(&out, &cc));
    assert!(
        cc_n > root_n,
        "normalization must surface the specific child: cc {cc_n} vs root {root_n}"
    );
}

#[test]
fn deterministic() {
    let scope = ScopeKey::turn_group(1, 1);
    let x = ChildKey::turn(0);
    let seq = [10.0f32, 900.0, 30.0, 500.0, 5.0];

    let run = || {
        let mut c = NormalizationCache::default();
        // Distinct source per step: the sequence is the thing under test, and a
        // fresh cache must fold all of it.
        for (i, &r) in seq.iter().enumerate() {
            c.observe(&scope, i as u64, &[(x.clone(), r)]);
        }
        c.normalize(&scope, &[(x.clone(), 500.0)])
    };
    assert_eq!(run(), run());
}

/// Concept A.4: a positive per-child floor REPLACES the cold-start prior and
/// scope floor — a tiny fragment's high floor mutes it, while a quiet
/// full-window child's small floor lets a rare hit amplify (the design's
/// stand-out contract). No floors means today's behavior exactly.
#[test]
fn per_child_floors_replace_the_prior_and_scope_floor() {
    let scope = ScopeKey::turn_group(1, 1);
    let frag = ChildKey::named("fragment");
    let full = ChildKey::named("full");
    let cache = NormalizationCache::default();

    // Both unobserved, equal raw score 400. The caller's policy computed a
    // high size floor for the fragment (muted) and a small one for the
    // full-window child (its rare hit amplifies well past the flat prior).
    let raw = [(frag.clone(), 400.0), (full.clone(), 400.0)];
    let out = cache.normalize_with_floors(&scope, &raw, &[6400.0, 2.0]);
    approx(val(&out, &frag), 1000.0 * 400.0 / 6400.0, 0.1); // 62.5 — muted
    approx(val(&out, &full), 1000.0 * 400.0 / 2.0, 0.5); // 200_000 — stands out

    // A zero floor keeps the standard prior path for that child.
    let out = cache.normalize_with_floors(&scope, &raw, &[6400.0, 0.0]);
    approx(val(&out, &full), 1000.0 * 400.0 / 400.0, 0.1); // cold prior 400

    // Empty floors ⇒ identical to plain normalize.
    let a = cache.normalize_with_floors(&scope, &raw, &[]);
    let b = cache.normalize(&scope, &raw);
    assert_eq!(a, b);
}

/// The floored path normalizes against the child's observed TRAFFIC PEAK when
/// it exceeds the floor — a hit at the peak lands at the scale, and the prior
/// seed never inflates the denominator.
#[test]
fn traffic_peak_dominates_the_supplied_floor() {
    let scope = ScopeKey::turn_group(1, 1);
    let x = ChildKey::named("x");
    let mut cache = NormalizationCache::default();
    // Observed traffic peaked at 5000 (regardless of the EWMA's blend).
    obs(&mut cache, &scope, &[(x.clone(), 5000.0)]);
    obs(&mut cache, &scope, &[(x.clone(), 100.0)]);
    let out = cache.normalize_with_floors(&scope, &[(x.clone(), 5000.0)], &[100.0]);
    approx(val(&out, &x), 1000.0, 1.0);
    // A weak hit relative to that peak mutes proportionally.
    let out = cache.normalize_with_floors(&scope, &[(x.clone(), 500.0)], &[100.0]);
    approx(val(&out, &x), 100.0, 0.5);
}

/// Fold `n` distinct pieces of evidence at `raw` into `scope`, to bring a child
/// over the warm threshold a subdivided scope is asked for.
fn obs_n(cache: &mut NormalizationCache, scope: &ScopeKey, raw: &[(ChildKey, f32)], n: u32) {
    for _ in 0..n {
        obs(cache, scope, raw);
    }
}

/// **A subdivided scope normalizes only what it has learned; the rest falls back
/// to the parent.**
///
/// This is the whole safety property of subdividing by phase. Splitting the
/// traffic means some children are cold in the child scope for a long time, and
/// dividing by a learning-starved level does not blur the ranking — it inverts
/// it. So a cold child must come out on the PARENT's denominator, and a warm one
/// on the child scope's, in one call, in caller order.
#[test]
fn a_cold_child_in_a_subdivided_scope_falls_back_to_its_parent() {
    let parent = ScopeKey::collection(1, "tools");
    let child = ScopeKey::collection_phase(1, "tools", super::Phase::User);
    let warm = ChildKey::named("warm");
    let cold = ChildKey::named("cold");
    let mut cache = NormalizationCache::default();

    // The parent scores whole turns, so its hit level is ~3600.
    obs_n(
        &mut cache,
        &parent,
        &[(warm.clone(), 3600.0), (cold.clone(), 3600.0)],
        12,
    );
    // The phase scope reads only the user region — a fraction of the turn — so
    // the same hit lands at ~900 on its band.
    obs_n(&mut cache, &child, &[(warm.clone(), 900.0)], 12);

    let raw = [(warm.clone(), 900.0), (cold.clone(), 900.0)];
    let out = cache.normalize_with_fallback(&child, &parent, &raw, &[], 8);

    // Order is the caller's, not warm-then-cold.
    assert_eq!(out[0].0, warm);
    assert_eq!(out[1].0, cold);
    // `warm` is a full hit ON ITS OWN BAND: 900 against a learned level of ~900.
    approx(val(&out, &warm), 1000.0, 30.0);
    // `cold` is scored on the parent's ~3600 level — a quarter hit, NOT the 1000
    // it would have reached had the starved child scope been trusted.
    approx(val(&out, &cold), 253.0, 20.0);
}

/// A scope the subdivision has never touched at all normalizes wholly on the
/// parent — the boot case, before any phase traffic exists.
#[test]
fn an_unseen_subdivided_scope_is_exactly_the_parent() {
    let parent = ScopeKey::collection(1, "tools");
    let child = ScopeKey::collection_phase(1, "tools", super::Phase::Response);
    let a = ChildKey::named("a");
    let mut cache = NormalizationCache::default();
    obs_n(&mut cache, &parent, &[(a.clone(), 900.0)], 10);

    let raw = [(a.clone(), 450.0)];
    assert_eq!(
        cache.normalize_with_fallback(&child, &parent, &raw, &[], 8),
        cache.normalize(&parent, &raw),
    );
}

/// The threshold is per CHILD, not per scope: a scope that is warm on one member
/// must not thereby vouch for a member it has barely seen.
#[test]
fn the_warm_threshold_is_counted_per_child() {
    let parent = ScopeKey::collection(1, "tools");
    let child = ScopeKey::collection_phase(1, "tools", super::Phase::User);
    let a = ChildKey::named("a");
    let b = ChildKey::named("b");
    let mut cache = NormalizationCache::default();
    obs_n(
        &mut cache,
        &parent,
        &[(a.clone(), 3600.0), (b.clone(), 3600.0)],
        12,
    );
    obs_n(&mut cache, &child, &[(a.clone(), 900.0)], 12);
    // `b` appears in the child scope, but only 3 times — under the threshold.
    obs_n(&mut cache, &child, &[(b.clone(), 900.0)], 3);

    let raw = [(a.clone(), 900.0), (b.clone(), 900.0)];
    let out = cache.normalize_with_fallback(&child, &parent, &raw, &[], 8);
    approx(val(&out, &a), 1000.0, 30.0);
    // `b` fell back despite being present in the child scope.
    approx(val(&out, &b), 253.0, 20.0);
}

/// **A child that never SCORES in a scope stays cold, however many folds the
/// scope sees.**
///
/// The production shape, which `the_warm_threshold_is_counted_per_child` misses
/// by observing each child in its own call: the scorer hands `observe` every
/// section on every fold, most of them zero. Counting those made the per-child
/// gate a per-scope one — after `min_observations` folds every member read warm,
/// including ones the subdivision had never seen score, and each was then
/// normalized on a denominator learned from nothing.
#[test]
fn a_child_that_only_ever_scores_zero_never_goes_warm() {
    let parent = ScopeKey::collection(1, "tools");
    let child = ScopeKey::collection_phase(1, "tools", super::Phase::User);
    let scorer = ChildKey::named("scorer");
    let silent = ChildKey::named("silent");
    let mut cache = NormalizationCache::default();

    obs_n(
        &mut cache,
        &parent,
        &[(scorer.clone(), 3600.0), (silent.clone(), 3600.0)],
        12,
    );
    // 20 folds of the WHOLE slice — far past the threshold — but `silent` only
    // ever appears with a zero.
    obs_n(
        &mut cache,
        &child,
        &[(scorer.clone(), 900.0), (silent.clone(), 0.0)],
        20,
    );

    assert!(
        cache.count_of(&child, &scorer).unwrap_or(0) >= 8,
        "the scoring child learned from those folds",
    );
    assert_eq!(
        cache.count_of(&child, &silent),
        Some(0),
        "a child that only ever scored zero learned nothing, however many folds \
         the scope saw",
    );

    let raw = [(scorer.clone(), 900.0), (silent.clone(), 900.0)];
    let out = cache.normalize_with_fallback(&child, &parent, &raw, &[], 8);
    approx(val(&out, &scorer), 1000.0, 30.0);
    assert!(
        val(&out, &silent) < 400.0,
        "the never-scoring child must fall back to the parent, not be normalized \
         on a level it never taught; got {}",
        val(&out, &silent),
    );
}

/// Per-child floors follow their child through the warm/cold split — the
/// reassembly must not shift them by one when the halves are interleaved.
#[test]
fn fallback_keeps_each_childs_floor_with_that_child() {
    let parent = ScopeKey::collection(1, "tools");
    let child = ScopeKey::collection_phase(1, "tools", super::Phase::User);
    let cold_a = ChildKey::named("a");
    let warm_b = ChildKey::named("b");
    let cold_c = ChildKey::named("c");
    let mut cache = NormalizationCache::default();
    // Only the middle child is warm in the phase scope, so warm and cold
    // interleave and a naive reassembly puts the floors on the wrong children.
    obs_n(&mut cache, &child, &[(warm_b.clone(), 100.0)], 12);

    let raw = [
        (cold_a.clone(), 400.0),
        (warm_b.clone(), 400.0),
        (cold_c.clone(), 400.0),
    ];
    let floors = [4000.0, 8000.0, 2000.0];
    let out = cache.normalize_with_fallback(&child, &parent, &raw, &floors, 8);
    assert_eq!(
        out.iter().map(|(k, _)| k.clone()).collect::<Vec<_>>(),
        vec![cold_a.clone(), warm_b.clone(), cold_c.clone()]
    );
    // Floored path: denominator is max(observed peak, floor). No peak for the
    // cold pair, and `b`'s peak (100) is under its floor, so each is its floor.
    approx(val(&out, &cold_a), 1000.0 * 400.0 / 4000.0, 0.1);
    approx(val(&out, &warm_b), 1000.0 * 400.0 / 8000.0, 0.1);
    approx(val(&out, &cold_c), 1000.0 * 400.0 / 2000.0, 0.1);
}

/// A phase scope is a different scope, not a relabelling of the collection: what
/// the `user` lens learns must never move the `response` lens or the parent.
#[test]
fn phase_scopes_do_not_leak_into_each_other() {
    let parent = ScopeKey::collection(1, "tools");
    let user = ScopeKey::collection_phase(1, "tools", super::Phase::User);
    let resp = ScopeKey::collection_phase(1, "tools", super::Phase::Response);
    let x = ChildKey::named("x");
    let mut cache = NormalizationCache::default();
    obs_n(&mut cache, &user, &[(x.clone(), 900.0)], 12);

    let raw = [(x.clone(), 900.0)];
    approx(val(&cache.normalize(&user, &raw), &x), 1000.0, 60.0);
    // Untouched scopes still sit on the cold-start prior of 400.
    approx(
        val(&cache.normalize(&resp, &raw), &x),
        1000.0 * 900.0 / 400.0,
        1.0,
    );
    approx(
        val(&cache.normalize(&parent, &raw), &x),
        1000.0 * 900.0 / 400.0,
        1.0,
    );
}
