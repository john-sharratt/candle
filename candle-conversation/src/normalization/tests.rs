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

#[test]
fn hit_level_asymmetric_ewma() {
    // Default: alpha_up 0.30, alpha_dn 0.02, prior 400.
    let scope = ScopeKey::turn_group(1, 1);
    let cc = ChildKey::named("cc");
    let mut cache = NormalizationCache::default();

    // Rise fast: 400 + 0.30*(1000-400) = 580.
    cache.observe(&scope, &[(cc.clone(), 1000.0)]);
    approx(cache.level_of(&scope, &cc).unwrap(), 580.0, 0.01);

    // Again: 580 + 0.30*(1000-580) = 706.
    cache.observe(&scope, &[(cc.clone(), 1000.0)]);
    approx(cache.level_of(&scope, &cc).unwrap(), 706.0, 0.01);

    // Decay slow: 706 + 0.02*(0-706) = 691.88.
    cache.observe(&scope, &[(cc.clone(), 0.0)]);
    approx(cache.level_of(&scope, &cc).unwrap(), 691.88, 0.01);

    assert_eq!(cache.count_of(&scope, &cc), Some(3));
}

#[test]
fn normalize_puts_a_hit_at_the_scale() {
    let scope = ScopeKey::turn_group(1, 1);
    let cc = ChildKey::named("cc");
    let mut cache = NormalizationCache::default();

    // Warm cc to a known level (400 -> 580).
    cache.observe(&scope, &[(cc.clone(), 1000.0)]);
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

    cache.observe(&a, &[(x.clone(), 1000.0)]); // -> 580
    cache.observe(&b, &[(x.clone(), 0.0)]); // 400 + 0.02*(0-400) = 392

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

    cache.observe(&old, &[(c.clone(), 1000.0)]);
    cache.observe(&other, &[(c.clone(), 1000.0)]);
    assert!(cache.level_of(&old, &c).is_some());

    cache.observe(&new, &[(c.clone(), 1000.0)]);
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
        cache.observe(&scope, probe);
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
        for &r in &seq {
            c.observe(&scope, &[(x.clone(), r)]);
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
    cache.observe(&scope, &[(x.clone(), 5000.0)]);
    cache.observe(&scope, &[(x.clone(), 100.0)]);
    let out = cache.normalize_with_floors(&scope, &[(x.clone(), 5000.0)], &[100.0]);
    approx(val(&out, &x), 1000.0, 1.0);
    // A weak hit relative to that peak mutes proportionally.
    let out = cache.normalize_with_floors(&scope, &[(x.clone(), 500.0)], &[100.0]);
    approx(val(&out, &x), 100.0, 0.5);
}
