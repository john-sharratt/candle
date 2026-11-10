//! Unit tests for the normalization module — std-only, deterministic.
//!
//! The band unit is `1000 / ln 100 = 217.147` per nat of surprise, and the bin
//! axis is `8 ln(1 + x)`; the expected values below are worked from those.

use super::{ChildKey, NormalizationCache, ScopeKey};

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

/// Observe as a piece of evidence never seen before, about no known child.
///
/// `observe` is idempotent per SOURCE, so a test about the fold itself has to
/// present distinct evidence each time — otherwise it is exercising the dedup
/// instead of the thing it means to. Tests about idempotency call `observe`
/// directly with a fixed source.
fn obs(cache: &NormalizationCache, scope: &ScopeKey, raw: &[(ChildKey, f32)]) {
    obs_about(cache, scope, raw, None);
}

/// [`obs`] for a probe known to be about `answer`.
fn obs_about(
    cache: &NormalizationCache,
    scope: &ScopeKey,
    raw: &[(ChildKey, f32)],
    answer: Option<&ChildKey>,
) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    cache.observe(
        scope,
        NEXT.fetch_add(1, Ordering::Relaxed),
        raw,
        answer,
        T_REF,
    );
}

/// Probe length used by every test that is not *about* probe length: equal to
/// `NormConfig::probe_t_ref`, so the reference scaling is exactly 1.0.
const T_REF: usize = 256;

/// `x` at exactly `bins` on the bin axis.
fn at(bins: f32) -> f32 {
    (bins / 8.0).exp() - 1.0
}

/// **A score beyond 99 of 100 background samples is a full hit**: against 99
/// samples of 1.0, the top of their bin is `ln 100` nats — exactly 1000 — and
/// 1.0 itself is 54.5% into its bin, `ln(100 / 46.03) = 0.776` nats, 168.5.
#[test]
fn a_score_beyond_ninety_nine_in_a_hundred_is_a_full_hit() {
    let scope = ScopeKey::collection(1, "tools");
    let x = ChildKey::named("x");
    let cache = NormalizationCache::default();
    for _ in 0..99 {
        obs(&cache, &scope, &[(x.clone(), 1.0)]);
    }
    approx(
        val(&cache.normalize(&scope, &[(x.clone(), at(6.0))], T_REF), &x),
        1000.0,
        0.1,
    );
    approx(
        val(&cache.normalize(&scope, &[(x.clone(), 1.0)], T_REF), &x),
        168.5,
        0.2,
    );
    assert_eq!(
        val(&cache.normalize(&scope, &[(x.clone(), 0.0)], T_REF), &x),
        0.0,
        "a zero is no evidence"
    );
}

/// **The warm-up's order no longer decides a member's score.** Three tools,
/// eight exemplars each, folded one tool after another as the warm-up does:
/// each exemplar scores 100 on its own tool and 10 on the others. Each tool's
/// background is the sixteen 10s of the other tools' exemplars, so a hit of
/// 100 reads the same for all three — `ln 17` plus 2.115 nats past the top
/// bin, 1074.5 — and folding them in the reverse order changes nothing. The
/// hit-level EWMA this replaced left the first tool's level decayed to 10 and
/// the last tool's at 100, a tenfold difference set by the catalog order.
#[test]
fn the_warm_up_order_does_not_decide_a_score() {
    let scope = ScopeKey::collection(1, "tools");
    let tools = [
        ChildKey::named("a"),
        ChildKey::named("b"),
        ChildKey::named("c"),
    ];
    let warm = |order: &[usize]| {
        let cache = NormalizationCache::default();
        for &own in order {
            for _ in 0..8 {
                let probe: Vec<(ChildKey, f32)> = tools
                    .iter()
                    .enumerate()
                    .map(|(i, t)| (t.clone(), if i == own { 100.0 } else { 10.0 }))
                    .collect();
                obs_about(&cache, &scope, &probe, Some(&tools[own]));
            }
        }
        let hit: Vec<(ChildKey, f32)> = tools.iter().map(|t| (t.clone(), 100.0)).collect();
        cache.normalize(&scope, &hit, T_REF)
    };
    let forward = warm(&[0, 1, 2]);
    assert_eq!(forward, warm(&[2, 1, 0]));
    for (_, score) in &forward {
        approx(*score, 1074.5, 0.2);
    }
}

/// **A loud child is muted by its own loudness, and the specific one wins its
/// probe.** `root` scores 600 on every probe; `cc` 40 on twenty unrelated ones
/// and 500 on its own two. On a cc probe raw ranks root first; read against
/// their backgrounds, root's 600 is its median (43.2) and cc's 500 is past
/// twenty of its 22 samples (587.9).
#[test]
fn a_loud_child_is_muted_and_the_specific_one_wins() {
    let scope = ScopeKey::pool(1);
    let root = ChildKey::timeline(1);
    let cc = ChildKey::timeline(2);
    let cache = NormalizationCache::default();
    let other = [(root.clone(), 600.0), (cc.clone(), 40.0)];
    let cc_probe = [(root.clone(), 600.0), (cc.clone(), 500.0)];
    for i in 0..22 {
        obs(&cache, &scope, if i % 11 == 5 { &cc_probe } else { &other });
    }
    let out = cache.normalize(&scope, &cc_probe, T_REF);
    approx(val(&out, &root), 43.2, 0.2);
    approx(val(&out, &cc), 587.9, 0.2);
}

/// **A probe's known answer is left out of its background**; every other child
/// it scored is taught.
#[test]
fn a_known_answer_is_not_its_own_background() {
    let scope = ScopeKey::collection(1, "tools");
    let a = ChildKey::named("a");
    let b = ChildKey::named("b");
    let cache = NormalizationCache::default();
    obs_about(
        &cache,
        &scope,
        &[(a.clone(), 100.0), (b.clone(), 10.0)],
        Some(&a),
    );
    assert_eq!(cache.background_of(&scope, &a), None);
    assert_eq!(cache.background_of(&scope, &b), Some(1));
}

/// **A child too new for a background of its own reads against the scope's
/// pool**: the same score lands where it does for the child the pool was
/// taught by.
#[test]
fn a_new_child_reads_against_the_pool() {
    let scope = ScopeKey::collection(1, "tools");
    let known = ChildKey::named("known");
    let new = ChildKey::named("new");
    let cache = NormalizationCache::default();
    for i in 0..20 {
        obs(&cache, &scope, &[(known.clone(), 5.0 + i as f32)]);
    }
    let seen = val(
        &cache.normalize(&scope, &[(known.clone(), 12.0)], T_REF),
        &known,
    );
    let fresh = val(
        &cache.normalize(&scope, &[(new.clone(), 12.0)], T_REF),
        &new,
    );
    assert!(seen > 0.0);
    assert_eq!(fresh, seen);
}

/// **A scope never taught reads each child against the probe's own scores**:
/// 100 against `{100, 10, 0}` is 7.9% into the top sample's bin, `ln(4 /
/// 1.079)` nats, 284.5; 10 is below the 100 and 81.7% of its own bin, `ln(4 /
/// 2.817)`, 76.1.
#[test]
fn an_untaught_scope_reads_the_probe_against_itself() {
    let cache = NormalizationCache::default();
    let scope = ScopeKey::turn_group(2, 2);
    let (a, b, c) = (ChildKey::turn(0), ChildKey::turn(1), ChildKey::turn(2));
    let out = cache.normalize(
        &scope,
        &[(a.clone(), 100.0), (b.clone(), 10.0), (c.clone(), 0.0)],
        T_REF,
    );
    approx(val(&out, &a), 284.5, 0.2);
    approx(val(&out, &b), 76.1, 0.2);
    assert_eq!(val(&out, &c), 0.0);
}

/// Re-observing evidence a scope has already folded changes nothing.
///
/// This is what lets the backgrounds be rebuilt on EVERY load, from empty,
/// against a substrate already on disk, while the same scopes keep learning
/// from live traffic afterwards. It is also the hot-path fast case — one lookup
/// and return.
#[test]
fn re_observing_the_same_evidence_changes_nothing() {
    const TURN: u64 = 7;
    let scope = ScopeKey::turn_group(1, 1);
    let cc = ChildKey::named("cc");
    let cache = NormalizationCache::default();

    cache.observe(&scope, TURN, &[(cc.clone(), 1000.0)], None, T_REF);
    for _ in 0..25 {
        cache.observe(&scope, TURN, &[(cc.clone(), 1000.0)], None, T_REF);
        cache.observe(&scope, TURN, &[(cc.clone(), 10.0)], None, T_REF);
    }
    assert_eq!(cache.background_of(&scope, &cc), Some(1));
    cache.observe(&scope, TURN + 1, &[(cc.clone(), 500.0)], None, T_REF);
    assert_eq!(
        cache.background_of(&scope, &cc),
        Some(2),
        "new evidence teaches"
    );
}

#[test]
fn scopes_are_isolated() {
    let a = ScopeKey::turn_group(1, 1);
    let b = ScopeKey::collection(2, "b");
    let x = ChildKey::turn(0);
    let cache = NormalizationCache::default();
    obs(&cache, &a, &[(x.clone(), 1000.0)]);
    obs(&cache, &a, &[(x.clone(), 1000.0)]);
    obs(&cache, &b, &[(x.clone(), 0.0)]);
    assert_eq!(cache.background_of(&a, &x), Some(2));
    assert_eq!(cache.background_of(&b, &x), Some(1));
}

#[test]
fn observe_retains_the_groups_sibling_timeline_scopes() {
    // Same group, two timelines. A belief group like `code_reading` has MANY
    // simultaneously-active timelines (one per file), each with its own
    // backgrounds — observing one timeline must NOT evict the group's others.
    let old = ScopeKey::turn_group(7, 100);
    let new = ScopeKey::turn_group(7, 200);
    let other = ScopeKey::turn_group(9, 100);
    let c = ChildKey::turn(0);
    let cache = NormalizationCache::default();

    obs(&cache, &old, &[(c.clone(), 1000.0)]);
    obs(&cache, &other, &[(c.clone(), 1000.0)]);
    obs(&cache, &new, &[(c.clone(), 1000.0)]);
    assert!(cache.background_of(&new, &c).is_some());
    assert!(
        cache.background_of(&old, &c).is_some(),
        "sibling-timeline scope must be retained (one scope per active file)"
    );
    assert!(
        cache.background_of(&other, &c).is_some(),
        "a different group's scope must be retained"
    );
}

// ── The band is a RATE, not a total ────────────────────────────────────────
//
// A raw provenance score is a sum over probe tokens, and the read path runs at
// every reprojection with whatever window is current while the write path folds
// a capped tail at seal. These pin the unit.

/// Twenty background samples of 600 over the reference window, and a scope of
/// `x` taught them.
fn taught_at_the_reference(group: u64) -> (NormalizationCache, ScopeKey, ChildKey) {
    let scope = ScopeKey::turn_group(group, group);
    let x = ChildKey::named("x");
    let cache = NormalizationCache::default();
    for i in 0..20 {
        obs(&cache, &scope, &[(x.clone(), 300.0 + 20.0 * i as f32)]);
    }
    (cache, scope, x)
}

/// Equal agreement PER TOKEN lands on the same band value, whatever the probe
/// length.
#[test]
fn the_band_is_invariant_to_probe_length() {
    let (cache, scope, x) = taught_at_the_reference(9);
    let long = val(&cache.normalize(&scope, &[(x.clone(), 600.0)], 256), &x);
    let short = val(&cache.normalize(&scope, &[(x.clone(), 150.0)], 64), &x);
    assert!(long > 0.0);
    approx(short, long, 1e-3);
}

/// The converse: an equal raw TOTAL from a shorter probe is four times the
/// agreement per token, so it reads as the score 2400 over the reference —
/// past every sample, where 600 is not.
#[test]
fn an_equal_raw_total_from_a_shorter_probe_scores_higher() {
    let (cache, scope, x) = taught_at_the_reference(10);
    let long = val(&cache.normalize(&scope, &[(x.clone(), 600.0)], 256), &x);
    let short = val(&cache.normalize(&scope, &[(x.clone(), 600.0)], 64), &x);
    let as_reference = val(&cache.normalize(&scope, &[(x.clone(), 2400.0)], 256), &x);
    assert!(short > long);
    approx(short, as_reference, 1e-3);
}

/// The WRITE path scales too: a background taught by a short probe and one
/// taught by a long probe at the same rate read every score alike.
#[test]
fn the_write_path_learns_a_rate_not_a_total() {
    let from_long = ScopeKey::turn_group(11, 11);
    let from_short = ScopeKey::turn_group(12, 12);
    let x = ChildKey::named("x");
    let cache = NormalizationCache::default();
    for i in 0..10u64 {
        cache.observe(&from_long, i, &[(x.clone(), 600.0)], None, 256);
        cache.observe(&from_short, i, &[(x.clone(), 150.0)], None, 64);
    }
    for probe in [300.0, 600.0, 1200.0] {
        assert_eq!(
            cache.normalize(&from_long, &[(x.clone(), probe)], 256),
            cache.normalize(&from_short, &[(x.clone(), probe)], 256),
        );
    }
}

/// A scan with no probe produces no scores to scale, and must not reach the
/// band as an infinity — the length is treated as one token.
#[test]
fn a_zero_length_probe_stays_finite() {
    let scope = ScopeKey::turn_group(13, 13);
    let x = ChildKey::named("x");
    let cache = NormalizationCache::default();
    let got = val(&cache.normalize(&scope, &[(x.clone(), 0.0)], 0), &x);
    assert!(got.is_finite(), "a zero-length probe produced {got}");
    assert_eq!(got, 0.0);
}
