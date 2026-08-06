//! Unit tests for the VRAM Governor. Everything here runs on CPU via the
//! scripted `FakeProbe` / `FakeBalloonAllocator` — no GPU required. The matrix
//! mirrors `docs/vram_governor_design.md` §14: budget/measurement, the relief
//! ladder (gentle-early, escalation, Critical-only sync, no-spin), the forecast,
//! expert budget (all- vs some-resident), external pressure, and the diag table.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use super::balloon::FakeBalloonAllocator;
use super::reading::fake::FakeVram;
use super::*;
use crate::{Error, Result};

const GIB: u64 = 1024 * 1024 * 1024;
const MIB: u64 = 1024 * 1024;

fn test_config() -> GovernorConfig {
    GovernorConfig {
        kv_floor_abs: 3 * GIB,
        kv_floor_pct: 0.15,
        scratch_margin: GIB,
        ladder: [
            LadderTier::new(2 * GIB, 0.040),
            LadderTier::new(3 * GIB / 2, 0.030),
            LadderTier::new(GIB, 0.015),
            LadderTier::new(GIB / 2, 0.005),
            LadderTier::new(0, 0.0),
        ],
        balloon_target_frac: 0.90,
        // Small enough not to bind on the 64 GiB test cards (the 0.90 fraction
        // stays the binding term), so existing target assertions hold.
        balloon_headroom_abs: 512 * MIB,
        balloon_floor: 512 * MIB,
        balloon_chunk: 256 * MIB,
        critical_min_interval_ms: 0, // deterministic: every Critical proceeds
    }
}

/// Build a governor over a fake vram cell, with C and weights set so the floor
/// and thresholds are live.
fn governed(vram: &FakeVram, c: u64, weights: u64) -> VramGovernor {
    let gov = VramGovernor::new(0, Box::new(vram.probe()), test_config());
    gov.set_capacity(c);
    gov.set_class(AllocClass::Weights, weights);
    gov
}

/// A model load records the session's WHOLE dense footprint, so recording it
/// again must replace it, not stack on top.
///
/// Added, a second load drives `C - weights` to zero and `kv_floor` collapses
/// to `kv_floor_abs` — the KV reserve silently disappears on exactly the card
/// that needs it.
#[test]
fn reloading_the_model_does_not_collapse_the_kv_floor() {
    let vram = FakeVram::new(64 * GIB, 64 * GIB);
    let gov = governed(&vram, 16 * GIB, 8 * GIB);
    let floor = gov.kv_floor();
    assert!(floor > gov.config().kv_floor_abs, "{floor}");

    gov.set_class(AllocClass::Weights, 8 * GIB);
    assert_eq!(gov.kv_floor(), floor, "a second load must not move the floor");
    assert_eq!(gov.class_reserved(AllocClass::Weights), 8 * GIB);
}

/// Register a relief closure that counts its calls and releases `release` bytes
/// of headroom (bounded by `want`), reporting the same as `evictable`.
fn counting_relief(
    gov: &VramGovernor,
    vram: &FakeVram,
    class: AllocClass,
    tier: Criticality,
    release: u64,
) -> Arc<AtomicU64> {
    let calls = Arc::new(AtomicU64::new(0));
    let c = calls.clone();
    let v = vram.clone();
    gov.register_relief(
        class,
        tier,
        move |req| {
            c.fetch_add(1, Ordering::Relaxed);
            let freed = req.want.min(release);
            v.release(freed);
            ReliefOutcome::new(freed)
        },
        move || release,
    );
    calls
}

// ── Budget & measurement ─────────────────────────────────────────────────────

#[test]
fn starvation_signal_accumulates_and_drains_once() {
    let vram = FakeVram::new(0, 73 * GIB);
    let gov = governed(&vram, 73 * GIB, 2 * GIB);
    // Nothing reported yet.
    assert_eq!(gov.take_starvation(), 0);
    // A burst of background-compressor failures accumulates.
    gov.signal_starvation();
    gov.signal_starvation();
    gov.signal_starvation();
    // Draining returns the count once, then resets — so the scheduler escalates
    // recovery exactly once per burst, not on every loop iteration.
    assert_eq!(gov.take_starvation(), 3);
    assert_eq!(gov.take_starvation(), 0);
}

#[test]
fn kv_floor_abs_plus_pct() {
    let vram = FakeVram::new(0, 73 * GIB);
    let gov = governed(&vram, 73 * GIB, 2 * GIB);
    // 3 GiB + 0.15 × (73 − 2) GiB = 3 + 10.65 = 13.65 GiB.
    let expected = 3 * GIB + ((0.15 * (71 * GIB) as f64) as u64);
    assert_eq!(gov.kv_floor(), expected);

    // Small card: absolute term dominates.
    let vram2 = FakeVram::new(0, 16 * GIB);
    let gov2 = governed(&vram2, 16 * GIB, 2 * GIB);
    let expected2 = 3 * GIB + ((0.15 * (14 * GIB) as f64) as u64);
    assert_eq!(gov2.kv_floor(), expected2);
    assert!(gov2.kv_floor() < gov.kv_floor());
}

#[test]
fn thresholds_descend_and_bracket_floor() {
    let vram = FakeVram::new(0, 73 * GIB);
    let gov = governed(&vram, 73 * GIB, 2 * GIB);
    let t = [
        gov.tier_threshold(Criticality::Trivial),
        gov.tier_threshold(Criticality::Cheap),
        gov.tier_threshold(Criticality::Moderate),
        gov.tier_threshold(Criticality::Costly),
        gov.tier_threshold(Criticality::Critical),
    ];
    // Strictly descending; Critical sits exactly at the floor.
    for w in t.windows(2) {
        assert!(w[0] > w[1], "thresholds must descend: {t:?}");
    }
    assert_eq!(t[4], gov.kv_floor());
}

#[test]
fn budget_evolves_with_allocations() {
    let vram = FakeVram::new(50 * GIB, 73 * GIB);
    let gov = governed(&vram, 73 * GIB, 2 * GIB);
    gov.reserve(AllocClass::Expert, 15 * GIB, || Ok::<_, Error>(()))
        .unwrap();
    // Model the experts consuming headroom (no accounting gate — measurement).
    vram.consume(15 * GIB);
    let table = gov.budget_table();
    assert_eq!(table.reserved(AllocClass::Weights), 2 * GIB);
    assert_eq!(table.reserved(AllocClass::Expert), 15 * GIB);
    assert_eq!(table.headroom, 35 * GIB);
}

// ── Balloon ──────────────────────────────────────────────────────────────────

#[test]
fn balloon_skips_when_card_already_free() -> Result<()> {
    // Headroom at/above target ⇒ no squatters ⇒ skip the touch; C = headroom.
    let total = 64 * GIB;
    let vram = FakeVram::new(total, total);
    let gov = VramGovernor::new(0, Box::new(vram.probe()), test_config());
    let mut alloc = FakeBalloonAllocator::new(vram.clone(), 60 * GIB);
    let c = gov.run_balloon(&mut alloc)?;
    assert_eq!(c, total, "free card: C = measured headroom");
    // The allocator was never touched — headroom is exactly as it started.
    assert_eq!(vram.headroom(), total);
    Ok(())
}

#[test]
fn balloon_loop_claims_to_target() -> Result<()> {
    // The loop itself (no fast path): fills to target on an uncontended fake.
    let total = 64 * GIB;
    let vram = FakeVram::new(total, total);
    let mut alloc = FakeBalloonAllocator::new(vram.clone(), 60 * GIB);
    let claimed = super::balloon::balloon_measure(&vram.probe(), &mut alloc, &test_config())?;
    let target = (0.90 * total as f64) as u64;
    assert_eq!(claimed, target);
    // Balloon freed: headroom restored.
    assert_eq!(vram.headroom(), total);
    Ok(())
}

#[test]
fn balloon_target_combines_fraction_and_absolute() -> Result<()> {
    // C = min(frac × total, total − headroom_abs): the absolute reserve binds on
    // a small card, the fraction on a large one — so neither the 16 GiB minimum
    // (which needs an absolute scratch reserve larger than 5%) nor a 72 GiB card
    // (which must not surrender a fixed slice of a huge card) is penalised.
    let mut cfg = test_config();
    cfg.balloon_target_frac = 0.95;
    cfg.balloon_headroom_abs = 2560 * MIB; // 2.5 GiB

    // 16 GiB: 0.95 × 16 = 15.2 GiB, but 16 − 2.5 = 13.5 GiB is smaller ⇒ abs binds.
    let total = 16 * GIB;
    let vram = FakeVram::new(total, total);
    let mut alloc = FakeBalloonAllocator::new(vram.clone(), total);
    let c = super::balloon::balloon_measure(&vram.probe(), &mut alloc, &cfg)?;
    assert_eq!(c, total - 2560 * MIB);

    // 72 GiB: 0.95 × 72 = 68.4 GiB, 72 − 2.5 = 69.5 GiB ⇒ the fraction binds.
    let total = 72 * GIB;
    let vram = FakeVram::new(total, total);
    let mut alloc = FakeBalloonAllocator::new(vram.clone(), total);
    let c = super::balloon::balloon_measure(&vram.probe(), &mut alloc, &cfg)?;
    assert_eq!(c, (0.95 * total as f64) as u64);
    Ok(())
}

#[test]
fn balloon_undersized_falls_back() -> Result<()> {
    let total = 64 * GIB;
    // Headroom below target ⇒ the loop runs; a low ceiling makes the claim tiny
    // ⇒ circuit-breaker fallback to total − margin.
    let vram = FakeVram::new(total / 2, total);
    let gov = VramGovernor::new(0, Box::new(vram.probe()), test_config());
    let mut alloc = FakeBalloonAllocator::new(vram.clone(), GIB);
    let c = gov.run_balloon(&mut alloc)?;
    assert_eq!(c, total - GIB);
    Ok(())
}

// ── Relief ladder ────────────────────────────────────────────────────────────

#[test]
fn gentle_relief_early_no_eviction() -> Result<()> {
    let vram = FakeVram::new(0, 73 * GIB);
    let gov = governed(&vram, 73 * GIB, 2 * GIB);
    let triv = counting_relief(&gov, &vram, AllocClass::Kv, Criticality::Trivial, 4 * GIB);
    let cheap = counting_relief(&gov, &vram, AllocClass::Kv, Criticality::Cheap, 4 * GIB);
    let mod_ = counting_relief(&gov, &vram, AllocClass::Kv, Criticality::Moderate, 4 * GIB);
    let costly = counting_relief(&gov, &vram, AllocClass::Kv, Criticality::Costly, 4 * GIB);
    let crit = counting_relief(&gov, &vram, AllocClass::Kv, Criticality::Critical, 4 * GIB);

    // Headroom just below the Trivial threshold, above Cheap → only Trivial trips.
    let t_trivial = gov.tier_threshold(Criticality::Trivial);
    let t_cheap = gov.tier_threshold(Criticality::Cheap);
    vram.set((t_trivial + t_cheap) / 2);

    gov.relieve_pressure(AllocClass::Kv)?;
    assert_eq!(triv.load(Ordering::Relaxed), 1, "Trivial should engage");
    assert_eq!(cheap.load(Ordering::Relaxed), 0);
    assert_eq!(mod_.load(Ordering::Relaxed), 0, "no KV eviction this early");
    assert_eq!(costly.load(Ordering::Relaxed), 0);
    assert_eq!(crit.load(Ordering::Relaxed), 0);
    assert_eq!(gov.sync_count(), 0, "no GPU sync below Critical");
    Ok(())
}

#[test]
fn escalates_but_withholds_critical_above_floor() -> Result<()> {
    let vram = FakeVram::new(0, 73 * GIB);
    let gov = governed(&vram, 73 * GIB, 2 * GIB);
    // Relievers that free nothing → forces the ladder to climb.
    let triv = counting_relief(&gov, &vram, AllocClass::Kv, Criticality::Trivial, 0);
    let cheap = counting_relief(&gov, &vram, AllocClass::Kv, Criticality::Cheap, 0);
    let mod_ = counting_relief(&gov, &vram, AllocClass::Kv, Criticality::Moderate, 0);
    let costly = counting_relief(&gov, &vram, AllocClass::Kv, Criticality::Costly, 0);
    let crit = counting_relief(&gov, &vram, AllocClass::Kv, Criticality::Critical, 0);

    // Just above the floor (Critical), below Costly → climbs to Costly, not Critical.
    let floor = gov.kv_floor();
    let t_costly = gov.tier_threshold(Criticality::Costly);
    vram.set((floor + t_costly) / 2);

    gov.relieve_pressure(AllocClass::Kv)?;
    assert_eq!(triv.load(Ordering::Relaxed), 1);
    assert_eq!(cheap.load(Ordering::Relaxed), 1);
    assert_eq!(mod_.load(Ordering::Relaxed), 1);
    assert_eq!(costly.load(Ordering::Relaxed), 1);
    assert_eq!(
        crit.load(Ordering::Relaxed),
        0,
        "Critical withheld above floor"
    );
    assert_eq!(gov.sync_count(), 0, "no sync until Critical");
    Ok(())
}

#[test]
fn critical_syncs_before_and_after() -> Result<()> {
    let vram = FakeVram::new(0, 73 * GIB);
    let gov = governed(&vram, 73 * GIB, 2 * GIB);
    let crit = counting_relief(&gov, &vram, AllocClass::Kv, Criticality::Critical, 0);
    // Below the floor → Critical engages.
    vram.set(gov.kv_floor() - GIB);
    let res = gov.relieve_pressure(AllocClass::Kv)?;
    assert!(crit.load(Ordering::Relaxed) >= 1, "Critical engaged");
    assert_eq!(
        gov.sync_count(),
        2,
        "sync before AND after aggressive relief"
    );
    assert_eq!(res, ReliefResult::Exhausted(0), "nothing freed → exhausted");
    Ok(())
}

#[test]
fn ladder_exhausted_no_spin() -> Result<()> {
    let vram = FakeVram::new(0, 73 * GIB);
    let gov = governed(&vram, 73 * GIB, 2 * GIB);
    // No relievers at all; headroom pinned at zero.
    vram.set(0);
    let res = gov.relieve_pressure(AllocClass::Kv)?;
    assert!(matches!(res, ReliefResult::Exhausted(_)));
    // Bounded work: the probe was read a finite number of times.
    assert!(vram.read_count() < 100, "must not spin");
    Ok(())
}

#[test]
fn relief_stops_when_recovered() -> Result<()> {
    let vram = FakeVram::new(0, 73 * GIB);
    let gov = governed(&vram, 73 * GIB, 2 * GIB);
    // A generous Trivial reliever recovers headroom above healthy in one pass.
    let triv = counting_relief(&gov, &vram, AllocClass::Kv, Criticality::Trivial, 40 * GIB);
    let mod_ = counting_relief(&gov, &vram, AllocClass::Kv, Criticality::Moderate, 40 * GIB);
    vram.set(gov.kv_floor()); // deep
    gov.relieve_pressure(AllocClass::Kv)?;
    assert_eq!(triv.load(Ordering::Relaxed), 1);
    assert_eq!(
        mod_.load(Ordering::Relaxed),
        0,
        "recovered before eviction rungs"
    );
    Ok(())
}

// ── Managed allocation & retry ───────────────────────────────────────────────

#[test]
fn allocate_retries_through_relief() -> Result<()> {
    let vram = FakeVram::new(0, 73 * GIB);
    let gov = governed(&vram, 73 * GIB, 2 * GIB);
    // A Moderate reliever that frees 2 GiB when asked.
    counting_relief(&gov, &vram, AllocClass::Kv, Criticality::Moderate, 2 * GIB);
    let need = GIB;
    let vr = vram.clone();
    // The alloc "succeeds" once headroom covers the request (models a real alloc).
    gov.allocate(AllocClass::Kv, need, || {
        if vr.headroom() >= need {
            Ok(())
        } else {
            Err(Error::Msg("out of memory".into()))
        }
    })?;
    assert_eq!(gov.class_reserved(AllocClass::Kv), need);
    Ok(())
}

#[test]
fn allocate_gives_up_after_full_ladder() {
    let vram = FakeVram::new(0, 73 * GIB);
    let gov = governed(&vram, 73 * GIB, 2 * GIB);
    // No relievers; alloc always OOMs.
    let res: Result<()> = gov.allocate(AllocClass::Kv, GIB, || {
        Err(Error::Msg("out of memory".into()))
    });
    assert!(res.is_err());
    // Reached Critical (rate limit 0), so it did sync while trying.
    assert!(gov.sync_count() >= 1);
    // Nothing credited on failure.
    assert_eq!(gov.class_reserved(AllocClass::Kv), 0);
}

#[test]
fn allocate_propagates_non_oom_error() {
    let vram = FakeVram::new(0, 73 * GIB);
    let gov = governed(&vram, 73 * GIB, 2 * GIB);
    let res: Result<()> = gov.allocate(AllocClass::Kv, GIB, || {
        Err(Error::Msg("shape mismatch".into()))
    });
    assert!(res.is_err());
    // A non-OOM error must NOT trigger the relief ladder.
    assert_eq!(gov.sync_count(), 0);
}

#[test]
fn reserve_credits_class_without_relief() -> Result<()> {
    let vram = FakeVram::new(10 * GIB, 73 * GIB);
    let gov = governed(&vram, 73 * GIB, 2 * GIB);
    gov.reserve(AllocClass::Scratch, GIB, || Ok::<_, Error>(()))?;
    assert_eq!(gov.class_reserved(AllocClass::Scratch), GIB);
    assert_eq!(gov.sync_count(), 0);
    Ok(())
}

// ── Forecast ─────────────────────────────────────────────────────────────────

#[test]
fn forecast_counts_reversible_evictable_only() {
    let vram = FakeVram::new(4 * GIB, 73 * GIB);
    let gov = governed(&vram, 73 * GIB, 2 * GIB);
    // 2 GiB reversibly evictable at Moderate, plus a huge Critical-only pool.
    counting_relief(&gov, &vram, AllocClass::Kv, Criticality::Moderate, 2 * GIB);
    counting_relief(
        &gov,
        &vram,
        AllocClass::Kv,
        Criticality::Critical,
        100 * GIB,
    );
    // Units of 1 GiB: headroom(4) + reversible(2) = 6, excludes the Critical pool.
    assert_eq!(gov.forecast_units(GIB), 6);
}

// ── Expert budget (all- vs some-resident) ────────────────────────────────────

#[test]
fn expert_budget_leaves_floor_and_scratch() -> Result<()> {
    let vram = FakeVram::new(60 * GIB, 73 * GIB);
    let gov = governed(&vram, 73 * GIB, 2 * GIB);
    let budget = gov.expert_budget()?;
    assert_eq!(budget, 60 * GIB - gov.kv_floor() - GIB);
    // Experts never cross the floor: after taking the whole budget, ≥ floor remains.
    assert!(60 * GIB - budget >= gov.kv_floor());
    Ok(())
}

#[test]
fn all_experts_resident_when_they_fit() -> Result<()> {
    let vram = FakeVram::new(60 * GIB, 73 * GIB);
    let gov = governed(&vram, 73 * GIB, 2 * GIB);
    let max_expert = 4 * MIB;
    let total_experts = 6144u64;
    let total_expert_bytes = total_experts * max_expert; // ~24 GiB
    let budget = gov.expert_budget()?;
    let num_slots = (budget / max_expert).min(total_experts);
    let all_resident = num_slots >= total_experts;
    assert!(
        budget >= total_expert_bytes,
        "budget should fit all experts here"
    );
    assert!(all_resident);
    Ok(())
}

#[test]
fn some_experts_resident_when_tight() -> Result<()> {
    // Tight card: headroom leaves only a little above the floor for experts.
    let vram = FakeVram::new(20 * GIB, 24 * GIB);
    let gov = governed(&vram, 24 * GIB, 2 * GIB);
    let max_expert = 4 * MIB;
    let total_experts = 6144u64;
    let budget = gov.expert_budget()?;
    let num_slots = (budget / max_expert).min(total_experts);
    let all_resident = num_slots >= total_experts;
    assert!(!all_resident, "not all experts fit on a tight card");
    assert!(num_slots > 0, "but some are resident");
    Ok(())
}

// ── External pressure & recovery ─────────────────────────────────────────────

#[test]
fn external_theft_drives_relief() -> Result<()> {
    let vram = FakeVram::new(30 * GIB, 73 * GIB);
    let gov = governed(&vram, 73 * GIB, 2 * GIB);
    counting_relief(&gov, &vram, AllocClass::Kv, Criticality::Moderate, 8 * GIB);
    // Another process grabs VRAM → headroom collapses below the eviction rungs.
    vram.set(gov.kv_floor() + 100 * MIB);
    let res = gov.relieve_pressure(AllocClass::Kv)?;
    assert!(res.freed() > 0, "eviction engaged to shed KV under theft");
    Ok(())
}

#[test]
fn forecast_recovers_when_headroom_returns() {
    let vram = FakeVram::new(2 * GIB, 73 * GIB);
    let gov = governed(&vram, 73 * GIB, 2 * GIB);
    let low = gov.forecast_units(GIB);
    vram.set(40 * GIB);
    let high = gov.forecast_units(GIB);
    assert!(
        high > low,
        "forecast grows back as headroom returns ({low} -> {high})"
    );
}

// ── Diagnostics ──────────────────────────────────────────────────────────────

#[test]
fn budget_table_shape() {
    let vram = FakeVram::new(20 * GIB, 73 * GIB);
    let gov = governed(&vram, 73 * GIB, 2 * GIB);
    counting_relief(&gov, &vram, AllocClass::Kv, Criticality::Moderate, 5 * GIB);
    let t = gov.budget_table();
    assert_eq!(t.capacity_c, 73 * GIB);
    assert_eq!(t.total, 73 * GIB);
    assert_eq!(t.headroom, 20 * GIB);
    assert_eq!(t.rows.len(), AllocClass::COUNT);
    assert_eq!(t.reserved(AllocClass::Weights), 2 * GIB);
    assert_eq!(t.kv_floor, gov.kv_floor());
    assert_eq!(t.thresholds[4], gov.kv_floor());
    assert_eq!(t.evictable_reversible, 5 * GIB);
    // Rendering must not panic and must mention the source.
    let rendered = gov.render_budget("test");
    assert!(rendered.contains("vram budget [test]"));
}

// ── Global registry ──────────────────────────────────────────────────────────

#[test]
fn registry_install_and_get() {
    let vram = FakeVram::new(GIB, 73 * GIB);
    let gov = Arc::new(VramGovernor::new(7, Box::new(vram.probe()), test_config()));
    install(gov.clone());
    assert!(get(7).is_some());
    assert_eq!(get(7).unwrap().gpu_id(), 7);
    remove(7);
    assert!(get(7).is_none());
}

// ── Relief registration lifecycle ────────────────────────────────────────────

#[test]
fn debit_class_reverses_credit() {
    let vram = FakeVram::new(GIB, 73 * GIB);
    let gov = governed(&vram, 73 * GIB, 2 * GIB);
    gov.credit_class(AllocClass::Kv, 5 * GIB);
    gov.debit_class(AllocClass::Kv, 2 * GIB);
    assert_eq!(gov.class_reserved(AllocClass::Kv), 3 * GIB);
    // Saturating: never underflows past zero.
    gov.debit_class(AllocClass::Kv, 100 * GIB);
    assert_eq!(gov.class_reserved(AllocClass::Kv), 0);
}

#[test]
fn relieve_with_driver_climbs_cheapest_first() -> Result<()> {
    // A borrowed driver whose rungs release scripted amounts of headroom.
    struct Driver {
        vram: FakeVram,
        calls: Vec<Criticality>,
    }
    impl super::KvReliefDriver for Driver {
        fn relieve(&mut self, tier: Criticality, want: u64) -> u64 {
            self.calls.push(tier);
            // Only the Moderate rung frees enough here.
            let freed = match tier {
                Criticality::Trivial | Criticality::Cheap => GIB / 2,
                Criticality::Moderate => want.min(8 * GIB),
                _ => 0,
            };
            self.vram.release(freed);
            freed
        }
    }
    let vram = FakeVram::new(0, 73 * GIB);
    let gov = governed(&vram, 73 * GIB, 2 * GIB);
    vram.set(GIB);
    let mut driver = Driver {
        vram: vram.clone(),
        calls: Vec::new(),
    };
    let res = gov.relieve_with(6 * GIB, &mut driver)?;
    assert!(res.is_relieved());
    assert!(vram.headroom() >= 6 * GIB);
    // Cheapest-first: Trivial and Cheap ran before Moderate.
    assert_eq!(driver.calls[0], Criticality::Trivial);
    assert_eq!(driver.calls[1], Criticality::Cheap);
    assert!(driver.calls.contains(&Criticality::Moderate));
    // Never reached Critical (Moderate satisfied the target) → no sync.
    assert_eq!(gov.sync_count(), 0);
    Ok(())
}

#[test]
fn relieve_to_target_climbs_until_met() -> Result<()> {
    let vram = FakeVram::new(0, 73 * GIB);
    let gov = governed(&vram, 73 * GIB, 2 * GIB);
    // Moderate reliever frees 6 GiB when asked.
    counting_relief(&gov, &vram, AllocClass::Kv, Criticality::Moderate, 6 * GIB);
    vram.set(GIB);
    let res = gov.relieve_to(AllocClass::Kv, 5 * GIB)?;
    assert!(res.is_relieved());
    assert!(vram.headroom() >= 5 * GIB);
    Ok(())
}

#[test]
fn unregister_relief_removes_closure() {
    let vram = FakeVram::new(GIB, 73 * GIB);
    let gov = governed(&vram, 73 * GIB, 2 * GIB);
    let h = gov.register_relief(
        AllocClass::Kv,
        Criticality::Moderate,
        |_| ReliefOutcome::new(0),
        || 123,
    );
    assert_eq!(gov.relief_count(), 1);
    assert_eq!(gov.evictable_estimate(Criticality::Moderate), 123);
    assert!(gov.unregister_relief(h));
    assert_eq!(gov.relief_count(), 0);
    assert_eq!(gov.evictable_estimate(Criticality::Moderate), 0);
}

// ── Inference-engine run scenarios (CPU, modelled on the real lifecycle) ──────
//
// A `Sim` wires the governor to a fake VRAM cell and a KV/expert residency model
// the way the real subsystems will register relief, then the tests drive the
// actual engine flow: boot → load weights/experts/scratch → grow KV → forecast
// prefill width → relieve pressure → external contention → steady state.

mod scenarios {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    use std::time::Instant;

    const CARD: u64 = 73 * GIB; // model the real Blackwell
    const CONTEXT: u64 = GIB; // CUDA context overhead

    fn cfg() -> GovernorConfig {
        test_config()
    }

    /// A simulated engine: governor + fake VRAM + a KV/expert residency model.
    struct Sim {
        gov: VramGovernor,
        vram: FakeVram,
        kv_held: Arc<AtomicU64>,
        frag: Arc<AtomicU64>,
        experts: Arc<AtomicU64>,
        floor: u64,
    }

    impl Sim {
        /// Boot on a free card and measure capacity (balloon fast-path).
        fn boot(total: u64) -> Self {
            let vram = FakeVram::new(total - CONTEXT, total);
            let gov = VramGovernor::new(0, Box::new(vram.probe()), cfg());
            gov.set_capacity(total - CONTEXT);
            Sim {
                gov,
                vram,
                kv_held: Arc::new(AtomicU64::new(0)),
                frag: Arc::new(AtomicU64::new(0)),
                experts: Arc::new(AtomicU64::new(0)),
                floor: 0,
            }
        }

        fn load_weights(&self, bytes: u64) {
            self.gov
                .reserve(AllocClass::Weights, bytes, || Ok::<_, crate::Error>(()))
                .unwrap();
            self.vram.consume(bytes);
        }

        fn load_experts(&self, resident: u64, all_resident: bool) {
            self.gov
                .reserve(AllocClass::Expert, resident, || Ok::<_, crate::Error>(()))
                .unwrap();
            self.vram.consume(resident);
            self.experts.store(resident, Ordering::Relaxed);
            if !all_resident {
                // Partial residency: experts can be shed only at Critical.
                let v = self.vram.clone();
                let e = self.experts.clone();
                let e2 = self.experts.clone();
                self.gov.register_relief(
                    AllocClass::Expert,
                    Criticality::Critical,
                    move |req| {
                        let avail = e.load(Ordering::Relaxed) / 2; // shrink pool by up to half
                        let x = req.want.min(avail);
                        e.fetch_sub(x, Ordering::Relaxed);
                        v.release(x);
                        ReliefOutcome::new(x)
                    },
                    move || e2.load(Ordering::Relaxed) / 2,
                );
            }
        }

        fn load_scratch(&self, bytes: u64) {
            self.gov
                .reserve(AllocClass::Scratch, bytes, || Ok::<_, crate::Error>(()))
                .unwrap();
            self.vram.consume(bytes);
        }

        /// Register the KV relief ladder (after weights, so the floor is fixed):
        /// Trivial reclaims fragmentation (no hit-rate cost); Moderate evicts KV
        /// reversibly down to the floor; Critical evicts below the floor (lossy).
        fn wire_kv(&mut self) {
            self.floor = self.gov.kv_floor();
            let floor = self.floor;

            let (v, f, f2) = (self.vram.clone(), self.frag.clone(), self.frag.clone());
            self.gov.register_relief(
                AllocClass::Kv,
                Criticality::Trivial,
                move |req| {
                    let x = req.want.min(f.load(Ordering::Relaxed));
                    f.fetch_sub(x, Ordering::Relaxed);
                    v.release(x);
                    ReliefOutcome::new(x)
                },
                move || f2.load(Ordering::Relaxed),
            );

            let (v, kv, kv2) = (
                self.vram.clone(),
                self.kv_held.clone(),
                self.kv_held.clone(),
            );
            self.gov.register_relief(
                AllocClass::Kv,
                Criticality::Moderate,
                move |req| {
                    let evictable = kv.load(Ordering::Relaxed).saturating_sub(floor);
                    let x = req.want.min(evictable);
                    kv.fetch_sub(x, Ordering::Relaxed);
                    v.release(x);
                    ReliefOutcome::new(x)
                },
                move || kv2.load(Ordering::Relaxed).saturating_sub(floor),
            );

            let (v, kv) = (self.vram.clone(), self.kv_held.clone());
            self.gov.register_relief(
                AllocClass::Kv,
                Criticality::Critical,
                move |req| {
                    let x = req.want.min(kv.load(Ordering::Relaxed));
                    kv.fetch_sub(x, Ordering::Relaxed);
                    v.release(x);
                    ReliefOutcome::new(x)
                },
                || 0, // lossy last resort — not offered to the forecast
            );
        }

        /// Model a KV arena allocation: consume headroom, grow residency, accrue
        /// ~6% reclaimable fragmentation.
        fn grow_kv(&self, bytes: u64) {
            self.vram.consume(bytes);
            self.kv_held.fetch_add(bytes, Ordering::Relaxed);
            self.gov.credit_class(AllocClass::Kv, bytes);
            self.frag.fetch_add(bytes / 16, Ordering::Relaxed);
        }

        fn headroom(&self) -> u64 {
            self.vram.headroom()
        }
        fn kv(&self) -> u64 {
            self.kv_held.load(Ordering::Relaxed)
        }
        fn experts(&self) -> u64 {
            self.experts.load(Ordering::Relaxed)
        }
    }

    /// A fully loaded engine ready to serve: weights + experts + scratch in,
    /// KV relief wired. Big card ⇒ all experts resident.
    fn loaded(total: u64, all_resident: bool) -> Sim {
        let mut s = Sim::boot(total);
        s.load_weights(2 * GIB);
        let experts = if all_resident { 16 * GIB } else { 30 * GIB };
        s.load_experts(experts, all_resident);
        s.load_scratch(2 * GIB);
        s.wire_kv();
        s
    }

    #[test]
    fn startup_partitions_evolve_and_leave_kv_floor() {
        let s = loaded(CARD, true);
        let t = s.gov.budget_table();
        assert_eq!(t.reserved(AllocClass::Weights), 2 * GIB);
        assert_eq!(t.reserved(AllocClass::Expert), 16 * GIB);
        assert_eq!(t.reserved(AllocClass::Scratch), 2 * GIB);
        // KV region (current headroom) is well above the floor at boot.
        assert!(s.headroom() > s.floor, "KV headroom above floor at boot");
        // Floor is 3 GiB + 15% of (C − weights).
        assert_eq!(t.kv_floor, s.gov.kv_floor());
    }

    #[test]
    fn expert_budget_all_resident_vs_partial() -> Result<()> {
        // Big card: budget fits all experts → all-resident.
        let big = Sim::boot(CARD);
        big.load_weights(2 * GIB);
        let budget = big.gov.expert_budget()?;
        let total_expert_bytes = 16 * GIB;
        assert!(budget >= total_expert_bytes, "big card fits all experts");

        // Small card: budget can't fit all → partial residency.
        let small = Sim::boot(24 * GIB);
        small.load_weights(2 * GIB);
        let budget_s = small.gov.expert_budget()?;
        assert!(budget_s < 30 * GIB, "tight card can't fit all experts");
        assert!(budget_s > 0);
        // Experts never eat the KV floor.
        assert!(small.headroom() - budget_s >= small.gov.kv_floor());
        Ok(())
    }

    #[test]
    fn full_kv_does_not_starve_prefill_forecast() {
        // As KV fills, raw headroom shrinks — but the forecast counts reversibly-
        // evictable KV, so prefill can always plan on evicting cold KV to make
        // room. The forecast stays well above what raw headroom alone would allow,
        // and doesn't collapse. This is the property that keeps the engine from
        // deadlocking once the cache is warm.
        let s = loaded(CARD, true);
        let per_seq = 512 * MIB;
        let f_empty = s.gov.forecast_units(per_seq);
        for _ in 0..16 {
            s.grow_kv(2 * GIB); // fill ~32 GiB of KV, headroom stays positive
        }
        let f_full = s.gov.forecast_units(per_seq);
        let raw_full = (s.headroom() / per_seq) as usize;
        assert!(
            f_full > raw_full,
            "eviction accounting keeps the forecast above raw headroom ({f_full} vs raw {raw_full})"
        );
        assert!(
            f_full >= f_empty / 2,
            "full KV doesn't starve prefills ({f_empty} -> {f_full})"
        );
    }

    #[test]
    fn forecast_admits_more_than_raw_headroom_via_evictable() {
        let s = loaded(CARD, true);
        s.grow_kv(30 * GIB); // build a big evictable KV pool
        let per_seq = GIB;
        let raw = s.headroom() / per_seq;
        let forecast = s.gov.forecast_units(per_seq) as u64;
        assert!(
            forecast > raw,
            "forecast ({forecast}) counts reversibly-evictable KV beyond raw headroom ({raw})"
        );
    }

    #[test]
    fn gentle_relief_reclaims_fragmentation_without_evicting_kv() -> Result<()> {
        let s = loaded(CARD, true);
        s.grow_kv(30 * GIB); // accrues fragmentation
        let kv_before = s.kv();
        // Position headroom just below Trivial, above Cheap.
        let t_triv = s.gov.tier_threshold(Criticality::Trivial);
        let t_cheap = s.gov.tier_threshold(Criticality::Cheap);
        s.vram.set((t_triv + t_cheap) / 2);
        s.gov.relieve_pressure(AllocClass::Kv)?;
        assert_eq!(s.kv(), kv_before, "no KV evicted at the gentle rung");
        assert_eq!(s.gov.sync_count(), 0, "no GPU sync for gentle relief");
        Ok(())
    }

    #[test]
    fn heavy_pressure_evicts_kv_but_never_below_floor() -> Result<()> {
        let mut s = loaded(CARD, true);
        // Grow KV until we're under the Moderate threshold.
        while s.headroom() > s.gov.tier_threshold(Criticality::Moderate) {
            s.grow_kv(2 * GIB);
        }
        let _ = &mut s;
        s.gov.relieve_pressure(AllocClass::Kv)?;
        // Moderate eviction ran and left KV at (or above) the floor.
        assert!(
            s.kv() >= s.floor,
            "KV eviction stops at the floor (kv={}MiB floor={}MiB)",
            s.kv() / MIB,
            s.floor / MIB
        );
        assert!(
            s.headroom() >= s.gov.tier_threshold(Criticality::Moderate),
            "headroom recovered above the eviction threshold"
        );
        Ok(())
    }

    #[test]
    fn oom_on_arena_alloc_retries_through_relief() -> Result<()> {
        let s = loaded(CARD, true);
        s.grow_kv(40 * GIB); // fill toward the floor
                             // Drive headroom below one arena so the alloc "fails".
        let arena = 512 * MIB;
        s.vram.set(arena / 2);
        let vr = s.vram.clone();
        // Real arena alloc: fails until headroom covers it; relief frees KV.
        s.gov.allocate(AllocClass::Kv, arena, || {
            if vr.headroom() >= arena {
                Ok(())
            } else {
                Err(crate::Error::Msg("out of memory".into()))
            }
        })?;
        assert!(s.vram.headroom() >= arena || s.kv() <= s.floor);
        Ok(())
    }

    #[test]
    fn experts_untouched_until_critical() -> Result<()> {
        let mut s = loaded(CARD, false); // partial residency ⇒ expert relief exists
        let experts_before = s.experts();
        // Grow into the Costly band (above the floor) but not to Critical.
        while s.headroom() > s.gov.tier_threshold(Criticality::Costly) {
            s.grow_kv(2 * GIB);
        }
        let _ = &mut s;
        // Position just above the floor: Costly trips, Critical does not.
        let floor = s.gov.kv_floor();
        let t_costly = s.gov.tier_threshold(Criticality::Costly);
        s.vram.set((floor + t_costly) / 2);
        s.gov.relieve_pressure(AllocClass::Kv)?;
        assert_eq!(
            s.experts(),
            experts_before,
            "experts not shed above the floor (Critical withheld)"
        );

        // Now drive below the floor → Critical → experts may be borrowed.
        s.grow_kv(0);
        s.vram.set(floor.saturating_sub(GIB));
        // Exhaust KV first so Critical must reach cross-class into experts.
        s.kv_held.store(0, Ordering::Relaxed);
        s.gov.relieve_pressure(AllocClass::Kv)?;
        assert!(
            s.experts() < experts_before,
            "experts borrowed only at Critical ({}MiB -> {}MiB)",
            experts_before / MIB,
            s.experts() / MIB
        );
        assert!(s.gov.sync_count() >= 2, "Critical synced");
        Ok(())
    }

    #[test]
    fn external_contention_sheds_kv_holds_floor_then_recovers() -> Result<()> {
        let s = loaded(CARD, true);
        s.grow_kv(20 * GIB);
        s.frag.store(0, Ordering::Relaxed); // isolate the KV-eviction behaviour
        let kv_calm = s.kv();
        let forecast_calm = s.gov.forecast_units(GIB);

        // Another process grabs VRAM: headroom collapses into the Moderate band
        // (below the eviction threshold but where the floor still fits).
        let theft = 19 * GIB;
        s.vram.consume(theft);
        s.gov.relieve_pressure(AllocClass::Kv)?;
        assert!(s.kv() < kv_calm, "KV was shed under contention");
        assert!(s.kv() >= s.floor, "but never below the floor");
        assert_eq!(
            s.gov.sync_count(),
            0,
            "Moderate handled it — no Critical sync"
        );

        // Contention clears: forecast recovers.
        s.vram.release(theft);
        let forecast_after = s.gov.forecast_units(GIB);
        assert!(
            forecast_after >= forecast_calm / 2,
            "forecast recovers after contention clears ({forecast_calm} -> {forecast_after})"
        );
        Ok(())
    }

    #[test]
    fn sustained_wave_loop_is_stable_and_fast() -> Result<()> {
        let s = loaded(CARD, true);
        let per_seq = 512 * MIB;
        let waves = 2000;
        let floor = s.floor;

        let t = Instant::now();
        for _ in 0..waves {
            // Each wave: admit prefills per the forecast, grow KV, relieve.
            let width = s.gov.forecast_units(per_seq);
            if width > 0 {
                s.grow_kv(per_seq); // one wave's worth of new KV
            }
            if s.gov.measure()?.headroom < s.gov.tier_threshold(Criticality::Trivial) {
                s.gov.relieve_pressure(AllocClass::Kv)?;
            }
            // Invariant every wave: we never drive headroom to zero (no paging).
            assert!(s.headroom() > 0, "headroom must never hit zero (paging)");
        }
        let elapsed = t.elapsed();
        eprintln!(
            "[scenarios] {waves} waves in {:.1}ms ({:.2}µs/wave); final headroom={}MiB kv={}MiB floor={}MiB",
            elapsed.as_secs_f64() * 1000.0,
            elapsed.as_nanos() as f64 / waves as f64 / 1000.0,
            s.headroom() / MIB,
            s.kv() / MIB,
            floor / MIB,
        );
        // Steady state stays healthy and the loop is cheap.
        assert!(s.kv() >= floor / 2, "KV working set retained");
        assert!(
            elapsed.as_millis() < 500,
            "2000 wave cycles must be fast (was {}ms)",
            elapsed.as_millis()
        );
        Ok(())
    }

    #[test]
    fn decode_and_prefill_mix() -> Result<()> {
        let s = loaded(CARD, true);
        // Interleave large prefill arenas and tiny decode-step growth.
        for _ in 0..50 {
            s.grow_kv(512 * MIB); // prefill chunk
            for _ in 0..32 {
                s.grow_kv(2 * MIB); // decode steps
            }
            if s.gov.measure()?.headroom < s.gov.tier_threshold(Criticality::Cheap) {
                s.gov.relieve_pressure(AllocClass::Kv)?;
            }
            assert!(s.headroom() > 0);
        }
        assert!(s.gov.budget_table().headroom > 0);
        Ok(())
    }

    #[test]
    fn all_resident_never_touches_experts() -> Result<()> {
        let s = loaded(CARD, true); // all-resident ⇒ no expert relief registered
        let experts_before = s.experts();
        s.vram.set(s.gov.kv_floor().saturating_sub(GIB)); // below floor → Critical
        s.kv_held.store(0, Ordering::Relaxed); // no KV to shed
        s.gov.relieve_pressure(AllocClass::Kv)?;
        assert_eq!(
            s.experts(),
            experts_before,
            "all-resident experts are never shed"
        );
        Ok(())
    }
}

// ── Real CUDA device tests (feature = "cuda", need a GPU) ─────────────────────
//
// Run with:  cargo test -p candle-core --features cuda vram::tests::real_cuda -- --nocapture
// They skip gracefully (pass) if no CUDA device is present.

#[cfg(feature = "cuda")]
mod real_cuda {
    use super::*;
    use crate::vram::balloon::DeviceBalloonAllocator;
    use crate::{Device, Tensor};
    use std::sync::atomic::AtomicBool;
    use std::sync::Mutex;
    use std::time::Instant;

    // These tests each claim large amounts of real VRAM, so they must not run
    // concurrently (the default test harness runs tests in parallel). Serialize
    // them through one process-wide lock, recovering from poisoning so one
    // failure doesn't cascade.
    static GPU_LOCK: Mutex<()> = Mutex::new(());

    fn gpu_guard() -> std::sync::MutexGuard<'static, ()> {
        GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn cuda_device() -> Option<Device> {
        match Device::new_cuda(0) {
            Ok(d) => Some(d),
            Err(e) => {
                eprintln!("[real_cuda] no CUDA device, skipping: {e}");
                None
            }
        }
    }

    fn mib(b: u64) -> u64 {
        b / MIB
    }

    /// Allocate `bytes` of real, touched device VRAM (tagged Kv), holding it.
    fn alloc_kv(device: &Device, bytes: u64) -> Result<Tensor> {
        Tensor::zeros(bytes as usize, crate::DType::U8, device)
    }

    #[test]
    fn real_cuda_balloon_measure_and_track() -> Result<()> {
        let Some(device) = cuda_device() else {
            return Ok(());
        };
        let _gpu = gpu_guard();
        let gov = VramGovernor::from_device(&device, 0)?;

        let before = gov.measure()?;
        eprintln!(
            "[real_cuda] before balloon: headroom={}MiB total={}MiB source={:?}",
            mib(before.headroom),
            mib(before.total),
            before.source
        );
        assert!(before.total > 0);

        // Probe latency: production calls measure() on a per-wave cadence, so it
        // must be cheap.
        let n = 1000;
        let t = Instant::now();
        for _ in 0..n {
            let _ = gov.measure()?;
        }
        let per_read_us = t.elapsed().as_nanos() as f64 / n as f64 / 1000.0;
        eprintln!(
            "[real_cuda] probe.read() latency: {per_read_us:.2}µs/call ({:?})",
            before.source
        );

        // Balloon to 95% (default), then free+trim → capacity C. Time it: the
        // balloon runs at startup and must not move init time.
        let mut balloon = DeviceBalloonAllocator::new(device.clone());
        let t = Instant::now();
        let c = gov.run_balloon(&mut balloon)?;
        let balloon_ms = t.elapsed().as_secs_f64() * 1000.0;
        let skipped = balloon_ms < 10.0;
        eprintln!(
            "[real_cuda] run_balloon → C={}MiB ({}% of total) in {:.2}ms {}",
            mib(c),
            c.saturating_mul(100) / before.total,
            balloon_ms,
            if skipped {
                "(fast path: card already free, touch skipped)"
            } else {
                "(full touch-balloon)"
            }
        );
        assert!(c > 0, "balloon must claim something");
        assert!(c <= before.total, "C cannot exceed total");

        // After the balloon freed + trimmed, headroom must have RECOVERED (proves
        // free_all's sync+trim actually returned the balloon to the OS).
        let after_balloon = gov.measure()?;
        eprintln!(
            "[real_cuda] after balloon free+trim: headroom={}MiB",
            mib(after_balloon.headroom)
        );
        assert!(
            after_balloon.headroom >= before.headroom / 2,
            "headroom should recover after balloon free+trim (got {}MiB, before {}MiB)",
            mib(after_balloon.headroom),
            mib(before.headroom)
        );

        // Allocate ~2 GiB of real KV and confirm the probe SEES headroom drop.
        let h_pre = gov.measure()?.headroom;
        let mut held: Vec<Tensor> = Vec::new();
        let alloc_bytes = (2 * GIB).min(h_pre.saturating_sub(GIB)); // leave a GiB
        let mut done = 0u64;
        while done < alloc_bytes {
            let chunk = (256 * MIB).min(alloc_bytes - done);
            match alloc_kv(&device, chunk) {
                Ok(t) => {
                    held.push(t);
                    done += chunk;
                }
                Err(_) => break,
            }
        }
        let h_alloc = gov.measure()?.headroom;
        eprintln!(
            "[real_cuda] allocated {}MiB KV: headroom {}MiB -> {}MiB",
            mib(done),
            mib(h_pre),
            mib(h_alloc)
        );
        if done > 0 {
            assert!(
                h_alloc < h_pre,
                "headroom must drop after real allocation ({}MiB -> {}MiB)",
                mib(h_pre),
                mib(h_alloc)
            );
        }

        // Free + sync + trim → headroom recovers. Time the trim (the Critical
        // rung pays this cost).
        held.clear();
        let t = Instant::now();
        device.synchronize()?;
        if let Device::Cuda(d) = &device {
            let _ = d.trim_pool(0);
        }
        let trim_ms = t.elapsed().as_secs_f64() * 1000.0;
        let h_free = gov.measure()?.headroom;
        eprintln!(
            "[real_cuda] after free+trim: headroom={}MiB (sync+trim {:.1}ms)",
            mib(h_free),
            trim_ms
        );
        assert!(
            h_free >= h_alloc,
            "headroom must recover after free+trim ({}MiB -> {}MiB)",
            mib(h_alloc),
            mib(h_free)
        );
        gov.log_budget("real_cuda balloon test");
        Ok(())
    }

    /// Measure the governor's real hot-path overhead: what a scheduler wave
    /// actually calls (forecast the admit width + check/relieve pressure), timed
    /// against a fully-populated relief registry and the LIVE DXGI probe. This is
    /// the number that decides whether the governor is free to run per wave / per
    /// arena allocation in the engine.
    #[test]
    fn real_cuda_hotpath_overhead() -> Result<()> {
        let Some(device) = cuda_device() else {
            return Ok(());
        };
        let _gpu = gpu_guard();
        let gov = VramGovernor::from_device(&device, 0)?;
        gov.set_capacity(gov.measure()?.total);
        // Populate the registry the way the real engine will (KV: 5 rungs; experts:
        // 2 rungs), each with an evictable reporter the forecast must sum.
        for tier in Criticality::ALL {
            gov.register_relief(AllocClass::Kv, tier, |_| ReliefOutcome::new(0), || 4 * GIB);
        }
        gov.register_relief(
            AllocClass::Expert,
            Criticality::Moderate,
            |_| ReliefOutcome::new(0),
            || GIB,
        );
        gov.register_relief(
            AllocClass::Expert,
            Criticality::Critical,
            |_| ReliefOutcome::new(0),
            || GIB,
        );

        let per_seq = 512 * MIB;
        let n = 100_000u64;

        // (a) forecast_units — measure + evictable_estimate over the registry.
        let t = Instant::now();
        let mut acc = 0usize;
        for _ in 0..n {
            acc = acc.wrapping_add(gov.forecast_units(per_seq));
        }
        let forecast_ns = t.elapsed().as_nanos() as f64 / n as f64;

        // (b) relieve_pressure while healthy — measure + threshold compare, no
        //     relief invoked (the common per-wave case).
        let t = Instant::now();
        for _ in 0..n {
            let _ = gov.relieve_pressure(AllocClass::Kv)?;
        }
        let pressure_ns = t.elapsed().as_nanos() as f64 / n as f64;

        // (c) a single measure() (the DXGI query alone).
        let t = Instant::now();
        for _ in 0..n {
            let _ = gov.measure()?;
        }
        let measure_ns = t.elapsed().as_nanos() as f64 / n as f64;

        let per_wave_ns = forecast_ns + pressure_ns; // what a wave actually pays
        eprintln!("[real_cuda] governor hot-path overhead (live DXGI, full registry):");
        eprintln!("  measure()          : {measure_ns:.0} ns");
        eprintln!("  forecast_units()   : {forecast_ns:.0} ns");
        eprintln!("  relieve_pressure() : {pressure_ns:.0} ns  (healthy, no eviction)");
        eprintln!(
            "  → per-wave check   : {per_wave_ns:.0} ns  = {:.3} µs",
            per_wave_ns / 1000.0
        );
        // Compare to representative wave times.
        for (label, wave_ms) in [
            ("decode wave ~25ms", 25.0),
            ("prefill wave ~2000ms", 2000.0),
        ] {
            let frac = (per_wave_ns / 1e9) / (wave_ms / 1e3) * 100.0;
            eprintln!("  vs {label}: {frac:.5}% of the wave");
        }
        assert!(acc > 0);
        // Hard ceiling: the per-wave governor check must be well under 100µs so it
        // is immaterial against even a fast decode wave.
        assert!(
            per_wave_ns < 100_000.0,
            "per-wave governor overhead must be immaterial (was {per_wave_ns:.0} ns)"
        );
        Ok(())
    }

    /// Force the full touch-balloon (bypassing run_balloon's fast-path skip) to
    /// measure the real claim throughput and confirm it evicts + recovers.
    #[test]
    fn real_cuda_full_balloon_throughput() -> Result<()> {
        let Some(device) = cuda_device() else {
            return Ok(());
        };
        let _gpu = gpu_guard();
        let gov = VramGovernor::from_device(&device, 0)?;
        let before = gov.measure()?;
        let mut balloon = DeviceBalloonAllocator::new(device.clone());
        let t = Instant::now();
        // run_full_balloon always touches, unlike run_balloon's fast path.
        let claimed = gov.run_full_balloon(&mut balloon)?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[real_cuda] FULL balloon: claimed {}MiB in {:.0}ms = {:.1} GiB/s (touch+free+trim)",
            mib(claimed),
            ms,
            (claimed as f64 / GIB as f64) / (ms / 1000.0)
        );
        assert!(claimed > 0);
        // Fully recovered after the balloon's own free+trim.
        let after = gov.measure()?.headroom;
        assert!(
            after >= before.headroom / 2,
            "headroom recovers after full balloon ({}MiB -> {}MiB)",
            mib(before.headroom),
            mib(after)
        );
        Ok(())
    }

    /// Isolate the balloon's per-chunk cost: alloc+touch of one 256 MiB chunk,
    /// and the full-card sync+trim, so we can see whether the balloon is
    /// alloc-bound, memset-bound, or trim-bound.
    #[test]
    fn real_cuda_balloon_chunk_cost() -> Result<()> {
        let Some(device) = cuda_device() else {
            return Ok(());
        };
        let _gpu = gpu_guard();
        let chunk = 256 * MIB;
        // Warm up the allocator/pool once.
        let _ = alloc_kv(&device, chunk)?;
        device.synchronize()?;

        let iters = 16u64;
        let t = Instant::now();
        let mut held = Vec::new();
        for _ in 0..iters {
            held.push(alloc_kv(&device, chunk)?);
        }
        device.synchronize()?;
        let touch_ms = t.elapsed().as_secs_f64() * 1000.0;
        let gib = (iters * chunk) as f64 / GIB as f64;
        eprintln!(
            "[real_cuda] alloc+touch {} × {}MiB = {:.2}GiB in {:.1}ms = {:.1} GiB/s ({:.2}ms/chunk)",
            iters,
            mib(chunk),
            gib,
            touch_ms,
            gib / (touch_ms / 1000.0),
            touch_ms / iters as f64,
        );

        held.clear();
        let t = Instant::now();
        device.synchronize()?;
        if let Device::Cuda(d) = &device {
            let _ = d.trim_pool(0);
        }
        eprintln!(
            "[real_cuda] free+sync+trim of {:.2}GiB: {:.1}ms",
            gib,
            t.elapsed().as_secs_f64() * 1000.0
        );
        Ok(())
    }

    #[test]
    fn real_cuda_relief_ladder_and_critical() -> Result<()> {
        let Some(device) = cuda_device() else {
            return Ok(());
        };
        let _gpu = gpu_guard();

        // Custom config: put the floor ~2 GiB below CURRENT headroom (and zero the
        // percentage terms) so a small, card-size-independent allocation drives us
        // to the Critical rung — no need to fill the whole card.
        let h0 = {
            let g = VramGovernor::from_device(&device, 0)?;
            g.measure()?.headroom
        };
        let floor = h0.saturating_sub(2 * GIB).max(GIB);
        let cfg = GovernorConfig {
            kv_floor_abs: floor,
            kv_floor_pct: 0.0,
            scratch_margin: 0,
            ladder: [
                LadderTier::new(600 * MIB, 0.0), // Trivial
                LadderTier::new(450 * MIB, 0.0), // Cheap
                LadderTier::new(300 * MIB, 0.0), // Moderate
                LadderTier::new(150 * MIB, 0.0), // Costly
                LadderTier::new(0, 0.0),         // Critical == floor
            ],
            balloon_target_frac: 0.95,
            balloon_headroom_abs: 512 * MIB,
            balloon_floor: 512 * MIB,
            balloon_chunk: 256 * MIB,
            critical_min_interval_ms: 0,
        };
        let gov = VramGovernor::from_device_with_config(&device, 0, cfg)?;
        gov.set_capacity(gov.measure()?.total);

        // Shared holder the relief closure can free from.
        let held: Arc<Mutex<Vec<Tensor>>> = Arc::new(Mutex::new(Vec::new()));

        // KV relief at Critical: drop everything held, report the bytes. The
        // governor does the sync+trim around this rung, so the freed VRAM returns
        // to the OS and the remeasure sees it.
        let held_relief = held.clone();
        let bytes_per = Arc::new(AtomicU64::new(0));
        let bp = bytes_per.clone();
        gov.register_relief(
            AllocClass::Kv,
            Criticality::Critical,
            move |_req| {
                let mut v = held_relief.lock().unwrap();
                let n = v.len() as u64;
                v.clear();
                let freed = n.saturating_mul(bp.load(Ordering::Relaxed));
                ReliefOutcome::new(freed)
            },
            {
                let held_ev = held.clone();
                let bp2 = bytes_per.clone();
                move || held_ev.lock().unwrap().len() as u64 * bp2.load(Ordering::Relaxed)
            },
        );

        // Allocate to just below the floor so Critical trips. Bounded, guarded.
        let chunk = 256 * MIB;
        bytes_per.store(chunk, Ordering::Relaxed);
        let target_headroom = floor.saturating_sub(chunk); // strictly below floor
        let mut allocated = 0u64;
        loop {
            let hr = gov.measure()?.headroom;
            if hr <= target_headroom {
                break;
            }
            match alloc_kv(&device, chunk) {
                Ok(t) => {
                    held.lock().unwrap().push(t);
                    gov.credit_class(AllocClass::Kv, chunk);
                    allocated += chunk;
                }
                Err(_) => break, // real OOM: stop, still exercise relief
            }
            if allocated > h0 {
                break; // safety
            }
        }
        let h_pressure = gov.measure()?.headroom;
        eprintln!(
            "[real_cuda] pressure: allocated {}MiB, headroom={}MiB, floor={}MiB",
            mib(allocated),
            mib(h_pressure),
            mib(floor)
        );
        eprintln!("{}", gov.render_budget("under pressure"));

        // Relieve: should climb to Critical (sync+trim) and drop the held KV.
        let t = Instant::now();
        let res = gov.relieve_pressure(AllocClass::Kv)?;
        let relief_ms = t.elapsed().as_secs_f64() * 1000.0;
        let h_after = gov.measure()?.headroom;
        eprintln!(
            "[real_cuda] relief result={:?} sync_count={} headroom {}MiB -> {}MiB in {:.1}ms",
            res,
            gov.sync_count(),
            mib(h_pressure),
            mib(h_after),
            relief_ms
        );

        if allocated >= 2 * GIB {
            // We genuinely reached pressure: Critical must have fired (2 syncs) and
            // headroom must have recovered past what we freed.
            assert_eq!(gov.sync_count(), 2, "Critical syncs before and after");
            assert!(res.freed() > 0, "relief freed the held KV");
            assert!(
                h_after > h_pressure,
                "headroom recovers after Critical relief ({}MiB -> {}MiB)",
                mib(h_pressure),
                mib(h_after)
            );
        } else {
            eprintln!("[real_cuda] only {}MiB free to allocate; pressure not reached, relief exercised without hard asserts", mib(allocated));
        }

        // Cleanup.
        held.lock().unwrap().clear();
        device.synchronize()?;
        if let Device::Cuda(d) = &device {
            let _ = d.trim_pool(0);
        }
        Ok(())
    }

    /// The pool trim routes through the registered arena-topology guard: the
    /// wrapper decides whether the trim runs — skipping it while a migrate holds
    /// the topology, running it otherwise.
    ///
    /// The wrapper is gated on a flag this test clears again before returning,
    /// and the test holds [`gpu_guard`], because `POOL_TRIM_GUARD` is a
    /// `OnceLock`: process-global, first-registration-wins, and impossible to
    /// remove. A wrapper that skipped unconditionally therefore disabled the
    /// Critical rung's trim for the whole rest of the binary — after which
    /// `real_cuda_relief_ladder_and_critical` measured no headroom recovery
    /// (11936MiB -> 11936MiB, relief "Relieved(2.5GiB)" in 0.1ms instead of
    /// 47.7ms) and failed. It failed only when the harness happened to schedule
    /// this test first, which is what made it look like GPU flakiness.
    #[test]
    fn guarded_pool_trim_routes_through_the_registered_guard() {
        static GUARD_CALLS: AtomicU64 = AtomicU64::new(0);
        static TRIM_RUNS: AtomicU64 = AtomicU64::new(0);
        static SKIP_TRIM: AtomicBool = AtomicBool::new(false);
        let _gpu = gpu_guard();
        set_pool_trim_guard(Box::new(|trim| {
            GUARD_CALLS.fetch_add(1, Ordering::SeqCst);
            if !SKIP_TRIM.load(Ordering::SeqCst) {
                trim();
            }
        }));

        // Migrate in flight: the wrapper swallows the trim.
        SKIP_TRIM.store(true, Ordering::SeqCst);
        guarded_pool_trim(|| {
            TRIM_RUNS.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(
            GUARD_CALLS.load(Ordering::SeqCst),
            1,
            "the registered guard wrapper was invoked"
        );
        assert_eq!(
            TRIM_RUNS.load(Ordering::SeqCst),
            0,
            "the guard skipped the underlying trim"
        );

        // Topology free again: the same wrapper must let the trim through — and
        // leaving it in THIS state is what keeps the rest of the binary honest.
        SKIP_TRIM.store(false, Ordering::SeqCst);
        guarded_pool_trim(|| {
            TRIM_RUNS.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(
            GUARD_CALLS.load(Ordering::SeqCst),
            2,
            "the wrapper is consulted on every trim"
        );
        assert_eq!(
            TRIM_RUNS.load(Ordering::SeqCst),
            1,
            "the guard runs the trim once the topology is free"
        );
    }
}
