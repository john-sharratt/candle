//! Unit tests for the VRAM Governor. Everything here runs on CPU via the
//! scripted `FakeProbe` / `FakeBalloonAllocator` — no GPU required, except the
//! `real_cuda` module. What is covered: measurement and the balloon, the KV
//! floor and the expert budget, the per-class tallies, and the diag table.
//!
//! The relief-ladder half of the matrix — gentle-early relief, escalation,
//! Critical-only sync, no-spin, the concurrency forecast and the OOM retry —
//! went with the ladder itself (`docs/archived/arena_unification.md` §5).

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
        balloon_target_frac: 0.90,
        // Small enough not to bind on the 64 GiB test cards (the 0.90 fraction
        // stays the binding term), so existing target assertions hold.
        balloon_headroom_abs: 512 * MIB,
        balloon_floor: 512 * MIB,
        balloon_chunk: 256 * MIB,
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
    assert_eq!(
        gov.kv_floor(),
        floor,
        "a second load must not move the floor"
    );
    assert_eq!(gov.class_reserved(AllocClass::Weights), 8 * GIB);
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
fn budget_evolves_with_allocations() {
    let vram = FakeVram::new(50 * GIB, 73 * GIB);
    let gov = governed(&vram, 73 * GIB, 2 * GIB);
    gov.credit_class(AllocClass::Expert, 15 * GIB);
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

/// The balloon's absolute reserve must not re-book the scratch cushion.
///
/// `scratch_margin` is already subtracted in `expert_budget` and sits below
/// every relief rung, so reserving the transient peak again in the balloon cap
/// books the same bytes twice — and unlike `scratch_margin`, those bytes are
/// never handed to *any* allocator, because `C` is the ceiling every budget is
/// derived from. Measured on the 16 GiB card: a 2.5 GiB cap held `C` at 13488
/// MiB while the ceiling the balloon finds when allowed to look is 14592.
///
/// The cap's job is only to stop the *measurement* destabilising the desktop.
/// It must therefore stay at or below the scratch cushion — anything larger is
/// reserving engine headroom a second time.
///
/// Reads the shipped constants directly rather than `GovernorConfig::default()`,
/// which resolves `CANDLE_VRAM_*` overrides: this project uses those knobs
/// routinely, and a developer with one exported would otherwise see this fail
/// for reasons that have nothing to do with the defaults it exists to pin.
#[test]
fn the_balloon_reserve_does_not_double_book_the_scratch_cushion() {
    let cfg = GovernorConfig::defaults_ignoring_env();
    assert!(
        cfg.balloon_headroom_abs <= cfg.scratch_margin,
        "balloon reserve {} would re-book the {} scratch cushion",
        cfg.balloon_headroom_abs,
        cfg.scratch_margin
    );
}

/// A cap set above the card's real ceiling costs nothing: the balloon stops
/// where the driver refuses, which is the honest capacity. This is what lets
/// the default be generous — on a card that genuinely cannot hold that much,
/// the allocation failure binds first.
#[test]
fn balloon_stops_at_the_real_ceiling_when_the_cap_is_generous() -> Result<()> {
    let mut cfg = test_config();
    cfg.balloon_target_frac = 0.99;
    cfg.balloon_headroom_abs = 128 * MIB; // effectively no cap

    // The card reports 16 GiB but only 12 GiB can actually be claimed.
    let total = 16 * GIB;
    let vram = FakeVram::new(total, total);
    let mut alloc = FakeBalloonAllocator::new(vram.clone(), 12 * GIB);
    let c = super::balloon::balloon_measure(&vram.probe(), &mut alloc, &cfg)?;
    assert_eq!(
        c,
        12 * GIB,
        "the driver's refusal is the ceiling, not the cap"
    );
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

// ── Managed allocation & retry ───────────────────────────────────────────────

// ── Forecast ─────────────────────────────────────────────────────────────────

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

/// The expert cache must be sized inside the balloon-measured capacity, not
/// against raw driver headroom.
///
/// On a WDDM card the driver reports materially more free memory than can
/// actually be held resident — finding that gap is what the balloon is for.
/// Measured on the 16 GiB dev card: `C` = 13488 MiB, headroom at expert load
/// ~15000 MiB. Sizing against headroom took 8888 MiB for experts where the
/// capacity allowed ~6000, and since nothing can ever shed an expert slot, the
/// pool then sat permanently above `C` with the driver still reporting free
/// memory — every later KV allocation refused on the pool's own ceiling.
#[test]
fn expert_budget_is_bounded_by_capacity_not_driver_headroom() -> Result<()> {
    const MIB_U: u64 = 1 << 20;
    // Card reports 15000 MiB free of 16375; the balloon proved only 13488 is
    // holdable. `set_capacity` records 15000 as the baseline, so nothing of `C`
    // is spent yet and the whole of it is available beyond floor + cushion.
    let vram = FakeVram::new(15000 * MIB_U, 16375 * MIB_U);
    let gov = VramGovernor::new(0, Box::new(vram.probe()), test_config());
    gov.set_capacity(13488 * MIB_U);

    let budget = gov.expert_budget()?;
    assert_eq!(
        budget,
        13488 * MIB_U - gov.kv_floor() - gov.scratch_margin(),
        "capacity, not headroom, is the ceiling"
    );
    assert!(
        budget < 15000 * MIB_U - gov.kv_floor() - gov.scratch_margin(),
        "sizing against headroom would have allowed more than the card can hold"
    );
    Ok(())
}

/// The spend against `C` is the drop in headroom since `C` was measured — NOT
/// `total - headroom`.
///
/// DXGI reports `headroom = Budget - CurrentUsage`, so `total - headroom` also
/// carries `total - Budget`: the OS reserve, which the balloon already excluded
/// from `C`. Charging it again costs roughly a gigabyte of expert residency on
/// the 16 GiB card. Here the reserve is 1375 MiB and we have since spent 500;
/// only the 500 may be deducted.
#[test]
fn expert_budget_charges_our_own_spend_not_the_os_reserve() -> Result<()> {
    const MIB_U: u64 = 1 << 20;
    let total = 16375 * MIB_U;
    let vram = FakeVram::new(15000 * MIB_U, total);
    let gov = VramGovernor::new(0, Box::new(vram.probe()), test_config());
    gov.set_capacity(13488 * MIB_U); // baseline headroom = 15000 MiB

    vram.set(14500 * MIB_U); // we allocated 500 MiB
    let budget = gov.expert_budget()?;
    assert_eq!(
        budget,
        13488 * MIB_U - 500 * MIB_U - gov.kv_floor() - gov.scratch_margin(),
        "only our own 500 MiB is spent; the 1375 MiB OS reserve is already in C"
    );
    // The discredited form would have charged `total - headroom` = 1875 MiB.
    let double_booked = 13488 * MIB_U - 1875 * MIB_U - gov.kv_floor() - gov.scratch_margin();
    assert!(
        budget > double_booked,
        "double-booking the reserve costs {} MiB",
        (budget - double_booked) / MIB_U
    );
    Ok(())
}

/// Headroom recovering above the baseline (another process released memory)
/// must not underflow into a bogus spend.
#[test]
fn expert_budget_handles_headroom_returning_above_the_baseline() -> Result<()> {
    const MIB_U: u64 = 1 << 20;
    let vram = FakeVram::new(12000 * MIB_U, 16375 * MIB_U);
    let gov = VramGovernor::new(0, Box::new(vram.probe()), test_config());
    gov.set_capacity(13488 * MIB_U); // baseline headroom = 12000 MiB

    vram.set(15000 * MIB_U); // a neighbour freed 3 GiB
    let budget = gov.expert_budget()?;
    assert_eq!(
        budget,
        13488 * MIB_U - gov.kv_floor() - gov.scratch_margin(),
        "nothing of C is spent, and the surplus does not inflate past C"
    );
    Ok(())
}

/// Before the balloon has run there is no capacity to bound against, so the
/// live reading is the only measurement available.
#[test]
fn expert_budget_falls_back_to_headroom_before_the_balloon_runs() -> Result<()> {
    let vram = FakeVram::new(20 * GIB, 24 * GIB);
    let gov = VramGovernor::new(0, Box::new(vram.probe()), test_config());
    // No `set_capacity` — C is still 0.
    let budget = gov.expert_budget()?;
    assert_eq!(
        budget,
        20 * GIB - gov.kv_floor() - gov.scratch_margin(),
        "with no capacity measurement the budget is headroom-derived"
    );
    Ok(())
}

// ── External pressure & recovery ─────────────────────────────────────────────

// ── Diagnostics ──────────────────────────────────────────────────────────────

#[test]
fn budget_table_shape() {
    let vram = FakeVram::new(20 * GIB, 73 * GIB);
    let gov = governed(&vram, 73 * GIB, 2 * GIB);
    let t = gov.budget_table();
    assert_eq!(t.capacity_c, 73 * GIB);
    assert_eq!(t.total, 73 * GIB);
    assert_eq!(t.headroom, 20 * GIB);
    assert_eq!(t.rows.len(), AllocClass::COUNT);
    assert_eq!(t.reserved(AllocClass::Weights), 2 * GIB);
    assert_eq!(t.kv_floor, gov.kv_floor());
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

// ── Inference-engine run scenarios (CPU, modelled on the real lifecycle) ──────
//
// A `Sim` wires the governor to a fake VRAM cell and a KV/expert residency model
// the way the real subsystems will register relief, then the tests drive the
// actual engine flow: boot → load weights/experts/scratch → check that the
// startup partition leaves the KV floor intact.

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
    }

    impl Sim {
        /// Boot on a free card and measure capacity (balloon fast-path).
        fn boot(total: u64) -> Self {
            let vram = FakeVram::new(total - CONTEXT, total);
            let gov = VramGovernor::new(0, Box::new(vram.probe()), cfg());
            gov.set_capacity(total - CONTEXT);
            Sim { gov, vram }
        }

        fn load_weights(&self, bytes: u64) {
            self.gov.credit_class(AllocClass::Weights, bytes);
            self.vram.consume(bytes);
        }

        fn load_experts(&self, resident: u64) {
            self.gov.credit_class(AllocClass::Expert, resident);
            self.vram.consume(resident);
        }

        fn load_scratch(&self, bytes: u64) {
            self.gov.credit_class(AllocClass::Scratch, bytes);
            self.vram.consume(bytes);
        }

        fn headroom(&self) -> u64 {
            self.vram.headroom()
        }
    }

    /// A fully loaded engine ready to serve: weights + experts + scratch in.
    fn loaded(total: u64) -> Sim {
        let s = Sim::boot(total);
        s.load_weights(2 * GIB);
        s.load_experts(16 * GIB);
        s.load_scratch(2 * GIB);
        s
    }

    #[test]
    fn startup_partitions_evolve_and_leave_kv_floor() {
        let s = loaded(CARD);
        let t = s.gov.budget_table();
        assert_eq!(t.reserved(AllocClass::Weights), 2 * GIB);
        assert_eq!(t.reserved(AllocClass::Expert), 16 * GIB);
        assert_eq!(t.reserved(AllocClass::Scratch), 2 * GIB);
        // KV region (current headroom) is well above the floor at boot.
        assert!(
            s.headroom() > s.gov.kv_floor(),
            "KV headroom above floor at boot"
        );
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
}
