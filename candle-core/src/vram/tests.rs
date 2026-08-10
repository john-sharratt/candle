//! Unit tests for the VRAM Governor. Everything here runs on CPU via the
//! scripted `FakeProbe` / `FakeBalloonAllocator` — no GPU required, except the
//! `real_cuda` module. What is covered: measurement and the balloon, `usable`
//! and the span it sizes, the per-class tallies, and the diag table.
//!
//! The relief-ladder half of the matrix — gentle-early relief, escalation,
//! Critical-only sync, no-spin, the concurrency forecast and the OOM retry —
//! went with the ladder itself (`docs/archived/arena_unification.md` §5). The
//! `kv_floor` / `expert_budget` half went with the static partition
//! (`docs/elastic_vram_partition.md` §9).

use std::sync::Arc;

use super::balloon::FakeBalloonAllocator;
use super::reading::fake::FakeVram;
use super::*;
use crate::Result;

const GIB: u64 = 1024 * 1024 * 1024;
const MIB: u64 = 1024 * 1024;

fn test_config() -> GovernorConfig {
    GovernorConfig {
        scratch_margin: GIB,
        capacity_reserve: 512 * MIB,
        balloon_chunk: 256 * MIB,
        balloon_min_chunk: 2 * MIB,
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
/// The tally no longer sizes anything, but it is still what every report of the
/// card's decomposition is read from, and a doubled one is wrong by a whole
/// model.
#[test]
fn reloading_the_model_replaces_the_weight_tally() {
    let vram = FakeVram::new(64 * GIB, 64 * GIB);
    let gov = governed(&vram, 16 * GIB, 8 * GIB);
    assert_eq!(gov.class_reserved(AllocClass::Weights), 8 * GIB);

    gov.set_class(AllocClass::Weights, 8 * GIB);
    assert_eq!(
        gov.class_reserved(AllocClass::Weights),
        8 * GIB,
        "a second load must replace the tally, not stack on it"
    );
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
fn the_pool_cushion_is_the_only_thing_held_back() {
    let vram = FakeVram::new(0, 73 * GIB);
    let gov = governed(&vram, 73 * GIB, 2 * GIB);
    assert_eq!(gov.pool_cushion(), GIB, "test config's cushion");
    // Independent of the card and of what is loaded: it covers the CUDA pool,
    // not a share of anything. `kv_floor` was the term that scaled with both,
    // and the reason it had to go is that the quantity it was trying to size —
    // how much KV the workload needs — is not knowable at load.
    let small = governed(&FakeVram::new(0, 16 * GIB), 16 * GIB, 8 * GIB);
    assert_eq!(small.pool_cushion(), gov.pool_cushion());
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

/// **The reserve applies on the path that actually runs.**
///
/// `run_balloon` skips the touch when the card is already free, which is the
/// normal startup case, and that path used to take `headroom.min(total)` — the
/// whole card. The reserve existed only as the *threshold* deciding whether to
/// skip. So the one number that was supposed to guarantee "never allocate
/// everything" was, on an uncontended card, not applied at all.
#[test]
fn the_reserve_applies_on_the_fast_path_too() -> Result<()> {
    let total = 64 * GIB;
    let vram = FakeVram::new(total, total);
    let cfg = test_config();
    let gov = VramGovernor::new(0, Box::new(vram.probe()), cfg.clone());
    let mut alloc = FakeBalloonAllocator::new(vram.clone(), 60 * GIB);
    let c = gov.run_balloon(&mut alloc)?;
    assert_eq!(
        c,
        total - cfg.capacity_reserve,
        "a free card must still hold back the reserve"
    );
    assert!(c < total, "C must never be the whole card");
    // The allocator was never touched — headroom is exactly as it started.
    assert_eq!(vram.headroom(), total);
    Ok(())
}

/// Both paths agree, on every card size. The two used to be written separately
/// (a fraction here, an absolute there) and disagreed by 818 MiB on the dev card.
#[test]
fn both_balloon_paths_target_total_less_the_reserve() -> Result<()> {
    let cfg = test_config();
    for total in [8 * GIB, 16 * GIB, 24 * GIB, 64 * GIB, 96 * GIB] {
        let target = super::balloon::capacity_target(total, cfg.capacity_reserve);
        assert_eq!(target, total - cfg.capacity_reserve);

        // Fast path (headroom ≥ target).
        let vram = FakeVram::new(total, total);
        let gov = VramGovernor::new(0, Box::new(vram.probe()), cfg.clone());
        let mut alloc = FakeBalloonAllocator::new(vram.clone(), total);
        assert_eq!(gov.run_balloon(&mut alloc)?, target, "fast path @ {total}");

        // Growth loop (forced), uncontended so nothing refuses.
        let vram = FakeVram::new(total, total);
        let mut alloc = FakeBalloonAllocator::new(vram.clone(), total);
        let claimed = super::balloon::balloon_measure(&vram.probe(), &mut alloc, &cfg)?;
        assert_eq!(claimed, target, "growth loop @ {total}");
        // Balloon freed: headroom restored.
        assert_eq!(vram.headroom(), total);
    }
    Ok(())
}

/// The driver's refusal is the ceiling, and the claim lands **within one granule
/// of it** rather than within one 256 MiB chunk.
///
/// This is the whole point of refining the chunk on refusal: `C` is what every
/// later partition is sized from, so an under-measurement here is permanent.
/// The ceiling is deliberately not a multiple of the chunk — a ceiling that
/// happened to land on a chunk boundary would pass with or without refinement.
#[test]
fn refusal_refines_the_chunk_to_within_a_granule_of_the_ceiling() -> Result<()> {
    let cfg = test_config();
    let total = 16 * GIB;
    let ceiling = 12 * GIB + 100 * MIB; // 100 MiB past a chunk boundary

    let vram = FakeVram::new(total, total);
    let mut alloc = FakeBalloonAllocator::new(vram.clone(), ceiling);
    let c = super::balloon::balloon_measure(&vram.probe(), &mut alloc, &cfg)?;

    assert!(c <= ceiling, "claimed {c} past the ceiling {ceiling}");
    assert!(
        ceiling - c < cfg.balloon_min_chunk,
        "left {} B unclaimed; refinement should get within {} B",
        ceiling - c,
        cfg.balloon_min_chunk,
    );
    // Specifically: a fixed 256 MiB chunk would have stopped at 12 GiB exactly.
    assert!(
        c > 12 * GIB,
        "did not refine past the chunk boundary at all (got {c})"
    );
    Ok(())
}

/// Refinement must not run past the granule: a ceiling that is not a multiple of
/// `balloon_min_chunk` leaves the remainder unclaimed rather than looping.
#[test]
fn refinement_stops_at_the_minimum_chunk() -> Result<()> {
    let mut cfg = test_config();
    cfg.balloon_min_chunk = 64 * MIB;
    let total = 16 * GIB;
    let ceiling = 12 * GIB + 100 * MIB;

    let vram = FakeVram::new(total, total);
    let mut alloc = FakeBalloonAllocator::new(vram.clone(), ceiling);
    let c = super::balloon::balloon_measure(&vram.probe(), &mut alloc, &cfg)?;
    assert_eq!(
        c,
        12 * GIB + 64 * MIB,
        "one 64 MiB step fits inside the 100 MiB remainder; a second does not"
    );
    Ok(())
}

/// A target above the card's real ceiling costs nothing: the balloon stops where
/// the driver refuses, which is the honest capacity.
#[test]
fn balloon_stops_at_the_real_ceiling() -> Result<()> {
    let mut cfg = test_config();
    cfg.capacity_reserve = 128 * MIB; // effectively no cap

    // The card reports 16 GiB but only 12 GiB can actually be claimed.
    let total = 16 * GIB;
    let vram = FakeVram::new(total, total);
    let mut alloc = FakeBalloonAllocator::new(vram.clone(), 12 * GIB);
    let c = super::balloon::balloon_measure(&vram.probe(), &mut alloc, &cfg)?;
    assert_eq!(
        c,
        12 * GIB,
        "the driver's refusal is the ceiling, not the target"
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

// ── `usable`: what the reservation is sized from ─────────────────────────────
//
// These were `expert_budget` tests. The function is gone — the resident-expert
// count is the weight zone's capacity now, not a byte budget divided by an
// expert size — but every property they pinned belongs to `usable()`, which is
// the one measurement the span is taken from. Nothing here changed except the
// subtraction that used to sit on top of it.

/// **The dense weights are subtracted exactly once.**
///
/// The load order now makes them resident *and* tallied when the span is sized,
/// so `usable − class_reserved(Weights)` looks like prudence. It is the same
/// bytes twice: the weights are already inside the headroom drop `usable`
/// measures. This codebase has made that mistake twice (`balloon_headroom_abs`
/// against `expert_budget`, 1,104 MiB; `scratch_margin` against the transient
/// tier, "same bytes, two places, opposite signs"), and neither was found by
/// reasoning about it — so the tally is deliberately recorded here, and a future
/// reader who reaches for it fails this test instead of silently halving the KV
/// side.
#[test]
fn the_weights_are_subtracted_once_not_twice() -> Result<()> {
    const MIB_U: u64 = 1 << 20;
    let total = 16375 * MIB_U;
    let capacity = 14592 * MIB_U;
    for dense_mib in [0u64, 1024, 4096] {
        let dense = dense_mib * MIB_U;
        let vram = FakeVram::new(capacity, total);
        let gov = VramGovernor::new(0, Box::new(vram.probe()), test_config());
        gov.set_capacity(capacity); // baseline headroom = capacity

        // The dense weights load: headroom drops, and the tally records them.
        vram.set(capacity - dense);
        gov.set_class(AllocClass::Weights, dense);

        let usable = gov.usable()?;
        assert_eq!(
            usable,
            capacity - dense,
            "usable already nets out the resident weights at {dense_mib} MiB"
        );
        // The span. Note what is *absent*: no second `- gov.class_reserved(Weights)`.
        let span = usable - gov.pool_cushion();
        assert_eq!(span, capacity - dense - gov.pool_cushion());
        assert!(
            gov.class_reserved(AllocClass::Weights) == dense,
            "the tally is recorded — the point is that it is not also subtracted"
        );
    }
    Ok(())
}

/// The span must be sized inside the balloon-measured capacity, not against raw
/// driver headroom.
///
/// On a WDDM card the driver reports materially more free memory than can
/// actually be held resident — finding that gap is what the balloon is for.
/// Measured on the 16 GiB dev card: `C` = 13488 MiB, headroom at expert load
/// ~15000 MiB. Sizing against headroom claimed 1512 MiB the card could not hold,
/// and every later allocation ran into a pool whose `used` sat above `C` with
/// the driver still reporting free memory.
#[test]
fn usable_is_bounded_by_capacity_not_driver_headroom() -> Result<()> {
    const MIB_U: u64 = 1 << 20;
    let vram = FakeVram::new(15000 * MIB_U, 16375 * MIB_U);
    let gov = VramGovernor::new(0, Box::new(vram.probe()), test_config());
    gov.set_capacity(13488 * MIB_U);

    assert_eq!(
        gov.usable()?,
        13488 * MIB_U,
        "capacity, not headroom, is the ceiling"
    );
    assert!(gov.usable()? < 15000 * MIB_U);
    Ok(())
}

/// The spend against `C` is the drop in headroom since `C` was measured — NOT
/// `total - headroom`.
///
/// DXGI reports `headroom = Budget - CurrentUsage`, so `total - headroom` also
/// carries `total - Budget`: the OS reserve, which the balloon already excluded
/// from `C`. Charging it again costs roughly a gigabyte on the 16 GiB card. Here
/// the reserve is 1375 MiB and we have since spent 500; only the 500 counts.
#[test]
fn usable_charges_our_own_spend_not_the_os_reserve() -> Result<()> {
    const MIB_U: u64 = 1 << 20;
    let vram = FakeVram::new(15000 * MIB_U, 16375 * MIB_U);
    let gov = VramGovernor::new(0, Box::new(vram.probe()), test_config());
    gov.set_capacity(13488 * MIB_U); // baseline headroom = 15000 MiB

    vram.set(14500 * MIB_U); // we allocated 500 MiB
    assert_eq!(
        gov.usable()?,
        13488 * MIB_U - 500 * MIB_U,
        "only our own 500 MiB is spent; the 1375 MiB OS reserve is already in C"
    );
    // The discredited form would have charged `total - headroom` = 1875 MiB.
    assert!(gov.usable()? > 13488 * MIB_U - 1875 * MIB_U);
    Ok(())
}

/// Headroom recovering above the baseline (another process released memory)
/// must not underflow into a bogus spend.
#[test]
fn usable_handles_headroom_returning_above_the_baseline() -> Result<()> {
    const MIB_U: u64 = 1 << 20;
    let vram = FakeVram::new(12000 * MIB_U, 16375 * MIB_U);
    let gov = VramGovernor::new(0, Box::new(vram.probe()), test_config());
    gov.set_capacity(13488 * MIB_U); // baseline headroom = 12000 MiB

    vram.set(15000 * MIB_U); // a neighbour freed 3 GiB
    assert_eq!(
        gov.usable()?,
        13488 * MIB_U,
        "nothing of C is spent, and the surplus does not inflate past C"
    );
    Ok(())
}

/// Before the balloon has run there is no capacity to bound against, so the
/// live reading is the only measurement available.
#[test]
fn usable_falls_back_to_headroom_before_the_balloon_runs() -> Result<()> {
    let vram = FakeVram::new(20 * GIB, 24 * GIB);
    let gov = VramGovernor::new(0, Box::new(vram.probe()), test_config());
    // No `set_capacity` — C is still 0.
    assert_eq!(gov.usable()?, 20 * GIB);
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
    assert_eq!(t.pool_cushion, gov.pool_cushion());
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
// actual engine flow: boot → load weights → check that the span the reservation
// takes is exactly what they left behind.

mod scenarios {
    use super::*;

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
    fn startup_partitions_evolve_and_report() {
        let s = loaded(CARD);
        let t = s.gov.budget_table();
        assert_eq!(t.reserved(AllocClass::Weights), 2 * GIB);
        assert_eq!(t.reserved(AllocClass::Expert), 16 * GIB);
        assert_eq!(t.reserved(AllocClass::Scratch), 2 * GIB);
        assert_eq!(t.pool_cushion, s.gov.pool_cushion());
    }

    /// The span the reservation takes, on a roomy card and a tight one.
    ///
    /// This was `expert_budget_all_resident_vs_partial`. The residency question
    /// moved to the weight zone, which answers it from the span rather than from
    /// a budget — so what is left to pin here is the span itself: everything
    /// `usable` reports once the dense weights are in, less the pool cushion,
    /// and nothing else.
    #[test]
    fn the_span_is_what_the_weights_left_behind() -> Result<()> {
        for (total, dense) in [(CARD, 2 * GIB), (24 * GIB, 2 * GIB)] {
            let s = Sim::boot(total);
            s.load_weights(dense);
            let span = s.gov.usable()? - s.gov.pool_cushion();
            assert_eq!(
                span,
                (total - CONTEXT) - dense - s.gov.pool_cushion(),
                "the span on a {} GiB card",
                total / GIB
            );
            assert!(span > 0);
        }
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
