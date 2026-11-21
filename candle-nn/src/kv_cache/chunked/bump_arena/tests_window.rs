//! Tests for when an arena may be created and which callers are refused.
//!
//! An arena cannot be created while a wave is open. A sealing thread is told so
//! with a retry marker and not parked, while the forward's own thread is refused
//! without the marker, since a retry there would spin. A tier left standing
//! between forwards does not block a window, and a tier handed back for an arena
//! takes its plan with it. An open window also holds the stream lease, so a
//! recording cannot start under it.

use super::*;

#[cfg(all(test, feature = "cuda"))]
pub(super) mod wave_tests {
    use super::{
        begin_forward, begin_wave, enter_arena_window, plan_wave_transient, tier_footprint,
        LayerPhase,
    };

    use candle::{Device, Result};

    /// The wave domain is process-global and `cargo test` runs tests in
    /// parallel, so these take turns. It is the *crate-wide* lock rather than a
    /// local one: the KV selection tests build backings on the same device, and
    /// a lock that only excludes this module would not exclude them.
    use super::super::super::gpu_test_lock::gpu_serial as serial;

    pub(in super::super) fn stream() -> Option<candle::cuda_backend::StreamLease> {
        match Device::new_cuda(0) {
            Ok(Device::Cuda(d)) => Some(d.cuda_stream()),
            _ => None,
        }
    }

    /// A lease for a window, on the device [`stream`] found.
    pub(in super::super) fn lease() -> candle::cuda_backend::StreamLease {
        stream().expect("the device the test's stream came from")
    }

    /// **An arena cannot be created while a wave is open — and the caller is
    /// told so rather than parked.**
    ///
    /// The wave is pre-allocated end to end and its transient tier is placed
    /// against the arena frontier as it stands when the forward begins, so
    /// creating an arena mid-wave moves that frontier under it. Refusing is what
    /// keeps that from being a *blocking* edge: this is a leaf of the allocator,
    /// reached under whatever locks its caller holds, and a wait here deadlocked
    /// the daemon three times against three different locks.
    ///
    /// The refusal carries [`KV_ARENA_MID_WAVE`] so a sealing pass can tell it
    /// from a real allocation failure and retry on the next one.
    #[test]
    fn creating_an_arena_mid_wave_is_refused_not_parked() -> Result<()> {
        use std::sync::mpsc;

        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let guard = begin_wave(&s, LayerPhase::Attention)?;

        // From another thread, so this is the ordinary sealing case rather than
        // the forward's own thread.
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(
                enter_arena_window(lease(), "a test tenant")
                    .err()
                    .map(|e| e.to_string()),
            );
        })
        .join()
        .expect("prober panicked");

        let err = rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("the prober must answer immediately, never park")
            .expect("an arena window must not open with a wave generation live");
        assert!(
            err.contains(super::KV_ARENA_MID_WAVE),
            "the refusal must be marked retryable so the sealing pass comes back: {err}"
        );

        // Once the wave is over the same request succeeds.
        drop(guard);
        drop(enter_arena_window(lease(), "a test tenant")?);
        Ok(())
    }

    /// **An open window keeps a recording from starting on another thread.**
    /// The window holds its stream lease for as long as it stands, so a wave
    /// opened beside it runs its segment as issued; once it closes, the same
    /// wave records. Entered and let go at once, the lease let a segment start
    /// recording under a standing window, wait on the window for its tier, and
    /// hold the window's thread on the recording — a deadlock that stalled both
    /// persistence threads on `.5` for 30 s at a time.
    #[test]
    fn an_open_window_keeps_a_recording_from_starting() -> Result<()> {
        use std::sync::mpsc;

        let _serial = serial();
        let Ok(Device::Cuda(dev)) = Device::new_cuda(0) else {
            return Ok(());
        };
        let (opened, window_open) = mpsc::channel();
        let (close, closing) = mpsc::channel::<()>();
        let holder = std::thread::spawn(move || -> Result<()> {
            let window = enter_arena_window(lease(), "a test tenant")?;
            opened.send(()).expect("the test is listening");
            let _ = closing.recv();
            drop(window);
            Ok(())
        });
        window_open
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("the window opened");

        let wave = dev.begin_wave_capture()?;
        dev.record_launches()?;
        assert_eq!(
            dev.recording_segment(),
            None,
            "a segment began recording while another thread's window stood"
        );
        drop(wave);

        close.send(()).expect("the holder is waiting");
        holder.join().expect("the holder panicked")?;
        let wave = dev.begin_wave_capture()?;
        dev.record_launches()?;
        assert!(
            dev.recording_segment().is_some(),
            "with the window closed the wave records"
        );
        drop(wave);
        Ok(())
    }

    /// **The forward's own thread gets a different refusal, and deliberately no
    /// retry marker.**
    ///
    /// A sealing thread refused mid-wave should come back later; the forward
    /// thread asking for an arena it should have claimed in admit is a placement
    /// bug, and a caller looping on [`KV_ARENA_MID_WAVE`] would spin on it
    /// forever. Opening the forward before admit did exactly this and stopped
    /// the engine.
    #[test]
    fn the_forward_thread_is_told_it_is_a_placement_bug() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let open = begin_forward(&s);
        let err = enter_arena_window(lease(), "a test tenant")
            .err()
            .expect("the forward's own thread must be refused")
            .to_string();
        assert!(
            err.contains("admit_wave_kv"),
            "the error must say where the allocation belonged instead: {err}"
        );
        // The tenant, because "which tenant asked" is the first question this
        // refusal raises and the message used to answer it with the wrong one.
        assert!(
            err.contains("a test tenant"),
            "the error must name the tenant that asked: {err}"
        );
        assert!(
            !err.contains(super::KV_ARENA_MID_WAVE),
            "a placement bug must not be marked retryable, or the caller spins: {err}"
        );
        drop(open);
        // And once the forward is over, the same thread is an ordinary caller.
        drop(enter_arena_window(lease(), "a test tenant")?);
        Ok(())
    }

    /// **A tier left standing between forwards must not refuse an arena window.**
    ///
    /// The tier is not handed back when a forward ends — it stands until the
    /// next forward's phase 0 returns it. So a caller that keyed on `placed_at`
    /// would be refused (or, in the first version, parked) over ground that is
    /// nobody's: 55 threads asleep at zero CPU with `tier=96MiB` above an
    /// otherwise idle partition and the status endpoint frozen at `Prefilling
    /// tool sections`. The caller has to hand the tier back itself.
    #[test]
    fn an_idle_tier_does_not_block_an_arena_window() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let ordinal = s.context().ordinal();

        // A forward places its tier and returns; nothing releases it.
        {
            let _open = begin_forward(&s);
            plan_wave_transient(&s, [1 << 20, 1 << 20, 1 << 20])?;
        }
        let placed = super::super::super::region_pool::region_stats(ordinal)
            .map(|r| r.transient_bytes)
            .unwrap_or(0);
        assert!(placed > 0, "the forward should have left a tier standing");

        // The window opens anyway, and the ground comes back with it.
        let window = enter_arena_window(lease(), "a test tenant")?;
        assert_eq!(
            super::super::super::region_pool::region_stats(ordinal)
                .map(|r| r.transient_bytes)
                .unwrap_or(0),
            0,
            "the waiter must hand back the tier it waited past, or the ground it \
             is about to carve an arena from is still spoken for"
        );
        drop(window);
        Ok(())
    }

    /// **A tier handed back for an arena takes its plan with it.** Left behind,
    /// the plan is re-placed by the next guard that finds no tier: a draft walk
    /// that priced itself into a prefill's standing tier, then claimed its KV
    /// through an arena window, re-bought the prefill's whole plan and was
    /// refused against the KV that had grown since.
    #[test]
    fn a_tier_handed_back_for_an_arena_takes_its_plan_with_it() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let ordinal = s.context().ordinal();
        let fallback = tier_footprint(super::fallback_plan());
        // A forward wider than anything the fallback plans for.
        let wide = super::fallback_plan().map(|b| b + (16 << 20));
        {
            let _open = begin_forward(&s);
            plan_wave_transient(&s, wide)?;
            // Finished: it laid its spans out, as every forward that ran has.
            drop(begin_wave(&s, LayerPhase::Forward)?);
        }
        drop(enter_arena_window(lease(), "a test tenant")?);
        let tier = |ordinal| {
            super::super::super::region_pool::region_stats(ordinal)
                .map(|r| r.transient_bytes)
                .unwrap_or(0)
        };
        assert_eq!(tier(ordinal), 0, "the window handed the tier back");
        // A guard with no forward behind it now prices itself, and places the
        // fallback rather than the finished forward's plan.
        // The pool places whole regions, so the fallback reads rounded up — and
        // still regions short of the wide plan, which is 48 MiB wider.
        let guard = begin_wave(&s, LayerPhase::Attention)?;
        let placed = tier(ordinal);
        let wide_bytes = tier_footprint(wide);
        assert!(
            placed >= fallback && placed < wide_bytes,
            "the guard placed {placed} B: the fallback is {fallback} B, the finished \
             forward's plan {wide_bytes} B"
        );
        drop(guard);
        Ok(())
    }
}
