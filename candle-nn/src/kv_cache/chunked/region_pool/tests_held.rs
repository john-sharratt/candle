//! Tests for held ground: it grows to what a take needs and returns when idle.
//!
//! A take wider than anything held is refused once, and the next provisioning
//! holds an arena wide enough, so the same take then succeeds. Once nothing
//! holds that arena, a window of quiet provisionings gives it back. An arena
//! standing high moves down onto freed ground rather than remaining the
//! frontier. A take inside a recording forward must not read a fence, because
//! the driver would invalidate the capture. These tests use their own held
//! ground, since the device's is shared with other tests.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::{claim_region, REGION_BYTES};

    use candle::{Device, Result};

    use super::{SpanHeldGround, BOOTSTRAP_HELD_BYTES};

    use candle::cuda_backend::{vacate_epoch, HeldGround};

    use std::panic::Location;

    use std::sync::Arc;

    /// The pool is process-global and cargo runs tests in parallel, so region
    /// counts are only stable while one test at a time is looking at them — and
    /// that includes tests outside this module, which is why it is the
    /// crate-wide lock and not a local `static`.
    use super::super::super::gpu_test_lock::gpu_serial as serial;

    /// A held ground of the tests' own, with a bootstrap block like the device's:
    /// the device's ground is the process's, and every CUDA test running beside
    /// these takes from it, so its arenas, headroom and growth are theirs as much
    /// as this test's. Only the span regions its arenas claim are shared. The
    /// bootstrap base is never dereferenced — nothing here touches the memory.
    pub(in super::super) fn own_held_ground() -> Arc<SpanHeldGround> {
        SpanHeldGround::new(0x7f00_0000_0000, BOOTSTRAP_HELD_BYTES)
    }

    /// **The held ground grows to what was refused and gives it back when idle.**
    /// A take wider than anything held is refused, once; the next provisioning
    /// claims an arena that holds it, so the same take is held ground; and once
    /// nothing holds the arena, a window of quiet provisionings returns it to the
    /// span, keeping only the region in hand.
    #[test]
    fn held_arenas_cover_a_refused_take_and_go_back_when_idle() -> Result<()> {
        let _serial = serial();
        let Ok(Device::Cuda(dev)) = Device::new_cuda(0) else {
            return Ok(());
        };
        let s = dev.cuda_stream();
        let held = own_held_ground();
        let site = Location::caller();
        held.provision(&dev);
        let wide = 40 << 20;

        assert_eq!(
            held.take(wide, site),
            None,
            "nothing held is wide enough yet"
        );

        held.provision(&dev);
        let grown = held.stats();
        assert!(
            grown.arena_bytes >= wide,
            "an arena of at least the refused width: {grown:?}"
        );
        let second = held
            .take(wide, site)
            .expect("the same take is held ground now");
        held.give_back(second, &s);
        s.synchronize().map_err(candle::Error::wrap)?;

        for _ in 0..=super::super::super::held_ground::RECENT_WINDOW {
            held.provision(&dev);
        }
        assert_idle(&held);
        Ok(())
    }

    /// **A take inside a recording forward keeps the forward's capture.** A
    /// take reads the fences of the ranges given back before it, and the
    /// driver refuses a fence read on a thread that is capturing and
    /// invalidates the capture: on BlackWorld the first take of a decode after
    /// a give-back lost the wave at every boot. Given back, then taken twice
    /// inside a wave capture — once with room, once past the room so the take
    /// waits out the oldest fence — the segment still records and ends.
    #[test]
    fn a_take_inside_a_capture_keeps_it() -> Result<()> {
        let _serial = serial();
        let Ok(Device::Cuda(dev)) = Device::new_cuda(0) else {
            return Ok(());
        };
        let s = dev.cuda_stream();
        let held = own_held_ground();
        let site = Location::caller();
        held.provision(&dev);
        let room = held.stats().largest_free;
        let given = held.take(room, site).expect("the ground's widest range");
        held.give_back(given, &s);

        let capture = dev.begin_wave_capture()?;
        dev.record_launches()?;
        let small = held.take(64, site);
        let past = held.take(room, site);
        assert!(
            dev.recording_segment().is_some(),
            "the segment still records after the takes"
        );
        capture.finish()?;
        for ptr in [small, past].into_iter().flatten() {
            held.give_back(ptr, &s);
        }
        s.synchronize().map_err(candle::Error::wrap)?;
        Ok(())
    }

    /// Once a window of quiet provisionings has passed, every idle arena has
    /// gone back to the span but the one keeping the region in hand: the
    /// headroom is arena ground, never the bootstrap's slack.
    pub(in super::super) fn assert_idle(held: &SpanHeldGround) {
        let idle = held.stats();
        assert_eq!(idle.arenas, 1, "only the region in hand stays: {idle:?}");
        assert!(idle.arena_bytes >= REGION_BYTES, "{idle:?}");
    }

    /// **An idle held arena moves down onto ground freed beneath it.** Claimed
    /// while the low ground was taken, it stands high; once that ground is free,
    /// the next provisioning moves it there, so it is not left as the frontier.
    #[test]
    fn an_idle_held_arena_moves_down_onto_freed_ground() -> Result<()> {
        let _serial = serial();
        let Ok(Device::Cuda(dev)) = Device::new_cuda(0) else {
            return Ok(());
        };
        let s = dev.cuda_stream();
        let held = own_held_ground();
        held.provision(&dev);
        // The low ground, taken, so the arena is claimed above it.
        let mut low = Vec::new();
        for _ in 0..8 {
            low.extend(claim_region(&s)?);
        }
        let wide = 72 << 20;
        assert_eq!(
            held.take(wide, Location::caller()),
            None,
            "refused, so it only records the width"
        );
        held.provision(&dev);
        let high = held.stats().arena_top;
        assert!(
            held.stats().arena_bytes >= wide,
            "an arena of the refused width"
        );

        drop(low);
        s.synchronize().map_err(candle::Error::wrap)?;
        held.provision(&dev);
        let lowered = held.stats().arena_top;
        assert!(
            lowered < high,
            "the idle arena moved down: top {high} → {lowered}"
        );

        for _ in 0..=super::super::super::held_ground::RECENT_WINDOW {
            held.provision(&dev);
        }
        assert_idle(&held);
        Ok(())
    }

    /// **A live held arena is vacated onto ground freed beneath it.** Its holder
    /// has its addresses, so it cannot be moved for it: the next provisioning
    /// claims a run of its width below, asks its holders to move, and takes
    /// nothing more from it. The holder's fresh take lands below, and once the
    /// old range is let go the arena goes back — it is no longer the frontier.
    #[test]
    fn a_live_held_arena_is_vacated_onto_freed_ground() -> Result<()> {
        let _serial = serial();
        let Ok(Device::Cuda(dev)) = Device::new_cuda(0) else {
            return Ok(());
        };
        let s = dev.cuda_stream();
        let held = own_held_ground();
        let site = Location::caller();
        held.provision(&dev);
        // The low ground, taken, so the arena is claimed above it.
        let mut low = Vec::new();
        for _ in 0..8 {
            low.extend(claim_region(&s)?);
        }
        let wide = 40 << 20;
        assert_eq!(
            held.take(wide, site),
            None,
            "refused, so it only records the width"
        );
        held.provision(&dev);
        let old = held.take(wide, site).expect("held ground now");
        let high = held.stats().arena_top;
        assert!(!held.vacating(old), "nothing below it is free yet");

        drop(low);
        s.synchronize().map_err(candle::Error::wrap)?;
        let epoch = vacate_epoch();
        held.provision(&dev);
        assert!(
            held.vacating(old),
            "a live arena over freed ground is vacated"
        );
        assert!(vacate_epoch() > epoch, "its holders were asked to move");
        let moved = held.take(wide, site).expect("room was claimed below it");
        assert!(
            moved < old,
            "the fresh take landed below: {moved:#x} against {old:#x}"
        );
        assert!(!held.vacating(moved), "the new range stays");

        held.give_back(old, &s);
        s.synchronize().map_err(candle::Error::wrap)?;
        held.provision(&dev);
        let lowered = held.stats().arena_top;
        assert!(
            lowered < high,
            "the vacated arena went back: top {high} → {lowered}"
        );

        held.give_back(moved, &s);
        s.synchronize().map_err(candle::Error::wrap)?;
        for _ in 0..=super::super::super::held_ground::RECENT_WINDOW {
            held.provision(&dev);
        }
        assert_idle(&held);
        Ok(())
    }
}
