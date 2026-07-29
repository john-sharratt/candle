//! The balloon-and-measure bootstrap: claim ~90% of VRAM (touching it to force
//! residency, which evicts other processes' cold allocations on WDDM), record
//! the resident high-water as the capacity `C`, then free it. See
//! `docs/vram_governor_design.md` §5.

use super::budget::GovernorConfig;
use super::reading::VramProbe;
use crate::{DType, Device, Result, Tensor};

/// A source of touched device memory the balloon can claim and release. Abstracted
/// so the balloon *logic* is unit-testable without a GPU.
pub trait BalloonAllocator {
    /// Allocate `bytes` of device memory and touch every part of it (force
    /// residency). Hold it until [`Self::free_all`]. `Err` on allocation failure
    /// (the hard ceiling).
    fn alloc_and_touch(&mut self, bytes: u64) -> Result<()>;
    /// Release everything claimed so far.
    fn free_all(&mut self);
}

/// Grow the balloon until it reaches the capacity target `C`, the probe's
/// headroom hits `balloon_floor`, or an allocation fails — whichever comes first.
/// Returns the resident high-water we claimed (`C`), having freed the balloon.
///
/// The target combines a fractional and an absolute reserve as
/// `C = min(balloon_target_frac × total, total − balloon_headroom_abs)`: the
/// fraction governs large cards and the absolute headroom protects small ones
/// (see [`GovernorConfig::balloon_headroom_abs`]). `balloon_floor` remains a
/// separate, deeper safety net for the growth loop itself.
pub fn balloon_measure(
    probe: &dyn VramProbe,
    alloc: &mut dyn BalloonAllocator,
    config: &GovernorConfig,
) -> Result<u64> {
    let total = probe.read()?.total;
    let frac_target = (config.balloon_target_frac * total as f64) as u64;
    let abs_target = total.saturating_sub(config.balloon_headroom_abs);
    let target = frac_target.min(abs_target);
    let chunk = config.balloon_chunk.max(1);
    let mut reserved = 0u64;
    while reserved < target {
        let headroom = probe.read()?.headroom;
        if headroom <= config.balloon_floor {
            break;
        }
        let want = chunk.min(target - reserved);
        if want == 0 {
            break;
        }
        match alloc.alloc_and_touch(want) {
            Ok(()) => reserved = reserved.saturating_add(want),
            Err(_) => break, // hard ceiling reached
        }
    }
    alloc.free_all();
    Ok(reserved)
}

/// A [`BalloonAllocator`] backed by real device memory. `Tensor::zeros` allocates
/// **and** zero-fills (a full memset), which touches every page and forces
/// residency — the mechanism that evicts colder tenants on WDDM.
pub struct DeviceBalloonAllocator {
    device: Device,
    held: Vec<Tensor>,
}

impl DeviceBalloonAllocator {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            held: Vec::new(),
        }
    }
}

impl BalloonAllocator for DeviceBalloonAllocator {
    fn alloc_and_touch(&mut self, bytes: u64) -> Result<()> {
        // U8 tensor of `bytes` elements: alloc + zero-fill = touch every page.
        let t = Tensor::zeros(bytes as usize, DType::U8, &self.device)?;
        self.held.push(t);
        Ok(())
    }
    fn free_all(&mut self) {
        // Drop → cuMemFreeAsync back to the pool, then retire the frees and
        // trim the pool so the ballooned bytes actually return to the OS.
        // Without the sync+trim the async pool retains them and the post-balloon
        // measurement would read them as still-used.
        self.held.clear();
        let _ = self.device.synchronize();
        #[cfg(feature = "cuda")]
        if let Device::Cuda(d) = &self.device {
            let _ = d.trim_pool(0);
        }
    }
}

/// A scripted balloon allocator for unit tests: consumes/releases a shared
/// [`FakeVram`](super::reading::fake::FakeVram) so the probe's headroom moves the
/// way real residency claims would move it, and fails once a scripted ceiling is
/// hit.
#[cfg(any(test, feature = "vram-test-util"))]
pub struct FakeBalloonAllocator {
    vram: super::reading::fake::FakeVram,
    claimed: u64,
    /// Fail `alloc_and_touch` once cumulative claimed would exceed this.
    ceiling: u64,
}

#[cfg(any(test, feature = "vram-test-util"))]
impl FakeBalloonAllocator {
    pub fn new(vram: super::reading::fake::FakeVram, ceiling: u64) -> Self {
        Self {
            vram,
            claimed: 0,
            ceiling,
        }
    }
}

#[cfg(any(test, feature = "vram-test-util"))]
impl BalloonAllocator for FakeBalloonAllocator {
    fn alloc_and_touch(&mut self, bytes: u64) -> Result<()> {
        if self.claimed.saturating_add(bytes) > self.ceiling {
            return Err(crate::Error::Msg("fake balloon: out of memory".into()));
        }
        self.claimed += bytes;
        self.vram.consume(bytes);
        Ok(())
    }
    fn free_all(&mut self) {
        self.vram.release(self.claimed);
        self.claimed = 0;
    }
}
