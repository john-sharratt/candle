//! The balloon-and-measure bootstrap: claim everything except a fixed reserve
//! (touching it to force residency, which evicts other processes' cold
//! allocations on WDDM), record the resident high-water as the capacity `C`,
//! then free it. See `docs/elastic_vram_partition.md` §5.

use super::budget::GovernorConfig;
use super::reading::{ProbeKind, VramProbe, VramReading};
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

/// The capacity target: everything on the card except a fixed absolute reserve.
///
/// One expression, used by the growth loop **and** by the fast path that skips
/// the balloon entirely, so the two cannot disagree about what "as much as
/// possible" means. It was two — a fraction of total for the growth loop and
/// nothing at all on the fast path — and the fast path is the one that runs on
/// an uncontended card, so in practice the reserve was not being applied.
///
/// A fraction of the card is not a fact about anything: it says "leave 5%",
/// which on a 16 GiB card means 818 MiB and on a 96 GiB card means 4.9 GiB, for
/// no reason that scales. What has to be left is a fixed working margin for the
/// display driver and the OS, and that is an absolute quantity.
///
/// This is the CEILING of a claim, not its residency bound — the growth loop
/// additionally stops when the probe's live headroom falls to the reserve
/// (see [`balloon_measure`]), which on WDDM is what actually limits `C`.
pub fn capacity_target(total: u64, reserve: u64) -> u64 {
    total.saturating_sub(reserve)
}

/// The residency wobble margin for a reading.
///
/// On a WDDM budget reading ([`ProbeKind::Dxgi`]) the startup budget is a
/// dynamic target, not a floor — desktop activity dips it by gigabytes at
/// runtime and WDDM demotes rather than re-promotes, so a capacity set right
/// under the startup budget still thrashes (measured: budget − 512 MiB scored
/// 89.9 t/s on the widest config and hard-OOMed the following run; ~4 GB
/// under budget ran stable best-ever). A sixteenth of the budget, floored at
/// the absolute reserve, scales that slack with the card: ~4.4 GiB on the
/// 73 GiB dev card, under a GiB on a 16 GiB one — where a fixed 4 GiB would
/// cost a quarter of the card.
///
/// Everywhere else the refusal mechanism is honest and the margin is just the
/// configured reserve.
pub fn wobble_margin(reading: &VramReading, config: &GovernorConfig) -> u64 {
    match reading.source {
        ProbeKind::Dxgi => (reading.headroom / 16).max(config.capacity_reserve),
        _ => config.capacity_reserve,
    }
}

/// Grow the balloon until it reaches [`capacity_target`], or until the driver
/// refuses a chunk smaller than [`GovernorConfig::balloon_min_chunk`] — whichever
/// comes first. Returns the resident high-water we claimed (`C`), having freed
/// the balloon.
///
/// # Refusal refines the chunk instead of ending the claim
///
/// The loop grows in `balloon_chunk` steps (256 MiB). A refusal at that size
/// says nothing about 128 MiB, so ending there leaves up to a whole chunk of
/// claimable memory unmeasured — and `C` is what every later partition is sized
/// from, so an under-measurement is permanent. Halving on refusal and continuing
/// walks the claim down to within `balloon_min_chunk` of the true ceiling, at a
/// cost of at most `log2(chunk / min_chunk)` extra failed allocations (three, at
/// the shipped values).
///
/// The floor is the reservation's granule size: below it a claim cannot be
/// expressed in the allocator the capacity is being measured *for*, so refining
/// further would measure memory that could never be mapped.
///
/// The chunk only ever shrinks, so the loop provably terminates and pays at most
/// `log2(chunk / min_chunk)` failed allocations near the ceiling. A refusal far
/// from the ceiling — a transient squatter rather than the limit — leaves the
/// rest of the claim to be made in `min_chunk` steps, bounded at
/// `target / min_chunk` iterations. Restoring the chunk on success would avoid
/// that, at the cost of re-failing at every doubling once the ceiling is close;
/// the bound is a fraction of a second and the failure path is the one that
/// matters, so it stays monotone.
///
/// # The live-headroom stop: WDDM's refusal never comes
///
/// On WDDM, `cuMemAlloc` + memset succeed PAST the OS's residency budget —
/// the memory manager silently demotes pages to system RAM instead of
/// refusing, so a loop that waits for a refusal measures COMMIT, not
/// residency. Measured on the 73,045 MiB RTX PRO 5000 (per-process DXGI
/// budget 71,977 MiB): the refusal-only balloon claimed 72,574 MiB with zero
/// refusals; the widest sweep config then spent that `C`, WDDM demoted 2–5 GB
/// of live pages, and identical runs scored 141↔900 t/s depending on WHICH
/// pages the OS chose (page-fault signature: big-buffer spans inflated
/// 6–20×, small-buffer spans flat; the `\GPU Adapter Memory` counter trace
/// put demotion onset within ~1 GiB of the budget).
///
/// So the loop ALSO stops when the probe's live headroom falls to the wobble
/// margin below. On the DXGI probe headroom is `Budget − CurrentUsage` — the
/// OS's residency promise — and it is read fresh each chunk because the
/// budget GROWS as the balloon's touches demote other processes' cold pages:
/// a startup snapshot would under-measure a contended card exactly where the
/// balloon matters most. On the plain CUDA probe headroom is free device
/// memory and the refusal arrives as before.
///
/// # The wobble margin: the budget itself over-promises
///
/// The startup budget is a dynamic target, not a floor — desktop activity
/// (DWM composition, a browser paint) dips it by gigabytes at runtime, and
/// WDDM demotes rather than re-promotes, so a `C` set right under the
/// startup budget still thrashes. Measured on the same card: `C` at budget
/// − 512 MiB (74.9 GB) scored 89.9 t/s on the widest config and hard-OOMed
/// the following run; `C` ~4 GB under budget ran {880, 889, 867, 888} —
/// stable best-ever. The margin scales with the budget (1/16th, floored at
/// the absolute reserve) because a fixed number cannot serve both a 73 GiB
/// card (needs ~4 GiB) and a 16 GiB one (where 4 GiB is a quarter of the
/// card): the budget the OS grants and the amount it later claws back both
/// grow with the card its co-tenants render against.
pub fn balloon_measure(
    probe: &dyn VramProbe,
    alloc: &mut dyn BalloonAllocator,
    config: &GovernorConfig,
) -> Result<u64> {
    let first = probe.read()?;
    let target = capacity_target(first.total, config.capacity_reserve);
    let margin = wobble_margin(&first, config);
    let min_chunk = config.balloon_min_chunk.max(1);
    let mut chunk = config.balloon_chunk.max(min_chunk);
    let mut reserved = 0u64;
    while reserved < target {
        // Live residency bound: how much the OS will still keep resident for
        // us beyond what the balloon already holds, minus the wobble margin.
        let headroom = probe.read()?.headroom;
        let room = headroom.saturating_sub(margin);
        if room == 0 {
            break;
        }
        let want = chunk.min(target - reserved).min(room);
        if want == 0 {
            break;
        }
        match alloc.alloc_and_touch(want) {
            Ok(()) => reserved = reserved.saturating_add(want),
            Err(_) => {
                // Not the ceiling — the ceiling *for this chunk size*. Halve and
                // ask again; stop only when even the smallest useful claim is
                // refused.
                if chunk <= min_chunk {
                    break;
                }
                chunk = (chunk / 2).max(min_chunk);
            }
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
