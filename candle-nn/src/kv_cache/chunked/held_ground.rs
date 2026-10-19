//! The reservation's held ground: device buffers that outlive a forward but may
//! be asked for at any moment — inside one, or from another thread — where a span
//! region claim would be refused: a sampler's argument pack, decode slot-state
//! slabs, a cuBLAS workspace, a cold-load batch's staging.
//!
//! # Elastic, not fixed
//!
//! The ground under these buffers is two parts (`docs/held_ground_elimination.md`):
//!
//! - **A bootstrap block** at the very left of the span, [`BOOTSTRAP_HELD_BYTES`],
//!   for the takes made while a model loads. Regions cannot be claimed then — the
//!   dense block is still growing and every region is measured from its right
//!   edge — so the load's own buffers (a thread's cuBLAS workspace, the stager's
//!   device slab) need ground that exists before any region does.
//! - **Held arenas**: span regions claimed as a tenant, one region or a contiguous
//!   run of them per arena, sub-allocated by the same first-fit extents.
//!   [`SpanHeldGround::provision`] tops them up **between forwards** to a headroom
//!   learned from what recent forwards took, so a take inside a forward finds room
//!   already there; it releases arenas nothing holds once the headroom is covered
//!   without them. A take that finds no room goes to the CUDA pool, reported, and
//!   the next provisioning covers it — a take never claims, because it arrives
//!   under whatever locks its caller holds.
//!
//! What the fixed 1,280 MiB block this replaces held for the busiest daemon is now
//! held only while something is holding it.
//!
//! # Release is fenced
//!
//! A range given back is fenced on the stream it was released on and reused only
//! once the fence has fired, so a launch queued before the release still reads its
//! bytes. An arena goes back to the span only with no range live or pending.

#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::panic::Location;
#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "cuda")]
use candle::cuda_backend::cudarc::driver::{sys, CudaEvent, CudaStream};
#[cfg(feature = "cuda")]
use candle::cuda_backend::{ask_holders_to_move, HeldGround};
#[cfg(feature = "cuda")]
use candle::{CudaDevice, Device};

#[cfg(feature = "cuda")]
use super::held_extents::{Extents, Taken};
#[cfg(feature = "cuda")]
use super::held_ledger::{HeldLedger, HeldSite};
#[cfg(feature = "cuda")]
use super::region_pool::{regions_open, run_free_below, SpanClaims, SpanRegion};
use super::types::TARGET_ARENA_BYTES;

/// Holders listed beside a refusal: enough to name what filled the ground.
#[cfg(feature = "cuda")]
const REFUSAL_REPORT_SITES: usize = 8;

/// Bytes of the span kept as held ground before any region exists — the takes
/// a model load makes.
///
/// Sized from the load's own holders: the loading thread's cuBLAS workspace
/// (4 MiB, `thread_blas`), the expert stager's device slab (4 MiB,
/// `pinned_staging`), the launch-table and quantize staging the repack grows
/// (a few MiB), and the link probe's window — under 16 MiB on Flash-Next. Twice
/// that, so a model with a second GEMM thread at load does not spill. Everything
/// after the load is held arenas; this block also serves the runtime first, since
/// it is there anyway.
pub const BOOTSTRAP_HELD_BYTES: usize = 32 * 1024 * 1024;

/// Every range the ground hands out starts on this boundary.
pub(crate) const HELD_ALIGN: usize = 256;

/// Bytes in one held arena region — the span's region size.
const REGION: usize = TARGET_ARENA_BYTES;

/// Forwards whose held-ground growth the headroom remembers.
///
/// The same window the growth negotiation keeps for the tier
/// (`region_pool::RECENT_FORWARDS`): a decode step is several forwards, and
/// a slot-state promotion or a sampler pack growing lands in whichever of them
/// crosses the boundary.
#[cfg(feature = "cuda")]
const RECENT: usize = 16;

/// [`RECENT`], for a test that runs out a window.
#[cfg(all(test, feature = "cuda"))]
pub(crate) const RECENT_WINDOW: usize = RECENT;

/// Which ground a range was taken from.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ground {
    Bootstrap,
    /// The held arena whose first byte is at this address.
    Arena(u64),
}

/// One held arena: a region, or a run of consecutive regions, sub-allocated.
#[cfg(feature = "cuda")]
struct HeldArena {
    base: u64,
    extents: Extents,
    /// Dropped with the arena, which is what returns the ground to the span.
    regions: Vec<SpanRegion>,
}

#[cfg(feature = "cuda")]
struct Inner {
    bootstrap: Extents,
    /// Ascending by base, so first-fit takes the lowest arena first.
    arenas: Vec<HeldArena>,
    /// Bases of the arenas whose holders are asked to move
    /// ([`SpanHeldGround::vacate`]): live, with room claimed below them. Nothing new is taken from them, and their free
    /// ground is not counted as headroom.
    vacating: Vec<u64>,
    /// Ranges given back whose fence may not have fired.
    pending: Vec<(CudaEvent, Ground, Taken)>,
    /// The range behind each live address, and the site that holds it.
    live: HashMap<u64, (Ground, Taken, &'static Location<'static>)>,
    /// What each site holds and has been refused.
    ledger: HeldLedger,
    /// Bytes held when the last provisioning ended, the most held since, and what
    /// was refused since — together, what the forwards in between took.
    used_at_provision: usize,
    peak_since: usize,
    refused_since: usize,
    /// The widest single request refused since the last provisioning: an arena
    /// of at least this is claimed for it.
    widest_refused: usize,
    /// Growth per provisioning window, newest at `growth_at`.
    growth: [usize; RECENT],
    growth_at: usize,
    /// The most held at once since the process started.
    peak: usize,
}

#[cfg(feature = "cuda")]
impl Inner {
    fn used(&self) -> usize {
        self.bootstrap.used() + self.arenas.iter().map(|a| a.extents.used()).sum::<usize>()
    }

    /// Free ground a take can land in — not a vacating arena's.
    fn free(&self) -> usize {
        self.bootstrap.capacity() - self.bootstrap.used()
            + self
                .staying()
                .map(|a| a.extents.capacity() - a.extents.used())
                .sum::<usize>()
    }

    /// [`SpanHeldGround::provision`]'s claims, worked out as it works them out
    /// and changing nothing: the window since the last pass folded into the
    /// growth record, the headroom short of free ground in whole regions, the
    /// widest refused take's arena, and the landings of the live arenas.
    fn next_claim(&self) -> HeldClaim {
        let grew = self.peak_since.saturating_sub(self.used_at_provision) + self.refused_since;
        let mut growth = self.growth;
        growth[self.growth_at] = grew;
        let short = headroom_for(&growth).saturating_sub(self.free());
        let widest = if self.widest_refused > self.largest_free() {
            self.widest_refused.div_ceil(REGION).max(1) * REGION
        } else {
            0
        };
        let landing = self
            .arenas
            .iter()
            .filter(|a| a.extents.used() > 0 && !self.vacating.contains(&a.base))
            .map(|a| {
                let ranges: Vec<usize> = self
                    .live
                    .values()
                    .filter(|(g, _, _)| *g == Ground::Arena(a.base))
                    .map(|(_, t, _)| t.len)
                    .collect();
                landing_runs(&ranges).iter().sum::<usize>() * REGION
            })
            .sum();
        let per_window = growth.iter().copied().max().unwrap_or(0);
        HeldClaim {
            growth: widest + short.div_ceil(REGION) * REGION,
            landing,
            per_window: per_window.div_ceil(REGION) * REGION,
        }
    }

    /// Bytes standing in vacating arenas — the room their holders move into.
    fn vacating_bytes(&self) -> usize {
        self.arenas
            .iter()
            .filter(|a| self.vacating.contains(&a.base))
            .map(|a| a.extents.used())
            .sum()
    }

    /// The arenas a take may land in.
    fn staying(&self) -> impl Iterator<Item = &HeldArena> {
        self.arenas
            .iter()
            .filter(|a| !self.vacating.contains(&a.base))
    }

    fn capacity(&self) -> usize {
        self.bootstrap.capacity()
            + self
                .arenas
                .iter()
                .map(|a| a.extents.capacity())
                .sum::<usize>()
    }

    fn largest_free(&self) -> usize {
        self.staying()
            .map(|a| a.extents.largest_free())
            .fold(self.bootstrap.largest_free(), usize::max)
    }

    fn extents_mut(&mut self, ground: Ground) -> &mut Extents {
        match ground {
            Ground::Bootstrap => &mut self.bootstrap,
            Ground::Arena(base) => {
                &mut self
                    .arenas
                    .iter_mut()
                    .find(|a| a.base == base)
                    .expect("a range's arena stays while the range is live or pending")
                    .extents
            }
        }
    }

    /// Return every pending range whose fence has fired; with `wait`, the
    /// oldest one is waited for first, so at least one comes back.
    fn reclaim(&mut self, wait: bool) {
        if wait {
            if let Some((event, _, _)) = self.pending.first() {
                // A failed wait leaves the range pending rather than reusing
                // memory a launch may still read.
                let _ = event.synchronize();
            }
        }
        let mut i = 0;
        while i < self.pending.len() {
            // `cuEventQuery`: SUCCESS once every launch queued before the
            // record has completed; anything else leaves the range pending.
            let fired = unsafe { sys::cuEventQuery(self.pending[i].0.cu_event()) }
                == sys::CUresult::CUDA_SUCCESS;
            if fired {
                let (_, ground, t) = self.pending.swap_remove(i);
                self.extents_mut(ground).release(t.start, t.len);
            } else {
                i += 1;
            }
        }
    }

    /// `bytes` from the bootstrap block, else the lowest arena that holds them and
    /// is not being vacated.
    fn take_from(&mut self, bytes: usize) -> Option<(Ground, Taken, u64)> {
        if let Some(t) = self.bootstrap.take(bytes, HELD_ALIGN) {
            return Some((Ground::Bootstrap, t, 0));
        }
        let vacating = &self.vacating;
        self.arenas
            .iter_mut()
            .filter(|a| !vacating.contains(&a.base))
            .find_map(|a| {
                a.extents
                    .take(bytes, HELD_ALIGN)
                    .map(|t| (Ground::Arena(a.base), t, a.base))
            })
    }

    /// Whether `ground` has a range live or behind a fence.
    fn holds(&self, ground: Ground) -> bool {
        self.live.values().any(|(g, _, _)| *g == ground)
            || self.pending.iter().any(|(_, g, _)| *g == ground)
    }

    fn insert_arena(&mut self, regions: Vec<SpanRegion>) {
        let base = regions[0].base();
        let arena = HeldArena {
            base,
            extents: Extents::new(regions.len() * REGION),
            regions,
        };
        let at = self.arenas.partition_point(|a| a.base < base);
        self.arenas.insert(at, arena);
    }
}

/// The held ground of one device.
#[cfg(feature = "cuda")]
pub(crate) struct SpanHeldGround {
    bootstrap_base: u64,
    inner: Mutex<Inner>,
}

#[cfg(feature = "cuda")]
impl SpanHeldGround {
    /// Ground whose bootstrap block is `[base, base + bootstrap)`, set aside for it
    /// for the life of the process; its arenas are claimed on the device that
    /// provisions it.
    pub(crate) fn new(base: u64, bootstrap: usize) -> Arc<Self> {
        Arc::new(Self {
            bootstrap_base: base,
            inner: Mutex::new(Inner {
                bootstrap: Extents::new(bootstrap),
                arenas: Vec::new(),
                vacating: Vec::new(),
                pending: Vec::new(),
                live: HashMap::new(),
                ledger: HeldLedger::default(),
                used_at_provision: 0,
                peak_since: 0,
                refused_since: 0,
                widest_refused: 0,
                growth: [0; RECENT],
                growth_at: 0,
                peak: 0,
            }),
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// What the ground holds now — padding included, and a release counted as
    /// held until its fence fires.
    pub(crate) fn stats(&self) -> HeldStats {
        stats_of(&self.lock())
    }

    /// The highest region the held arenas stand on — whether held ground is what
    /// holds the arena frontier.
    pub(crate) fn top_rank(&self) -> Option<usize> {
        self.lock()
            .arenas
            .iter()
            .flat_map(|a| a.regions.iter().map(|r| r.index()))
            .max()
    }

    /// What [`Self::provision`] would claim if it ran now ([`HeldClaim`]).
    pub(crate) fn next_claim(&self) -> HeldClaim {
        self.lock().next_claim()
    }

    /// Every holder's account, the largest live holder first.
    pub(crate) fn sites(&self) -> Vec<HeldSite> {
        self.lock().ledger.sites()
    }

    /// Claim `bytes` worth of held arena — one region, or a run when `bytes` is
    /// wider than a region — or `None` when the span will not give it now: a
    /// model still loading, a forward owning the partition, no ground spare.
    fn claim_arena(device: &CudaDevice, bytes: usize) -> Option<Vec<SpanRegion>> {
        if !regions_open(&device.cuda_stream()) {
            return None;
        }
        let claims = SpanClaims::open(&Device::Cuda(device.clone()), "held ground").ok()?;
        let k = bytes.div_ceil(REGION).max(1);
        if k == 1 {
            claims.claim().ok().flatten().map(|r| vec![r])
        } else {
            claims.claim_run(k).ok().flatten()
        }
    }

    /// **Between forwards:** fit the held arenas to what recent forwards took.
    ///
    /// Folds the window since the last call into the growth record — the most held
    /// past where the last provisioning left it, plus everything refused — and keeps
    /// free ground of the widest recent growth plus one region, claiming arenas when
    /// short and releasing ones nothing holds when over. A request refused for being
    /// wider than any arena gets an arena that holds it.
    ///
    /// Refused claims — a forward open, the span full — leave the arenas as they
    /// are; the next call tries again.
    pub(crate) fn provision(&self, device: &CudaDevice) {
        {
            let mut inner = self.lock();
            inner.reclaim(false);
            let grew =
                inner.peak_since.saturating_sub(inner.used_at_provision) + inner.refused_since;
            let at = inner.growth_at;
            inner.growth[at] = grew;
            inner.growth_at = (at + 1) % RECENT;
            inner.refused_since = 0;
        }
        // Before the headroom is reckoned: a vacating arena's free ground is not
        // headroom.
        self.vacate(device);
        let (short, widest) = {
            let mut inner = self.lock();
            let headroom = headroom_for(&inner.growth);
            let widest = std::mem::take(&mut inner.widest_refused);
            let widest = (widest > inner.largest_free()).then_some(widest);
            (headroom.saturating_sub(inner.free()), widest)
        };
        if let Some(bytes) = widest {
            if let Some(regions) = Self::claim_arena(device, bytes) {
                self.lock().insert_arena(regions);
            }
        }
        let mut need = short;
        while need > 0 {
            let Some(regions) = Self::claim_arena(device, REGION) else {
                break;
            };
            self.lock().insert_arena(regions);
            need = need.saturating_sub(REGION);
        }
        // Released outside the lock: an arena's regions go back to the span
        // through the pool's lock.
        let released = {
            let mut inner = self.lock();
            let headroom = headroom_for(&inner.growth);
            let mut released = Vec::new();
            // Highest first, so the arenas that stay are the low ones. A vacated
            // arena goes whatever the headroom: none of it was counted there.
            // What still stands in vacating arenas keeps room of its own on top
            // of the headroom, so a holder that has not yet moved finds it.
            let mut i = inner.arenas.len();
            while i > 0 {
                i -= 1;
                let a = &inner.arenas[i];
                let base = a.base;
                let idle = a.extents.used() == 0 && !inner.holds(Ground::Arena(base));
                let vacated = inner.vacating.contains(&base);
                let spare = inner.free().saturating_sub(a.extents.capacity());
                if idle && (vacated || spare >= headroom + inner.vacating_bytes()) {
                    released.push(inner.arenas.remove(i));
                    inner.vacating.retain(|&b| b != base);
                }
            }
            inner.used_at_provision = inner.used();
            inner.peak_since = inner.used_at_provision;
            released
        };
        drop(released);
        self.lower(device);
    }

    /// Mark the arenas whose holders should move: something lives in them, and
    /// room for it is free below.
    ///
    /// [`Self::lower`] moves an arena nothing holds; one with a live range cannot
    /// move, because its holders have its addresses. A slot-state slab or a
    /// scratch lives for as long as its owner does, so under a long decode cohort
    /// such an arena never falls idle and stands as the frontier over the holes
    /// packing opened below it — measured on WhiteWorld: held ground at region 629
    /// over 165–194 free regions, the weight side and the wave tier shut out of
    /// all of it. Asked to move, a holder takes afresh at its next natural point
    /// (`Held::asked_to_move`), the take skips the vacating arena, and the arena
    /// falls idle and goes back.
    ///
    /// Each newly marked arena first gets runs claimed below it that hold what
    /// lives in it ([`landing_runs`]), all of them or none; one that cannot land
    /// is not marked, since a holder that found no room would go to the CUDA
    /// pool — outside the span, worse than standing high. An arena stays
    /// marked until it goes back: its room was claimed when it was marked, and
    /// once idle it is released whatever the headroom.
    fn vacate(&self, device: &CudaDevice) {
        let stream = device.cuda_stream();
        // Live arenas not yet marked, highest first: each with its first region's
        // index, its width, and the runs below that would hold what lives in it.
        let live: Vec<(u64, usize, usize, Vec<usize>)> = {
            let inner = self.lock();
            inner
                .arenas
                .iter()
                .rev()
                .filter(|a| a.extents.used() > 0 && !inner.vacating.contains(&a.base))
                .map(|a| {
                    let ranges: Vec<usize> = inner
                        .live
                        .values()
                        .filter(|(g, _, _)| *g == Ground::Arena(a.base))
                        .map(|(_, t, _)| t.len)
                        .collect();
                    (
                        a.base,
                        a.regions[0].index(),
                        a.regions.len(),
                        landing_runs(&ranges),
                    )
                })
                .collect()
        };
        // Probed outside the held lock: the probe takes the pool's. The widest
        // run is the one least likely to be free.
        let fresh: Vec<_> = live
            .into_iter()
            .filter(|(_, start, _, runs)| {
                let widest = runs.iter().copied().max().unwrap_or(1);
                run_free_below(&stream, widest, *start)
            })
            .collect();
        let mut marked = false;
        if !fresh.is_empty() && regions_open(&stream) {
            if let Ok(claims) = SpanClaims::open(&Device::Cuda(device.clone()), "held ground") {
                for (base, start, width, runs) in fresh {
                    // Every run or none: a partial landing would leave a holder
                    // with nowhere in the span to go.
                    let mut landed = Vec::new();
                    for &k in &runs {
                        match claims.claim_run_below(k, start) {
                            Ok(Some(lower)) => landed.push(lower),
                            _ => break,
                        }
                    }
                    if landed.len() < runs.len() {
                        drop(landed);
                        continue;
                    }
                    tracing::info!(
                        "held ground: vacating the arena at region {start} ({width} regions) \
                         onto runs of {} regions below it",
                        runs.iter()
                            .map(|k| format!("{k}"))
                            .collect::<Vec<_>>()
                            .join("+")
                    );
                    let mut inner = self.lock();
                    for lower in landed {
                        inner.insert_arena(lower);
                    }
                    inner.vacating.push(base);
                    marked = true;
                }
            }
        }
        if marked {
            ask_holders_to_move();
        }
    }

    /// Move every arena nothing holds onto a free run below it.
    ///
    /// An arena is claimed wherever the lowest free run stood when it was needed —
    /// mid-prefill, high in the span. Packing then lowers the KV beneath it, and an
    /// arena kept as headroom never falls idle long enough to go back, so it stood
    /// as the frontier with the free ground under it unreachable to the weight side:
    /// Flash-Next's gate held 99 free regions below a held arena at region 116 and
    /// the expert zone 1.1 GiB short of its maximum. An empty arena holds no
    /// address, so it moves by claiming the lower run and dropping itself.
    ///
    /// The claims window opens only once a lower run is known to be free: opening it
    /// hands back a standing tier, which no forward should re-place for nothing.
    fn lower(&self, device: &CudaDevice) {
        let stream = device.cuda_stream();
        // Highest first, each with its first region's index and its width.
        let idle: Vec<(u64, usize, usize)> = {
            let inner = self.lock();
            inner
                .arenas
                .iter()
                .rev()
                .filter(|a| a.extents.used() == 0 && !inner.holds(Ground::Arena(a.base)))
                .map(|a| (a.base, a.regions[0].index(), a.regions.len()))
                .collect()
        };
        // Probed outside the held lock: the probe takes the pool's.
        let movable: Vec<_> = idle
            .into_iter()
            .filter(|&(_, start, k)| run_free_below(&stream, k, start))
            .collect();
        if movable.is_empty() || !regions_open(&stream) {
            return;
        }
        let Ok(claims) = SpanClaims::open(&Device::Cuda(device.clone()), "held ground") else {
            return;
        };
        // Dropped outside the lock, as in `provision`.
        let mut moved_off = Vec::new();
        for (base, start, k) in movable {
            let Ok(Some(lower)) = claims.claim_run_below(k, start) else {
                continue;
            };
            let mut inner = self.lock();
            // A take may have landed in it since it was found idle; then the
            // lower run goes straight back.
            let still_idle = !inner.holds(Ground::Arena(base));
            match inner.arenas.iter().position(|a| a.base == base) {
                Some(i) if still_idle && inner.arenas[i].extents.used() == 0 => {
                    moved_off.push(inner.arenas.remove(i).regions);
                    inner.vacating.retain(|&b| b != base);
                    inner.insert_arena(lower);
                }
                _ => moved_off.push(lower),
            }
        }
        drop(moved_off);
    }
}

#[cfg(feature = "cuda")]
impl SpanHeldGround {
    /// **With nothing in flight:** forget the recent forwards' growth and fit the
    /// arenas to that — every arena nothing holds goes back past one region of
    /// headroom. The growth record predicts the next forward of a busy engine; an
    /// idle one has none coming, and its peak would otherwise stand until 16
    /// forwards had passed.
    pub(crate) fn settle(&self, device: &CudaDevice) {
        {
            let mut inner = self.lock();
            inner.growth = [0; RECENT];
            inner.refused_since = 0;
            inner.peak_since = inner.used();
            inner.used_at_provision = inner.peak_since;
        }
        self.provision(device);
    }
}

/// Free ground the held arenas keep between forwards: the widest growth any of
/// the recent windows saw, plus one region.
#[cfg(feature = "cuda")]
fn headroom_for(growth: &[usize]) -> usize {
    growth.iter().copied().max().unwrap_or(0) + REGION
}

/// The runs, in regions, that hold live ranges of `ranges` bytes: widest first,
/// each into the first planned run with room, as a take lands first-fit, else
/// into a new run of its own width.
///
/// Sized from what lives in an arena, not from the arena: one claimed for a
/// wide take that has since gone may hold a few slot-state slabs across seven
/// regions, and needing seven free in a row below it is what kept it from ever
/// moving.
#[cfg(feature = "cuda")]
fn landing_runs(ranges: &[usize]) -> Vec<usize> {
    let mut widest_first = ranges.to_vec();
    widest_first.sort_unstable_by(|a, b| b.cmp(a));
    let mut runs: Vec<Extents> = Vec::new();
    for len in widest_first {
        if runs.iter_mut().any(|r| r.take(len, HELD_ALIGN).is_some()) {
            continue;
        }
        let mut run = Extents::new(len.div_ceil(REGION).max(1) * REGION);
        run.take(len, HELD_ALIGN)
            .expect("a run of a range's own width holds it");
        runs.push(run);
    }
    runs.iter().map(|r| r.capacity() / REGION).collect()
}

#[cfg(feature = "cuda")]
fn stats_of(inner: &Inner) -> HeldStats {
    HeldStats {
        used: inner.used(),
        pending: inner.pending.iter().map(|(_, _, t)| t.len).sum(),
        peak: inner.peak,
        capacity: inner.capacity(),
        largest_free: inner.largest_free(),
        arenas: inner.arenas.len(),
        arena_bytes: inner.arenas.iter().map(|a| a.extents.capacity()).sum(),
        arena_top: inner
            .arenas
            .iter()
            .flat_map(|a| a.regions.iter().map(|r| r.index() + 1))
            .max()
            .unwrap_or(0),
    }
}

/// File `site`'s refused request of `bytes` and, the first time that site is
/// refused, report it with the holders that filled the ground.
///
/// The refused buffer goes to the CUDA pool, and on a full card from there to
/// the driver — which answers `CUDA_ERROR_OUT_OF_MEMORY` with no name attached.
/// This line is the name: who asked, for how much, what the ground looked like,
/// and who was holding it. Once per site, because a ground that is short stays
/// short until the next provisioning; the ledger keeps counting.
#[cfg(feature = "cuda")]
fn refuse(inner: &mut Inner, site: &'static Location<'static>, bytes: usize) {
    inner.refused_since += bytes;
    inner.widest_refused = inner.widest_refused.max(bytes);
    if !inner.ledger.decline(site, bytes) {
        return;
    }
    let s = stats_of(inner);
    let holders = inner
        .ledger
        .sites()
        .into_iter()
        .filter(|h| h.live_bytes > 0)
        .take(REFUSAL_REPORT_SITES)
        .map(|h| {
            format!(
                "{} {} KiB in {} (peak {} KiB)",
                h.site,
                h.live_bytes >> 10,
                h.live_count,
                h.peak_bytes >> 10
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    tracing::warn!(
        target: "candle_nn::kv_cache::held_ground",
        %site,
        bytes,
        used = s.used,
        pending = s.pending,
        capacity = s.capacity,
        largest_free = s.largest_free,
        arenas = s.arenas,
        holders = %holders,
        "held ground had no room for a buffer inside a forward; it goes to the CUDA pool, \
         outside the reservation, and the next provisioning covers it"
    );
}

/// A reading of one device's held ground, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HeldStats {
    pub used: usize,
    /// The part of `used` given back but still behind its fence.
    pub pending: usize,
    /// The most it has held at once since the process started.
    pub peak: usize,
    /// The bootstrap block and every held arena.
    pub capacity: usize,
    /// The widest buffer it could hand out now.
    pub largest_free: usize,
    /// Held arenas standing, and the span ground they hold.
    pub arenas: usize,
    pub arena_bytes: usize,
    /// One past the highest region a held arena stands on; 0 with none.
    pub arena_top: usize,
}

/// What the next provisioning pass will claim from the span, in whole regions'
/// bytes — ground a forward planned now cannot otherwise see coming.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HeldClaim {
    /// The headroom top-up and the arena for the widest refused take: claimed
    /// at the lowest free regions, wherever they stand.
    pub growth: usize,
    /// At most this much more, to land the live arenas not yet being vacated
    /// — claimed only below an arena with room under it, and given back once
    /// its holders have moved. An upper bound: an arena with no room below
    /// claims nothing.
    pub landing: usize,
    /// The most any one recent window grew held ground, in whole regions'
    /// bytes. Provisioning runs at every forward's entry, so a sweep of many
    /// windows meets a pass at each: past the first, which `growth` prices,
    /// each later window may claim up to this again.
    pub per_window: usize,
}

#[cfg(feature = "cuda")]
impl HeldGround for SpanHeldGround {
    fn take(&self, bytes: usize, site: &'static Location<'static>) -> Option<u64> {
        let mut inner = self.lock();
        inner.reclaim(false);
        let mut taken = inner.take_from(bytes);
        if taken.is_none() && !inner.pending.is_empty() {
            // Room may be waiting behind a fence: wait out the oldest and try
            // once more before reaching for more ground.
            inner.reclaim(true);
            taken = inner.take_from(bytes);
        }
        // **No claim from here.** A take arrives from arbitrary depth — a slot-state
        // slab is taken under the sequence-state lock, a staging scratch under the
        // device's — and a region claim sweeps empty arenas and may buy ground from
        // the weight side, both of which take locks of their own. Any of them can
        // close a cycle with whatever the caller holds. Only `provision`, called
        // between forwards from a known point, claims; a take it did not cover
        // goes to the pool once and is covered from then on.
        let Some((ground, taken, base)) = taken else {
            refuse(&mut inner, site, bytes);
            return None;
        };
        inner.ledger.take(site, taken.len);
        let ptr = match ground {
            Ground::Bootstrap => self.bootstrap_base + taken.at as u64,
            Ground::Arena(_) => base + taken.at as u64,
        };
        inner.live.insert(ptr, (ground, taken, site));
        let used = inner.used();
        inner.peak = inner.peak.max(used);
        inner.peak_since = inner.peak_since.max(used);
        Some(ptr)
    }

    fn vacating(&self, ptr: u64) -> bool {
        let inner = self.lock();
        match inner.live.get(&ptr) {
            Some((Ground::Arena(base), _, _)) => inner.vacating.contains(base),
            _ => false,
        }
    }

    fn give_back(&self, ptr: u64, stream: &Arc<CudaStream>) {
        let mut inner = self.lock();
        let Some((ground, taken, site)) = inner.live.remove(&ptr) else {
            panic!("held ground: {ptr:#x} given back but never taken");
        };
        inner.ledger.release(site, taken.len);
        match stream.record_event(None) {
            Ok(event) => inner.pending.push((event, ground, taken)),
            // No fence to wait behind: wait for the stream itself rather than
            // reuse bytes a queued launch may still read.
            Err(_) => {
                let _ = stream.synchronize();
                inner.extents_mut(ground).release(taken.start, taken.len);
            }
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    /// The headroom is the widest recent growth plus one region: a window that
    /// grew nothing still keeps a region in hand.
    #[test]
    fn the_headroom_is_the_widest_recent_growth_and_a_region() {
        assert_eq!(headroom_for(&[0; RECENT]), REGION);
        let mut g = [0; RECENT];
        g[3] = 5 * REGION;
        g[7] = 2 * REGION;
        assert_eq!(headroom_for(&g), 6 * REGION);
    }

    /// **The next claim is the headroom short of free ground and the widest
    /// refused take's arena, in whole regions.** On a ground of bootstrap only,
    /// with no device: a quiet one claims nothing; a take of a region and a half
    /// leaves a half free against two and a half of headroom, so two regions; a
    /// refused take of three regions adds its own three-region arena and grows
    /// the headroom by three more. The window's growth, rounded to regions, is
    /// what each later window of a sweep may claim again.
    #[test]
    fn the_next_claim_is_what_provisioning_takes() {
        let held = SpanHeldGround::new(0x7f00_0000_0000, 2 * REGION);
        let site = Location::caller();
        assert_eq!(held.next_claim(), HeldClaim::default(), "a quiet ground");

        held.take(REGION + REGION / 2, site)
            .expect("bootstrap room");
        assert_eq!(held.next_claim().growth, 2 * REGION);

        assert_eq!(held.take(3 * REGION, site), None, "wider than any ground");
        // Grew 4.5 regions → headroom 5.5, free 0.5: five regions short, and the
        // refused take's own arena of three.
        assert_eq!(held.next_claim().growth, (5 + 3) * REGION);
        assert_eq!(held.next_claim().landing, 0, "no arena lives yet");
        assert_eq!(
            held.next_claim().per_window,
            5 * REGION,
            "the window grew four and a half regions"
        );
    }

    /// **The landing is sized by what lives in an arena, not by the arena.** A
    /// range and a half takes a run of two with half a region left, which the
    /// two quarter ranges fill exactly; the whole range gets a run of its own.
    #[test]
    fn landing_runs_fit_the_live_ranges() {
        assert_eq!(
            landing_runs(&[REGION / 4, REGION, REGION / 4, REGION + REGION / 2]),
            vec![2, 1]
        );
        assert_eq!(
            landing_runs(&[100]),
            vec![1],
            "a sliver still takes a region"
        );
        assert!(
            landing_runs(&[]).is_empty(),
            "nothing lives there, nothing to land"
        );
    }
}
