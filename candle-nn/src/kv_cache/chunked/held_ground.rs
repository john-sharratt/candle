//! The reservation's held ground: a fixed block at the far left of the span for
//! the device buffers that outlive every forward — launch-table scratch, a
//! sampler's argument pack, resident slot headers, attention split partials.
//!
//! It sits beside the persistence staging block and for the same reason: its
//! address never moves, so neither the KV frontier nor the weight floor ever
//! has to reason about it, and a buffer can be taken from it at any moment —
//! inside a forward as readily as between two, where a region claim would be
//! refused. That is what lets these buffers leave the CUDA pool.
//!
//! Ranges are managed by first-fit extents ([`super::held_extents`]). A range
//! given back is fenced on the stream it was released on and reused only once
//! the fence has fired, so a launch queued before the release still reads its
//! bytes.

#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::panic::Location;
#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "cuda")]
use candle::cuda_backend::cudarc::driver::{sys, CudaEvent, CudaStream};
#[cfg(feature = "cuda")]
use candle::cuda_backend::HeldGround;

#[cfg(feature = "cuda")]
use super::held_extents::{Extents, Taken};
#[cfg(feature = "cuda")]
use super::held_ledger::{HeldLedger, HeldSite};

/// Holders listed beside a refusal: enough to name what filled the ground.
#[cfg(feature = "cuda")]
const REFUSAL_REPORT_SITES: usize = 8;

/// Bytes of the span kept as held ground for the buffers every model holds. No
/// model widens it: a window's residual lives on the step's forward span or a
/// creep's arena (`docs/wave_step.md`).
///
/// Sized from what the serving loop holds this way on Flash-Next under a
/// conversation and an ingest (`.5`, 2026-10-17): the launchers' kernel scratch
/// 436 MiB at its peak (INT8 prefill and decode split partials, the provenance
/// scans' accumulators), decode slot-state slabs 304 MiB, the penalty counts
/// 76 MiB, the gallery scan 68 MiB, launch tables 23 MiB, and a cold-load
/// batch's staging coming and going at up to 64 MiB — ~970 MiB, against the
/// 512 MiB this was. Every byte past it was refused into the CUDA pool, outside
/// the reservation, where it pushed the card past WDDM's residency budget.
///
/// The margin over the measurement is for placement: ranges are first-fit
/// ([`super::held_extents`]), so a ground near full refuses a request its free
/// bytes would cover. A request the ground cannot hold is still served from the
/// pool and reported, so the figure is a budget to measure against
/// ([`super::region_pool::held_ground_stats`]'s `peak`), not a wall.
pub const HELD_GROUND_BYTES: usize = 1280 * 1024 * 1024;

/// Every range the ground hands out starts on this boundary.
pub(crate) const HELD_ALIGN: usize = 256;

#[cfg(feature = "cuda")]
struct Inner {
    extents: Extents,
    /// Ranges given back whose fence may not have fired.
    pending: Vec<(CudaEvent, Taken)>,
    /// The range behind each live address, and the site that holds it.
    live: HashMap<u64, (Taken, &'static Location<'static>)>,
    /// What each site holds and has been refused.
    ledger: HeldLedger,
}

#[cfg(feature = "cuda")]
impl Inner {
    /// Return every pending range whose fence has fired; with `wait`, the
    /// oldest one is waited for first, so at least one comes back.
    fn reclaim(&mut self, wait: bool) {
        if wait {
            if let Some((event, _)) = self.pending.first() {
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
                let (_, t) = self.pending.swap_remove(i);
                self.extents.release(t.start, t.len);
            } else {
                i += 1;
            }
        }
    }
}

/// The held ground of one device.
#[cfg(feature = "cuda")]
pub(crate) struct SpanHeldGround {
    base: u64,
    inner: Mutex<Inner>,
}

#[cfg(feature = "cuda")]
impl SpanHeldGround {
    /// Ground over `[base, base + capacity)`, which the caller has set aside
    /// for it for the life of the process.
    pub(crate) fn new(base: u64, capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            base,
            inner: Mutex::new(Inner {
                extents: Extents::new(capacity),
                pending: Vec::new(),
                live: HashMap::new(),
                ledger: HeldLedger::default(),
            }),
        })
    }

    /// What the ground holds now — padding included, and a release counted as
    /// held until its fence fires.
    pub(crate) fn stats(&self) -> HeldStats {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        stats_of(&inner)
    }

    /// Every holder's account, the largest live holder first.
    pub(crate) fn sites(&self) -> Vec<HeldSite> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .ledger
            .sites()
    }
}

#[cfg(feature = "cuda")]
fn stats_of(inner: &Inner) -> HeldStats {
    HeldStats {
        used: inner.extents.used(),
        pending: inner.pending.iter().map(|(_, t)| t.len).sum(),
        peak: inner.extents.peak(),
        capacity: inner.extents.capacity(),
        largest_free: inner.extents.largest_free(),
    }
}

/// File `site`'s refused request of `bytes` and, the first time that site is
/// refused, report it with the holders that filled the ground.
///
/// The refused buffer goes to the CUDA pool, and on a full card from there to
/// the driver — which answers `CUDA_ERROR_OUT_OF_MEMORY` with no name attached.
/// This line is the name: who asked, for how much, what the ground looked like,
/// and who was holding it. Once per site, because a ground that is full stays
/// full and refuses the same holder every wave; the ledger keeps counting.
#[cfg(feature = "cuda")]
fn refuse(inner: &mut Inner, site: &'static Location<'static>, bytes: usize) {
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
        peak = s.peak,
        capacity = s.capacity,
        largest_free = s.largest_free,
        holders = %holders,
        "held ground refused a buffer; it goes to the CUDA pool, outside the reservation"
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
    pub capacity: usize,
    /// The widest buffer it could hand out now.
    pub largest_free: usize,
}

#[cfg(feature = "cuda")]
impl HeldGround for SpanHeldGround {
    fn take(&self, bytes: usize, site: &'static Location<'static>) -> Option<u64> {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.reclaim(false);
        let mut taken = inner.extents.take(bytes, HELD_ALIGN);
        if taken.is_none() && !inner.pending.is_empty() {
            // Room may be waiting behind a fence: wait out the oldest and try
            // once more before sending the request to the pool.
            inner.reclaim(true);
            taken = inner.extents.take(bytes, HELD_ALIGN);
        }
        let Some(taken) = taken else {
            refuse(&mut inner, site, bytes);
            return None;
        };
        inner.ledger.take(site, taken.len);
        let ptr = self.base + taken.at as u64;
        inner.live.insert(ptr, (taken, site));
        Some(ptr)
    }

    fn give_back(&self, ptr: u64, stream: &Arc<CudaStream>) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let Some((taken, site)) = inner.live.remove(&ptr) else {
            panic!("held ground: {ptr:#x} given back but never taken");
        };
        inner.ledger.release(site, taken.len);
        match stream.record_event(None) {
            Ok(event) => inner.pending.push((event, taken)),
            // No fence to wait behind: wait for the stream itself rather than
            // reuse bytes a queued launch may still read.
            Err(_) => {
                let _ = stream.synchronize();
                inner.extents.release(taken.start, taken.len);
            }
        }
    }
}
