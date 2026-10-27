//! The elastic weight/KV boundary, moved by the pipeline thread.
//!
//! It runs on the pipeline thread because that thread owns the cache. *When* it
//! may run is a stricter condition: no wave generation open on the span, and
//! every invocation the forward thread has begun already served — checked under
//! the pass lock, which is held across the move so no invocation can begin
//! under it. Ground then changes hands only behind a device-wide quiesce; a
//! worker waiting on a cold expert waits only on the stager, which needs no
//! driver, so the quiesce always ends.

use super::cache::ZoneFloor;
use super::pipeline::PipelineState;
use super::slot_image::{build_slot_view, slot_offsets};
use super::types::BoundaryAsk;
use candle::cuda_backend::graph::{try_without_recording, Quiet};
use candle::{Device, Result};
use candle_nn::kv_cache::{
    kv_spare_regions, set_weight_floor, wave_is_live, weight_floor_after, REGION_BYTES,
};
use cudarc::driver::CudaStream;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Regions the weight side leaves above the KV side's recent high-water mark.
///
/// Headroom for the KV side to grow into without the weight side having to give
/// anything back — 32 regions is 512 MiB, which is several turns' worth at the
/// measured steady state. Taking right up to the mark would make every small
/// increase in KV demand a boundary move, and a boundary move costs an eviction.
const KV_REGION_SLACK: usize = 32;

/// The smallest grant this cache can spend, in regions. An expert slot is far
/// smaller than a region, so every whole region handed over becomes residency.
const EXPERT_MIN_GRANT_REGIONS: usize = 8;

/// Why a growth negotiation ended where it did.
///
/// The boundary is asked to grow once per wave and answers zero almost every
/// time, and the reasons are not interchangeable: "the KV side reports no spare"
/// wants the spare calculation looked at, "the floor refused" wants the call
/// site moved, and "the target did not change" wants the region→slot conversion
/// looked at.
#[derive(Clone, Copy)]
pub(crate) enum GrowOutcome {
    Asked,
    NoSpare,
    SpareOffered(usize),
    TargetUnchanged,
    TargetWentBackwards,
    FloorRefused,
    AtLimit,
    Gained(usize),
}

/// Tallies for [`GrowOutcome`], in its variant order.
static GROW_TALLY: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

pub(crate) fn grow_note(outcome: GrowOutcome) {
    let (idx, add) = match outcome {
        GrowOutcome::Asked => (0, 1),
        GrowOutcome::NoSpare => (1, 1),
        GrowOutcome::SpareOffered(n) => (2, n as u64),
        GrowOutcome::TargetUnchanged => (3, 1),
        GrowOutcome::TargetWentBackwards => (4, 1),
        GrowOutcome::FloorRefused => (5, 1),
        GrowOutcome::AtLimit => (6, 1),
        GrowOutcome::Gained(n) => (7, n as u64),
    };
    GROW_TALLY[idx].fetch_add(add, Ordering::Relaxed);
}

/// `(asked, no_spare, spare_regions_offered, target_unchanged, target_backwards,
/// floor_refused, at_limit, slots_gained)` since boot.
pub fn grow_tally() -> [u64; 8] {
    std::array::from_fn(|i| GROW_TALLY[i].load(Ordering::Relaxed))
}

/// The growth question, asked by the thread between forwards: the regions the
/// KV side holds spare that this cache could take, 0 for none.
///
/// **Asked here, not on the pipeline thread.** The answer is the region pool's
/// present occupancy, knowable right after the wave's transient tier is handed
/// back and the empty arenas are swept — which is where the caller stands. The
/// pipeline thread answers its messages in order, so asked there the question
/// first waited for every routed layer still queued: the device to reach the
/// forward's last MoE layer, then the thread to serve the backlog — on
/// Qwen3.8-Flash-Next single-session decode (RTX PRO 5000) 2–3 ms a forward,
/// nearly always to learn there was nothing spare. Only a non-zero answer goes
/// to the pipeline thread ([`BoundaryAsk::Take`]), which owns the move.
pub(crate) fn spare_for_growth(stream: &Arc<CudaStream>) -> Result<usize> {
    grow_note(GrowOutcome::Asked);
    let spare = kv_spare_regions(stream, KV_REGION_SLACK, EXPERT_MIN_GRANT_REGIONS)?;
    grow_note(if spare == 0 {
        GrowOutcome::NoSpare
    } else {
        GrowOutcome::SpareOffered(spare)
    });
    Ok(spare)
}

impl PipelineState {
    /// Move the boundary if it may move now, deciding before anything is
    /// touched, and answer with the bytes conceded.
    pub(crate) fn renegotiate_if_quiet(&mut self, ask: BoundaryAsk) -> Result<u64> {
        let Device::Cuda(cd) = &self.device else {
            return Ok(0);
        };
        let pass_state = self.pass_state.clone();
        let p = pass_state
            .lock()
            .map_err(|_| candle::Error::Msg("expert pipeline: pass state poisoned".into()))?;
        let live = wave_is_live(cd.cuda_stream().context().ordinal());
        let unserved = p.reserved != self.routed_served.load(Ordering::Acquire);
        if live || unserved {
            // A KV purchase refused is the one refusal a caller may fail on — a
            // tier that cannot be placed fails its wave — so it says which gate
            // closed. The give-back direction is refused this way every forward
            // and stays quiet.
            match ask {
                BoundaryAsk::Sell(wanted) | BoundaryAsk::SellPaused(wanted) => tracing::info!(
                    target: "candle_transformers::expert_lre",
                    wanted,
                    wave_live = live,
                    invocation_unserved = unserved,
                    "weight side refused a KV purchase: the boundary moves only between \
                     forwards"
                ),
                BoundaryAsk::StashPinned => tracing::info!(
                    target: "candle_transformers::expert_lre",
                    wave_live = live,
                    invocation_unserved = unserved,
                    "the pinned experts were not stashed: the boundary moves only between \
                     forwards"
                ),
                // The one ask that must not be lost: a zone left below its floor
                // cannot serve a token. Its caller asks again until it lands.
                BoundaryAsk::RestoreFloor => tracing::warn!(
                    target: "candle_transformers::expert_lre",
                    wave_live = live,
                    invocation_unserved = unserved,
                    "the zone's floor could not be restored yet: the boundary moves only \
                     between forwards"
                ),
                BoundaryAsk::Take(_) => {}
            }
            return Ok(0);
        }
        // **Never while a wave records.** The move quiesces the device on both
        // sides of the handover, and the driver refuses a context-wide
        // synchronise while any stream captures — refusing it invalidates the
        // capture on the recording thread. Between forwards is not enough: the
        // draft walk and the speculative rewind record there. Measured in zend:
        // a requested retraction landed inside a draft walk, its next launch
        // failed, and the expert pipeline aborted. Held for the whole move, so no
        // recording can begin between the two quiesces; refused, it concedes
        // nothing now and the next negotiation asks again.
        let conceded = match try_without_recording(|quiet| self.renegotiate_boundary(quiet, ask)) {
            Some(conceded) => conceded,
            None => {
                // A guest's concession or its restore deferred is one the caller
                // is waiting on with the engine stopped, so it is said at INFO;
                // the per-forward growth question is deferred this way routinely.
                if matches!(
                    ask,
                    BoundaryAsk::SellPaused(_) | BoundaryAsk::RestoreFloor | BoundaryAsk::StashPinned
                ) {
                    tracing::info!(
                        target: "candle_transformers::expert_lre",
                        ?ask,
                        "boundary move deferred: a wave capture is recording"
                    );
                } else {
                    tracing::debug!(
                        target: "candle_transformers::expert_lre",
                        "boundary move deferred: a wave capture is recording"
                    );
                }
                Ok(0)
            }
        };
        // **The zone's gauges follow the boundary, not only the next routed
        // layer.** A move between forwards — a new sequence's state buying
        // ground, the weight side taking spare regions back — changes how much
        // the zone could still concede, and the prefill width cap reads that
        // figure before the next forward routes anything. Left to the next
        // layer, it priced a Qwen3.8-Flash-Next ×8 prefill against ground the
        // zone had already sold, and the wave's tier came up short.
        self.publish_gauges();
        drop(p);
        conceded
    }

    /// Move the weight/KV boundary: sell the regions the KV side is asking for,
    /// or take back the regions it is holding spare.
    ///
    /// **Both quantities are stated by the caller, never inferred here.** A running count
    /// of refused claims was once spent as regions: one failed drain left 4,436
    /// behind it against a KV side 28 regions short, and the retraction that
    /// followed evicted the zone below its own pinned working set.
    ///
    /// Both directions are **non-destructive by preference**. Growing takes only
    /// free regions. Shrinking relocates the hottest doomed experts into free
    /// slots below the new frontier and drops the rest — the worst case is a
    /// promotion later, never a loss: every expert has a copy in the pack.
    ///
    /// The retraction stops at the zone's floor (`minimum_resident_slots`, the
    /// fewest slots this cache can serve a token with) and nowhere else.
    ///
    /// Answers with the **bytes conceded to the KV side** — zero when the
    /// boundary held or moved the other way.
    fn renegotiate_boundary(&mut self, quiet: &Quiet, ask: BoundaryAsk) -> Result<u64> {
        let Device::Cuda(cd) = &self.device else {
            return Ok(0);
        };
        let stream = cd.cuda_stream();
        // **The pinned set's ground comes back first, and alone.** While it is
        // out, the zone believes it holds slots whose ground the KV side has,
        // so no other move may be priced against it: a refused unstash leaves
        // the stash standing and the restore is asked again.
        match ask {
            BoundaryAsk::StashPinned => return self.stash_pinned(quiet),
            BoundaryAsk::RestoreFloor => self.unstash_pinned(quiet)?,
            _ if self.pinned_stash.is_some() => return Ok(0),
            _ => {}
        }
        let growing = matches!(ask, BoundaryAsk::Take(_) | BoundaryAsk::RestoreFloor);
        let within = match ask {
            BoundaryAsk::SellPaused(_) => ZoneFloor::Paused,
            _ => ZoneFloor::Serving,
        };
        let delta = match ask {
            BoundaryAsk::Take(0)
            | BoundaryAsk::Sell(0)
            | BoundaryAsk::SellPaused(0)
            | BoundaryAsk::StashPinned => return Ok(0),
            BoundaryAsk::Take(spare) => -(spare as isize),
            BoundaryAsk::Sell(wanted) | BoundaryAsk::SellPaused(wanted) => wanted as isize,
            BoundaryAsk::RestoreFloor => {
                let short = self
                    .inner
                    .zone
                    .stated_floor()
                    .saturating_sub(self.inner.zone.capacity());
                if short == 0 {
                    return Ok(0);
                }
                -((short * self.inner.zone.slot_bytes()).div_ceil(REGION_BYTES) as isize)
            }
        };
        let floor = weight_floor_after(&stream, delta)?;
        let before = self.inner.zone.capacity();
        let mut target = self.inner.zone.capacity_for_frontier(floor);
        if ask == BoundaryAsk::RestoreFloor {
            // Back to the floor and no further: what lies past it is the
            // ordinary growth question's, with its slack.
            target = target.min(self.inner.zone.stated_floor());
        }
        if matches!(ask, BoundaryAsk::SellPaused(_) | BoundaryAsk::RestoreFloor) {
            tracing::info!(
                target: "candle_transformers::expert_lre",
                ?ask,
                delta,
                slots = before,
                target,
                pinned_slots = self.inner.pinned_slots(),
                stated_floor = self.inner.zone.stated_floor(),
                "a guest's boundary move"
            );
        }
        if target == before {
            if growing {
                // At the limit is not the same fact as nothing to take: a cache
                // holding every expert reports an unchanged target however much
                // ground it is offered.
                if before >= self.inner.zone.limit() {
                    grow_note(GrowOutcome::AtLimit);
                } else {
                    grow_note(GrowOutcome::TargetUnchanged);
                }
            }
            return Ok(0);
        }
        if growing && target < before {
            grow_note(GrowOutcome::TargetWentBackwards);
        }

        if target > self.inner.zone.capacity() {
            // Taking ground is a handover too: those regions were the KV side's
            // until this instant, and "free" there means no host-side gid names
            // them, not that no kernel is still reading them.
            self.quiesce_before_handover(quiet)?;
            // **The floor moves first, and the zone follows it.** A refusal
            // (a wave generation open on the span) then lands with nothing yet
            // moved; growing the zone first once left it one slot wider than
            // the published boundary, and a promotion into that slot wrote KV
            // ground.
            let grown_floor = self.inner.zone.frontier_after_growth(target);
            let gained = if grown_floor < self.inner.zone.frontier_for_capacity() {
                match set_weight_floor(&stream, grown_floor) {
                    Ok(_) => self.inner.grow_zone(target),
                    Err(e) => {
                        grow_note(GrowOutcome::FloorRefused);
                        return Err(e);
                    }
                }
            } else {
                grow_note(GrowOutcome::AtLimit);
                0
            };
            grow_note(GrowOutcome::Gained(gained));
            if gained > 0 {
                tracing::trace!(
                    target: "candle_transformers::expert_lre",
                    gained,
                    spare = -delta,
                    slots = self.inner.zone.capacity(),
                    "weight side took free KV regions"
                );
            }
            return Ok(0);
        }

        // **Refuse before touching anything**, exactly as the growth path does:
        // a retraction refused half-way once left the zone believing it was
        // smaller while experts were still live past the new capacity.
        let floor_slots = match within {
            ZoneFloor::Serving => self.inner.zone.min_capacity(),
            ZoneFloor::Paused => self.inner.pinned_slots().min(before),
        };
        if target.max(floor_slots) >= before {
            tracing::info!(
                target: "candle_transformers::expert_lre",
                wanted = delta,
                slots = before,
                floor_slots,
                paused = within == ZoneFloor::Paused,
                "weight side is on its floor and can concede no further ground"
            );
            return Ok(0);
        }
        // The device is idle after this and no invocation can begin (the pass
        // lock is held), so every entry may change at once: nothing can read
        // the table until the move is done.
        self.quiesce_before_handover(quiet)?;
        self.drain_ring(self.routed_served.load(Ordering::Acquire))?;

        // The zone decides who moves and who goes; this performs it. A
        // displaced survivor is evicted first: the relocation into its slot
        // takes over the slot's bookkeeping.
        let plan = self.inner.retract_zone(target, within);
        let displaced: Vec<(usize, usize)> = plan
            .displace
            .iter()
            .filter_map(|&slot_idx| self.inner.evict(slot_idx))
            .collect();
        let mut moved = Vec::with_capacity(plan.relocate.len());
        for &(from, to) in &plan.relocate {
            if let Some(key) = self.relocate_slot(from, to)? {
                moved.push((key, to));
            }
        }
        // The relocation copies must land before an entry names their
        // destinations.
        self.copy_stream
            .synchronize()
            .map_err(candle::Error::wrap)?;
        {
            let mut r = self
                .residency
                .lock()
                .map_err(|_| candle::Error::Msg("expert pipeline: residency poisoned".into()))?;
            for &(row, expert) in &displaced {
                r.set_vram(row, expert, None);
            }
            for &((row, expert), to) in &moved {
                r.set_vram(row, expert, Some((to, self.inner.slot_base(to))));
            }
            for &slot_idx in &plan.evict {
                if let Some((row, expert)) = self.inner.evict(slot_idx) {
                    r.set_vram(row, expert, None);
                }
            }
        }
        if let Ok(mut s) = self.stats.lock() {
            s.evictions += plan.evict.len() + displaced.len();
        }
        // Only now: the relocations above read `slot_to_key` for the slots the
        // truncation removes.
        self.inner.truncate_tables();
        // The conceded slots stop being ours the moment the floor moves.
        self.quiesce_before_handover(quiet)?;
        // **A refused publish here has to be undone, not carried out.** The
        // doomed slots were evicted or relocated, so they are free; growing
        // the zone back restores the agreement between the zone and the
        // published floor, and costs only promotions of dropped experts.
        if let Err(e) = set_weight_floor(&stream, self.inner.zone.frontier_for_capacity()) {
            self.inner.grow_zone(before);
            return Err(e);
        }
        let conceded =
            (before - self.inner.zone.capacity()) as u64 * self.inner.zone.slot_bytes() as u64;
        tracing::trace!(
            target: "candle_transformers::expert_lre",
            wanted = delta,
            relocated = plan.relocate.len(),
            evicted = plan.evict.len(),
            slots = self.inner.zone.capacity(),
            floor_slots = self.inner.zone.min_capacity(),
            conceded_mib = conceded / (1 << 20),
            "weight side gave ground to KV"
        );
        Ok(conceded)
    }

    /// Copy the pinned set to host memory and hand its ground to the KV side
    /// ([`BoundaryAsk::StashPinned`]). Answers the bytes conceded.
    ///
    /// **Only from the pinned set, and nothing in the zone moves.** The zone
    /// must already stand on its pinned slots (a `SellPaused`), which are
    /// slots `0..n`, contiguous against the span's right edge; their bytes go
    /// to the host whole and the published floor moves to the edge. Tables,
    /// slot views and every captured address stay as they are, and
    /// [`Self::unstash_pinned`] writes the same bytes back to the same
    /// addresses, so a pinned expert's address never changes. Nothing forwards
    /// until it has: [`PipelineStats::pinned_stashed`] holds the serving floor
    /// unmet.
    ///
    /// [`PipelineStats::pinned_stashed`]: super::types::PipelineStats::pinned_stashed
    fn stash_pinned(&mut self, quiet: &Quiet) -> Result<u64> {
        let Device::Cuda(cd) = &self.device else {
            return Ok(0);
        };
        if self.pinned_stash.is_some() {
            return Ok(0);
        }
        let n = self.inner.zone.capacity();
        let pinned = self.inner.pinned_slots();
        if n == 0 || n > pinned {
            tracing::info!(
                target: "candle_transformers::expert_lre",
                slots = n,
                pinned_slots = pinned,
                "the pinned experts were not stashed: the zone does not stand on its pinned set"
            );
            return Ok(0);
        }
        let stream = cd.cuda_stream();
        let frontier = self.inner.zone.frontier_for_capacity();
        let len = n * self.inner.zone.slot_bytes();
        let edge = frontier + len as u64;
        self.quiesce_before_handover(quiet)?;
        self.drain_ring(self.routed_served.load(Ordering::Acquire))?;
        let started = std::time::Instant::now();
        let mut host = vec![0u8; len];
        // SAFETY: `[frontier, edge)` is the zone's own ground — its `n` slots,
        // live and quiesced — and `host` is exactly that long.
        unsafe {
            cudarc::driver::result::memcpy_dtoh_async(
                &mut host,
                frontier,
                self.copy_stream.cu_stream(),
            )
        }
        .map_err(candle::Error::wrap)?;
        self.copy_stream
            .synchronize()
            .map_err(candle::Error::wrap)?;
        self.quiesce_before_handover(quiet)?;
        set_weight_floor(&stream, edge)?;
        self.pinned_stash = Some((frontier, host));
        if let Ok(mut s) = self.stats.lock() {
            s.pinned_stashed = true;
        }
        self.publish_gauges();
        tracing::info!(
            target: "candle_transformers::expert_lre",
            slots = n,
            stashed_mib = len >> 20,
            ms = started.elapsed().as_millis() as u64,
            "the pinned experts stand in host memory for a guest"
        );
        Ok(len as u64)
    }

    /// Put the pinned set back where [`Self::stash_pinned`] took it from:
    /// the floor first, so a refusal (KV still standing in that ground) lands
    /// with nothing moved and the stash kept for the next ask, then the bytes.
    fn unstash_pinned(&mut self, quiet: &Quiet) -> Result<()> {
        let Device::Cuda(cd) = &self.device else {
            return Ok(());
        };
        let Some((frontier, _)) = self.pinned_stash.as_ref() else {
            return Ok(());
        };
        let frontier = *frontier;
        let stream = cd.cuda_stream();
        self.quiesce_before_handover(quiet)?;
        set_weight_floor(&stream, frontier)?;
        let Some((_, host)) = self.pinned_stash.take() else {
            return Ok(());
        };
        let started = std::time::Instant::now();
        // SAFETY: the floor now stands at `frontier` again, so the ground is
        // the zone's; nothing reads a pinned slot until this returns.
        let copied = unsafe {
            cudarc::driver::result::memcpy_htod_async(frontier, &host, self.copy_stream.cu_stream())
        }
        .and_then(|()| self.copy_stream.synchronize());
        if let Err(e) = copied {
            self.pinned_stash = Some((frontier, host));
            return Err(candle::Error::wrap(e));
        }
        if let Ok(mut s) = self.stats.lock() {
            s.pinned_stashed = false;
        }
        self.publish_gauges();
        tracing::info!(
            target: "candle_transformers::expert_lre",
            restored_mib = host.len() >> 20,
            ms = started.elapsed().as_millis() as u64,
            "the pinned experts are back in VRAM"
        );
        Ok(())
    }

    /// Retire every kernel in flight before a byte changes owner.
    ///
    /// **The boundary is the one place where memory changes side**, and neither
    /// side's own ordering reaches across it: ground arriving from the weight
    /// side is fresh to the KV side, claimed with no wait. A device-wide
    /// synchronize, paid only when the boundary actually moves — and only with
    /// the record gate held (`quiet`), since it would invalidate any capture.
    fn quiesce_before_handover(&self, quiet: &Quiet) -> Result<()> {
        let Device::Cuda(cd) = &self.device else {
            return Ok(());
        };
        let stream = cd.cuda_stream();
        quiet
            .synchronize_context(stream.context())
            .map_err(candle::Error::wrap)
    }

    /// Move one expert's bytes from slot `from` to slot `to` on the copy
    /// stream, and its bookkeeping with them. Returns the expert, or `None` for
    /// an empty source — whose destination goes straight back to the zone,
    /// which marked it occupied when it built the plan.
    ///
    /// The caller synchronizes the copy stream before any entry names `to`.
    fn relocate_slot(&mut self, from: usize, to: usize) -> Result<Option<(usize, usize)>> {
        let Device::Cuda(cd) = &self.device else {
            return Ok(None);
        };
        let Some(key) = self.inner.slot_to_key[from] else {
            self.inner.zone.release(to);
            return Ok(None);
        };
        let geom = &self.layer_geometries[key.0];
        let bytes = slot_offsets(geom).3;
        let src = self.inner.slot_base(from);
        let dst = self.inner.slot_base(to);
        // SAFETY: both addresses name whole slots of the zone; the source is
        // live past the frontier and the destination below it — free, or
        // emptied by its displaced occupant's eviction — so the two cannot
        // alias. The device is quiesced.
        unsafe {
            cudarc::driver::result::memcpy_dtod_async(
                dst,
                src,
                bytes,
                self.copy_stream.cu_stream(),
            )
            .map_err(candle::Error::wrap)?;
        }
        // SAFETY: the copy above puts this layer's three projections at `dst`
        // before anything reads the views (the caller synchronizes first).
        let moved = unsafe { build_slot_view(geom, cd, dst)? };
        self.inner.slots[from] = None;
        self.inner.slot_to_key[from] = None;
        self.inner.slots[to] = Some(moved);
        self.inner.slot_to_key[to] = Some(key);
        #[cfg(feature = "tensor-assert")]
        self.inner.mirror_tenant(to);
        self.inner.key_to_slot.insert(key, to);
        self.inner.last_used[to] = self.inner.last_used[from];
        Ok(Some(key))
    }
}
