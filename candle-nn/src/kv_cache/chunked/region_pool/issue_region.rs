//! Hands a region to a new owner, cleaning it first if a previous tenant left bytes.
//!
//! Every issued region passes through here whichever path picked it, because a
//! region can arrive dirty from the free list, from the wave tier, or from the
//! weight boundary, and all three need the same clean. The clean waits for the
//! device to quiesce, but only once per batch of releases, and it is deferred
//! while a wave is recording. It is skipped when the claiming stream already
//! orders the clean behind the old readers. A region that is still held by an
//! owner is refused with an error rather than `None`, since that is a
//! bookkeeping fault that would otherwise pass silently.

use super::*;

/// Run `f` against this device's pool, creating the reservation on first use.
/// Recycled-claim cost, split into its two halves because they have different
/// answers if either ever gets expensive.
///
/// The **sync** is the whole-device quiesce of [`claim_region`], amortised by
/// the epoch stamp: `quiesced` counts the claims that actually paid it. If that
/// ratio climbs back toward 1 the batching has stopped working, and the next
/// move is a per-region release event. The **fill** is a synchronous 16 MiB
/// memset; if *it* dominates the answer is to zero on release instead, off the
/// allocation path.
///
/// Reported under `KV_ARENA_STATS`, next to `[arena-create]`, so one gate run
/// with the flag set gives the whole allocation fast path.
pub(super) static RECYCLE_COUNT: AtomicU64 = AtomicU64::new(0);

pub(super) static RECYCLE_QUIESCED: AtomicU64 = AtomicU64::new(0);

pub(super) static RECYCLE_SYNC_NS: AtomicU64 = AtomicU64::new(0);

pub(super) static RECYCLE_FILL_NS: AtomicU64 = AtomicU64::new(0);

pub(super) static RECYCLE_SYNC_MAX_NS: AtomicU64 = AtomicU64::new(0);

/// Every nanosecond a recycled claim has spent in its device-wide quiesce,
/// counted whatever reports it, so a caller can charge the wait inside its own
/// span to the quiesce rather than to its own work ([`quiesce_wait_ns`]).
pub(super) static QUIESCE_WAIT_NS: AtomicU64 = AtomicU64::new(0);

/// The time recycled claims have waited in their device-wide quiesce, in all,
/// since the process started. Read before and after a span of work, the
/// difference is how much of it was the wait for the device to drain.
pub fn quiesce_wait_ns() -> u64 {
    QUIESCE_WAIT_NS.load(Ordering::Relaxed)
}

pub(super) fn record_recycle(quiesced: bool, sync_ns: u64, fill_ns: u64) {
    if quiesced {
        QUIESCE_WAIT_NS.fetch_add(sync_ns, Ordering::Relaxed);
    }
    if !super::super::alloc::arena_stats_enabled() {
        return;
    }
    let n = RECYCLE_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    let q =
        RECYCLE_QUIESCED.fetch_add(u64::from(quiesced), Ordering::Relaxed) + u64::from(quiesced);
    let sync_total = RECYCLE_SYNC_NS.fetch_add(sync_ns, Ordering::Relaxed) + sync_ns;
    let fill_total = RECYCLE_FILL_NS.fetch_add(fill_ns, Ordering::Relaxed) + fill_ns;
    RECYCLE_SYNC_MAX_NS.fetch_max(sync_ns, Ordering::Relaxed);
    let ms = |ns: u64| ns as f64 / 1_000_000.0;
    eprintln!(
        "[region-recycle] quiesced={quiesced} sync_ms={:.3} fill_ms={:.3} | n={n} \
         quiesces={q} sync_total_ms={:.1} sync_max_ms={:.3} fill_total_ms={:.1} \
         fill_avg_ms={:.3}",
        ms(sync_ns),
        ms(fill_ns),
        ms(sync_total),
        ms(RECYCLE_SYNC_MAX_NS.load(Ordering::Relaxed)),
        ms(fill_total),
        ms(fill_total) / n as f64,
    );
}

/// Hand region `index` to a new owner: clean it if a previous tenant left bytes
/// in it, and mark it held. `recycled` says whether it came off the free list
/// rather than past `next`. The pick is the caller's ([`try_claim`] for one
/// region, [`try_claim_run`] for a contiguous run); this is the part every
/// issued region goes through, whoever picked it.
pub(super) fn issue_region(
    pool: &mut RegionPool,
    stream: &std::sync::Arc<CudaStream>,
    index: usize,
    recycled: bool,
) -> Result<RegionHandle> {
    {
        let base = pool.region_base(index);
        // **Whether it needs cleaning, not where it came from.** This used to key
        // on `recycled` — a region taken off the free list — which silently
        // assumed the free list was the only way a region could arrive with
        // someone else's bytes in it. It is not, and both of the others are on
        // the hot path:
        //
        // - **The wave transient tier.** It stands on KV ground for a whole
        //   forward and writes intermediates across all of it. When it is
        //   released, `next` may then advance straight into that ground as a
        //   *fresh* claim — no free-list entry, so no quiesce and no zeroing,
        //   while the forward's attention and FFN kernels are still in flight
        //   writing there. A `tier_reserve` used to keep the KV side off that
        //   band until the next placement moved it, which hid the omission; with
        //   the reserve gone the race is direct, and it produces exactly what it
        //   sounds like — KV chunks interleaved with activation bytes, decoding
        //   to NaN.
        // - **The weight boundary.** Ground conceded by the expert zone held
        //   expert weights a moment ago. The handover quiesces
        //   (`quiesce_before_handover`), so there are no readers left, but the
        //   bytes are still there and a fresh claim would hand them to a KV
        //   arena unzeroed.
        let dirty = pool.dirt[index].epoch != 0;
        if dirty {
            // **Quiesce before re-tenanting.** A region returns to the free list
            // the moment its arena drops, and that says only that no host-side
            // gid still names it — kernels launched earlier can still be reading
            // it, on the compute stream or the persistence thread's copy stream.
            // Handing those bytes to a new owner and memsetting them is then a
            // read-after-free, and it reports as `CUDA_ERROR_ILLEGAL_ADDRESS` in
            // whichever unrelated kernel happens to be in flight.
            //
            // The allocator used to supply this ordering for free: an arena's
            // slab was released with `cuMemFreeAsync` and re-allocated with
            // `cuMemAllocAsync` on the same stream, so reuse could not overtake
            // the reads. A free-list push has no such ordering, so the wait is
            // explicit — device-wide, because the copy stream is one of the
            // readers and draining the compute stream alone would not cover it.
            //
            // **But at most once per batch of releases.** A quiesce retires every
            // kernel launched before it, on every stream, so it discharges the
            // debt of *all* regions released up to that point — not just the one
            // that triggered it. Regions come back in bulk (an eviction pass
            // drops a turn's arenas together) and are re-claimed in bulk, so the
            // epoch stamp turns what was a wait per claim into a wait per batch.
            //
            // This was measured, not assumed. Per-claim, the gate run paid
            // **2,837 ms** across 395 recycled claims — 7.2 ms average, 35.3 ms
            // worst — which is 2.4 % of the run spent stalled on an allocation.
            // (Step 4 measured this as free, at 0.029 ms per arena creation. It
            // was: back then the scheduler synchronised the device every wave in
            // `release_empty_arenas`, so the queue was always shallow here. Step
            // 5 removed that sweep-path sync — correctly, it was guarding an
            // unmap that no longer happens — and this became the real cost of
            // the ordering rather than a free ride on someone else's.)
            //
            // SAFETY: the region is inside the reservation and mapped, and no
            // kernel is reading it — either because the sync below just retired
            // them all, or because an earlier one did and nothing has re-read
            // the region since (it has been on the free list throughout).
            // Bound unconditionally: the fill below needs a current context on
            // this thread whether or not the sync ran.
            //
            // **Never while a wave records.** The driver refuses a context-wide
            // synchronise while any stream captures, and the refusal invalidates
            // the capture on the recording thread — measured in zend as a draft
            // walk killed by the persistence thread's arena creation, and the
            // expert pipeline aborting behind it. The gate runs the clean only
            // when nothing records, and holds recordings off until it is done;
            // otherwise the region goes back and the claim is deferred like any
            // other mid-wave arena creation.
            //
            // **Nor when this stream already orders the clean behind them.** Ground
            // the wave tier left is touched by the compute stream alone; a claim on
            // that stream queues its memset behind the forward, as the next
            // forward's tier placement does, and needs no device-wide wait
            // ([`Dirt`]). Measured on the RTX 3090: the gap's first lease after a
            // prefill waited 180–580 ms here for the prefill to drain.
            let needs_sync = pool.dirt[index].needs_sync(pool.quiesce_epoch, stream_id(stream));
            let quiesce_epoch = &mut pool.quiesce_epoch;
            let cleaned = try_without_recording(|quiet| {
                let bound = stream.context().bind_to_thread();
                let t_sync = Instant::now();
                let synced = bound.and_then(|()| {
                    if !needs_sync {
                        return Ok(());
                    }
                    let r = quiet.synchronize_context(stream.context());
                    if r.is_ok() {
                        *quiesce_epoch += 1;
                    }
                    r
                });
                let sync_ns = t_sync.elapsed().as_nanos() as u64;
                let t_fill = Instant::now();
                // On the device's stream, ahead of the region's next tenant;
                // nothing records while the gate's `quiet` is held.
                let zeroed = synced.and_then(|()| unsafe {
                    memset_d8_async(base, 0, REGION_BYTES, stream.cu_stream())
                });
                record_recycle(needs_sync, sync_ns, t_fill.elapsed().as_nanos() as u64);
                zeroed
            });
            let Some(zeroed) = cleaned else {
                pool.free.push(Reverse(index));
                candle::bail!(
                    "{KV_ARENA_MID_WAVE} (recycling region {index}): a wave capture is \
                     recording, and the clean's device-wide synchronise would invalidate it"
                )
            };
            if let Err(e) = zeroed {
                // Put it back: a region that could not be cleaned is still the
                // pool's, and losing it here would shrink the span silently.
                // Its epoch stamp stands, so the retry re-evaluates the wait
                // against whatever has happened in the meantime.
                pool.free.push(Reverse(index));
                candle::bail!("recycling region {index}: {e}")
            }
            // Clean now, and its next tenant learns that from the stamp rather
            // than from having watched it happen.
            pool.dirt[index] = Dirt::default();
        }
        // **Refuse to hand out a region somebody still holds.**
        //
        // Reached only if the free list carried an index that was never released,
        // or if `next` advanced over a live region. Both are bookkeeping faults,
        // and both are silent without this: the second tenant's writes land on
        // the first tenant's live data, every address involved is mapped, and
        // nothing faults. It has been measured as a sequence's whole recurrent
        // state going non-finite while the KV side read perfectly — the reader of
        // the damage was in a different subsystem from the writer, with no path
        // between them to inspect.
        //
        // **An error, which reaches the turn — not `Ok(None)`, which would not.**
        //
        // The distinction is the whole point. `None` is this function's pressure
        // signal: the caller buys ground from the weight side, retries, and carries
        // on. Returning it here would turn a broken partition into a slightly
        // slower run with nothing in the output naming it. An `Err` takes a
        // different path — every caller propagates it with `?` (`alloc.rs`'s arena
        // creation, `SpanClaims::claim`, the guest ground) up to the prefill or
        // decode that asked, which fails the turn.
        //
        // That is the right blast radius, and it is why this is not a panic: the
        // conversation that asked for the region fails and says why, while the
        // daemon and every other session stay up to be inspected. Nothing is
        // recovered — the duplicate is already in the free list or already past
        // `next`, and this line is where it is noticed, not where it was made.
        if pool.owned[index] {
            candle::bail!(
                "reservation: region {index} was issued while still held — \
                 {} live of {}, {} free, next {}, watermark {}. The free list or \
                 the fresh-claim frontier has lost track of an owner; handing it \
                 over would write over a live tenant (a KV arena, the wave tier, \
                 or a sequence's recurrent store) without faulting.",
                pool.live,
                pool.total,
                pool.free.len(),
                pool.next,
                pool.live_watermark(),
            )
        }
        pool.owned[index] = true;
        pool.live += 1;
        pool.peak_live = pool.peak_live.max(pool.live);
        // Paired with the release log below: a fault correlated between a
        // release of region N and its re-claim is the re-tenancy signature, and
        // the region index is the only handle a log has on which bytes those
        // were.
        log::debug!(
            "region {index} claimed ({}{}), {} live of {}",
            if recycled { "recycled" } else { "fresh" },
            if dirty { ", cleaned" } else { "" },
            pool.live,
            pool.total,
        );
        Ok(RegionHandle {
            ordinal: stream.context().ordinal(),
            index,
            base,
        })
    }
}
