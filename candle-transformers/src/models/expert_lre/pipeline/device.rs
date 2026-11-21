//! Lands the expert slots the GPU has filled, and drops the demand claims that a
//! faulted forward gave up.
//!
//! A promotion is published only once its invocation has completed. A
//! bucketize that finds an expert unmarked and cold may hand it to the CPU pool,
//! which reads only host memory, so no VRAM publish may follow an unmarked cold
//! read. A read-ahead claim is kept through a fault, because its copies are
//! already published by the device. A demand claim whose slot was never written
//! is dropped, and its zone slot is returned. The fault word is read only after
//! the completion it is ordered behind.

use super::*;

/// Whether a device promotion lands now: its invocation is `complete`, and
/// either it is a read-ahead claim (which never waits on a cold expert) or no
/// forward has `faulted` — a demand claim's item may have been given up, its
/// slot never written, and it waits for the fault to be taken, which drops it
/// (`PipelineState::drop_demand_promotions`).
///
/// `faulted` is read only after `complete` was: a worker claims the fault word
/// and fences before any later launch begins, and a later launch's start is
/// what makes a claim complete — so a claim of the faulted launch judged
/// complete always reads the word set.
pub(super) fn lands(complete: bool, ahead: bool, faulted: impl FnOnce() -> bool) -> bool {
    complete && (ahead || !faulted())
}

/// Pending promotions split into the read-ahead claims, which a fault leaves to
/// land, and the demand claims it drops — each in its pending order.
pub(super) fn split_ahead(
    pending: Vec<DevicePromotion>,
) -> (Vec<DevicePromotion>, Vec<DevicePromotion>) {
    pending.into_iter().partition(|p| p.ahead)
}

/// A slot the GPU is filling with a missed expert: whole once invocation
/// `ticket` has completed.
pub(super) struct DevicePromotion {
    pub(super) row: usize,
    pub(super) expert: usize,
    pub(super) slot: usize,
    pub(super) ticket: u64,
    /// A read-ahead claim: its gate launch's workers copy it whole and publish
    /// it, whatever else the launch gives up (`fault`).
    pub(super) ahead: bool,
}

impl PipelineState {
    /// Collect what the GPU has taken from the promotion ring, in log order.
    /// A skipped victim's offer comes back with its expert still resident. A
    /// claimed victim was evicted by the device — its entries retargeted before
    /// its slot was written — so its bookkeeping follows at once, ahead of any
    /// later claim that could promote the same expert again. Each claimed slot
    /// becomes a device promotion, whole once its invocation is done; a
    /// read-ahead claim's expert is already published by the device, and its
    /// row's routing will judge the prediction. `near` is the ticket being
    /// served — the logged words are resolved against it.
    pub(super) fn collect_device_promotions(&mut self, near: u64) -> Result<()> {
        let Some(ring) = self.promo_ring.clone() else {
            return Ok(());
        };
        let head = ring.head();
        let (mut taken, mut claimed, mut skipped) = (0usize, 0usize, 0usize);
        let (mut ahead, mut ahead_bytes, mut ahead_pad) = (0usize, 0usize, 0usize);
        let mut carried = 0usize;
        while self.ring_taken != head {
            let i = self.ring_taken as usize % ring.cap();
            let logged = ring.log(self.ring_taken);
            let offer = self.ring_slots[i]
                .take()
                .expect("a taken ring index holds the offer made there");
            self.ring_taken = self.ring_taken.wrapping_add(1);
            self.offered.remove(&offer.slot);
            if logged.skipped() {
                self.release_offer(&offer)?;
                skipped += 1;
                continue;
            }
            if let Some(((vr, ve), _)) = offer.victim {
                let evicted = self.inner.evict(offer.slot);
                if evicted != Some((vr, ve)) {
                    candle::bail!(
                        "expert pipeline: ring slot {} was offered holding ({vr}, {ve}) but held \
                         {evicted:?} when the device claimed it",
                        offer.slot
                    );
                }
                // The entry now names the fallback the device wrote (the
                // stager may have booked a cold one first); only then may the
                // stager evict a pad copy it names.
                self.residency()?.device_evicted(vr, ve);
                self.release_offer(&offer)?;
                let tag = self.victim_tag(vr, ve);
                // A carry promotes an expert its row routed, as a demand
                // claim does: judged with them, not with the predictions.
                self.regret.evicted(
                    vr,
                    ve,
                    logged.ahead && !logged.carry,
                    ticket_from_word(logged.word, near),
                    tag,
                );
                claimed += 1;
            }
            taken += 1;
            if logged.carry {
                carried += 1;
                ahead_bytes += slot_offsets(&self.layer_geometries[logged.row]).3;
            } else if logged.ahead {
                ahead += 1;
                ahead_bytes += slot_offsets(&self.layer_geometries[logged.row]).3;
                // Listed from the pad: its pin holds until this lands.
                ahead_pad += usize::from(self.ahead_pins.lists(logged.row, logged.expert));
                self.speculative_loads.insert((logged.row, logged.expert));
            }
            self.promoting.insert((logged.row, logged.expert));
            self.regret.promoted(
                logged.row,
                logged.expert,
                logged.ahead && !logged.carry,
                ticket_from_word(logged.word, near),
            );
            self.device_promotions.push(DevicePromotion {
                row: logged.row,
                expert: logged.expert,
                slot: offer.slot,
                ticket: ticket_from_word(logged.word, near),
                ahead: logged.ahead,
            });
        }
        // Both are copies over the link.
        self.ahead_uncounted += ahead + carried;
        if let Ok(mut s) = self.stats.lock() {
            s.ring_taken += taken;
            s.victims_claimed += claimed;
            s.victims_skipped += skipped;
            s.evictions += claimed;
            s.ahead_claims += ahead;
            s.carry_claims += carried;
            s.ahead_bytes += ahead_bytes;
            s.ahead_pad_claims += ahead_pad;
        }
        Ok(())
    }

    /// Point every device-promoted expert whose invocation has completed — all
    /// of them when `idle` (the device has been synchronized) — at its slot.
    pub(super) fn land_device_promotions(&mut self, idle: bool) -> Result<()> {
        let Device::Cuda(cd) = self.device.clone() else {
            return Ok(());
        };
        let mut i = 0;
        let mut landed = 0usize;
        while i < self.device_promotions.len() {
            let p = &self.device_promotions[i];
            let complete = idle || self.clock.reclaimable(p.ticket);
            if !lands(complete, p.ahead, || self.fault.is_set()) {
                i += 1;
                continue;
            }
            let p = self.device_promotions.swap_remove(i);
            self.promoting.remove(&(p.row, p.expert));
            // Promoted twice — by the GPU on two invocations: nothing ever
            // named this slot, so it is free.
            if self
                .inner
                .key_to_slot
                .get(&(p.row, p.expert))
                .is_some_and(|&s| self.inner.slots[s].is_some())
            {
                if let Some(ring) = &self.promo_ring {
                    ring.clear_mark(p.row, p.expert);
                }
                self.inner.put_free(p.slot);
                continue;
            }
            let base = self.inner.slot_base(p.slot);
            // SAFETY: the invocation whose workers filled the slot has completed.
            let view = unsafe { build_slot_view(&self.layer_geometries[p.row], &cd, base)? };
            self.inner.install(p.slot, p.row, p.expert, view);
            self.residency()?
                .set_vram(p.row, p.expert, Some((p.slot, base)));
            // The mark only after the VRAM entry is published: a bucketize that
            // finds the expert unmarked and cold may hand it to the CPU pool,
            // which can read only host memory, so no VRAM publish may follow
            // an unmarked cold read (`moe_bucketize_cpu.cuh`).
            if let Some(ring) = &self.promo_ring {
                ring.clear_mark(p.row, p.expert);
            }
            landed += 1;
        }
        if landed > 0 {
            if let Ok(mut s) = self.stats.lock() {
                s.worker_promotions += landed;
            }
        }
        Ok(())
    }

    /// A forward faulted (`fault`) and the device has been synchronized: its
    /// workers gave demand items up, and a given-up item's promotion slot was
    /// never written. Collect what the device took, and drop every demand
    /// promotion still to land — its slot back to the zone, its mark cleared —
    /// instead of installing bytes that were never copied. Every demand claim
    /// made since the fault is still pending (`land_device_promotions` holds
    /// them while the word is set). Read-ahead claims stay: they never wait on
    /// a cold expert, and the device has published them. `near` is the ticket
    /// last served.
    pub(crate) fn drop_demand_promotions(&mut self, near: u64) -> Result<()> {
        self.collect_device_promotions(near)?;
        self.drop_pending_demand();
        Ok(())
    }

    /// Drop every demand promotion still to land — its slot back to the zone,
    /// its mark cleared — keeping the read-ahead claims.
    pub(super) fn drop_pending_demand(&mut self) {
        let (ahead, demand) = split_ahead(std::mem::take(&mut self.device_promotions));
        self.device_promotions = ahead;
        for p in demand {
            self.promoting.remove(&(p.row, p.expert));
            self.regret.withdraw_promotion(p.row, p.expert);
            if let Some(ring) = &self.promo_ring {
                ring.clear_mark(p.row, p.expert);
            }
            self.inner.put_free(p.slot);
        }
    }
}
