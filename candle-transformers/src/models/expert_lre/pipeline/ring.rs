//! Keeps the promotion ring stocked with free or evictable VRAM slots for the
//! GPU to claim when it misses an expert.
//!
//! Every empty slot is stocked, because a miss that finds one is promoted by the
//! copy its worker made anyway. Lazy victims are stocked to the prediction. A
//! launch that claims more experts than the next rows were predicted to want is
//! a sweep, and claims nothing, so a prompt passing over the table cannot evict
//! the experts decode reuses. Offers the device does not take are returned to
//! the zone, and the order of evictions follows each class's measured regret.

use super::*;

/// What the promotion ring is kept at: how many offers it holds, the reserve
/// below which a prompt-only expert takes no slot, and the sweep word — the
/// most claiming experts a launch may have and still claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RingStock {
    pub(super) target: usize,
    pub(super) reserve: u32,
    pub(super) sweep: u32,
}

/// The promotion ring's stock, from the stock the next rows' decode misses
/// want (`decode_ahead`), the stock all their misses want (`all_ahead`), and
/// the zone's **empty** slots — the free list plus the empty offers already
/// standing in the ring.
///
/// **Every empty slot is stocked.** A ring slot is where a worker's copy of a
/// missed expert lands and is computed from (`moe_live_worker`), so a miss that
/// finds one is promoted for the copy the layer made anyway. An empty slot
/// evicts nothing to stand there, so the ring holds all of them (to half its
/// capacity) whatever the prediction says; the prediction sizes only the stock
/// of lazy victims, which a claim evicts. Stocked to the prediction alone, a
/// 16 GB card's zone stayed ~1,400 slots short of full for a whole
/// Qwen3.6-35B-A3B run while its misses crossed the link and were thrown away.
///
/// With room (`PROMPT_ROOM`) nothing is reserved, so a prompt-only miss takes
/// the stock like a decode one — in a launch that is not a sweep. Without room
/// decode's own target is the reserve, so the stock a prompt-only expert may
/// take is only what stands past it: empty slots, never a victim decode would
/// not have made.
///
/// **The sweep word is the prediction, never the empties.** A launch whose
/// claiming experts outnumber what the next rows were predicted to want is a
/// prompt passing over the table, and claims nothing: its misses would evict
/// one-shot experts into the zone decode reuses. Counting empty slots into it
/// would let a cold prompt claim victims as soon as the zone had a hole. That
/// holds with room too: a prefill launch's misses run to thousands against a
/// prediction capped at half the ring, so even a model whose zone holds most of
/// its experts promotes a prompt's misses only from launches narrow enough to
/// fit the prediction — stencils, short prompts, the rows a decode step
/// verifies.
///
/// **Read-ahead stocks on top.** A launch reads ahead with up to `window` offers
/// past its own misses, so the stock grows by the window — in the target only:
/// the reserve and the sweep word decide what misses may claim, and read-ahead
/// spends only the stock above the reserve (`moe_bucketize.cu`). When decode's
/// own target reaches half the ring, the target is clamped there too and equals
/// the reserve: read-ahead then gets only the zone's empty slots, since the
/// ring has no room left to stock a window past what decode's misses want.
pub(super) fn ring_stock(
    decode_ahead: usize,
    all_ahead: usize,
    prompt_room: bool,
    empty_slots: usize,
    window: usize,
    cap: usize,
) -> RingStock {
    let target = |ahead: usize| (ahead + ahead / 4).clamp(RING_TARGET, cap / 2);
    let empties = empty_slots.min(cap / 2);
    let stocked = |t: usize| (t + window).min(cap / 2).max(empties);
    if prompt_room {
        let all = target(all_ahead);
        return RingStock {
            target: stocked(all),
            reserve: 0,
            sweep: all as u32,
        };
    }
    let decode = target(decode_ahead);
    RingStock {
        target: stocked(decode),
        reserve: decode as u32,
        sweep: decode as u32,
    }
}

/// What stands at one ring index: a zone slot, and — when it was offered as a
/// lazy victim — the resident expert it still holds and the fallback held for
/// it as long as the offer stands (a pad slot pinned, a cold one marked).
#[derive(Clone, Copy, Debug)]
pub(super) struct Offer {
    pub(super) slot: usize,
    pub(super) victim: Option<((usize, usize), Fallback)>,
}

/// A lazy victim chosen for an offer: its slot, its expert, what the device is
/// told, and the fallback held for it.
pub(super) struct Chosen {
    pub(super) slot: usize,
    pub(super) key: (usize, usize),
    pub(super) victim: Victim,
    pub(super) fallback: Fallback,
}

/// The share of the pad lazy victims may hold pinned (`1 / PAD_PIN_SHARE`). A
/// pinned pad slot cannot be evicted, and the stager stages every cold miss
/// into the pad: offers holding most of it would leave a cold wait nowhere to
/// land.
pub(super) const PAD_PIN_SHARE: usize = 4;

impl PipelineState {
    /// Empty offers standing in the ring, untaken.
    pub(super) fn empty_offers(&self, ring: &PromotionRing) -> usize {
        let (head, tail) = (ring.head(), ring.tail());
        let mut i = head;
        let mut n = 0usize;
        while i != tail {
            if self.ring_slots[i as usize % ring.cap()].is_some_and(|o| o.victim.is_none()) {
                n += 1;
            }
            i = i.wrapping_add(1);
        }
        n
    }

    /// Up to `n` lazy victims, ranked against `row`: resident experts whose
    /// slot a miss may claim, each with the entries the device retargets it
    /// to (`Residency::displaced_entries`), never a slot already offered or
    /// one being promoted into. The fallback is held here for the life of the
    /// offer: a pad slot pinned, while fewer than a `PAD_PIN_SHARE`th of the
    /// pad is pinned so (past that, a pad-backed expert is passed over); a cold
    /// one marked, so the stager reads a cold summary of it as the device's
    /// eviction. Rows with an invocation enqueued are offered last: the wave is
    /// about to read their experts.
    ///
    /// The ranking runs without the residency lock — it is a scan of the whole
    /// zone, and the stager publishes pad copies under that lock while cold
    /// workers wait on them — so it asks for twice what it needs, and the lock
    /// is held only to confirm and hold the ranked slots' fallbacks, in rank
    /// order, keeping the first `n`.
    pub(super) fn lazy_victims(&mut self, row: usize, n: usize) -> Result<Vec<Chosen>> {
        let upcoming: Vec<bool> = (0..self.num_moe_layers)
            .map(|r| self.clock.upcoming(r))
            .collect();
        let keys = &self.inner.slot_to_key;
        let offered = &self.offered;
        let promoting = &self.promoting;
        let width = self.inner.experts_per_layer;
        let candidate = |slot: usize| {
            !offered.contains(&slot)
                && keys[slot].is_some_and(|(r, e)| !promoting.contains(&(r, e)))
        };
        let ask = 2 * n;
        let mut slots = self
            .inner
            .rank_victims(row, ask, |slot, layer| !upcoming[layer] && candidate(slot));
        if slots.len() < ask {
            let more = self
                .inner
                .rank_victims(row, ask - slots.len(), |slot, layer| {
                    upcoming[layer] && candidate(slot)
                });
            slots.extend(more);
        }
        let pad_cap = self.stats.lock().map_or(0, |s| s.pad_slots / PAD_PIN_SHARE);
        let mut pinned = self.pad_pinned_offers;
        let mut chosen = Vec::with_capacity(n);
        let mut places = self
            .residency
            .lock()
            .map_err(|_| candle::Error::Msg("expert pipeline: residency poisoned".into()))?;
        for slot in slots {
            if chosen.len() == n {
                break;
            }
            let Some((r, e)) = keys[slot] else { continue };
            let Some((retarget, fallback)) = places.displaced_entries(r, e) else {
                continue;
            };
            match fallback {
                Fallback::Pad if pinned >= pad_cap => continue,
                Fallback::Pad => {
                    places.pin_pad(r, e, 1);
                    pinned += 1;
                }
                Fallback::Cold => places.set_offered_cold(r, e, true),
            }
            chosen.push(Chosen {
                slot,
                key: (r, e),
                victim: Victim {
                    index: (r * width + e) as u64,
                    retarget,
                },
                fallback,
            });
        }
        drop(places);
        self.pad_pinned_offers = pinned;
        Ok(chosen)
    }

    /// An offer is over — claimed, skipped or taken back: release the fallback
    /// it held.
    pub(super) fn release_offer(&mut self, offer: &Offer) -> Result<()> {
        match offer.victim {
            Some(((r, e), Fallback::Pad)) => {
                self.residency()?.pin_pad(r, e, -1);
                self.pad_pinned_offers -= 1;
            }
            Some(((r, e), Fallback::Cold)) => self.residency()?.set_offered_cold(r, e, false),
            None => {}
        }
        Ok(())
    }

    /// Keep `ring_target` offers standing in the promotion ring: every empty
    /// slot first, then lazy victims, which stay resident until a miss claims
    /// them.
    pub(super) fn refill_ring(&mut self, row: usize) -> Result<()> {
        let Some(ring) = self.promo_ring.clone() else {
            return Ok(());
        };
        let tail = ring.tail();
        let stocked = tail.wrapping_sub(ring.head()) as usize;
        let room = ring.cap() - tail.wrapping_sub(self.ring_taken) as usize;
        let want = self.ring_target.saturating_sub(stocked).min(room);
        if want == 0 {
            return Ok(());
        }
        let mut offers: Vec<(Offer, Option<Victim>)> = Vec::with_capacity(want);
        while offers.len() < want {
            match self.inner.take_free() {
                Some(slot) => offers.push((Offer { slot, victim: None }, None)),
                None => break,
            }
        }
        if offers.len() < want {
            for c in self.lazy_victims(row, want - offers.len())? {
                offers.push((
                    Offer {
                        slot: c.slot,
                        victim: Some((c.key, c.fallback)),
                    },
                    Some(c.victim),
                ));
            }
        }
        for (offer, victim) in offers {
            let at = ring.tail() as usize % ring.cap();
            self.ring_slots[at] = Some(offer);
            self.offered.insert(offer.slot);
            ring.offer(self.inner.slot_base(offer.slot), victim);
        }
        Ok(())
    }

    /// Return an offer the device never took: an empty slot to the zone's free
    /// list, a lazy victim to plain residency — its expert never left — with
    /// its fallback released.
    pub(super) fn take_back(&mut self, offer: Offer) -> Result<()> {
        self.offered.remove(&offer.slot);
        self.release_offer(&offer)?;
        if offer.victim.is_none() {
            self.inner.put_free(offer.slot);
        }
        Ok(())
    }

    /// Take back the offers past `ring_target` — stock the demand no longer
    /// wants.
    ///
    /// Only when no bucketize can be reading the ring: under the pass lock (no
    /// invocation can begin) with every invocation begun observed (its
    /// bucketize has run). Between a decode step and the next that holds; deep
    /// in a run-ahead it does not, and the stock waits.
    pub(super) fn trim_ring(&mut self) -> Result<()> {
        let Some(ring) = self.promo_ring.clone() else {
            return Ok(());
        };
        let pass_state = self.pass_state.clone();
        let p = pass_state
            .lock()
            .map_err(|_| candle::Error::Msg("expert pipeline: pass state poisoned".into()))?;
        if p.reserved != self.clock.observed() {
            return Ok(());
        }
        let (head, tail) = (ring.head(), ring.tail());
        let stocked = tail.wrapping_sub(head) as usize;
        if stocked <= self.ring_target {
            return Ok(());
        }
        let keep = head.wrapping_add(self.ring_target as u32);
        ring.withdraw_to(keep);
        drop(p);
        let mut i = keep;
        while i != tail {
            if let Some(offer) = self.ring_slots[i as usize % ring.cap()].take() {
                self.take_back(offer)?;
            }
            i = i.wrapping_add(1);
        }
        Ok(())
    }

    /// Take every offer out of the promotion ring — before a boundary move,
    /// with the device synchronized: land what the GPU filled, take back what
    /// it never took, and release the pad pins read-ahead's old listings still
    /// held, whose readers are all done. With a fault not yet taken, its demand
    /// promotions are dropped here rather than left pending: the move may
    /// retract the zone below their slots.
    pub(crate) fn drain_ring(&mut self, near: u64) -> Result<()> {
        let Some(ring) = self.promo_ring.clone() else {
            return Ok(());
        };
        self.collect_device_promotions(near)?;
        if self.fault.is_set() {
            self.drop_pending_demand();
        }
        self.land_device_promotions(true)?;
        ring.withdraw();
        let untaken: Vec<Offer> = self
            .ring_slots
            .iter_mut()
            .filter_map(Option::take)
            .collect();
        for offer in untaken {
            self.take_back(offer)?;
        }
        self.ring_taken = ring.head();
        self.release_ahead_pins(true)
    }

    /// What was known of `row`'s expert `e` as a claim evicts it: the best
    /// measured probability a live prediction for the row gave it (bands below
    /// and from one half), whether the row routed it on its last visit,
    /// whether the row lies just ahead of the one last served, and whether a
    /// warm copy backs it. "Last visit" and "last served" are as this thread
    /// has processed them: while it trails the GPU, a claim from a later
    /// invocation is tagged against the visit before.
    pub(super) fn victim_tag(&self, row: usize, e: usize) -> VictimTag {
        let best = self.pending[row]
            .iter()
            .enumerate()
            .filter_map(|(h, p)| Some((h + 1, p.as_ref()?)))
            .flat_map(|(hop, p)| {
                p.held_named
                    .iter()
                    .filter(move |&&(x, _)| x == e)
                    .map(move |&(_, cell)| self.cells.probability(hop, cell))
            })
            .fold(None, |m: Option<f64>, p| Some(m.map_or(p, |m| m.max(p))));
        let n = self.num_moe_layers;
        let served = self.last_summary.map_or(row, |(_, r)| r);
        VictimTag {
            predicted: prediction_band(best),
            hit_last: self.inner.hit_last_visit(row, e),
            ahead: just_ahead(row, served, n),
            warm: self.inner.is_warm_backed(row, e),
        }
    }

    /// Order the eviction passes by each class's expected miss cost — its
    /// running regret rate (`regret`) times what restoring one costs, priced by
    /// its backing (`SplitParams::refill_secs`, `docs/cpu_expert_compute.md`
    /// §5.11): for a warm-backed expert the copy over the link (one image at
    /// the measured link rate), or the CPU pool's path where that is cheaper;
    /// for a pack-only one the pack read the layer then waits on (the stager's
    /// mean, latest measured), weighted by `cold_bias`, plus that copy. Before
    /// the first pack read is measured the default order stands.
    pub(super) fn set_eviction_order(&mut self) {
        if let Ok(s) = self.stats.lock() {
            if let Some(read) = s.pack_reads.mean_secs() {
                self.pack_read_secs = Some(read);
            }
        }
        let Some(read) = self.pack_read_secs else {
            return;
        };
        if self.link_rate <= 0.0 {
            return;
        }
        let copy = self.image_bytes as f64 / self.link_rate;
        let (warm, pack) = cpu_params()
            .lock()
            .map(|p| p.refill_secs(copy, read))
            .unwrap_or((copy, read + copy));
        let cost = eviction_costs(self.regret.class_rates(), warm, pack);
        self.inner.set_eviction_costs(cost);
        if let Ok(mut s) = self.stats.lock() {
            s.eviction_cost_us = cost.map(|c| c * 1e6);
        }
    }
}
