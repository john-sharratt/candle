//! Writes each row's read-ahead listing of experts the coming launches will read,
//! and stages the cold ones for the stager.
//!
//! A listing names experts with a pinned copy in the pad, and each pad slot is
//! pinned for the listing so the stager cannot reuse it under the reader. A
//! listing is replaced only while the launch that reads it has not begun. An
//! expert with no copy the device can read goes to the stager, and is listed
//! from the pad once it lands. A prefill launch stages the experts its rows
//! voted for instead of listing them, since its own misses already fill the
//! link.

use super::*;

/// The share of the pad read-ahead's listings may hold pinned
/// (`1 / AHEAD_PIN_SHARE`), beside the lazy victims' share: together they leave
/// the stager most of the pad for cold demand rows.
pub(super) const AHEAD_PIN_SHARE: usize = 8;

impl PipelineState {
    /// Whether a launch that reads `row`'s listing has yet to begin
    /// (`read_ahead::listing_readable`): the served row is the last whose
    /// summary this thread read, and the served ticket is where the forward
    /// thread's ring hold stands — the previous pass's last between passes.
    pub(super) fn listing_readable(&self, row: usize) -> bool {
        let served_row = self.last_summary.map(|(_, r)| r);
        let served_ticket = self.routed_served.load(Ordering::Acquire);
        listing_readable(row, served_row, served_ticket, |r| self.clock.begun(r))
    }

    /// Write `row`'s read-ahead listing from its latest prediction: the
    /// predicted experts not in VRAM and not being promoted that have a pinned
    /// copy, each with its slot image as the vetted source, replacing the last
    /// listing — only while a launch that reads it has yet to begin
    /// (`listing_readable`), since a launch at `r` reads the lists of `r + 2`
    /// on. Every pinned copy is a pad slot, listed only with the slot pinned
    /// for the listing (`ahead_pins`), since the stager may otherwise reuse it
    /// under another row's launch. With `stage`, the cold ones (no
    /// device-readable copy) go to the stager to stage into the pad, whether
    /// or not the row is listed now — staged in time for its own workers at the
    /// least, and listed from the pad as each lands (`list_landed`). Returns
    /// the experts listed.
    pub(super) fn list_row(&mut self, row: usize, stage: bool) -> Result<Vec<usize>> {
        let Some(ring) = self.promo_ring.clone() else {
            return Ok(Vec::new());
        };
        let readable = self.listing_readable(row);
        if !readable && !stage {
            return Ok(Vec::new());
        }
        let cap = self
            .stats
            .lock()
            .map_or(0, |s| s.pad_slots / AHEAD_PIN_SHARE);
        let mut to_stage = Vec::new();
        let mut planned = None;
        {
            let mut places = self
                .residency
                .lock()
                .map_err(|_| candle::Error::Msg("expert pipeline: residency poisoned".into()))?;
            let predicted = &self.predicted[row];
            let mut candidates = Vec::with_capacity(predicted.len().min(AHEAD_CAP));
            for &e in predicted {
                if self.is_held(row, e) {
                    continue;
                }
                let p = places.place(row, e);
                if p.entry() == 0 {
                    if stage {
                        to_stage.push(e);
                    }
                } else if p.vram.is_none() && candidates.len() < AHEAD_CAP {
                    candidates.push(Candidate {
                        expert: e,
                        image: p.entry(),
                        pad: p.pad.is_some(),
                    });
                }
            }
            if readable {
                // Pinned under the residency lock the candidates were read
                // under, so each pad slot listed is the one its entry names.
                let plan = self.ahead_pins.plan(row, &candidates, cap);
                for &e in &plan.pin {
                    places.pin_pad(row, e, 1);
                }
                planned = Some(plan.listing);
            }
        }
        let mut listed = Vec::new();
        if let Some(listing) = planned {
            ring.predict(row, &listing);
            self.ahead_pins.commit(self.clock.readers_key());
            if let Ok(mut s) = self.stats.lock() {
                s.ahead_pinned = self.ahead_pins.held();
            }
            listed = listing.into_iter().map(|(e, _)| e).collect();
        }
        if !to_stage.is_empty() {
            if let Ok(mut s) = self.stats.lock() {
                s.stage_requests += to_stage.len();
            }
            // A closed channel means the stager is gone; its guard has raised
            // the abort word, which the next routed layer reports.
            let _ = self.stager.send(StagerMsg::Stage {
                row,
                experts: to_stage,
            });
        }
        Ok(listed)
    }

    /// A prefill-width launch at `row` voted on the rows after it: stage every
    /// expert each of those rows' routers put in some token's top `k`
    /// (`voted_set`) and that the device cannot read yet, in expert order, so
    /// the drive reads the coming layers while this one computes.
    ///
    /// **The drive, not the link.** A prefill launch's own misses saturate the
    /// link, which is why it lists nothing for the device's read-ahead (above);
    /// staging moves a pack record into the pad, which only the drive pays for,
    /// and a prefill pass is bound by the drive. Ranking by the pad's scores
    /// instead measured ~1,000 unrouted reads a pass on Flash-Next's ×8 prefill
    /// and covered ~40% of the next row's cold set — those scores are decode's.
    pub(super) fn stage_voted(&mut self, row: usize, votes: &Votes) -> Result<()> {
        let width = self.inner.experts_per_layer;
        // Nearest row first: the stager serves its speculative queue in order.
        for hop in 1..=votes.hops(width) {
            let target = row + hop;
            if target >= self.num_moe_layers {
                break;
            }
            let voted = voted_set(votes.hop(hop, width).0);
            let cold: Vec<usize> = {
                let places = self.residency.lock().map_err(|_| {
                    candle::Error::Msg("expert pipeline: residency poisoned".into())
                })?;
                voted
                    .into_iter()
                    .filter(|&e| places.place(target, e).entry() == 0)
                    .collect()
            };
            if cold.is_empty() {
                continue;
            }
            if let Ok(mut s) = self.stats.lock() {
                s.stage_requests += cold.len();
            }
            // A closed channel means the stager is gone; its guard has raised
            // the abort word, which the next routed layer reports.
            let _ = self.stager.send(StagerMsg::Stage {
                row: target,
                experts: cold,
            });
        }
        Ok(())
    }

    /// The experts the stager has staged ahead since this was last called,
    /// each now in the pad: every row among them whose latest prediction still
    /// names the expert, and that a launch has yet to read, is listed again —
    /// from that prediction, so now with the staged expert from the pad.
    pub(super) fn list_landed(&mut self) -> Result<()> {
        let mut landed: Vec<(usize, usize)> = Vec::new();
        while let Ok(l) = self.landed_ahead.try_recv() {
            landed.push(l);
        }
        if landed.is_empty() {
            return Ok(());
        }
        let mut rows: Vec<usize> = landed.iter().map(|&(r, _)| r).collect();
        rows.sort_unstable();
        rows.dedup();
        let mut listed = 0usize;
        for row in rows {
            let staged: Vec<usize> = landed
                .iter()
                .filter(|&&(r, e)| r == row && self.predicted[row].contains(&e))
                .map(|&(_, e)| e)
                .collect();
            if staged.is_empty() {
                continue;
            }
            let now = self.list_row(row, false)?;
            listed += staged.iter().filter(|e| now.contains(e)).count();
        }
        if listed > 0 {
            if let Ok(mut s) = self.stats.lock() {
                s.staged_listed += listed;
            }
        }
        Ok(())
    }

    /// Release the pad pins read-ahead's listings dropped once every
    /// invocation that could have read them has finished — all of them when
    /// `idle` (the device has been synchronized).
    pub(super) fn release_ahead_pins(&mut self, idle: bool) -> Result<()> {
        let clock = &self.clock;
        let due = self.ahead_pins.due(|key| idle || clock.reclaimable(key));
        if !due.is_empty() {
            let mut places = self.residency()?;
            for (row, e) in due {
                places.pin_pad(row, e, -1);
            }
            drop(places);
            if let Ok(mut s) = self.stats.lock() {
                s.ahead_pinned = self.ahead_pins.held();
            }
        }
        Ok(())
    }
}
