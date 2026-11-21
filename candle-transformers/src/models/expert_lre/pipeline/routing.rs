//! Serves one routed layer: reads its summary, scores it, promotes its misses and
//! prefetches for the layers ahead.
//!
//! The thread lands what the GPU has finished and reads each routing summary
//! while the GPU carries on, so nothing on the device waits for it. Only decode
//! routing judges predictions and claims. A prompt launch routes most of a row,
//! and would teach the statistics the wrong thing. Whether to read ahead is
//! decided at the end of a pass from the link time decode leaves free. A
//! prefill-width launch does not read ahead, because its own misses use the link.

use super::*;

impl PipelineState {
    /// Serve one routed layer: read its summary, score it, promote its misses,
    /// and prefetch ahead. Computes nothing and gates nothing — the layer's
    /// expert chain was enqueued without it.
    pub(crate) fn process_routed(&mut self, msg: RoutedLayer) -> Result<()> {
        let row = msg.row;

        // ── A new pass: the forward thread started its rows over ──
        if self.pass != Some(msg.pass) {
            if self.pass.is_some() {
                self.transition_matrix.reset_pass();
                self.set_read_ahead_window();
                self.set_eviction_order();
                self.decide_read_ahead();
            }
            self.pass = Some(msg.pass);
            self.last_summary = None;
        }

        // ── The routing summary ──
        let t = profile_now();
        // While it waits, land what the GPU has finished promoting, so the
        // residency this thread ranks victims and predictions on is current,
        // and list what the stager has staged ahead while a launch can still
        // read the listing.
        self.list_landed()?;
        let ring = self.ring.clone();
        // Already written when the wait begins: the summary sat waiting for
        // this thread, so its arrival times the thread, not the device.
        let late = ring.ready(msg.slot, msg.summary_word);
        ring.wait(msg.slot, msg.summary_word, || {
            self.land_device_promotions(false)?;
            self.list_landed()
        })?;
        let arrived = Instant::now();
        if let Some((at, prev)) = self.last_summary {
            if row == prev + 1 {
                self.layer_intervals
                    .push(arrived.duration_since(at).as_secs_f64());
            }
        }
        self.last_summary = Some((arrived, row));
        self.clock.observe(msg.ticket);
        let lag = self.clock.latest_started().saturating_sub(msg.ticket);
        self.profile.record("pipe_routed_wait", t);
        // SAFETY: `wait` returned for this invocation, and the forward thread
        // does not rewrite the slot — or its vote slot — until `routed_served`
        // passes this ticket.
        let summary: Vec<u32> = unsafe { self.ring.read(msg.slot) }.to_vec();
        let votes =
            (msg.voted_hops > 0).then(|| unsafe { self.votes.read(msg.slot, msg.voted_hops) });
        self.routed_served.store(msg.ticket, Ordering::Release);

        let t = profile_now();
        self.collect_device_promotions(msg.ticket)?;
        self.land_device_promotions(false)?;
        self.release_ahead_pins(false)?;

        let mut expert_ids: Vec<usize> = Vec::new();
        let mut decode: HashSet<usize> = HashSet::new();
        let (mut pinned, mut cold, mut cpu) = (0usize, 0usize, 0usize);
        for (e, &w) in summary.iter().enumerate() {
            if w & SUMMARY_COUNT == 0 {
                continue;
            }
            expert_ids.push(e);
            if w & SUMMARY_DECODE != 0 {
                decode.insert(e);
            }
            pinned += usize::from(w & SUMMARY_PINNED != 0);
            cold += usize::from(w & SUMMARY_COLD != 0);
            // The pinned misses the pool served; a cold one it served is
            // counted cold — it waited on the pack either way.
            cpu += usize::from(w & SUMMARY_CPU != 0 && w & SUMMARY_COLD == 0);
        }
        let routing = RowRouting {
            pass: msg.pass,
            row,
            arrived,
            prefill: msg.prefill_width,
            routed: expert_ids.len(),
            pinned,
            cold,
            late,
        };
        if let Some(trace) = self.layer_trace.visit(routing, self.num_moe_layers) {
            publish_layer_trace(trace);
        }
        if let Ok(mut t) = self.trace.lock() {
            let routed: Vec<(usize, u32, bool)> = expert_ids
                .iter()
                .map(|&e| (e, summary[e] & SUMMARY_COUNT, decode.contains(&e)))
                .collect();
            t.push(row, &routed, row < self.inner.pinned_layers);
        }
        // Claims are judged, and the row's visit recorded, by decode routing
        // alone: a prompt launch routes most of the row, so it would count
        // nearly every victim routed again and mark nearly every resident just
        // hit — the regret rates that order eviction (`set_eviction_order`)
        // and the recency passes would learn the prompt, not decode.
        if !msg.prefill_width {
            let v = self.regret.judge(row, msg.ticket, &expert_ids);
            if let Ok(mut s) = self.stats.lock() {
                for k in 0..2 {
                    s.claim_evicted[k] += v.evicted[k];
                    s.claim_regretted[k] += v.regretted[k];
                    s.claim_promoted[k] += v.promoted[k];
                    s.claim_paid[k] += v.paid[k];
                }
                for t in 0..TAGS {
                    s.victim_tag_evicted[t] += v.tag_evicted[t];
                    s.victim_tag_regretted[t] += v.tag_regretted[t];
                }
            }
            self.inner.mark_visit(row, &expert_ids);
        }

        // ── Read-ahead precision: of the experts read ahead for this row, the
        // ones it routed ──
        let spec_for_layer: Vec<usize> = self
            .speculative_loads
            .iter()
            .filter(|&&(l, _)| l == row)
            .map(|&(_, e)| e)
            .collect();
        if !spec_for_layer.is_empty() {
            let hits = spec_for_layer
                .iter()
                .filter(|e| expert_ids.binary_search(e).is_ok())
                .count();
            if let Ok(mut s) = self.stats.lock() {
                s.predicted_total += spec_for_layer.len();
                s.predicted_hits += hits;
            }
            for &eid in &spec_for_layer {
                if expert_ids.binary_search(&eid).is_ok() {
                    self.inner.record_prediction_hit(row, eid);
                }
            }
        }
        self.speculative_loads.retain(|&(l, _)| l != row);

        // ── The latest predictions for this row, judged — by decode routing
        // only, for the cells learn what decode's next rows route; a prompt
        // reaching the row first discards them ──
        for hop in 1..=HOPS {
            let Some(p) = self.pending[row][hop - 1].take() else {
                continue;
            };
            if msg.prefill_width {
                continue;
            }
            let routed = |e: &usize| expert_ids.binary_search(e).is_ok();
            let (mut union_hits, mut markov, mut recent) = (0, (0, 0), (0, 0));
            for &(e, cell) in &p.candidates {
                let hit = routed(&e);
                self.cells.record(hop, cell, hit);
                union_hits += usize::from(hit);
                let (_, m, r) = cell_axes(cell);
                if m > 0 {
                    markov.0 += 1;
                    markov.1 += usize::from(hit);
                }
                if r {
                    recent.0 += 1;
                    recent.1 += usize::from(hit);
                }
            }
            let selected_hits = p.selected.iter().filter(|e| routed(e)).count();
            let copy_routed = expert_ids
                .iter()
                .filter(|&&e| !p.held.get(e).copied().unwrap_or(false))
                .count();
            if let Ok(mut s) = self.stats.lock() {
                let h = hop - 1;
                if let Some(votes) = &p.votes {
                    s.lookahead_judged[h] += votes.len();
                    s.lookahead_hits[h] += votes.iter().filter(|e| routed(e)).count();
                    s.lookahead_routed[h] += expert_ids.len();
                }
                s.copy_routed[h] += copy_routed;
                s.union_hits[h] += union_hits;
                s.markov_judged[h] += markov.0;
                s.markov_hits[h] += markov.1;
                s.recent_judged[h] += recent.0;
                s.recent_hits[h] += recent.1;
                s.selected_judged[h] += p.selected.len();
                s.selected_hits[h] += selected_hits;
            }
        }
        if let Ok(mut s) = self.stats.lock() {
            s.cells = self.cells;
        }

        // The predictor learns from decode routing alone: a prompt launch
        // routes most of the row, so its transitions are near-uniform and
        // would wash out the sharper statistics decode's steps carry. A cache
        // holding every expert never predicts (`predict_rows` below), so it
        // keeps no tables — at a wide decode step their update is hundreds of
        // thousands of scattered writes an invocation, on the thread every
        // host meeting point waits for.
        if !msg.prefill_width && !self.all_resident {
            let mut decode_ids: Vec<usize> = decode.iter().copied().collect();
            decode_ids.sort_unstable();
            self.transition_matrix.observe(row, &decode_ids);
        }

        // ── Hits and misses, by this thread's own bookkeeping ──
        let mut hits = 0usize;
        let mut misses = 0usize;
        // Misses that took no ring slot and are not otherwise being promoted:
        // their workers computed them from scratch, and the next launch that
        // routes them claims one. Those a decode row routed are counted apart —
        // a decode row co-batched into a sweep defers its claims this way.
        let (mut unslotted, mut decode_unslotted) = (0usize, 0usize);
        // Whether prompt-only experts are promoted (`PROMPT_ROOM`).
        let prompt_room =
            self.inner.zone.limit() * PROMPT_ROOM.1 >= self.inner.total_experts() * PROMPT_ROOM.0;
        for &e in &expert_ids {
            match self.inner.key_to_slot.get(&(row, e)) {
                Some(&slot) if self.inner.slots[slot].is_some() => {
                    self.inner.promote(slot);
                    // A decode row's reuse of this expert is near-certain from
                    // one step to the next; a prefill row's is close to zero, so
                    // the two must not bid for residency on equal footing.
                    if decode.contains(&e) {
                        self.inner.record_hit(row, e);
                    } else {
                        self.inner.record_prefill_hit(row, e);
                    }
                    hits += 1;
                }
                _ => {
                    misses += 1;
                    // The GPU promotes it if bucketize had a ring slot for it.
                    // A decode miss holds its slot to the next step, which
                    // routes to it again; a prefill-only miss earns no benefit
                    // of the doubt, so its score biases it toward the next
                    // eviction.
                    let routed_by_decode = decode.contains(&e);
                    if routed_by_decode {
                        self.inner.record_decode_miss(row, e);
                    } else {
                        self.inner.record_prefill_elevate(row, e);
                    }
                    if !self.promoting.contains(&(row, e)) {
                        unslotted += 1;
                        decode_unslotted += usize::from(routed_by_decode);
                    }
                }
            }
        }
        if let Ok(mut s) = self.stats.lock() {
            s.routed_messages += 1;
            s.pipeline_lag += lag;
            s.ring_unslotted += unslotted;
            s.decode_unslotted += decode_unslotted;
            s.expert_hits += hits;
            s.expert_misses += misses;
            s.worker_pinned += pinned;
            s.worker_cold += cold;
            s.worker_cpu += cpu;
        }
        // ── Whether reading ahead pays (`read_ahead_gate`), judged on decode
        // invocations — the width the look-ahead runs at. The misses bucketize
        // found cold wait on a pack read, and those it gave the CPU pool cross
        // no link; the rest, and the read-ahead claims, are pinned copies over
        // the link. Counting only those is what lets a CPU-heavy split OPEN
        // read-ahead on the link time it freed ──
        if !msg.prefill_width {
            let copies =
                misses.saturating_sub(cold + cpu) + std::mem::take(&mut self.ahead_uncounted);
            let cold_secs = self
                .pack_read_secs
                .unwrap_or(self.image_bytes as f64 / self.link_rate.max(1.0));
            let us = link_us(copies, cold, self.image_bytes, self.link_rate, cold_secs);
            // Decided at the pass's end (`decide_read_ahead`), not here.
            self.link_time.observe(us);
        }
        self.profile.record("pipe_classify", t);

        if !self.all_resident {
            // Stock every empty slot, and past them what the next
            // `RING_HORIZON` rows will miss — their non-resident experts, at the
            // share of experts this row routed that take ring slots — since this
            // thread's refill trails the GPU's bucketize; never below the floor.
            // Without room, prompt traffic gets only stock past decode's
            // reserve, and a launch claiming more than the predicted demand is a
            // sweep (`ring_stock`). Stock the demand no longer wants is taken
            // back (`trim_ring`).
            if let Some(ring) = &self.promo_ring {
                let empty_slots = self.inner.free_len() + self.empty_offers(ring);
                let width = self.inner.experts_per_layer.max(1);
                let absent: usize = (row + 1..(row + 1 + RING_HORIZON).min(self.num_moe_layers))
                    .map(|r| {
                        (0..width)
                            .filter(|&e| !self.inner.key_to_slot.contains_key(&(r, e)))
                            .count()
                    })
                    .sum();
                let ahead = |promoted: usize| (absent * promoted).div_ceil(width);
                let stock = ring_stock(
                    ahead(decode.len()),
                    ahead(expert_ids.len()),
                    prompt_room,
                    empty_slots,
                    self.window as usize,
                    ring.cap(),
                );
                self.ring_target = stock.target;
                ring.set_reserve(stock.reserve);
                ring.set_sweep(stock.sweep);
            }

            // ── Stock the ring for the layers to come ──
            let t = profile_now();
            self.trim_ring()?;
            self.refill_ring(row)?;
            self.profile.record("pipe_ring", t);

            // ── Predictions for read-ahead, only where the link has room ──
            // A prefill-width launch's misses are pulled by enough workers to
            // saturate the link, so reading ahead for it takes bandwidth from
            // them and saves nothing — its launches' own misses use up their
            // window anyway (`read_ahead`). Measured with copy-engine prefetch
            // on Qwen3.8-Flash-Next ×16 prefill (RTX 3090, a working set 3× the
            // zone): 40,823 promotions, 52.6 GiB, nearly every one late —
            // prefill 845 t/s with them, 1,075 without. A decode-width launch
            // leaves the link mostly idle, and there reading ahead pays: ×16
            // decode 237 t/s with it, 220 without.
            //
            // A narrower launch that routes prompt-only experts — any expert
            // no decode-scored row routed — predicts only without room
            // (`PROMPT_ROOM`); one whose experts are all decode's predicts as a
            // decode launch does. With room the ring already promotes the
            // prompt's misses and reading ahead only competes with them:
            // Qwen3.6-35B-A3B's one-sequence prompts ran 4–9% slower with it.
            // Without room, it is the one way a prompt's experts reach VRAM
            // ahead of the workers: Qwen3.8-Flash-Next's warm one-sequence
            // prompt ran 4% faster with it.
            let t = profile_now();
            if !msg.prefill_width && (decode.len() == expert_ids.len() || !prompt_room) {
                let wide = expert_ids.len() * 2 >= self.inner.experts_per_layer.max(1);
                // Invocations the device has begun past this one, as of now.
                let begun_past = self.clock.latest_started().saturating_sub(msg.ticket);
                self.predict_rows(row, &expert_ids, votes.as_ref(), wide, begun_past)?;
            } else if msg.prefill_width {
                if let Some(v) = votes.as_ref() {
                    self.stage_voted(row, v)?;
                }
            }
            self.profile.record("pipe_prefetch", t);
        }
        // Scores age once per pass that routed a decode-scored row, at its last
        // MoE layer. That is nearly every pass — a prompt's last row is scored
        // as decode (`residency_rows`) — so the recency term counts passes; a
        // launch with no decode-scored row at all leaves the scores as they are.
        if row + 1 == self.num_moe_layers && !decode.is_empty() {
            self.inner.decay_scores(DECODE_RECENCY_DECAY);
        }
        self.publish_gauges();
        Ok(())
    }

    /// The zone's gauges, refreshed each routed layer and after every boundary
    /// move.
    pub(in super::super) fn publish_gauges(&self) {
        let slot_bytes = self.inner.zone.slot_bytes();
        let occupied = self.inner.num_slots() - self.inner.free_len();
        if let Ok(mut s) = self.stats.lock() {
            s.resident_vram_bytes = occupied * slot_bytes;
            s.zone_cedeable_bytes = self
                .inner
                .zone
                .capacity()
                .saturating_sub(self.inner.zone.min_capacity())
                * slot_bytes;
            s.zone_bytes = self.inner.zone.capacity() * slot_bytes;
            s.zone_min_bytes = self.inner.zone.min_capacity() * slot_bytes;
            s.zone_max_bytes = self.inner.zone.limit() * slot_bytes;
            s.expert_slot_bytes = slot_bytes;
            s.link_bytes_per_sec = self.link_rate;
        }
    }
}
