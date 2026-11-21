//! Predicts which experts the rows ahead will route, and keeps each prediction
//! until its row's routing can judge it.
//!
//! Candidates come from the router look-ahead, from a Markov table keyed on the
//! current row's routing, and from the experts the zone recently let go of. They
//! are ranked by the evidence that has been right for each combination. A
//! prediction is kept whole so that the routing of its row can update the cells
//! from every candidate. A wrong prediction costs a pad slot or a ring offer and
//! the link time its window allowed, never correctness. A row the device has
//! already begun is skipped, because a read-ahead for it could not land in time.

use super::*;

/// One prediction for a row ahead, kept until that row's routing judges it: the
/// look-ahead's own list (for its precision and recall), every candidate any
/// predictor proposed that a copy could serve, with its cell (`blend`), the
/// list used, which of the row's experts were held — resident or being
/// promoted — when it was made, so the list is also judged on the experts a
/// copy could have served, and the held experts a predictor named, with their
/// cell: what an eviction of one of them knew it was throwing away (`regret`).
#[derive(Clone, Debug, Default)]
pub(super) struct Prediction {
    pub(super) votes: Option<Vec<usize>>,
    pub(super) candidates: Vec<(usize, usize)>,
    pub(super) selected: Vec<usize>,
    pub(super) held: Vec<bool>,
    pub(super) held_named: Vec<(usize, usize)>,
}

/// Markov candidates proposed per row ahead — past any cap, so the cells see
/// Markov's weaker evidence too.
pub(super) const MARKOV_CANDIDATES: usize = 64;

/// Recently routed candidates proposed per row ahead, for the same reason.
pub(super) const RECENT_CANDIDATES: usize = 64;

/// Experts predicted for a row within two of the served one: too near for a
/// launch to read ahead (`read_ahead::listing_readable`), so the prediction
/// only stages its cold experts into the pad for the row's own workers, and
/// the pad — one layer of slots, shared with every demand row — is what
/// bounds it. A row past that is listed for read-ahead as well and takes the
/// ring's list (`AHEAD_CAP`).
///
/// 48 so the look-ahead's margin picks (`LOOK_AHEAD_MARGIN`) reach the stager:
/// at 32 a verify wave's routed picks alone filled the list. See the margin for
/// what the pair measured.
pub(super) const STAGE_CAP: usize = 48;

/// The score a non-resident expert needs to be predicted for its row from
/// decode's own history (`scored_absent`): a decode routing credits 1.0 and a
/// pass decays it by `DECODE_RECENCY_DECAY` (0.85), so this admits an expert
/// decode routed there within the last four steps.
pub(super) const RECENT_MIN_SCORE: f32 = 0.5;

/// How many experts a row is predicted at `hop` rows ahead.
pub(super) fn prediction_cap(hop: usize) -> usize {
    if hop >= 3 {
        AHEAD_CAP
    } else {
        STAGE_CAP
    }
}

impl PipelineState {
    /// `experts` predicted for row `row`, best first — the row's latest
    /// prediction, which its listing is vetted from now (`list_row`) and again
    /// as the stager lands what it stages ahead for it (`list_landed`).
    pub(super) fn predict_ahead(&mut self, row: usize, experts: &[usize]) -> Result<()> {
        if row < self.inner.pinned_layers {
            return Ok(());
        }
        self.predicted[row] = experts.to_vec();
        self.list_row(row, true)?;
        Ok(())
    }

    /// Whether `row`'s expert `e` is resident or being promoted: listing it
    /// would buy no copy.
    pub(super) fn is_held(&self, row: usize, e: usize) -> bool {
        self.inner
            .key_to_slot
            .get(&(row, e))
            .is_some_and(|&s| self.inner.slots[s].is_some())
            || self.promoting.contains(&(row, e))
    }

    /// `row`'s experts not in VRAM that decode routed recently — a score of at
    /// least `min` (`RECENT_MIN_SCORE`: within the last few steps) — highest
    /// first, at most `cap`: what the zone let go of and the next step is
    /// likely to want back.
    pub(super) fn scored_absent(&self, row: usize, min: f32, cap: usize) -> Vec<usize> {
        let mut cands: Vec<(usize, f32)> = (0..self.inner.experts_per_layer)
            .filter(|&e| !self.inner.key_to_slot.contains_key(&(row, e)))
            .filter_map(|e| {
                let s = self.inner.score(row, e);
                (s >= min).then_some((e, s))
            })
            .collect();
        cands.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(CmpOrdering::Equal)
                .then(a.0.cmp(&b.0))
        });
        cands.truncate(cap);
        cands.into_iter().map(|(e, _)| e).collect()
    }

    /// Predict the experts the next rows will need, out to the last row a
    /// launch reads ahead for. Each hop's candidates come from three
    /// predictors:
    ///
    /// 1. the router look-ahead's votes (`votes`) — that row's own router
    ///    applied to this layer's input, its margin picks included;
    /// 2. for a narrow launch, the Markov table from this row's routing to the
    ///    hop's arrivals (`transition`), with its confidence — a launch routing
    ///    most of the row (`wide`) leaves it nothing specific to imply;
    /// 3. the target row's experts the zone let go of that decode routed
    ///    recently (`scored_absent`) — any scored one for a wide launch, which
    ///    will route most of that row too.
    ///
    /// The list is their union ranked by what each combination of evidence has
    /// been worth (`blend`), cut at the hop's cap — not the predictors in turn.
    /// In turn, the votes filled every cap and the Markov table never reached a
    /// list, though a strong Markov prediction alone is routed more often than a
    /// one-token vote from hop 3 on, and the two agreeing more often than
    /// either: on Qwen3.8-Flash-Next 92% at hop 1, 86% at hop 5. Ranked by the
    /// cells, the same candidates gave hops 3–5 recall 26/24/22 → 35/34/34% and
    /// precision 55–63 → 84–85% at ×8, and 38/35/31 → 43/41/38% recall at ×1;
    /// ×1 decode rose 28.5–31.6 → 35.3 t/s (warm), ×8 to 96.0 and 106.1.
    ///
    /// A wrong prediction costs a pad slot or a ring offer and the link time
    /// its window allowed, never correctness. Every candidate is kept to be
    /// judged by the row's routing (`Prediction`), so the cells learn from the
    /// whole set, not just the list.
    ///
    /// **Only rows the device has not begun.** The device has begun
    /// `begun_past` invocations past this one, so the rows that many hops on
    /// are already running: a read-ahead or a staging for them cannot land in
    /// time, and predicting them only deepens this thread's lag. Those hops are
    /// skipped. On a card whose GPU outruns this thread the skip is what keeps
    /// it from holding the forward at the summary ring — Qwen3.8-Flash-Next's
    /// gate (RTX PRO 5000) at one to four sequences ran the ring's full 64
    /// invocations behind, and with the skip at most 36: BF16 ×4 decode 677 →
    /// 797 t/s, ×1 warm 200 → 283, the ×2 rows 337–368 → 423–446. On a card
    /// that streams its experts the lag stays short and every hop runs.
    pub(super) fn predict_rows(
        &mut self,
        row: usize,
        current: &[usize],
        votes: Option<&Votes>,
        wide: bool,
        begun_past: u64,
    ) -> Result<()> {
        let width = self.inner.experts_per_layer;
        let voted_hops = votes.map_or(0, |v| v.hops(width));
        for hop in 1..=HOPS {
            let target = row + hop;
            if target >= self.num_moe_layers {
                break;
            }
            if hop as u64 <= begun_past {
                continue;
            }
            let cap = prediction_cap(hop);
            let hop_votes = votes
                .filter(|_| hop <= voted_hops)
                .map(|v| v.hop(hop, width));
            let by_votes = votes
                .filter(|_| hop <= voted_hops)
                .map(|v| ranked(&v.words, width, hop, cap));
            let markov = if wide {
                Vec::new()
            } else {
                self.transition_matrix.candidates(
                    row,
                    hop,
                    current,
                    MARKOV_MIN_CONF,
                    MARKOV_CANDIDATES,
                )
            };
            let min = if wide {
                f32::MIN_POSITIVE
            } else {
                RECENT_MIN_SCORE
            };
            let recent = self.scored_absent(target, min, RECENT_CANDIDATES);
            let held: Vec<bool> = (0..width).map(|e| self.is_held(target, e)).collect();
            // Only an expert a copy could serve is listed: a held one would
            // spend a slot of the cap and buy nothing.
            let (named_held, candidates): (Vec<_>, Vec<_>) = gather(hop_votes, &markov, &recent)
                .into_iter()
                .partition(|c| held[c.expert]);
            let selected = self.cells.select(hop, &candidates, cap);
            self.pending[target][hop - 1] = Some(Prediction {
                votes: by_votes,
                candidates: candidates.iter().map(|c| (c.expert, c.cell)).collect(),
                selected: selected.clone(),
                held,
                held_named: named_held.iter().map(|c| (c.expert, c.cell)).collect(),
            });
            if selected.is_empty() {
                continue;
            }
            // A pinned layer's experts are all resident; `predict_ahead`
            // passes it over.
            self.predict_ahead(target, &selected)?;
        }
        Ok(())
    }

    /// Decide whether the next pass reads ahead (`read_ahead_gate`), from the
    /// pass just finished — after `set_read_ahead_window`, so the fillable
    /// ground is measured against the window that pass will read with.
    pub(super) fn decide_read_ahead(&mut self) {
        let fillable = self.fillable_ground();
        let was = self.link_time.on();
        let on = self.link_time.end_pass(fillable);
        self.read_ahead.set(on);
        if on != was {
            tracing::debug!(
                link_us_per_invocation = self.link_time.us(),
                fillable,
                "expert cache: read-ahead {}",
                if on { "on" } else { "off" }
            );
        }
        if let Ok(mut s) = self.stats.lock() {
            s.ahead_link_us = self.link_time.us();
            s.ahead_passes += 1;
            s.ahead_passes_on += usize::from(on);
            s.ahead_switches += usize::from(on != was);
        }
    }

    /// Whether the zone holds ground read-ahead can fill without evicting: at
    /// least a layer's window of empty slots — free, or empty offers standing
    /// in the ring — and at least as many experts decode routed recently that
    /// the zone does not hold (`scored_absent`'s rule), which is what a
    /// concession handed back and the zone grew into leaves. Counted only to
    /// the window: the answer is a threshold, not a census.
    pub(super) fn fillable_ground(&self) -> bool {
        if self.all_resident {
            return false;
        }
        let window = self.window as usize;
        if window == 0 {
            return false;
        }
        let empty = self.inner.free_len()
            + self
                .promo_ring
                .as_ref()
                .map_or(0, |ring| self.empty_offers(ring));
        if empty < window {
            return false;
        }
        let width = self.inner.experts_per_layer;
        let mut absent = 0usize;
        for row in self.inner.pinned_layers..self.num_moe_layers {
            for e in 0..width {
                if self.inner.score(row, e) >= RECENT_MIN_SCORE
                    && !self.inner.key_to_slot.contains_key(&(row, e))
                {
                    absent += 1;
                    if absent >= window {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Size read-ahead's window from the pass just finished: the median
    /// interval between its consecutive rows' summaries, at the link's measured
    /// rate (`read_ahead`). A pass with no such interval leaves the window as
    /// it was.
    pub(super) fn set_read_ahead_window(&mut self) {
        let Some(layer_secs) = median(&mut self.layer_intervals) else {
            return;
        };
        self.layer_intervals.clear();
        self.window = read_ahead_window(self.link_rate, layer_secs, self.image_bytes, AHEAD_MAX);
        if let Some(ring) = &self.promo_ring {
            ring.set_window(self.window);
        }
        if let Ok(mut s) = self.stats.lock() {
            s.ahead_window = self.window as usize;
        }
    }
}
