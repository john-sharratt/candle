//! The pinned resident set: which experts hold a pinned host slot through a
//! wave, re-chosen between waves from decode's own host accesses
//! (`docs/expert_cache_design.md`, "The pinned tiers").
//!
//! All pinned expert memory is the pad's (`pad.rs`): the churn slots, which a
//! cold read lands in and evicts from inside a wave, and the resident slots —
//! the warm tier's pinned block — which nothing evicts. A slot is resident by
//! label alone, so changing the set moves no bytes: at a pass boundary the
//! stager **promotes** churn slots whose experts decode keeps reaching the
//! host for, and **demotes** the residents it reaches for least, which become
//! churn and age out as any churn slot does. An expert decode misses on often
//! has always just been read into a churn slot, so a promotion is a relabel of
//! a copy already there.
//!
//! **The data is decode's alone, over many waves.** A host access is a decode
//! routing that found its expert outside VRAM — the routing summary's pinned or
//! cold bit with its decode bit ([`HostScores::credit`]). A prompt's routings
//! are not counted: a prompt sweeps most of every layer once, and residency
//! chosen from it would churn for a pass that never comes back. The scores
//! decay once per pass that credited anything, so a prompt's pass does not age
//! decode's history either.
//!
//! **Nothing the device reads moves.** The labels change only at a pass
//! boundary — when the stager reads the next pass's first routing — by at most
//! [`MAX_RELABELS`], and only past [`PROMOTE_MARGIN`]. A label decides only
//! which pad slots the stager may evict; no relabel moves a byte or changes a
//! live-table entry, so the set the device reads through is the one its wave
//! started with, and a candidate has to beat an incumbent clearly to displace
//! it.
//!
//! **Spread across the layers, as it churns — within bounds, not fixed.** The
//! startup fill drew the same share of every layer
//! (`pinned::stratified_membership`), and each layer's resident count may move
//! from there between [`LAYER_FLOOR`] and [`LAYER_CEIL`] times it: a layer
//! that misses more takes more of the budget, but none is drained. The wave is
//! sequential, and a layer's misses are hidden behind the layers before it only
//! while each layer's are few: a set that drifted wholly into the hottest
//! layers would leave the others' misses on the critical path at once. A swap
//! stays inside one layer, or moves a slot between two that both stay inside
//! their bands ([`Bands`]).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};

/// The least a layer's resident count may fall to, as a share of its startup
/// count.
pub(crate) const LAYER_FLOOR: f32 = 0.5;
/// The most a layer's resident count may grow to, as a multiple of its startup
/// count.
pub(crate) const LAYER_CEIL: f32 = 2.0;

/// Each pass that credited a decode access decays every score by this: a
/// window of ~20 decode steps.
pub(crate) const HOST_DECAY: f32 = 0.95;

/// Relabels one pass boundary may make, promotions and demotions together. A
/// relabel moves no bytes, so the cap bounds only how fast the set turns over;
/// [`PROMOTE_MARGIN`] is what keeps it from thrashing. At 64 the set was still
/// replacing the startup's random draw 128 steps in (Flash-Next, the laptop:
/// every boundary at the cap).
pub(crate) const MAX_RELABELS: usize = 512;

/// A candidate displaces a resident only when its score beats the resident's by
/// this factor, plus [`PROMOTE_FLOOR`]: hysteresis, so two experts of nearly
/// equal heat do not trade places every pass.
pub(crate) const PROMOTE_MARGIN: f32 = 1.5;

/// The least score a churn slot needs to be promoted at all — about one decode
/// host access within the window.
pub(crate) const PROMOTE_FLOOR: f32 = 0.5;

/// Per `(row, expert)`: does a host copy hold it that a reload needs no pack
/// read for — a resident pinned slot, or the warm tier's pageable copy? Set
/// from the startup fill, kept by the stager as it labels slots (a demoted
/// resident stays backed while a pageable copy holds it), and read by the
/// VRAM eviction's cost classes (`cache::ReloadTier`).
pub(crate) struct HostBacked {
    flags: Box<[AtomicBool]>,
    n_experts: usize,
}

impl HostBacked {
    pub(crate) fn new(rows: usize, n_experts: usize) -> Self {
        Self {
            flags: (0..rows * n_experts)
                .map(|_| AtomicBool::new(false))
                .collect(),
            n_experts,
        }
    }

    pub(crate) fn set(&self, row: usize, expert: usize, backed: bool) {
        if let Some(f) = self.flags.get(row * self.n_experts + expert) {
            f.store(backed, Ordering::Relaxed);
        }
    }

    pub(crate) fn get(&self, row: usize, expert: usize) -> bool {
        self.flags
            .get(row * self.n_experts + expert)
            .is_some_and(|f| f.load(Ordering::Relaxed))
    }

    pub(crate) fn clear(&self) {
        self.flags
            .iter()
            .for_each(|f| f.store(false, Ordering::Relaxed));
    }
}

/// Decode's host accesses per `(row, expert)`, decayed per decode pass.
pub(crate) struct HostScores {
    scores: Vec<f32>,
    n_experts: usize,
    /// This pass credited at least one access.
    credited: bool,
}

impl HostScores {
    pub(crate) fn new(rows: usize, n_experts: usize) -> Self {
        Self {
            scores: vec![0.0; rows * n_experts],
            n_experts,
            credited: false,
        }
    }

    /// A decode routing reached the host for `(row, expert)`.
    pub(crate) fn credit(&mut self, row: usize, expert: usize) {
        self.scores[row * self.n_experts + expert] += 1.0;
        self.credited = true;
    }

    /// A pass ended: decay, if it credited anything. Returns whether it did.
    pub(crate) fn end_pass(&mut self) -> bool {
        let credited = std::mem::take(&mut self.credited);
        if credited {
            self.scores.iter_mut().for_each(|s| *s *= HOST_DECAY);
        }
        credited
    }

    pub(crate) fn score(&self, row: usize, expert: usize) -> f32 {
        self.scores[row * self.n_experts + expert]
    }
}

/// One relabel: `promote` becomes resident; `demote`, when given, becomes
/// churn in its place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Relabel {
    pub(crate) promote: usize,
    pub(crate) demote: Option<usize>,
}

/// Per layer, the resident counts it must stay between: [`LAYER_FLOOR`] to
/// [`LAYER_CEIL`] times its startup count.
pub(crate) struct Bands(Vec<(usize, usize)>);

impl Bands {
    /// The bands around `counts`, each layer's startup resident count.
    pub(crate) fn around(counts: &[usize]) -> Self {
        Self(
            counts
                .iter()
                .map(|&c| {
                    (
                        (c as f32 * LAYER_FLOOR) as usize,
                        (c as f32 * LAYER_CEIL).ceil() as usize,
                    )
                })
                .collect(),
        )
    }

    fn min(&self, row: usize) -> usize {
        self.0.get(row).map_or(0, |b| b.0)
    }

    fn max(&self, row: usize) -> usize {
        self.0.get(row).map_or(0, |b| b.1)
    }
}

/// A held slot, as the plan sees it: `(slot, row, score)`.
pub(crate) type Held = (usize, usize, f32);

/// The relabels for one pass boundary. `churn` and `residents` are the held
/// slots of each label with their rows and their experts' scores; `capacity`
/// is how many slots may be resident; `bands` how many each row may hold.
///
/// Candidates go best first, each from above [`PROMOTE_FLOOR`]. While
/// residents number fewer than `capacity`, one is promoted outright if its row
/// is below its band's top. Otherwise it displaces the coldest resident it
/// beats by [`PROMOTE_MARGIN`] and the floor among those it may: one of its own
/// row, or — if its row is below its top — one of a row above its band's
/// bottom. At most [`MAX_RELABELS`] slots change label, each counted once.
pub(crate) fn plan(
    mut churn: Vec<Held>,
    residents: Vec<Held>,
    capacity: usize,
    bands: &Bands,
) -> Vec<Relabel> {
    churn.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)));
    let rows = bands.0.len();
    let mut count = vec![0usize; rows];
    // Per row, its residents coldest first.
    let mut by_row: Vec<VecDeque<(usize, f32)>> = vec![VecDeque::new(); rows];
    let mut total = 0usize;
    for (slot, row, score) in residents {
        if row < rows {
            count[row] += 1;
            by_row[row].push_back((slot, score));
            total += 1;
        }
    }
    for q in &mut by_row {
        q.make_contiguous()
            .sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    }
    let mut room = capacity.saturating_sub(total);
    let mut out = Vec::new();
    let mut changed = 0usize;
    for (slot, row, score) in churn {
        if changed >= MAX_RELABELS || score < PROMOTE_FLOOR {
            break;
        }
        if row >= rows {
            continue;
        }
        if room > 0 && count[row] < bands.max(row) {
            out.push(Relabel {
                promote: slot,
                demote: None,
            });
            count[row] += 1;
            room -= 1;
            changed += 1;
            continue;
        }
        if changed + 2 > MAX_RELABELS {
            break;
        }
        // The coldest resident this candidate may displace: its own row's,
        // or, while its row has room in its band, any row's above its bottom.
        let grow = count[row] < bands.max(row);
        let victim_row = (0..rows)
            .filter(|&r| r == row || (grow && count[r] > bands.min(r)))
            .filter_map(|r| by_row[r].front().map(|&(_, s)| (r, s)))
            .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        let Some((vr, v_score)) = victim_row else {
            continue;
        };
        if score <= v_score * PROMOTE_MARGIN + PROMOTE_FLOOR {
            continue;
        }
        let (v_slot, _) = by_row[vr].pop_front().expect("the row's coldest");
        out.push(Relabel {
            promote: slot,
            demote: Some(v_slot),
        });
        count[vr] -= 1;
        count[row] += 1;
        changed += 2;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Credits accumulate; a pass decays only if it credited; flags are per
    /// expert.
    #[test]
    fn scores_decay_per_decode_pass_and_flags_are_per_expert() {
        let mut s = HostScores::new(2, 4);
        s.credit(1, 2);
        s.credit(1, 2);
        assert_eq!(s.score(1, 2), 2.0);
        assert!(s.end_pass());
        assert_eq!(s.score(1, 2), 2.0 * HOST_DECAY);
        assert!(!s.end_pass(), "a pass with no decode access");
        assert_eq!(s.score(1, 2), 2.0 * HOST_DECAY, "and no decay");
        let b = HostBacked::new(2, 4);
        b.set(1, 2, true);
        assert!(b.get(1, 2) && !b.get(1, 3) && !b.get(0, 2));
        b.clear();
        assert!(!b.get(1, 2));
    }

    fn promote(slot: usize) -> Relabel {
        Relabel {
            promote: slot,
            demote: None,
        }
    }

    fn swap(slot: usize, demote: usize) -> Relabel {
        Relabel {
            promote: slot,
            demote: Some(demote),
        }
    }

    /// Bands run from half to twice each layer's startup count.
    #[test]
    fn a_band_is_half_to_twice_the_startup_share() {
        let b = Bands::around(&[100, 7, 0]);
        assert_eq!((b.min(0), b.max(0)), (50, 200));
        assert_eq!((b.min(1), b.max(1)), (3, 14));
        assert_eq!((b.min(2), b.max(2)), (0, 0), "a layer the fill gave none");
    }

    /// With room the hottest churn slots above the floor are promoted
    /// outright, best first.
    #[test]
    fn room_is_filled_by_the_hottest_above_the_floor() {
        let churn = vec![(10, 0, 0.4), (11, 0, 3.0), (12, 0, 1.0), (13, 0, 2.0)];
        let p = plan(churn, vec![(1, 0, 5.0)], 3, &Bands::around(&[4]));
        assert_eq!(p, vec![promote(11), promote(13)]);
    }

    /// Full: a candidate displaces the coldest resident only past the margin;
    /// one that does not is passed over, not the end of the scan.
    #[test]
    fn a_full_set_swaps_only_past_the_margin() {
        let residents = vec![(1, 0, 0.2), (2, 0, 1.0), (3, 0, 4.0)];
        // 9.0 and 3.0 beat 0.2×1.5+0.5 and 1.0×1.5+0.5; 1.6 does not beat
        // 4.0×1.5+0.5.
        let churn = vec![(20, 0, 3.0), (21, 0, 1.6), (22, 0, 9.0)];
        let p = plan(churn, residents, 3, &Bands::around(&[3]));
        assert_eq!(p, vec![swap(22, 1), swap(20, 2)]);
    }

    /// **The set moves toward the layer that misses, inside the bands.** Two
    /// layers of four, all cold; ten hot candidates in layer 1. Layer 0 gives
    /// up slots only down to its floor of two, then layer 1's own cold
    /// residents go, and with none left and layer 0 at its floor the rest wait.
    #[test]
    fn the_set_shifts_between_layers_only_inside_their_bands() {
        let residents: Vec<Held> = (0..8).map(|s| (s, s / 4, 0.0)).collect();
        let churn: Vec<Held> = (0..10).map(|s| (100 + s, 1, 10.0)).collect();
        let p = plan(churn, residents, 8, &Bands::around(&[4, 4]));
        assert_eq!(p.len(), 6);
        let demoted: Vec<usize> = p.iter().filter_map(|r| r.demote).collect();
        assert_eq!(
            demoted.iter().filter(|&&s| s < 4).count(),
            2,
            "layer 0 kept its floor of two"
        );
        assert_eq!(
            demoted.iter().filter(|&&s| s >= 4).count(),
            4,
            "layer 1's own cold residents"
        );
    }

    /// No more than `MAX_RELABELS` labels change in one boundary.
    #[test]
    fn a_boundary_changes_at_most_the_cap() {
        let n = MAX_RELABELS + 100;
        let churn: Vec<Held> = (0..n).map(|s| (10_000 + s, 0, 10.0)).collect();
        let p = plan(churn.clone(), Vec::new(), 2 * n, &Bands::around(&[2 * n]));
        assert_eq!(p.len(), MAX_RELABELS);
        let residents: Vec<Held> = (0..n).map(|s| (s, 0, 0.0)).collect();
        let p = plan(churn, residents, n, &Bands::around(&[n]));
        assert_eq!(p.len(), MAX_RELABELS / 2, "a swap changes two labels");
        assert!(p.iter().all(|r| r.demote.is_some()));
    }
}
