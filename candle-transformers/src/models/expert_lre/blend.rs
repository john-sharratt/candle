//! One prediction list from three predictors, ranked by what each kind of
//! evidence has turned out to be worth.
//!
//! A row ahead is predicted three ways: the router look-ahead's votes (`votes`),
//! the Markov tables (`transition`), and the row's recent decode routing. Each
//! speaks its own units — vote counts, a per-source confidence, a decayed score
//! — so none ranks against another directly. What does compare is how often a
//! candidate carrying a given combination of evidence is then routed.
//!
//! So every candidate is placed in a **cell** — what the votes said about it,
//! what Markov said, whether decode routed it recently — and each cell keeps,
//! per hop, how many of its candidates were proposed and how many the target row
//! then routed. That ratio is the cell's probability, and the list is the
//! candidates in order of it, cut at the hop's cap. A list filled by probability
//! holds the most expected hits the cap can buy, and no source has a fixed share:
//! whichever carries the stronger evidence on that row, at that hop, wins the
//! slots.
//!
//! **Cells are measured jointly, not combined.** The votes and the Markov tables
//! both look at the current row, so their evidence is correlated, and a
//! probability built from each alone would over-count their agreement. The cell
//! "both said so" is its own cell and learns what agreement is worth.
//!
//! **Every candidate is judged, chosen or not.** Whether a row routes an expert
//! does not depend on whether it was loaded, so the cells learn from the whole
//! candidate set and a cell the cap keeps cutting still converges.
//!
//! **A cold cell ranks low.** Each starts from a weak prior well under any
//! measured cell, so the votes — measured from the first rows — hold the list
//! while the Markov tables are still warming, and a Markov cell earns its place
//! as its own hit rate shows.
//!
//! **A cell under even odds lists nothing** (`BREAK_EVEN`). A listed expert is
//! copied over the link whether or not its row then routes it, so the list is
//! worth its cap only while each entry is more likely used than not; a cell
//! below that still has its candidates judged, and is listed again once its
//! record climbs past it.

use super::transition::HOPS;
use std::collections::HashMap;

/// What the votes said about a candidate: nothing, a margin pick only, routed
/// by one token, routed by several.
const VOTE_BUCKETS: usize = 4;
/// What Markov said: nothing, then its per-source confidence in three bands.
const MARKOV_BUCKETS: usize = 4;
/// Cells per hop: every vote bucket × Markov bucket × recently routed or not.
pub const CELLS: usize = VOTE_BUCKETS * MARKOV_BUCKETS * 2;

/// The weak prior every cell starts from: `PRIOR_HITS` hits in `PRIOR_PREDS`
/// predictions — 10%, under the hit rate of any evidence measured here, and
/// worth ten judged candidates, so a cell's own record outweighs it within a
/// row or two.
const PRIOR_HITS: f64 = 1.0;
const PRIOR_PREDS: f64 = 10.0;

/// The lowest probability a candidate is listed at. Its read-ahead copy is
/// made by the gate launch's workers, in the stream, over the link a demand
/// miss crosses too: a right one saves the copy the miss would have made, a
/// wrong one costs a copy as long, so under even odds the list spends more link
/// time than it saves. DeepSeek-V4-Flash (RTX PRO 5000), whose sweeps cast no
/// look-ahead votes, listed from its Markov and recent cells at 4–33% — 4.9
/// claims a routed layer, 9% of them routed, 170 GiB of copies in its gate's
/// warm ×1 run — and that run's decode fell from 13.7 to 9.8 t/s.
const BREAK_EVEN: f64 = 0.5;

/// The lowest Markov confidence a candidate is proposed at, and the bands above
/// it: `[MARKOV_MIN_CONF, 0.5)`, `[0.5, 0.8)`, `[0.8, 1]`. The confidence is per
/// source relative (`transition`), 1.0 for each active expert's strongest
/// successor.
pub(crate) const MARKOV_MIN_CONF: f32 = 0.25;

/// The cell of a candidate: `vote` its look-ahead vote word (`routed << 16 |
/// margin`, 0 for none), `markov` its Markov confidence when Markov proposed it,
/// `recent` whether decode routed it at that row recently.
pub(crate) fn cell(vote: u32, markov: Option<f32>, recent: bool) -> usize {
    let routed = vote >> 16;
    let margin = vote & 0xffff;
    let v = if routed >= 2 {
        3
    } else if routed == 1 {
        2
    } else if margin > 0 {
        1
    } else {
        0
    };
    let m = match markov {
        None => 0,
        Some(c) if c < 0.5 => 1,
        Some(c) if c < 0.8 => 2,
        Some(_) => 3,
    };
    (v * MARKOV_BUCKETS + m) * 2 + usize::from(recent)
}

/// A cell's three axes: the vote bucket, the Markov bucket (0 for none), and
/// whether decode routed it recently — the inverse of [`cell`].
pub(crate) fn cell_axes(cell: usize) -> (usize, usize, bool) {
    (
        cell / 2 / MARKOV_BUCKETS,
        (cell / 2) % MARKOV_BUCKETS,
        cell % 2 == 1,
    )
}

/// A cell's label for the gate's log — `v{votes}m{markov}r{recent}`.
pub fn cell_label(cell: usize) -> String {
    let (v, m, r) = cell_axes(cell);
    format!("v{v}m{m}r{}", u8::from(r))
}

/// One candidate for a row ahead: the expert, its cell, and a tiebreak within
/// the cell — the router probability its votes carried, then the vote word,
/// then the Markov confidence, so a cell's stronger members come first.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Candidate {
    pub(crate) expert: usize,
    pub(crate) cell: usize,
    pub(crate) mass: f32,
    pub(crate) vote: u32,
    pub(crate) markov: f32,
}

/// Every candidate for one row ahead, each with its evidence: `votes` the hop's
/// vote words and their router mass (`[n_experts]` each), `markov` the Markov
/// candidates with their confidences, `recent` the recently routed ones. An
/// expert appears once.
pub(crate) fn gather(
    votes: Option<(&[u32], &[f32])>,
    markov: &[(usize, f32)],
    recent: &[usize],
) -> Vec<Candidate> {
    let mut out: Vec<(usize, u32, f32, Option<f32>, bool)> = Vec::new();
    // Each expert's place in `out`, so a later predictor's evidence merges
    // into its entry in constant time.
    let mut at: HashMap<usize, usize> = HashMap::new();
    if let Some((words, mass)) = votes {
        for (e, (&w, &m)) in words.iter().zip(mass).enumerate() {
            if w > 0 {
                at.insert(e, out.len());
                out.push((e, w, m, None, false));
            }
        }
    }
    for &(e, c) in markov {
        match at.get(&e) {
            Some(&i) => out[i].3 = Some(c),
            None => {
                at.insert(e, out.len());
                out.push((e, 0, 0.0, Some(c), false));
            }
        }
    }
    for &e in recent {
        match at.get(&e) {
            Some(&i) => out[i].4 = true,
            None => {
                at.insert(e, out.len());
                out.push((e, 0, 0.0, None, true));
            }
        }
    }
    out.into_iter()
        .map(|(expert, vote, mass, markov, recent)| Candidate {
            expert,
            cell: cell(vote, markov, recent),
            mass,
            vote,
            markov: markov.unwrap_or(0.0),
        })
        .collect()
}

/// Each hop's cells: candidates proposed, and of them the ones the target row
/// routed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellTable {
    pub preds: [[u32; CELLS]; HOPS],
    pub hits: [[u32; CELLS]; HOPS],
}

impl Default for CellTable {
    fn default() -> Self {
        Self {
            preds: [[0; CELLS]; HOPS],
            hits: [[0; CELLS]; HOPS],
        }
    }
}

impl CellTable {
    /// The probability a candidate in `cell` is routed `hop` rows ahead.
    pub fn probability(&self, hop: usize, cell: usize) -> f64 {
        (self.hits[hop - 1][cell] as f64 + PRIOR_HITS)
            / (self.preds[hop - 1][cell] as f64 + PRIOR_PREDS)
    }

    /// One candidate judged against its row's routing.
    pub(crate) fn record(&mut self, hop: usize, cell: usize, routed: bool) {
        let p = &mut self.preds[hop - 1][cell];
        *p = p.saturating_add(1);
        if routed {
            let h = &mut self.hits[hop - 1][cell];
            *h = h.saturating_add(1);
        }
    }

    /// The candidates whose cell's probability at `hop` reaches `BREAK_EVEN`,
    /// in order of it — then the tiebreak, then the expert — cut at `cap`.
    pub(crate) fn select(&self, hop: usize, candidates: &[Candidate], cap: usize) -> Vec<usize> {
        let mut ranked: Vec<(f64, &Candidate)> = candidates
            .iter()
            .map(|c| (self.probability(hop, c.cell), c))
            .filter(|&(p, _)| p >= BREAK_EVEN)
            .collect();
        ranked.sort_by(|a, b| {
            b.0.total_cmp(&a.0)
                .then(b.1.mass.total_cmp(&a.1.mass))
                .then(b.1.vote.cmp(&a.1.vote))
                .then(b.1.markov.total_cmp(&a.1.markov))
                .then(a.1.expert.cmp(&b.1.expert))
        });
        ranked.truncate(cap);
        ranked.into_iter().map(|(_, c)| c.expert).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The cell index, raw, across each axis.
    #[test]
    fn a_candidates_evidence_names_its_cell() {
        assert_eq!(cell(0, None, false), 0);
        assert_eq!(cell(0, None, true), 1);
        assert_eq!(cell(3, None, false), 8, "margin only");
        assert_eq!(cell(1 << 16, None, false), 16, "routed by one");
        assert_eq!(cell((4 << 16) | 2, None, false), 24, "routed by several");
        assert_eq!(cell(0, Some(0.3), false), 2);
        assert_eq!(cell(0, Some(0.6), false), 4);
        assert_eq!(cell(0, Some(1.0), true), 7);
        assert_eq!(cell(2 << 16, Some(0.9), true), CELLS - 1);
        assert_eq!(cell_label(CELLS - 1), "v3m3r1");
        assert_eq!(cell_axes(cell(1 << 16, Some(0.6), true)), (2, 2, true));
        assert_eq!(cell_label(cell(5, Some(0.6), false)), "v1m2r0");
    }

    /// Each expert once, its evidence merged across the predictors.
    #[test]
    fn gathering_merges_each_experts_evidence() {
        // Expert 1 routed by one token, 2 a margin pick, 3 routed by two.
        let votes = [0, 1 << 16, 1, 2 << 16, 0, 0];
        let mass = [0.0, 0.5, 0.125, 0.75, 0.0, 0.0];
        let markov = [(2, 0.9), (4, 0.3), (3, 1.0)];
        let recent = [4, 5];
        let got = gather(Some((&votes, &mass)), &markov, &recent);
        let want = [
            (1, cell(1 << 16, None, false), 0.5, 1 << 16, 0.0),
            (2, cell(1, Some(0.9), false), 0.125, 1, 0.9),
            (3, cell(2 << 16, Some(1.0), false), 0.75, 2 << 16, 1.0),
            (4, cell(0, Some(0.3), true), 0.0, 0, 0.3),
            (5, cell(0, None, true), 0.0, 0, 0.0),
        ]
        .map(|(expert, cell, mass, vote, markov)| Candidate {
            expert,
            cell,
            mass,
            vote,
            markov,
        });
        assert_eq!(got, want);
    }

    /// A cell starts at the prior and moves to its record: 10% unjudged, then
    /// (hits + 1) / (preds + 10).
    #[test]
    fn a_cells_probability_is_its_record_over_a_weak_prior() {
        let mut t = CellTable::default();
        assert_eq!(t.probability(1, 5), 0.1);
        for routed in [true, true, true, false] {
            t.record(1, 5, routed);
        }
        assert_eq!(t.probability(1, 5), 4.0 / 14.0);
        assert_eq!(t.probability(2, 5), 0.1, "hops learn apart");
    }

    /// The list follows the cells' records, not the sources' order: the
    /// Markov-only cell (20 of 21, 21/31) ranks under the routed cell (20 of
    /// 20, 21/30), the margin cell (0 of 20, 1/30) lists nothing with room to
    /// spare, and the cap cuts the weakest of the rest. Within a cell the router
    /// mass breaks the tie — e9 routed as firmly as one token can, e2 barely,
    /// the same vote word.
    #[test]
    fn the_list_is_ordered_by_measured_probability_and_cut_at_the_cap() {
        let mut t = CellTable::default();
        let routed = cell(1 << 16, None, false);
        let margin = cell(1, None, false);
        let markov = cell(0, Some(0.9), false);
        for _ in 0..20 {
            t.record(1, routed, true);
            t.record(1, markov, true);
            t.record(1, margin, false);
        }
        t.record(1, markov, false);
        let c = |expert, cell, mass, vote, markov| Candidate {
            expert,
            cell,
            mass,
            vote,
            markov,
        };
        let candidates = [
            c(7, margin, 0.25, 1, 0.0),
            c(3, markov, 0.0, 0, 0.9),
            c(2, routed, 0.0625, 1 << 16, 0.0),
            c(9, routed, 0.875, 1 << 16, 0.0),
        ];
        assert_eq!(t.select(1, &candidates, 8), vec![9, 2, 3]);
        assert_eq!(t.select(1, &candidates, 2), vec![9, 2]);
    }

    /// Even odds is listed, a hair under is not: 9 routed of 10 is (9 + 1) /
    /// (10 + 10) = 1/2, 8 of 10 is 9/20. The cell under it is still judged and
    /// is listed once its record climbs — one more routed candidate, 10/21 → 11/22.
    #[test]
    fn a_cell_under_even_odds_lists_nothing() {
        let mut t = CellTable::default();
        let even = cell(0, Some(0.9), true);
        let under = cell(0, Some(0.9), false);
        for i in 0..10 {
            t.record(1, even, i < 9);
            t.record(1, under, i < 8);
        }
        assert_eq!(t.probability(1, even), 0.5);
        assert_eq!(t.probability(1, under), 0.45);
        let c = |expert, cell| Candidate {
            expert,
            cell,
            mass: 0.0,
            vote: 0,
            markov: 0.9,
        };
        let candidates = [c(4, under), c(6, even)];
        assert_eq!(t.select(1, &candidates, 8), vec![6]);
        t.record(1, under, true);
        assert_eq!(t.probability(1, under), 10.0 / 21.0);
        assert_eq!(t.select(1, &candidates, 8), vec![6]);
        t.record(1, under, true);
        assert_eq!(t.probability(1, under), 0.5);
        assert_eq!(t.select(1, &candidates, 8), vec![4, 6], "tied, by expert");
    }
}
