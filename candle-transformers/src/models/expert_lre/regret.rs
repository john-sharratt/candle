//! Evictions and promotions judged at the row's next visit.
//!
//! Every claim from the promotion ring promotes an expert into a slot, and when
//! the slot held a lazy victim it evicts that victim — a demand miss's claim or
//! a read-ahead copy's. The claim paid if its row routes the promoted expert
//! the next time the row runs; the eviction was a mistake if the row routes the
//! victim then. This ledger holds both until the row's next routing judges
//! them, by the kind of claim, so the eviction ranking (`cache::rank_victims`)
//! and the admission of promotions are measured where they act rather than
//! through the hit rate they only share: a claim is worth making only while the
//! experts it promotes are routed again more often than the ones it displaces.
//!
//! Each victim also carries a [`VictimTag`] — what was known about it when it
//! was evicted: whether a live prediction named it for its row, whether its row
//! routed it on its last visit, whether it sat just ahead of the wave or behind
//! it, and whether a warm copy backs it — so the regret splits by the evidence
//! a better ranking could have used to keep it, and by what its miss costs.
//!
//! An entry is judged only by an invocation of its row that began after the
//! claim (`ticket` above the claim's): the pipeline thread may collect a claim
//! the device made later than the routing it is serving.

use std::array;

/// How many layers past the served row count as just ahead of the wave: the
/// rows the router look-ahead predicts for.
const AHEAD_ROWS: usize = 5;

/// What was known about a victim when it was evicted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct VictimTag {
    /// The best measured probability a live prediction for its row gave it:
    /// 0 none, 1 below one half, 2 one half or more.
    pub(crate) predicted: usize,
    /// Its row routed it on its last visit.
    pub(crate) hit_last: bool,
    /// Its row lies within [`AHEAD_ROWS`] past the row being served.
    pub(crate) ahead: bool,
    /// A warm (host) copy backs it — the cheap reload — rather than the pack
    /// alone.
    pub(crate) warm: bool,
}

/// Distinct tags: 3 prediction bands × last-visit hit × ahead or behind ×
/// warm or pack-only.
pub const TAGS: usize = 24;

/// Eviction classes — what the ranking orders its passes by
/// (`cache::ExpertCacheInner::set_eviction_costs`): whether the row routed the
/// victim on its last visit × whether a warm copy backs it, as
/// `hit * 2 + warm`.
pub(crate) const CLASSES: usize = 4;

/// The eviction class of a resident: whether its row routed it on its last
/// visit, whether a warm copy backs it.
pub(crate) fn class_index(hit_last: bool, warm: bool) -> usize {
    usize::from(hit_last) * 2 + usize::from(warm)
}

/// Each eviction class's expected miss cost, by class index: its regret rate
/// times what restoring one costs — `copy` (one image over the link) for a
/// warm-backed class, `read + copy` (the pack read the layer waits on, then
/// the copy) for a pack-only one.
pub(crate) fn eviction_costs(rates: [f64; CLASSES], copy: f64, read: f64) -> [f64; CLASSES] {
    array::from_fn(|c| {
        let warm = c == class_index(false, true) || c == class_index(true, true);
        rates[c] * if warm { copy } else { read + copy }
    })
}

/// The band of the best measured probability a live prediction gave a
/// victim: 0 none, 1 below one half, 2 one half or more.
pub(crate) fn prediction_band(best: Option<f64>) -> usize {
    match best {
        None => 0,
        Some(p) if p < 0.5 => 1,
        Some(_) => 2,
    }
}

/// Whether `row` lies within [`AHEAD_ROWS`] past `served`, of `n` rows,
/// through the wrap — not `served` itself.
pub(crate) fn just_ahead(row: usize, served: usize, n: usize) -> bool {
    (1..=AHEAD_ROWS).contains(&((row + n - served) % n))
}

impl VictimTag {
    /// This victim's eviction class.
    pub(crate) fn class(self) -> usize {
        class_index(self.hit_last, self.warm)
    }

    pub(crate) fn index(self) -> usize {
        ((self.predicted * 2 + usize::from(self.hit_last)) * 2 + usize::from(self.ahead)) * 2
            + usize::from(self.warm)
    }
}

/// A tag's label for the gate's log: `p{0-2} {hit|cold} {ahead|behind}
/// {warm|pack}`.
pub fn tag_label(index: usize) -> String {
    let warm = if index % 2 == 1 { "warm" } else { "pack" };
    let ahead = if (index / 2) % 2 == 1 {
        "ahead"
    } else {
        "behind"
    };
    let hit = if (index / 4) % 2 == 1 { "hit" } else { "cold" };
    format!("p{} {hit} {ahead} {warm}", index / 8)
}

/// One claimed expert awaiting its row's next routing.
#[derive(Clone, Copy, Debug)]
struct Entry {
    expert: usize,
    /// Taken by a read-ahead copy's claim, not a demand miss's.
    ahead: bool,
    /// Promoted by the claim — or evicted by it, with what was known of it.
    victim: Option<VictimTag>,
    ticket: u64,
}

/// How much each judged victim's weight decays per later one of its class: a
/// memory of about a thousand victims, so the rates follow a change of width
/// within a few hundred evictions.
const RATE_MEMORY: f64 = 0.999;

/// The prior every class starts from: [`PRIOR_RATE`] over [`PRIOR_VICTIMS`]
/// judged victims — equal across classes, so until the ledger has measured
/// them the reload costs alone order the passes.
const PRIOR_VICTIMS: f64 = 10.0;
const PRIOR_RATE: f64 = 0.25;

/// Each eviction class's share of victims routed again at their row's next
/// visit, as a decayed running rate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ClassRates {
    evicted: [f64; CLASSES],
    regretted: [f64; CLASSES],
}

impl Default for ClassRates {
    fn default() -> Self {
        Self {
            evicted: [PRIOR_VICTIMS; CLASSES],
            regretted: [PRIOR_VICTIMS * PRIOR_RATE; CLASSES],
        }
    }
}

impl ClassRates {
    /// One victim of `class` judged: routed again or not.
    fn record(&mut self, class: usize, again: bool) {
        self.evicted[class] = self.evicted[class] * RATE_MEMORY + 1.0;
        self.regretted[class] = self.regretted[class] * RATE_MEMORY + f64::from(u8::from(again));
    }

    /// Each class's rate, by class index.
    pub(crate) fn rates(&self) -> [f64; CLASSES] {
        array::from_fn(|c| self.regretted[c] / self.evicted[c])
    }
}

/// Claimed experts awaiting judgement, per row, and the running regret rate
/// of each eviction class.
pub(crate) struct RegretLedger {
    pending: Vec<Vec<Entry>>,
    rates: ClassRates,
}

/// Judged and routed-again counts, by claim kind: `[demand, read-ahead]`,
/// and the victims by tag (both kinds).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Verdicts {
    /// Victims judged, and those routed again — regretted.
    pub(crate) evicted: [usize; 2],
    pub(crate) regretted: [usize; 2],
    /// Promotions judged, and those routed again — paid.
    pub(crate) promoted: [usize; 2],
    pub(crate) paid: [usize; 2],
    pub(crate) tag_evicted: [usize; TAGS],
    pub(crate) tag_regretted: [usize; TAGS],
}

impl RegretLedger {
    pub(crate) fn new(rows: usize) -> Self {
        Self {
            pending: vec![Vec::new(); rows],
            rates: ClassRates::default(),
        }
    }

    /// Each eviction class's running regret rate, by class index.
    pub(crate) fn class_rates(&self) -> [f64; CLASSES] {
        self.rates.rates()
    }

    /// `row`'s `expert` was evicted by a claim of invocation `ticket` — a
    /// read-ahead copy's when `ahead` — knowing `tag` of it.
    pub(crate) fn evicted(
        &mut self,
        row: usize,
        expert: usize,
        ahead: bool,
        ticket: u64,
        tag: VictimTag,
    ) {
        self.pending[row].push(Entry {
            expert,
            ahead,
            victim: Some(tag),
            ticket,
        });
    }

    /// `row`'s `expert` was promoted by a claim of invocation `ticket`.
    pub(crate) fn promoted(&mut self, row: usize, expert: usize, ahead: bool, ticket: u64) {
        self.pending[row].push(Entry {
            expert,
            ahead,
            victim: None,
            ticket,
        });
    }

    /// `row`'s demand promotion of `expert` was dropped before it landed — a
    /// faulted forward gave its item up and its slot was never written — so
    /// there is no promotion to judge.
    pub(crate) fn withdraw_promotion(&mut self, row: usize, expert: usize) {
        self.pending[row].retain(|e| !(e.victim.is_none() && !e.ahead && e.expert == expert));
    }

    /// Invocation `ticket` of `row` routed `routed` (sorted): judge every
    /// entry of the row claimed before it began.
    pub(crate) fn judge(&mut self, row: usize, ticket: u64, routed: &[usize]) -> Verdicts {
        let mut v = Verdicts::default();
        let rates = &mut self.rates;
        self.pending[row].retain(|e| {
            if e.ticket >= ticket {
                return true;
            }
            let k = usize::from(e.ahead);
            let again = usize::from(routed.binary_search(&e.expert).is_ok());
            match e.victim {
                None => {
                    v.promoted[k] += 1;
                    v.paid[k] += again;
                }
                Some(tag) => {
                    v.evicted[k] += 1;
                    v.regretted[k] += again;
                    v.tag_evicted[tag.index()] += 1;
                    v.tag_regretted[tag.index()] += again;
                    rates.record(tag.class(), again == 1);
                }
            }
            false
        });
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const COLD_BEHIND: VictimTag = VictimTag {
        predicted: 0,
        hit_last: false,
        ahead: false,
        warm: false,
    };

    /// A victim the row routes again is regretted and a promotion it routes
    /// again paid, each by claim kind and each judged once.
    #[test]
    fn the_rows_next_routing_judges_victims_and_promotions() {
        let mut l = RegretLedger::new(4);
        l.evicted(2, 7, false, 10, COLD_BEHIND); // routed again: regretted
        l.evicted(2, 9, false, 10, COLD_BEHIND); // not
        l.evicted(2, 5, true, 11, COLD_BEHIND); // read-ahead claim's victim, routed again
        l.promoted(2, 1, false, 10); // routed again: paid
        l.promoted(2, 4, true, 11); // read-ahead promotion, not routed
        l.evicted(3, 7, false, 10, COLD_BEHIND); // another row, untouched
        let mut tag_evicted = [0; TAGS];
        let mut tag_regretted = [0; TAGS];
        tag_evicted[0] = 3;
        tag_regretted[0] = 2;
        assert_eq!(
            l.judge(2, 20, &[1, 5, 7]),
            Verdicts {
                evicted: [2, 1],
                regretted: [1, 1],
                promoted: [1, 1],
                paid: [1, 0],
                tag_evicted,
                tag_regretted,
            }
        );
        assert_eq!(l.judge(2, 30, &[1, 4, 5, 7, 9]), Verdicts::default());
        assert_eq!(l.judge(3, 30, &[]).evicted, [1, 0]);
    }

    /// A claim the device made after the invocation being judged began waits
    /// for the row's next one.
    #[test]
    fn a_claim_after_the_routing_waits_for_the_next_visit() {
        let mut l = RegretLedger::new(1);
        l.evicted(0, 3, false, 15, COLD_BEHIND);
        l.promoted(0, 4, false, 15);
        assert_eq!(l.judge(0, 15, &[3, 4]), Verdicts::default());
        let v = l.judge(0, 16, &[3, 4]);
        assert_eq!((v.regretted, v.paid), ([1, 0], [1, 0]));
    }

    /// Classes are `hit * 2 + warm`; a warm-backed class costs its rate × the
    /// copy, a pack-only one its rate × (read + copy). Raw: copy 0.25, read
    /// 1.75, every rate 0.5.
    #[test]
    fn eviction_costs_price_each_class_by_its_restore() {
        assert_eq!(
            [
                class_index(false, false),
                class_index(false, true),
                class_index(true, false),
                class_index(true, true)
            ],
            [0, 1, 2, 3]
        );
        assert_eq!(
            eviction_costs([0.5; CLASSES], 0.25, 1.75),
            [1.0, 0.125, 1.0, 0.125]
        );
    }

    /// The prediction bands, and placement through the wrap: rows 1..=5 past
    /// the served one are just ahead, the served row itself is not.
    #[test]
    fn victims_are_banded_and_placed() {
        assert_eq!(
            [
                prediction_band(None),
                prediction_band(Some(0.49)),
                prediction_band(Some(0.5))
            ],
            [0, 1, 2]
        );
        assert!(!just_ahead(10, 10, 48), "the served row");
        assert!(just_ahead(11, 10, 48) && just_ahead(15, 10, 48));
        assert!(!just_ahead(16, 10, 48) && !just_ahead(9, 10, 48));
        assert!(just_ahead(2, 46, 48), "through the wrap");
    }

    /// A dropped demand promotion is not judged; the same expert's read-ahead
    /// promotion and its victims are.
    #[test]
    fn a_withdrawn_demand_promotion_is_not_judged() {
        let mut l = RegretLedger::new(1);
        l.promoted(0, 4, false, 1);
        l.promoted(0, 4, true, 1);
        l.evicted(0, 4, false, 1, COLD_BEHIND);
        l.withdraw_promotion(0, 4);
        let v = l.judge(0, 2, &[4]);
        assert_eq!((v.promoted, v.paid, v.evicted), ([0, 1], [0, 1], [1, 0]));
    }

    /// Each class starts at the prior, 0.25, and moves to its record:
    /// (regretted + prior) / (judged + prior), the older judgements decayed.
    #[test]
    fn each_class_learns_its_own_regret_rate() {
        let mut l = RegretLedger::new(1);
        assert_eq!(l.class_rates(), [0.25; CLASSES]);
        let hit_warm = VictimTag {
            predicted: 0,
            hit_last: true,
            ahead: false,
            warm: true,
        };
        assert_eq!((COLD_BEHIND.class(), hit_warm.class()), (0, 3));
        l.evicted(0, 3, false, 1, hit_warm);
        l.evicted(0, 4, false, 1, COLD_BEHIND);
        l.judge(0, 2, &[3]);
        let r = l.class_rates();
        // hit, warm: (2.5 × 0.999 + 1) / (10 × 0.999 + 1); cold, pack: one
        // victim not routed again.
        assert_eq!(r[3], (2.5 * 0.999 + 1.0) / (10.0 * 0.999 + 1.0));
        assert_eq!(r[0], (2.5 * 0.999) / (10.0 * 0.999 + 1.0));
        assert_eq!(
            (r[1], r[2]),
            (0.25, 0.25),
            "untouched classes keep the prior"
        );
    }

    /// Each tag has its own index and label, and a victim's verdict lands
    /// under its tag.
    #[test]
    fn a_victims_verdict_lands_under_its_tag() {
        let strong_hit_ahead_warm = VictimTag {
            predicted: 2,
            hit_last: true,
            ahead: true,
            warm: true,
        };
        let weak_cold_behind_pack = VictimTag {
            predicted: 1,
            hit_last: false,
            ahead: false,
            warm: false,
        };
        assert_eq!(strong_hit_ahead_warm.index(), TAGS - 1);
        assert_eq!(weak_cold_behind_pack.index(), 8);
        assert_eq!(tag_label(TAGS - 1), "p2 hit ahead warm");
        assert_eq!(tag_label(8), "p1 cold behind pack");
        assert_eq!(tag_label(5), "p0 hit behind warm");

        let mut l = RegretLedger::new(1);
        l.evicted(0, 3, false, 1, strong_hit_ahead_warm);
        l.evicted(0, 5, true, 1, weak_cold_behind_pack);
        let v = l.judge(0, 2, &[3]);
        assert_eq!(
            (
                v.tag_evicted[TAGS - 1],
                v.tag_regretted[TAGS - 1],
                v.tag_evicted[8],
                v.tag_regretted[8]
            ),
            (1, 1, 1, 0)
        );
    }
}
