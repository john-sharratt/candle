//! The `Ground` trait the fill is run against: what the engine reports to it
//! and what it asks of it, one offer at a time.
//!
//! The fill walks each band's queue and asks the engine for the next candidate,
//! its cost, the residency it is judged against and the work the wave has
//! already been charged. Residency is read again before every offer, because it
//! moves inside a fill from places admission cannot see, and a figure read
//! earlier would judge a card that has since changed. A band is strictly first
//! in, first out: an item that does not fit ends the band, and the cheaper items
//! behind it are not offered in its place.

use super::*;

/// The wave's opening terms, read once per fill from the settled device.
///
/// Everything [`rate::WaveRate::reset`] needs, gathered in one place so the
/// four figures are taken from the same moment — a floor read after a purchase
/// that a residency read preceded would let a wave admit into ground that had
/// already moved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Budget {
    /// Resident weights as the wave opens.
    pub resident: u64,
    /// The residency a **prefill** admission may not take them under — lower than
    /// [`Self::decode_floor`], because a prefill's copy amortises over its rows
    /// while a decode's is charged per layer per step. See
    /// `WaveRate::prefill_floor`.
    pub prefill_floor: u64,
    /// The residency a **decode** admission may not take them under.
    pub decode_floor: u64,
    /// The widest wave whose transient tier the partition could place — the
    /// gap as it stands plus what the weight side could concede for it. A rate
    /// the tier cannot hold is not a rate the engine can run.
    pub max_rows: usize,
    /// The most decodes a wave will carry.
    pub max_decodes: usize,
}

/// The engine, as the fill needs to see it.
pub(crate) trait Ground {
    /// Slots admitted and not yet finished — prefilling or decoding alike.
    fn active(&self) -> usize;

    /// Active slots currently in their decode phase.
    fn decodes_active(&self) -> usize;

    /// Whether anything finished since the last admission pass.
    ///
    /// `false` is the fast path: no slot freed, so nothing new can fit that did
    /// not fit before, and the wave runs what it holds. Admission is O(slots
    /// completed), not O(waves).
    fn settled(&self) -> bool;

    /// The device after the eviction pass — see the module header.
    fn headroom(&self) -> Headroom;

    /// The wave's opening terms, from that same settled reading.
    fn budget(&self) -> Budget;

    /// Resident weights **right now**, re-read before every offer.
    ///
    /// Not carried forward from the last admission: residency moves inside a
    /// fill from places admission cannot see. The expert cache grows back into
    /// spare K/V ground at phase 0 of every forward, evicts slots mid-forward,
    /// and concedes ground to any claim that runs the K/V side out. A wave
    /// judged against a figure from three offers ago is judged against a card
    /// that has since moved.
    fn resident_weights(&self) -> u64;

    /// Rows the next wave already carries before this fill adds anything: the
    /// creep group held from the last wave, which rides the next one whole.
    ///
    /// Charged to the wave at the open, so the offers that follow are judged
    /// against what the forward will actually run — and so a wave that is
    /// already carrying rows does not also take a head unconditionally.
    fn standing_rows(&self) -> usize;

    /// What admitting this band's next FIFO candidate would take, or `None`
    /// when the band has nothing left to offer.
    ///
    /// Strictly FIFO: an item that does not fit is **not** passed over for one
    /// that does. Cheapest-first starves the expensive work permanently, and
    /// the expensive work is never the cheapest.
    fn peek(&mut self, kind: Kind, prio: DecodePriority) -> Option<Cost>;

    /// Whether some active slot belongs to a band below `prio` — work an offer
    /// of `prio` outranks ([`rate::WaveRate::try_outranking`]). A section is
    /// `Low`.
    fn outranks_active(&self, prio: DecodePriority) -> bool;

    /// Running decodes of `prio`'s band or above: the ones an outranking
    /// offer of `prio` still trades against. The default, every running
    /// decode, is a ground with one band.
    fn decodes_at_or_above(&self, _prio: DecodePriority) -> usize {
        self.decodes_active()
    }

    /// The weight waiting prompts put on prefill throughput when a turn is
    /// judged on the trade against decode ([`rate::WaveRate::try_widen_trading`],
    /// whose other weight is the decodes running): one per prompt waiting, each
    /// counted more the longer it has waited, so prefill's claim on the engine
    /// grows with both how much is waiting and for how long. It reorders
    /// nothing and pre-empts nothing: admitted turns still get the forwards
    /// that finish them, which is how their ground comes back. `prio` is the
    /// band being offered. The default, one, is a lone prompt that has not
    /// waited.
    fn prefill_weight(&self, _prio: DecodePriority) -> f64 {
        1.0
    }

    /// Admit the candidate [`Self::peek`] just priced at `cost`. `false` when the
    /// allocators refused it after all.
    ///
    /// **The judgement is made on the whole turn; the ground is bought as the
    /// turn runs.** The gate has already said the turn's price — all of its K/V
    /// and the tier its deepest chunk stands — stays above the residency the
    /// engine defends. What admission does *not* do is buy that price up front:
    /// each forward's placement concedes its own tier and claims between forwards
    /// (before the forward opens; the boundary refuses to move while one is), so
    /// the experts the later chunks will dislodge keep serving the earlier ones.
    /// Measured on Flash-Next's 128K Strata prefill, buying the whole turn at
    /// admission conceded 6,650 slots at once, the first forwards' hit rate fell
    /// from 95% to 77%, and prefill fell 3.9% (4,592 → 4,414 t/s); at 4K, 3.4%.
    /// Ground the turn's next forward needs is kept from the weight side's
    /// growth in the meantime by the last forward's tier (`GrowthPolicy::spare`).
    fn admit(&mut self, kind: Kind, prio: DecodePriority, cost: Cost) -> bool;

    /// The rate model has judged that more prefill rows will not make the
    /// forward faster: spend no more rows this pass, and offer the rest of the
    /// band as **joins** — turns that share the forwards to come at the width
    /// already admitted rather than widening this one. `true` when the ground
    /// re-offers that way; a ground that cannot ends the band on the refusal.
    fn close_rows(&mut self) -> bool {
        false
    }

    /// File `took` of this fill as `part`, for the window's admission split. A
    /// ground that keeps no split ignores it.
    fn charge(&mut self, part: FillPart, took: Duration) {
        let _ = (part, took);
    }
}

/// The parts of a fill [`Ground::charge`] is told of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FillPart {
    /// Finding the band's next offer and pricing it ([`Ground::peek`]).
    Peek,
    /// Reading the residency an offer is judged against
    /// ([`Ground::resident_weights`]).
    Residency,
    /// The rate model's verdict on an offer, and its refusal's log line.
    Judge,
}
