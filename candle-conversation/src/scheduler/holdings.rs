//! Who is holding the span, how much, and **why** — one snapshot, per slot,
//! with the reason attached.
//!
//! # Why this exists
//!
//! [`super::memory_report`] answers *how much* of each tenant is resident. It
//! cannot answer *why a byte is still there*, and every hard question about
//! this engine turns out to be that second one. A run whose weight zone sits
//! 600 MiB above its hold for four hours, with eviction running on every pass
//! and freeing nothing, is not explained by any total: the totals say the K/V
//! is live, and live is exactly what they would say whether the holder was a
//! conversation mid-reply, a turn merely queued behind ninety others, or a view
//! nobody ever finalised. Those three call for completely different fixes.
//!
//! `demote_idle_slots` decides what it may take by building one `busy` set out
//! of **eleven** different sources. A slot in that set is skipped, and nothing
//! anywhere records which of the eleven put it there. Reconstructing it from a
//! log means correlating `busy_slots` against `queued` and `views` across
//! thousands of lines and inferring — which is guessing with arithmetic on top,
//! and it produced two wrong diagnoses before this module was written.
//!
//! So the census names the holder. Every live slot appears with the tokens and
//! bytes it holds, the substrate residences behind it, whether those are
//! actually evictable, and the list of reasons it is being held. Aggregated by
//! reason, that is a direct answer to "what would I have to change to get this
//! ground back", which no amount of totals can give.
//!
//! # What it costs
//!
//! One pass over the live slots, building small sets from collections the
//! scheduler already holds in memory. No device query and no engine lock; the
//! only per-slot work that leaves the scheduler is one `hot_residency` read per
//! conversation. It is taken **when the engine is short of ground**, not every
//! wave, so a healthy run pays for none of it — and a run that is short is
//! exactly the one worth the line.

use std::collections::{HashMap, HashSet};

use serde::Serialize;

use crate::sequence_handle::SequenceId;

/// Why a slot is being held — one variant per source that can put it in
/// `demote_idle_slots`' busy set, plus the two "not held at all" cases.
///
/// **The order of the variants is the order they are reported in**, and it runs
/// from "genuinely working" to "merely waiting", because that is the axis the
/// reader cares about: everything below [`Holder::PrefillQueued`] is ground
/// held by work that has not started.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Holder {
    /// Stepping in the wave right now.
    ActiveDecode,
    /// Prefilling — admitted and feeding chunks.
    ActivePrefill,
    /// A section ingest in flight.
    ActiveSection,
    /// A member of the creep group the next wave carries.
    WaveMember,
    /// Queued for reprojection; the drain will rebuild it.
    PendingReprojection,
    /// Deferred glue is pending against it.
    DeferredGlue,
    /// A scratch slot the engine owns for the duration of an operation.
    Ephemeral,
    /// A section waiting in the queue — submitted, not admitted.
    SectionQueued,
    /// **A prefill waiting in the queue — submitted, not admitted.** The one
    /// that matters most: this slot is holding K/V for a turn that has not
    /// started and may not start for minutes.
    PrefillQueued,
    /// A live turn view stands on this slot.
    TurnView,
    /// This slot is the *parent* of a live turn view, so the view's borrowed
    /// chunks keep its ground alive.
    ViewParent,
}

impl Holder {
    /// Whether this reason represents work the engine is actually running.
    ///
    /// The complement — queued prefills, queued sections, and the views that
    /// stand on them — is ground held by work that has not begun, which is the
    /// population an eviction pass should be able to reach and currently
    /// cannot.
    ///
    /// # `ViewParent` is deliberately not running, and that is a trap
    ///
    /// A parent carries the same tag whether its view is decoding or sitting in
    /// the prefill queue, because the tag describes *why the ground is held*,
    /// not what the holder is doing. Counting it as running would protect the
    /// parents of queued views — exactly the population `demote_unadmitted_slots`
    /// exists to reclaim — so it stays out.
    ///
    /// The cost is that **a caller asking "is this slot busy?" gets `false` for
    /// the parent of a live decode**, and any set built from
    /// [`running_slots`](super::holdings::running_slots) is therefore keyed on
    /// *views* alone. A consumer that then looks something up by parent id
    /// silently matches nothing: `demote_unadmitted_slots` intersected this set
    /// with `slot_projection_state` (parent-keyed) to build its protect-list,
    /// got an empty list every time, and dropped the hot copies of turns that
    /// were actively decoding. Widen to owners first — see the `running_owners`
    /// mapping there — or ask about the view, never the parent.
    pub fn is_running(self) -> bool {
        matches!(
            self,
            Holder::ActiveDecode
                | Holder::ActivePrefill
                | Holder::ActiveSection
                | Holder::WaveMember
                | Holder::PendingReprojection
                | Holder::DeferredGlue
                | Holder::Ephemeral
        )
    }
}

/// One slot's holding, with the reasons it is held.
#[derive(Debug, Clone, Serialize)]
pub struct SlotHolding {
    pub slot: usize,
    /// Tokens in the slot's block table — the materialised projection.
    pub tokens: usize,
    /// Those tokens priced through the model's live K/V geometry.
    pub kv_bytes: u64,
    /// Substrate residences this conversation holds in VRAM, and their bytes.
    pub hot_residences: usize,
    pub hot_bytes: u64,
    /// Of those, the ones eviction could take right now — a hot copy whose warm
    /// copy has already landed. The rest are hot KV whose migration is still in
    /// flight and which no pass may drop.
    pub evictable_residences: usize,
    pub evictable_bytes: u64,
    /// Whether the model holds this sequence's recurrent state on the card.
    pub recurrent_resident: bool,
    /// Every reason this slot is held, in [`Holder`] order. Empty means idle —
    /// eviction may take it.
    pub holders: Vec<Holder>,
}

impl SlotHolding {
    /// Whether any reason for holding this slot is work actually running.
    pub fn is_running(&self) -> bool {
        self.holders.iter().any(|h| h.is_running())
    }

    /// Held, but by nothing that is running — the population an eviction pass
    /// wants and the busy set currently protects.
    pub fn is_waiting_only(&self) -> bool {
        !self.holders.is_empty() && !self.is_running()
    }
}

/// Bytes and slots attributable to one reason.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct HolderTally {
    pub slots: usize,
    pub tokens: usize,
    pub kv_bytes: u64,
    pub hot_bytes: u64,
    pub evictable_bytes: u64,
}

/// The whole census.
#[derive(Debug, Clone, Serialize)]
pub struct Holdings {
    pub captured_unix_ms: u64,
    /// Every live slot.
    pub slots: Vec<SlotHolding>,
    /// Per-reason totals. A slot with several reasons contributes its bytes to
    /// **each** of them, so these sum to more than the span — they answer "how
    /// much ground would this reason alone keep alive", which is the question
    /// worth asking before removing one.
    pub by_holder: Vec<(Holder, HolderTally)>,
    /// Slots held only by work that has not started, and what they hold.
    pub waiting_only: HolderTally,
    /// Slots nothing holds — what eviction may already take.
    pub idle: HolderTally,
    /// Every slot, running or not.
    pub total: HolderTally,
    /// Slots holding a recurrent store while doing no work — neither decoding
    /// nor prefilling.
    ///
    /// **The invariant this counts is `claim_recurrent`'s own**: "a slot waiting
    /// in the queue holds nothing". Nothing enforces it, and it has now been
    /// broken twice by two different requeue paths — a store is ~160 MiB, so a
    /// handful of them is gigabytes the expert zone cannot grow into, and the
    /// symptom is a zone that simply stops recovering with no error anywhere.
    /// Run 34 measured 7 such slots (~1.1 GiB) against a zone plateaued at
    /// 6,116 MiB. A number in the census is what turns the next occurrence into
    /// something visible rather than something inferred.
    pub idle_recurrent_slots: usize,
    /// Those slots' stores, priced at the model's store size.
    pub idle_recurrent_bytes: u64,
}

impl Holdings {
    /// The census as one line, ordered by how much each reason holds.
    ///
    /// Deliberately compact: the JSON goes to the report, this goes in the log
    /// beside the wave lines so a run's history carries the attribution without
    /// anyone having to fetch anything.
    pub fn summary(&self) -> String {
        let mut parts: Vec<String> = self
            .by_holder
            .iter()
            .filter(|(_, t)| t.slots > 0)
            .map(|(h, t)| {
                format!(
                    "{:?}={}slots/{}MiB",
                    h,
                    t.slots,
                    (t.kv_bytes + t.hot_bytes) >> 20
                )
            })
            .collect();
        parts.sort();
        format!(
            "total={}slots/{}MiB waiting_only={}slots/{}MiB idle={}slots/{}MiB evictable={}MiB \
             idle_stores={}slots/{}MiB | {}",
            self.total.slots,
            (self.total.kv_bytes + self.total.hot_bytes) >> 20,
            self.waiting_only.slots,
            (self.waiting_only.kv_bytes + self.waiting_only.hot_bytes) >> 20,
            self.idle.slots,
            (self.idle.kv_bytes + self.idle.hot_bytes) >> 20,
            self.total.evictable_bytes >> 20,
            self.idle_recurrent_slots,
            self.idle_recurrent_bytes >> 20,
            parts.join(" "),
        )
    }
}

/// Builds a [`Holdings`] from the scheduler's own collections.
///
/// Kept as a builder rather than one long function on `Scheduler` so the
/// attribution is testable without a device or a session: the caller hands over
/// the sets it already has, and the arithmetic that turns them into per-slot
/// reasons is pure.
#[derive(Debug, Default)]
pub struct HoldingsBuilder {
    holders: HashMap<SequenceId, Vec<Holder>>,
}

impl HoldingsBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that `slot` is held for `reason`.
    pub fn hold(&mut self, slot: SequenceId, reason: Holder) {
        let entry = self.holders.entry(slot).or_default();
        if !entry.contains(&reason) {
            entry.push(reason);
        }
    }

    /// Record a whole set at once.
    pub fn hold_all<I: IntoIterator<Item = SequenceId>>(&mut self, slots: I, reason: Holder) {
        for s in slots {
            self.hold(s, reason);
        }
    }

    /// Every slot something claims.
    pub fn held_slots(&self) -> Vec<SequenceId> {
        self.holders.keys().copied().collect()
    }

    /// The reasons recorded for `slot`, in [`Holder`] order.
    pub fn reasons(&self, slot: SequenceId) -> Vec<Holder> {
        let mut v = self.holders.get(&slot).cloned().unwrap_or_default();
        v.sort();
        v
    }

    /// Assemble the census over `slots`, pricing each with `measure`.
    ///
    /// `measure` answers the per-slot quantities the builder cannot know —
    /// tokens, K/V bytes, substrate residency, recurrent residency — so this
    /// stays free of the session and the substrate.
    /// `store_bytes` is what one recurrent store costs, so the census can price
    /// the ones held by slots that are not running — see
    /// [`Holdings::idle_recurrent_slots`].
    pub fn build<F>(
        &self,
        slots: &[SequenceId],
        captured_unix_ms: u64,
        store_bytes: u64,
        mut measure: F,
    ) -> Holdings
    where
        F: FnMut(SequenceId) -> SlotMeasure,
    {
        let mut out: Vec<SlotHolding> = Vec::with_capacity(slots.len());
        for &id in slots {
            let m = measure(id);
            out.push(SlotHolding {
                slot: id.0,
                tokens: m.tokens,
                kv_bytes: m.kv_bytes,
                hot_residences: m.hot_residences,
                hot_bytes: m.hot_bytes,
                evictable_residences: m.evictable_residences,
                evictable_bytes: m.evictable_bytes,
                recurrent_resident: m.recurrent_resident,
                holders: self.reasons(id),
            });
        }

        let mut by: HashMap<Holder, HolderTally> = HashMap::new();
        let (mut waiting, mut idle, mut total) = (
            HolderTally::default(),
            HolderTally::default(),
            HolderTally::default(),
        );
        for s in &out {
            add(&mut total, s);
            if s.holders.is_empty() {
                add(&mut idle, s);
            } else if !s.is_running() {
                add(&mut waiting, s);
            }
            for h in &s.holders {
                add(by.entry(*h).or_default(), s);
            }
        }
        let mut by_holder: Vec<(Holder, HolderTally)> = by.into_iter().collect();
        // Heaviest first — the reader is looking for what to remove.
        by_holder.sort_by(|a, b| {
            (b.1.kv_bytes + b.1.hot_bytes)
                .cmp(&(a.1.kv_bytes + a.1.hot_bytes))
                .then(a.0.cmp(&b.0))
        });

        // A store is justified only while the slot is actually running, and
        // [`Holder::is_running`] is the definition of that — a *section* in
        // flight holds its store exactly as a prefill does, so listing the
        // running holders by hand here reports every live section as a leak.
        // Measured: 22 running sections read as 3,520 MiB of idle stores.
        let idle_recurrent_slots = out
            .iter()
            .filter(|s| s.recurrent_resident && !s.holders.iter().any(|h| h.is_running()))
            .count();

        Holdings {
            captured_unix_ms,
            slots: out,
            by_holder,
            waiting_only: waiting,
            idle,
            total,
            idle_recurrent_slots,
            idle_recurrent_bytes: idle_recurrent_slots as u64 * store_bytes,
        }
    }
}

fn add(t: &mut HolderTally, s: &SlotHolding) {
    t.slots += 1;
    t.tokens += s.tokens;
    t.kv_bytes += s.kv_bytes;
    t.hot_bytes += s.hot_bytes;
    t.evictable_bytes += s.evictable_bytes;
}

/// What the caller measures for one slot.
#[derive(Debug, Clone, Copy, Default)]
pub struct SlotMeasure {
    pub tokens: usize,
    pub kv_bytes: u64,
    pub hot_residences: usize,
    pub hot_bytes: u64,
    pub evictable_residences: usize,
    pub evictable_bytes: u64,
    pub recurrent_resident: bool,
}

/// Slots held only by work that has not started, given a census.
///
/// The candidate set for an eviction pass that is willing to take ground back
/// from the queue — see `Scheduler::demote_unadmitted_slots`. Pure over the
/// census so the selection is testable without a device.
pub fn waiting_only_slots(h: &Holdings) -> Vec<SequenceId> {
    h.slots
        .iter()
        .filter(|s| s.is_waiting_only())
        .map(|s| SequenceId(s.slot))
        .collect()
}

/// Slots the census says are running, as a set — the keep-list an eviction pass
/// must never cross.
pub fn running_slots(h: &Holdings) -> HashSet<SequenceId> {
    h.slots
        .iter()
        .filter(|s| s.is_running())
        .map(|s| SequenceId(s.slot))
        .collect()
}

/// Latest census, for the HTTP surface — same process-global slot pattern as
/// [`super::memory_report`], so `zend` can serve it without an engine lock.
static LATEST: std::sync::OnceLock<std::sync::Mutex<Option<Holdings>>> = std::sync::OnceLock::new();

fn slot() -> &'static std::sync::Mutex<Option<Holdings>> {
    LATEST.get_or_init(|| std::sync::Mutex::new(None))
}

/// Store `h` as the latest census.
pub fn publish(h: Holdings) {
    *slot().lock().unwrap() = Some(h);
}

/// The latest census, if one has been taken.
pub fn latest() -> Option<Holdings> {
    slot().lock().unwrap().clone()
}

impl super::Scheduler {
    /// Take the holdings census.
    ///
    /// **Mirrors `demote_idle_slots`' busy set source for source**, which is
    /// the entire point: if the two ever disagree, the census is explaining a
    /// decision the engine did not make. They are written adjacently and any
    /// change to one belongs in the other.
    pub(super) fn census(&self) -> Holdings {
        use super::WaveMember;

        let mut b = HoldingsBuilder::new();
        b.hold_all(self.active_decodes.keys().copied(), Holder::ActiveDecode);
        b.hold_all(
            self.active_prefills.iter().map(|p| p.work.sequence_id),
            Holder::ActivePrefill,
        );
        b.hold_all(
            self.active_section_ingests.iter().map(|s| s.sequence_id),
            Holder::ActiveSection,
        );
        b.hold_all(
            self.section_queue.iter().map(|s| s.sequence_id),
            Holder::SectionQueued,
        );
        b.hold_all(
            self.prefill_queue.iter().map(|w| w.sequence_id),
            Holder::PrefillQueued,
        );
        b.hold_all(
            self.pending_reprojections.iter().copied(),
            Holder::PendingReprojection,
        );
        b.hold_all(
            self.deferred_glue_fires.iter().map(|p| p.parent_id),
            Holder::DeferredGlue,
        );
        b.hold_all(self.ephemeral_slots.iter().copied(), Holder::Ephemeral);
        for (view, st) in &self.turn_views {
            b.hold(*view, Holder::TurnView);
            b.hold(st.parent_id, Holder::ViewParent);
        }
        for m in &self.wave_prefill_members {
            let seq_id = match m {
                WaveMember::Prefill { seq_id, .. } | WaveMember::Section { seq_id, .. } => *seq_id,
            };
            b.hold(SequenceId(seq_id), Holder::WaveMember);
        }

        // **Every live slot, and every slot anything claims.** The conversation
        // map is the authoritative list of live slots, but a slot held by
        // something that has no row there would otherwise be invisible —
        // and an invisible holder is exactly the failure this module exists to
        // end. The union costs nothing and cannot under-report.
        let mut seen: HashSet<SequenceId> = self.slot_conversations.keys().copied().collect();
        seen.extend(b.held_slots());
        let mut slots: Vec<SequenceId> = seen.into_iter().collect();
        slots.sort_by_key(|s| s.0);
        let per_block = self.per_block_kv_bytes();
        let captured = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        b.build(
            &slots,
            captured,
            self.model.recurrent_store_bytes() as u64,
            |id| {
                let tokens = self.session.sequence_offset(id.0).unwrap_or(0);
                let blocks = tokens.div_ceil(candle_nn::kv_cache::CHUNK_SIZE) as u64;
                let (hot_residences, hot_bytes, evictable_residences, evictable_bytes) = self
                    .slot_conversations
                    .get(&id)
                    .map(|c| c.read().hot_residency())
                    .unwrap_or((0, 0, 0, 0));
                SlotMeasure {
                    tokens,
                    kv_bytes: blocks.saturating_mul(per_block),
                    hot_residences,
                    hot_bytes,
                    evictable_residences,
                    evictable_bytes,
                    recurrent_resident: self.model.recurrent_resident(id.0),
                }
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One recurrent store, as the 35B measures it.
    const STORE: u64 = 160 << 20;

    fn m(tokens: usize, hot: u64, evictable: u64) -> SlotMeasure {
        SlotMeasure {
            tokens,
            kv_bytes: tokens as u64 * 1024,
            hot_residences: if hot > 0 { 1 } else { 0 },
            hot_bytes: hot,
            evictable_residences: if evictable > 0 { 1 } else { 0 },
            evictable_bytes: evictable,
            recurrent_resident: false,
        }
    }

    /// A slot measured with a recurrent store on the card.
    fn with_store(tokens: usize) -> SlotMeasure {
        SlotMeasure {
            recurrent_resident: true,
            ..m(tokens, 0, 0)
        }
    }

    /// **A store is only justified while the slot is running.** `claim_recurrent`
    /// places one at admission on the stated rule that "a slot waiting in the
    /// queue holds nothing", and nothing enforces that — it has been broken by
    /// two different requeue paths, each time leaving ~160 MiB per slot standing
    /// in ground the expert zone needed, with no error anywhere. Run 34 measured
    /// 7 such slots against a zone plateaued at 6,116 MiB.
    #[test]
    fn a_store_held_by_a_slot_that_is_not_running_is_counted() {
        let mut b = HoldingsBuilder::new();
        let queued = SequenceId(1);
        let parent = SequenceId(2);
        let decoding = SequenceId(3);
        let prefilling = SequenceId(4);
        let sectioning = SequenceId(5);
        b.hold(queued, Holder::PrefillQueued);
        b.hold(parent, Holder::ViewParent);
        b.hold(decoding, Holder::ActiveDecode);
        b.hold(prefilling, Holder::ActivePrefill);
        b.hold(sectioning, Holder::ActiveSection);

        let h = b.build(
            &[queued, parent, decoding, prefilling, sectioning],
            0,
            STORE,
            |_| with_store(0),
        );

        assert_eq!(
            h.idle_recurrent_slots, 2,
            "the queued turn and the parent are holding stores for no work"
        );
        assert_eq!(h.idle_recurrent_bytes, 2 * STORE);
        assert!(h.summary().contains("idle_stores=2slots/320MiB"));
    }

    /// **A running section holds its store legitimately.** It keeps its K/V, its
    /// store and its tier rows until it seals, exactly as a prefill does — and
    /// counting it as idle reported 22 live sections as 3,520 MiB of leak, which
    /// is what a hand-written list of "running" holders costs.
    #[test]
    fn a_running_section_is_not_an_idle_store() {
        let mut b = HoldingsBuilder::new();
        let s = SequenceId(1);
        b.hold(s, Holder::ActiveSection);
        let h = b.build(&[s], 0, STORE, |_| with_store(500));
        assert_eq!(h.idle_recurrent_slots, 0);
        assert_eq!(h.idle_recurrent_bytes, 0);
    }

    /// A slot with no store contributes nothing, whatever holds it.
    #[test]
    fn a_slot_without_a_store_is_not_counted() {
        let mut b = HoldingsBuilder::new();
        let queued = SequenceId(1);
        b.hold(queued, Holder::PrefillQueued);
        let h = b.build(&[queued], 0, STORE, |_| m(0, 0, 0));
        assert_eq!(h.idle_recurrent_slots, 0);
        assert_eq!(h.idle_recurrent_bytes, 0);
    }

    /// **The census names every reason, not the first one found.** A slot held
    /// by three things is three different fixes; reporting one of them is how
    /// the wrong one gets removed.
    #[test]
    fn a_slot_carries_every_reason_it_is_held_for() {
        let mut b = HoldingsBuilder::new();
        let s = SequenceId(7);
        b.hold(s, Holder::PrefillQueued);
        b.hold(s, Holder::ViewParent);
        b.hold(s, Holder::ActiveDecode);
        assert_eq!(
            b.reasons(s),
            vec![
                Holder::ActiveDecode,
                Holder::PrefillQueued,
                Holder::ViewParent
            ],
            "reported in Holder order: running first, waiting last",
        );
        // Recorded twice is recorded once.
        b.hold(s, Holder::PrefillQueued);
        assert_eq!(b.reasons(s).len(), 3);
    }

    /// **"Waiting only" is the population that matters**, and it is not the
    /// same as "not running": a slot that is both decoding and queued is
    /// running, and taking its ground would be wrong.
    #[test]
    fn waiting_only_excludes_anything_also_running() {
        let mut b = HoldingsBuilder::new();
        let queued = SequenceId(1);
        let both = SequenceId(2);
        let idle = SequenceId(3);
        b.hold(queued, Holder::PrefillQueued);
        b.hold(both, Holder::PrefillQueued);
        b.hold(both, Holder::ActiveDecode);
        let h = b.build(&[queued, both, idle], 0, STORE, |_| m(100, 0, 0));

        assert_eq!(waiting_only_slots(&h), vec![queued]);
        assert!(running_slots(&h).contains(&both));
        assert!(!running_slots(&h).contains(&queued));
        assert_eq!(h.waiting_only.slots, 1);
        assert_eq!(h.idle.slots, 1, "held by nothing at all");
        assert_eq!(h.total.slots, 3);
    }

    /// A slot's bytes count toward **every** reason holding it, so a tally
    /// answers "how much would this reason alone keep alive" rather than
    /// splitting the ground arbitrarily between co-holders.
    #[test]
    fn tallies_attribute_the_whole_slot_to_each_of_its_reasons() {
        let mut b = HoldingsBuilder::new();
        let s = SequenceId(1);
        b.hold(s, Holder::PrefillQueued);
        b.hold(s, Holder::TurnView);
        let h = b.build(&[s], 0, STORE, |_| m(1_000, 0, 0));

        let get = |want: Holder| h.by_holder.iter().find(|(x, _)| *x == want).unwrap().1;
        assert_eq!(get(Holder::PrefillQueued).kv_bytes, 1_024_000);
        assert_eq!(get(Holder::TurnView).kv_bytes, 1_024_000);
        assert_eq!(h.total.kv_bytes, 1_024_000, "the span counts it once");
    }

    /// **Evictable is not the same as hot**, and conflating them is what makes
    /// an eviction pass that frees nothing indistinguishable from one with
    /// nothing to free. Hot KV whose warm copy has not landed cannot be taken.
    #[test]
    fn the_census_separates_hot_from_what_can_actually_be_taken() {
        let mut b = HoldingsBuilder::new();
        let s = SequenceId(1);
        b.hold(s, Holder::PrefillQueued);
        let h = b.build(&[s], 0, STORE, |_| m(0, 900 << 20, 300 << 20));
        assert_eq!(h.total.hot_bytes, 900 << 20);
        assert_eq!(
            h.total.evictable_bytes,
            300 << 20,
            "600 MiB still migrating"
        );
        assert!(h.summary().contains("evictable=300MiB"));
    }

    /// The heaviest reason is reported first, because the reader is looking for
    /// what to change.
    #[test]
    fn reasons_are_ordered_by_how_much_they_hold() {
        let mut b = HoldingsBuilder::new();
        let (small, big) = (SequenceId(1), SequenceId(2));
        b.hold(small, Holder::ActiveDecode);
        b.hold(big, Holder::PrefillQueued);
        let h = b.build(&[small, big], 0, STORE, |id| {
            if id == big {
                m(10_000, 0, 0)
            } else {
                m(10, 0, 0)
            }
        });
        assert_eq!(h.by_holder[0].0, Holder::PrefillQueued);
    }

    /// An engine holding nothing reports nothing, rather than an empty shape
    /// that reads like a measurement failure.
    #[test]
    fn an_empty_engine_censuses_cleanly() {
        let h = HoldingsBuilder::new().build(&[], 0, STORE, |_| SlotMeasure::default());
        assert_eq!(h.total.slots, 0);
        assert!(h.by_holder.is_empty());
        assert!(waiting_only_slots(&h).is_empty());
    }
}
