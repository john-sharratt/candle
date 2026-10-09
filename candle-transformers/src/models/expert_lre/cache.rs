//! Expert cache bookkeeping — slot management, eviction policy, score-based.
//!
//! This module contains [`ExpertCacheInner`], the mutable bookkeeping
//! structure that tracks which experts are resident in VRAM, manages
//! slot allocation, and implements the score-based eviction policy.
//!
//! ## Eviction policy
//!
//! Evict behind the wave, frequency within. In brief:
//!
//! 1. **Ranked victims, taken by the caller** — the promotion ring's lazy
//!    victims are ranked by [`ExpertCacheInner::rank_victims`] in one scan,
//!    scored at the row the pipeline is serving, and a claim — a demand miss
//!    or a read-ahead copy — takes the next one. Eviction is a retarget (the
//!    cold pack holds every expert; the warm tier is immutable), so there is no
//!    copy to hide and nothing to do ahead of time.
//! 2. **Behind the wave first, fading** — each resident's frequency is scaled
//!    by how far behind the wave its layer lies ([`behind_protection`]): L−1
//!    by [`FADE_FLOOR`], each layer further back a little more, wrapping round
//!    to the layers about to run, which keep nearly their whole score. At the
//!    start of a pass the eviction front is the previous pass's last layers;
//!    as the wave advances it moves to the early layers it has just left.
//! 3. **Early-layer pinning** — the first [`PINNED_LAYERS`] layers are never
//!    evicted (they run first every pass with no compute to hide a reload).
//! 4. **Eviction-class passes** — the residents fall into four classes,
//!    reload tier (warm-backed or NVMe-pack-only) × whether their row routed
//!    them on its last visit, and each of the above runs over one class at a
//!    time, a later class only for what the earlier could not free. The
//!    classes run cheapest expected miss cost first
//!    ([`ExpertCacheInner::set_eviction_costs`]: regret rate × restore cost),
//!    and in [`EVICTION_PASSES`]'s order — warm-backed first, stale before
//!    just-routed — until those costs are measured.
//!
//! ## Score table
//!
//! A flat `Vec<f32>` indexed by `layer * experts_per_layer + expert` records a
//! lightly-decayed access frequency: higher = more valuable = evicted last.
//! Updated by pipeline events:
//!
//! - **Cache hit (decode-attributed)**: +1.0, and +1.0 of decode reuse
//! - **Miss (decode-attributed)**: +1.0, no decode reuse
//! - **Cache hit (prefill-attributed)**: +0.1 — see [`PREFILL_HIT_SCORE`]
//! - **Prefill-attributed elevation** (a fresh cold load streamed in to serve a
//!   prefill row): set to −0.1 — see [`PREFILL_ELEVATE_PENALTY`]
//! - **Prediction hit**: +0.3 (a speculative load the layer actually routed to)
//! - **End-of-pass decay**: ×[`DECODE_RECENCY_DECAY`] (recency-weighting of the
//!   frequency); decode reuse ×[`DECODE_REUSE_DECAY`]
//!
//! The eviction score is the recency-weighted frequency plus
//! [`DECODE_REUSE_WEIGHT`] × decode reuse: a short memory that keeps the
//! experts the current decode steps are routing, and a long one that keeps the
//! experts decode keeps coming back to.

use super::regret::class_index;
#[cfg(feature = "tensor-assert")]
use super::slot_owners::SlotOwners;
use super::types::ExpertSlot;
use candle_nn::kv_cache::WeightZone;
use std::cmp::Ordering;
use std::collections::HashMap;
#[cfg(feature = "tensor-assert")]
use std::sync::Arc;

/// Number of early MoE layers whose experts are never evicted.
///
/// These layers run first every pass and have zero compute to overlap
/// with DMA — evicting them guarantees cold misses with maximum stall.
/// A single decode step routes top-8 per layer, so pinning layers 0–1 holds
/// ~16 experts; a batch wide enough to route everywhere holds all of them, which
/// is what [`minimum_resident_slots`] prices.
///
/// # Why this is a constant and not a function of capacity
///
/// It used to be derived from the zone's live capacity, so a small card
/// degraded to less pinning rather than to a cache that could not evict. That
/// is no longer available, because the pinned set is now **the set of experts
/// with no host or disk copy at all**: a permanently-resident expert is never
/// reloaded, so [`pack`](super::pack) omits its record and the warm tier omits
/// it from the draw. A pinned count that varied with capacity would make a
/// pack built on one machine unreadable on another — the second machine would
/// unpin a layer whose records were never written and then fail every load
/// against it.
///
/// So the depth is fixed, and the *floor* is what guarantees it can be paid:
/// [`minimum_resident_slots`] prices the pinned set plus a full working layer,
/// and the zone may never retract below it.
pub const PINNED_LAYERS: usize = 2;

/// Score credit for a prefill-attributed hit on an already-resident expert.
///
/// Far below [`ExpertCacheInner::record_hit`]'s +1.0: a prefill row's own
/// reuse of a specific expert across waves is close to zero (a long prefill
/// sweep touches most of the table roughly once each), so treating a prefill
/// hit as equally valuable as a decode hit let a diverse enough prefill
/// out-bid decode's genuinely-reused residents for warm/hot slots — decode
/// then had to re-earn its own working set from a cache full of prefill's
/// one-shot leftovers. This still lets an expert that prefill itself reuses
/// often (a broadly-useful "generalist") earn some protection, just far less
/// than a resident decode actually depends on.
pub const PREFILL_HIT_SCORE: f32 = 0.1;

/// Score penalty applied when a prefill row causes a *fresh* elevation (a
/// cold miss that streams an expert in), as opposed to a hit on something
/// already resident.
///
/// Negative, not merely small: the expert a prefill row just paid a full DMA
/// to load is, on the evidence, the LEAST likely thing in the zone to be
/// touched again before the pass wraps — the layer it serves has just been
/// left behind, and prefill's own within-sweep reuse of one specific expert
/// is close to zero. Pushing its score negative (rather than leaving it at
/// whatever it decayed to from an earlier, unrelated occupancy) makes it the
/// eviction scan's preferred victim immediately, freeing the slot for the
/// layer about to run instead of leaving it to be found by the normal
/// low-score-first scan.
///
/// **A negative score must never be multiplied by a protection factor.** Every
/// such factor is positive, so multiplying makes a negative base *more*
/// negative — the better-protected slot sorts lower and is evicted first, which
/// is backwards. Each eviction key therefore goes through
/// [`ExpertCacheInner::protected_by`], which divides below zero; that also keeps
/// the sign a hard tier, so an elevated expert outranks nothing in its pass
/// however far behind or ahead of the wave it lies ([`behind_protection`]).
pub const PREFILL_ELEVATE_PENALTY: f32 = -0.1;

/// Per-pass decay of the recency-weighted score: a memory of about ten passes.
pub const DECODE_RECENCY_DECAY: f32 = 0.85;

/// Per-pass decay of [`ExpertCacheInner::decode_reuse`]: a memory of about a
/// hundred decode steps, where the recency score's ×0.85 forgets in about ten.
///
/// One score cannot serve both needs a decode miss has. Credited as a hit, the
/// expert holds its slot until the next step routes to it again — without that
/// credit Qwen3.8-Flash-Next's wide decode loses a third (RTX 3090, ×16: 368 →
/// 249 t/s) — but it then outbids every expert decode last routed more than a
/// few steps ago, and a reply that returns to them pays for all of them again
/// (a 13-step two-sequence row: 155 → 120 t/s, its first step 57 → 107 ms).
/// A slower decay of the one score keeps those and loses the wide rows instead
/// (×0.99: ×16 308 t/s). Reuse earned only by hits, decayed slowly and
/// weighted in, keeps both. ×0.98 measured 1–3% under ×0.99 on most rows.
pub const DECODE_REUSE_DECAY: f32 = 0.99;

/// Weight of [`ExpertCacheInner::decode_reuse`] in the eviction score.
///
/// Measured on Qwen3.8-Flash-Next (RTX 3090): 0.02, 0.04 and 0.1 each raised
/// every decode row a little over the one before; 0.3 raised the two- and
/// eight-sequence rows another 1–2% and cost ×16 7% (372 → 347 t/s), the row
/// whose working set most exceeds the zone, where old reuse is least worth
/// keeping.
pub const DECODE_REUSE_WEIGHT: f32 = 0.1;

/// How many layers this model actually pins.
///
/// [`PINNED_LAYERS`], except for a model with fewer MoE layers than that — in
/// which case every layer is pinned and there is no evictable set: every
/// expert stays in VRAM and none needs a reload path.
///
/// Both the pack writer and the warm-tier draw derive their skip from this, so
/// a single number decides which experts have a reload path.
pub(crate) fn pinned_layer_count(num_moe_layers: usize) -> usize {
    PINNED_LAYERS.min(num_moe_layers)
}

/// The fewest slots the cache can serve a token with, for a model with
/// `experts_per_layer` experts in each MoE layer.
///
/// **The eviction scan cannot touch layers `0..PINNED_LAYERS`.** A batch wide
/// enough to route to every expert in those layers fills
/// `PINNED_LAYERS × experts_per_layer` slots that no victim search will ever
/// select. Give the zone fewer slots than that and it can reach a state where
/// every resident slot holds a pinned-layer expert: the `layer >= pinned_layers`
/// filter in [`ExpertCacheInner::rank_victims`] matches nothing, and no
/// promotion can take a slot from then on — for the life of the process,
/// because nothing in that state can ever free one. Fewer slots than the pinned
/// set itself strands a pinned expert outright: it has no warm or pack copy to
/// be served from.
///
/// The daemon reached it: the boundary retracted to 297 slots against a
/// pinned-eligible set of 384, and the next 1,774 wave steps all failed
/// identically. This is the number that must never be crossed, and since the
/// boundary is otherwise free to trade expert residency for KV ground on demand,
/// it is the **only** thing standing between a hungry KV side and a dead engine.
///
/// Priced as the pinned set **plus a full working layer, plus one**, because
/// [`PINNED_LAYERS`] is fixed: nothing sheds pinned layers under pressure, so
/// this floor is all that keeps the eviction scan supplied with candidates. A
/// zone sitting exactly on it holds every pinned expert, a worst-case routed set
/// for the layer executing, and one slot for the load that triggered the
/// eviction to land in.
///
/// The arithmetic is unchanged from when pinning was three capacity-derived
/// layers and this priced only `3 × n + 1`: two pinned layers plus a working
/// layer is the same `3 × n`. So no model's boundary placement moves — which
/// matters, because this number feeds the opening size of every MoE model and
/// raising it starves the KV side (Qwen3-30B OOMs at load).
pub fn minimum_resident_slots(experts_per_layer: usize) -> usize {
    (PINNED_LAYERS + 1) * experts_per_layer + 1
}

/// Where an evicted expert's next miss is served from.
///
/// The warm tier and the NVMe pack hold disjoint experts, and the model's
/// routing is trained balanced, so which experts VRAM holds barely moves its
/// hit rate — it decides where the misses land. Every slot VRAM spends on a
/// warm-backed expert leaves one more pack-only expert to miss on disk (a
/// 2.9 MB page-cache-bypassing read near a millisecond, against ~116 µs H2D
/// from pinned host memory). So every eviction runs its policy one reload tier
/// at a time — warm-backed first by default ([`EVICTION_PASSES`]), and in the
/// order of each class's measured expected miss cost once that is known
/// ([`ExpertCacheInner::set_eviction_costs`]), which prices exactly this
/// difference in restore cost.
///
/// Measured on Flash-Next at 16 GB against the single-pass policy: pack share
/// of loads 68 % → 61 / 63 / 50 % (BF16×1 / BF16×8 / C5×2), decode 16.2 /
/// 72.1 / 30.1 → 16.2 / 74.4 / 32.1 t/s, for a hit rate 2–3 points lower.
/// Protecting the hottest 20 % or 40 % of residents from the first pass
/// recovered under a point of that hit rate, gave back the pack saving about
/// as fast, and decoded no faster — the frequency signal is spread across the
/// warm-backed experts, not concentrated at the top, and a warm miss is cheap
/// enough that the extra ones cost less than the pack reads they replace.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ReloadTier {
    /// A warm (host) copy exists — the cheap reload.
    Warm,
    /// Only the NVMe pack holds it.
    Nvme,
}

/// The order eviction passes visit residents in until their costs are measured
/// ([`ExpertCacheInner::set_eviction_costs`]): each reload tier, and within it
/// first the experts their row did **not** route on its last visit, then the
/// ones it did (`hit_last`).
///
/// **Recency is a pass, not a weight.** Decode routing carries over from one
/// step to the next: of the experts a row routed on its last visit, 37–63% are
/// routed again on its next (Qwen3.8-Flash-Next, RTX 4090 Laptop, decode ×1 to
/// ×8), against 10–21% of the rest. The decayed frequency score cannot see
/// that — a demand miss just installed scores about 1, a resident hit a few
/// passes ago still 2–3 — so ranked by score alone those just-hit experts were
/// 54% of the victims at ×1 and 82% at ×8, and accounted for most of the
/// evictions their rows then missed. As a pass they go only once every other
/// resident of the tier has.
///
/// **Inside the reload tier, not above it**, because the costs say so: a warm
/// miss reloads in ~116 µs and a pack-only one in ~1 ms, so a warm just-hit
/// victim (37–63% back × 116 µs) is still cheaper than a pack-only cold one
/// (10–21% × 1 ms).
const EVICTION_PASSES: [(ReloadTier, bool); 4] = [
    (ReloadTier::Warm, false),
    (ReloadTier::Warm, true),
    (ReloadTier::Nvme, false),
    (ReloadTier::Nvme, true),
];

/// A pass's eviction class, as the ledger indexes them (`regret`).
fn pass_class((tier, hit): (ReloadTier, bool)) -> usize {
    class_index(hit, tier == ReloadTier::Warm)
}

/// One eviction candidate as the batch scans rank it within a pass.
#[derive(Clone, Copy)]
struct VictimKey {
    slot: usize,
    /// Frequency, with whatever positional factor the scan applies.
    score: f32,
    /// Wrapped forward distance from the current layer.
    dist: usize,
    lru: u32,
}

impl VictimKey {
    /// Best victim first: lowest score, then farthest, then least recently
    /// used.
    fn order(a: &Self, b: &Self) -> Ordering {
        a.score
            .partial_cmp(&b.score)
            .unwrap_or(Ordering::Equal)
            .then(b.dist.cmp(&a.dist))
            .then(a.lru.cmp(&b.lru))
    }
}

/// How well a resident at wrapped forward distance `dist` from the wave is
/// protected, of `n` layers: [`FADE_FLOOR`] plus the rest in proportion to the
/// layers it lies *behind* the wave, `(n − dist) / n`. L−1 (`dist = n − 1`) is
/// protected least, each layer further back a little more, through the wrap,
/// up to the layers about to run (L+1) and the one being computed (1).
///
/// The layer just executed is the one whose next use is furthest away — a full
/// pass — so it is evicted first, and the preference fades the further back a
/// layer is. Wrapping is what moves the eviction front with the wave: at the
/// start of a pass the layers just behind are the previous pass's last, and as
/// the wave advances it evicts from the early layers it has just left.
fn behind_protection(dist: usize, n: usize) -> f32 {
    FADE_FLOOR + (1.0 - FADE_FLOOR) * (n - dist) as f32 / n as f32
}

/// The least protection [`behind_protection`] gives, at L−1: the fade spans
/// about `1 / FADE_FLOOR` (~4×) from the layer just behind to the one about to
/// run, so distance steers eviction without overruling frequency.
///
/// Measured on Qwen3.8-Flash-Next (RTX 4090 Laptop), as the share of evicted
/// experts their row routed again at its next visit (decode ×1 / ×4 / ×8 /
/// C5×8, two rounds each): a floor of 0 — the full `1/n … 1` fade, ~48× — rose
/// to 26.3 / 38.4 / 58.6 / 54.5% from the flat 2× window's 25.8 / 36.8 / 55.6 /
/// 51.3%, evicting hot experts just behind that the next pass needed; 0.5 gave
/// 26.3 / 37.6 / 54.3 / 50.6%, and 0.25 25.5 / 36.4 / 54.3 / 49.5%, the least.
/// Against the flat window, order-balanced over four rounds, 0.25 decodes level
/// on every row but C5×8, where it runs ~7% slower (83.1 → 77.4 t/s) with ~40%
/// more read-ahead claims.
const FADE_FLOOR: f32 = 0.25;

/// Mutable bookkeeping owned exclusively by the pipeline thread.
///
/// All fields are plain data — no `Arc`, no atomic types.
///
/// ## Slot lifecycle
///
/// Each slot is either free (in the zone's free list), or occupied (has an
/// `ExpertSlot` and a `slot_to_key` entry).  Occupied slots have a
/// `last_used` timestamp that determines eviction order.
///
/// ```text
/// Free:     slots[i] = None,  slot_to_key[i] = None
/// Occupied: slots[i] = Some(ExpertSlot), slot_to_key[i] = Some((moe, exp))
/// ```
///
/// ## Where a slot's bytes come from
///
/// The [`WeightZone`] owns the addresses. It is the right-hand side of the
/// device reservation, and it is also **the free list** — there is no second one
/// here. That matters for more than tidiness: the zone hands out the *rightmost*
/// free slot, which keeps live experts packed away from the boundary the KV side
/// pushes against, and a duplicate `Vec` free list here would have silently
/// undone that ordering on every eviction (`push` puts the freed index on top,
/// so the next load takes it back regardless of where it sits).
pub struct ExpertCacheInner {
    /// VRAM slots — created on-demand, indexed by slot_idx.
    /// **No Arc wrapping** — sole ownership.
    pub(crate) slots: Vec<Option<ExpertSlot>>,
    /// The addresses, and the free list over them.
    pub(crate) zone: WeightZone,
    /// Forward lookup: `(moe_layer_idx, expert_idx) -> slot_idx`.
    pub(crate) key_to_slot: HashMap<(usize, usize), usize>,
    /// Per-slot usage timestamp — higher = more recently used.
    /// Kept for recency tie-breaking within score-based eviction.
    pub(crate) last_used: Vec<u32>,
    /// Monotonically increasing counter, bumped on each cache access.
    pub(crate) generation: u32,
    /// Reverse map: `slot_idx -> (moe_layer_idx, expert_idx)` for eviction.
    pub(crate) slot_to_key: Vec<Option<(usize, usize)>>,

    // ── Score-based eviction state ──
    /// Flat score table: `expert_scores[layer * experts_per_layer + expert]`.
    /// A lightly-decayed access frequency — higher = more valuable = evicted last.
    pub(crate) expert_scores: Vec<f32>,
    /// Long-memory decode reuse, indexed like `expert_scores`: decode hits
    /// only, decayed by [`DECODE_REUSE_DECAY`] a pass. Weighted into
    /// [`Self::score`] by [`DECODE_REUSE_WEIGHT`].
    pub(crate) decode_reuse: Vec<f32>,
    /// Number of MoE layers (e.g. 48).
    pub(crate) num_moe_layers: usize,
    /// Experts per MoE layer (e.g. 128).
    pub(crate) experts_per_layer: usize,
    /// Early MoE layers exempt from eviction — [`pinned_layer_count`], fixed
    /// for the life of the cache and independent of capacity.
    ///
    /// It cannot track the boundary, because these are exactly the experts the
    /// pack and the warm tier do not store: unpinning one would leave it with
    /// nowhere to reload from. The zone's floor ([`minimum_resident_slots`]) is
    /// what guarantees the cache can always afford it.
    pub(crate) pinned_layers: usize,
    /// Flat `layer * experts_per_layer + expert` → does a warm (pinned host)
    /// copy of this expert exist?
    ///
    /// Written once when the warm tier is filled and never again — the tier is
    /// immutable, which is what lets the eviction policy treat this as a
    /// property of the expert rather than something to re-check.
    pub(crate) warm_backed: Vec<bool>,
    /// Flat `layer * experts_per_layer + expert` → did this expert's row route
    /// it on the row's last visit? Rewritten per row as each is served
    /// ([`Self::mark_visit`]); the recency half of [`EVICTION_PASSES`].
    hit_last: Vec<bool>,
    /// The order the eviction passes run in: [`EVICTION_PASSES`] until
    /// [`Self::set_eviction_costs`] orders them by expected miss cost.
    passes: [(ReloadTier, bool); 4],
    /// The slot-tenancy tags bucketize's owner check reads, once a device
    /// cache has attached them ([`Self::attach_owners`]); every tenant a slot
    /// gains after that is written through.
    #[cfg(feature = "tensor-assert")]
    owners: Option<Arc<SlotOwners>>,
}

impl ExpertCacheInner {
    /// Create a new empty cache over `zone`'s slots.
    ///
    /// * `num_moe_layers` — total MoE layers (e.g. 48)
    /// * `experts_per_layer` — experts per layer (e.g. 128)
    pub(crate) fn new(zone: WeightZone, num_moe_layers: usize, experts_per_layer: usize) -> Self {
        let num_slots = zone.capacity();
        Self {
            slots: (0..num_slots).map(|_| None).collect(),
            zone,
            key_to_slot: HashMap::new(),
            last_used: vec![0u32; num_slots],
            generation: 0,
            slot_to_key: vec![None; num_slots],
            expert_scores: vec![0.0f32; num_moe_layers * experts_per_layer],
            decode_reuse: vec![0.0f32; num_moe_layers * experts_per_layer],
            num_moe_layers,
            experts_per_layer,
            pinned_layers: pinned_layer_count(num_moe_layers),
            warm_backed: vec![false; num_moe_layers * experts_per_layer],
            hit_last: vec![false; num_moe_layers * experts_per_layer],
            passes: EVICTION_PASSES,
            #[cfg(feature = "tensor-assert")]
            owners: None,
        }
    }

    /// Tag every slot with its tenant, and write through every tenant a slot
    /// gains from here on — once the startup fill has installed its experts.
    #[cfg(feature = "tensor-assert")]
    pub(crate) fn attach_owners(&mut self, owners: Arc<SlotOwners>) {
        for (slot, key) in self.slot_to_key.iter().enumerate() {
            if let Some((row, expert)) = *key {
                owners.set(slot, row, expert);
            }
        }
        self.owners = Some(owners);
    }

    /// Slot `slot` gained the tenant `slot_to_key` names: write its tag.
    #[cfg(feature = "tensor-assert")]
    pub(crate) fn mirror_tenant(&self, slot: usize) {
        if let (Some(owners), Some((row, expert))) = (&self.owners, self.slot_to_key[slot]) {
            owners.set(slot, row, expert);
        }
    }

    /// Record which experts the warm tier holds, once its fill is decided.
    ///
    /// Called before the first forward and never again.
    pub(crate) fn set_warm_backed(&mut self, membership: &[(usize, usize)]) {
        self.warm_backed.iter_mut().for_each(|b| *b = false);
        for &(layer, expert) in membership {
            let idx = layer * self.experts_per_layer + expert;
            if idx < self.warm_backed.len() {
                self.warm_backed[idx] = true;
            }
        }
    }

    /// Row `layer` was served, routing `routed`: those of its experts are now
    /// the ones it hit on its last visit, and no others.
    pub(crate) fn mark_visit(&mut self, layer: usize, routed: &[usize]) {
        let row = layer * self.experts_per_layer;
        self.hit_last[row..row + self.experts_per_layer].fill(false);
        for &e in routed {
            self.hit_last[row + e] = true;
        }
    }

    /// Order the eviction passes by each class's expected miss cost, cheapest
    /// first: `cost[hit * 2 + warm]` is how often a victim of that class is
    /// routed again at its row's next visit times what restoring it costs.
    /// Ties keep [`EVICTION_PASSES`]'s order, which takes the warm-backed
    /// first.
    ///
    /// Restoring a pack-only expert costs the drive read *and* the copy that
    /// follows, against the copy alone for a warm-backed one, so pack-only
    /// classes are protected by that ratio — about an order of magnitude on an
    /// NVMe pack — and an eviction takes one only when it is that much less
    /// likely to be wanted back.
    pub(crate) fn set_eviction_costs(&mut self, cost: [f64; 4]) {
        let mut passes = EVICTION_PASSES;
        passes.sort_by(|&a, &b| cost[pass_class(a)].total_cmp(&cost[pass_class(b)]));
        self.passes = passes;
    }

    /// Whether `layer`'s row routed `expert` on its last visit.
    #[inline]
    pub(crate) fn hit_last_visit(&self, layer: usize, expert: usize) -> bool {
        self.hit_last[layer * self.experts_per_layer + expert]
    }

    /// Whether a warm (host) copy backs `(layer, expert)` — its cheap reload.
    #[inline]
    pub(crate) fn is_warm_backed(&self, layer: usize, expert: usize) -> bool {
        self.reload_tier(layer, expert) == ReloadTier::Warm
    }

    /// Where `(layer, expert)` reloads from after an eviction — see
    /// [`ReloadTier`] for why eviction passes split by tier.
    #[inline]
    fn reload_tier(&self, layer: usize, expert: usize) -> ReloadTier {
        let idx = layer * self.experts_per_layer + expert;
        if self.warm_backed.get(idx).copied().unwrap_or(false) {
            ReloadTier::Warm
        } else {
            ReloadTier::Nvme
        }
    }

    /// Apply a protection factor to a base score, so that "larger factor ⇒
    /// better protected" holds on both sides of zero.
    ///
    /// [`behind_protection`] means "how much this slot is worth protecting", so
    /// a larger one must always raise the key. A plain multiply only does that
    /// above zero. [`PREFILL_ELEVATE_PENALTY`] drives a freshly prefill-loaded
    /// expert *below* zero on purpose, and multiplying a negative base by a
    /// protection factor makes it more negative: the slot about to run would be
    /// taken before the one just behind — exactly backwards, and invisible,
    /// because the result is a plausible number in the wrong order.
    ///
    /// Dividing below zero fixes the direction while keeping the sign a hard
    /// tier: whatever its protection, an elevated expert still sorts below every
    /// scored one in its pass, which is the separation
    /// [`PREFILL_ELEVATE_PENALTY`] exists to create. `protect` is always in
    /// `[FADE_FLOOR, 1]`, so this never divides by zero.
    #[inline]
    fn protected_by(base: f32, protect: f32) -> f32 {
        if base < 0.0 {
            base / protect
        } else {
            base * protect
        }
    }

    /// Slots that exist — the zone's capacity, which is also `slots.len()`.
    pub(crate) fn num_slots(&self) -> usize {
        self.zone.capacity()
    }

    /// Experts in the model — every MoE layer's full complement.
    pub(crate) fn total_experts(&self) -> usize {
        self.num_moe_layers * self.experts_per_layer
    }

    /// Slots not currently holding an expert.
    pub(crate) fn free_len(&self) -> usize {
        self.zone.free_count()
    }

    /// Take the rightmost free slot, without evicting anything.
    ///
    /// `None` means every slot is occupied — the signal to consult the eviction
    /// policy, never a reason to skip it. Position decides *where*; temperature
    /// decides *who*.
    pub(crate) fn take_free(&mut self) -> Option<usize> {
        self.zone.alloc()
    }

    /// Return a slot whose contents are gone.
    pub(crate) fn put_free(&mut self, slot_idx: usize) {
        self.slots[slot_idx] = None;
        self.zone.release(slot_idx);
    }

    /// Device address of slot `slot_idx`'s first byte.
    pub(crate) fn slot_base(&self, slot_idx: usize) -> u64 {
        self.zone.slot_base(slot_idx)
    }

    /// Take `target - capacity` more slots from the KV side. Returns how many.
    ///
    /// The per-slot bookkeeping grows with the zone. Nothing is loaded into the
    /// new slots here — they join the free list and the next miss takes them,
    /// after every closer hole.
    pub(crate) fn grow_zone(&mut self, target: usize) -> usize {
        let gained = self.zone.grow_to(target);
        if gained > 0 {
            let n = self.zone.capacity();
            self.slots.resize_with(n, || None);
            self.last_used.resize(n, 0);
            self.slot_to_key.resize(n, None);
        }
        gained
    }

    /// Give `capacity - target` slots back to the KV side.
    ///
    /// Returns the plan the caller must perform on the bytes: relocate the
    /// hottest doomed occupants into surviving free slots, evict the rest. The
    /// bookkeeping *inside the zone* is already applied; the per-slot tables
    /// here are truncated once the caller has moved what it is going to move,
    /// which is why this returns before touching them.
    pub(crate) fn retract_zone(&mut self, target: usize) -> candle_nn::kv_cache::RetractPlan {
        // The pinned set does not move with the boundary — those experts have no
        // record in the pack and no warm slot, so unpinning one under pressure
        // would strand it. What keeps the eviction scan supplied instead is the
        // zone's floor: `minimum_resident_slots` prices the pinned set plus a
        // full working layer, and `WeightZone::retract_to` will not go below it.
        // (The failure this guards against is measured: a zone that retracted to
        // 297 slots against a 384-expert pinned set failed 1,774 consecutive
        // slot allocations, because `layer >= pinned_layers` matched nothing.)
        debug_assert!(
            target.max(self.zone.min_capacity())
                >= minimum_resident_slots(self.experts_per_layer).min(self.total_experts()),
            "retraction target is below the floor that prices the fixed pinned set"
        );
        // Keep key: pack-only experts survive a concession ahead of
        // warm-backed ones, then the hotter within each tier — the eviction
        // passes, inverted. The same key decides which survivor below the
        // frontier a hotter doomed expert displaces; a pinned-layer expert is
        // never displaced, having nowhere to reload from.
        let keep: Vec<(bool, f32)> = (0..self.zone.capacity())
            .map(|i| {
                self.slot_to_key[i].map_or((false, 0.0), |(layer, expert)| {
                    (
                        self.reload_tier(layer, expert) == ReloadTier::Nvme,
                        self.score(layer, expert),
                    )
                })
            })
            .collect();
        let before = self.zone.capacity();
        let pinned_layers = self.pinned_layers;
        let slot_to_key = &self.slot_to_key;
        let plan = self.zone.retract_to(
            target,
            |i| keep[i],
            |i| slot_to_key[i].is_some_and(|(layer, _)| layer >= pinned_layers),
        );
        // Keyed on the CAPACITY changing, not on the plan being non-empty: a
        // concession of slots that happened to be free moves the boundary just
        // the same, while asking nothing to be relocated or evicted.
        if self.zone.capacity() != before {
            // At INFO, not DEBUG. A concession retires expert slots and hands
            // their ground to the KV side; it is the event that turns an
            // all-resident grid into a paged one. It went unnoticed for a whole
            // campaign because it was logged at DEBUG while the daemon runs at
            // INFO — 4.92 GiB of expert ground changed hands and the log said
            // nothing.
            let n = self.zone.capacity();
            tracing::info!(
                before,
                after = n,
                conceded = before.saturating_sub(n),
                frontier = format!("{:#x}", self.zone.slot_base(n.saturating_sub(1))),
                "expert cache: weight zone conceded ground to the KV side"
            );
        }
        plan
    }

    /// Drop the per-slot tables to the zone's current capacity.
    ///
    /// Separate from [`Self::retract_zone`] because the caller has to move bytes
    /// and rewrite `slot_to_key` for the relocations in between; truncating
    /// first would take the entries it still needs to read.
    pub(crate) fn truncate_tables(&mut self) {
        let n = self.zone.capacity();
        self.slots.truncate(n);
        self.last_used.truncate(n);
        self.slot_to_key.truncate(n);
        // Anything the truncation removed is gone from VRAM; the location map
        // must not still point at it.
        self.key_to_slot.retain(|_, &mut slot| slot < n);
    }

    /// Promote a slot's timestamp (the hot path — one array write).
    #[inline]
    pub(crate) fn promote(&mut self, slot_idx: usize) {
        self.last_used[slot_idx] = self.generation;
        self.generation += 1;
    }

    /// Evict a slot: drop its contents and remove it from the lookup tables.
    ///
    /// Returns the evicted `(moe_layer, expert_idx)` key, or `None` if the slot
    /// was already empty.
    ///
    /// **Eviction moves no bytes.** The cold tier holds a valid copy of every
    /// expert at all times, so there is nothing to write back and nowhere to
    /// write it — this used to hand the `ExpertSlot` to the caller for a D2H
    /// copy into pinned RAM, and that copy duplicated data the pack file
    /// already held. Dropping the slot here releases the three `QMatMul` views;
    /// the zone owns the bytes and keeps them.
    pub(crate) fn evict(&mut self, slot_idx: usize) -> Option<(usize, usize)> {
        let evicted = self.slot_to_key[slot_idx];
        if let Some(evict_key) = evicted {
            self.key_to_slot.remove(&evict_key);
        }
        self.slot_to_key[slot_idx] = None;
        self.slots[slot_idx] = None;
        evicted
    }

    // ── Score update methods ─────────────────────────────────────────

    /// Index into `expert_scores` for a given (layer, expert) pair.
    #[inline]
    fn score_idx(&self, layer: usize, expert: usize) -> usize {
        layer * self.experts_per_layer + expert
    }

    /// Get the current score for a (layer, expert) pair.
    #[inline]
    pub(crate) fn score(&self, layer: usize, expert: usize) -> f32 {
        let idx = self.score_idx(layer, expert);
        self.expert_scores[idx] + DECODE_REUSE_WEIGHT * self.decode_reuse[idx]
    }

    /// Record a decode hit: +1.0, and +1.0 of long-memory decode reuse.
    #[inline]
    pub(crate) fn record_hit(&mut self, layer: usize, expert: usize) {
        let idx = self.score_idx(layer, expert);
        self.expert_scores[idx] += 1.0;
        self.decode_reuse[idx] += 1.0;
    }

    /// Record a decode miss: +1.0, so the expert holds its slot to the next
    /// step, which near-certainly routes to it again — but no decode reuse
    /// until it is hit.
    #[inline]
    pub(crate) fn record_decode_miss(&mut self, layer: usize, expert: usize) {
        let idx = self.score_idx(layer, expert);
        self.expert_scores[idx] += 1.0;
    }

    /// Record a prefill-attributed hit on an already-resident expert:
    /// [`PREFILL_HIT_SCORE`] (+0.1) rather than the full decode weight.
    #[inline]
    pub(crate) fn record_prefill_hit(&mut self, layer: usize, expert: usize) {
        let idx = self.score_idx(layer, expert);
        self.expert_scores[idx] += PREFILL_HIT_SCORE;
    }

    /// Record a prefill-attributed elevation (a fresh cold load, not a hit):
    /// [`PREFILL_ELEVATE_PENALTY`] (−0.1), biasing it toward the next
    /// eviction scan rather than leaving its score at whatever an earlier,
    /// unrelated occupancy left behind.
    ///
    /// **Assigned, not accumulated.** `install` does not clear `expert_scores`,
    /// so the slot's previous tenant's score is still standing when a fresh
    /// expert lands on it. Adding the penalty to that leaves a formerly
    /// decode-hot expert at, say, 11.9 — comfortably protected, which is the
    /// opposite of what an elevation means and exactly the "earlier, unrelated
    /// occupancy" the constant's doc says this must not inherit. An elevation is
    /// one event with one meaning: this expert was just paid for and is the
    /// least likely thing in the zone to be wanted again — by recency.
    ///
    /// **Decode reuse is kept.** It belongs to the expert, not to the slot, and
    /// was earned by decode hits on this same expert, so an expert decode keeps
    /// returning to still outranks a prompt's one-shot fetch after a prompt
    /// happens to load it: its score is −0.1 plus [`DECODE_REUSE_WEIGHT`] × its
    /// reuse.
    #[inline]
    pub(crate) fn record_prefill_elevate(&mut self, layer: usize, expert: usize) {
        let idx = self.score_idx(layer, expert);
        self.expert_scores[idx] = PREFILL_ELEVATE_PENALTY;
    }

    /// Record a successful speculative prediction: +0.3.
    #[inline]
    pub(crate) fn record_prediction_hit(&mut self, layer: usize, expert: usize) {
        let idx = self.score_idx(layer, expert);
        self.expert_scores[idx] += 0.3;
    }

    /// End-of-pass exponential decay: multiply all scores by `factor` (e.g. 0.85),
    /// and decode reuse by [`DECODE_REUSE_DECAY`].
    pub(crate) fn decay_scores(&mut self, factor: f32) {
        for s in self.expert_scores.iter_mut() {
            *s *= factor;
        }
        for r in self.decode_reuse.iter_mut() {
            *r *= DECODE_REUSE_DECAY;
        }
    }

    /// Forward (wrapped) distance from `current_layer` to `layer`: how many
    /// layers until the wave reaches `layer` again. Distance 0 = the layer being
    /// computed right now; distance `n-1` = the layer that just executed.
    #[inline]
    fn forward_distance(&self, layer: usize, current_layer: usize) -> usize {
        if layer >= current_layer {
            layer - current_layer
        } else {
            self.num_moe_layers - current_layer + layer
        }
    }

    /// Up to `count` batch victims, ranked by [`VictimKey::order`] within each
    /// [`EVICTION_PASSES`] pass: a later pass only for what the earlier ones
    /// could not supply. Best victim first.
    ///
    /// `weight(slot_idx, moe_layer, dist)` admits a non-pinned occupied slot
    /// by returning the protection factor its frequency is scaled by (through
    /// [`Self::protected_by`], so an elevated expert's negative score is not
    /// inverted), or `None` to skip it.
    fn batch_victims(
        &self,
        current_layer: usize,
        count: usize,
        weight: impl Fn(usize, usize, usize) -> Option<f32>,
    ) -> Vec<usize> {
        let mut victims = Vec::with_capacity(count);
        for (tier, hit) in self.passes {
            let need = count - victims.len();
            if need == 0 {
                break;
            }
            let mut cands: Vec<VictimKey> = self
                .slot_to_key
                .iter()
                .enumerate()
                .filter_map(|(idx, key)| {
                    let (layer, expert) = (*key)?;
                    if layer < self.pinned_layers
                        || self.reload_tier(layer, expert) != tier
                        || self.hit_last_visit(layer, expert) != hit
                    {
                        return None;
                    }
                    let dist = self.forward_distance(layer, current_layer);
                    let factor = weight(idx, layer, dist)?;
                    Some(VictimKey {
                        slot: idx,
                        score: Self::protected_by(self.score(layer, expert), factor),
                        dist,
                        lru: self.last_used[idx],
                    })
                })
                .collect();
            // O(n) partition to the best `need`, then order just those.
            if need < cands.len() {
                cands.select_nth_unstable_by(need, VictimKey::order);
                cands.truncate(need);
            }
            cands.sort_by(VictimKey::order);
            victims.extend(cands.iter().map(|c| c.slot));
        }
        victims
    }

    /// Up to `count` victims, best first, **without evicting them** — the
    /// caller decides, per victim, whether it may be taken now (the reclaim
    /// rule: an expert whose row is still being computed cannot give up its
    /// slot) and evicts the ones it takes with [`Self::evict`].
    ///
    /// ## Victim key: `frequency × behind_protection`, per reload tier
    ///
    /// Lowest key first. The protection fades with how far behind the wave a
    /// layer is ([`behind_protection`]): L−1 is scaled by [`FADE_FLOOR`] and
    /// each layer further back a little more, wrapping round to the layers about
    /// to run, which keep nearly their whole score. So the wave protects what
    /// is ahead of it at the expense of what is behind, and an expert just
    /// behind survives one about to run only with about `1 / FADE_FLOOR` its
    /// score — enough to steer churn behind the wave, not enough to evict a hot
    /// expert the next pass needs for a cold one ahead.
    ///
    /// Demand and read-ahead claims spend the same ranked offers, so a read-ahead
    /// copy for a row ahead is bought from the layers the wave has left, not
    /// from the ones it is about to route.
    ///
    /// The policy runs once per eviction pass — a reload tier × whether the
    /// row routed the expert on its last visit — cheapest expected miss cost
    /// first ([`Self::set_eviction_costs`]), and a later pass only for the
    /// shortfall. So neither distance nor frequency outranks a pass: a
    /// warm-backed expert ahead of the wave goes before a pack-only one just
    /// behind it, and a stale hot expert before one its row just routed,
    /// whenever the measured costs order the passes that way. A hard
    /// distance tier across the reload tiers was measured here and tripled cold
    /// pack reads (2.8k→8.2k at config-8, bulk −9%) precisely because it let
    /// position override the cold shield.
    ///
    /// Ties break farther-first then LRU, so a never-used expert goes from the
    /// layer just behind first. Pinned layers are never candidates.
    /// `admit(slot, layer)` refuses a slot outright.
    ///
    /// One O(slots) scan + an O(n) `select_nth` partition per tier pass.
    pub(crate) fn rank_victims(
        &self,
        current_layer: usize,
        count: usize,
        admit: impl Fn(usize, usize) -> bool,
    ) -> Vec<usize> {
        if count == 0 {
            return Vec::new();
        }
        let n = self.num_moe_layers;
        self.batch_victims(current_layer, count, |idx, layer, dist| {
            admit(idx, layer).then(|| behind_protection(dist, n))
        })
    }

    /// Install an expert into a slot and update all lookup tables.
    pub(crate) fn install(
        &mut self,
        slot_idx: usize,
        moe_idx: usize,
        expert_idx: usize,
        slot: ExpertSlot,
    ) {
        self.slots[slot_idx] = Some(slot);
        self.key_to_slot.insert((moe_idx, expert_idx), slot_idx);
        self.slot_to_key[slot_idx] = Some((moe_idx, expert_idx));
        #[cfg(feature = "tensor-assert")]
        self.mirror_tenant(slot_idx);
        self.promote(slot_idx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cache of `n` slots over a zone with no device behind it.
    ///
    /// The zone is pure arithmetic — addresses and indices — so the whole
    /// eviction policy still exercises with no GPU and no model load, exactly as
    /// it did when the free list was a local `Vec`.
    /// Pins the standard head layers, as every cache now does.
    fn cache(n: usize) -> ExpertCacheInner {
        ExpertCacheInner::new(WeightZone::new(1 << 30, 4096, n, n, 0), 48, 128)
    }

    /// A cache sized as the loader would size it.
    fn sized_cache(n: usize, experts_per_layer: usize) -> ExpertCacheInner {
        ExpertCacheInner::new(
            WeightZone::new(1 << 30, 4096, n, n, 0),
            48,
            experts_per_layer,
        )
    }

    /// **A cache at or above the floor can always evict**, at any size.
    ///
    /// The failure this rules out is not a slow cache, it is a dead one: once
    /// every resident slot holds a pinned-layer expert, the `layer >= pinned`
    /// filter matches nothing, and no promotion can take a slot from then on —
    /// permanently, because escaping the state requires an eviction the state
    /// forbids. The
    /// daemon reached it with 297 slots against a 384-expert pinned set and
    /// failed 1,774 consecutive forwards.
    ///
    /// The pinned count is fixed, so the *floor* is what rules the state out:
    /// [`minimum_resident_slots`] prices the pinned set plus a full working
    /// layer. Swept over sizes from the floor upward, and over every model's
    /// expert count.
    #[test]
    fn a_cache_above_the_floor_always_has_a_victim() {
        for experts_per_layer in [8usize, 128, 256] {
            let floor = minimum_resident_slots(experts_per_layer);
            for capacity in [
                floor,
                floor + 1,
                4 * experts_per_layer,
                9 * experts_per_layer,
            ] {
                assert!(
                    PINNED_LAYERS * experts_per_layer + experts_per_layer <= capacity,
                    "the floor must leave the working layer room in {capacity} slots"
                );

                // The worst case that actually produced the deadlock: a batch
                // wide enough to fill every slot from layer 0 upward.
                let mut inner = sized_cache(capacity, experts_per_layer);
                for slot in 0..capacity {
                    let layer = slot / experts_per_layer;
                    occupy(&mut inner, slot, layer, slot % experts_per_layer, 0, 0.0);
                }
                assert!(
                    best_victim(&inner, capacity / experts_per_layer, &[]).is_some(),
                    "a full cache of {capacity} slots at {experts_per_layer} experts \
                     a layer had no evictable slot"
                );
            }
        }
    }

    /// **The pinned depth never varies.** A pack omits the pinned layers'
    /// records, so a cache that unpinned a layer under pressure would strand it
    /// with nowhere to reload from — and a pack written on one machine would be
    /// unreadable on a smaller one. Capacity must not enter into it.
    #[test]
    fn the_pinned_depth_is_fixed_across_every_capacity() {
        for experts_per_layer in [8usize, 128, 256] {
            for capacity in [
                minimum_resident_slots(experts_per_layer),
                4 * experts_per_layer,
                9 * experts_per_layer,
                48 * experts_per_layer,
            ] {
                let inner = sized_cache(capacity, experts_per_layer);
                assert_eq!(
                    inner.pinned_layers, PINNED_LAYERS,
                    "{capacity} slots at {experts_per_layer}/layer changed the pinned depth"
                );
            }
        }
        // A model with fewer layers than the constant pins all of them.
        let tiny = ExpertCacheInner::new(WeightZone::new(1 << 30, 4096, 64, 64, 0), 1, 8);
        assert_eq!(tiny.pinned_layers, 1);
    }

    /// **A zone at the floor still has a victim**, which is the entire reason
    /// the floor exists.
    ///
    /// Fill every slot of a minimum-sized cache with pinned-layer experts — the
    /// worst case, a batch wide enough to route to all of them — and the scan
    /// must still find something to evict. Below this size it cannot: the
    /// pinned-layer filter matches nothing and no promotion can take a slot from
    /// then on, permanently, because escaping the state requires an eviction the state
    /// forbids. The daemon sat at 297 slots against a 384-slot floor and failed
    /// 1,774 consecutive forwards that way.
    #[test]
    fn a_cache_at_its_floor_can_still_evict() {
        let experts_per_layer = 128;
        let floor = minimum_resident_slots(experts_per_layer);
        assert_eq!(floor, (PINNED_LAYERS + 1) * experts_per_layer + 1);

        let mut inner = cache(floor);
        // Every pinned-layer expert resident, and one slot beyond them.
        for slot in 0..floor {
            let layer = slot / experts_per_layer;
            occupy(&mut inner, slot, layer, slot % experts_per_layer, 0, 0.0);
        }
        assert!(
            best_victim(&inner, PINNED_LAYERS, &[]).is_some(),
            "a zone sized to the pinned working set always has one slot the \
             scan is allowed to take"
        );

        // A cache with room for only the pinned set reaches the dead state: the
        // same fill leaves every resident slot holding a pinned-layer expert and
        // nothing can be freed. This is what the floor keeps the boundary out
        // of, with a full working layer of margin on top.
        let pinned_only = PINNED_LAYERS * experts_per_layer;
        assert!(pinned_only < floor, "the floor must clear the pinned set");
        let mut starved = cache(pinned_only);
        for slot in 0..pinned_only {
            let layer = slot / experts_per_layer;
            occupy(&mut starved, slot, layer, slot % experts_per_layer, 0, 0.0);
        }
        assert!(
            best_victim(&starved, PINNED_LAYERS, &[]).is_none(),
            "a cache holding only pinned-layer experts has nothing it may free"
        );
    }

    /// **The floor's arithmetic did not move when the pinned depth did.**
    ///
    /// This number feeds the opening boundary placement of every MoE model, and
    /// raising it starves the KV side (Qwen3-30B OOMs at load). Two pinned
    /// layers plus a working layer prices the same `3 × n + 1` that three
    /// capacity-derived layers did, so every model's placement is untouched.
    #[test]
    fn the_floor_is_arithmetically_unchanged() {
        for experts_per_layer in [8usize, 128, 256] {
            assert_eq!(
                minimum_resident_slots(experts_per_layer),
                3 * experts_per_layer + 1
            );
        }
        // The two models this actually places.
        assert_eq!(minimum_resident_slots(128), 385); // Qwen3-30B-A3B
        assert_eq!(minimum_resident_slots(256), 769); // Qwen3.5/3.6-35B-A3B
    }

    /// Mark a slot occupied by `(layer, expert)` without a real `ExpertSlot`
    /// (eviction selection reads only the bookkeeping tables, never the VRAM
    /// buffers).
    fn occupy(
        inner: &mut ExpertCacheInner,
        slot: usize,
        layer: usize,
        expert: usize,
        last_used: u32,
        freq: f32,
    ) {
        // The zone hands out the rightmost free index, and every test here
        // occupies ascending from 0, so the two agree. Asserting rather than
        // searching keeps the fixture honest: a test that stopped filling in
        // order would fail here rather than quietly occupy a different slot than
        // the one its assertions name.
        let taken = inner.zone.alloc().expect("a free slot");
        assert_eq!(taken, slot, "fixture must occupy slots in ascending order");
        inner.slot_to_key[slot] = Some((layer, expert));
        inner.key_to_slot.insert((layer, expert), slot);
        inner.last_used[slot] = last_used;
        inner.expert_scores[layer * inner.experts_per_layer + expert] = freq;
    }

    /// The demand-driven choice with every candidate admissible but `protect`:
    /// rank, then evict and free each victim. Returns the evicted keys.
    fn demand_eviction(
        inner: &mut ExpertCacheInner,
        current: usize,
        count: usize,
        protect: &[usize],
    ) -> Vec<(usize, usize)> {
        let victims = inner.rank_victims(current, count, |s, _| !protect.contains(&s));
        victims
            .into_iter()
            .map(|s| {
                let key = inner.evict(s).expect("a ranked victim holds an expert");
                inner.zone.release(s);
                key
            })
            .collect()
    }

    /// The single best demand victim, `(slot, key)`, if there is one.
    fn best_victim(
        inner: &ExpertCacheInner,
        current: usize,
        protect: &[usize],
    ) -> Option<(usize, (usize, usize))> {
        let s = *inner
            .rank_victims(current, 1, |s, _| !protect.contains(&s))
            .first()?;
        Some((
            s,
            inner.slot_to_key[s].expect("a ranked victim holds an expert"),
        ))
    }

    #[test]
    fn forced_eviction_targets_lowest_frequency() {
        // Four behind-the-wave experts at the same layer; the least-frequently
        // used (lowest score) is evicted, keeping the hot experts resident.
        let mut inner = cache(4);
        occupy(&mut inner, 0, 10, 100, 1, 8.0);
        occupy(&mut inner, 1, 10, 101, 2, 3.0);
        occupy(&mut inner, 2, 10, 102, 3, 0.5); // coldest
        occupy(&mut inner, 3, 10, 103, 4, 5.0);
        assert!(inner.free_len() == 0);

        assert_eq!(best_victim(&inner, 20, &[]), Some((2, (10, 102))));
    }

    #[test]
    fn record_prefill_hit_bumps_by_the_small_credit_only() {
        let mut inner = cache(1);
        occupy(&mut inner, 0, 10, 100, 1, 0.0);
        inner.record_prefill_hit(10, 100);
        assert_eq!(inner.score(10, 100), 0.1);
        inner.record_prefill_hit(10, 100);
        assert_eq!(inner.score(10, 100), 0.2);
    }

    #[test]
    fn record_hit_still_bumps_by_the_full_decode_credit() {
        let mut inner = cache(1);
        occupy(&mut inner, 0, 10, 100, 1, 0.0);
        inner.record_hit(10, 100);
        let idx = 10 * inner.experts_per_layer + 100;
        assert_eq!(inner.expert_scores[idx], 1.0);
        assert_eq!(inner.decode_reuse[idx], 1.0);
        assert_eq!(inner.score(10, 100), 1.1);
    }

    #[test]
    fn a_decode_miss_earns_the_decode_credit_but_no_reuse() {
        let mut inner = cache(1);
        occupy(&mut inner, 0, 10, 100, 1, 0.0);
        inner.record_decode_miss(10, 100);
        let idx = 10 * inner.experts_per_layer + 100;
        assert_eq!(inner.expert_scores[idx], 1.0);
        assert_eq!(inner.decode_reuse[idx], 0.0);
        assert_eq!(inner.score(10, 100), 1.0);
    }

    #[test]
    fn decode_reuse_decays_on_its_own_slower_clock() {
        let mut inner = cache(1);
        occupy(&mut inner, 0, 10, 100, 1, 0.0);
        inner.record_hit(10, 100);
        inner.decay_scores(0.5);
        let idx = 10 * inner.experts_per_layer + 100;
        assert_eq!(inner.expert_scores[idx], 0.5);
        assert_eq!(inner.decode_reuse[idx], 0.99);
    }

    /// Two experts with the same recency score: the one decode has hit outranks
    /// the one it has only missed.
    #[test]
    fn a_one_off_decode_miss_is_evicted_before_a_reused_expert() {
        let mut inner = cache(2);
        occupy(&mut inner, 0, 10, 100, 1, 0.0);
        occupy(&mut inner, 1, 10, 101, 1, 0.0);
        inner.record_hit(10, 100);
        inner.record_decode_miss(10, 101);
        assert_eq!(best_victim(&inner, 20, &[]), Some((1, (10, 101))));
    }

    #[test]
    fn record_prefill_elevate_pushes_the_score_negative() {
        let mut inner = cache(1);
        occupy(&mut inner, 0, 10, 100, 1, 0.0);
        inner.record_prefill_elevate(10, 100);
        assert_eq!(inner.score(10, 100), -0.1);
    }

    /// An elevation resets the recency score but keeps the decode reuse the
    /// same expert earned: two decode hits leave −0.1 + 0.1 × 2.
    #[test]
    fn a_prefill_elevation_keeps_the_experts_decode_reuse() {
        let mut inner = cache(1);
        occupy(&mut inner, 0, 10, 100, 1, 0.0);
        inner.record_hit(10, 100);
        inner.record_hit(10, 100);
        inner.record_prefill_elevate(10, 100);
        let idx = inner.score_idx(10, 100);
        assert_eq!(inner.expert_scores[idx], -0.1);
        assert_eq!(inner.decode_reuse[idx], 2.0);
        assert_eq!(inner.score(10, 100), 0.1);
    }

    /// **A previous tenant's score is not inherited.** `install` leaves
    /// `expert_scores` alone, so a slot that held a decode-hot expert still
    /// carries its score when a prefill miss lands a different expert on it.
    /// Adding the penalty there would leave the newcomer protected at 11.9;
    /// assigning it states what an elevation actually means.
    #[test]
    fn a_prefill_elevation_does_not_inherit_a_hot_predecessors_score() {
        let mut inner = cache(1);
        occupy(&mut inner, 0, 10, 100, 1, 0.0);
        let idx = inner.score_idx(10, 100);
        inner.expert_scores[idx] = 12.0;
        inner.record_prefill_elevate(10, 100);
        assert_eq!(inner.score(10, 100), -0.1);
    }

    /// The protection, raw, for 48 layers: the floor plus 0.75 × 1/48
    /// (= 1/64) per layer behind, from L−1 through the wrap to the layer being
    /// computed.
    #[test]
    fn protection_fades_from_the_layer_just_behind() {
        let n = 48;
        assert_eq!(behind_protection(47, n), 0.265625, "L-1");
        assert_eq!(behind_protection(45, n), 0.296875, "L-3");
        assert_eq!(behind_protection(24, n), 0.625, "half a pass behind");
        assert_eq!(behind_protection(1, n), 0.984375, "L+1");
        assert_eq!(behind_protection(0, n), 1.0, "the layer being computed");
    }

    /// **A prefill-lifted expert is evicted before any decode-scored one of its
    /// reload tier, at every position.** This is what keeps the cache hot for
    /// decode: a prefill sweep touches most of the table roughly once, so what
    /// it drags in must not displace the working set decode actually reuses.
    /// Checked at the extremes of the protection rather than at one convenient
    /// layer, because the ordering has to hold across the whole range — the
    /// least-evictable prefill lift (the layer being computed) against the
    /// most-evictable decode hit (L−1).
    #[test]
    fn a_prefill_lift_sorts_below_every_decode_hit_at_both_extremes() {
        let n = 48usize;
        let worst_protect = behind_protection(n - 1, n); // L-1
        let best_protect = behind_protection(0, n); // the layer being computed
        assert_eq!((worst_protect, best_protect), (0.265625, 1.0));
        // The least-evictable a prefill lift can be, and the most-evictable a
        // decode hit can be.
        let softest_lift = PREFILL_ELEVATE_PENALTY / best_protect;
        let weakest_decode = 1.0f32 * worst_protect;
        assert!(
            softest_lift < weakest_decode,
            "a prefill lift ({softest_lift}) must still evict before the weakest \
             decode hit ({weakest_decode})"
        );
        // And below a prefill HIT, which is the nearest positive neighbour.
        let weakest_prefill_hit = PREFILL_HIT_SCORE * worst_protect;
        assert!(
            softest_lift < weakest_prefill_hit,
            "a prefill lift ({softest_lift}) must evict before a prefill hit \
             ({weakest_prefill_hit})"
        );
    }

    /// Two elevated experts, one warm and one pack-only, through the
    /// single-victim path: the warm pass runs first, so the warm-backed one is
    /// taken however the two scores compare — the elevation cannot pull the
    /// pack-only expert ahead of its pass.
    #[test]
    fn a_prefill_elevated_pack_only_expert_is_not_evicted_before_a_warm_one() {
        let mut inner = cache(2);
        occupy(&mut inner, 0, 10, 100, 1, 0.0);
        occupy(&mut inner, 1, 10, 101, 2, 0.0);
        inner.record_prefill_elevate(10, 100);
        inner.record_prefill_elevate(10, 101);
        // 100 keeps a warm copy; 101 lives only in the pack, so it is the one
        // worth keeping.
        inner.set_warm_backed(&[(10, 100)]);
        assert_eq!(
            best_victim(&inner, 20, &[]),
            Some((0, (10, 100))),
            "the warm-backed expert is the cheaper victim"
        );
    }

    /// The same in `demand_eviction`: the elevated pack-only expert is in the
    /// second pass, which is never reached while the warm pass can supply.
    #[test]
    fn a_demand_batch_does_not_evict_the_pack_only_elevated_expert_first() {
        let mut inner = cache(2);
        occupy(&mut inner, 0, 10, 100, 1, 0.0);
        occupy(&mut inner, 1, 10, 101, 2, 0.0);
        inner.record_prefill_elevate(10, 100);
        inner.record_prefill_elevate(10, 101);
        inner.set_warm_backed(&[(10, 100)]);
        let evicted = demand_eviction(&mut inner, 15, 1, &[]);
        assert_eq!(
            evicted,
            vec![(10, 100)],
            "the warm-backed expert is the cheaper victim"
        );
    }

    /// **The protection must not invert an elevated score onto a layer ahead.**
    /// Two elevated warm-backed experts: layer 10 is five layers behind the wave
    /// (protection 0.328) and layer 20 five ahead of it (0.922). Multiplying
    /// would give the one behind −0.033 and the one ahead −0.092, evicting the
    /// expert the wave is about to route to and keeping the one it has just
    /// left; dividing gives −0.305 and −0.108.
    #[test]
    fn the_protection_does_not_invert_an_elevated_score_onto_an_upcoming_layer() {
        let mut inner = cache(2);
        occupy(&mut inner, 0, 10, 100, 1, 0.0); // dist 43 → five behind
        occupy(&mut inner, 1, 20, 101, 2, 0.0); // dist 5 → about to be used
        inner.record_prefill_elevate(10, 100);
        inner.record_prefill_elevate(20, 101);
        inner.set_warm_backed(&[(10, 100), (20, 101)]);
        let evicted = demand_eviction(&mut inner, 15, 1, &[]);
        assert_eq!(
            evicted,
            vec![(10, 100)],
            "the behind-the-wave expert goes, not the one five layers ahead"
        );
    }

    /// **The cold shield reaches the never-used group, where every key is
    /// exactly zero.**
    ///
    /// No score can separate two never-used experts, so within one pass the
    /// scan falls through to LRU — which knows nothing about what a reload
    /// costs. The tier pass is what keeps the shield: the LRU order here is set
    /// against it on purpose (the pack-only expert is the least recently used),
    /// and the warm-backed one still goes first because its pass runs first.
    ///
    /// Never-used residents are not an edge case: a decode-attributed elevation
    /// records no score at all, and a speculative prefetch's score stays at zero
    /// until its prediction validates.
    #[test]
    fn among_never_used_experts_the_pack_only_one_is_not_the_victim() {
        let mut inner = cache(2);
        occupy(&mut inner, 0, 10, 100, 5, 0.0); // warm, most recently used
        occupy(&mut inner, 1, 10, 101, 1, 0.0); // pack-only, least recently used
        inner.set_warm_backed(&[(10, 100)]);
        assert_eq!(inner.score(10, 100), 0.0);
        assert_eq!(inner.score(10, 101), 0.0);

        let evicted = demand_eviction(&mut inner, 15, 1, &[]);
        assert_eq!(
            evicted,
            vec![(10, 100)],
            "the reload tier outranks LRU among equal scores"
        );
    }

    /// **The sign stays a hard tier within a pass in the batch paths too.** An
    /// elevated expert ahead of the wave must still be taken before one with a
    /// single decode hit behind it: −0.1 / 0.922 against 1.0 × 0.328. This is
    /// what `protected_by`'s divide preserves and what a shift-then-multiply
    /// scheme would lose. Both are pack-only, so the two share a pass and only
    /// the key decides.
    #[test]
    fn an_elevated_expert_outranks_nothing_scored_in_a_demand_batch() {
        let mut inner = cache(2);
        occupy(&mut inner, 0, 20, 100, 1, 0.0); // five ahead of the wave
        occupy(&mut inner, 1, 10, 101, 2, 1.0); // one decode hit, five behind
        inner.record_prefill_elevate(20, 100);
        let evicted = demand_eviction(&mut inner, 15, 1, &[]);
        assert_eq!(
            evicted,
            vec![(20, 100)],
            "an elevated expert is evicted before any scored one"
        );
    }

    /// **A prefill-elevated expert is evicted before any hit, decode or
    /// prefill, once the pass has moved past its layer.** This is the whole
    /// point of the negative score: a fresh prefill load at layer 10 must not
    /// out-survive genuinely reused experts once the sweep is at layer 20.
    #[test]
    fn prefill_elevated_expert_is_the_preferred_victim_over_any_hit() {
        let mut inner = cache(3);
        occupy(&mut inner, 0, 10, 100, 1, 0.0); // freshly prefill-elevated
        inner.record_prefill_elevate(10, 100);
        occupy(&mut inner, 1, 10, 101, 2, 0.1); // one prefill hit
        occupy(&mut inner, 2, 10, 102, 3, 1.0); // one decode hit

        assert_eq!(best_victim(&inner, 20, &[]), Some((0, (10, 100))));
    }

    #[test]
    fn demand_eviction_prefers_the_layers_just_behind() {
        // current=35, n=48. Layer 30 is five behind (key 0.5 × 0.328 = 0.164),
        // layer 20 fifteen behind (0.4 × 0.484 = 0.194): the warmer expert just
        // behind goes BEFORE the colder one further back, steering churn onto
        // the layers the wave just left.
        let mut inner = cache(4);
        occupy(&mut inner, 0, 1, 100, 1, 0.1); // pinned (layer < PINNED_LAYERS)
        occupy(&mut inner, 1, 10, 101, 2, 5.0);
        occupy(&mut inner, 2, 20, 102, 3, 0.4); // colder, further back
        occupy(&mut inner, 3, 30, 103, 4, 0.5); // just behind → victim

        let evicted = demand_eviction(&mut inner, 35, 1, &[]);
        assert_eq!(evicted, vec![(30, 103)], "just-behind expert goes first");
        assert!(inner.key_to_slot.contains_key(&(10, 101)));
        assert!(inner.key_to_slot.contains_key(&(20, 102)));
        assert!(
            inner.key_to_slot.contains_key(&(1, 100)),
            "pinned layer was evicted"
        );
        assert_eq!(inner.free_len(), 1, "exactly the demanded count freed");
    }

    #[test]
    fn demand_eviction_lets_frequency_outweigh_a_few_layers() {
        // Three candidates just behind current=35: a cold expert four back
        // (0.3 × 0.3125 = 0.094) goes before a warm one at L-1 (2.0 × 0.266 =
        // 0.53) and a hot one at L-2 (6.0 × 0.281 = 1.69).
        let mut inner = cache(3);
        occupy(&mut inner, 0, 33, 100, 1, 6.0); // L-2, hot
        occupy(&mut inner, 1, 31, 101, 2, 0.3); // L-4, coldest → victim
        occupy(&mut inner, 2, 34, 102, 3, 2.0); // L-1, warm
        let evicted = demand_eviction(&mut inner, 35, 1, &[]);
        assert_eq!(evicted, vec![(31, 101)]);
    }

    #[test]
    fn demand_eviction_cold_shield_outranks_distance() {
        // The reload tier dominates the distance: a warm-backed expert AHEAD of
        // the wave (cheap RAM reload) is evicted before an equally-used
        // cold-only one just behind it (whose reload is a pack read), because
        // the warm pass runs first. A hard distance tier across the reload
        // tiers inverted this trade and tripled cold pack reads.
        let mut inner = cache(2);
        occupy(&mut inner, 0, 32, 100, 1, 1.0); // just behind, cold-only
        occupy(&mut inner, 1, 40, 101, 2, 1.0); // ahead, warm-backed
        inner.set_warm_backed(&[(40, 101)]);
        let evicted = demand_eviction(&mut inner, 35, 1, &[]);
        assert_eq!(
            evicted,
            vec![(40, 101)],
            "warm reload chosen over pack read"
        );
        assert!(inner.key_to_slot.contains_key(&(32, 100)));
    }

    #[test]
    fn demand_eviction_protects_the_layer_hits() {
        // A protected slot (the caller lists the current layer's hits) is
        // spared even when it is the best victim on the card; the eviction
        // takes the next one instead.
        let mut inner = cache(3);
        occupy(&mut inner, 0, 30, 100, 1, 0.1); // best victim — but a HIT, protected
        occupy(&mut inner, 1, 10, 101, 2, 5.0);
        occupy(&mut inner, 2, 33, 102, 3, 0.4); // L-2, next best → victim
        let evicted = demand_eviction(&mut inner, 35, 1, &[0]);
        assert_eq!(evicted, vec![(33, 102)], "protected hit slot spared");
        assert!(inner.key_to_slot.contains_key(&(30, 100)));
    }

    #[test]
    fn demand_eviction_protects_in_flight_installs() {
        // An in-flight prefetch install (near-future layer, score 0 — the
        // coldest-looking slot on the card) survives when listed in `protect`;
        // the eviction takes the next candidate instead.
        let mut inner = cache(3);
        occupy(&mut inner, 0, 36, 100, 1, 0.0); // in-flight install for L+1, protected
        occupy(&mut inner, 1, 20, 101, 2, 0.5); // fifteen behind → victim
        occupy(&mut inner, 2, 37, 102, 3, 4.0);
        let evicted = demand_eviction(&mut inner, 35, 1, &[0]);
        assert_eq!(evicted, vec![(20, 101)], "in-flight install spared");
        assert!(inner.key_to_slot.contains_key(&(36, 100)));
    }

    #[test]
    fn demand_eviction_caps_at_the_candidates() {
        // Asking for more than the non-pinned population frees what exists and
        // no more; a promotion short of victims takes fewer slots.
        let mut inner = cache(2);
        occupy(&mut inner, 0, 1, 100, 1, 0.1); // pinned
        occupy(&mut inner, 1, 10, 101, 2, 0.2);
        let evicted = demand_eviction(&mut inner, 20, 5, &[]);
        assert_eq!(evicted, vec![(10, 101)]);
        assert!(inner.key_to_slot.contains_key(&(1, 100)), "pinned survives");
    }

    #[test]
    fn a_refused_slot_is_never_ranked() {
        // A slot the caller refuses — a row still being computed, which cannot
        // give its slot up yet — is skipped even when it is the lowest-scored
        // slot on the card.
        let mut inner = cache(2);
        occupy(&mut inner, 0, 36, 100, 1, 0.0); // refused
        occupy(&mut inner, 1, 40, 101, 2, 9.0); // hot, but the only legal victim
        assert_eq!(best_victim(&inner, 35, &[0]), Some((1, (40, 101))));
    }

    #[test]
    fn among_equals_ahead_the_furthest_future_goes() {
        // Everything resident is ahead of the wave, equal frequency: the
        // FURTHEST-future expert goes (next use latest — Belady), not the one
        // about to be routed.
        let mut inner = cache(2);
        occupy(&mut inner, 0, 36, 100, 1, 2.0); // L+1 — about to be routed, kept
        occupy(&mut inner, 1, 45, 101, 2, 2.0); // L+10 — furthest future → victim
        assert_eq!(best_victim(&inner, 35, &[]), Some((1, (45, 101))));
    }

    /// **At equal scores the order is the walk backwards from the wave**: L−1,
    /// then L−3, then L+1 — the layer about to run last of all.
    #[test]
    fn equal_scores_go_from_the_layer_just_behind_backwards() {
        let mut inner = cache(3);
        occupy(&mut inner, 0, 11, 100, 1, 1.0); // L+1
        occupy(&mut inner, 1, 7, 101, 2, 1.0); // L-3
        occupy(&mut inner, 2, 9, 102, 3, 1.0); // L-1
        assert_eq!(inner.rank_victims(10, 3, |_, _| true), vec![2, 1, 0]);
    }

    /// **The eviction front moves with the wave, through the wrap.** At the
    /// start of a pass (current=2 of 48) the layers just behind are the
    /// previous pass's last: layer 47 (three behind) goes before layer 10
    /// (forty behind — about to run). Well into the pass (current=30) the order
    /// turns over: layer 10 is twenty behind and layer 47 seventeen ahead.
    #[test]
    fn the_eviction_front_wraps_with_the_wave() {
        let mut inner = cache(2);
        occupy(&mut inner, 0, 47, 100, 1, 1.0);
        occupy(&mut inner, 1, 10, 101, 2, 1.0);
        assert_eq!(inner.rank_victims(2, 2, |_, _| true), vec![0, 1]);
        assert_eq!(inner.rank_victims(30, 2, |_, _| true), vec![1, 0]);
    }

    /// **Distance steers, frequency decides.** A hot expert just behind the
    /// wave (9.0 × 0.266 = 2.39) outlasts a near-cold one about to run (0.2 ×
    /// 0.984 = 0.197): the next pass needs the hot one. Between two experts of
    /// like temperature the one just behind goes — 1.0 × 0.266 against 0.3 ×
    /// 0.984 = 0.295 — and a never-used one goes first wherever it sits.
    #[test]
    fn frequency_ranks_within_the_fade() {
        let mut inner = cache(2);
        occupy(&mut inner, 0, 29, 100, 1, 9.0); // L-1, hot
        occupy(&mut inner, 1, 31, 101, 2, 0.2); // L+1, near cold
        assert_eq!(inner.rank_victims(30, 1, |_, _| true), vec![1]);

        let mut inner = cache(2);
        occupy(&mut inner, 0, 29, 100, 1, 1.0); // L-1
        occupy(&mut inner, 1, 31, 101, 2, 0.3); // L+1
        assert_eq!(inner.rank_victims(30, 1, |_, _| true), vec![0]);

        let mut inner = cache(2);
        occupy(&mut inner, 0, 10, 100, 1, 4.0); // L-20, hot
        occupy(&mut inner, 1, 31, 101, 2, 0.0); // L+1, never used
        assert_eq!(inner.rank_victims(30, 1, |_, _| true), vec![1]);
    }

    #[test]
    fn pinned_layers_never_evicted() {
        let mut inner = cache(PINNED_LAYERS);
        for layer in 0..PINNED_LAYERS {
            occupy(&mut inner, layer, layer, 100 + layer, layer as u32 + 1, 0.0);
        }
        // Every resident expert is pinned → no legal victim.
        assert!(best_victim(&inner, 5, &[]).is_none());
    }

    /// An expert with no warm copy costs an NVMe read to bring back, so it is
    /// kept in preference to an equally-cold one that reloads over PCIe.
    #[test]
    fn the_expert_with_no_warm_copy_is_kept() {
        let mut inner = cache(2);
        occupy(&mut inner, 0, 10, 50, 5, 1.0); // warm-backed
        occupy(&mut inner, 1, 10, 51, 5, 1.0); // cold-only, same temperature
        inner.set_warm_backed(&[(10, 50)]);

        assert_eq!(
            best_victim(&inner, 20, &[]).map(|v| v.1),
            Some((10, 50)),
            "evicted the expert that would have to come back from disk"
        );
    }

    /// The reload tier is a pass, not a tilt: even a hot warm-backed expert
    /// goes before a pack-only one nobody is using.
    #[test]
    fn a_hot_warm_backed_expert_goes_before_a_cold_pack_only_one() {
        let mut inner = cache(5);
        occupy(&mut inner, 0, 10, 50, 1, 40.0); // warm-backed and very hot
        occupy(&mut inner, 1, 11, 51, 2, 0.1); // pack-only and cold
        occupy(&mut inner, 2, 12, 52, 3, 0.2);
        occupy(&mut inner, 3, 13, 53, 4, 0.3);
        occupy(&mut inner, 4, 14, 54, 5, 0.4);
        inner.set_warm_backed(&[(10, 50)]);

        assert_eq!(best_victim(&inner, 20, &[]), Some((0, (10, 50))));
    }

    /// The warm-backed experts are exhausted before a pack-only one is
    /// considered, so a warm-backed expert AHEAD of the wave goes before a
    /// pack-only one behind it.
    #[test]
    fn the_warm_pass_is_exhausted_before_the_pack() {
        let mut inner = cache(3);
        occupy(&mut inner, 0, 10, 50, 1, 0.0); // behind, pack-only, never used
        occupy(&mut inner, 1, 30, 51, 2, 7.0); // ahead, warm-backed, hot
        occupy(&mut inner, 2, 40, 52, 3, 2.0); // ahead, warm-backed, cooler
        inner.set_warm_backed(&[(30, 51), (40, 52)]);
        assert_eq!(
            best_victim(&inner, 20, &[]),
            Some((2, (40, 52))),
            "coolest warm-backed expert"
        );
    }

    /// The batch path takes what the warm pass can supply, then runs the same
    /// policy over the pack-only experts for the shortfall — each pass in its
    /// own frequency × protection order.
    #[test]
    fn demand_eviction_runs_the_pack_pass_only_for_the_shortfall() {
        let mut inner = cache(5);
        occupy(&mut inner, 0, 10, 50, 1, 6.0); // warm, 25 behind: 3.84
        occupy(&mut inner, 1, 11, 51, 2, 3.0); // warm, 24 behind: 1.875
        occupy(&mut inner, 2, 12, 52, 3, 0.5); // pack-only, 23 behind: 0.305
        occupy(&mut inner, 3, 13, 53, 4, 2.0); // pack-only, 22 behind: 1.19
        occupy(&mut inner, 4, 32, 54, 5, 3.0); // pack-only, 3 behind: 0.891
        inner.set_warm_backed(&[(10, 50), (11, 51)]);

        let evicted = demand_eviction(&mut inner, 35, 4, &[]);
        assert_eq!(
            evicted,
            vec![(11, 51), (10, 50), (12, 52), (32, 54)],
            "both warm (by key), then the two best pack-only (0.305, then 0.891)"
        );
        assert!(inner.key_to_slot.contains_key(&(13, 53)));
        assert_eq!(inner.free_len(), 4);
    }

    /// With no warm tier every expert is pack-only, and the passes collapse to
    /// the plain policy: the same victims in the same order.
    #[test]
    fn with_no_warm_tier_the_policy_is_unchanged() {
        let mut inner = cache(3);
        occupy(&mut inner, 0, 10, 50, 1, 6.0);
        occupy(&mut inner, 1, 20, 51, 2, 0.5);
        occupy(&mut inner, 2, 33, 52, 3, 0.8); // L-2: key 0.225, under 20's 0.242
        let evicted = demand_eviction(&mut inner, 35, 2, &[]);
        assert_eq!(evicted, vec![(33, 52), (20, 51)]);
    }

    /// A concession inverts the passes: a warm-backed expert is dropped before
    /// a pack-only one, even a colder one.
    #[test]
    fn a_concession_relocates_the_pack_only_expert() {
        let mut inner = sized_cache(6, 1);
        for slot in 0..6 {
            occupy(&mut inner, slot, 10 + slot, 0, slot as u32, 1.0);
        }
        // One free destination below the new frontier.
        inner.evict(0);
        inner.put_free(0);
        inner.expert_scores[15] = 0.5; // slot 5: pack-only, colder
        inner.expert_scores[14] = 9.0; // slot 4: warm-backed and hot
        inner.set_warm_backed(&[(14, 0)]);

        let plan = inner.retract_zone(4);
        assert_eq!(plan.relocate, vec![(5, 0)]);
        assert_eq!(plan.evict, vec![4]);
    }

    /// A concession on a full zone trades a hot doomed expert for the coldest
    /// survivor below the frontier — never for a pinned-layer expert, which has
    /// no tier to reload from, however cold.
    #[test]
    fn a_full_concession_displaces_the_coldest_unpinned_survivor() {
        let mut inner = sized_cache(6, 1);
        let p = inner.pinned_layers;
        assert!(p >= 1, "the fixture needs a pinned layer");
        occupy(&mut inner, 0, 0, 0, 0, 0.0); // pinned, coldest of all
        occupy(&mut inner, 1, p + 1, 0, 1, 0.1); // the coldest unpinned survivor
        occupy(&mut inner, 2, p + 2, 0, 2, 5.0);
        occupy(&mut inner, 3, p + 3, 0, 3, 5.0);
        occupy(&mut inner, 4, p + 4, 0, 4, 9.0); // doomed, hot
        occupy(&mut inner, 5, p + 5, 0, 5, 0.05); // doomed, cold

        let plan = inner.retract_zone(4);
        assert_eq!(plan.displace, vec![1]);
        assert_eq!(plan.relocate, vec![(4, 1)]);
        assert_eq!(plan.evict, vec![5]);
    }

    #[test]
    fn hot_expert_survives_a_cold_one() {
        // Same layer (same protection): the frequently-used expert is kept
        // and the cold one evicted — the cache is frequency-dominated.
        let mut inner = cache(2);
        occupy(&mut inner, 0, 10, 50, 9, 9.0);
        occupy(&mut inner, 1, 10, 51, 1, 0.0);

        assert_eq!(
            best_victim(&inner, 20, &[]).map(|v| v.1),
            Some((10, 51)),
            "cold expert was not evicted in preference to the hot one"
        );
    }

    /// **An expert its row just routed goes only after every one it did not,**
    /// whatever their scores: here a just-hit expert scored 1.0 outlasts a
    /// stale one at 3.0, until the row's next visit routes it no more.
    #[test]
    fn a_just_hit_expert_outlasts_every_stale_one_of_its_tier() {
        let mut inner = cache(3);
        occupy(&mut inner, 0, 10, 50, 1, 1.0); // just hit
        occupy(&mut inner, 1, 10, 51, 2, 3.0); // stale, hot
        occupy(&mut inner, 2, 20, 52, 3, 5.0); // stale, hotter
        inner.mark_visit(10, &[50]);
        assert_eq!(inner.rank_victims(30, 3, |_, _| true), vec![1, 2, 0]);
        inner.mark_visit(10, &[51]);
        assert_eq!(inner.rank_victims(30, 1, |_, _| true), vec![0]);
    }

    /// The reload tier still comes first: a warm-backed expert just hit goes
    /// before a pack-only stale one.
    #[test]
    fn recency_sits_inside_the_reload_tier() {
        let mut inner = cache(2);
        occupy(&mut inner, 0, 10, 50, 1, 1.0); // warm, just hit
        occupy(&mut inner, 1, 10, 51, 2, 0.0); // pack-only, stale
        inner.set_warm_backed(&[(10, 50)]);
        inner.mark_visit(10, &[50]);
        assert_eq!(inner.rank_victims(30, 2, |_, _| true), vec![0, 1]);
    }

    /// **Measured costs order the passes.** Classes, `hit * 2 + warm`: a
    /// pack-only stale expert (6% back × 2.2 ms) costs more to evict than a
    /// warm just-hit one (37% × 0.2 ms), so the warm just-hit one goes first;
    /// where a pack-only class is the cheaper, it goes first; ties keep the
    /// warm-backed ahead.
    #[test]
    fn measured_costs_order_the_eviction_passes() {
        let mut inner = cache(4);
        occupy(&mut inner, 0, 10, 50, 1, 0.0); // warm, stale
        occupy(&mut inner, 1, 10, 51, 2, 0.0); // warm, just hit
        occupy(&mut inner, 2, 10, 52, 3, 0.0); // pack, stale
        occupy(&mut inner, 3, 10, 53, 4, 0.0); // pack, just hit
        inner.set_warm_backed(&[(10, 50), (10, 51)]);
        inner.mark_visit(10, &[51, 53]);
        // [cold pack, cold warm, hit pack, hit warm]
        inner.set_eviction_costs([0.06 * 2.2, 0.16 * 0.2, 0.05 * 2.2, 0.37 * 0.2]);
        assert_eq!(inner.rank_victims(30, 4, |_, _| true), vec![0, 1, 3, 2]);
        // A pack-only class cheap enough goes first.
        inner.set_eviction_costs([0.01, 0.5, 0.9, 0.9]);
        assert_eq!(inner.rank_victims(30, 4, |_, _| true), vec![2, 0, 1, 3]);
        // Equal costs keep the default: warm stale, warm hit, pack stale, pack hit.
        inner.set_eviction_costs([1.0; 4]);
        assert_eq!(inner.rank_victims(30, 4, |_, _| true), vec![0, 1, 2, 3]);
    }

    /// A visit marks exactly the experts it routed in its row, clearing the
    /// row's previous visit and leaving other rows alone.
    #[test]
    fn a_visit_marks_its_routed_experts_and_clears_the_rest() {
        let mut inner = cache(1);
        inner.mark_visit(10, &[3, 7]);
        inner.mark_visit(11, &[3]);
        inner.mark_visit(10, &[7, 9]);
        let row10: Vec<usize> = (0..12).filter(|&e| inner.hit_last_visit(10, e)).collect();
        let row11: Vec<usize> = (0..12).filter(|&e| inner.hit_last_visit(11, e)).collect();
        assert_eq!((row10, row11), (vec![7, 9], vec![3]));
    }

    /// Ranking evicts nothing: the caller decides per victim.
    #[test]
    fn ranking_leaves_the_tables_alone() {
        let mut inner = cache(2);
        occupy(&mut inner, 0, 10, 50, 5, 5.0);
        occupy(&mut inner, 1, 30, 51, 0, 0.0);
        assert_eq!(inner.rank_victims(20, 2, |_, _| true), vec![1, 0]);
        assert!(inner.key_to_slot.contains_key(&(10, 50)));
        assert!(inner.key_to_slot.contains_key(&(30, 51)));
        assert_eq!(inner.free_len(), 0);
    }
}
