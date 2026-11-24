//! The constants that fix which early layers are pinned, how the eviction
//! scores decay and how far the expert zone may be retracted.
//!
//! The first MoE layers are pinned because they run first in every pass with no
//! compute to hide a reload behind, and their depth is a constant rather than a
//! function of capacity so a pack written on one machine reads on another. The
//! zone floor prices the pinned set plus a full working layer plus one slot, the
//! least the eviction scan needs to have a candidate, since a zone below it can
//! reach a state where nothing is evictable and the state never clears. A
//! prefill-elevated expert carries a negative score, which the protection
//! factors must divide rather than multiply, or the better-protected slot sorts
//! first.

use super::*;

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

/// How far a retraction may take the zone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ZoneFloor {
    /// To [`minimum_resident_slots`] and no further: the engine serves tokens,
    /// and a forward needs the pinned set plus a working layer.
    Serving,
    /// To the pinned set alone. Only while nothing forwards — a guest drain —
    /// and only with the zone grown back to its floor before anything does.
    Paused,
}

/// Where an evicted expert's next miss is served from.
///
/// A pack-only expert costs a drive read and a copy to bring back (a 2.9 MB
/// page-cache-bypassing read near a millisecond), a warm-backed one the copy
/// alone (~116 µs H2D from pinned host memory). Eviction weighs that as a
/// **tilt, not a pass**: a pack-only resident's key is scaled by
/// [`PACK_ONLY_PROTECTION`] and it competes with every warm-backed one on the
/// same least-used order ([`tier_protection`]).
///
/// A pass — every warm-backed resident before any pack-only one — let the tier
/// decide alone wherever the two disagreed, and the startup fill made that
/// permanent. The experts VRAM holds at startup are exactly the ones the warm
/// draw skips first, so they were all pack-only; behind a hard pass they never
/// turned over, however little decode routed them. Measured on WhiteWorld
/// (Flash-Next, RTX 3090, build 5482): layers 2–10 held all 512 experts each on
/// the card — ~4,600 slots for a decode that routes ten a layer — while layers
/// 11–47 churned through the ~1,100 slots left and decoded 6.6 of 10 from VRAM.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum ReloadTier {
    /// A warm (host) copy exists — the cheap reload.
    Warm,
    /// Only the NVMe pack holds it.
    Nvme,
}

/// How much more a pack-only resident's key is worth than a warm-backed one's
/// at the same score: twice.
///
/// Small on purpose. It sits inside the same order as frequency and distance
/// rather than above them: a pack-only expert decode has stopped routing still
/// goes before a warm-backed one it keeps routing, and among never-used
/// residents (score zero, where no factor separates anything) the tier only
/// breaks a tie of distance ([`VictimKey::order`]). Twice is well inside the
/// ~4× span of [`behind_protection`], so position still steers churn behind
/// the wave across both tiers.
pub(super) const PACK_ONLY_PROTECTION: f32 = 2.0;

/// The reload-cost factor a resident's key is scaled by: 1 for a warm-backed
/// expert, [`PACK_ONLY_PROTECTION`] for a pack-only one.
pub(super) fn tier_protection(tier: ReloadTier) -> f32 {
    match tier {
        ReloadTier::Warm => 1.0,
        ReloadTier::Nvme => PACK_ONLY_PROTECTION,
    }
}

/// The order eviction passes visit residents in: first the experts their row
/// did **not** route on its last visit, then the ones it did (`hit_last`) —
/// each pass over both reload tiers.
///
/// **Recency is a pass, not a weight.** Decode routing carries over from one
/// step to the next: of the experts a row routed on its last visit, 37–63% are
/// routed again on its next (Qwen3.8-Flash-Next, RTX 4090 Laptop, decode ×1 to
/// ×8), against 10–21% of the rest. The decayed frequency score cannot see
/// that — a demand miss just installed scores about 1, a resident hit a few
/// passes ago still 2–3 — so ranked by score alone those just-hit experts were
/// 54% of the victims at ×1 and 82% at ×8, and accounted for most of the
/// evictions their rows then missed. As a pass they go only once every other
/// resident has.
pub(super) const RECENCY_PASSES: [bool; 2] = [false, true];

/// One eviction candidate as the batch scans rank it within a pass.
#[derive(Clone, Copy)]
pub(super) struct VictimKey {
    pub(super) slot: usize,
    /// Frequency, with the positional and reload-cost factors the scan applies.
    pub(super) score: f32,
    /// Wrapped forward distance from the current layer.
    pub(super) dist: usize,
    /// A warm copy backs it: among exact ties, the cheaper reload goes first.
    pub(super) warm: bool,
    pub(super) lru: u32,
}

impl VictimKey {
    /// Best victim first: lowest score, then farthest, then warm-backed, then
    /// least recently used.
    pub(super) fn order(a: &Self, b: &Self) -> Ordering {
        a.score
            .partial_cmp(&b.score)
            .unwrap_or(Ordering::Equal)
            .then(b.dist.cmp(&a.dist))
            .then(b.warm.cmp(&a.warm))
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
pub(super) fn behind_protection(dist: usize, n: usize) -> f32 {
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
pub(super) const FADE_FLOOR: f32 = 0.25;
