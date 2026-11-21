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
pub(super) enum ReloadTier {
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
pub(super) const EVICTION_PASSES: [(ReloadTier, bool); 4] = [
    (ReloadTier::Warm, false),
    (ReloadTier::Warm, true),
    (ReloadTier::Nvme, false),
    (ReloadTier::Nvme, true),
];

/// A pass's eviction class, as the ledger indexes them (`regret`).
pub(super) fn pass_class((tier, hit): (ReloadTier, bool)) -> usize {
    class_index(hit, tier == ReloadTier::Warm)
}

/// One eviction candidate as the batch scans rank it within a pass.
#[derive(Clone, Copy)]
pub(super) struct VictimKey {
    pub(super) slot: usize,
    /// Frequency, with whatever positional factor the scan applies.
    pub(super) score: f32,
    /// Wrapped forward distance from the current layer.
    pub(super) dist: usize,
    pub(super) lru: u32,
}

impl VictimKey {
    /// Best victim first: lowest score, then farthest, then least recently
    /// used.
    pub(super) fn order(a: &Self, b: &Self) -> Ordering {
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
