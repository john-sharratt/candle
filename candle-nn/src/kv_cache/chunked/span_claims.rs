//! The KV side's extents, published to [`candle::span_audit`].
//!
//! Two providers, because they have very different costs and the fast one is
//! the one that catches invariant 7:
//!
//! * **`span/zones`** — the four tenants of the reservation, read from
//!   [`region_pool::span_layout`] in one lock. Constant work per wave, and it is
//!   what detects a tier placed above `weight_floor`, a floor lowered through a
//!   standing tier, or a region range that has grown into the weight side.
//! * **`span/kv-arenas`** — every materialised arena, as a leaf inside the KV
//!   zone. Proportional to the arena count, so it is the one to watch for cost;
//!   it is what detects an arena that has ended up outside the region range, or
//!   two arenas on one address.
//!
//! Both report **live** state, re-read on every call. The partition is dynamic
//! by design — the floor moves, the tier comes and goes, arenas are created and
//! reclaimed — so a provider that cached anything would report a layout that no
//! longer exists and manufacture overlaps that were never there.

use candle::span_audit::{Claim, ClaimKind, Tenant};

use super::region_pool::{self, SpanLayout};

/// Push the four zone extents for `ordinal`.
///
/// The zones are `Container`s: they are *expected* to hold their own tenants'
/// buffers, and only a foreign leaf inside one, or two zones overlapping each
/// other, is a fault.
fn zones(ordinal: usize, out: &mut Vec<Claim>) {
    let Some(l) = region_pool::span_layout(ordinal) else {
        return;
    };
    push_zones(&l, out);
}

/// The zone extents of `l`, split out so the mapping from layout to claims can
/// be tested without a device.
pub(crate) fn push_zones(l: &SpanLayout, out: &mut Vec<Claim>) {
    // The persistence staging block, at the foot of the span. `persist_carved`
    // is what has actually been handed out, which is the extent that matters —
    // the gap between it and `region_base` is unclaimed ground, and reporting
    // that as owned would hide a region range that had grown down into it.
    if l.persist_carved > 0 {
        out.push(Claim::new(
            "span/persist",
            Tenant::Persist,
            ClaimKind::Container,
            l.span_base,
            l.persist_carved,
        ));
    }
    // **The KV side's OCCUPIED extent, not the regions it owns.**
    //
    // `region_end()` is capacity: every region below the weight floor the pool
    // has claimed, live or not. The transient tier is placed at `live_end`, the
    // arena frontier, so it stands on owned-but-free ground *inside* that range
    // by design — claiming capacity here reported the tier as overlapping the KV
    // zone on every single wave, 50–84 MiB of it, which is the whole tier.
    //
    // A first live run found that immediately, which is the argument for
    // reporting occupancy: it is the figure that answers "who holds this byte",
    // and capacity is not.
    if l.live_end > l.region_base {
        out.push(Claim::new(
            "span/kv-regions",
            Tenant::KvRegion,
            ClaimKind::Container,
            l.region_base,
            (l.live_end - l.region_base) as usize,
        ));
    }
    // The wave transient tier, only while one stands. Registering a released
    // tier's last extent would collide with whatever the KV side has since put
    // there — the exact false positive a stale declaration produces.
    if let Some(base) = l.transient_base {
        if l.transient_bytes > 0 {
            out.push(Claim::new(
                "span/tier",
                Tenant::TransientTier,
                ClaimKind::Container,
                base,
                l.transient_bytes,
            ));
        }
    }
    // The weight side: everything from the floor to the top of the span.
    if l.span_end > l.weight_floor {
        out.push(Claim::new(
            "span/weights",
            Tenant::ExpertWeight,
            ClaimKind::Container,
            l.weight_floor,
            (l.span_end - l.weight_floor) as usize,
        ));
    }
}

/// Register every KV-side provider for `ordinal`.
///
/// Idempotent — `span_audit::register` replaces by name, so a second call after
/// a reservation is rebuilt refreshes each closure rather than double-reporting.
///
/// Registered separately rather than as one closure so the audit's own report
/// can attribute cost, and so a provider that turns out to be expensive can be
/// dropped by name without losing the cheap ones.
pub fn register_zones(ordinal: usize) {
    candle::span_audit::register("span/zones", move |out| zones(ordinal, out));
    candle::span_audit::register("span/activations", move |out| {
        super::bump_arena::push_activation_claims(ordinal, out)
    });
    candle::span_audit::register("span/kv-arenas", |out| {
        super::backing::push_arena_claims(out)
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::span_audit::{find_overlaps, Relation};

    /// A layout with the four tenants in their proper order and no overlap.
    fn healthy() -> SpanLayout {
        SpanLayout {
            span_base: 0x1000,
            span_end: 0x9000,
            region_base: 0x2000,
            persist_carved: 0x1000,
            total: 0,
            live_end: 0x2000,
            weight_floor: 0x8000,
            transient_base: None,
            transient_bytes: 0,
        }
    }

    #[test]
    fn a_healthy_layout_has_no_overlaps() {
        let mut c = Vec::new();
        push_zones(&healthy(), &mut c);
        assert!(find_overlaps(&c).is_empty(), "{c:?}");
    }

    #[test]
    fn a_tier_placed_above_the_weight_floor_is_caught() {
        // `region_pool::tier_fits` is supposed to refuse this. Measured in
        // production once: a tier topping out at 0x51fbc00000 against a floor of
        // 0x51f9c00000, 32 MiB of tier inside expert ground, with the `wave-ffn`
        // span writing activations over resident expert slots.
        let mut l = healthy();
        l.transient_base = Some(0x7800);
        l.transient_bytes = 0x1000; // ends at 0x8800, past the floor
        let mut c = Vec::new();
        push_zones(&l, &mut c);
        let o = find_overlaps(&c);
        assert!(
            o.iter().any(|x| x.relation == Relation::ZoneOnZone),
            "tier crossing the weight floor must be flagged, got {o:?}"
        );
    }

    #[test]
    fn a_floor_lowered_through_a_standing_tier_is_caught() {
        // The mirror of the case above, and the one the floor guard missed: the
        // tier was legal when placed, then the weight side grew *down* onto the
        // ground it stands on.
        let mut l = healthy();
        l.transient_base = Some(0x7000);
        l.transient_bytes = 0x1000;
        l.weight_floor = 0x7400; // moved down, under the live tier
        let mut c = Vec::new();
        push_zones(&l, &mut c);
        assert!(
            find_overlaps(&c)
                .iter()
                .any(|x| x.relation == Relation::ZoneOnZone),
            "a floor cutting a standing tier must be flagged"
        );
    }

    #[test]
    fn kv_occupancy_grown_into_the_weight_side_is_caught() {
        let mut l = healthy();
        l.live_end = 0x8800; // live arenas run past the weight floor
        let mut c = Vec::new();
        push_zones(&l, &mut c);
        assert!(
            find_overlaps(&c)
                .iter()
                .any(|x| x.relation == Relation::ZoneOnZone),
            "live KV arenas overlapping the weight zone must be flagged"
        );
    }

    /// **The tier standing on owned-but-free KV ground is legal.**
    ///
    /// `place_transient` puts the tier at `live_end`, which is inside the region
    /// range whenever the pool owns more regions than it has filled — the normal
    /// state. Claiming capacity rather than occupancy made this fire on every
    /// wave of a live run, which is how the distinction was found.
    #[test]
    fn a_tier_above_the_live_arenas_is_not_an_overlap() {
        let mut l = healthy();
        l.total = 6; // the pool owns far more ground than it has filled
        l.live_end = 0x3000; // arenas reach here
        l.transient_base = Some(0x3000); // tier sits directly on the frontier
        l.transient_bytes = 0x1000;
        let mut c = Vec::new();
        push_zones(&l, &mut c);
        assert!(
            find_overlaps(&c).is_empty(),
            "a tier on free ground inside the region range must not report: {c:?}"
        );
    }

    #[test]
    fn a_released_tier_is_not_reported() {
        // A stale extent is the classic false positive: the KV side reuses that
        // ground the moment the tier lets go of it.
        let mut l = healthy();
        l.transient_base = Some(0x7000);
        l.transient_bytes = 0;
        let mut c = Vec::new();
        push_zones(&l, &mut c);
        assert!(
            !c.iter().any(|x| x.tenant == Tenant::TransientTier),
            "a zero-length tier must not be claimed"
        );
    }
}
