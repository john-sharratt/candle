//! The weight side's extents, published to [`candle::span_audit`].
//!
//! The expert grid is a stack of fixed-size slots growing **downwards** from the
//! top of the reservation:
//!
//! ```text
//! slot_base(i) = span_end − (i + 1) · slot_bytes
//! ```
//!
//! so slot 0 is the highest and rising indices walk toward `weight_floor`. That
//! direction is the whole reason this tenant needs auditing: the zone's capacity
//! is what keeps the lowest slot above the floor, and a capacity that outruns
//! the floor puts resident expert weights on ground the KV side owns — with no
//! fault, because every address in the span is mapped.
//!
//! Everything here takes its boundaries as arguments rather than reading them
//! from a live cache, so the rules can be tested without a device, a grid, or a
//! reservation. The one function that does read live state is
//! [`register`], which is the only part that cannot be exercised offline.

use candle::span_audit::{Claim, ClaimKind, Tenant};

/// Push one claim per resident expert slot.
///
/// Slots are **leaves**: each is one contiguous weight allocation, and two of
/// them sharing a byte means the zone handed the same ground out twice — the
/// aliasing that produced `gate weights for expert 27` and `expert 28` at one
/// address.
///
/// `capacity` is the zone's live slot count. Reading it from the caller rather
/// than from published geometry is deliberate: the geometry is exactly what goes
/// stale across a concession, and an audit that trusted it would go blind at the
/// moment it matters.
pub fn push_slot_claims(span_end: u64, slot_bytes: usize, capacity: usize, out: &mut Vec<Claim>) {
    if slot_bytes == 0 {
        return;
    }
    for i in 0..capacity {
        // Saturating: a capacity that would walk below address zero is itself
        // the bug, and wrapping would hide it by producing a plausible high
        // address instead of an obviously wrong low one.
        let base = span_end.saturating_sub(((i + 1) * slot_bytes) as u64);
        out.push(Claim::new(
            format!("expert/slot{i}"),
            Tenant::ExpertWeight,
            ClaimKind::Leaf,
            base,
            slot_bytes,
        ));
    }
}

/// Register the expert-side provider.
///
/// `live` is asked for the zone's current `(span_end, slot_bytes, capacity)` on
/// every call, because all three move: the floor is renegotiated, the zone
/// concedes and grows back, and a cached triple would describe a grid that no
/// longer exists.
pub fn register(live: impl Fn() -> Option<(u64, usize, usize)> + Send + Sync + 'static) {
    candle::span_audit::register("span/expert-slots", move |out| {
        if let Some((span_end, slot_bytes, capacity)) = live() {
            push_slot_claims(span_end, slot_bytes, capacity, out);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::span_audit::{find_overlaps, Relation};

    const SPAN_END: u64 = 0x10_0000;
    const SLOT: usize = 0x1000;

    #[test]
    fn slots_descend_from_the_top_of_the_span_and_do_not_overlap() {
        let mut c = Vec::new();
        push_slot_claims(SPAN_END, SLOT, 4, &mut c);
        assert_eq!(c.len(), 4);
        // Slot 0 is the highest, and its top is the span's top.
        assert_eq!(c[0].base, SPAN_END - SLOT as u64);
        assert_eq!(c[0].end(), SPAN_END);
        // Rising index walks down.
        assert_eq!(c[3].base, SPAN_END - 4 * SLOT as u64);
        assert!(
            find_overlaps(&c).is_empty(),
            "a healthy grid must not self-report"
        );
    }

    #[test]
    fn a_grid_that_outruns_the_weight_floor_is_caught() {
        // The capacity the zone believes it has reaches below the floor, so the
        // lowest slots stand on KV ground. This is the shape of a concession
        // that was never reflected in capacity.
        let floor = SPAN_END - 4 * SLOT as u64;
        let mut c = vec![
            Claim::new(
                "span/kv-regions",
                Tenant::KvRegion,
                ClaimKind::Container,
                0x0,
                floor as usize,
            ),
            Claim::new(
                "span/weights",
                Tenant::ExpertWeight,
                ClaimKind::Container,
                floor,
                (SPAN_END - floor) as usize,
            ),
        ];
        push_slot_claims(SPAN_END, SLOT, 6, &mut c); // two slots too many
        let o = find_overlaps(&c);
        assert!(
            o.iter().any(|x| x.relation == Relation::ForeignContainment),
            "slots below the floor must be flagged as trespass, got {o:?}"
        );
    }

    #[test]
    fn a_slot_aliased_onto_another_is_caught() {
        // Two experts resolving to one slot — measured in production as
        // `expert 27` and `expert 28` at the same address.
        let mut c = Vec::new();
        push_slot_claims(SPAN_END, SLOT, 3, &mut c);
        c.push(Claim::new(
            "expert/slot28",
            Tenant::ExpertWeight,
            ClaimKind::Leaf,
            c[1].base,
            SLOT,
        ));
        let o = find_overlaps(&c);
        assert!(
            o.iter().any(|x| x.relation == Relation::LeafOnLeaf),
            "two experts on one slot must be flagged, got {o:?}"
        );
    }

    #[test]
    fn an_empty_or_unsized_grid_claims_nothing() {
        let mut c = Vec::new();
        push_slot_claims(SPAN_END, SLOT, 0, &mut c);
        push_slot_claims(SPAN_END, 0, 8, &mut c);
        assert!(c.is_empty(), "nothing resident means nothing to claim");
    }

    #[test]
    fn a_capacity_that_would_walk_below_zero_saturates_rather_than_wrapping() {
        // Wrapping would turn an impossible slot into a plausible high address
        // and hide the fault; saturating leaves it at zero, which overlaps the
        // foot of the span and reports.
        let mut c = Vec::new();
        push_slot_claims(0x2000, 0x1000, 4, &mut c);
        assert!(c.iter().all(|x| x.base < 0x2000));
        assert_eq!(c[3].base, 0, "must clamp, not wrap");
    }
}
