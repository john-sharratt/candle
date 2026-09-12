//! The rules, pinned. Every case here is a layout that has actually occurred in
//! this system or is one boundary move away from occurring.

use super::{check_pointer_table, find_overlaps, Claim, ClaimKind, Relation, Tenant};

fn zone(name: &'static str, tenant: Tenant, base: u64, len: usize) -> Claim {
    Claim::new(name, tenant, ClaimKind::Container, base, len)
}
fn buf(name: &'static str, tenant: Tenant, base: u64, len: usize) -> Claim {
    Claim::new(name, tenant, ClaimKind::Leaf, base, len)
}

/// The healthy layout: the partition CLAUDE.md describes, each zone holding its
/// own buffers, nothing shared.
fn healthy() -> Vec<Claim> {
    vec![
        zone("persist", Tenant::Persist, 0x1000, 0x1000),
        zone("kv", Tenant::KvRegion, 0x2000, 0x4000),
        zone("tier", Tenant::TransientTier, 0x6000, 0x2000),
        zone("weights", Tenant::ExpertWeight, 0x8000, 0x4000),
        buf("kv/arena0", Tenant::KvRegion, 0x2000, 0x1000),
        buf("kv/arena1", Tenant::KvRegion, 0x3000, 0x1000),
        buf("acts/wave", Tenant::TransientTier, 0x6000, 0x800),
        buf("expert/17", Tenant::ExpertWeight, 0x8000, 0x400),
    ]
}

#[test]
fn a_correct_partition_reports_nothing() {
    assert!(
        find_overlaps(&healthy()).is_empty(),
        "nested zones and their own buffers must not be flagged"
    );
}

#[test]
fn two_live_buffers_on_one_byte_is_always_wrong() {
    // The allocator handing one slot to two owners — what a recycled gid that is
    // still referenced looks like.
    let mut c = healthy();
    c.push(buf("kv/arena1-alias", Tenant::KvRegion, 0x3800, 0x1000));
    let o = find_overlaps(&c);
    assert_eq!(o.len(), 1, "expected exactly one pair, got {o:?}");
    assert_eq!(o[0].relation, Relation::LeafOnLeaf);
    assert_eq!(o[0].bytes, 0x800, "shared extent");
}

#[test]
fn a_buffer_inside_another_tenants_zone_is_a_trespass() {
    // Invariant 7's headline case: the tier placed above `weight_floor`, so an
    // activation buffer stands on resident expert ground.
    let mut c = healthy();
    c.push(buf("acts/spill", Tenant::TransientTier, 0x8100, 0x100));
    let o = find_overlaps(&c);
    assert!(
        o.iter().any(|x| x.relation == Relation::ForeignContainment),
        "an activation inside the weight zone must be flagged, got {o:?}"
    );
}

#[test]
fn a_buffer_crossing_its_own_zones_edge_is_wrong_too() {
    // The floor moved under a standing tier: the buffer is half in its zone.
    // Within one tenant, so nothing about *ownership* looks wrong — only the
    // geometry does, which is why this case needs its own rule.
    let mut c = healthy();
    c.push(buf("tier/straddle", Tenant::TransientTier, 0x7F00, 0x400));
    let o = find_overlaps(&c);
    assert!(
        o.iter().any(|x| x.relation == Relation::StraddlesBoundary),
        "a leaf straddling its own container edge must be flagged, got {o:?}"
    );
}

#[test]
fn two_tenants_zones_overlapping_is_a_misplaced_boundary() {
    let c = vec![
        zone("kv", Tenant::KvRegion, 0x2000, 0x4000),
        zone("weights", Tenant::ExpertWeight, 0x5000, 0x4000),
    ];
    let o = find_overlaps(&c);
    assert_eq!(o.len(), 1);
    assert_eq!(o[0].relation, Relation::ZoneOnZone);
    assert_eq!(o[0].bytes, 0x1000);
}

#[test]
fn nested_zones_of_one_tenant_are_fine() {
    // A region inside the KV zone, an arena inside the region.
    let c = vec![
        zone("kv", Tenant::KvRegion, 0x2000, 0x4000),
        zone("kv/region0", Tenant::KvRegion, 0x2000, 0x2000),
        zone("kv/region0/arena3", Tenant::KvRegion, 0x2400, 0x400),
    ];
    assert!(find_overlaps(&c).is_empty());
}

#[test]
fn zero_length_claims_cannot_collide() {
    // A registered-but-unmaterialised arena. Reporting it would make every
    // fresh registration look like a collision with whatever starts there.
    let c = vec![
        buf("empty", Tenant::KvRegion, 0x3000, 0),
        buf("real", Tenant::KvRegion, 0x3000, 0x100),
    ];
    assert!(find_overlaps(&c).is_empty());
}

#[test]
fn abutting_extents_do_not_overlap() {
    // `end` is exclusive. An off-by-one here would flag every adjacent arena in
    // the pool and bury the real finding.
    let c = vec![
        buf("a", Tenant::KvRegion, 0x1000, 0x100),
        buf("b", Tenant::KvRegion, 0x1100, 0x100),
    ];
    assert!(find_overlaps(&c).is_empty());
}

#[test]
fn a_length_that_would_wrap_the_address_space_saturates() {
    // A garbage length must not wrap to a low `end` and silently stop
    // overlapping everything above it.
    let c = vec![
        buf("huge", Tenant::Other, u64::MAX - 16, usize::MAX),
        buf("victim", Tenant::KvRegion, u64::MAX - 8, 4),
    ];
    let o = find_overlaps(&c);
    assert_eq!(o.len(), 1, "saturating end must still overlap, got {o:?}");
}

#[test]
fn every_offending_pair_is_reported_not_just_adjacent_ones() {
    // Three buffers all sharing one byte range: the sweep must yield all three
    // pairs, or a chain of aliases reports as a single collision and the
    // narrowing stalls.
    let c = vec![
        buf("a", Tenant::KvRegion, 0x1000, 0x300),
        buf("b", Tenant::KvRegion, 0x1100, 0x300),
        buf("c", Tenant::KvRegion, 0x1200, 0x300),
    ];
    assert_eq!(find_overlaps(&c).len(), 3);
}

#[test]
fn a_pointer_table_entry_in_the_wrong_tenant_is_named() {
    // The standing MoE hazard: a slot address cached before a concession, still
    // a valid pointer afterwards, now naming ground the KV side owns.
    let claims = healthy();
    let entries = vec![
        ("expert0".to_string(), 0x8000u64),
        ("expert1".to_string(), 0x3100u64), // conceded — now KV
    ];
    let bad = check_pointer_table("gate_ptrs", Tenant::ExpertWeight, &entries, &claims);
    assert_eq!(bad.len(), 1, "got {bad:?}");
    assert!(bad[0].contains("expert1"), "{}", bad[0]);
    assert!(bad[0].contains("kv"), "{}", bad[0]);
}

#[test]
fn a_pointer_table_entry_owned_by_nobody_is_named() {
    let claims = healthy();
    let entries = vec![("expert9".to_string(), 0xdead_0000u64)];
    let bad = check_pointer_table("gate_ptrs", Tenant::ExpertWeight, &entries, &claims);
    assert_eq!(bad.len(), 1);
    assert!(bad[0].contains("NO registered extent"), "{}", bad[0]);
}

#[test]
fn a_pointer_table_resolves_to_the_most_specific_owner() {
    // An address inside both the zone and a slot must be attributed to the slot,
    // otherwise every entry resolves to the enclosing zone and a slot-level
    // mix-up is invisible.
    let claims = vec![
        zone("weights", Tenant::ExpertWeight, 0x8000, 0x4000),
        buf("kv/stray", Tenant::KvRegion, 0x8100, 0x100),
    ];
    let entries = vec![("expert3".to_string(), 0x8180u64)];
    let bad = check_pointer_table("gate_ptrs", Tenant::ExpertWeight, &entries, &claims);
    assert_eq!(bad.len(), 1, "the inner KV buffer must win over the zone");
    assert!(bad[0].contains("kv/stray"), "{}", bad[0]);
}
