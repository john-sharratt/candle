//! The overlap rules, separated from registration so they can be tested
//! without standing up a device or a single tenant.

use super::{Claim, ClaimKind, Tenant};

/// Why a pair of extents is a problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Relation {
    /// Two live buffers share bytes. Always wrong, whoever owns them.
    LeafOnLeaf,
    /// A buffer lies inside a container belonging to a different tenant — the
    /// trespass invariant 7 describes.
    ForeignContainment,
    /// A buffer crosses the edge of a container, so part of it is outside the
    /// zone it belongs to. Wrong even within one tenant: that is a buffer half
    /// on ground its owner does not hold.
    StraddlesBoundary,
    /// Two containers of different tenants overlap — a boundary is in the wrong
    /// place, and everything inside both is now suspect.
    ZoneOnZone,
}

impl Relation {
    pub fn as_str(self) -> &'static str {
        match self {
            Relation::LeafOnLeaf => "two live buffers on the same bytes",
            Relation::ForeignContainment => "buffer sits inside another tenant's zone",
            Relation::StraddlesBoundary => "buffer crosses its zone's edge",
            Relation::ZoneOnZone => "two tenants' zones overlap",
        }
    }
}

/// One offending pair.
#[derive(Debug, Clone)]
pub struct Overlap {
    pub a: Claim,
    pub b: Claim,
    pub relation: Relation,
    /// Bytes the two share.
    pub bytes: u64,
}

impl std::fmt::Display for Overlap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{rel}: {an} [{ab:#x}..{ae:#x}) ({at}) vs {bn} [{bb:#x}..{be:#x}) ({bt}) — {bytes} bytes shared",
            rel = self.relation.as_str(),
            an = self.a.name,
            ab = self.a.base,
            ae = self.a.end(),
            at = self.a.tenant.as_str(),
            bn = self.b.name,
            bb = self.b.base,
            be = self.b.end(),
            bt = self.b.tenant.as_str(),
            bytes = self.bytes,
        )
    }
}

/// Whether `inner` lies wholly within `outer`.
fn contains(outer: &Claim, inner: &Claim) -> bool {
    outer.base <= inner.base && inner.end() <= outer.end()
}

/// Classify an overlapping pair, or `None` when the pair is legitimate.
///
/// Only ever called on pairs already known to share at least one byte.
fn classify(a: &Claim, b: &Claim) -> Option<Relation> {
    let same_tenant = a.tenant == b.tenant;
    match (a.kind, b.kind) {
        // Two live buffers. Never legitimate — not even within one tenant,
        // which is the case that matters most here: one arena handing the same
        // slot to two owners looks exactly like this.
        (ClaimKind::Leaf, ClaimKind::Leaf) => Some(Relation::LeafOnLeaf),
        // Nesting is what containers are for, but only within one tenant.
        (ClaimKind::Container, ClaimKind::Container) => {
            (!same_tenant).then_some(Relation::ZoneOnZone)
        }
        // A buffer against a zone: fine only when it is wholly inside a zone of
        // its own tenant.
        (ClaimKind::Container, ClaimKind::Leaf) | (ClaimKind::Leaf, ClaimKind::Container) => {
            let (zone, leaf) = if a.kind == ClaimKind::Container {
                (a, b)
            } else {
                (b, a)
            };
            if !contains(zone, leaf) {
                // Partial: it is half in and half out. Wrong either way, but
                // name the tenant case distinctly — a leaf straddling a foreign
                // zone is a trespass, straddling its own is a boundary that
                // moved under it.
                return Some(if same_tenant {
                    Relation::StraddlesBoundary
                } else {
                    Relation::ForeignContainment
                });
            }
            (!same_tenant).then_some(Relation::ForeignContainment)
        }
    }
}

/// Every offending pair among `claims`.
///
/// Sorts by base and sweeps, comparing each extent only against those still
/// open at its start, so the cost is `O(n log n)` plus the number of overlapping
/// pairs rather than `O(n²)` — which matters because the claim list is every
/// arena, every resident expert and every live activation buffer.
///
/// Zero-length claims are dropped: they cannot share a byte with anything, and
/// keeping them would report a freshly-registered empty arena as colliding with
/// whatever happens to start at the same address.
pub fn find_overlaps(claims: &[Claim]) -> Vec<Overlap> {
    let mut live: Vec<&Claim> = claims.iter().filter(|c| c.len > 0).collect();
    live.sort_by_key(|c| (c.base, std::cmp::Reverse(c.end())));

    let mut out = Vec::new();
    for (i, a) in live.iter().enumerate() {
        for b in live.iter().skip(i + 1) {
            // Sorted by base, so once one starts at or after `a` ends, so does
            // everything after it.
            if b.base >= a.end() {
                break;
            }
            let shared = a.end().min(b.end()).saturating_sub(b.base);
            if shared == 0 {
                continue;
            }
            if let Some(relation) = classify(a, b) {
                out.push(Overlap {
                    a: (*a).clone(),
                    b: (*b).clone(),
                    relation,
                    bytes: shared,
                });
            }
        }
    }
    out
}

/// Every claim overlapping `[base, base + len)`, described.
///
/// The whole-partition audit answers "does anything overlap anything", which is
/// the right question between waves and the wrong one at a fault: by then the
/// question has narrowed to *one* buffer, and what is wanted is the list of
/// everyone else who claims its bytes.
///
/// It is also the only form that can see a fault which exists solely *inside* a
/// wave. The between-waves audit runs when the bump cursor has just reset and no
/// activation carve exists, so a buffer that collides only while a wave is
/// running is invisible to it — but perfectly visible here, asked at the moment
/// the buffer is being read.
///
/// Ordered most specific first, so the smallest extent containing the range —
/// the one that actually describes it — leads the report.
pub fn owners_of(claims: &[Claim], base: u64, len: usize) -> Vec<String> {
    if len == 0 {
        return Vec::new();
    }
    let end = base.saturating_add(len as u64);
    let mut hit: Vec<&Claim> = claims
        .iter()
        .filter(|c| c.len > 0 && c.base < end && base < c.end())
        .collect();
    hit.sort_by_key(|c| c.len);
    hit.iter()
        .map(|c| {
            let shared = end.min(c.end()).saturating_sub(base.max(c.base));
            format!(
                "{} ({}, {:?}) [{:#x}..{:#x}) — {} bytes shared",
                c.name,
                c.tenant.as_str(),
                c.kind,
                c.base,
                c.end(),
                shared
            )
        })
        .collect()
}

/// A pointer table's entries, checked against the extents they are supposed to
/// name.
///
/// A dispatch table holds one device address per expert, and the standing bug
/// this guards is a *stale* one: an address captured before a boundary move
/// still reads as a plausible pointer afterwards, because the ground is still
/// mapped and the geometry reads exactly as it did at load. Comparing the
/// address against the tenant that currently owns it is what makes that visible.
///
/// `entries` are `(label, address)` pairs read from the table. An address that
/// lands in no claim at all is reported too — it names ground nobody admits to
/// owning, which is worse than landing in the wrong tenant.
pub fn check_pointer_table(
    table: &str,
    expect: Tenant,
    entries: &[(String, u64)],
    claims: &[Claim],
) -> Vec<String> {
    let mut owners: Vec<&Claim> = claims.iter().filter(|c| c.len > 0).collect();
    owners.sort_by_key(|c| c.base);
    let mut bad = Vec::new();
    for (label, addr) in entries {
        // The most specific owner is the smallest extent containing the address,
        // so a slot beats the arena beats the zone.
        let owner = owners
            .iter()
            .filter(|c| c.base <= *addr && *addr < c.end())
            .min_by_key(|c| c.len);
        match owner {
            None => bad.push(format!(
                "{table}[{label}] -> {addr:#x} is inside NO registered extent — it names ground \
                 no tenant claims"
            )),
            Some(c) if c.tenant != expect => bad.push(format!(
                "{table}[{label}] -> {addr:#x} is owned by {} ({}), expected {}",
                c.name,
                c.tenant.as_str(),
                expect.as_str()
            )),
            Some(_) => {}
        }
    }
    bad
}
