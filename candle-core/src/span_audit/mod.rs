//! Who owns which bytes of the device reservation, checked between waves.
//!
//! The span is one contiguous reservation shared by several tenants —
//! `| persist | KV regions | wave transient tier | expert weights |` — and
//! **nothing in CUDA enforces the boundaries**. Every address inside the span is
//! mapped, so a tenant that walks into another's ground reads and writes live
//! data and raises nothing at all. It surfaces as a wrong number many layers
//! later, never as a fault (CLAUDE.md hot-path invariant 7).
//!
//! Every instrument beside this one answers *"is this value finite"*, which
//! catches the symptom. This one answers *"does anybody's memory overlap anybody
//! else's"*, which catches the cause, and it can answer it **before** a single
//! bad number has been produced.
//!
//! # Nesting is legal; trespass is not
//!
//! A naive checker that flags any two overlapping ranges is useless here,
//! because the layout is deliberately nested: the span contains regions, regions
//! contain arenas, an arena contains chunk slots. So each claim declares both
//! **whose** it is ([`Tenant`]) and **what kind of extent** it is
//! ([`ClaimKind`]), and the rules follow from the invariant rather than from
//! geometry alone:
//!
//! | | same tenant | different tenant |
//! |---|---|---|
//! | leaf ∩ leaf | **error** — two live buffers on one byte | **error** |
//! | leaf ⊂ container | fine — an arena inside its own zone | **error** — trespass |
//! | leaf ⋂ container, partial | **error** — straddles its own boundary | **error** |
//! | container ∩ container | fine — nesting | **error** |
//!
//! A leaf straddling its own tenant's container edge is an error on purpose:
//! that is a buffer half inside its zone, which is how a boundary move leaves a
//! tenant standing on ground it no longer owns.
//!
//! # Registration, not plumbing
//!
//! Each subsystem knows its own extents and nothing else does, so tenants
//! register a provider closure and the audit calls them all. That keeps this
//! module free of any dependency on the KV cache, the expert grid or the wave
//! arena — each of which lives in a different crate and none of which can see
//! the others.
//!
//! Providers are expected to report **live** extents. A provider that reports a
//! freed buffer manufactures a false positive, exactly as a stale
//! `readonly_regions` declaration does.
//!
//! # Cost
//!
//! Host-side only: no device work, no synchronisation, no readback. It is a
//! `Vec` build plus a sort, so it belongs at a wave boundary where a few
//! microseconds do not matter — never inside a wave. Behind `tensor-assert`, so
//! it compiles to nothing without the feature.

use std::borrow::Cow;
use std::sync::RwLock;

mod detect;
#[cfg(test)]
mod tests;

pub use detect::{check_pointer_table, find_overlaps, owners_of, Overlap, Relation};

/// Which tenant of the reservation a claim belongs to.
///
/// Two claims of different tenants may never share a byte, whatever their kind.
/// The variants are the partition CLAUDE.md names, plus the buffers that are not
/// span tenants at all but can still collide with one another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Tenant {
    /// The persist region at the foot of the span.
    Persist,
    /// KV regions — arenas and the chunk slots inside them.
    KvRegion,
    /// The wave transient tier.
    TransientTier,
    /// Resident expert weights above `weight_floor`.
    ExpertWeight,
    /// Per-wave activation buffers.
    Activation,
    /// Device tables of addresses — the MoE dispatch tables, the paged-attention
    /// arena tables. Their *contents* point into other tenants, which is checked
    /// separately from where the table itself lives.
    PointerTable,
    /// Anything registered that does not belong to the partition.
    Other,
}

impl Tenant {
    pub fn as_str(self) -> &'static str {
        match self {
            Tenant::Persist => "persist",
            Tenant::KvRegion => "kv",
            Tenant::TransientTier => "tier",
            Tenant::ExpertWeight => "weights",
            Tenant::Activation => "activation",
            Tenant::PointerTable => "ptr-table",
            Tenant::Other => "other",
        }
    }
}

/// Whether a claim is an extent that legitimately holds others of its tenant, or
/// an individual live buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimKind {
    /// A zone, region or arena — expected to contain leaves of the same tenant.
    Container,
    /// One live buffer. Two of these may never share a byte.
    Leaf,
}

/// One registered extent of device memory.
#[derive(Debug, Clone)]
pub struct Claim {
    /// What to call this in a report — an arena index, an expert id, a site.
    pub name: Cow<'static, str>,
    pub tenant: Tenant,
    pub kind: ClaimKind,
    pub base: u64,
    pub len: usize,
}

impl Claim {
    pub fn new(
        name: impl Into<Cow<'static, str>>,
        tenant: Tenant,
        kind: ClaimKind,
        base: u64,
        len: usize,
    ) -> Self {
        Self {
            name: name.into(),
            tenant,
            kind,
            base,
            len,
        }
    }

    /// One past the last byte. Saturating, so a nonsense length cannot wrap the
    /// address space and silently stop overlapping everything.
    pub fn end(&self) -> u64 {
        self.base.saturating_add(self.len as u64)
    }
}

type Provider = Box<dyn Fn(&mut Vec<Claim>) + Send + Sync + 'static>;

fn providers() -> &'static RwLock<Vec<(&'static str, Provider)>> {
    static P: std::sync::OnceLock<RwLock<Vec<(&'static str, Provider)>>> = std::sync::OnceLock::new();
    P.get_or_init(|| RwLock::new(Vec::new()))
}

/// Register a tenant's extent provider. Idempotent per `name`: registering the
/// same name twice replaces the first, so a subsystem rebuilt mid-run does not
/// end up reporting its extents twice and colliding with itself.
pub fn register(name: &'static str, f: impl Fn(&mut Vec<Claim>) + Send + Sync + 'static) {
    if let Ok(mut p) = providers().write() {
        if let Some(slot) = p.iter_mut().find(|(n, _)| *n == name) {
            slot.1 = Box::new(f);
        } else {
            p.push((name, Box::new(f)));
        }
    }
}

/// Drop a provider — for a subsystem being torn down, whose extents would
/// otherwise be reported after the memory is gone.
pub fn unregister(name: &'static str) {
    if let Ok(mut p) = providers().write() {
        p.retain(|(n, _)| *n != name);
    }
}

/// Collect every registered claim into `out`, which is cleared first.
///
/// Takes the buffer rather than returning one so a caller running this every
/// wave can keep a single allocation alive across the whole run — see
/// [`audit`]. Providers append; none of them clears.
pub fn collect_into(out: &mut Vec<Claim>) {
    out.clear();
    if let Ok(p) = providers().read() {
        for (_, f) in p.iter() {
            f(out);
        }
    }
}

/// Collect every registered claim.
pub fn collect() -> Vec<Claim> {
    let mut out = Vec::new();
    collect_into(&mut out);
    out
}

/// How many providers are registered, so a caller can tell "nothing overlaps"
/// from "nobody is looking" — the distinction that made every capture's
/// `drifted=probe-not-registered` meaningless until it was read carefully.
pub fn provider_count() -> usize {
    providers().read().map(|p| p.len()).unwrap_or(0)
}

/// Who else claims `[base, base + len)`, asked of the live providers.
///
/// For a caller that has one suspect buffer and wants the tenants sharing it —
/// see [`owners_of`]. Collects fresh rather than reusing a cached claim list,
/// because this is called at a fault, where the layout as it is *now* is the
/// whole point.
pub fn who_owns(base: u64, len: usize) -> Vec<String> {
    owners_of(&collect(), base, len)
}

/// Audit every registered extent and log what overlaps.
///
/// Returns the overlaps so a caller can panic on them if it would rather stop
/// than continue; logging alone is the default because the first run of this
/// check on a live system is as likely to find a mis-registered provider as a
/// real trespass.
pub fn audit(context: &str) -> Vec<Overlap> {
    // **Reused across waves.** This runs at every wave boundary, and the claim
    // list is every zone, every arena, every resident expert and every live
    // activation buffer — a fresh `Vec` per wave would be a real allocation on
    // a path whose whole justification is that it is cheap. The buffer keeps
    // its capacity, so after the first wave the collect is pure writes.
    thread_local! {
        static SCRATCH: std::cell::RefCell<Vec<Claim>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    SCRATCH.with(|s| {
        let mut claims = s.borrow_mut();
        collect_into(&mut claims);
        audit_claims(context, &claims)
    })
}

/// [`audit`] over a claim list the caller already holds.
pub fn audit_claims(context: &str, claims: &[Claim]) -> Vec<Overlap> {
    if claims.is_empty() {
        tracing::warn!(
            target: "candle_core::span_audit",
            context, providers = provider_count(),
            "span audit: NO extents registered — this reports nothing, which is not the \
             same as reporting no overlaps"
        );
        return Vec::new();
    }
    let overlaps = find_overlaps(claims);
    if overlaps.is_empty() {
        tracing::debug!(
            target: "candle_core::span_audit",
            context, claims = claims.len(), providers = provider_count(),
            "span audit: clean"
        );
        return overlaps;
    }
    tracing::error!(
        target: "candle_core::span_audit",
        context, claims = claims.len(), overlaps = overlaps.len(),
        "span audit: TENANTS SHARE MEMORY — every overlap below is two owners of one byte, \
         which CUDA will not fault on and which surfaces as a wrong number many layers later"
    );
    for o in &overlaps {
        tracing::error!(target: "candle_core::span_audit", "  {o}");
    }
    overlaps
}
