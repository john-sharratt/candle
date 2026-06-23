//! Opaque identifier newtypes.
//!
//! All identifiers are **assigned by the crate** — never constructed by the
//! caller. They are the addressing mechanism between the user's content store
//! and the projection engine: the engine emits ids in a [`Projection`], the
//! caller looks up content by id.
//!
//! # Identifier scope
//!
//! ```text
//!  ┌────────────────────────────────────────────────────────────────┐
//!  │  Schema                                                        │
//!  │  ┌──────────────────────────────────────────────────────────┐  │
//!  │  │  SystemPrompt                                            │  │
//!  │  │   • SectionId(1), SectionId(2), …  (whole-schema scope) │  │
//!  │  └──────────────────────────────────────────────────────────┘  │
//!  │  ┌──────────────────────────────────────────────────────────┐  │
//!  │  │  Layer  LayerId(1)         (whole-schema scope)         │  │
//!  │  │   ├─ Group GroupId(1)      (GLOBALLY unique across       │  │
//!  │  │   │    └─ TurnIndex(0..n)   all layers)                  │  │
//!  │  │   └─ Group GroupId(2)                                    │  │
//!  │  │        └─ TurnIndex(0..n)  (group-local, monotonic)     │  │
//!  │  │  Layer  LayerId(2)                                       │  │
//!  │  │   └─ Group GroupId(3)                                    │  │
//!  │  └──────────────────────────────────────────────────────────┘  │
//!  └────────────────────────────────────────────────────────────────┘
//! ```
//!
//! - **`LayerId`**, **`GroupId`**, **`SectionId`** — assigned at schema
//!   construction. Stable for the lifetime of the [`super::Builder`]. Use
//!   [`NonZeroU32`] internally so `Option<LayerId>` etc. are pointer-sized.
//! - **`TurnIndex`** — assigned at append time, scoped to a single group,
//!   monotonically increasing from `0`.
//!
//! [`NonZeroU32`]: std::num::NonZeroU32
//! [`Projection`]: super::Projection

use std::num::{NonZeroU32, NonZeroU64};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Names an *engine-internal* projection-id category — a "well-known" kind
/// of conversation the engine builds on its own (not user-defined YAML).
///
/// Reserved ids occupy slots at the top of the [`u32`] range so they can
/// never collide with the `1..n` ids allocated to user schemas by YAML
/// parsing. The compiler is the only thing that can construct one — there
/// is no public way to fabricate a reserved id from a magic integer.
///
/// Adding a new internal kind is one new variant plus one [`Self::slot`]
/// arm. Each kind reuses the same slot across all three of
/// [`LayerId`] / [`GroupId`] / [`SectionId`] — they're disjoint newtypes
/// so cross-type collisions can't happen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Reserved {
    /// The daemon's titler conversation — generates sidebar labels from
    /// the first user message of each main conversation. Lives on its
    /// own layer/group/section so its turns never enter a user
    /// conversation's projection.
    Titler,
    /// The cached tool-catalog summary section for "Comprehensive" tools mode —
    /// an overview of the full catalog. Sealed at runtime (its content is
    /// model-generated, not in the schema) and pinned under this reserved
    /// [`SectionId`] so it can be injected just before the `tools` collection.
    ToolSummary,
    /// The cached tool-catalog summary section for "Restricted" tools mode — an
    /// overview built from the safe (non-high-risk) tool subset only. The
    /// Restricted-mode projection points the `tools` collection at this section
    /// instead of [`Reserved::ToolSummary`]; "None" mode emits neither.
    ToolSummaryRestricted,
}

impl Reserved {
    /// Number of reserved kinds — the width of the band at the very top of the
    /// u32 space that is disjoint from the `1..n` ids YAML allocates. Bump this
    /// when adding a `Reserved` variant.
    pub const COUNT: u32 = 3;

    /// Per-kind offset from the top of the u32 range. Slot 0 = `u32::MAX`,
    /// slot 1 = `u32::MAX - 1`, etc.
    const fn slot(self) -> u32 {
        match self {
            Reserved::Titler => 0,
            Reserved::ToolSummary => 1,
            Reserved::ToolSummaryRestricted => 2,
        }
    }

    /// Raw numeric value used by every reserved id of this kind.
    const fn raw(self) -> u32 {
        u32::MAX - self.slot()
    }
}

/// Opaque identifier for a layer. Assigned in declaration order at construction
/// (first layer = `LayerId(1)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LayerId(pub(super) NonZeroU32);

impl LayerId {
    /// Constructor restricted to the projection module.
    pub(super) fn new(n: u32) -> Self {
        Self(NonZeroU32::new(n).expect("LayerId must be non-zero"))
    }

    /// The reserved layer id for an engine-internal kind. Disjoint from
    /// the `1..n` range YAML allocates.
    pub const fn reserved(kind: Reserved) -> Self {
        // Safety: every `Reserved::*` slot maps to a value in
        // `(0, u32::MAX]`, all of which are non-zero.
        match NonZeroU32::new(kind.raw()) {
            Some(nz) => Self(nz),
            None => panic!("Reserved::raw produced zero"),
        }
    }

    #[cfg(any(test, feature = "test-helpers"))]
    pub fn for_test(n: u32) -> Self {
        Self(NonZeroU32::new(n.max(1)).unwrap())
    }

    /// Raw integer for diagnostics or external persistence.
    pub fn raw(self) -> u32 {
        self.0.get()
    }

    /// True if this id is in the reserved engine-internal band at the top of
    /// the u32 space (e.g. [`Reserved::Titler`]), rather than the `1..n` range
    /// YAML allocates for user layers. Engine-internal conversations have no
    /// user-facing projection/summary and are excluded from compression.
    pub fn is_reserved(self) -> bool {
        self.0.get() > u32::MAX - Reserved::COUNT
    }

    /// Rebuild a `LayerId` from a persisted [`Self::raw`] value — the resume
    /// path. `None` for the reserved `0` sentinel.
    pub fn from_raw(n: u32) -> Option<Self> {
        NonZeroU32::new(n).map(Self)
    }
}

/// Opaque identifier for a group. **Globally unique** across all layers in a
/// schema — not scoped per-layer. Assigned in (layer order × group order) at
/// construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GroupId(pub(super) NonZeroU32);

impl GroupId {
    pub(super) fn new(n: u32) -> Self {
        Self(NonZeroU32::new(n).expect("GroupId must be non-zero"))
    }

    /// The reserved group id for an engine-internal kind. Disjoint from
    /// the `1..n` range YAML allocates.
    pub const fn reserved(kind: Reserved) -> Self {
        match NonZeroU32::new(kind.raw()) {
            Some(nz) => Self(nz),
            None => panic!("Reserved::raw produced zero"),
        }
    }

    #[cfg(any(test, feature = "test-helpers"))]
    pub fn for_test(n: u32) -> Self {
        Self(NonZeroU32::new(n.max(1)).unwrap())
    }

    pub fn raw(self) -> u32 {
        self.0.get()
    }

    /// Rebuild a `GroupId` from a persisted [`Self::raw`] value — the resume
    /// path. `None` for the reserved `0` sentinel.
    pub fn from_raw(n: u32) -> Option<Self> {
        NonZeroU32::new(n).map(Self)
    }
}

/// Opaque identifier for a system-prompt section. Assigned in declaration
/// order at construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SectionId(pub(super) NonZeroU32);

impl SectionId {
    pub fn new(n: u32) -> Self {
        Self(NonZeroU32::new(n).expect("SectionId must be non-zero"))
    }

    /// The reserved section id for an engine-internal kind. Disjoint from
    /// the `1..n` range YAML allocates.
    pub const fn reserved(kind: Reserved) -> Self {
        match NonZeroU32::new(kind.raw()) {
            Some(nz) => Self(nz),
            None => panic!("Reserved::raw produced zero"),
        }
    }

    pub fn raw(self) -> u32 {
        self.0.get()
    }
}

/// Opaque identifier for a [`super::SectionCollection`].  Globally unique
/// across the schema; assigned in declaration order alongside other ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CollectionId(pub(super) NonZeroU32);

impl CollectionId {
    pub(super) fn new(n: u32) -> Self {
        Self(NonZeroU32::new(n).expect("CollectionId must be non-zero"))
    }

    pub fn raw(self) -> u32 {
        self.0.get()
    }
}

/// Turn index within a single timeline. Monotonically increasing from `0`,
/// assigned by the substrate at append time. Scoped to its timeline — a
/// `TurnIndex` from one timeline cannot be used to address turns in another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TurnIndex(pub u32);

impl TurnIndex {
    pub fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl std::fmt::Display for TurnIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "t{}", self.0)
    }
}

/// Fully-qualified address of one turn in the workspace — the `(timeline,
/// index)` pair as a strongly-typed key.
///
/// Use [`TurnKey`] anywhere two-arg `(TimelineId, TurnIndex)` would
/// otherwise be passed as an anonymous tuple: HashMap keys, iterator
/// items, function returns, eviction candidate handles. Keeps call sites
/// self-documenting and rules out the silent positional-swap bug.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct TurnKey {
    pub timeline: TimelineId,
    pub index: TurnIndex,
}

impl TurnKey {
    pub fn new(timeline: TimelineId, index: TurnIndex) -> Self {
        Self { timeline, index }
    }
}

/// Workspace-stable identifier for one *instance* of a group's shape — i.e.
/// one conversation timeline within a (layer, group) pair.
///
/// # Why this exists
///
/// `GroupId` describes a **kind** of group (the shape declared in YAML:
/// selection rule, budget, score formula).  Multiple parallel conversations
/// of the same kind — e.g. two open chats both targeting
/// `dialogue/primary_conversation` — need separate timelines so their turns
/// don't interleave.  `TimelineId` is that per-instance address: each
/// conversation has its own monotonic timeline, and the substrate maps
/// `TimelineId → (LayerId, GroupId)` to recover the shape on resume.
///
/// # Encoding
///
/// 64-bit integer that **is** a microsecond UNIX timestamp under typical
/// conditions.  Bursts and clock skew are absorbed by [`TimelineAllocator`]
/// (each allocation returns `max(now_micros, last_issued + 1)`), so the
/// stored value may run slightly ahead of real time during a burst — but
/// the value is always strictly monotonic and unique, and the high bits
/// always read as a recognisable timestamp in logs.
///
/// `NonZeroU64` so `Option<TimelineId>` is pointer-sized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TimelineId(NonZeroU64);

impl TimelineId {
    /// Construct from a raw `NonZeroU64`.  Restricted to the projection
    /// crate so callers can't forge IDs — they must go through
    /// [`TimelineAllocator::next`] or the on-disk replay path.
    #[allow(dead_code)] // used by Phase 2's `Conversation::open` replay path
    pub(crate) fn new(raw: NonZeroU64) -> Self {
        Self(raw)
    }

    /// Test-only constructor that returns a fixed, deterministic id.
    /// Projection unit tests use a mock resolver that ignores
    /// `target.timeline` semantics, so any non-zero value works — this
    /// helper just keeps the test code terse.
    #[cfg(any(test, feature = "test-helpers"))]
    pub fn for_test(n: u64) -> Self {
        Self(NonZeroU64::new(n.max(1)).unwrap())
    }

    /// Raw integer for diagnostics or external persistence.
    pub fn raw(self) -> u64 {
        self.0.get()
    }

    /// Rebuild a `TimelineId` from a persisted [`Self::raw`] value — the
    /// on-disk substrate-reload path. `None` for the reserved `0` sentinel.
    pub fn from_raw(n: u64) -> Option<Self> {
        NonZeroU64::new(n).map(Self)
    }
}

impl std::fmt::Display for TimelineId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.get())
    }
}

/// Workspace-scoped allocator for [`TimelineId`].
///
/// Each call to [`Self::next`] returns `max(now_micros, last_issued + 1)`,
/// which guarantees:
///
/// - **Strict monotonicity** — IDs are assigned in allocation order.
/// - **Burst-safe uniqueness** — many allocations within the same microsecond
///   never collide; the second through Nth get `last + 1`, `last + 2`, …
/// - **Clock-skew safety** — a backward NTP jump can't issue a duplicate
///   because we always advance past `last_issued`.
/// - **Resume safety** — when [`crate::projection::Conversation::open`]
///   replays a persisted substrate, the allocator is seeded with the largest
///   on-disk `TimelineId` so freshly minted ids are guaranteed unique against
///   the recovered set.
///
/// Internally an [`AtomicU64`] so allocation is lock-free.
#[derive(Debug)]
pub struct TimelineAllocator {
    last: AtomicU64,
}

impl TimelineAllocator {
    /// Create a fresh allocator that has never issued an id.  The first
    /// [`Self::next`] call will return the current microsecond timestamp.
    pub fn new() -> Self {
        Self {
            last: AtomicU64::new(0),
        }
    }

    /// Seed the allocator with a high-water mark.  Used by the
    /// `Conversation::open` replay path to ensure ids minted after a
    /// resume don't collide with anything already on disk.
    pub fn seed(&self, high_water: u64) {
        self.last.fetch_max(high_water, Ordering::Relaxed);
    }

    /// Allocate the next [`TimelineId`].
    ///
    /// Returns `max(now_micros, last_issued + 1)` and advances the
    /// internal high-water mark to that value.  Lock-free.
    pub fn next(&self) -> TimelineId {
        let now_us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);
        // CAS loop: read last, compute next = max(now, last+1), store.
        let mut last = self.last.load(Ordering::Relaxed);
        loop {
            let next = now_us.max(last.saturating_add(1));
            match self
                .last
                .compare_exchange_weak(last, next, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => {
                    let nz = NonZeroU64::new(next).unwrap_or_else(|| {
                        // Only reachable if `now_us == 0` AND last was 0,
                        // which can only happen on the first allocation
                        // when SystemTime::now() failed.  Fall back to 1.
                        NonZeroU64::new(1).unwrap()
                    });
                    return TimelineId(nz);
                }
                Err(observed) => last = observed,
            }
        }
    }
}

impl Default for TimelineAllocator {
    fn default() -> Self {
        Self::new()
    }
}

/// Fully self-describing turn identifier.
///
/// Three pieces of addressing information locate any turn within a
/// workspace's substrate.  The substrate IS the workspace
/// [`super::Conversation`] handle, so there's no per-turn
/// "conversation id" — every turn in this substrate was produced
/// within this workspace by definition.
///
/// - **`layer_id`** — denormalised; derivable from `group_id` via the schema,
///   carried inline so projection emit doesn't need a back-lookup.
/// - **`group_id`** — globally unique across the schema.
/// - **`index`** — monotonic per-group, assigned by [`super::Builder::append`].
///
/// Two [`TurnId`]s compare equal iff all three fields match.  This is the
/// primary key for the workspace substrate's per-turn record map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TurnId {
    pub layer_id: LayerId,
    pub group_id: GroupId,
    pub index: TurnIndex,
}
