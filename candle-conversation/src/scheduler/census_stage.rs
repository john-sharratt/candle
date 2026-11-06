//! The memory census's scopes for the scheduler loop's stages
//! (`docs/memory_census.md` §10).
//!
//! The scheduler thread names itself `Scheduler` as it starts; each stage of its
//! loop narrows that for as long as the stage runs, so a block the stage makes
//! is counted as the stage's. The guard is the census's own, and without the
//! `memory-census` feature [`enter`] is an empty inline function returning an
//! empty guard — no call site carries a `#[cfg]`, and a build without the
//! census pays nothing for them.

#[cfg(feature = "memory-census")]
use heap_census::{Scope, Scoped};

/// A stage of the scheduler's loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Stage {
    /// The decode quantum.
    Decode,
    /// The prefill quantum.
    Prefill,
    /// Sealing finished turns, and ingesting sections.
    Seal,
    /// The gallery's keep and sweep, and the published rates.
    Gallery,
    /// Answering one request.
    Requests,
}

#[cfg(feature = "memory-census")]
impl Stage {
    /// The census scope the stage is counted under.
    fn scope(self) -> Scope {
        match self {
            Stage::Decode => Scope::Decode,
            Stage::Prefill => Scope::Prefill,
            Stage::Seal => Scope::TurnSeal,
            Stage::Gallery => Scope::Gallery,
            Stage::Requests => Scope::Requests,
        }
    }
}

/// Count what the calling thread makes as `stage`'s until the guard drops.
#[cfg(feature = "memory-census")]
#[inline(always)]
pub(super) fn enter(stage: Stage) -> Scoped {
    heap_census::enter(stage.scope())
}

/// The guard [`enter`] returns without the census: nothing to put back.
#[cfg(not(feature = "memory-census"))]
pub(super) struct Uncounted;

/// Without the census there is nothing to count under.
#[cfg(not(feature = "memory-census"))]
#[inline(always)]
pub(super) fn enter(_stage: Stage) -> Uncounted {
    Uncounted
}

#[cfg(all(test, feature = "memory-census"))]
mod tests {
    use super::*;

    /// **Each stage has a scope of its own**, so two stages' blocks are never
    /// counted together.
    #[test]
    fn every_stage_names_its_own_scope() {
        let stages = [
            Stage::Decode,
            Stage::Prefill,
            Stage::Seal,
            Stage::Gallery,
            Stage::Requests,
        ];
        let names: Vec<&str> = stages.iter().map(|s| s.scope().name()).collect();
        assert_eq!(
            names,
            ["decode", "prefill", "turn_seal", "gallery", "requests"]
        );
    }
}
