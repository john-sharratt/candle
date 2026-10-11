//! Where a rejoining member's generation comes from
//! (`docs/replicated_substrate.md` §6.7).
//!
//! Generations are minted only by the group: generation `k` exists once the
//! Paxos decision `gen/<k>` names the member that rejoined with it, and the
//! decided generations run 1, 2, 3 … with no gap. The substrate does not run
//! the decision; it asks for one through [`MintGeneration`], which the
//! cluster's consensus implements, and fixes the generation of what it wrote
//! while cut off to the answer.

use std::sync::Mutex;

/// Decides a fresh generation for a member that rejoins.
pub trait MintGeneration {
    /// A generation decided for this member, above `known` — the greatest
    /// generation it knows to be decided. A decision for another member at
    /// `known + 1` raises the answer past it.
    fn mint_generation(&self, known: u64) -> u64;
}

/// A [`MintGeneration`] for members of one process: the decisions are one
/// counter, so each rejoin takes the next generation above both the counter
/// and what it already knows.
#[derive(Debug, Default)]
pub struct LocalGenerations {
    decided: Mutex<u64>,
}

impl LocalGenerations {
    pub fn new() -> Self {
        Self::default()
    }

    /// The greatest generation decided so far.
    pub fn decided(&self) -> u64 {
        *self.decided.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl MintGeneration for LocalGenerations {
    fn mint_generation(&self, known: u64) -> u64 {
        let mut decided = self.decided.lock().unwrap_or_else(|e| e.into_inner());
        *decided = (*decided).max(known) + 1;
        *decided
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decisions run without a gap, each above what the asker knows and above
    /// every decision before it.
    #[test]
    fn decisions_run_without_a_gap() {
        let g = LocalGenerations::new();
        assert_eq!(g.mint_generation(0), 1);
        assert_eq!(g.mint_generation(0), 2, "1 is taken: the next is above it");
        assert_eq!(g.mint_generation(5), 6);
        assert_eq!(g.decided(), 6);
    }
}
