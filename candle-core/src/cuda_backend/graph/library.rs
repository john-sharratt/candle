//! The executables kept across waves, found by what a segment is rather than
//! by where in its wave it falls.
//!
//! A segment ends wherever the forward meets the device eagerly — the MoE host
//! protocol's flushes, an upload, a readback — and where those fall moves with
//! the wave's contents. Keyed by ordinal, segment `i` held a different run of
//! layers from one wave to the next: on npcd's mind ingest (Qwen3.8-Flash-Next,
//! RTX 3090) ordinal 6 was offered executables of 1,304, 665, 652, 1,395, 792
//! and 1,256 nodes, beginning at a combine kernel in one wave and a copy in the
//! next, folded into none of them and was instantiated afresh. That came to
//! 10.8 s of `cuGraphInstantiate` in 30 s of prefill, up to 3.4 s a segment,
//! with the GPU idle behind it.
//!
//! Keyed by its topology — every node's kind, in order — a run of layers finds
//! the executables it was folded into before, wherever the cuts put it. The
//! kernel at a node is not part of the key: a wave's width chooses among a
//! projection's int8 variants (`_m4`, `_nw`, `_sk`) and a GEMM's tiles, and
//! the driver rewrites the function on a fold, so a segment that differs only
//! in which variant it launched is the same topology. Keyed by kernel as well,
//! each combination of variants was a topology of its own, and npcd's ingest
//! met a new one in most waves.
//!
//! **Kept by bytes, not by count** — see [`super::budget`] for the policy and
//! the failure that made it necessary.

use std::collections::HashMap;

use super::budget;
use super::exec::GraphExec;
use super::regime::Key;
use super::slot::ExecSlot;

/// Executables by segment topology, within a fixed device-memory budget.
#[derive(Default)]
pub(super) struct ExecLibrary {
    slots: HashMap<u64, ExecSlot>,
    /// Bytes charged across every kept executable.
    used: u64,
    /// `None` until the card's size is known ([`Self::set_budget`]).
    budget: Option<u64>,
}

impl ExecLibrary {
    /// A library with a budget of exactly `budget` bytes, whatever the card.
    #[cfg(test)]
    pub(super) fn with_budget(budget: u64) -> Self {
        Self {
            budget: Some(budget),
            ..Self::default()
        }
    }

    /// Set the budget ([`budget::graph_budget_bytes`]), unless it already has one.
    pub(super) fn set_budget(&mut self, bytes: u64) {
        self.budget.get_or_insert(bytes);
    }

    /// Whether the budget has been set, by the card or explicitly.
    pub(super) fn budget_known(&self) -> bool {
        self.budget.is_some()
    }

    /// The budget in force — the smallest band's until the card is known.
    pub(super) fn budget(&self) -> u64 {
        self.budget.unwrap_or_else(|| budget::budget_for(0))
    }

    /// The executables of topology `key`, made empty on first sight.
    pub(super) fn slot(&mut self, key: u64) -> &mut ExecSlot {
        self.slots.entry(key).or_default()
    }

    /// The executable topology `key`'s last kept fold put first.
    pub(super) fn front(&self, key: u64) -> Option<&GraphExec> {
        self.slots.get(&key).and_then(ExecSlot::front)
    }

    /// Topologies with executables kept.
    pub(super) fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether topology `key` has executables kept.
    pub(super) fn holds(&self, key: u64) -> bool {
        self.slots.get(&key).is_some_and(|s| !s.execs().is_empty())
    }

    /// Every topology with executables kept — what a trial judges.
    pub(super) fn held_topologies(&self) -> Vec<Key> {
        self.slots
            .iter()
            .filter(|(_, s)| !s.execs().is_empty())
            .map(|(&k, _)| k)
            .collect()
    }

    /// Whether an executable holds exactly `shape` of topology `key` — what a
    /// segment needs to replay without reshaping anything.
    pub(super) fn holds_shape(&self, key: Key, shape: u64) -> bool {
        self.slots.get(&key).is_some_and(|s| s.holds_shape(shape))
    }

    /// Executables kept across every topology, their nodes, and the bytes they
    /// are charged.
    pub(super) fn held(&self) -> (usize, usize, u64) {
        let (execs, nodes) = self
            .slots
            .values()
            .map(ExecSlot::held)
            .fold((0, 0), |(e, n), (se, sn)| (e + se, n + sn));
        (execs, nodes, self.used)
    }

    /// Drop topology `key`'s slot if it keeps nothing — a slot made for a fold
    /// that was not kept.
    pub(super) fn drop_if_empty(&mut self, key: u64) {
        if self.slots.get(&key).is_some_and(|s| s.execs().is_empty()) {
            self.slots.remove(&key);
        }
    }

    /// Topology `key` has just kept a new executable at its front: charge it
    /// against the budget. `false` when it does not fit — it is removed again,
    /// uncharged, and the caller ends the learn; nothing else is evicted to
    /// make room. Call once the executable has been launched: destroying one
    /// in flight is legal, and the driver frees it when its launch completes.
    pub(super) fn admitted(&mut self, key: u64) -> bool {
        let Some(bytes) = self.front(key).map(GraphExec::bytes) else {
            return true;
        };
        if self.used + bytes <= self.budget() {
            self.used += bytes;
            return true;
        }
        if let Some(slot) = self.slots.get_mut(&key) {
            drop(slot.remove_front());
        }
        self.drop_if_empty(key);
        false
    }

    /// Drop every executable of a topology a trial did not prove — banned or
    /// benched — giving their bytes back to the budget. Returns how many there
    /// were.
    pub(super) fn drop_topology(&mut self, key: Key) -> usize {
        let Some(slot) = self.slots.remove(&key) else {
            return 0;
        };
        for e in slot.execs() {
            self.used = self.used.saturating_sub(e.bytes());
        }
        // Destroying an executable still in flight is legal.
        slot.execs().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A topology seen again finds its own slot, however many others came
    /// between.
    #[test]
    fn a_topology_finds_its_slot_again() {
        let mut lib = ExecLibrary::with_budget(1 << 30);
        lib.slot(7);
        lib.slot(9);
        lib.slot(7);
        assert_eq!(lib.len(), 2);
    }

    /// A slot that kept nothing is not a topology the library holds, and is
    /// dropped when asked.
    #[test]
    fn an_empty_slot_holds_nothing_and_is_dropped() {
        let mut lib = ExecLibrary::with_budget(1 << 30);
        lib.slot(3);
        assert!(!lib.holds(3));
        lib.drop_if_empty(3);
        assert_eq!(lib.len(), 0);
        assert_eq!(lib.held(), (0, 0, 0));
    }

    /// The budget is set once; an explicit budget is not reset.
    #[test]
    fn the_budget_is_set_once() {
        let mut lib = ExecLibrary::default();
        assert_eq!(lib.budget(), 256 << 20, "the smallest band until known");
        lib.set_budget(512 << 20);
        assert_eq!(lib.budget(), 512 << 20);
        lib.set_budget(2 << 30);
        assert_eq!(lib.budget(), 512 << 20, "set once");
        let mut fixed = ExecLibrary::with_budget(1);
        fixed.set_budget(2 << 30);
        assert_eq!(fixed.budget(), 1);
    }
}
