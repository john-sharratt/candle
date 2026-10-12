//! The executable library's memory policy: what an executable costs, which
//! ones make room for a new one, and which keys may not come back yet.
//!
//! Every instantiated executable is device memory the driver hands out
//! directly — outside the reservation, invisible to the partition, and the
//! only thing left to take when the card is full. Kept by count (512
//! topologies of up to 16 executables), npcd's library grew to 1,098
//! executables of 312,417 nodes in seven minutes and the next instantiation
//! failed with `CUDA_ERROR_OUT_OF_MEMORY` in the middle of a forward. So the
//! library is kept by bytes:
//!
//! - **A budget**, sized by the card ([`budget_for`]). An admission that takes
//!   the library past it evicts until it fits.
//! - **Least-invoked eviction.** Each launch credits its executable one
//!   invocation; the fewest go first. Counts decay by [`INVOCATION_DECAY`] each
//!   wave, so a graph that was hot an hour ago is not kept for that alone, and
//!   a newcomer is not doomed by starting from nothing against counts that
//!   never shrink.
//! - **A cooldown.** An evicted key may not be admitted again for
//!   [`COOLDOWN_WAVES`] waves. Without it, a key evicted to make room returns
//!   the next wave, evicts the next-least-invoked, and the library spends its
//!   time instantiating — the thrash the cooldown exists to stop.
//!
//! Pure arithmetic over costs, scores and wave numbers, so every rule is
//! tested without a device.

use std::collections::HashMap;

/// Device memory the library may hold across all its executables, by how much
/// the card has.
///
/// The driver allocates an executable's memory itself — no API places it — so
/// it cannot come from the reservation's span. It is planned instead: the span
/// is sized to leave exactly this much of the card outside it
/// (`region_pool::span_target`), and the library is held to it.
///
/// | card | budget |
/// |---|---|
/// | under 20 GiB (the 16 GB laptop) | 512 MiB |
/// | under 32 GiB (the 24 GB RTX 3090) | 768 MiB |
/// | up to 64 GiB | 1 GiB |
/// | over 64 GiB (the 72 GB RTX PRO 5000) | 2 GiB |
pub fn budget_for(device_total: u64) -> u64 {
    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;
    match device_total {
        t if t < 20 * GIB => 512 * MIB,
        t if t < 32 * GIB => 768 * MIB,
        t if t <= 64 * GIB => GIB,
        _ => 2 * GIB,
    }
}

/// The graph budget of the card the calling thread's context is on: the one
/// figure the reservation leaves outside its span for executables and the
/// library holds them to, so the two cannot disagree.
pub fn graph_budget_bytes() -> crate::Result<u64> {
    use crate::cuda_backend::WrapErr;
    let (_, total) = cudarc::driver::result::mem_get_info().w()?;
    Ok(budget_for(total as u64))
}

/// Waves an evicted key waits before it may be admitted again.
pub(crate) const COOLDOWN_WAVES: u64 = 256;

/// What each executable's invocation count keeps from one wave to the next —
/// a half-life of about 70 waves.
pub(crate) const INVOCATION_DECAY: f64 = 0.99;

/// The least an executable costs, whatever its size — measured on an RTX 3090
/// (`graph::tests::device_bytes_per_executable`, one node: 98,304 B).
const EXEC_MIN_BYTES: u64 = 96 * 1024;

/// Device bytes per kernel node — the same probe, 100 and 400 nodes: 9,503 B.
const EXEC_BYTES_PER_NODE: u64 = 9_503;

/// What an executable of `nodes` nodes is expected to cost.
pub(crate) fn estimate_bytes(nodes: usize) -> u64 {
    (nodes as u64 * EXEC_BYTES_PER_NODE).max(EXEC_MIN_BYTES)
}

/// What to charge an executable of `nodes` nodes, given the driver's free
/// memory `measured` to have fallen by that much across its instantiation.
///
/// The reading is trusted only near the estimate — between an eighth of it
/// and twice it. Another thread's allocation in the same moment inflates it,
/// a free deflates it, and a card with nothing left reads zero before and
/// after; any of those is answered with the estimate instead.
pub(crate) fn charge(measured: Option<u64>, nodes: usize) -> u64 {
    let estimate = estimate_bytes(nodes);
    match measured {
        Some(m) if m >= estimate / 8 && m <= estimate * 2 => m,
        _ => estimate,
    }
}

/// One executable as eviction sees it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Candidate<K> {
    pub(crate) id: K,
    /// Its decayed invocation count.
    pub(crate) invocations: f64,
    pub(crate) bytes: u64,
}

/// Which of `candidates` to evict so that `used` bytes fit `budget`: the least
/// invoked first, ties to the larger (more room per eviction). Empty when the
/// library already fits; every candidate when even that is not enough.
pub(crate) fn victims<K: Copy>(candidates: &[Candidate<K>], used: u64, budget: u64) -> Vec<K> {
    if used <= budget {
        return Vec::new();
    }
    let mut order: Vec<&Candidate<K>> = candidates.iter().collect();
    order.sort_by(|a, b| {
        a.invocations
            .total_cmp(&b.invocations)
            .then(b.bytes.cmp(&a.bytes))
    });
    let mut over = used - budget;
    let mut out = Vec::new();
    for c in order {
        if over == 0 {
            break;
        }
        out.push(c.id);
        over = over.saturating_sub(c.bytes);
    }
    out
}

/// Keys evicted recently, and the wave each may be admitted again.
#[derive(Debug, Default)]
pub(crate) struct Cooldown<K> {
    until: HashMap<K, u64>,
}

impl<K: std::hash::Hash + Eq + Copy> Cooldown<K> {
    /// `key` was evicted in wave `now`.
    pub(crate) fn cool(&mut self, key: K, now: u64) {
        self.until.insert(key, now + COOLDOWN_WAVES);
    }

    /// Whether `key` may not be admitted in wave `now`.
    pub(crate) fn cooling(&self, key: &K, now: u64) -> bool {
        self.until.get(key).is_some_and(|&until| now < until)
    }

    /// Forget every key whose cooldown is over by wave `now`.
    pub(crate) fn prune(&mut self, now: u64) {
        self.until.retain(|_, &mut until| now < until);
    }

    /// Keys cooling.
    pub(crate) fn len(&self) -> usize {
        self.until.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(id: u32, invocations: f64, bytes: u64) -> Candidate<u32> {
        Candidate {
            id,
            invocations,
            bytes,
        }
    }

    /// The budget follows the card: the three dev machines' cards as the
    /// driver reports them, and the band edges.
    #[test]
    fn the_budget_follows_the_card() {
        const GIB: u64 = 1 << 30;
        const MIB: u64 = 1 << 20;
        // RTX 4090 Mobile, RTX 3090, RTX PRO 5000 Blackwell.
        assert_eq!(budget_for(17_175_150_592), 512 * MIB);
        assert_eq!(budget_for(25_769_279_488), 768 * MIB);
        assert_eq!(budget_for(76_000_000_000), 2 * GIB);
        assert_eq!(budget_for(20 * GIB - 1), 512 * MIB);
        assert_eq!(budget_for(20 * GIB), 768 * MIB);
        assert_eq!(budget_for(32 * GIB), GIB);
        assert_eq!(budget_for(64 * GIB), GIB);
        assert_eq!(budget_for(64 * GIB + 1), 2 * GIB);
    }

    /// A library within its budget evicts nothing.
    #[test]
    fn a_library_within_budget_keeps_everything() {
        assert!(victims(&[c(1, 0.0, 100)], 1_000, 1_000).is_empty());
    }

    /// The least invoked go first, and only as many as the overrun needs.
    #[test]
    fn the_least_invoked_make_room_and_no_more() {
        let lib = [
            c(1, 9.0, 400),
            c(2, 0.5, 300),
            c(3, 2.0, 300),
            c(4, 0.1, 100),
        ];
        // 1,100 held against 700: 400 to free — 4 (100) then 2 (300).
        assert_eq!(victims(&lib, 1_100, 700), vec![4, 2]);
        // One byte over: the single least invoked.
        assert_eq!(victims(&lib, 701, 700), vec![4]);
    }

    /// Among equally invoked executables, the larger goes first.
    #[test]
    fn a_tie_evicts_the_larger() {
        let lib = [c(1, 1.0, 100), c(2, 1.0, 500)];
        assert_eq!(victims(&lib, 600, 500), vec![2]);
    }

    /// Too far over to fit: everything offered is evicted.
    #[test]
    fn an_overrun_past_every_candidate_evicts_them_all() {
        let lib = [c(1, 1.0, 100), c(2, 2.0, 100)];
        assert_eq!(victims(&lib, 10_000, 500), vec![1, 2]);
    }

    /// The estimate has its floor and its per-node rate.
    #[test]
    fn the_estimate_follows_the_probe() {
        assert_eq!(estimate_bytes(1), 98_304);
        assert_eq!(estimate_bytes(100), 950_300);
        assert_eq!(estimate_bytes(400), 3_801_200);
    }

    /// A believable reading is charged as measured; an implausible one, or none,
    /// is charged the estimate.
    #[test]
    fn a_reading_is_trusted_only_near_the_estimate() {
        // 400 nodes: estimate 3,801,200; trusted between 475,150 and 7,602,400.
        assert_eq!(charge(Some(1_500_000), 400), 1_500_000);
        assert_eq!(charge(Some(0), 400), 3_801_200);
        assert_eq!(charge(Some(400_000), 400), 3_801_200);
        assert_eq!(charge(Some(9_000_000), 400), 3_801_200);
        assert_eq!(charge(None, 400), 3_801_200);
    }

    /// An evicted key cools for exactly `COOLDOWN_WAVES` waves, then is free —
    /// and pruning forgets it only then.
    #[test]
    fn a_cooled_key_returns_after_its_cooldown() {
        let mut cd = Cooldown::default();
        cd.cool(7u64, 100);
        assert!(cd.cooling(&7, 100));
        assert!(cd.cooling(&7, 100 + COOLDOWN_WAVES - 1));
        assert!(!cd.cooling(&7, 100 + COOLDOWN_WAVES));
        assert!(!cd.cooling(&8, 100));
        cd.prune(100 + COOLDOWN_WAVES - 1);
        assert_eq!(cd.len(), 1);
        cd.prune(100 + COOLDOWN_WAVES);
        assert_eq!(cd.len(), 0);
    }
}
