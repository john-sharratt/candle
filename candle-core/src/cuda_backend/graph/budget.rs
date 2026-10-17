//! The executable library's memory: how much it may hold, and what each
//! executable is charged against that.
//!
//! Every instantiated executable is device memory the driver hands out
//! directly — outside the reservation, invisible to the partition, and the
//! only thing left to take when the card is full. Kept by count (512
//! topologies of up to 16 executables), npcd's library grew to 1,098
//! executables of 312,417 nodes in seven minutes and the next instantiation
//! failed with `CUDA_ERROR_OUT_OF_MEMORY` in the middle of a forward. So the
//! library is kept by bytes, within a **fixed budget** sized by the card
//! ([`budget_for`]). Nothing is evicted to make room: an admission that does
//! not fit ends the library's learn (`regime`), and only a trial's bans give
//! ground back.
//!
//! Pure arithmetic over costs, so every rule is tested without a device.

/// Device memory the library may hold across all its executables, by how much
/// the card has.
///
/// The driver allocates an executable's memory itself — no API places it — so
/// it cannot come from the reservation's span. It is planned instead: the span
/// is sized to leave exactly this much of the card outside it
/// (`region_pool::span_target`), and the library is held to it. Every byte of
/// it is a byte the weight zone and the KV side never get, so the small cards
/// are held tightest: on the 16 GB laptop the experts stream through what is
/// left, and a learn that fills a smaller budget only locks sooner (`regime`).
///
/// | card | budget |
/// |---|---|
/// | under 20 GiB (the 16 GB laptop) | 256 MiB |
/// | under 28 GiB (the 24 GB RTX 3090) | 512 MiB |
/// | up to 64 GiB (the 32 GB RTX 5090, 48 GB cards) | 1 GiB |
/// | over 64 GiB (the 72 GB RTX PRO 5000) | 2 GiB |
///
/// **Each edge sits between card sizes, never at one.** A card reports a
/// little less than its nominal size — the 32 GB RTX 5090 reports about
/// 31.8 GiB — so an edge at a nominal size puts that card in the band below
/// it. The edges are mid-gap ([`EDGES`]), and the bands' test holds every card
/// the fleet runs at least `EDGE_MARGIN` (2 GiB) from all of them.
pub fn budget_for(device_total: u64) -> u64 {
    const MIB: u64 = 1 << 20;
    match device_total {
        t if t < EDGES[0] => 256 * MIB,
        t if t < EDGES[1] => 512 * MIB,
        t if t <= EDGES[2] => GIB,
        _ => 2 * GIB,
    }
}

const GIB: u64 = 1 << 30;

/// Where one budget band ends and the next begins, by reported card total:
/// between 16 and 24 GB cards, between 24 and 32 GB, between 48 and 72 GB.
const EDGES: [u64; 3] = [20 * GIB, 28 * GIB, 64 * GIB];

/// The least distance between a card the fleet runs and a band edge.
#[cfg(test)]
const EDGE_MARGIN: u64 = 2 * GIB;

/// The graph budget of the card the calling thread's context is on: the one
/// figure the reservation leaves outside its span for executables and the
/// library holds them to, so the two cannot disagree.
pub fn graph_budget_bytes() -> crate::Result<u64> {
    use crate::cuda_backend::WrapErr;
    let (_, total) = cudarc::driver::result::mem_get_info().w()?;
    Ok(budget_for(total as u64))
}

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

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    /// What the driver reports for each card the fleet runs, and the budget it
    /// gets: RTX 4090 Mobile, RTX 3090, RTX 5090 (the production workstation),
    /// a 48 GB card, RTX PRO 5000 Blackwell.
    const CARDS: [(u64, u64); 5] = [
        (17_175_150_592, 256 * MIB),
        (25_769_279_488, 512 * MIB),
        (34_190_917_632, GIB),
        (51_527_024_640, GIB),
        (76_000_000_000, 2 * GIB),
    ];

    /// The budget follows the card, and the edges fall where they say.
    #[test]
    fn the_budget_follows_the_card() {
        for (total, budget) in CARDS {
            assert_eq!(budget_for(total), budget, "card of {total} B");
        }
        assert_eq!(budget_for(EDGES[0] - 1), 256 * MIB);
        assert_eq!(budget_for(EDGES[0]), 512 * MIB);
        assert_eq!(budget_for(EDGES[1] - 1), 512 * MIB);
        assert_eq!(budget_for(EDGES[1]), GIB);
        assert_eq!(budget_for(EDGES[2]), GIB);
        assert_eq!(budget_for(EDGES[2] + 1), 2 * GIB);
    }

    /// **No card sits near an edge**: a card a few hundred MiB short of its
    /// nominal size, or a driver that reserves a little more, keeps its band.
    #[test]
    fn every_card_is_clear_of_every_edge() {
        for (total, budget) in CARDS {
            for edge in EDGES {
                assert!(
                    total.abs_diff(edge) >= EDGE_MARGIN,
                    "card of {total} B is within {} MiB of the edge at {} GiB",
                    total.abs_diff(edge) >> 20,
                    edge / GIB
                );
            }
            assert_eq!(budget_for(total - EDGE_MARGIN / 2), budget);
            assert_eq!(budget_for(total + EDGE_MARGIN / 2), budget);
        }
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
}
