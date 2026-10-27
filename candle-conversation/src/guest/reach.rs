//! How large a picture this node's card draws: a table from card size to area.
//!
//! A draw's cost is its pixel count — the image guest claims about 2 KiB of
//! activations a pixel ([`super::image`]) on top of its weights, out of ground
//! the engine frees by pausing and evicting its own working set. Whether that
//! ground exists is therefore a property of the card, not of the moment, and a
//! card too small for an area should say so at once rather than evict the
//! engine and wait out a patience it can never satisfy.
//!
//! So each node serves draws up to the area its card's row allows, and refuses
//! a larger one with [`super::GuestError::BeyondReach`], which names the reach
//! so the caller — or the cluster — can send it to a node that has it.
//!
//! The rows sit between the estate's three cards:
//!
//! | card                  | VRAM  | reach       | basis                                       |
//! |-----------------------|-------|-------------|---------------------------------------------|
//! | RTX 4090 Mobile (AW)  | 16 GB | 512 × 512   | the user's ruling, 2026-10-27               |
//! | RTX 3090 (WhiteWorld) | 24 GB | 1024 × 1024 | measured: a 1024² draw's 9.1 GB fits beside the Makers after a wait |
//! | RTX 5090 (BlackWorld) | 32 GB | 2048 × 2048 | priced, not yet measured: 4.2 Mpx × 2 KiB is 8.6 GB of activations |
//!
//! Reach is an **area**, not a side: 768 × 1344 is about 1 Mpx and is served
//! where 1024 × 1024 is. Cutouts are not limited by it: the matte network runs
//! at its own fixed side whatever the picture, scaling down to it and the alpha
//! back up, so its cost does not follow the picture at all.

use std::sync::atomic::{AtomicU32, Ordering};

/// The reach this process's engine stated, for what it tells its cluster.
static STATED: AtomicU32 = AtomicU32::new(0);

/// Say what this process draws: `side`, the largest square its card serves.
/// Read by the cluster member's `CAPACITY` ([`stated`]), so a gateway places a
/// draw where a card reaches it.
pub fn state(side: u32) {
    STATED.store(side, Ordering::Relaxed);
}

/// The largest square side this process draws, as its engine stated it; zero
/// before an engine has.
pub fn stated() -> u32 {
    STATED.load(Ordering::Relaxed)
}

/// `(smallest card, largest square side drawn on it)`, largest card first.
///
/// The thresholds fall between the estate's cards (16, 24 and 32 GB) so a
/// card's driver-reported total — a few hundred MB under its label — lands in
/// its own row.
const REACH: &[(u64, u32)] = &[(28 << 30, 2048), (20 << 30, 1024), (0, 512)];

/// The largest square side a card of `card_bytes` draws.
pub fn side_for_card(card_bytes: u64) -> u32 {
    REACH
        .iter()
        .find(|(min, _)| card_bytes >= *min)
        .map_or(REACH[REACH.len() - 1].1, |&(_, side)| side)
}

/// The reach a node tells its cluster: its card's `side` when it has an image
/// guest to draw with, and 0 when it has none — a gateway then sends it no
/// draw, which it could only refuse.
pub fn advertised(side: u32, draws: bool) -> u32 {
    if draws {
        side
    } else {
        0
    }
}

/// Whether a `width × height` draw is within a reach of `side × side`.
pub fn within(width: u32, height: u32, side: u32) -> bool {
    width as u64 * height as u64 <= side as u64 * side as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    /// Each of the estate's cards, at the total its driver reports, in its row.
    #[test]
    fn each_card_lands_in_its_own_row() {
        assert_eq!(side_for_card(16 * GIB - 300 * (1 << 20)), 512);
        assert_eq!(side_for_card(24 * GIB - 300 * (1 << 20)), 1024);
        assert_eq!(side_for_card(32 * GIB - 300 * (1 << 20)), 2048);
        assert_eq!(side_for_card(0), 512);
        assert_eq!(side_for_card(96 * GIB), 2048);
    }

    /// **A node with no image guest tells its cluster it draws nothing**,
    /// whatever its card; one with a guest, its card's side.
    #[test]
    fn a_node_without_an_image_guest_advertises_no_reach() {
        assert_eq!(advertised(2048, true), 2048);
        assert_eq!(advertised(2048, false), 0);
    }

    /// The reach is an area: a tall picture of the same pixel count is served
    /// where the square is, and one past it is not.
    #[test]
    fn reach_is_an_area() {
        assert!(within(1024, 1024, 1024));
        assert!(within(768, 1344, 1024));
        assert!(!within(1024, 1040, 1024));
        assert!(within(512, 512, 512));
        assert!(!within(768, 512, 512));
    }
}
