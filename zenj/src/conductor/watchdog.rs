//! The watchdog (§7.3), checked every bar: the deck on air has more than 32
//! bars left, or the next track is loaded and its plan armed. If neither,
//! Automix takes over and the screens say why. A blind reader is the same
//! fault: ZenJ cannot mix what it cannot see.

use crate::deck::Snapshot;
use crate::music::Grid;

/// Bars left on air below which the next must be ready.
pub const BARS_LEFT: f64 = 32.0;

/// What the conductor knows of the night when the watchdog looks.
pub struct Watched<'a> {
    pub snapshot: &'a Snapshot,
    /// The playing track's grid and length, when the reader names it.
    pub on_air: Option<(Grid, u64)>,
    pub next_loaded: bool,
    pub armed: bool,
}

/// Why Automix must take over, if it must.
pub fn fault(w: &Watched<'_>) -> Option<String> {
    if let Some(why) = &w.snapshot.blind {
        return Some(format!("the deck reader is blind: {why}"));
    }
    let deck = w.snapshot.on_air()?;
    let (grid, length_ms) = w.on_air?;
    let left_ms = length_ms as f64 - deck.position_ms;
    let bars_left = left_ms / grid.ms_per_beat() / 4.0 * (deck.bpm as f64 / grid.bpm as f64);
    if bars_left > BARS_LEFT || (w.next_loaded && w.armed) {
        return None;
    }
    Some(format!(
        "deck {} has {:.0} bars left and the next is {}",
        deck.number,
        bars_left.max(0.0),
        if w.next_loaded { "loaded but not armed" } else { "not loaded" }
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deck::Deck;

    fn playing(position_ms: f64) -> Snapshot {
        Snapshot {
            decks: vec![Deck {
                number: 1,
                track: None,
                bpm: 120.0,
                position_ms,
                beat: None,
                playing: true,
            }],
            master: Some(1),
            blind: None,
        }
    }

    const GRID: Grid = Grid {
        bpm: 120.0,
        first_beat_ms: 0.0,
    };

    /// **Past 32 bars left nothing is wrong; inside them the next must be
    /// loaded and armed.** At 120 BPM a bar is 2 s.
    #[test]
    fn inside_32_bars_the_next_must_be_ready() {
        let far = playing(100_000.0);
        let w = |s, loaded, armed| Watched { snapshot: s, on_air: Some((GRID, 200_000)), next_loaded: loaded, armed };
        assert_eq!(fault(&w(&far, false, false)), None, "50 bars left");
        let near = playing(150_000.0);
        assert_eq!(
            fault(&w(&near, false, false)).as_deref(),
            Some("deck 1 has 25 bars left and the next is not loaded")
        );
        assert_eq!(
            fault(&w(&near, true, false)).as_deref(),
            Some("deck 1 has 25 bars left and the next is loaded but not armed")
        );
        assert_eq!(fault(&w(&near, true, true)), None);
        let blind = Snapshot::blind("rekordbox is not running");
        assert_eq!(
            fault(&w(&blind, true, true)).as_deref(),
            Some("the deck reader is blind: rekordbox is not running")
        );
    }
}
