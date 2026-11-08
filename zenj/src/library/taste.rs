//! How much a person likes a track, from what Spotify recorded of them
//! (`spotify::export::Heard`): listens on a log scale, a like, being in a
//! playlist of theirs, less for skips — folded into `0..=1`.

use crate::spotify::export::Heard;

/// Listens past which more say nothing new.
const LISTENS_FULL: f32 = 50.0;

pub fn weight(h: &Heard) -> f32 {
    let listens = ((1.0 + h.listens as f32).ln() / (1.0 + LISTENS_FULL).ln()).min(1.0);
    let plays = h.listens + h.skips;
    let skips = if plays == 0 {
        0.0
    } else {
        h.skips as f32 / plays as f32
    };
    let w = 0.55 * listens
        + 0.3 * h.liked as u8 as f32
        + 0.15 * (!h.playlists.is_empty()) as u8 as f32
        - 0.25 * skips;
    w.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn heard(listens: u32, skips: u32, liked: bool, playlists: usize) -> Heard {
        Heard {
            listens,
            skips,
            liked,
            playlists: vec!["p".into(); playlists],
            ..Heard::default()
        }
    }

    #[test]
    fn a_weight_rises_with_listens_and_likes_and_falls_with_skips() {
        assert_eq!(weight(&heard(0, 0, false, 0)), 0.0);
        assert!((weight(&heard(50, 0, true, 1)) - 1.0).abs() < 1e-6);
        assert!((weight(&heard(0, 0, true, 0)) - 0.3).abs() < 1e-6);
        assert!(weight(&heard(10, 0, false, 0)) > weight(&heard(3, 0, false, 0)));
        assert!(weight(&heard(10, 10, false, 0)) < weight(&heard(10, 0, false, 0)));
        assert_eq!(weight(&heard(0, 20, false, 0)), 0.0, "held at zero");
    }
}
