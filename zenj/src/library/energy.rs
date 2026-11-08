//! A track's energy, `0..=1`, from what its analysis says (§4.3): the mood
//! rekordbox's phrase analysis gives it, how much of it is chorus and build,
//! and its tempo. The room's response refines it live (`room`); this is the
//! prior it starts from.

use crate::music::Phrase;

/// The energy of a track of `bpm`, in phrase mood `mood` (1 high, 2 mid, 3
/// low), with `phrases`.
pub fn estimate(bpm: f32, mood: Option<u16>, phrases: &[Phrase]) -> f32 {
    let base = match mood {
        Some(1) => 0.7,
        Some(2) => 0.55,
        Some(3) => 0.4,
        _ => 0.5,
    };
    let total: u32 = phrases.iter().map(Phrase::beats).sum();
    let shape = if total == 0 {
        0.0
    } else {
        let weighted: f32 = phrases
            .iter()
            .map(|p| p.kind.weight() * p.beats() as f32)
            .sum();
        weighted / total as f32 - 0.6
    };
    let tempo = ((bpm - 120.0) / 100.0).clamp(-0.15, 0.15);
    (base + 0.4 * shape + tempo).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::music::track::fixture::club;

    #[test]
    fn energy_rises_with_mood_shape_and_tempo() {
        let club = club("c", 128.0, "8A", 0.0).phrases;
        let high = estimate(128.0, Some(1), &club);
        let low = estimate(128.0, Some(3), &club);
        assert!(high > low);
        assert!(estimate(140.0, Some(2), &club) > estimate(100.0, Some(2), &club));
        // The club shape: (0.4·32 + 0.7·32 + 1·64 + 0.35·32 + 1·64 + 0.4·32) / 256.
        let shape = (0.4 * 32.0 + 0.7 * 32.0 + 64.0 + 0.35 * 32.0 + 64.0 + 0.4 * 32.0) / 256.0 - 0.6;
        assert!((estimate(120.0, Some(2), &club) - (0.55 + 0.4 * shape)).abs() < 1e-6);
        assert_eq!(estimate(120.0, None, &[]), 0.5);
    }
}
