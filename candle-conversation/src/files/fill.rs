//! What of a kind's sections a projection places
//! (`docs/provenance_system_collection.md` §4, `docs/placed_section_layout.md`
//! §3.5c).
//!
//! The kind's sections go strongest first under the kind's share of the room:
//! each that fits in what is left is placed, and one that does not is passed
//! over for the next. A section stands alone — it carries its own heading and
//! closing line — so a large file no longer places as one block: each of its
//! sections competes on its own score.

use super::key::SectionKey;

/// The sections of `ranked` — strongest first, each sized by `tokens_of` —
/// placed under `cap` tokens, in rank order. A section `tokens_of` does not
/// know is not placed.
pub fn choose(
    ranked: &[(SectionKey, f32)],
    tokens_of: impl Fn(SectionKey) -> Option<usize>,
    cap: usize,
) -> Vec<SectionKey> {
    let mut left = cap;
    let mut out = Vec::new();
    for &(section, _) in ranked {
        let Some(t) = tokens_of(section) else {
            continue;
        };
        if t > left {
            continue;
        }
        left -= t;
        out.push(section);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Strongest first, each section that fits in what is left, one that
    /// does not passed over for the next**: under 1,000 tokens, a (300) and b
    /// (400) fit; c (500) does not fit the 300 left and is passed over; d
    /// (250) fits.
    #[test]
    fn sections_are_placed_strongest_first_each_that_fits() {
        let (a, b, c, d) = (SectionKey(1), SectionKey(2), SectionKey(3), SectionKey(4));
        let tokens = |s: SectionKey| match s.0 {
            1 => Some(300),
            2 => Some(400),
            3 => Some(500),
            4 => Some(250),
            _ => None,
        };
        let ranked = [
            (a, 900.0),
            (b, 800.0),
            (c, 700.0),
            (d, 600.0),
            (SectionKey(9), 500.0),
        ];
        assert_eq!(choose(&ranked, tokens, 1_000), vec![a, b, d]);
        assert!(choose(&ranked, tokens, 200).is_empty());
    }
}
