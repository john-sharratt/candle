//! A unit's anchor: four base64url characters hashed from its text.
//!
//! ```text
//! anchor(p) = base64url( SHA-256( seed ‖ normalise(text) ) )[0..4]
//! ```
//!
//! `seed` is a 32-bit little-endian integer. A document's anchors are computed
//! together, in document order, each at seed 0; a unit whose anchor an earlier
//! unit already holds is hashed again at seed 1, then 2, until its anchor is
//! free — so two identical paragraphs need no mechanism of their own, and
//! every anchor in a document is unique and four characters long.

use std::collections::HashSet;

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use sha2::{Digest, Sha256};

/// Characters in an anchor: 24 bits of the hash.
pub const ANCHOR_CHARS: usize = 4;

/// `text` as its anchor sees it: each line's trailing whitespace (and a `\r`
/// before its `\n`) removed, and trailing blank lines dropped — so formatting
/// noise and line endings do not change an anchor.
pub fn normalise(text: &str) -> String {
    let lines: Vec<&str> = text.split('\n').map(str::trim_end).collect();
    lines.join("\n").trim_end_matches('\n').to_string()
}

/// The anchor of `text` at `seed`.
pub fn anchor(seed: u32, text: &str) -> String {
    let mut hash = Sha256::new();
    hash.update(seed.to_le_bytes());
    hash.update(normalise(text).as_bytes());
    let digest = hash.finalize();
    URL_SAFE_NO_PAD.encode(&digest[..3])
}

/// The anchors of a document's units, given their texts in document order:
/// each at the lowest seed whose anchor no earlier unit holds.
pub fn assign<'a>(texts: impl IntoIterator<Item = &'a str>) -> Vec<String> {
    let mut taken: HashSet<String> = HashSet::new();
    texts
        .into_iter()
        .map(|text| {
            let mut seed = 0u32;
            loop {
                let a = anchor(seed, text);
                if taken.insert(a.clone()) {
                    return a;
                }
                seed += 1;
            }
        })
        .collect()
}

/// Whether `s` is shaped like an anchor: four base64url characters.
pub fn is_anchor(s: &str) -> bool {
    s.len() == ANCHOR_CHARS
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_anchor_is_four_url_safe_characters() {
        for text in ["You said fourteen.", "", "fn a() {}", "No."] {
            let a = anchor(0, text);
            assert!(is_anchor(&a), "{a:?}");
        }
    }

    /// Fixed values: an anchor is a stored contract with the text it names,
    /// so the hash, the seed's encoding and the alphabet may never drift.
    #[test]
    fn anchors_are_these_exact_strings() {
        assert_eq!(
            [
                anchor(0, "You said fourteen."),
                anchor(1, "You said fourteen."),
                anchor(0, "fn a() {}"),
                anchor(0, ""),
            ],
            ["G2QO", "LyaR", "Gx01", "3z9h"]
        );
    }

    #[test]
    fn formatting_noise_does_not_change_an_anchor() {
        assert_eq!(anchor(0, "a\nb"), anchor(0, "a  \r\nb\t"));
        assert_eq!(anchor(0, "a\nb"), anchor(0, "a\nb\n\n"));
        assert_ne!(anchor(0, "a\nb"), anchor(0, "a\n b"));
        assert_ne!(anchor(0, "a b"), anchor(0, "a  b"));
    }

    #[test]
    fn the_seed_changes_the_anchor() {
        assert_ne!(anchor(0, "No."), anchor(1, "No."));
    }

    /// Duplicates are collisions: the second copy takes seed 1.
    #[test]
    fn identical_units_take_successive_seeds() {
        let got = assign(["No.", "Yes.", "No.", "No."]);
        assert_eq!(got[0], anchor(0, "No."));
        assert_eq!(got[1], anchor(0, "Yes."));
        assert_eq!(got[2], anchor(1, "No."));
        assert_eq!(got[3], anchor(2, "No."));
        let unique: HashSet<&String> = got.iter().collect();
        assert_eq!(unique.len(), 4);
    }

    /// Two different texts that collide at seed 0 are told apart by a bump,
    /// and the bumped anchor is still four characters. A collision is found
    /// by search over a small family of texts.
    #[test]
    fn a_collision_between_different_texts_is_bumped() {
        let mut seen: std::collections::HashMap<String, String> = Default::default();
        let (first, second) = (0..200_000)
            .map(|i| format!("line {i}"))
            .find_map(|t| {
                let a = anchor(0, &t);
                match seen.get(&a) {
                    Some(prev) => Some((prev.clone(), t)),
                    None => {
                        seen.insert(a, t);
                        None
                    }
                }
            })
            .expect("24 bits collide well inside 200k texts");
        let got = assign([first.as_str(), second.as_str()]);
        assert_eq!(got[0], anchor(0, &first));
        assert_eq!(got[1], anchor(1, &second));
        assert_ne!(got[0], got[1]);
        assert!(is_anchor(&got[1]));
    }

    #[test]
    fn is_anchor_checks_the_shape() {
        assert!(is_anchor("Qm7x"));
        assert!(is_anchor("b4_K"));
        assert!(!is_anchor("Qm7"));
        assert!(!is_anchor("Qm7x9"));
        assert!(!is_anchor("Qm+x"));
    }
}
