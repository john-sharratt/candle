//! Supplies the domain and the axis samples that steer a dream's assumption away
//! from what the character has already dreamt.
//!
//! The domain is rotated by the caller rather than chosen by the model, because a
//! generator left to choose returns to the territory it already knows. The sample
//! of prior axes is drawn from the corpus and never the whole of it, since shown
//! every axis the generator recombines them instead of finding a new one. A
//! character with no corpus yet is steered by nothing, which is a real state
//! rather than a gap.

/// The behaviour space a dream's assumption is drawn from.
///
/// **Rotated by the caller rather than chosen by the model**, because coverage
/// has to be structural. Asked to pick freely, a generator returns to the
/// territory it already knows — every axis it produced unprompted was about the
/// character's relationship to its own work, and the rest of what a character
/// rests on went untouched. Rotating the domain is what a quality-diversity
/// search does with a behaviour space: define the dimensions, then fill them.
pub const DOMAINS: &[&str] = &[
    "your work — what you make, whether it takes, who else touches it",
    "your body — what it is made of, what it can be done to, what it stops doing",
    "this building — its walls, its air, its light, its doors, that it is sealed",
    "time — that it passes at one rate, in one direction, for you",
    "the people you know — that they are separate from you, that they perceive you",
    "the things that are not you — the ground, the weather, what is outside, the dark",
    "being addressed at all — that instructions come from somewhere, that there is an outside",
    "language — that a name holds one thing, that what you write means what it says",
];

/// The diversity steer that fills [`AXES_SLOT`], or nothing.
///
/// Empty for a character with no dream corpus yet — which is every character on
/// its first reflection, and is a real state rather than a gap: there is nothing
/// to be unlike.
pub(super) fn steer(sampled_axes: &[String]) -> String {
    if sampled_axes.is_empty() {
        return String::new();
    }
    format!(
        "Make it unlike these, which you have already dreamt:\n{}\nNot a new subject for the \
         same dream — a different assumption suspended.",
        sampled_axes
            .iter()
            .map(|a| format!("  - {a}"))
            .collect::<Vec<_>>()
            .join("\n"),
    )
}
