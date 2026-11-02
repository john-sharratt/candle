//! A character's agenda (`docs/zen_moon_mind.md` §17): the missions a character
//! that takes no orders makes for itself, several at once by priority — the
//! acts that keep it ([`acts`]), and what they do ([`perform`]). The missions
//! themselves are kept with everybody else's, in `sim::missions`.

pub mod acts;
mod perform;

pub use acts::{is_mine, AGENDA_ACTS};
pub use perform::perform;

#[cfg(test)]
mod tests;
