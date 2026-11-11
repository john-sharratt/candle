//! A character's agenda (`docs/zen_moon_mind.md` §17): the missions a character
//! that takes no orders makes for itself, several at once by priority — the
//! acts that keep it ([`acts`]), and what they do ([`perform`]). The missions
//! themselves are kept with everybody else's, in `sim::missions`. The same
//! acts as a character that takes orders has them, changed through a
//! conversation, are [`ordered`].

pub mod acts;
pub mod ordered;
mod perform;

pub use acts::{is_mine, AGENDA_ACTS};
pub use perform::perform;

#[cfg(test)]
mod tests;
