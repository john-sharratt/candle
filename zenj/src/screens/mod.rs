//! The screens (§8): which scene each shows and when it changes
//! ([`scene`]), what is done ahead of a change so it lands on the phrase
//! ([`ahead`]), ZenJ's layer over it ([`overlay`]), and the windows that draw
//! both ([`window`]).

pub mod ahead;
pub mod overlay;
pub mod scene;
pub mod window;

pub use window::{Screens, Shared};
