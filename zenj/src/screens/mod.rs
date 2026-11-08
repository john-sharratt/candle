//! The screens (§8): which scene each shows and when it changes
//! ([`scene`]), ZenJ's layer over it ([`overlay`]), and the windows that draw
//! both ([`window`]).

pub mod overlay;
pub mod scene;
pub mod window;

pub use window::{Screens, Shared};
