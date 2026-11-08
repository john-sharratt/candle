//! The room (§4.4): the FLX4's output captured ([`capture`]), each block's
//! levels ([`levels`]), and how the room is taking the music ([`landing`]).

pub mod capture;
pub mod landing;
pub mod levels;

pub use landing::Room;
pub use levels::{Analyser, Levels, BLOCK};
