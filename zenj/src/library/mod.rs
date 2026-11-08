//! The library (§4.1, §4.3): Spotify tracks matched to TIDAL ([`matching`]),
//! how much each person likes each ([`taste`]), its energy ([`energy`]), the
//! crates it is filed in ([`crates`]), the kept library ([`store`]), and the
//! passes that build it ahead of the night ([`sync`]).

pub mod crates;
pub mod energy;
pub mod matching;
pub mod store;
pub mod sync;
pub mod taste;

pub use store::Library;
