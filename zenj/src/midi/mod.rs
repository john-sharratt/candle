//! MIDI (§6.1): the bytes on the wire ([`wire`]), a controller's map of
//! messages to controls ([`map`]), and the ports — the FLX4 observed, the
//! virtual controller driven ([`ports`]).

pub mod map;
pub mod ports;
pub mod wire;

pub use map::{Binding, Map, Reading};
pub use ports::{Out, Touch};
pub use wire::{Decoder, Message};
