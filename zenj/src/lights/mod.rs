//! The lights (§9): the WiZ bulbs on the ceiling track, driven on the beat
//! while the house is in Party — rekordbox open — and left to the house's own
//! table the rest of the time (`docs/house_modes.md`). They never touch the
//! audio: nothing on the audio path reads a light.
//!
//! | Module | |
//! |---|---|
//! | [`fixtures`] | `zenj/lights.yaml`: every bulb, its address, place and aim |
//! | [`cue`] | a lighting move of a transition plan |
//! | [`palette`] | the colours and brightness the energy asks for |
//! | [`scene`] | the named scenes, each drawn whole from the beat |
//! | [`director`] | what the room looks like at a beat, by the rules in order |
//! | [`timeline`] | two bars planned ahead on the beat clock, each send at its instant |
//! | [`gate`] | the rate budget: a token bucket per bulb, coalescing |
//! | [`health`] | which bulbs answer, and how long they take |
//! | [`lead`] | how early each bulb is sent its state |
//! | [`wire`] | the one non-blocking UDP socket |
//! | [`run`] | the planner and the sender |

pub mod cue;
pub mod director;
pub mod fixtures;
pub mod gate;
pub mod health;
pub mod lead;
pub mod palette;
pub mod run;
pub mod scene;
pub mod timeline;
pub mod wire;

pub use cue::{Cue, Cues};
pub use director::Ask;
pub use fixtures::Fixtures;
pub use run::{Lights, View};
pub use scene::Scene;
