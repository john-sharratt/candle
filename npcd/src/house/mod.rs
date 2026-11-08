//! The house's mode (`docs/house_modes.md`): what the mirrored home is set up
//! for — away, asleep, at home, listening, watching, a party — set by one act,
//! `house_mode`, which a body that takes no orders (Zen) makes from the room
//! the house's devices stand in.
//!
//! | Module | |
//! |---|---|
//! | [`mode`] | the modes, and which of them opens the decks |
//! | [`table`] | `house/<world>.yaml`: the room, and each mode's device steps |
//! | [`acts`] | the act |
//! | [`switch`] | a switch carried out: the decks over the bus, each device through its bridge |

pub mod acts;
pub mod mode;
pub mod switch;
pub mod table;

pub use acts::{is_mine, HOUSE_ACTS};
pub use mode::Mode;
pub use table::Table;
