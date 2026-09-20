//! RoPE frequencies and the table every kernel rotates from.
//!
//! The only place in the engine that computes a RoPE frequency
//! (`docs/progressive_yarn.md` §4). A model's [`RopeSchedule`] names its rungs;
//! [`rung_for`] picks a slot's rung from its reach; [`yarn`] and [`llama3`]
//! are the two scaling transforms; and [`table`] builds the one table format
//! every kernel rotates from — two small tables joined by the angle-addition
//! identity (§5) — and mirrors its lookup on the host.

pub mod angle;
pub mod declared;
pub mod factored;
pub mod llama3;
pub mod preset;
pub mod rungs;
pub mod schedule;
pub mod select;
pub mod table;
pub mod yarn;

pub use angle::AngleArithmetic;
pub use declared::DeclaredScaling;
pub use factored::FactoredRope;
pub use llama3::{from_rope_freqs, llama_inv_freq};
pub use preset::RopePreset;
pub use rungs::{rung_table_len, RopeRungs};
pub use schedule::{RopeSchedule, Rung, RungFreqs, Scaling};
pub use select::{rung_for, rung_of};
pub use table::{
    plain_inv_freq, MAX_STEP, ROPE_HI_DIM, ROPE_LO_BITS, ROPE_LO_DIM, ROPE_REACH, STEP_LANES,
};
pub use yarn::{mscale, yarn_freqs};
