//! Cooperative admission for a zend cluster (`docs/cooperative_admission.md`):
//! what each member's forwards cost, measured from its own scheduler wave log,
//! and a simulator of the cluster that compares placement policies on those
//! costs before any of them is built into the gateway.
//!
//! | Module | |
//! |---|---|
//! | [`sample`] | one channel of one wave window |
//! | [`wave_log`] | the scheduler's `wave` line, read back into samples |
//! | [`experts`] | the expert bytes a forward of `t` tokens reads |
//! | [`cost`] | a member's forward cost, term by term |
//! | [`fit`] | that cost fitted to a member's samples |
//! | [`curve`] | measured step time by batch size |

pub mod cost;
pub mod curve;
pub mod experts;
pub mod fit;
pub mod sample;
pub mod wave_log;
