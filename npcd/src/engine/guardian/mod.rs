//! The guardian: a configurable set of modules that watch every character
//! against the mission it carries, and press on the ones that wander.
//!
//! | Module | What it owns |
//! |---|---|
//! | [`view`] | what a module is shown, and what it may conclude |
//! | [`module`] | the trait a check implements |
//! | [`modules`] | the checks: drift, looping, stall, step tracking, journal |
//! | [`ladder`] | how hard to press, and when to stop |
//! | [`nudge`] | what each rung says to the character |
//! | [`config`] | `<data>/guardian.yaml` |
//! | [`builder`] | assembling a guardian from modules or a config |
//! | [`probe`] | the window onto the engine |
//! | [`runner`] | the scan loop |
//! | [`log`] | what it did, for an operator |

pub mod builder;
pub mod config;
pub mod ladder;
pub mod log;
pub mod module;
pub mod modules;
pub mod nudge;
pub mod probe;
pub mod runner;
pub mod view;

pub use builder::GuardianBuilder;
pub use config::GuardianConfig;
pub use log::GuardianLog;
pub use probe::RuntimeProbe;
pub use runner::Guardian;
