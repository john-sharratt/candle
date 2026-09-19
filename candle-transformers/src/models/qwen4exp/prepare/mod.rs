//! The prepared engine artifact: its recipe and identity ([`recipe`]), the
//! sources it is built from ([`store`]), the expert requantizer ([`requant`]),
//! and the resolve-or-build flow ([`build`]).

#[cfg(feature = "cuda")]
pub mod build;
pub mod recipe;
#[cfg(feature = "cuda")]
pub mod requant;
pub mod store;

#[cfg(feature = "cuda")]
pub use build::{prepare_engine, prepared};
pub use recipe::{ExpertSource, Recipe, SourceFile, SourceRole};
pub use store::SourceStore;
