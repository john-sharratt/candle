//! The layers built on other layers — a layer of file units on each file's
//! `code_reading` (the bundled schema's `dependency_analysis`), or of day units
//! on each closed day of dialogue — each declared in the projection schema by
//! an `ingest:` block ([`spec`]) and read by one engine.
//!
//! A unit of such a layer is one hidden conversation, forked from its layer's
//! base with the unit it builds on as its lineage: a file's
//! `dependency_analysis` continues the file's own reading, and a layer declared
//! `after` it would continue that — each layer's question asked of a
//! conversation that has already read the file and every analysis of it below. **The layers
//! compound**: what a unit retrieves is also the layers below it, of every
//! file on its branches, by provenance (`Substrate::set_ingest_sees`), while
//! its own layer stays self-local — a unit never retrieves its siblings.
//!
//! **Lower layers first, and what is stale goes at once** ([`plan`]): a pass
//! walks the layers bottom up, a unit stands only while the unit below it
//! stands and the files it depends on are where it found them, and a stale
//! one is tombstoned as soon as it is seen — the layer briefly has nothing
//! for the file rather than an analysis of another version. A pass asked for
//! while a layer above reads stops it between units, so the layers below are
//! read first ([`crate::ingest_worker::preempted`]).

pub mod days;
pub mod edges;
pub mod keys;
pub mod pass;
pub mod plan;
pub mod run;
pub mod spec;

pub use pass::{pass_layer, sweep_layer, Ground, LayerRun, Reach};
pub use spec::{LayerSpec, UnitKind};
