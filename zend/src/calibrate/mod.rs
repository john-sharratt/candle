//! Calibrating a layer's conversations for provenance: each unit — a file's
//! reading, an analysis of it, a day — writes the questions its content
//! answers, and those questions, prefilled alone, stand in the gallery for it
//! (`docs/provenance_scoped_signatures.md` §2.5).
//!
//! A unit's own gallery windows are its content: the request that read it,
//! the tool calls, the file, its summary. A live probe is a question. Scored
//! against content, the question finds whatever shares its words; against
//! questions the content answers, it finds what answers it — the lever the
//! tool catalog measured at +10.6 points Top-1 when its exemplars became
//! questions (`docs/tool_selection_provenance_results.md` §26.3).
//!
//! Each question is asked again alone, with thinking on, and what the model
//! reasons opens its calibration answer ([`thinking`]): a live probe is
//! mostly the model's thinking, and the question's thinking is the gallery's
//! counterpart to it.
//!
//! Any layer opts in with a `calibrate:` block ([`spec`]). Calibration is the
//! tools' resume filter applied to units ([`plan`]): a pass finds every ready
//! unit without a calibration carrying its current marker and calibrates it
//! ([`run`]), so units read before the layer declared the block
//! are calibrated without being read again, and a unit read again — a new
//! version, a changed prompt — is calibrated again.

pub mod permits;
pub mod plan;
pub mod questions;
pub mod run;
pub mod spec;
pub mod thinking;

pub use run::{calibrate_committed, calibrate_layer, Frame, Layer, Report};
pub use spec::CalibrateSpec;
