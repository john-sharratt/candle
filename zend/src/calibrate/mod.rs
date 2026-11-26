//! Calibrating the knowledge base's files for provenance
//! (`docs/provenance_system_collection.md` §7.1): each file writes the
//! questions its content answers, and those questions, asked behind the empty
//! knowledge base and behind the file alone, stand in the gallery for it
//! (`docs/provenance_scoped_signatures.md` §2.5).
//!
//! A file's own gallery windows are its content. A live probe is a question.
//! Scored against content, the question finds whatever shares its words;
//! against questions the content answers, it finds what answers it — the lever
//! the tool catalog measured at +10.6 points Top-1 when its exemplars became
//! questions (`docs/tool_selection_provenance_results.md` §26.3).
//!
//! Each question is asked again, with thinking on, and what the model reasons
//! opens its calibration answer ([`thinking`]): a live probe is mostly the
//! model's thinking, and the question's thinking is the gallery's counterpart
//! to it. A file's questions alternate between the two frames a live query
//! has ([`kb_frame`]).
//!
//! The files layer opts in with a `calibrate:` block ([`spec`]). Calibration
//! is the tools' resume filter applied to files ([`plan`]): a pass finds every
//! sealed file without a calibration carrying its current marker behind each
//! frame and calibrates it ([`run`]), so a file sealed again — a new version, a
//! changed prompt — is calibrated again.

pub mod ask;
pub mod count;
mod draft_sight;
pub mod failures;
pub mod kb_frame;
pub mod mark;
pub mod owed;
pub mod permits;
pub mod plan;
pub mod profile;
pub mod questions;
pub mod retire_replies;
pub mod run;
pub mod seal;
pub mod sections;
pub mod spec;
pub mod thinking;

pub use mark::mark_sealed;
pub(crate) use run::calibration_meta;
pub use run::{calibrate_layer, Frame, Layer, Report};
pub(crate) use seal::seal_calibration;
pub use spec::CalibrateSpec;
