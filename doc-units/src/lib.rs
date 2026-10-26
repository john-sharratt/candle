//! A file as the units an author edits, each named by a content anchor.
//!
//! `docs/npcd_anchored_edits.md` is the design. In brief:
//!
//! - [`carve`] cuts a file into units — paragraphs of prose, lines of YAML,
//!   semantic scopes of code — that partition its non-blank lines.
//! - [`anchor`] names each unit with four base64url characters hashed from
//!   its own text, a seed making every anchor in a document unique.
//! - [`Doc`] is a file carved and anchored; [`render`] shows a page of it the
//!   way a read returns it.
//! - [`edit`] applies one anchored operation — replace, insert, delete, move —
//!   and reports the units it touched under their new anchors.
//!
//! Nothing here is stored: anchors are computed from the text every time it is
//! read, so the same text always reads with the same anchors and a unit
//! somebody else changed simply no longer has the anchor it was read under.

pub mod anchor;
pub mod carve;
mod doc;
pub mod edit;
mod language;
pub mod render;

pub use doc::{Doc, Unit};
pub use edit::{apply, Applied, Edit, EditError, Op, Place};
pub use language::Language;
