//! Guidance calibrated from use (`docs/guidance_layer.md` §2).
//!
//! The guides and the journals are `code_reading` units that a drawing group
//! places apart from the code. A drafted calibration asks a unit for the
//! questions it answers, which suits code; a guide is wanted at a point in a
//! worker's own reasoning, which rarely reads as a question about the guide.
//!
//! So every read of a drawn unit — a `guide_read`, or a `file_read` of a file
//! the group draws — is a labelled exemplar: the reasoning the turn wrote
//! before the call is the state in which that unit was wanted. It is sealed as
//! one more calibration of the unit ([`record`]), so its signature votes for
//! the unit wherever the unit is scored, from the next turn of every
//! conversation. A unit keeps its newest [`record::MAX_EXEMPLARS`].

pub mod exemplar;
pub mod reads;
pub mod record;

pub use exemplar::use_case;
pub use reads::{read_path, unit_of, Read};
pub use record::{record_use, USE_KEY};
