//! The YAML the journal files are written in, both ways.
//!
//! Every file is written by this crate's own renderers, so its bytes are
//! exactly what the tests say — field order, quoting, one line per entry —
//! and read through `serde_yaml` into a tree whose fields are taken one by
//! one, so a wrong or unknown field is answered as an error naming its file
//! and line (§3.3).

mod doc;
mod fields;
mod scalar;

pub(crate) use doc::{item_line, parse_doc, parse_flow_line};
pub(crate) use fields::Fields;
pub(crate) use scalar::{quote, quote_list, scalar, scalar_list};
