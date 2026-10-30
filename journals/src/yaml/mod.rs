//! The YAML the journal files are written in, both ways.
//!
//! Every file is written by this crate's own renderers, so its bytes are
//! exactly what the tests say — field order, quoting, one field to a line,
//! long text folded — and read through `serde_yaml` into a tree whose fields
//! are taken one by one, so a wrong or unknown field is answered as an error
//! naming its file and line (§3.3).

mod doc;
mod fields;
mod scalar;
mod text;

pub(crate) use doc::{item_line, parse_doc};
pub(crate) use fields::Fields;
pub(crate) use scalar::{quote, scalar, scalar_list};
pub(crate) use text::text_field;
