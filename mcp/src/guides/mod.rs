//! The guides this connector serves (`docs/mcp_guides.md`): the library is the
//! `guides` crate's — the Markdown files beside this one, embedded at build
//! time, which Zen's own guide tools read too — and its tools are answered
//! here.
//!
//! | Module | What |
//! |---|---|
//! | [`tools`] | `guide_list` and `guide_read`, the aggregator's own tools |

pub mod tools;

pub use guides::{embedded, Guide, Library};
