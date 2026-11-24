//! Images in a turn, for a model with a vision tower.
//!
//! | Module | Concern |
//! |---|---|
//! | `prepare` | an upload decoded, sized to its patch grid, and named by its digest |
//! | `rows_codec` | an image's rows as bytes, in bf16 |
//! | `store` | an image's original and rows in the substrate, named by its id |
//! | `marker` | how a turn's text and ids say where its images are |
//! | `spans` | where a prompt's images are, and its ids as the engine reads them |
//! | `stored` | the image spans of stored turns, for a path that prefills them again |
//! | `encode` | uploads into a turn's images: prepared, read by the tower, registered |
//! | `tenancy` | who may read a kept image, and how long it is kept |
//! | `thumb` | an image's thumbnail, made once and kept beside it |
//!
//! Design: `docs/flash_next_vision.md`.

pub mod encode;
pub mod marker;
pub mod prepare;
pub mod rows_codec;
pub mod spans;
pub mod store;
pub mod stored;
pub mod tenancy;
pub mod thumb;

pub use encode::{encode, turn_images, ImageSource, TurnImage};
pub use marker::{marker, split_markers};
pub use prepare::{prepare, PreparedImage};
pub use spans::{engine_ids, turn_spans, ImageSpan, VisionTokens};
pub use store::{load_kept, load_original, load_rows, load_thumb, store_image, KeptRows};
pub use stored::stored_spans;
