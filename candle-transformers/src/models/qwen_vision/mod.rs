//! Images for a Qwen3-VL-family text trunk: the vision tower that turns an
//! image into rows, the geometry that places those rows, and the registry the
//! engine reads them from.
//!
//! | Module | Concern |
//! |---|---|
//! | `geometry` | an image's grid, patch order, position taps, rotary positions |
//! | `tower` | the ViT and merger, from an mmproj GGUF |
//! | `registry` | rows waiting to be read, by the virtual token ids that name them |
//! | `rotation` | an image row's Q and K turned by its place in the image (a CUDA kernel) |
//! | `source` | where a model's tower comes from, and its pack |
//! | `wave_images` | what a wave does with the virtual ids it was handed |
//!
//! Design: `docs/flash_next_vision.md`; the surface a model plugs into,
//! `docs/vision_models.md`.

pub mod geometry;
pub mod registry;
#[cfg(feature = "cuda")]
pub mod rotation;
pub mod source;
pub mod tower;
#[cfg(feature = "cuda")]
pub mod wave_images;

pub use geometry::{image_delta, smart_resize, trunk_axis, Grid};
pub use registry::{ImageHandle, ImageRows, VIRTUAL_BASE};
#[cfg(feature = "cuda")]
pub use rotation::{rotation_tables, ImageRotation, ImageRun, RotationTables};
pub use source::{VisionTowerSource, VISION_TOWER_MODE};
pub use tower::{VisionInput, VisionTower};
