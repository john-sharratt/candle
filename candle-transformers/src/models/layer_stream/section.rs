//! Opening the layer section of a model pack.
//!
//! Every repack pair the section holds is checked against what this build's
//! repack produces — the same per-pair fingerprint the expert section uses, so
//! a change to one format's repack invalidates the sections of either kind that
//! hold it and no others.

use super::pack::LayerPack;
use crate::models::repack_fingerprint::pair_fingerprint;
use candle::{CudaDevice, Result};
use std::path::Path;

/// Open the layer section at `base` of the model pack `path`.
pub(crate) fn open_layer_section(path: &Path, base: u64, device: &CudaDevice) -> Result<LayerPack> {
    LayerPack::open_section(path, base, |src, dtype| {
        pair_fingerprint(device, src, dtype)
    })
}
