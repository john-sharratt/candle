//! Images kept in the substrate ahead of the turn that names them, and read
//! back by their id.
//!
//! A kept image is read by the tower once, when it is kept; a turn that names
//! its id finds its rows and runs no tower ([`crate::images`]).

use serde::Serialize;

use super::*;

/// An image the substrate keeps: its id, its size as it arrived, and how many
/// rows a turn reads it as.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct KeptImage {
    pub id: String,
    pub width: u32,
    pub height: u32,
    pub rows: usize,
}

impl ZendSession {
    /// Keep each of `files` as an image: decoded, read by the tower — every
    /// one not already kept, in one job — and stored with its rows. In order.
    /// `None` before the model is loaded. Blocking: the tower runs as a guest
    /// between waves.
    pub fn keep_images(&self, files: Vec<Vec<u8>>) -> Option<Result<Vec<KeptImage>, String>> {
        let state = self.loaded()?;
        let substrate = state.engine.conversation().shared();
        let sources = files.into_iter().map(ImageSource::Bytes).collect();
        Some(
            images::resolve(&state.engine, &substrate, sources).map(|read| {
                read.into_iter()
                    .map(|image| KeptImage {
                        rows: image.handle.len(),
                        id: image.id,
                        width: image.width,
                        height: image.height,
                    })
                    .collect()
            }),
        )
    }

    /// The bytes image `id` arrived as. `None` before the model is loaded;
    /// `Some(Ok(None))` for an id the substrate keeps no image under. Reads a
    /// blob, so a caller on the async runtime runs it on the blocking pool.
    pub fn kept_image(&self, id: &str) -> Option<Result<Option<Vec<u8>>, String>> {
        let state = self.loaded()?;
        let substrate = state.engine.conversation().shared();
        Some(load_original(&substrate, id))
    }
}
