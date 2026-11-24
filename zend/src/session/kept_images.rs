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
    /// Keep `bytes` as an image: decoded, read by the tower, and stored with
    /// its rows. `None` before the model is loaded. Blocking: the tower runs
    /// as a guest between waves.
    pub fn keep_image(&self, bytes: Vec<u8>) -> Option<Result<KeptImage, String>> {
        let state = self.loaded()?;
        let substrate = state.engine.conversation().shared();
        Some(
            images::resolve(&state.engine, &substrate, vec![ImageRef::Bytes(bytes)]).map(
                |mut read| {
                    let image = read.remove(0);
                    KeptImage {
                        rows: image.handle.len(),
                        id: image.id,
                        width: image.width,
                        height: image.height,
                    }
                },
            ),
        )
    }

    /// The bytes image `id` arrived as. `None` before the model is loaded;
    /// `Some(Ok(None))` for an id the substrate keeps no image under.
    pub fn kept_image(&self, id: &str) -> Option<Result<Option<Vec<u8>>, String>> {
        let state = self.loaded()?;
        let substrate = state.engine.conversation().shared();
        Some(load_original(&substrate, id))
    }
}
