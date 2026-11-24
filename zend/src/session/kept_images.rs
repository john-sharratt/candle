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
    /// Keep each of `files` as an image `caller` sent: decoded, read by the
    /// tower — every one not already kept, in one job — stored with its rows,
    /// and held by them. In order. `None` before the model is loaded.
    /// Blocking: the tower runs as a guest between waves.
    ///
    /// Images kept with no conversation are swept here too: one that has stood
    /// unused past its grace is retired (`vision::tenancy::sweep_unused`).
    pub fn keep_images(
        &self,
        files: Vec<Vec<u8>>,
        caller: &Caller,
    ) -> Option<Result<Vec<KeptImage>, String>> {
        let state = self.loaded()?;
        let substrate = state.engine.conversation().shared();
        let sources: Vec<ImageSource> = files.into_iter().map(ImageSource::Bytes).collect();
        Some(
            images::claim(&substrate, &sources, caller)
                .and_then(|()| images::resolve(&state.engine, &substrate, sources))
                .and_then(|read| {
                    let retired = sweep_unused(&substrate, now_ms())?;
                    if !retired.is_empty() {
                        tracing::info!(retired = retired.len(), "unused kept images retired");
                    }
                    Ok(read
                        .into_iter()
                        .map(|image| KeptImage {
                            rows: image.handle.len(),
                            id: image.id,
                            width: image.width,
                            height: image.height,
                        })
                        .collect())
                }),
        )
    }

    /// The bytes image `id` arrived as, for `caller`: an image they sent, or
    /// any for an admin. `None` before the model is loaded; `Some(Ok(None))`
    /// for an id the substrate keeps no image under, and for one `caller` did
    /// not send — an id learned is not an image shared. Reads a blob, so a
    /// caller on the async runtime runs it on the blocking pool.
    pub fn kept_image(&self, id: &str, caller: &Caller) -> Option<Result<Option<Vec<u8>>, String>> {
        let state = self.loaded()?;
        let substrate = state.engine.conversation().shared();
        if !caller.is_admin() && !holds(&substrate, id, &caller.key()) {
            return Some(Ok(None));
        }
        Some(load_original(&substrate, id))
    }
}
