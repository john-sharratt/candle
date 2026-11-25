//! Images into a turn: each prepared, its rows read from the substrate or made
//! by the engine's vision tower and kept there, and held under virtual ids for
//! the prefill that reads them.
//!
//! Which tower is the engine's model's (`ConversationEngine::vision`,
//! `docs/vision_models.md` §2.2); a model that names none takes no image.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "hub")]
use candle_transformers::models::qwen_vision::VISION_TOWER_MODE;
use candle_transformers::models::qwen_vision::{Grid, ImageHandle, ImageRows, VisionTowerSource};

use super::prepare::{image_id, prepare, PreparedImage};
use super::store::{load_kept, load_original, store_image};
use crate::guest::{GuestOutcome, GuestRequest, VisionRequest};
use crate::persistence::SharedSubstrate;
use crate::ConversationEngine;

/// The engine's tower, or why its model takes no image.
fn tower_of(engine: &ConversationEngine) -> Result<&'static VisionTowerSource, String> {
    engine
        .vision()
        .ok_or_else(|| "this model reads no images: its spec names no vision tower".to_string())
}

/// An image a turn holds: its id and arrival size (the turn text's marker)
/// and its rows, held for the prefill.
#[derive(Clone, Debug)]
pub struct TurnImage {
    pub id: String,
    pub width: u32,
    pub height: u32,
    pub handle: Arc<ImageHandle>,
}

/// The rows of `images`, from the engine's tower.
pub fn encode(
    engine: &ConversationEngine,
    images: &[PreparedImage],
) -> Result<Vec<ImageRows>, String> {
    // The check stands in every build; only a build with the hub fetches the pack.
    #[cfg_attr(not(feature = "hub"), allow(unused_variables))]
    let tower = tower_of(engine)?;
    // Fetched and packed here, on the caller's thread — the first time, a
    // download and a few seconds of host conversion — so the drain only reads
    // the pack.
    #[cfg(feature = "hub")]
    tower
        .pack(VISION_TOWER_MODE)
        .map_err(|e| format!("the {} vision tower: {e}", tower.name))?;
    let request = GuestRequest::Vision(VisionRequest {
        images: images.iter().map(|i| i.image.clone()).collect(),
    });
    let receipt = engine.submit_guest(request).map_err(|e| e.to_string())?;
    let GuestOutcome::Vision(out) = receipt.wait().map_err(|e| e.to_string())? else {
        return Err("the vision guest answered with something else".into());
    };
    if out.rows.len() != images.len() {
        return Err(format!(
            "the vision guest answered {} images for {}",
            out.rows.len(),
            images.len()
        ));
    }
    let width = out.width;
    images
        .iter()
        .zip(out.rows)
        .map(|(i, values)| {
            let rows = ImageRows {
                grid: Grid {
                    height: i.image.height as usize,
                    width: i.image.width as usize,
                },
                width,
                values,
            };
            if rows.values.len() != rows.len() * width {
                return Err(format!(
                    "the tower returned {} values for {} rows",
                    rows.values.len(),
                    rows.len()
                ));
            }
            Ok(rows)
        })
        .collect()
}

/// An image a turn is given: the file itself, or the id of one the substrate
/// keeps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageSource {
    Bytes(Vec<u8>),
    Kept(String),
}

impl ImageSource {
    /// The id the image is kept under: a file's is its digest's, read
    /// without decoding it.
    pub fn id(&self) -> String {
        match self {
            ImageSource::Bytes(bytes) => image_id(bytes),
            ImageSource::Kept(id) => id.clone(),
        }
    }
}

/// `sources` as a turn's images, in order. An image whose rows the substrate
/// keeps for the engine's tower is read back as it was kept, its file never
/// decoded. Any other — a file, or a kept image only another tower read — is
/// decoded and sized, read by the tower in one job, and kept. Each is
/// registered for the prefill.
pub fn turn_images(
    engine: &ConversationEngine,
    substrate: &SharedSubstrate,
    sources: Vec<ImageSource>,
) -> Result<Vec<TurnImage>, String> {
    if sources.is_empty() {
        return Ok(Vec::new());
    }
    let tower = tower_of(engine)?.name;
    // Each source's image, once known: (id, width, height, rows).
    let mut known: Vec<Option<(String, u32, u32, ImageRows)>> = Vec::with_capacity(sources.len());
    // The files the tower still has to read, by their place in `sources`.
    let mut unread: Vec<(usize, PreparedImage, Vec<u8>)> = Vec::new();
    for (i, source) in sources.into_iter().enumerate() {
        let id = source.id();
        // Kept rows are the whole answer: the file is decoded only when the
        // tower has to read it.
        if let Some(k) = load_kept(substrate, &id, tower)? {
            known.push(Some((id, k.width, k.height, k.rows)));
            continue;
        }
        let bytes = match source {
            ImageSource::Bytes(bytes) => bytes,
            ImageSource::Kept(id) => {
                load_original(substrate, &id)?.ok_or_else(|| format!("no image {id} is kept"))?
            }
        };
        known.push(None);
        unread.push((i, prepare(&bytes)?, bytes));
    }
    if !unread.is_empty() {
        let prepared: Vec<PreparedImage> = unread.iter().map(|(_, p, _)| p.clone()).collect();
        let fresh = encode(engine, &prepared)?;
        let kept_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_millis() as u64);
        for ((i, p, bytes), rows) in unread.into_iter().zip(fresh) {
            store_image(substrate, &p, bytes, &rows, tower, kept_ms)?;
            known[i] = Some((p.id, p.width, p.height, rows));
        }
    }
    known
        .into_iter()
        .map(|k| {
            let (id, width, height, rows) =
                k.ok_or_else(|| "an image was neither kept nor read".to_string())?;
            Ok(TurnImage {
                id,
                width,
                height,
                handle: ImageHandle::register(rows)?,
            })
        })
        .collect()
}
