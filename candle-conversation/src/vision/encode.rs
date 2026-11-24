//! Images into a turn: each prepared, its rows read from the substrate or made
//! by the vision guest and kept there, and held under virtual ids for the
//! prefill that reads them.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "hub")]
use candle_transformers::models::quantized_qwen38_moe::{vision_tower_pack, VISION_TOWER_MODE};
use candle_transformers::models::qwen_vision::tower::OUT_HIDDEN;
use candle_transformers::models::qwen_vision::{Grid, ImageHandle, ImageRows};

use super::prepare::{prepare, PreparedImage};
use super::store::{load_kept, store_image};
use crate::guest::{GuestOutcome, GuestRequest, VisionRequest};
use crate::persistence::SharedSubstrate;
use crate::ConversationEngine;

/// An image a turn holds: its id and arrival size (the turn text's marker)
/// and its rows, held for the prefill.
#[derive(Clone, Debug)]
pub struct TurnImage {
    pub id: String,
    pub width: u32,
    pub height: u32,
    pub handle: Arc<ImageHandle>,
}

/// The rows of `images`, from the tower.
pub fn encode(
    engine: &ConversationEngine,
    images: &[PreparedImage],
) -> Result<Vec<ImageRows>, String> {
    // Fetched and packed here, on the caller's thread — the first time, a
    // download and a few seconds of host conversion — so the drain only reads
    // the pack.
    #[cfg(feature = "hub")]
    vision_tower_pack(VISION_TOWER_MODE).map_err(|e| format!("the vision tower: {e}"))?;
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
    images
        .iter()
        .zip(out.rows)
        .map(|(i, values)| {
            let rows = ImageRows {
                grid: Grid {
                    height: i.image.height as usize,
                    width: i.image.width as usize,
                },
                width: OUT_HIDDEN,
                values,
            };
            if rows.values.len() != rows.len() * OUT_HIDDEN {
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

/// `sources` as a turn's images, in order. A kept image is read from the
/// substrate as it was kept. A file is decoded and sized; its rows are taken
/// from the substrate when an earlier turn already read it, and the rest are
/// read by the tower in one job and kept. Each is registered for the prefill.
pub fn turn_images(
    engine: &ConversationEngine,
    substrate: &SharedSubstrate,
    sources: Vec<ImageSource>,
) -> Result<Vec<TurnImage>, String> {
    // Each source's image, once known: (id, width, height, rows).
    let mut known: Vec<Option<(String, u32, u32, ImageRows)>> = Vec::with_capacity(sources.len());
    // The files the tower still has to read, by their place in `sources`.
    let mut unread: Vec<(usize, PreparedImage, Vec<u8>)> = Vec::new();
    for (i, source) in sources.into_iter().enumerate() {
        match source {
            ImageSource::Kept(id) => {
                let k =
                    load_kept(substrate, &id)?.ok_or_else(|| format!("no image {id} is kept"))?;
                known.push(Some((id, k.width, k.height, k.rows)));
            }
            ImageSource::Bytes(bytes) => {
                let p = prepare(&bytes)?;
                match load_kept(substrate, &p.id)? {
                    Some(k) => known.push(Some((p.id, p.width, p.height, k.rows))),
                    None => {
                        known.push(None);
                        unread.push((i, p, bytes));
                    }
                }
            }
        }
    }
    if !unread.is_empty() {
        let prepared: Vec<PreparedImage> = unread.iter().map(|(_, p, _)| p.clone()).collect();
        let fresh = encode(engine, &prepared)?;
        let kept_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_millis() as u64);
        for ((i, p, bytes), rows) in unread.into_iter().zip(fresh) {
            store_image(substrate, &p, bytes, &rows, kept_ms)?;
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
