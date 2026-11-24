//! Images into a turn: each prepared, its rows read from the substrate or made
//! by the vision guest and kept there, and held under virtual ids for the
//! prefill that reads them.

use std::sync::Arc;

#[cfg(feature = "hub")]
use candle_transformers::models::quantized_qwen38_moe::vision_tower_file;
use candle_transformers::models::qwen_vision::tower::OUT_HIDDEN;
use candle_transformers::models::qwen_vision::{Grid, ImageHandle, ImageRows};

use super::prepare::{prepare, PreparedImage};
use super::store::{load_rows, store_image};
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
pub fn encode(engine: &ConversationEngine, images: &[PreparedImage]) -> Result<Vec<ImageRows>, String> {
    // Fetched here, on the caller's thread: the drain only reads the cache.
    #[cfg(feature = "hub")]
    vision_tower_file().map_err(|e| format!("the vision tower: {e}"))?;
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

/// `uploads` as a turn's images: each decoded and sized, its rows taken from
/// the substrate when an earlier turn already read it and made by the tower
/// otherwise (and kept), then registered for the prefill.
pub fn turn_images(
    engine: &ConversationEngine,
    substrate: &SharedSubstrate,
    uploads: Vec<Vec<u8>>,
) -> Result<Vec<TurnImage>, String> {
    let prepared: Vec<PreparedImage> = uploads
        .iter()
        .map(|b| prepare(b))
        .collect::<Result<_, _>>()?;
    let mut rows: Vec<Option<ImageRows>> = prepared
        .iter()
        .map(|p| load_rows(substrate, &p.id))
        .collect::<Result<_, _>>()?;
    let missing: Vec<usize> = (0..prepared.len()).filter(|&i| rows[i].is_none()).collect();
    if !missing.is_empty() {
        let fresh = encode(
            engine,
            &missing.iter().map(|&i| prepared[i].clone()).collect::<Vec<_>>(),
        )?;
        for (&i, r) in missing.iter().zip(fresh) {
            store_image(substrate, &prepared[i], uploads[i].clone(), &r)?;
            rows[i] = Some(r);
        }
    }
    prepared
        .into_iter()
        .zip(rows)
        .map(|(p, r)| {
            let handle = ImageHandle::register(r.expect("filled above"))?;
            Ok(TurnImage {
                id: p.id,
                width: p.width,
                height: p.height,
                handle,
            })
        })
        .collect()
}
