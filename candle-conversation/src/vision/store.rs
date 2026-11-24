//! An image in the substrate: its original bytes and its rows, each a
//! content-addressed blob, and a keyed record naming both by the image's id —
//! `vision/image/<id>` — so whatever holds a turn's text finds its rows.
//!
//! Blobs replicate with the substrate, so a member that adopts the turn finds
//! the image too; the rows mean no path that rebuilds a turn has to run the
//! tower again.

use candle_transformers::models::qwen_vision::ImageRows;
use serde_json::json;

use super::prepare::PreparedImage;
use super::rows_codec::{decode_rows, encode_rows};
use crate::persistence::record::CustomObjectPayload;
use crate::persistence::SharedSubstrate;

/// The record that names an image's blobs.
pub fn image_key(id: &str) -> String {
    format!("vision/image/{id}")
}

fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(text: &str) -> Option<[u8; 32]> {
    if text.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(&text[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(out)
}

/// Keep `image`'s original bytes and `rows`, and name them by its id.
pub fn store_image(
    substrate: &SharedSubstrate,
    image: &PreparedImage,
    original: Vec<u8>,
    rows: &ImageRows,
) -> Result<(), String> {
    let original = substrate.put_blob(original).map_err(|e| e.to_string())?;
    let rows = substrate
        .put_blob(encode_rows(rows))
        .map_err(|e| e.to_string())?;
    substrate
        .put_custom_object(CustomObjectPayload::new(
            image_key(&image.id),
            json!({
                "sha256": hex(&original),
                "rows": hex(&rows),
                "width": image.width,
                "height": image.height,
            }),
        ))
        .map_err(|e| e.to_string())
}

/// The original bytes of the image `id`, when the substrate holds it.
pub fn load_original(substrate: &SharedSubstrate, id: &str) -> Result<Option<Vec<u8>>, String> {
    let Some(obj) = substrate.custom_object(&image_key(id)) else {
        return Ok(None);
    };
    let Some(sha) = obj.blob["sha256"].as_str().and_then(unhex) else {
        return Err(format!("{}: no original named", image_key(id)));
    };
    substrate.blob(&sha).map_err(|e| e.to_string())
}

/// The rows of the image `id`, when the substrate holds them.
pub fn load_rows(substrate: &SharedSubstrate, id: &str) -> Result<Option<ImageRows>, String> {
    Ok(load_kept(substrate, id)?.map(|k| k.rows))
}

/// An image the substrate keeps: the size it arrived at, and its rows.
pub struct KeptRows {
    pub width: u32,
    pub height: u32,
    pub rows: ImageRows,
}

/// The image `id` as the substrate keeps it, when it does.
pub fn load_kept(substrate: &SharedSubstrate, id: &str) -> Result<Option<KeptRows>, String> {
    let Some(obj) = substrate.custom_object(&image_key(id)) else {
        return Ok(None);
    };
    let Some(sha) = obj.blob["rows"].as_str().and_then(unhex) else {
        return Err(format!("{}: no rows named", image_key(id)));
    };
    let side = |k: &str| {
        obj.blob[k]
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| format!("{}: no {k}", image_key(id)))
    };
    let (width, height) = (side("width")?, side("height")?);
    match substrate.blob(&sha).map_err(|e| e.to_string())? {
        Some(bytes) => Ok(Some(KeptRows {
            width,
            height,
            rows: decode_rows(&bytes)?,
        })),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digests_round_trip_through_hex() {
        let d: [u8; 32] = std::array::from_fn(|i| (i * 7) as u8);
        assert_eq!(unhex(&hex(&d)), Some(d));
        assert_eq!(unhex("00"), None);
        assert_eq!(&hex(&d)[..4], "0007");
    }
}
