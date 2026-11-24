//! An image in the substrate: its original bytes, its thumbnail and its rows,
//! each a content-addressed blob, and a keyed record naming all three by the
//! image's id — `vision/image/<id>` — so whatever holds a turn's text finds its
//! rows, and whatever draws it finds its thumbnail.
//!
//! Blobs replicate with the substrate, so a member that adopts the turn finds
//! the image too; the rows mean no path that rebuilds a turn has to run the
//! tower again.
//!
//! The rows are a tower's output, so the record names the tower that wrote
//! them (`docs/vision_models.md` §3): a turn on a model with another tower
//! reads the image again and keeps its own rows in their place.

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

pub(super) fn unhex(text: &str) -> Option<[u8; 32]> {
    if text.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(&text[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(out)
}

/// Keep `image`'s original bytes and the rows `tower` read it as, and name
/// them by its id, kept at `kept_ms` (Unix milliseconds) — the age an image no
/// conversation uses is retired by (`vision::tenancy`). Rows another tower
/// kept for the image before are retired once these replace them.
pub fn store_image(
    substrate: &SharedSubstrate,
    image: &PreparedImage,
    original: Vec<u8>,
    rows: &ImageRows,
    tower: &str,
    kept_ms: u64,
) -> Result<(), String> {
    let replaced = substrate
        .custom_object(&image_key(&image.id))
        .and_then(|obj| obj.blob["rows"].as_str().and_then(unhex));
    let original = substrate.put_blob(original).map_err(|e| e.to_string())?;
    let thumb = substrate
        .put_blob(image.thumb.clone())
        .map_err(|e| e.to_string())?;
    let rows = substrate
        .put_blob(encode_rows(rows))
        .map_err(|e| e.to_string())?;
    substrate
        .put_custom_object(CustomObjectPayload::new(
            image_key(&image.id),
            json!({
                "sha256": hex(&original),
                "thumb": hex(&thumb),
                "rows": hex(&rows),
                "tower": tower,
                "width": image.width,
                "height": image.height,
                "kept_ms": kept_ms,
            }),
        ))
        .map_err(|e| e.to_string())?;
    match replaced {
        Some(old) if old != rows => substrate.retire_blob(&old).map_err(|e| e.to_string()),
        _ => Ok(()),
    }
}

/// The blob the record of image `id` names under `field`, when the substrate
/// holds the image.
fn load_named(
    substrate: &SharedSubstrate,
    id: &str,
    field: &str,
) -> Result<Option<Vec<u8>>, String> {
    let Some(obj) = substrate.custom_object(&image_key(id)) else {
        return Ok(None);
    };
    let Some(sha) = obj.blob[field].as_str().and_then(unhex) else {
        return Err(format!("{}: no {field} named", image_key(id)));
    };
    substrate.blob(&sha).map_err(|e| e.to_string())
}

/// The original bytes of the image `id`, when the substrate holds it.
pub fn load_original(substrate: &SharedSubstrate, id: &str) -> Result<Option<Vec<u8>>, String> {
    load_named(substrate, id, "sha256")
}

/// The thumbnail of the image `id`, a JPEG, when the substrate holds it.
pub fn load_thumb(substrate: &SharedSubstrate, id: &str) -> Result<Option<Vec<u8>>, String> {
    load_named(substrate, id, "thumb")
}

/// The rows of the image `id`, whichever tower kept them, when the substrate
/// holds them — for a path replaying a stored turn, whose engine refuses rows
/// of another width.
pub fn load_rows(substrate: &SharedSubstrate, id: &str) -> Result<Option<ImageRows>, String> {
    let Some(obj) = substrate.custom_object(&image_key(id)) else {
        return Ok(None);
    };
    Ok(kept_from(substrate, id, &obj)?.map(|k| k.rows))
}

/// An image the substrate keeps: the size it arrived at, and its rows.
pub struct KeptRows {
    pub width: u32,
    pub height: u32,
    pub rows: ImageRows,
}

/// The image `id` as the substrate keeps it for `tower`: `None` when it keeps
/// no such image, or keeps it read by another tower.
pub fn load_kept(
    substrate: &SharedSubstrate,
    id: &str,
    tower: &str,
) -> Result<Option<KeptRows>, String> {
    let Some(obj) = substrate.custom_object(&image_key(id)) else {
        return Ok(None);
    };
    if obj.blob["tower"].as_str() != Some(tower) {
        return Ok(None);
    }
    kept_from(substrate, id, &obj)
}

/// What the record `obj` of image `id` names.
fn kept_from(
    substrate: &SharedSubstrate,
    id: &str,
    obj: &CustomObjectPayload,
) -> Result<Option<KeptRows>, String> {
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
    use crate::guest::VisionImage;
    use crate::persistence::Admit;
    use candle_transformers::models::qwen_vision::Grid;

    fn image() -> PreparedImage {
        PreparedImage {
            id: "00000000000000ab".into(),
            sha256: String::new(),
            width: 32,
            height: 32,
            image: VisionImage {
                rgb: Vec::new(),
                height: 32,
                width: 32,
            },
            thumb: b"thumb".to_vec(),
        }
    }

    fn rows(value: f32) -> ImageRows {
        ImageRows {
            grid: Grid {
                height: 32,
                width: 32,
            },
            width: 2,
            values: vec![value, value],
        }
    }

    /// **Rows are the tower's that read them**: a turn on another tower finds
    /// none, and its own replace them, the old rows retired; a replay reads
    /// whichever are kept.
    #[test]
    fn rows_are_kept_for_the_tower_that_read_them() {
        let dir = tempfile::tempdir().unwrap();
        let s = SharedSubstrate::open_in(dir.path(), Admit::Unbound).unwrap();
        let id = "00000000000000ab";
        store_image(&s, &image(), b"cat".to_vec(), &rows(1.0), "tower-a", 0).unwrap();
        assert!(load_kept(&s, id, "tower-a").unwrap().is_some());
        assert!(load_kept(&s, id, "tower-b").unwrap().is_none());
        let first = unhex(
            s.custom_object(&image_key(id)).unwrap().blob["rows"]
                .as_str()
                .unwrap(),
        )
        .unwrap();

        store_image(&s, &image(), b"cat".to_vec(), &rows(2.0), "tower-b", 0).unwrap();
        assert!(load_kept(&s, id, "tower-a").unwrap().is_none());
        let kept = load_kept(&s, id, "tower-b").unwrap().unwrap();
        assert_eq!(kept.rows.values, vec![2.0, 2.0]);
        assert_eq!(load_rows(&s, id).unwrap().unwrap().values, vec![2.0, 2.0]);
        assert!(
            !s.has_blob(&first),
            "the other tower's rows were not retired"
        );
        assert_eq!(load_original(&s, id).unwrap(), Some(b"cat".to_vec()));
    }

    #[test]
    fn digests_round_trip_through_hex() {
        let d: [u8; 32] = std::array::from_fn(|i| (i * 7) as u8);
        assert_eq!(unhex(&hex(&d)), Some(d));
        assert_eq!(unhex("00"), None);
        assert_eq!(&hex(&d)[..4], "0007");
    }
}
