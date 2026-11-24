//! An image in the substrate: its original bytes, its thumbnail and the rows
//! each tower read it as, each a content-addressed blob, and a keyed record
//! naming them all by the image's id — `vision/image/<id>` — so whatever holds
//! a turn's text finds its rows, and whatever draws it finds its thumbnail.
//!
//! Blobs replicate with the substrate, so a member that adopts the turn finds
//! the image too; the rows mean no path that rebuilds a turn has to run the
//! tower again. Every blob is named by the image through `blob_refs`, so one
//! another image shares goes only with the last of them.
//!
//! The rows are a tower's output, so the record keeps them by tower
//! (`docs/vision_models.md` §3): a turn on a model with another tower reads
//! the image again and keeps its rows beside the first tower's.

use candle_transformers::models::qwen_vision::ImageRows;
use serde_json::{json, Map, Value};

use super::blob_refs::{hex, put_named, release};
use super::prepare::PreparedImage;
use super::rows_codec::{decode_rows, encode_rows};
use crate::persistence::record::CustomObjectPayload;
use crate::persistence::SharedSubstrate;

/// The record that names an image's blobs.
pub fn image_key(id: &str) -> String {
    format!("vision/image/{id}")
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

/// The digest the record `blob` names its `tower`'s rows by.
fn rows_of(blob: &Value, tower: &str) -> Option<[u8; 32]> {
    blob["rows"][tower].as_str().and_then(unhex)
}

/// Every blob the record `blob` names: its original, its thumbnail, and each
/// tower's rows.
pub(super) fn blobs_of(blob: &Value) -> Vec<[u8; 32]> {
    let mut out: Vec<[u8; 32]> = ["sha256", "thumb"]
        .iter()
        .filter_map(|f| blob[*f].as_str().and_then(unhex))
        .collect();
    if let Some(rows) = blob["rows"].as_object() {
        out.extend(rows.values().filter_map(|v| v.as_str().and_then(unhex)));
    }
    out
}

/// Keep `image`'s original bytes and the rows `tower` read it as, and name
/// them by its id, kept at `kept_ms` (Unix milliseconds) — the age an image no
/// conversation uses is retired by (`vision::tenancy`). Rows other towers
/// kept stay; rows this tower kept before are released once these replace
/// them.
pub fn store_image(
    substrate: &SharedSubstrate,
    image: &PreparedImage,
    original: Vec<u8>,
    rows: &ImageRows,
    tower: &str,
    kept_ms: u64,
) -> Result<(), String> {
    let id = &image.id;
    let before = substrate.custom_object(&image_key(id));
    let replaced = before.as_ref().and_then(|obj| rows_of(&obj.blob, tower));
    let mut towers = before
        .as_ref()
        .and_then(|obj| obj.blob["rows"].as_object().cloned())
        .unwrap_or_else(Map::new);
    let original = put_named(substrate, original, id)?;
    let thumb = put_named(substrate, image.thumb.clone(), id)?;
    let fresh = put_named(substrate, encode_rows(rows), id)?;
    towers.insert(tower.to_string(), Value::String(hex(&fresh)));
    substrate
        .put_custom_object(CustomObjectPayload::new(
            image_key(id),
            json!({
                "sha256": hex(&original),
                "thumb": hex(&thumb),
                "rows": towers,
                "width": image.width,
                "height": image.height,
                "kept_ms": kept_ms,
            }),
        ))
        .map_err(|e| e.to_string())?;
    match replaced {
        Some(old) if old != fresh => release(substrate, &old, id),
        _ => Ok(()),
    }
}

/// Set the time image `id` was kept to `kept_ms`, so an image sent again is
/// as new as its sending. Nothing when the substrate keeps no such image.
pub fn touch(substrate: &SharedSubstrate, id: &str, kept_ms: u64) -> Result<(), String> {
    let Some(obj) = substrate.custom_object(&image_key(id)) else {
        return Ok(());
    };
    let mut blob = obj.blob;
    if blob["kept_ms"].as_u64() == Some(kept_ms) {
        return Ok(());
    }
    blob["kept_ms"] = json!(kept_ms);
    substrate
        .put_custom_object(CustomObjectPayload::new(image_key(id), blob))
        .map_err(|e| e.to_string())
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

/// The rows `tower` read image `id` as, when the substrate keeps them — for a
/// path replaying a stored turn.
pub fn load_rows(
    substrate: &SharedSubstrate,
    id: &str,
    tower: &str,
) -> Result<Option<ImageRows>, String> {
    Ok(load_kept(substrate, id, tower)?.map(|k| k.rows))
}

/// An image the substrate keeps: the size it arrived at, and its rows.
pub struct KeptRows {
    pub width: u32,
    pub height: u32,
    pub rows: ImageRows,
}

/// The image `id` as the substrate keeps it for `tower`: `None` when it keeps
/// no such image, or none of `tower`'s rows for it.
pub fn load_kept(
    substrate: &SharedSubstrate,
    id: &str,
    tower: &str,
) -> Result<Option<KeptRows>, String> {
    let Some(obj) = substrate.custom_object(&image_key(id)) else {
        return Ok(None);
    };
    let Some(sha) = rows_of(&obj.blob, tower) else {
        return Ok(None);
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

    /// **Each tower's rows are its own**: a second tower's sit beside the
    /// first's, and a tower reading the image again replaces only its own,
    /// the old rows retired.
    #[test]
    fn rows_are_kept_for_the_tower_that_read_them() {
        let dir = tempfile::tempdir().unwrap();
        let s = SharedSubstrate::open_in(dir.path(), Admit::Unbound).unwrap();
        let id = "00000000000000ab";
        store_image(&s, &image(), b"cat".to_vec(), &rows(1.0), "tower-a", 0).unwrap();
        assert!(load_kept(&s, id, "tower-a").unwrap().is_some());
        assert!(load_kept(&s, id, "tower-b").unwrap().is_none());

        store_image(&s, &image(), b"cat".to_vec(), &rows(2.0), "tower-b", 0).unwrap();
        let rows_a = load_rows(&s, id, "tower-a").unwrap().unwrap();
        assert_eq!(rows_a.values, vec![1.0, 1.0]);
        assert_eq!(
            load_rows(&s, id, "tower-b").unwrap().unwrap().values,
            vec![2.0, 2.0]
        );
        let first = rows_of(&s.custom_object(&image_key(id)).unwrap().blob, "tower-a").unwrap();

        store_image(&s, &image(), b"cat".to_vec(), &rows(3.0), "tower-a", 0).unwrap();
        assert_eq!(
            load_rows(&s, id, "tower-a").unwrap().unwrap().values,
            vec![3.0, 3.0]
        );
        assert!(!s.has_blob(&first), "the replaced rows were not retired");
        assert_eq!(load_original(&s, id).unwrap(), Some(b"cat".to_vec()));
    }

    /// An image sent again is as new as its sending; touching one not kept
    /// writes nothing.
    #[test]
    fn touching_an_image_renews_its_keeping() {
        let dir = tempfile::tempdir().unwrap();
        let s = SharedSubstrate::open_in(dir.path(), Admit::Unbound).unwrap();
        let id = "00000000000000ab";
        store_image(&s, &image(), b"cat".to_vec(), &rows(1.0), "tower-a", 5).unwrap();
        touch(&s, id, 9).unwrap();
        let obj = s.custom_object(&image_key(id)).unwrap();
        assert_eq!(obj.blob["kept_ms"].as_u64(), Some(9));
        assert_eq!(blobs_of(&obj.blob).len(), 3);
        touch(&s, "00000000000000ff", 9).unwrap();
        assert!(s.custom_object(&image_key("00000000000000ff")).is_none());
    }

    #[test]
    fn digests_round_trip_through_hex() {
        let d: [u8; 32] = std::array::from_fn(|i| (i * 7) as u8);
        assert_eq!(unhex(&hex(&d)), Some(d));
        assert_eq!(unhex("00"), None);
        assert_eq!(&hex(&d)[..4], "0007");
    }
}
