//! Who may read a kept image, and for how long it is kept.
//!
//! An image is the file a person sent. Its bytes (`GET /v1/images/{id}`) and
//! its rows, named by id in a turn, are read by a person who sent it
//! ([`holds`]) or by an admin, never by anyone who merely learned its id.
//!
//! An image lives as long as a timeline uses it ([`note_use`]) — a
//! conversation's, or any other a turn naming it was written to. When the last
//! timeline that used it is deleted ([`release_timeline`]), whichever path
//! deleted it, its blobs are released and every record of it ceases. An
//! archived conversation still uses its images: its text is kept, and the
//! history shows them. An image kept that no timeline ever used —
//! `POST /v1/images` for a turn that never came — is retired once it has stood
//! unused for [`UNUSED_GRACE_MS`] ([`sweep_unused`]); sending it again renews
//! its keeping (`store::touch`).
//!
//! Every fact is a keyed record of its own, never an entry in a list inside
//! one: two members adding a holder or a use at once write two keys, where a
//! list read, extended and written back would keep whichever wrote last.
//!
//! | key | means |
//! |---|---|
//! | `vision/held/<id>/<holder>` | `holder` sent image `id` |
//! | `vision/used/<id>/<timeline>` | timeline `timeline` reads image `id` |
//! | `vision/timeline/<timeline>/<id>` | the same, found from the timeline |
//!
//! `<holder>` is the caller's own key — the daemon's account key, a digest of
//! who they are — and `<timeline>` is the timeline's id in hex, itself a
//! digest, so neither ever puts a person's identity or a client's id into a
//! key.

use serde_json::json;

use super::blob_refs::release;
use super::store::{blobs_of, image_key};
use crate::persistence::record::CustomObjectPayload;
use crate::persistence::SharedSubstrate;
use crate::projection::TimelineId;

/// How long an image no timeline uses is kept: a day, ample for a client
/// that keeps an image ahead of the turn that names it.
pub const UNUSED_GRACE_MS: u64 = 24 * 60 * 60 * 1000;

/// The images every key of this module lives under.
const IMAGES: &str = "vision/image/";

/// The 16 hex digits a timeline is named by in these keys.
pub fn timeline_key(timeline: TimelineId) -> String {
    format!("{:016x}", timeline.raw())
}

fn held_prefix(id: &str) -> String {
    format!("vision/held/{id}/")
}

fn used_prefix(id: &str) -> String {
    format!("vision/used/{id}/")
}

fn timeline_prefix(timeline: &str) -> String {
    format!("vision/timeline/{timeline}/")
}

fn put(substrate: &SharedSubstrate, key: String) -> Result<(), String> {
    substrate
        .put_custom_object(CustomObjectPayload::new(key, json!({})))
        .map_err(|e| e.to_string())
}

fn cease(substrate: &SharedSubstrate, key: String) -> Result<(), String> {
    substrate
        .put_custom_object(CustomObjectPayload::ceased(key))
        .map_err(|e| e.to_string())
}

/// Record that `holder` sent image `id`. Written once: a holder already
/// recorded writes nothing.
pub fn hold(substrate: &SharedSubstrate, id: &str, holder: &str) -> Result<(), String> {
    if holds(substrate, id, holder) {
        return Ok(());
    }
    put(substrate, format!("{}{holder}", held_prefix(id)))
}

/// Whether `holder` sent image `id`.
pub fn holds(substrate: &SharedSubstrate, id: &str, holder: &str) -> bool {
    substrate
        .custom_object(&format!("{}{holder}", held_prefix(id)))
        .is_some()
}

/// Record that `timeline` reads image `id`, which keeps the image while the
/// timeline lasts. A turn notes its images before it reads them, so no delete
/// of another timeline between the read and the turn's writing finds the
/// image unused.
pub fn note_use(substrate: &SharedSubstrate, id: &str, timeline: TimelineId) -> Result<(), String> {
    let tl = timeline_key(timeline);
    let used = format!("{}{tl}", used_prefix(id));
    if substrate.custom_object(&used).is_some() {
        return Ok(());
    }
    put(substrate, used)?;
    put(substrate, format!("{}{id}", timeline_prefix(&tl)))
}

/// Forget every image the deleted `timeline` used, and retire each that no
/// other timeline uses. The ids retired, in key order.
pub fn release_timeline(
    substrate: &SharedSubstrate,
    timeline: TimelineId,
) -> Result<Vec<String>, String> {
    let tl = timeline_key(timeline);
    let prefix = timeline_prefix(&tl);
    let ids: Vec<String> = substrate
        .custom_objects_under(&prefix)
        .into_iter()
        .filter_map(|(obj, _)| obj.key.strip_prefix(&prefix).map(str::to_string))
        .collect();
    let mut retired = Vec::new();
    for id in ids {
        cease(substrate, format!("{}{tl}", used_prefix(&id)))?;
        cease(substrate, format!("{prefix}{id}"))?;
        if substrate.custom_objects_count_under(&used_prefix(&id)) == 0 {
            retire(substrate, &id)?;
            retired.push(id);
        }
    }
    Ok(retired)
}

/// Whether image `id` has stood unused for its grace at `now_ms`, read as it
/// stands now.
fn stale(substrate: &SharedSubstrate, id: &str, now_ms: u64) -> bool {
    let Some(obj) = substrate.custom_object(&image_key(id)) else {
        return false;
    };
    let kept_ms = obj.blob["kept_ms"].as_u64().unwrap_or(0);
    now_ms.saturating_sub(kept_ms) >= UNUSED_GRACE_MS
        && substrate.custom_objects_count_under(&used_prefix(id)) == 0
}

/// Retire every image no timeline uses that was kept at least
/// [`UNUSED_GRACE_MS`] before `now_ms`. The ids retired, in key order.
///
/// Each is judged again just before it is retired, so one sent again or used
/// while the sweep ran is kept. An image whose record carries no keeping time
/// predates the record's stating one, and is as old as any.
pub fn sweep_unused(substrate: &SharedSubstrate, now_ms: u64) -> Result<Vec<String>, String> {
    let ids: Vec<String> = substrate
        .custom_objects_under(IMAGES)
        .into_iter()
        .filter_map(|(obj, _)| obj.key.strip_prefix(IMAGES).map(str::to_string))
        .collect();
    let mut retired = Vec::new();
    for id in ids {
        if stale(substrate, &id, now_ms) {
            retire(substrate, &id)?;
            retired.push(id);
        }
    }
    Ok(retired)
}

/// Retire image `id`: release every blob it names, and cease its record and
/// who held it.
fn retire(substrate: &SharedSubstrate, id: &str) -> Result<(), String> {
    if let Some(obj) = substrate.custom_object(&image_key(id)) {
        for sha in blobs_of(&obj.blob) {
            release(substrate, &sha, id)?;
        }
        cease(substrate, image_key(id))?;
    }
    for (obj, _) in substrate.custom_objects_under(&held_prefix(id)) {
        cease(substrate, obj.key)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::guest::VisionImage;
    use crate::persistence::Admit;
    use crate::vision::prepare::PreparedImage;
    use crate::vision::store::{load_original, load_rows, load_thumb, store_image};
    use candle_transformers::models::qwen_vision::{Grid, ImageRows};
    use sha2::{Digest, Sha256};

    fn substrate(dir: &tempfile::TempDir) -> SharedSubstrate {
        SharedSubstrate::open_in(dir.path(), Admit::Unbound).unwrap()
    }

    fn timeline(n: u64) -> TimelineId {
        TimelineId::from_raw(n).unwrap()
    }

    /// Keep an image of id `id` whose file is `original`, kept at `kept_ms`,
    /// its rows `value`.
    fn keep(s: &SharedSubstrate, id: &str, original: &[u8], kept_ms: u64, value: f32) {
        let image = PreparedImage {
            id: id.into(),
            sha256: String::new(),
            width: 32,
            height: 32,
            image: VisionImage {
                rgb: Vec::new(),
                height: 32,
                width: 32,
            },
            thumb: [original, &b"-thumb"[..]].concat(),
        };
        let rows = ImageRows {
            grid: Grid {
                height: 32,
                width: 32,
            },
            width: 2,
            values: vec![value, -value],
        };
        store_image(s, &image, original.to_vec(), &rows, "test-tower", kept_ms).unwrap();
    }

    /// A timeline is named by its id's sixteen hex digits.
    #[test]
    fn a_timeline_is_named_by_its_id() {
        assert_eq!(timeline_key(timeline(0xab)), "00000000000000ab");
    }

    /// **Only the people who sent an image hold it**; holding it twice writes
    /// nothing new.
    #[test]
    fn an_image_is_held_by_the_people_who_sent_it() {
        let dir = tempfile::tempdir().unwrap();
        let s = substrate(&dir);
        keep(&s, "00000000000000aa", b"cat", 0, 0.5);
        assert!(!holds(&s, "00000000000000aa", "1111111111111111"));
        hold(&s, "00000000000000aa", "1111111111111111").unwrap();
        hold(&s, "00000000000000aa", "1111111111111111").unwrap();
        assert!(holds(&s, "00000000000000aa", "1111111111111111"));
        assert!(!holds(&s, "00000000000000aa", "2222222222222222"));
        assert_eq!(
            s.custom_objects_count_under("vision/held/00000000000000aa/"),
            1
        );
    }

    /// **An image goes with the last timeline that used it**: deleting one of
    /// two leaves it; deleting the second retires its blobs, its record and
    /// its holders.
    #[test]
    fn an_image_goes_with_the_last_timeline_that_used_it() {
        let dir = tempfile::tempdir().unwrap();
        let s = substrate(&dir);
        let id = "00000000000000bb";
        keep(&s, id, b"dog", 0, 0.5);
        hold(&s, id, "1111111111111111").unwrap();
        note_use(&s, id, timeline(1)).unwrap();
        note_use(&s, id, timeline(2)).unwrap();

        assert_eq!(
            release_timeline(&s, timeline(1)).unwrap(),
            Vec::<String>::new()
        );
        assert_eq!(load_original(&s, id).unwrap(), Some(b"dog".to_vec()));
        assert_eq!(load_thumb(&s, id).unwrap(), Some(b"dog-thumb".to_vec()));

        assert_eq!(
            release_timeline(&s, timeline(2)).unwrap(),
            vec![id.to_string()]
        );
        assert_eq!(load_original(&s, id).unwrap(), None);
        assert_eq!(load_thumb(&s, id).unwrap(), None);
        let thumb_sha: [u8; 32] = Sha256::digest(b"dog-thumb").into();
        assert!(!s.has_blob(&thumb_sha));
        assert!(s.custom_object(&image_key(id)).is_none());
        assert!(!holds(&s, id, "1111111111111111"));
        assert_eq!(s.custom_objects_count_under("vision/"), 0);
    }

    /// **Two images sharing rows keep them until both are gone**: same pixels
    /// in two files are one rows blob.
    #[test]
    fn rows_two_images_share_go_with_the_second() {
        let dir = tempfile::tempdir().unwrap();
        let s = substrate(&dir);
        let (a, b) = ("00000000000000a1", "00000000000000b2");
        keep(&s, a, b"one file", 0, 0.25);
        keep(&s, b, b"another file", 0, 0.25);
        note_use(&s, a, timeline(1)).unwrap();
        note_use(&s, b, timeline(2)).unwrap();
        release_timeline(&s, timeline(1)).unwrap();
        assert_eq!(
            load_rows(&s, b, "test-tower").unwrap().unwrap().values,
            vec![0.25, -0.25]
        );
        release_timeline(&s, timeline(2)).unwrap();
        assert_eq!(s.custom_objects_count_under("vision/"), 0);
    }

    /// A timeline that used nothing releases nothing.
    #[test]
    fn a_timeline_without_images_releases_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let s = substrate(&dir);
        keep(&s, "00000000000000cc", b"fox", 0, 0.5);
        assert!(release_timeline(&s, timeline(9)).unwrap().is_empty());
        assert!(s.custom_object(&image_key("00000000000000cc")).is_some());
    }

    /// **An image no timeline uses is kept for its grace, then retired**; one
    /// a timeline uses is kept however old.
    #[test]
    fn an_unused_image_is_retired_after_its_grace() {
        let dir = tempfile::tempdir().unwrap();
        let s = substrate(&dir);
        let kept_at = 1_000_000;
        keep(&s, "00000000000000dd", b"owl", kept_at, 0.5);
        keep(&s, "00000000000000ee", b"elk", kept_at, 0.75);
        note_use(&s, "00000000000000ee", timeline(1)).unwrap();

        let just_short = kept_at + UNUSED_GRACE_MS - 1;
        assert!(sweep_unused(&s, just_short).unwrap().is_empty());
        let after = kept_at + UNUSED_GRACE_MS;
        assert_eq!(
            sweep_unused(&s, after).unwrap(),
            vec!["00000000000000dd".to_string()]
        );
        assert_eq!(load_original(&s, "00000000000000dd").unwrap(), None);
        assert_eq!(
            load_original(&s, "00000000000000ee").unwrap(),
            Some(b"elk".to_vec())
        );
    }
}
