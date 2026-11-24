//! Who may read a kept image, and for how long it is kept.
//!
//! An image is the file a person sent. Its bytes (`GET /v1/images/{id}`) and
//! its rows, named by id in a turn, are read by a person who sent it
//! ([`holds`]) or by an admin, never by anyone who merely learned its id.
//!
//! An image lives as long as a conversation uses it ([`note_use`]). When the
//! last conversation that used it is deleted ([`release_conversation`]), its
//! bytes and rows are retired, and with them every record of it. An archived
//! conversation still uses its images: its text is kept, and the history shows
//! them. An image kept that no conversation ever used — `POST /v1/images` for a
//! turn that never came — is retired once it has stood unused for
//! [`UNUSED_GRACE_MS`] ([`sweep_unused`]).
//!
//! Every fact is a keyed record of its own, never an entry in a list inside
//! one: two members adding a holder or a use at once write two keys, where a
//! list read, extended and written back would keep whichever wrote last.
//!
//! | key | means |
//! |---|---|
//! | `vision/held/<id>/<holder>` | `holder` sent image `id` |
//! | `vision/used/<id>/<conv>` | conversation `conv` reads image `id` |
//! | `vision/conv/<conv>/<id>` | the same, found from the conversation |
//!
//! `<holder>` is the caller's own key — the daemon's account key, a digest of
//! who they are — and `<conv>` is [`conversation_key`] of the conversation's
//! id, so neither ever puts a person's identity or a client's id into a key.

use serde_json::json;
use sha2::{Digest, Sha256};

use super::store::{image_key, unhex};
use crate::persistence::record::CustomObjectPayload;
use crate::persistence::SharedSubstrate;

/// How long an image no conversation uses is kept: a day, ample for a client
/// that keeps an image ahead of the turn that names it.
pub const UNUSED_GRACE_MS: u64 = 24 * 60 * 60 * 1000;

/// The images every key of this module lives under.
const IMAGES: &str = "vision/image/";

/// The 16 hex digits a conversation's id is named by in these keys.
pub fn conversation_key(conv_id: &str) -> String {
    Sha256::digest(conv_id.as_bytes())[..8]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn held_prefix(id: &str) -> String {
    format!("vision/held/{id}/")
}

fn used_prefix(id: &str) -> String {
    format!("vision/used/{id}/")
}

fn conv_prefix(conv: &str) -> String {
    format!("vision/conv/{conv}/")
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

/// Record that the conversation `conv_id` reads image `id`, which keeps the
/// image while the conversation lasts.
pub fn note_use(substrate: &SharedSubstrate, id: &str, conv_id: &str) -> Result<(), String> {
    let conv = conversation_key(conv_id);
    let used = format!("{}{conv}", used_prefix(id));
    if substrate.custom_object(&used).is_some() {
        return Ok(());
    }
    put(substrate, used)?;
    put(substrate, format!("{}{id}", conv_prefix(&conv)))
}

/// Forget every image the deleted conversation `conv_id` used, and retire each
/// that no other conversation uses. The ids retired, in key order.
pub fn release_conversation(
    substrate: &SharedSubstrate,
    conv_id: &str,
) -> Result<Vec<String>, String> {
    let conv = conversation_key(conv_id);
    let prefix = conv_prefix(&conv);
    let ids: Vec<String> = substrate
        .custom_objects_under(&prefix)
        .into_iter()
        .filter_map(|(obj, _)| obj.key.strip_prefix(&prefix).map(str::to_string))
        .collect();
    let mut retired = Vec::new();
    for id in ids {
        cease(substrate, format!("{}{conv}", used_prefix(&id)))?;
        cease(substrate, format!("{prefix}{id}"))?;
        if substrate.custom_objects_count_under(&used_prefix(&id)) == 0 {
            retire(substrate, &id)?;
            retired.push(id);
        }
    }
    Ok(retired)
}

/// Retire every image no conversation uses that was kept at least
/// [`UNUSED_GRACE_MS`] before `now_ms`. The ids retired, in key order.
///
/// An image whose record carries no keeping time predates the record's
/// stating one, and is as old as any.
pub fn sweep_unused(substrate: &SharedSubstrate, now_ms: u64) -> Result<Vec<String>, String> {
    let stale: Vec<String> = substrate
        .custom_objects_under(IMAGES)
        .into_iter()
        .filter_map(|(obj, _)| {
            let id = obj.key.strip_prefix(IMAGES)?.to_string();
            let kept_ms = obj.blob["kept_ms"].as_u64().unwrap_or(0);
            (now_ms.saturating_sub(kept_ms) >= UNUSED_GRACE_MS
                && substrate.custom_objects_count_under(&used_prefix(&id)) == 0)
                .then_some(id)
        })
        .collect();
    for id in &stale {
        retire(substrate, id)?;
    }
    Ok(stale)
}

/// Retire image `id`: its bytes and its rows, its record, and who held it.
fn retire(substrate: &SharedSubstrate, id: &str) -> Result<(), String> {
    if let Some(obj) = substrate.custom_object(&image_key(id)) {
        for field in ["sha256", "rows"] {
            if let Some(digest) = obj.blob[field].as_str().and_then(unhex) {
                substrate.retire_blob(&digest).map_err(|e| e.to_string())?;
            }
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
    use crate::vision::store::{load_original, store_image};
    use candle_transformers::models::qwen_vision::{Grid, ImageRows};

    fn substrate(dir: &tempfile::TempDir) -> SharedSubstrate {
        SharedSubstrate::open_in(dir.path(), Admit::Unbound).unwrap()
    }

    /// Keep an image of id `id` whose file is `original`, kept at `kept_ms`.
    fn keep(s: &SharedSubstrate, id: &str, original: &[u8], kept_ms: u64) {
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
        };
        let rows = ImageRows {
            grid: Grid {
                height: 32,
                width: 32,
            },
            width: 2,
            values: vec![0.5, -0.5],
        };
        store_image(s, &image, original.to_vec(), &rows, kept_ms).unwrap();
    }

    /// The key a conversation is named by is the start of its id's SHA-256:
    /// `abc`'s is `ba7816bf8f01cfea`.
    #[test]
    fn a_conversation_is_named_by_its_ids_digest() {
        assert_eq!(conversation_key("abc"), "ba7816bf8f01cfea");
    }

    /// **Only the people who sent an image hold it**; holding it twice writes
    /// nothing new.
    #[test]
    fn an_image_is_held_by_the_people_who_sent_it() {
        let dir = tempfile::tempdir().unwrap();
        let s = substrate(&dir);
        keep(&s, "00000000000000aa", b"cat", 0);
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

    /// **An image goes with the last conversation that used it**: deleting one
    /// of two leaves it; deleting the second retires its bytes, its record and
    /// its holders.
    #[test]
    fn an_image_goes_with_the_last_conversation_that_used_it() {
        let dir = tempfile::tempdir().unwrap();
        let s = substrate(&dir);
        let id = "00000000000000bb";
        keep(&s, id, b"dog", 0);
        hold(&s, id, "1111111111111111").unwrap();
        note_use(&s, id, "conv-a").unwrap();
        note_use(&s, id, "conv-b").unwrap();

        assert_eq!(
            release_conversation(&s, "conv-a").unwrap(),
            Vec::<String>::new()
        );
        assert_eq!(load_original(&s, id).unwrap(), Some(b"dog".to_vec()));

        assert_eq!(
            release_conversation(&s, "conv-b").unwrap(),
            vec![id.to_string()]
        );
        assert_eq!(load_original(&s, id).unwrap(), None);
        assert!(s.custom_object(&image_key(id)).is_none());
        assert!(!holds(&s, id, "1111111111111111"));
        assert_eq!(s.custom_objects_count_under("vision/"), 0);
    }

    /// A conversation that used nothing releases nothing.
    #[test]
    fn a_conversation_without_images_releases_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let s = substrate(&dir);
        keep(&s, "00000000000000cc", b"fox", 0);
        assert!(release_conversation(&s, "conv-x").unwrap().is_empty());
        assert!(s.custom_object(&image_key("00000000000000cc")).is_some());
    }

    /// **An image no conversation uses is kept for its grace, then retired**;
    /// one a conversation uses is kept however old.
    #[test]
    fn an_unused_image_is_retired_after_its_grace() {
        let dir = tempfile::tempdir().unwrap();
        let s = substrate(&dir);
        let kept_at = 1_000_000;
        keep(&s, "00000000000000dd", b"owl", kept_at);
        keep(&s, "00000000000000ee", b"elk", kept_at);
        note_use(&s, "00000000000000ee", "conv-a").unwrap();

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
