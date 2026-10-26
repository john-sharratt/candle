//! Pictures drawn for a conversation (`image_generate`), kept for a day.
//!
//! A drawn picture is stored like any image, so its link can open it
//! (`crate::image_links`), and noted as `image/drawn/<id>` with the moment it
//! was drawn. Once [`KEPT`] has passed, [`Images::retire_drawn`] retires the
//! blob — a tombstone that replicates, and the bytes go at the next compaction
//! (`docs/replicated_substrate.md` §10.3) — unless a portrait has come to name
//! the same image, and ends the note either way. Storing the same bytes as an
//! ordinary image — a portrait made of the picture — ends the note at once, so
//! no member retires it after; one stored after the picture was retired
//! writes the bytes again, and that record outranks the tombstone.

use std::collections::HashSet;
use std::time::Duration;

use candle_conversation::persistence::record::CustomObjectPayload;
use serde_json::json;

use super::{valid_id, ImageError, Images};

/// How long a drawn picture is kept, and its link works.
pub const KEPT: Duration = Duration::from_secs(24 * 60 * 60);

/// What every drawn picture's note is keyed under, before its image id.
const PREFIX: &str = "image/drawn/";

/// The field of a note holding when it was drawn, in Unix milliseconds.
const DRAWN_MS: &str = "drawn_ms";

impl Images {
    /// Store a drawn picture, noting it was drawn at `now_ms`, and give back
    /// its id.
    pub fn put_drawn(&self, bytes: &[u8], now_ms: u64) -> Result<String, ImageError> {
        let id = self.store(bytes)?;
        self.shared.put_custom_object(CustomObjectPayload::new(
            note_key(&id),
            json!({ DRAWN_MS: now_ms }),
        ))?;
        Ok(id)
    }

    /// End image `id`'s drawn note, if it has one: it is kept from now on.
    pub(super) fn keep_drawn(&self, id: &str) -> Result<(), ImageError> {
        let key = note_key(id);
        if self.shared.custom_object(&key).is_some() {
            self.shared
                .put_custom_object(CustomObjectPayload::ceased(key))?;
        }
        Ok(())
    }

    /// Retire every drawn picture older than [`KEPT`] at `now_ms` that no
    /// portrait in `named` uses, and end the note of each one past it.
    /// Returns how many were retired.
    pub fn retire_drawn(&self, now_ms: u64, named: &HashSet<String>) -> Result<usize, ImageError> {
        let mut retired = 0;
        for (note, _) in self.shared.custom_objects_under(PREFIX) {
            // A note that does not say when it was drawn is left as it is:
            // retiring on a guess would take a picture whose link still works.
            let Some(drawn_ms) = note.blob[DRAWN_MS].as_u64() else {
                tracing::warn!("images: drawn note `{}` has no time; left alone", note.key);
                continue;
            };
            if now_ms.saturating_sub(drawn_ms) < KEPT.as_millis() as u64 {
                continue;
            }
            let id = &note.key[PREFIX.len()..];
            if !named.contains(id) {
                if let Some((sha256, _)) = valid_id(id) {
                    self.shared.retire_blob(&sha256)?;
                    retired += 1;
                }
            }
            self.shared
                .put_custom_object(CustomObjectPayload::ceased(note.key.clone()))?;
        }
        Ok(retired)
    }
}

/// The key image `id`'s drawn note is kept under.
fn note_key(id: &str) -> String {
    format!("{PREFIX}{id}")
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::npcs::Npcs;

    const HOUR_MS: u64 = 60 * 60 * 1000;
    const NOW: u64 = 1_800_000_000_000;

    fn jpeg(n: u8) -> Vec<u8> {
        vec![0xFF, 0xD8, 0xFF, 0xE0, n, n]
    }

    fn store(dir: &TempDir) -> Images {
        let shared = Npcs::load(dir.path()).unwrap().substrate();
        Images::open(shared, &dir.path().join("images")).unwrap().0
    }

    /// **A drawn picture is kept for a day, then retired** — read back until
    /// then, gone after, and not retired twice.
    #[test]
    fn a_drawn_picture_is_kept_a_day_then_retired() {
        let dir = TempDir::new().unwrap();
        let images = store(&dir);
        let id = images.put_drawn(&jpeg(1), NOW).unwrap();
        let none = HashSet::new();
        assert_eq!(images.retire_drawn(NOW + 23 * HOUR_MS, &none).unwrap(), 0);
        assert_eq!(images.get(&id).unwrap().0, jpeg(1));
        assert_eq!(images.retire_drawn(NOW + 24 * HOUR_MS, &none).unwrap(), 1);
        assert!(matches!(images.get(&id), Err(ImageError::NotFound)));
        assert_eq!(images.retire_drawn(NOW + 25 * HOUR_MS, &none).unwrap(), 0);
        drop(images);
        let again = store(&dir);
        assert!(
            matches!(again.get(&id), Err(ImageError::NotFound)),
            "retired across a restart"
        );
    }

    /// **A drawn picture a portrait uses is kept**, its note ended all the
    /// same; one drawn beside it is retired.
    #[test]
    fn a_picture_a_portrait_uses_is_kept() {
        let dir = TempDir::new().unwrap();
        let images = store(&dir);
        let portrait = images.put_drawn(&jpeg(1), NOW).unwrap();
        let other = images.put_drawn(&jpeg(2), NOW).unwrap();
        let named = HashSet::from([portrait.clone()]);
        assert_eq!(
            images
                .retire_drawn(NOW + KEPT.as_millis() as u64, &named)
                .unwrap(),
            1
        );
        assert_eq!(images.get(&portrait).unwrap().0, jpeg(1));
        assert!(matches!(images.get(&other), Err(ImageError::NotFound)));
        assert_eq!(
            images
                .retire_drawn(NOW + 48 * HOUR_MS, &HashSet::new())
                .unwrap(),
            0,
            "its note was ended, so it is not retired later"
        );
        assert_eq!(images.get(&portrait).unwrap().0, jpeg(1));
    }

    /// **A picture stored as an ordinary image within its day is kept** — a
    /// portrait made of it, on whichever member — and never retired after.
    #[test]
    fn a_picture_stored_as_an_image_is_kept() {
        let dir = TempDir::new().unwrap();
        let images = store(&dir);
        let id = images.put_drawn(&jpeg(4), NOW).unwrap();
        assert_eq!(images.put(&jpeg(4)).unwrap(), id);
        assert_eq!(
            images
                .retire_drawn(NOW + 48 * HOUR_MS, &HashSet::new())
                .unwrap(),
            0
        );
        assert_eq!(images.get(&id).unwrap().0, jpeg(4));
    }

    /// **The same bytes stored after their picture was retired live again** —
    /// the portrait someone uploads from the file they were sent.
    #[test]
    fn the_same_bytes_stored_again_live() {
        let dir = TempDir::new().unwrap();
        let images = store(&dir);
        let id = images.put_drawn(&jpeg(3), NOW).unwrap();
        images
            .retire_drawn(NOW + KEPT.as_millis() as u64, &HashSet::new())
            .unwrap();
        assert!(matches!(images.get(&id), Err(ImageError::NotFound)));
        assert_eq!(images.put(&jpeg(3)).unwrap(), id);
        assert_eq!(images.get(&id).unwrap().0, jpeg(3));
    }
}
