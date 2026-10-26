//! Pictures kept for a while, then retired: one drawn for a conversation
//! (`image_generate`, `image_edit`) for a day, one uploaded to be edited
//! (`PUT /v1/image/upload`) for an hour.
//!
//! Such a picture is stored like any image, so a link can open it
//! (`crate::image_links`), and noted as `image/<kind>/<id>` with the moment it
//! arrived. [`Images::retire_kept`] retires the blob — a tombstone that
//! replicates, and the bytes go at the next compaction
//! (`docs/replicated_substrate.md` §10.3) — once **every** note on it has run
//! its time and no portrait names it, and ends each note past its time. So a
//! drawn picture uploaded again keeps its day, and an upload drawn from keeps
//! nothing it was not promised.
//!
//! Storing the same bytes as an ordinary image — a portrait made of the
//! picture — ends its notes at once, so no member retires it after; one stored
//! after the picture was retired writes the bytes again, and that record
//! outranks the tombstone.

use std::collections::{BTreeMap, HashSet};
use std::time::Duration;

use candle_conversation::persistence::record::CustomObjectPayload;
use serde_json::json;

use super::{valid_id, ImageError, Images};

/// Why a picture is kept, and so for how long.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kept {
    /// Drawn for a conversation: as long as its link works.
    Drawn,
    /// Uploaded to be edited: long enough to edit, and no longer.
    Uploaded,
}

impl Kept {
    const ALL: [Kept; 2] = [Kept::Drawn, Kept::Uploaded];

    /// How long a picture kept for this is kept.
    pub const fn lifetime(self) -> Duration {
        match self {
            Kept::Drawn => Duration::from_secs(24 * 60 * 60),
            Kept::Uploaded => Duration::from_secs(60 * 60),
        }
    }

    /// What its notes are keyed under, before the image id.
    fn prefix(self) -> &'static str {
        match self {
            Kept::Drawn => "image/drawn/",
            Kept::Uploaded => "image/uploaded/",
        }
    }

    fn note_key(self, id: &str) -> String {
        format!("{}{id}", self.prefix())
    }

    /// The field of its notes holding when the picture arrived, in Unix
    /// milliseconds.
    fn at_field(self) -> &'static str {
        match self {
            Kept::Drawn => "drawn_ms",
            Kept::Uploaded => "uploaded_ms",
        }
    }
}

impl Images {
    /// Store a picture kept for `why`, noting it arrived at `now_ms`, and give
    /// back its id.
    pub fn put_kept(&self, bytes: &[u8], why: Kept, now_ms: u64) -> Result<String, ImageError> {
        let id = self.store(bytes)?;
        self.shared.put_custom_object(CustomObjectPayload::new(
            why.note_key(&id),
            json!({ why.at_field(): now_ms }),
        ))?;
        Ok(id)
    }

    /// End image `id`'s notes, if it has any: it is kept from now on.
    pub(super) fn keep_for_good(&self, id: &str) -> Result<(), ImageError> {
        for why in Kept::ALL {
            let key = why.note_key(id);
            if self.shared.custom_object(&key).is_some() {
                self.shared
                    .put_custom_object(CustomObjectPayload::ceased(key))?;
            }
        }
        Ok(())
    }

    /// Retire, at `now_ms`, every kept picture whose every note has run its
    /// time and that no portrait in `named` uses, and end each note past its
    /// time. Returns how many were retired.
    pub fn retire_kept(&self, now_ms: u64, named: &HashSet<String>) -> Result<usize, ImageError> {
        // Each image's notes: whether all have run out, and those that have.
        let mut notes: BTreeMap<String, (bool, Vec<String>)> = BTreeMap::new();
        for why in Kept::ALL {
            for (note, _) in self.shared.custom_objects_under(why.prefix()) {
                let id = note.key[why.prefix().len()..].to_string();
                let entry = notes.entry(id).or_insert((true, Vec::new()));
                // A note that does not say when the picture arrived is left as
                // it is, and keeps its picture: retiring on a guess would take
                // one whose link still works.
                let Some(at_ms) = note.blob[why.at_field()].as_u64() else {
                    tracing::warn!("images: kept note `{}` has no time; left alone", note.key);
                    entry.0 = false;
                    continue;
                };
                if now_ms.saturating_sub(at_ms) < why.lifetime().as_millis() as u64 {
                    entry.0 = false;
                } else {
                    entry.1.push(note.key.clone());
                }
            }
        }
        let mut retired = 0;
        for (id, (all_past, past)) in notes {
            if all_past && !named.contains(&id) {
                if let Some((sha256, _)) = valid_id(&id) {
                    self.shared.retire_blob(&sha256)?;
                    retired += 1;
                }
            }
            for key in past {
                self.shared
                    .put_custom_object(CustomObjectPayload::ceased(key))?;
            }
        }
        Ok(retired)
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::npcs::Npcs;

    const MIN_MS: u64 = 60 * 1000;
    const HOUR_MS: u64 = 60 * MIN_MS;
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
        let id = images.put_kept(&jpeg(1), Kept::Drawn, NOW).unwrap();
        let none = HashSet::new();
        assert_eq!(images.retire_kept(NOW + 23 * HOUR_MS, &none).unwrap(), 0);
        assert_eq!(images.get(&id).unwrap().0, jpeg(1));
        assert_eq!(images.retire_kept(NOW + 24 * HOUR_MS, &none).unwrap(), 1);
        assert!(matches!(images.get(&id), Err(ImageError::NotFound)));
        assert_eq!(images.retire_kept(NOW + 25 * HOUR_MS, &none).unwrap(), 0);
        drop(images);
        let again = store(&dir);
        assert!(
            matches!(again.get(&id), Err(ImageError::NotFound)),
            "retired across a restart"
        );
    }

    /// **An uploaded picture is kept for an hour.**
    #[test]
    fn an_uploaded_picture_is_kept_an_hour() {
        let dir = TempDir::new().unwrap();
        let images = store(&dir);
        let id = images.put_kept(&jpeg(5), Kept::Uploaded, NOW).unwrap();
        let none = HashSet::new();
        assert_eq!(images.retire_kept(NOW + 59 * MIN_MS, &none).unwrap(), 0);
        assert_eq!(images.get(&id).unwrap().0, jpeg(5));
        assert_eq!(images.retire_kept(NOW + HOUR_MS, &none).unwrap(), 1);
        assert!(matches!(images.get(&id), Err(ImageError::NotFound)));
    }

    /// **A picture goes only when every note on it has run out**: a drawn
    /// picture uploaded again keeps its day.
    #[test]
    fn a_picture_noted_twice_keeps_its_longest_time() {
        let dir = TempDir::new().unwrap();
        let images = store(&dir);
        let id = images.put_kept(&jpeg(6), Kept::Drawn, NOW).unwrap();
        assert_eq!(images.put_kept(&jpeg(6), Kept::Uploaded, NOW).unwrap(), id);
        let none = HashSet::new();
        assert_eq!(images.retire_kept(NOW + 2 * HOUR_MS, &none).unwrap(), 0);
        assert_eq!(images.get(&id).unwrap().0, jpeg(6));
        assert_eq!(images.retire_kept(NOW + 24 * HOUR_MS, &none).unwrap(), 1);
        assert!(matches!(images.get(&id), Err(ImageError::NotFound)));
    }

    /// **A kept picture a portrait uses is kept**, its note ended all the
    /// same; one kept beside it is retired.
    #[test]
    fn a_picture_a_portrait_uses_is_kept() {
        let dir = TempDir::new().unwrap();
        let images = store(&dir);
        let portrait = images.put_kept(&jpeg(1), Kept::Drawn, NOW).unwrap();
        let other = images.put_kept(&jpeg(2), Kept::Drawn, NOW).unwrap();
        let named = HashSet::from([portrait.clone()]);
        let day = Kept::Drawn.lifetime().as_millis() as u64;
        assert_eq!(images.retire_kept(NOW + day, &named).unwrap(), 1);
        assert_eq!(images.get(&portrait).unwrap().0, jpeg(1));
        assert!(matches!(images.get(&other), Err(ImageError::NotFound)));
        assert_eq!(
            images
                .retire_kept(NOW + 48 * HOUR_MS, &HashSet::new())
                .unwrap(),
            0,
            "its note was ended, so it is not retired later"
        );
        assert_eq!(images.get(&portrait).unwrap().0, jpeg(1));
    }

    /// **A kept picture stored as an ordinary image is kept for good** — a
    /// portrait made of it, on whichever member.
    #[test]
    fn a_picture_stored_as_an_image_is_kept() {
        let dir = TempDir::new().unwrap();
        let images = store(&dir);
        let id = images.put_kept(&jpeg(4), Kept::Uploaded, NOW).unwrap();
        assert_eq!(images.put(&jpeg(4)).unwrap(), id);
        assert_eq!(
            images
                .retire_kept(NOW + 48 * HOUR_MS, &HashSet::new())
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
        let id = images.put_kept(&jpeg(3), Kept::Drawn, NOW).unwrap();
        let day = Kept::Drawn.lifetime().as_millis() as u64;
        images.retire_kept(NOW + day, &HashSet::new()).unwrap();
        assert!(matches!(images.get(&id), Err(ImageError::NotFound)));
        assert_eq!(images.put(&jpeg(3)).unwrap(), id);
        assert_eq!(images.get(&id).unwrap().0, jpeg(3));
    }
}
