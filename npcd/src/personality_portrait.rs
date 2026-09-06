//! The portrait a personality carries: the words it is drawn from, and the
//! picture already chosen.
//!
//! # Why a personality has a prompt and a character does not
//!
//! [`crate::portrait`] draws a character from `persona.description`, and the
//! reason there is no prompt field *there* still holds: a character's
//! description and its picture must not be able to drift apart.
//!
//! A personality is the other thing. It is not one character — it is the mould
//! several are struck from, authored once in the mind and read by every daemon
//! that loads it. Its portrait is a piece of art direction: the framing, the
//! lens, the light, the coat. That is written to be *drawn*, and a description
//! is written to be *read*; asking one sentence to do both gets a worse version
//! of each. So the prompt lives here, beside the anchor, in the file a person
//! authors — not on the character, where it could disagree with the description
//! nobody would think to check it against.
//!
//! # Why the picture is a file in the mind
//!
//! `image:` names a file beside the personality rather than an id in this
//! daemon's image store. The store is content-addressed and local; an id in an
//! authored document would name bytes that exist on the machine that drew them
//! and nowhere else, so a fresh clone of the mind would show a broken portrait.
//! A file travels with the personality, because it *is* part of the
//! personality.
//!
//! At load the file is read once and put into the image store, which hands back
//! the id the console fetches through `GET /v1/image/:id`. Content addressing
//! makes that idempotent — a restart re-ingests the same bytes to the same id
//! and writes nothing.
//!
//! # The path is authored, so it is checked
//!
//! `image:` is a string in a file, and the file is edited through the console.
//! It is therefore treated as hostile: [`safe_relative`] admits a plain
//! relative path under the personalities directory and nothing else — no `..`,
//! no absolute root, no drive letter, no backslash, no reaching a file this
//! daemon was not pointed at. The check is on the *shape*, before any join, so
//! there is no canonicalisation race to lose.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::Value;

use crate::images::Images;
use crate::registry::Registry;

/// The block's key in the personality document.
pub const FIELD: &str = "portrait";

/// The most path segments an `image:` may have.
///
/// `portraits/keeper.png` is two. The limit exists so a pathological document
/// cannot make this walk an arbitrarily deep tree.
const MAX_SEGMENTS: usize = 8;

/// Extensions the image store can serve back. Kept in step with
/// [`crate::images`] — a file this admits but that cannot be served would be a
/// portrait that 404s.
const EXTENSIONS: [&str; 5] = ["png", "jpg", "jpeg", "gif", "webp"];

/// The prompt this personality's portrait is drawn from, if it has one.
///
/// Trimmed, and an empty prompt is the same as none: a `prompt: |` somebody
/// started and left blank should fall back to the description rather than draw
/// from whitespace.
pub fn prompt(body: &Value) -> Option<&str> {
    let p = body.get(FIELD)?.get("prompt")?.as_str()?.trim();
    (!p.is_empty()).then_some(p)
}

/// The chosen portrait's path, relative to the personalities directory —
/// **only if it is a path this may open**. See [`safe_relative`].
pub fn image_rel(body: &Value) -> Option<&str> {
    let rel = body.get(FIELD)?.get("image")?.as_str()?.trim();
    safe_relative(rel).then_some(rel)
}

/// Whether an authored path may be joined onto the personalities directory.
///
/// Shape only, and deliberately strict — it is easier to widen this later than
/// to explain why a document could read a file outside the mind.
///
/// Admitted: one or more `/`-separated segments of ordinary characters, ending
/// in an image extension. Refused: anything empty, anything absolute, any `..`
/// or `.` segment, any backslash (a Windows separator that a `/`-only check
/// would wave through), any drive letter, and any NUL.
pub fn safe_relative(rel: &str) -> bool {
    if rel.is_empty() || rel.len() > 255 {
        return false;
    }
    // A backslash is a separator on the platform this most often runs on, so a
    // check that only understood `/` would admit `..\..\secret.png`.
    if rel.contains('\\') || rel.contains('\0') {
        return false;
    }
    // Absolute, by root or by drive — `/etc/x.png`, `C:/x.png`.
    if rel.starts_with('/') || rel.as_bytes().get(1) == Some(&b':') {
        return false;
    }
    let segments: Vec<&str> = rel.split('/').collect();
    if segments.is_empty() || segments.len() > MAX_SEGMENTS {
        return false;
    }
    if segments
        .iter()
        .any(|s| s.is_empty() || *s == ".." || *s == ".")
    {
        return false;
    }
    // The extension has to be one the store serves, judged case-insensitively
    // because a person types `Keeper.PNG` and means the file.
    let Some((_, ext)) = rel.rsplit_once('.') else {
        return false;
    };
    let ext = ext.to_ascii_lowercase();
    EXTENSIONS.contains(&ext.as_str())
}

/// Read every personality's chosen portrait into the image store.
///
/// Returns `personality_id -> image_id`, which is what the listing adds to each
/// record so the console can fetch the picture through the ordinary image
/// route.
///
/// **A personality whose portrait cannot be read is logged and skipped**, never
/// fatal. A missing or corrupt picture must not stop a daemon from serving the
/// character it belongs to — the console falls back to the character's initial,
/// which is what it does for every personality that has no portrait at all.
pub fn ingest(reg: &Registry, dir: &Path, images: &Images) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for record in reg.iter() {
        let Some(rel) = image_rel(&record.body) else {
            // Either there is no portrait, or its path was refused. Say which,
            // because a typo'd path and a deliberate absence look identical in
            // the console and only one of them is a mistake.
            if record
                .body
                .get(FIELD)
                .and_then(|p| p.get("image"))
                .is_some()
            {
                tracing::warn!(
                    personality = %record.id,
                    "portrait image path is not a plain relative path under the personalities \
                     directory, ignoring it",
                );
            }
            continue;
        };
        let path = dir.join(rel);
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(
                    personality = %record.id, path = %path.display(), error = %e,
                    "portrait image could not be read, the character will show its initial",
                );
                continue;
            }
        };
        match images.put(&bytes) {
            Ok(id) => {
                tracing::info!(personality = %record.id, image = %id, "portrait image loaded");
                out.insert(record.id.clone(), id);
            }
            Err(e) => tracing::warn!(
                personality = %record.id, path = %path.display(), error = %e,
                "portrait image was read but could not be stored",
            ),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tmp(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("npcd-personality-portrait-{tag}"));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A one-pixel PNG, by its magic bytes — enough for the store to accept.
    fn png() -> Vec<u8> {
        let mut v = b"\x89PNG\r\n\x1a\n".to_vec();
        v.extend_from_slice(&[0u8; 64]);
        v
    }

    #[test]
    fn a_prompt_is_read_and_trimmed() {
        let body = json!({ "portrait": { "prompt": "  a weathered guardian  " } });
        assert_eq!(prompt(&body), Some("a weathered guardian"));
    }

    /// A blank prompt is the same as no prompt — it must fall through to the
    /// description rather than draw from an empty string.
    #[test]
    fn a_blank_prompt_is_no_prompt() {
        assert_eq!(prompt(&json!({ "portrait": { "prompt": "  \n " } })), None);
        assert_eq!(prompt(&json!({ "portrait": {} })), None);
        assert_eq!(prompt(&json!({})), None);
    }

    #[test]
    fn an_ordinary_relative_path_is_admitted() {
        for good in [
            "portraits/keeper.png",
            "keeper.png",
            "a/b/c/keeper.webp",
            "Keeper.PNG",
            "portraits/keeper-2.jpeg",
        ] {
            assert!(safe_relative(good), "{good} should be admitted");
        }
    }

    /// **The path comes out of a file a person edits through the console.**
    ///
    /// Every one of these is a way to name something outside the personalities
    /// directory, and the shape check is what stops them before a join.
    #[test]
    fn nothing_that_escapes_the_personalities_directory_is_admitted() {
        for bad in [
            "",
            "../secret.png",
            "portraits/../../secret.png",
            "./keeper.png",
            "/etc/passwd.png",
            "C:/Windows/system32/x.png",
            "c:keeper.png",
            "..\\secret.png",
            "portraits\\keeper.png",
            "portraits//keeper.png",
            "portraits/keeper.png\0.txt",
            "a/b/c/d/e/f/g/h/i/keeper.png",
        ] {
            assert!(!safe_relative(bad), "{bad:?} should be refused");
        }
    }

    /// Only what the image store can serve. A path this admitted but that the
    /// store would not hand back is a portrait that 404s in the console.
    #[test]
    fn only_servable_image_extensions_are_admitted() {
        for bad in [
            "keeper",
            "keeper.txt",
            "keeper.svg",
            "keeper.html",
            "keeper.png.html",
        ] {
            assert!(!safe_relative(bad), "{bad:?} should be refused");
        }
    }

    /// A refused path must not reach [`image_rel`]'s caller as something to
    /// open — the guard belongs on the read side, not only in the test above.
    #[test]
    fn a_refused_path_is_not_returned_as_an_image() {
        let body = json!({ "portrait": { "image": "../../secret.png" } });
        assert_eq!(image_rel(&body), None);
        let body = json!({ "portrait": { "image": "portraits/keeper.png" } });
        assert_eq!(image_rel(&body), Some("portraits/keeper.png"));
    }

    /// The end to end: a personality naming a real file gets an id, and one
    /// naming a missing file is skipped rather than failing the load.
    #[test]
    fn ingest_stores_what_it_can_and_skips_what_it_cannot() {
        let dir = tmp("ingest");
        std::fs::create_dir_all(dir.join("portraits")).unwrap();
        std::fs::write(dir.join("portraits/keeper.png"), png()).unwrap();
        std::fs::write(
            dir.join("keeper.yaml"),
            "id: keeper\nportrait:\n  image: portraits/keeper.png\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("ghost.yaml"),
            "id: ghost\nportrait:\n  image: portraits/nothing.png\n",
        )
        .unwrap();
        std::fs::write(dir.join("plain.yaml"), "id: plain\n").unwrap();

        let reg = Registry::load("personality", &dir).unwrap();
        let images = Images::new(&tmp("ingest-store"));
        let map = ingest(&reg, &dir, &images);

        let id = map.get("keeper").expect("keeper's portrait was stored");
        assert!(id.starts_with("img_") && id.ends_with(".png"), "{id}");
        assert!(images.get(id).is_ok(), "the id does not serve");
        assert!(!map.contains_key("ghost"), "a missing file became an id");
        assert!(!map.contains_key("plain"));
    }

    /// Content addressing means a restart re-ingests to the same id, so the
    /// console's cached portrait URL stays valid across one.
    #[test]
    fn re_ingesting_the_same_file_gives_the_same_id() {
        let dir = tmp("stable");
        std::fs::create_dir_all(dir.join("portraits")).unwrap();
        std::fs::write(dir.join("portraits/k.png"), png()).unwrap();
        std::fs::write(
            dir.join("k.yaml"),
            "id: k\nportrait:\n  image: portraits/k.png\n",
        )
        .unwrap();
        let reg = Registry::load("personality", &dir).unwrap();
        let images = Images::new(&tmp("stable-store"));
        assert_eq!(ingest(&reg, &dir, &images), ingest(&reg, &dir, &images));
    }
}
