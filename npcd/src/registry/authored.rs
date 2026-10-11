//! A world or a personality as the document an author keeps in the mind
//! (`docs/replicated_substrate.md` §10.3).
//!
//! These documents carry their reasoning in their comments, and none of that
//! is data. Saving through `serde_yaml` would round-trip the parsed value and
//! delete every comment and block scalar in the file — so a save edits the
//! text that is there through [`yaml_edit`], replacing only the values that
//! changed and copying every other byte through.
//!
//! The text edited is the one the author sees — their own draft of it, or the
//! record's — so a comment, a blank line or a rewrap somebody added survives
//! the next save. That is not protection against a lost update: a `PUT` is a
//! whole-document replacement, so a data field the caller sends overwrites an
//! edit to that same field whichever text is the base.

use anyhow::Context;
use serde_json::Value;

use super::{id, PutError, Record, MAX_BYTES};
use crate::registry::yaml_edit;

/// The folder of the mind the worlds are kept in.
pub const WORLDS: &str = "worlds";

/// The folder of the mind the personalities are kept in.
pub const PERSONALITIES: &str = "personalities";

/// Where record `id` of the collection kept in `folder` lives in the mind.
pub fn path_of(folder: &str, id: &str) -> String {
    format!("{folder}/{id}.yaml")
}

/// The record a document of the collection is, or `None` for a file that is
/// not one: a name that is not an id, or text that does not parse. A file
/// dropped in by hand — or arriving from a checkout made on another OS — must
/// not become an id the console then cannot save back.
pub fn record_of(kind: &str, file: &str, text: &str) -> Option<Record> {
    let stem = file.strip_suffix(".yaml")?;
    if let Err(e) = id::check(stem) {
        tracing::warn!(file, error = %e, "{kind}: unusable file name, skipping");
        return None;
    }
    match serde_yaml::from_str::<Value>(text) {
        Ok(body) => Some(Record {
            id: stem.to_string(),
            body,
        }),
        Err(e) => {
            tracing::warn!(file, error = %e, "{kind}: does not parse, skipping");
            None
        }
    }
}

/// The text to keep for `body`: an edit of `original` where there is one, a
/// fresh serialisation where there is nothing to edit.
pub fn render(
    kind: &str,
    id: &str,
    original: Option<&str>,
    body: &Value,
) -> Result<String, PutError> {
    id::check(id).map_err(PutError::BadId)?;
    let whole = || {
        serde_yaml::to_string(body)
            .with_context(|| format!("{kind}: serialising `{id}`"))
            .map_err(PutError::Io)
    };
    let yaml = match (body.as_object(), original) {
        (Some(map), Some(original)) => match yaml_edit::splice(original, map) {
            Some(edited) => edited,
            None => {
                // The document exists and could not be edited — an alias, a
                // root that is not a mapping, or a rendering that failed its
                // own read-back. Saying so matters: the save still succeeds,
                // and the cost is the author's comments.
                tracing::warn!(
                    "{kind}: `{id}` could not be edited in place — rewriting whole, comments will \
                     be lost"
                );
                whole()?
            }
        },
        _ => whole()?,
    };
    if yaml.len() > MAX_BYTES {
        return Err(PutError::TooLarge {
            bytes: yaml.len(),
            max: MAX_BYTES,
        });
    }
    Ok(yaml)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// **Saving one field changes the one line it must** and leaves the
    /// author's comments and the shape of the file alone.
    #[test]
    fn a_save_keeps_the_comments_and_the_shape_of_the_file() {
        let authored = "\
# Sandbox -- why this world is written the way it is.
#
# The setting says what IS there, not what is absent.

id: sandbox
name: Sandbox
public: false

setting: >-
  A room, a view from it, and objects that behave the same way
  every time you touch them.

# No shared canon: this world's whole content is one life.
selects: []
";
        let mut body: Value = serde_yaml::from_str(authored).unwrap();
        body["name"] = json!("Sandbox Prime");
        assert_eq!(
            render("world", "sandbox", Some(authored), &body).unwrap(),
            authored.replace("name: Sandbox\n", "name: Sandbox Prime\n")
        );
    }

    /// **A field absent from the body goes, and the comment that introduced
    /// it stays.** A `PUT` replaces the document — that is how the console
    /// removes a field — and a comment is a person's prose, which may explain
    /// why the field went.
    #[test]
    fn a_field_left_out_goes_and_its_comment_stays() {
        let original =
            "# written by hand\nname: Ardh\nsetting: Old.\n\n# and a new field\nmood: bleak\n";
        let out = render(
            "world",
            "ardh",
            Some(original),
            &json!({"name": "Low Fen", "setting": "Old."}),
        )
        .unwrap();
        assert!(out.contains("# written by hand"), "{out}");
        assert!(out.contains("name: Low Fen"), "{out}");
        assert!(!out.contains("mood: bleak"), "{out}");
        assert!(out.contains("# and a new field"), "{out}");
    }

    /// A document with nothing to edit is serialised whole.
    #[test]
    fn a_new_document_is_written_from_scratch() {
        assert_eq!(
            render(
                "world",
                "ardh",
                None,
                &json!({"name": "Ardh", "selects": ["north"]})
            )
            .unwrap(),
            "name: Ardh\nselects:\n- north\n"
        );
    }

    /// A document larger than the limit is refused rather than kept; one
    /// well inside it is kept.
    #[test]
    fn an_oversized_document_is_refused() {
        let big = json!({ "name": "x", "setting": "a".repeat(MAX_BYTES + 1) });
        assert!(matches!(
            render("world", "big", None, &big),
            Err(PutError::TooLarge { .. })
        ));
        let ok = json!({ "name": "x", "setting": "a".repeat(64 * 1024) });
        assert!(render("world", "fine", None, &ok).is_ok());
    }

    #[test]
    fn a_bad_id_is_refused() {
        for bad in ["../escape", "a/b", "con", "", "x.yaml", "-x"] {
            assert!(
                matches!(
                    render("world", bad, None, &json!({})),
                    Err(PutError::BadId(_))
                ),
                "accepted `{bad}`"
            );
        }
    }

    /// A file is a record when its name is an id and its text parses; a
    /// field the daemon has no struct for comes back unchanged.
    #[test]
    fn a_file_is_a_record_when_it_names_an_id_and_parses() {
        let r = record_of(
            "world",
            "ardh.yaml",
            "name: Ardh\nfuture: {deep: [1, 2, 3]}\n",
        )
        .unwrap();
        assert_eq!(r.id, "ardh");
        assert_eq!(r.body["future"]["deep"][2], 3);
        assert!(record_of("world", "Ardh.yaml", "name: A\n").is_none());
        assert!(record_of("world", "broken.yaml", "name: [unclosed\n").is_none());
        assert!(record_of("world", "notes.md", "name: A\n").is_none());
    }

    #[test]
    fn a_record_lives_in_its_collections_folder() {
        assert_eq!(path_of("worlds", "ardh"), "worlds/ardh.yaml");
    }
}
