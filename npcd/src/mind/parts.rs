//! One item inside a document, addressed on its own.
//!
//! Most of the corpus is a file per thing: a response is a file, a character is
//! a file, a canon page is a file. The projection schema is not. Its nine
//! layers live in one seven-hundred-line document, and until this existed the
//! only way to change a layer's budget was to find it in a textarea of YAML.
//!
//! So a settings document may declare that a key of it holds *parts* — see
//! `address::PARTED` — and each part gets an address. `settings/projection` is
//! still the whole file; `settings/projection/world` is the layer called
//! `world` inside it, and it lists, reads, and saves like anything else.
//!
//! # A part is edited in place, never re-serialised
//!
//! A save goes through [`crate::registry::yaml_edit::splice`] against the
//! *whole* document with just that part replaced. The splice compares all the
//! way down and patches only the scalars that moved, so changing one layer's
//! `window` is a one-line diff in a file whose other six hundred and ninety
//! lines — including every banner comment between the layers — are the exact
//! bytes they were.
//!
//! That is also why there is no add and no remove here. Either changes the
//! length of the list, which has no entry-for-entry correspondence left to walk
//! and so rewrites the list whole — taking the `# ── Environment ──` banners
//! with it. Adding a layer is an act for the whole document, where the author
//! can see the comments they are moving.

use serde_json::{Map, Value};

use crate::registry::yaml_edit;

/// Why a part could not be read or written.
#[derive(Debug)]
pub enum PartError {
    /// The document does not parse, or its root is not a mapping.
    Malformed,
    /// The document has no such list, or the list is not a list.
    NoList,
    /// No item of that list is named that.
    NotFound,
    /// The edit could not be made without rewriting the document.
    CannotPatch,
}

impl std::fmt::Display for PartError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PartError::Malformed => write!(f, "this document could not be read"),
            PartError::NoList => write!(f, "this document has no parts"),
            PartError::NotFound => write!(f, "there is nothing there"),
            PartError::CannotPatch => write!(
                f,
                "this part could not be edited without rewriting the document, \
                 which would lose its comments — edit it as text instead"
            ),
        }
    }
}

/// Every part of `list`, in the order the document writes them, with the name
/// each one goes by.
///
/// An item with no name, or a name that is not a string, is skipped rather than
/// given a made-up one: there would be no address for it, so offering a row
/// that cannot be opened is worse than leaving it to the text view.
pub fn list(text: &str, list_key: &str, id_key: &str) -> Result<Vec<(String, Value)>, PartError> {
    let doc: Value = serde_yaml::from_str(text).map_err(|_| PartError::Malformed)?;
    let items = doc
        .as_object()
        .ok_or(PartError::Malformed)?
        .get(list_key)
        .and_then(Value::as_array)
        .ok_or(PartError::NoList)?;
    Ok(items
        .iter()
        .filter_map(|item| {
            let name = item.get(id_key)?.as_str()?;
            Some((name.to_owned(), item.clone()))
        })
        .collect())
}

/// One part, as a document of its own.
pub fn read(text: &str, list_key: &str, id_key: &str, name: &str) -> Result<Value, PartError> {
    list(text, list_key, id_key)?
        .into_iter()
        .find(|(n, _)| n == name)
        .map(|(_, v)| v)
        .ok_or(PartError::NotFound)
}

/// The document `text` with one part replaced, changing nothing else.
///
/// The part keeps its position and its name: `next` is spliced over the item
/// that is there, and an attempt to rename it is refused — a name is the
/// address, and renaming through a form would move the thing out from under the
/// address that reached it.
pub fn write(
    text: &str,
    list_key: &str,
    id_key: &str,
    name: &str,
    next: &Map<String, Value>,
) -> Result<String, PartError> {
    let mut doc: Value = serde_yaml::from_str(text).map_err(|_| PartError::Malformed)?;
    let items = doc
        .as_object_mut()
        .ok_or(PartError::Malformed)?
        .get_mut(list_key)
        .and_then(Value::as_array_mut)
        .ok_or(PartError::NoList)?;
    let at = items
        .iter()
        .position(|item| item.get(id_key).and_then(Value::as_str) == Some(name))
        .ok_or(PartError::NotFound)?;

    let mut merged = next.clone();
    merged.insert(id_key.to_owned(), Value::String(name.to_owned()));
    items[at] = Value::Object(merged);

    let whole = doc.as_object().ok_or(PartError::Malformed)?;
    yaml_edit::splice(text, whole).ok_or(PartError::CannotPatch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Two layers with a banner comment above each, which is the shape that
    /// makes this module exist.
    const DOC: &str = r#"# The projection schema.
default_policy:
  preset: high_recall_scope

layers:
  # ── World ──────────────────────────────────────────────────────────────────
  - name: world
    description: |
      Shared knowledge about the setting.
    window: 8000
    score_threshold: 0.30
    budget:
      priority: 70
      max_percent: 20

  # ── Beliefs ────────────────────────────────────────────────────────────────
  - name: beliefs
    description: |
      What the character holds to be true.
    window: 4000
    score_threshold: 0.40
    budget:
      priority: 90
      max_percent: 15
"#;

    #[test]
    fn the_parts_are_listed_in_the_order_the_document_writes_them() {
        let parts = list(DOC, "layers", "name").expect("lists");
        let names: Vec<&str> = parts.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["world", "beliefs"]);
        assert_eq!(parts[0].1["window"], 8000);
    }

    #[test]
    fn a_part_reads_as_a_document_of_its_own() {
        let world = read(DOC, "layers", "name", "world").expect("reads");
        assert_eq!(world["name"], "world");
        assert_eq!(world["budget"]["priority"], 70);
        assert!(read(DOC, "layers", "name", "nope").is_err());
    }

    /// **The whole point.** Changing one field of one layer changes one line,
    /// and every banner comment is still where the author put it.
    #[test]
    fn saving_a_part_changes_that_field_and_leaves_the_comments() {
        let mut world = read(DOC, "layers", "name", "world").unwrap();
        world["window"] = json!(9000);
        let out = write(DOC, "layers", "name", "world", world.as_object().unwrap()).expect("saves");

        assert!(out.contains("# ── World ─"), "{out}");
        assert!(out.contains("# ── Beliefs ─"), "{out}");
        assert!(out.contains("# The projection schema."), "{out}");
        let before: Vec<&str> = DOC.lines().collect();
        let after: Vec<&str> = out.lines().collect();
        assert_eq!(after.len(), before.len(), "\n{out}");
        let moved: Vec<usize> = (0..before.len())
            .filter(|&i| before[i] != after[i])
            .collect();
        assert_eq!(moved.len(), 1, "changed lines {moved:?}:\n{out}");
        assert_eq!(after[moved[0]].trim(), "window: 9000");
        // And the other layer is untouched.
        assert_eq!(
            read(&out, "layers", "name", "beliefs").unwrap()["window"],
            4000
        );
    }

    /// The name is the address, so a save cannot change it. A form that could
    /// would move the document out from under the address that reached it.
    #[test]
    fn a_part_cannot_rename_itself() {
        let mut world = read(DOC, "layers", "name", "world").unwrap();
        world["name"] = json!("somewhere-else");
        world["window"] = json!(9000);
        let out = write(DOC, "layers", "name", "world", world.as_object().unwrap()).expect("saves");
        assert!(read(&out, "layers", "name", "world").is_ok(), "{out}");
        assert!(
            read(&out, "layers", "name", "somewhere-else").is_err(),
            "{out}"
        );
    }

    /// A nested value edits in place too — a budget is two levels down and
    /// still resolves to one line.
    #[test]
    fn a_nested_value_is_still_a_one_line_change() {
        let mut world = read(DOC, "layers", "name", "world").unwrap();
        world["budget"]["priority"] = json!(75);
        let out = write(DOC, "layers", "name", "world", world.as_object().unwrap()).expect("saves");
        let before: Vec<&str> = DOC.lines().collect();
        let after: Vec<&str> = out.lines().collect();
        let moved: Vec<usize> = (0..before.len())
            .filter(|&i| before[i] != after[i])
            .collect();
        assert_eq!(moved.len(), 1, "changed lines {moved:?}:\n{out}");
        assert_eq!(after[moved[0]].trim(), "priority: 75");
    }

    #[test]
    fn a_document_with_no_such_list_says_so() {
        assert!(matches!(
            list("a: 1\n", "layers", "name"),
            Err(PartError::NoList)
        ));
        assert!(matches!(
            list("- a\n- b\n", "layers", "name"),
            Err(PartError::Malformed)
        ));
    }

    /// An item with no name has no address, so it is left out rather than given
    /// a made-up one that could not be opened.
    #[test]
    fn an_unnamed_item_is_not_offered() {
        let parts = list(
            "layers:\n  - window: 10\n  - name: real\n",
            "layers",
            "name",
        )
        .unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].0, "real");
    }
}
