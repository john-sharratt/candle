//! A document as fields, so it can be edited without knowing YAML.
//!
//! The console used to show a section as its file: a textarea of raw YAML, with
//! the author responsible for indentation, block scalars, and not breaking the
//! `examples:` list. That is asking someone editing prose to also be a
//! serialisation format's proof-reader.
//!
//! So the document arrives as a list of [`Field`]s — a label, a kind, and a
//! value — and goes back the same way. The console renders a form; nothing on
//! either side needs to know what a block scalar is.
//!
//! # The comments are the reason this is not a full re-serialise
//!
//! **701 of the 712 section files carry comments**, and they are not decoration:
//!
//! ```text
//! # Provenance lead-ins — the context that PRODUCES the next (accepting) reply.
//! # FIXED SHAPE: 4 turns, user → assistant → user → assistant. Final assistant
//! # turn is the decode point: … Target: 16.
//! ```
//!
//! Parsing to a structure and writing it back out would delete every one of
//! them. So a save goes through [`crate::registry::yaml_edit::splice`], which
//! patches the values that changed and leaves the bytes around them exactly as
//! they were — and then re-reads its own output and refuses it if it does not
//! say what was asked.
//!
//! That is safe here for a reason worth writing down: **no file in the corpus
//! has a comment inside a value**, only above keys. Comments live between the
//! keys, so rewriting a value cannot reach them.
//!
//! # The note in the form is the author's own comment
//!
//! Those comments are the best documentation the corpus has, so the field that
//! follows one carries it as [`Field::note`] and the console shows it beside
//! the input. The guidance ends up where the editing happens instead of only in
//! the file.

use serde_json::{json, Map, Value};

/// How a value should be edited.
#[derive(Debug, Clone, PartialEq)]
pub enum Kind {
    /// A short string — one input.
    Line,
    /// A string with newlines in it, or a long one — a textarea.
    Text,
    /// A number — a numeric input, and a number on the way back. A layer's
    /// `window` typed into a text box comes back as the string `"8000"`, which
    /// is a different document.
    Number,
    /// True or false — a checkbox.
    Bool,
    /// One of a fixed set — a select.
    Choice(&'static [&'static str]),
    /// A list of short strings — add, remove, reorder.
    List,
    /// Conversations: a list of `{ note, turns: [{ role, content, thinking }] }`.
    /// The shape `examples:` has in every section file.
    Conversations,
    /// A mapping, as its own set of fields. A layer's `budget` is `priority`
    /// and `max_percent` and sometimes an `adaptive` inside that — three
    /// numbers, which is three numeric inputs and not a YAML box.
    Group(Vec<Field>),
    /// A list of mappings, each as its own set of fields — a layer's `groups`.
    Rows(Vec<Vec<Field>>),
    /// Anything this does not model. Edited as YAML, but only this value —
    /// never the whole document. An honest escape hatch beats a form that
    /// silently drops the half it did not understand.
    Raw,
}

impl Kind {
    fn as_str(&self) -> &'static str {
        match self {
            Kind::Line => "line",
            Kind::Text => "text",
            Kind::Number => "number",
            Kind::Bool => "bool",
            Kind::Choice(_) => "choice",
            Kind::List => "list",
            Kind::Conversations => "conversations",
            Kind::Group(_) => "group",
            Kind::Rows(_) => "rows",
            Kind::Raw => "raw",
        }
    }
}

/// One editable part of a document.
#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub key: String,
    pub label: String,
    pub kind: Kind,
    pub value: Value,
    /// The author's comment from directly above this key, if there is one.
    pub note: Option<String>,
    /// `id` is the address; changing it here would rename nothing and confuse
    /// everything.
    pub readonly: bool,
    /// For a [`Kind::Raw`] value only: the same value written as YAML, because
    /// that is what its escape-hatch editor shows. Sent rather than rendered in
    /// the browser so the text the author edits is the text this crate parses
    /// back — one serialiser, not two that can disagree.
    pub yaml: Option<String>,
}

impl Field {
    pub fn wire(&self) -> Value {
        json!({
            "key": self.key,
            "label": self.label,
            "kind": self.kind.as_str(),
            "value": self.value,
            "note": self.note,
            "readonly": self.readonly,
            "yaml": self.yaml,
            // Only whichever of these this kind has. A form renders one control
            // per field and reads the payload its kind names.
            "choices": match &self.kind {
                Kind::Choice(options) => json!(options),
                _ => Value::Null,
            },
            "fields": match &self.kind {
                Kind::Group(fields) => json!(fields.iter().map(Field::wire).collect::<Vec<_>>()),
                _ => Value::Null,
            },
            "rows": match &self.kind {
                Kind::Rows(rows) => json!(rows
                    .iter()
                    .map(|r| r.iter().map(Field::wire).collect::<Vec<_>>())
                    .collect::<Vec<_>>()),
                _ => Value::Null,
            },
        })
    }
}

/// Why a document could not be shown as fields.
#[derive(Debug)]
pub enum SectionError {
    /// Not YAML, or not a mapping at the top level. A list or a bare scalar has
    /// no fields to show.
    NotAMapping,
}

impl std::fmt::Display for SectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SectionError::NotAMapping => write!(
                f,
                "this document is not a set of fields, so it opens as text"
            ),
        }
    }
}

/// Read a document into fields, in the order the file writes them.
///
/// File order, not alphabetical: an author put `id`, `category`, `description`,
/// `template`, `examples` in that sequence because that is the order they are
/// understood in, and a form that sorted them would be re-teaching the document
/// to its own author.
/// `id_key` is the key that *is* the address — `id` for a section, `name` for a
/// projection layer. It is shown but not editable: changing it here would
/// rename nothing and confuse everything.
pub fn parse(text: &str, id_key: &str) -> Result<Vec<Field>, SectionError> {
    let doc: serde_yaml::Value =
        serde_yaml::from_str(text).map_err(|_| SectionError::NotAMapping)?;
    let map = doc.as_mapping().ok_or(SectionError::NotAMapping)?;

    let notes = comments(text);
    let mut out = Vec::new();
    for (k, _) in map {
        let Some(key) = k.as_str() else { continue };
        let value = to_json(&doc[k]);
        let mut field = field_for(key, value);
        field.note = notes.get(key).cloned();
        field.readonly = key == id_key;
        out.push(field);
    }
    Ok(out)
}

/// One field, from a key and its value. The nesting is here: a mapping becomes
/// a group of fields, and a list of mappings becomes rows of them, each built
/// the same way down to the scalars.
fn field_for(key: &str, value: Value) -> Field {
    let kind = kind_of(key, &value);
    Field {
        // Only the escape hatch needs the YAML text, and serialising every
        // value to a string the console will not show would be work done to be
        // discarded.
        yaml: (kind == Kind::Raw)
            .then(|| serde_yaml::to_string(&value).ok())
            .flatten(),
        kind,
        label: label_for(key),
        note: None,
        readonly: false,
        key: key.to_owned(),
        value,
    }
}

/// The fields of a mapping, in the order it writes them.
fn fields_of(map: &Map<String, Value>) -> Vec<Field> {
    map.iter().map(|(k, v)| field_for(k, v.clone())).collect()
}

/// What the console sends back, as the map [`splice`] wants.
///
/// Only keys the document already had, plus any the console added. Nothing is
/// dropped for being unrecognised: a `Raw` field round-trips through YAML text,
/// so a shape this module does not model still survives an edit to one beside
/// it.
///
/// [`splice`]: crate::registry::yaml_edit::splice
pub fn to_document(values: &Map<String, Value>) -> Result<Map<String, Value>, String> {
    let mut out = Map::new();
    for (k, v) in values {
        // A `Raw` field arrives as the YAML text the console showed. Parsing it
        // here means a malformed edit is refused with the key named, rather
        // than written out as a quoted string that silently changes the type.
        let parsed = match v {
            Value::Object(o) if o.len() == 1 && o.contains_key("__yaml") => {
                let text = o["__yaml"].as_str().unwrap_or_default();
                let y: serde_yaml::Value = serde_yaml::from_str(text)
                    .map_err(|e| format!("`{k}` is not valid YAML: {e}"))?;
                to_json(&y)
            }
            other => other.clone(),
        };
        out.insert(k.clone(), parsed);
    }
    Ok(out)
}

/// The comment block directly above each top-level key.
///
/// A plain scan rather than anything the YAML parser offers, because it does
/// not offer one: comments are not part of the data model, which is exactly why
/// re-serialising loses them. Only comments that sit immediately above a
/// top-level key count — a blank line between them ends the block, because a
/// comment separated by whitespace is about the file, not the field.
///
/// **The paragraphs survive.** An author wraps a comment across lines to fit
/// the file, so those line breaks are not meaning and are joined away — but a
/// bare `#` between two runs of them is a paragraph break the author typed, and
/// a bullet is a list item. Joining straight through both turns eight lines of
/// guidance into one unreadable wall, which is what the console showed the
/// first time anybody opened a response section.
fn comments(text: &str) -> std::collections::HashMap<String, String> {
    let mut out = std::collections::HashMap::new();
    let mut paragraphs: Vec<Vec<&str>> = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim_start();
        if line.starts_with('#') {
            let body = trimmed.trim_start_matches('#').trim();
            // A bare `#` ends the paragraph; a bullet starts its own.
            if body.is_empty() {
                paragraphs.push(Vec::new());
            } else if bullet(body) || paragraphs.is_empty() {
                paragraphs.push(vec![body]);
            } else {
                paragraphs.last_mut().expect("just checked").push(body);
            }
            continue;
        }
        if trimmed.is_empty() {
            paragraphs.clear();
            continue;
        }
        // A top-level key: no leading whitespace, and a colon in it.
        if !line.starts_with(char::is_whitespace) {
            if let Some((key, _)) = line.split_once(':') {
                let note = paragraphs
                    .iter()
                    .filter(|p| !p.is_empty())
                    .map(|p| p.join(" "))
                    .collect::<Vec<_>>()
                    .join("\n\n");
                if !note.is_empty() {
                    out.insert(key.trim().to_owned(), note);
                }
            }
        }
        paragraphs.clear();
    }
    out
}

/// A comment line that is a list item rather than a continuation of the one
/// above it.
fn bullet(body: &str) -> bool {
    body.starts_with("- ")
        || body.starts_with("* ")
        || body
            .split_once(". ")
            .is_some_and(|(n, _)| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()))
}

/// `blush_then_own` → `Blush then own`.
fn label_for(key: &str) -> String {
    let spaced = key.replace(['_', '-'], " ");
    let mut c = spaced.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

/// What kind of control this value wants.
fn kind_of(key: &str, v: &Value) -> Kind {
    match v {
        Value::String(s) => {
            if let Some(options) = choices(key, s) {
                return Kind::Choice(options);
            }
            // A newline or real length means prose, and prose in a one-line
            // input is unusable.
            if s.contains('\n') || s.chars().count() > 90 {
                Kind::Text
            } else {
                Kind::Line
            }
        }
        Value::Array(items) if items.is_empty() => Kind::List,
        Value::Array(items) if items.iter().all(|i| i.is_string()) => Kind::List,
        Value::Array(items) if items.iter().all(is_conversation) => Kind::Conversations,
        Value::Array(items) if items.iter().all(Value::is_object) => Kind::Rows(
            items
                .iter()
                .filter_map(|i| i.as_object().map(fields_of))
                .collect(),
        ),
        Value::Number(_) => Kind::Number,
        Value::Bool(_) => Kind::Bool,
        Value::Object(map) if !map.is_empty() => Kind::Group(fields_of(map)),
        _ => Kind::Raw,
    }
}

/// The fixed vocabularies, for the keys that have one.
///
/// These are the projection schema's own words — a layer gathers over a
/// `conversation` or over everything `shared`, and a typo in that is a layer
/// the engine cannot load. A select cannot be mistyped.
///
/// **Only offered when the value is already one of them.** The key names here
/// are ordinary words, and a document elsewhere with its own `kind` or `scope`
/// must not be told its value is invalid by a form that has never heard of it —
/// so an unrecognised value keeps its plain input and the vocabulary stays out
/// of the way.
fn choices(key: &str, value: &str) -> Option<&'static [&'static str]> {
    const VOCABULARY: [(&str, &[&str]); 4] = [
        ("gather_scope", &["conversation", "shared"]),
        ("decode_priority", &["low", "normal", "high"]),
        ("on_corrupt_turn", &["drop_turn", "drop_conversation"]),
        ("kind", &["conversation", "top_k"]),
    ];
    VOCABULARY
        .iter()
        .find(|(k, options)| *k == key && options.contains(&value))
        .map(|(_, options)| *options)
}

/// An item of `examples:` — an object with a `turns` array in it.
fn is_conversation(v: &Value) -> bool {
    v.get("turns").map(Value::is_array).unwrap_or(false)
}

/// `serde_yaml::Value` → `serde_json::Value`, through text.
///
/// The same reasoning as `yaml_edit::to_yaml` going the other way: two crates,
/// two `Value` types, and a hand-written match would be a second place for a
/// number's precision or a string's escaping to be decided.
fn to_json(v: &serde_yaml::Value) -> Value {
    serde_yaml::to_string(v)
        .ok()
        .and_then(|t| serde_yaml::from_str::<Value>(&t).ok())
        .unwrap_or(Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A raw string, because a `\`-continued literal strips the leading
    /// whitespace on each line — which in YAML is the structure.
    const REAL: &str = r#"id: accept_then_move_on
category: accept
description: Accepting what was offered, then letting the moment close.

# The frozen structural mode — its KV is loaded; the model decodes the NEXT turn
# into this once the section is selected.
template: |
  The tactical self is gone.
  What remains is the acceptance.

# Provenance lead-ins. FIXED SHAPE: 4 turns. Target: 16.
examples:
  - note: Late apology, no toll charged.
    turns:
      - role: user
        content: |
          "I'm late — sorry."
      - role: assistant
        thinking: |
          They will take it lightly.
"#;

    fn field<'a>(fs: &'a [Field], key: &str) -> &'a Field {
        fs.iter().find(|f| f.key == key).expect("field present")
    }

    #[test]
    fn a_document_becomes_fields_in_the_order_the_file_writes_them() {
        let fs = parse(REAL, "id").expect("parses");
        let keys: Vec<&str> = fs.iter().map(|f| f.key.as_str()).collect();
        assert_eq!(
            keys,
            ["id", "category", "description", "template", "examples"],
            "file order, not alphabetical"
        );
    }

    #[test]
    fn each_value_gets_the_control_it_wants() {
        let fs = parse(REAL, "id").unwrap();
        assert_eq!(field(&fs, "category").kind, Kind::Line);
        assert_eq!(field(&fs, "template").kind, Kind::Text, "it has newlines");
        assert_eq!(field(&fs, "examples").kind, Kind::Conversations);
        // A long single-line string is prose too, however it is stored.
        assert_eq!(field(&fs, "description").kind, Kind::Line);
        let long = parse(&format!("d: {}\n", "x".repeat(120)), "id").unwrap();
        assert_eq!(field(&long, "d").kind, Kind::Text);
        // A number is a number on the way back. Typed into a text box, a
        // layer's `window` returns as the string "8000", which is a different
        // document with the same appearance.
        let nums = parse("window: 8000\nthreshold: 0.3\npaused: false\n", "id").unwrap();
        assert_eq!(field(&nums, "window").kind, Kind::Number);
        assert_eq!(field(&nums, "threshold").kind, Kind::Number);
        assert_eq!(field(&nums, "paused").kind, Kind::Bool);
    }

    #[test]
    fn a_list_of_strings_is_a_list_and_anything_else_is_raw() {
        let fs = parse("selects:\n  - ammo\n  - armor\nempty: []\n", "id").unwrap();
        assert_eq!(field(&fs, "selects").kind, Kind::List);
        assert_eq!(field(&fs, "empty").kind, Kind::List);
        let plain = parse("a: b\n", "id").unwrap();
        assert_eq!(field(&plain, "a").yaml, None);
        // Only a shape with no structure to show falls through to YAML — a
        // list of mixed things, which is neither rows nor a list of strings.
        let odd = parse("mixed:\n  - a\n  - 2\n", "id").unwrap();
        assert_eq!(field(&odd, "mixed").kind, Kind::Raw);
        assert!(field(&odd, "mixed")
            .yaml
            .as_deref()
            .expect("raw carries its yaml")
            .contains("- a"));
    }

    /// **A mapping is a group of fields, not a YAML box.** A layer's `budget`
    /// is two or three numbers; showing it as text was the honest answer while
    /// nothing could render it, and is the wrong one now.
    #[test]
    fn a_mapping_becomes_its_own_fields_all_the_way_down() {
        let fs = parse(
            "budget:\n  priority: 70\n  adaptive:\n    gain: 2.0\n    max_percent: 40\n",
            "id",
        )
        .unwrap();
        let Kind::Group(inner) = &field(&fs, "budget").kind else {
            panic!("not a group: {:?}", field(&fs, "budget").kind);
        };
        assert_eq!(
            inner.iter().map(|f| f.key.as_str()).collect::<Vec<_>>(),
            ["adaptive", "priority"]
        );
        assert_eq!(field(inner, "priority").kind, Kind::Number);
        // And the nesting continues rather than stopping one level down.
        let Kind::Group(deeper) = &field(inner, "adaptive").kind else {
            panic!("nesting stopped");
        };
        assert_eq!(field(deeper, "gain").kind, Kind::Number);
        // A raw field still carries no yaml when it is not raw.
        assert_eq!(field(&fs, "budget").yaml, None);
    }

    /// A list of mappings — a layer's `groups` — is rows of fields.
    #[test]
    fn a_list_of_mappings_becomes_rows_of_fields() {
        let fs = parse(
            "groups:\n  - id: canon\n    budget:\n      priority: 100\n  - id: other\n    budget:\n      priority: 50\n",
            "id",
        )
        .unwrap();
        let Kind::Rows(rows) = &field(&fs, "groups").kind else {
            panic!("not rows: {:?}", field(&fs, "groups").kind);
        };
        assert_eq!(rows.len(), 2);
        assert_eq!(field(&rows[0], "id").value, "canon");
        let Kind::Group(budget) = &field(&rows[1], "budget").kind else {
            panic!("a row's mapping did not nest");
        };
        assert_eq!(field(budget, "priority").value, 50);
    }

    /// **A vocabulary the engine fixes becomes a select, but only where the
    /// value is already one of its words.**
    ///
    /// A typo in `gather_scope` is a layer the engine cannot load, and a select
    /// cannot be mistyped. But these are ordinary words: a document elsewhere
    /// with its own `kind` must not be told its value is invalid by a form that
    /// has never heard of it.
    #[test]
    fn a_fixed_vocabulary_becomes_a_choice_and_an_unknown_value_does_not() {
        let fs = parse(
            "gather_scope: shared\ndecode_priority: low\non_corrupt_turn: drop_turn\n",
            "name",
        )
        .unwrap();
        assert_eq!(
            field(&fs, "gather_scope").kind,
            Kind::Choice(&["conversation", "shared"])
        );
        assert_eq!(
            field(&fs, "decode_priority").kind,
            Kind::Choice(&["low", "normal", "high"])
        );
        assert!(matches!(
            field(&fs, "on_corrupt_turn").kind,
            Kind::Choice(_)
        ));
        // A word the vocabulary does not know keeps a plain input, rather than
        // a select that cannot express what the document already says.
        let other = parse("kind: something_else\ngather_scope: elsewhere\n", "id").unwrap();
        assert_eq!(field(&other, "kind").kind, Kind::Line);
        assert_eq!(field(&other, "gather_scope").kind, Kind::Line);
    }

    /// The id key is whichever key is the address. A section is keyed by `id`
    /// and a projection layer by `name`, and both are shown but not editable.
    #[test]
    fn whichever_key_is_the_address_is_the_readonly_one() {
        let fs = parse("name: world\nid: not-the-address\n", "name").unwrap();
        assert!(field(&fs, "name").readonly);
        assert!(!field(&fs, "id").readonly);
    }

    /// The conversation shape, which is what the examples editor is built on.
    #[test]
    fn the_examples_arrive_as_conversations_with_their_turns() {
        let fs = parse(REAL, "id").unwrap();
        let ex = field(&fs, "examples");
        let items = ex.value.as_array().unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["note"], "Late apology, no toll charged.");
        let turns = items[0]["turns"].as_array().unwrap();
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0]["role"], "user");
        assert!(turns[0]["content"].as_str().unwrap().contains("I'm late"));
        // The decode point: thinking and no content.
        assert_eq!(turns[1]["role"], "assistant");
        assert!(turns[1].get("content").is_none());
        assert!(turns[1]["thinking"].as_str().unwrap().contains("lightly"));
    }

    /// **The author's own comment becomes the field's note.** This is the best
    /// documentation the corpus has, and it belongs where the editing happens.
    #[test]
    fn the_comment_above_a_key_becomes_that_fields_note() {
        let fs = parse(REAL, "id").unwrap();
        let t = field(&fs, "template").note.as_deref().unwrap_or_default();
        assert!(t.contains("frozen structural mode"), "{t}");
        assert!(
            t.contains("decodes the NEXT turn"),
            "joined across lines: {t}"
        );
        let e = field(&fs, "examples").note.as_deref().unwrap_or_default();
        assert!(e.contains("FIXED SHAPE"), "{e}");
        // A key with nothing above it has no note.
        assert_eq!(field(&fs, "id").note, None);
        assert_eq!(field(&fs, "category").note, None);
    }

    /// A comment separated by a blank line is about the file, not the next key.
    #[test]
    fn a_detached_comment_is_not_attached_to_the_next_field() {
        let fs = parse("# about the file\n\nid: x\ncategory: y\n", "id").unwrap();
        assert_eq!(field(&fs, "id").note, None);
    }

    /// **The author's paragraphs survive the joining.**
    ///
    /// Wrapping is not meaning — an author breaks a line to fit the file, so
    /// those breaks are joined away. A bare `#` between two runs is a paragraph
    /// the author typed, and a bullet is a list item. The real note above
    /// `examples:` is eight lines and two paragraphs; joining straight through
    /// showed it as one unreadable wall.
    #[test]
    fn a_note_keeps_the_paragraphs_and_bullets_the_author_wrote() {
        let fs = parse(
            concat!(
                "# Provenance lead-ins — the context that PRODUCES the reply. FIXED SHAPE:\n",
                "# 4 turns, user → assistant → user → assistant.\n",
                "#\n",
                "# POV: user turns first-person; assistant beat third-person.\n",
                "# Scene third parties keep their own pronouns.\n",
                "examples: []\n",
            ),
            "id",
        )
        .unwrap();
        let note = field(&fs, "examples").note.as_deref().unwrap();
        assert_eq!(
            note,
            "Provenance lead-ins — the context that PRODUCES the reply. FIXED SHAPE: \
             4 turns, user → assistant → user → assistant.\n\n\
             POV: user turns first-person; assistant beat third-person. \
             Scene third parties keep their own pronouns."
        );

        // A bullet is its own paragraph, so a list does not run together.
        let fs = parse(
            concat!(
                "# A collection is one of:\n",
                "#   - a MOOD collection — re-evaluated at every barrier,\n",
                "#     spiking register\n",
                "#   - a RESPONSE collection — locked once\n",
                "items: []\n",
            ),
            "id",
        )
        .unwrap();
        assert_eq!(
            field(&fs, "items").note.as_deref().unwrap(),
            "A collection is one of:\n\n\
             - a MOOD collection — re-evaluated at every barrier, spiking register\n\n\
             - a RESPONSE collection — locked once"
        );
    }

    #[test]
    fn the_id_is_readonly_because_it_is_the_address() {
        let fs = parse(REAL, "id").unwrap();
        assert!(field(&fs, "id").readonly);
        assert!(!field(&fs, "category").readonly);
        assert!(!field(&fs, "examples").readonly);
    }

    #[test]
    fn a_document_that_is_not_a_mapping_has_no_fields() {
        for bad in ["- a\n- b\n", "just a string\n", "[1, 2]\n"] {
            assert!(
                matches!(parse(bad, "id"), Err(SectionError::NotAMapping)),
                "{bad}"
            );
        }
    }

    #[test]
    fn values_come_back_as_a_document_map() {
        let mut vals = Map::new();
        vals.insert("category".into(), json!("accept"));
        vals.insert("examples".into(), json!([{ "note": "n", "turns": [] }]));
        let out = to_document(&vals).expect("builds");
        assert_eq!(out["category"], "accept");
        assert_eq!(out["examples"][0]["note"], "n");
    }

    /// **A form opened and saved with nothing typed into it must produce the
    /// document it was given.**
    ///
    /// This is the property the whole editor rests on: the values that come out
    /// of [`parse`] have to go back through [`to_document`] as *exactly* the
    /// document the file already holds, or the splice sees a change in a field
    /// nobody touched and rewrites it. Block scalars are where that goes wrong —
    /// `|` keeps a trailing newline and `>-` folds — so the comparison is
    /// against the file's own parse, byte for byte in the values.
    #[test]
    fn opening_and_saving_without_typing_changes_nothing() {
        let on_disk: Value = serde_yaml::from_str(REAL).unwrap();
        let mut vals = Map::new();
        for f in parse(REAL, "id").unwrap() {
            let v = match (f.kind, f.yaml.as_deref()) {
                (Kind::Raw, Some(y)) => json!({ "__yaml": y }),
                _ => f.value.clone(),
            };
            vals.insert(f.key.clone(), v);
        }
        let back = to_document(&vals).expect("builds");
        assert_eq!(Value::Object(back), on_disk, "a no-op save changed a value");
    }

    /// A `Raw` field is YAML text on the way out and a value on the way back,
    /// so a shape this does not model survives an edit to one beside it.
    #[test]
    fn a_raw_field_is_parsed_back_from_its_yaml() {
        let mut vals = Map::new();
        vals.insert(
            "budget".into(),
            json!({ "__yaml": "priority: 3\nfloor: 10\n" }),
        );
        let out = to_document(&vals).expect("builds");
        assert_eq!(out["budget"]["priority"], 3);
        assert_eq!(out["budget"]["floor"], 10);
    }

    /// Malformed YAML in a raw field names the key rather than being written
    /// out as a quoted string that silently changes the value's type.
    #[test]
    fn a_malformed_raw_field_is_refused_with_its_key() {
        let mut vals = Map::new();
        vals.insert("budget".into(), json!({ "__yaml": "a: [1, 2\n" }));
        let err = to_document(&vals).expect_err("refused");
        assert!(err.contains("budget"), "{err}");
    }
}
