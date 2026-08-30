//! Saving an authored YAML file without rewriting it.
//!
//! # The problem this exists for
//!
//! An authored world or personality carries its reasoning in its comments. The
//! header on `worlds/sandbox.yaml` explains why the setting text says what IS
//! there rather than what is absent; the one on a personality explains why
//! biography is not in the file. That is the most valuable content in the
//! document and none of it is data.
//!
//! `serde_yaml` cannot see comments — it parses to a `Value` and can only
//! serialise a whole document back — so a registry that saved with it turned
//! every commented, block-scalar file into a flat list of quoted scalars the
//! first time anybody pressed Save in the console. Not a crash, not a warning:
//! a file that still loads perfectly and has lost the half of itself a person
//! wrote.
//!
//! # What this does instead
//!
//! It edits, through [`yamlpatch`], which works over a tree-sitter concrete
//! syntax tree where a comment is a node like any other. The original text is
//! the base and only the values that actually changed are replaced; every other
//! byte — comments, blank lines, block scalars, key order, the author's line
//! wrapping — is carried through untouched. A save that changes `name` changes
//! one line.
//!
//! `yamlpatch` and `yamlpath` come out of `zizmor`, which rewrites people's
//! checked-in GitHub Actions workflows and therefore has exactly this
//! requirement: the diff has to be reviewable by the person whose file it is.
//!
//! # Why this module is a diff rather than a thin wrapper
//!
//! The registry's API replaces a whole document — a `PUT` hands over the new
//! state — while `yamlpatch` takes operations. So the work here is deciding
//! *which* keys actually changed and emitting one op each: a `Replace` for a
//! changed value, an `Add` for a new key, a `Remove` for a departed one, and
//! nothing at all for the keys that match, which is the common case and the
//! reason a save is usually a one-line diff.
//!
//! # Why the result is verified before it is returned
//!
//! Editing text is a place to be wrong quietly. Rather than trust the op set,
//! [`splice`] parses what came out and compares it to what was asked for; if
//! they differ by so much as a field it discards the result and the caller
//! falls back to a plain full serialisation. The failure mode is then losing
//! comments — which is where this module started — and never a file that says
//! something the author did not.

use serde_json::{Map, Value};
use subfeature::{Fragment, Subfeature};
use yamlpatch::{Op, Patch};
use yamlpath::{Document, Route};

/// Rewrite `original` so it holds `next`, changing as little as possible.
///
/// Keys present in both and unchanged keep their exact original bytes. Changed
/// values are replaced in place. Keys only in `next` are added; keys only in
/// `original` are removed.
///
/// Returns `None` when the original cannot be edited — it does not parse, it is
/// not a top-level mapping, or the edit failed its own read-back check — and
/// the caller should fall back to serialising the document whole.
pub fn splice(original: &str, next: &Map<String, Value>) -> Option<String> {
    match edit(original, next) {
        Ok(out) => Some(out),
        Err(why) => {
            // Why it could not be edited, at debug. The caller logs the
            // consequence — the file is about to be rewritten whole and its
            // comments lost — but not the cause, and "which document, and what
            // about it" is the question anybody investigating that will have.
            tracing::debug!("yaml edit declined: {why}");
            None
        }
    }
}

/// The edit, with its reason for declining.
fn edit(original: &str, next: &Map<String, Value>) -> Result<String, String> {
    // A document whose root is not a mapping has no top-level keys to route to.
    // `Value::Object` is the only shape the registry writes, so this is a
    // corrupt or hand-made file rather than a case to support.
    let current: Value =
        serde_yaml::from_str(original).map_err(|e| format!("does not parse: {e}"))?;
    let current = current
        .as_object()
        .ok_or_else(|| "the document root is not a mapping".to_string())?;

    let doc = Document::new(original).map_err(|e| format!("tree-sitter parse: {e}"))?;
    let patches = plan(&doc, current, next)?;

    // No patch means no change: hand back the original bytes rather than
    // re-emitting them. `apply_yaml_patches` refuses an empty patch set anyway,
    // and an idle Save must not reformat the repository.
    if patches.is_empty() {
        return Ok(original.to_string());
    }

    let edited = yamlpatch::apply_yaml_patches(&doc, &patches)
        .map_err(|e| format!("applying {} patch(es): {e}", patches.len()))?;
    let out = edited.source().to_string();

    // The read-back check. An op set that did something other than what it said
    // must not be able to change what the file means.
    let back: Value =
        serde_yaml::from_str(&out).map_err(|e| format!("the edit did not parse back: {e}"))?;
    if back != Value::Object(next.clone()) {
        return Err("the edit changed the document's meaning".to_string());
    }
    Ok(out)
}

/// One op per key that actually differs, in a stable order.
///
/// `None` when a value cannot be expressed as a replacement — the caller then
/// serialises whole rather than writing a partial edit.
fn plan<'a>(
    doc: &'a Document,
    current: &Map<String, Value>,
    next: &'a Map<String, Value>,
) -> Result<Vec<Patch<'a>>, String> {
    let mut patches = Vec::new();

    for (key, want) in next {
        match current.get(key) {
            // Unchanged. The whole point: no op, so the bytes are untouched and
            // the author's block scalar, wrapping and inline comments survive
            // exactly as written.
            Some(have) if have == want => {}
            Some(_) if is_collection(want) => patches.push(Patch {
                route: route(key),
                operation: rewrite_collection(doc, key, want)?,
            }),
            Some(_) => patches.push(Patch {
                route: route(key),
                operation: Op::Replace(to_yaml(want)?),
            }),
            None => patches.push(Patch {
                // `Add` routes to the *mapping* that gains the key, which for a
                // top-level key is the document root — an empty route.
                route: Route::from(vec![]),
                operation: Op::Add {
                    key: key.clone(),
                    value: to_yaml(want)?,
                },
            }),
        }
    }

    // Departures. A `PUT` replaces the document, so a key the caller did not
    // send is one they removed — the same rule the console relies on to drop a
    // field, and indistinguishable here from a field they never knew about.
    for key in current.keys() {
        if !next.contains_key(key) {
            patches.push(Patch {
                route: route(key),
                operation: Op::Remove,
            });
        }
    }

    Ok(patches)
}

/// Whether a value is a sequence or mapping, which `Op::Replace` cannot render
/// in place — see [`rewrite_collection`].
fn is_collection(v: &Value) -> bool {
    matches!(v, Value::Array(_) | Value::Object(_))
}

/// Replace a collection by rewriting its value text, keeping the layout style
/// the author used.
///
/// `Op::Replace` renders a value with `yaml_serde::to_string` and joins it onto
/// the key's line. For a scalar that is right; for a populated collection it
/// produces
///
/// ```text
/// selects: - north
/// - hill-villages
/// ```
///
/// which is not YAML, and the patch fails with "input is not valid YAML". It is
/// the sequence limitation the crate documents, and every world in the mind has
/// a `selects` list, so it is squarely on the path rather than a corner.
///
/// `RewriteFragment` replaces the value's own text instead, which means this
/// module chooses the rendering — so it renders the way the file already reads.
/// A `[]` stays flow and a `- item` list stays a block, because an edit that
/// reflowed a forty-six entry list onto one line would bury the one entry that
/// changed under a diff of the whole file. That is the same reason the whole
/// module exists.
fn rewrite_collection<'a>(doc: &'a Document, key: &str, want: &Value) -> Result<Op<'a>, String> {
    // `query_exact`, not `query_pretty`: `RewriteFragment` searches within the
    // *exact* feature, which for a key route is the value alone. A `from` that
    // included the key would never match, and the patch fails with "no match
    // for … in feature" rather than doing something wrong quietly.
    let feature = doc
        .query_exact(&route(key))
        .map_err(|e| format!("locating `{key}`: {e}"))?
        .ok_or_else(|| format!("`{key}` has no value to rewrite"))?;
    // Borrowed from the document rather than copied, because `Fragment` keeps
    // the borrow and the patch outlives this call.
    let current = doc.extract(&feature);

    let to = match block_items(want) {
        // A block list, at the indent its neighbours already use.
        Some(items) if !items.is_empty() && !is_flow(current) => {
            let indent = block_indent(current);
            let mut out = String::new();
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push('\n');
                    out.push_str(&indent);
                }
                out.push_str("- ");
                out.push_str(item.trim_end());
            }
            out
        }
        // Flow: what the file already used, and the only form an empty
        // collection or a mapping has here.
        _ => yamlpatch::serialize_flow(&to_yaml(want)?)
            .map_err(|e| format!("flow-rendering `{key}`: {e}"))?,
    };

    Ok(Op::RewriteFragment {
        // `Fragment::new` turns whitespace into `\s+`, so a multi-line block
        // sequence matches its own text across the line breaks.
        from: Subfeature::new(0, Fragment::new(current)),
        to: to.into(),
    })
}

/// Whether the existing value text is written in flow style.
fn is_flow(current: &str) -> bool {
    let t = current.trim_start();
    t.starts_with('[') || t.starts_with('{')
}

/// A sequence's entries, each rendered as a single-line YAML scalar, or `None`
/// when the value is not a sequence of scalars.
///
/// Only flat sequences take the block treatment. A list of mappings is rare
/// here and renders correctly in flow, so it takes that path rather than
/// growing an indentation-aware renderer this file does not otherwise need.
fn block_items(v: &Value) -> Option<Vec<String>> {
    let arr = v.as_array()?;
    if arr.iter().any(is_collection) {
        return None;
    }
    arr.iter()
        .map(|item| {
            yamlpatch::serialize_flow(&to_yaml(item).ok()?)
                .ok()
                .map(|s| s.trim().to_string())
        })
        .collect()
}

/// The indent a block sequence's items sit at, read from the text being
/// replaced so a rewrite lines up with what is already there.
///
/// The first item carries no indent of its own — it follows the key's newline —
/// so the answer comes from a continuation line, and two spaces is the fallback
/// for a one-item list that has none.
fn block_indent(current: &str) -> String {
    current
        .lines()
        .skip(1)
        .find(|l| !l.trim().is_empty())
        .map(|l| l.chars().take_while(|c| *c == ' ').collect::<String>())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "  ".to_string())
}

/// The route to a top-level key.
fn route(key: &str) -> Route<'static> {
    Route::from(vec![yamlpath::Component::Key(key.to_owned().into())])
}

/// Convert a JSON value into the YAML value `yamlpatch` replaces with.
///
/// Through text rather than field by field: `yaml_serde` and `serde_json` are
/// different crates with different `Value` types, and a hand-written match over
/// both would be a second place for a number's precision or a string's escaping
/// to be decided.
fn to_yaml(v: &Value) -> Result<yaml_serde::Value, String> {
    let text = serde_yaml::to_string(v).map_err(|e| format!("serialising a value: {e}"))?;
    yaml_serde::from_str(&text).map_err(|e| format!("re-reading a value: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn obj(v: Value) -> Map<String, Value> {
        v.as_object().unwrap().clone()
    }

    /// The whole reason the module exists: the comments have to come back.
    #[test]
    fn a_changed_value_leaves_every_comment_where_it_was() {
        let original = "\
# Sandbox -- Elysia's world.
#
# It is not a deception and it is not a test chamber.

id: sandbox
name: Sandbox
public: false

# Nothing yet: the sandbox has no shared canon to admit.
selects: []
";
        let out = splice(
            original,
            &obj(json!({
                "id": "sandbox", "name": "Sandbox Prime",
                "public": false, "selects": []
            })),
        )
        .expect("editable");

        assert!(out.contains("# Sandbox -- Elysia's world."), "{out}");
        assert!(out.contains("# It is not a deception"), "{out}");
        assert!(out.contains("# Nothing yet:"), "{out}");
        assert!(out.contains("Sandbox Prime"), "{out}");
        assert!(
            !out.contains("name: Sandbox\n"),
            "the old name survived:\n{out}"
        );
    }

    /// A save that changes nothing must produce the file it was given, byte for
    /// byte. Anything else means an idle Save reformats the repository.
    #[test]
    fn an_unchanged_document_is_returned_unchanged() {
        let original = "\
# header
id: sandbox

# why this is here
setting: >-
  Your world is small and complete. A room, a view from it,
  and time that goes on passing.

selects: []
";
        let parsed: Value = serde_yaml::from_str(original).unwrap();
        let out = splice(original, &obj(parsed)).expect("editable");
        assert_eq!(out, original);
    }

    /// **The case that decided whether this crate could be used at all.**
    ///
    /// Every world in the mind carries `selects: []` — an empty *flow* sequence
    /// — and the survey that recommended these crates reported `Op::Replace`
    /// failing on sequences and flow style being rejected elsewhere. If that
    /// applied here the port was off, so it is pinned rather than assumed.
    #[test]
    fn a_flow_sequence_can_be_replaced_and_emptied() {
        let original = "# hdr\nname: Ardh\nselects: []\n";
        let out = edit(
            original,
            &obj(json!({ "name": "Ardh", "selects": ["north", "hill-villages"] })),
        )
        .expect("a flow sequence refused a Replace");
        let back: Value = serde_yaml::from_str(&out).unwrap();
        assert_eq!(back["selects"], json!(["north", "hill-villages"]));
        assert!(out.contains("# hdr"), "{out}");

        // And back to empty, which is the direction an author undoing a change
        // takes.
        let out2 = splice(&out, &obj(json!({ "name": "Ardh", "selects": [] })))
            .expect("emptying a sequence refused");
        let back: Value = serde_yaml::from_str(&out2).unwrap();
        assert_eq!(back["selects"], json!([]));
        assert!(out2.contains("# hdr"), "{out2}");
    }

    /// A block sequence — what `selects` becomes once it has entries — takes an
    /// edit too, and the comment introducing it stays put.
    #[test]
    fn a_block_sequence_can_be_replaced() {
        let original = "\
name: Battle Cities

# The tags this world admits.
selects:
  - combat
  - lore
";
        let out = splice(
            original,
            &obj(json!({ "name": "Battle Cities", "selects": ["combat", "lore", "towers"] })),
        )
        .expect("editable");
        let back: Value = serde_yaml::from_str(&out).unwrap();
        assert_eq!(back["selects"], json!(["combat", "lore", "towers"]));
        assert!(out.contains("# The tags this world admits."), "{out}");
        // And it is still a block list at the same indent, not a flow line. A
        // forty-six entry `selects` reflowed onto one line would bury the one
        // entry that changed under a diff of the whole file.
        assert!(
            out.contains("\n  - combat\n"),
            "lost the block style:\n{out}"
        );
        assert!(out.contains("\n  - towers\n"), "{out}");
        assert!(!out.contains('['), "collapsed to flow:\n{out}");
    }

    /// The other direction: a list the author wrote in flow stays in flow.
    #[test]
    fn a_flow_list_stays_flow_when_it_gains_entries() {
        let out = splice(
            "name: Ardh\nselects: [north]\n",
            &obj(json!({ "name": "Ardh", "selects": ["north", "south"] })),
        )
        .expect("editable");
        assert!(out.contains("selects: [north, south]"), "{out}");
    }

    /// Multi-line prose keeps its shape rather than collapsing into one quoted
    /// line of `\n` escapes.
    #[test]
    fn multi_line_text_stays_readable() {
        let original = "doctrine: >-\n  One line.\n";
        let out = splice(
            original,
            &obj(json!({ "doctrine": "First line.\nSecond line." })),
        )
        .expect("editable");
        let back: Value = serde_yaml::from_str(&out).unwrap();
        assert_eq!(back["doctrine"], json!("First line.\nSecond line."));
    }

    /// A trailing newline is part of the value, and the block indicator is the
    /// only thing that says so.
    #[test]
    fn chomping_preserves_whether_the_value_ends_in_a_newline() {
        for text in ["a\nb\n", "a\nb"] {
            let out = splice("k: x\n", &obj(json!({ "k": text }))).expect("editable");
            let back: Value = serde_yaml::from_str(&out).unwrap();
            assert_eq!(back["k"], json!(text), "{out}");
        }
    }

    #[test]
    fn a_new_key_is_added_and_the_rest_is_untouched() {
        let original = "# header\nname: Ardh\n";
        let out = splice(
            original,
            &obj(json!({ "name": "Ardh", "public": true, "selects": ["north"] })),
        )
        .expect("editable");

        assert!(out.contains("# header"), "{out}");
        let back: Value = serde_yaml::from_str(&out).unwrap();
        assert_eq!(back["name"], json!("Ardh"));
        assert_eq!(back["public"], json!(true));
        assert_eq!(back["selects"], json!(["north"]));
    }

    /// A `PUT` replaces the document, so a key the caller did not send is gone.
    /// That is the rule the console relies on to remove a field.
    ///
    /// The comment introducing a removed key **stays**. A person wrote it, and
    /// it may explain why the field went or be about to introduce its
    /// replacement; deleting prose because a neighbouring key left is the
    /// destructive reading of "replace the document". The hand-rolled editor
    /// this replaced took the comment with the key — a deliberate change, not a
    /// regression.
    #[test]
    fn a_key_the_caller_did_not_send_is_removed_and_its_comment_kept() {
        let original = "\
# The file header, which belongs to no key.
id: ardh

# Why zoom bands are declared per world.
zoom_bands: [local]
name: Ardh
";
        let out =
            splice(original, &obj(json!({ "id": "ardh", "name": "Ardh" }))).expect("editable");

        assert!(out.contains("# The file header"), "{out}");
        assert!(
            out.contains("# Why zoom bands"),
            "the comment went too:\n{out}"
        );
        let back: Value = serde_yaml::from_str(&out).unwrap();
        assert_eq!(back, json!({ "id": "ardh", "name": "Ardh" }), "{out}");
    }

    /// Nested structure round-trips, including through a replacement.
    #[test]
    fn a_nested_collection_survives_a_replacement() {
        let out = splice(
            "name: Ardh\ntime: {}\n",
            &obj(json!({ "name": "Ardh", "time": { "scale": 60, "paused": false } })),
        )
        .expect("editable");
        let back: Value = serde_yaml::from_str(&out).unwrap();
        assert_eq!(back["time"]["scale"], json!(60));
        assert_eq!(back["time"]["paused"], json!(false));
        assert_eq!(back["name"], json!("Ardh"));
    }

    /// Values that need quoting get it, via the serialiser rather than by hand.
    #[test]
    fn awkward_scalars_survive_the_round_trip() {
        for s in [
            "- not a list",
            "yes",
            "no",
            "null",
            "123",
            "1.5",
            "#not a comment",
            "a: colon",
            "trailing ",
            "  leading",
            "",
            "*alias",
            "@reserved",
            "emoji 🙂 and — dashes",
        ] {
            let Some(out) = splice("k: placeholder\n", &obj(json!({ "k": s }))) else {
                panic!("refused `{s}`");
            };
            let back: Value = serde_yaml::from_str(&out).unwrap();
            assert_eq!(back["k"], json!(s), "`{s}` came back wrong:\n{out}");
        }
    }

    /// Not every document can be edited, and the honest answer is to say so and
    /// let the caller serialise it whole.
    #[test]
    fn documents_this_cannot_edit_are_refused_rather_than_mangled() {
        for bad in [
            "- a\n- b\n",        // a sequence at the root
            "just a scalar\n",   // no mapping at all
            "",                  // empty
            "name: [unclosed\n", // does not parse
        ] {
            assert!(
                splice(bad, &obj(json!({ "name": "x" }))).is_none(),
                "accepted `{bad}`"
            );
        }
    }

    /// **Injection.** The values here are free prose arriving in an HTTP body —
    /// a world's `setting`, a personality's `doctrine` — and a key can be
    /// anything a JSON object may hold.
    ///
    /// The read-back check is what makes an escape unreachable rather than the
    /// serialisation being careful: whatever is produced is parsed and compared
    /// to the requested document, so anything that wrote an extra key or
    /// changed a value fails the comparison and the whole edit is discarded.
    /// This test exists to keep that true — a future "skip the check when
    /// nothing looks suspicious" reopens it.
    #[test]
    fn a_crafted_key_or_value_cannot_change_the_document() {
        let keys = [
            "a\nevil: injected",
            "a: b",
            "#comment",
            "a\n\n---\nevil: doc",
            "? explicit",
            "a\r\nevil: crlf",
            "\u{85}evil",
            "\u{2028}evil",
        ];
        for key in keys {
            let want = obj(json!({ "name": "Ardh", key.to_string(): "x" }));
            let Some(out) = splice("name: Ardh\n", &want) else {
                continue; // refused outright, which is also a correct answer
            };
            let back: Value = serde_yaml::from_str(&out)
                .unwrap_or_else(|e| panic!("unparseable for `{key:?}`: {e}\n{out}"));
            assert_eq!(back, Value::Object(want), "`{key:?}` changed it:\n{out}");
        }

        let values = [
            "harmless\nevil: injected",
            "x\n---\nevil: second document",
            "x\n...\nevil: after terminator",
            "\nevil: leading newline",
            "x\n  evil: indented",
            "*alias",
            "&anchor value",
            "!!python/object/apply:os.system ['calc']",
            "\u{0}embedded nul",
        ];
        for evil in values {
            let want = obj(json!({ "name": "Ardh", "setting": evil }));
            let Some(out) = splice("name: Ardh\nsetting: old\n", &want) else {
                continue;
            };
            let back: Value = serde_yaml::from_str(&out)
                .unwrap_or_else(|e| panic!("unparseable for {evil:?}: {e}\n{out}"));
            assert_eq!(back, Value::Object(want), "{evil:?} changed it:\n{out}");
            assert_eq!(
                back.as_object().unwrap().len(),
                2,
                "{evil:?} added a key:\n{out}"
            );
        }
    }

    /// A tag in the text must stay text. `serde_yaml` does not construct
    /// arbitrary types from tags the way some loaders do, and this pins that.
    #[test]
    fn a_value_is_data_and_never_a_constructed_type() {
        let want = obj(json!({ "k": "!!str not a tag" }));
        let out = splice("k: x\n", &want).expect("editable");
        let back: Value = serde_yaml::from_str(&out).unwrap();
        assert_eq!(back["k"], json!("!!str not a tag"), "{out}");
    }

    /// The read-back check is the safety net under everything above. Whatever
    /// comes out must parse to exactly what was asked for.
    #[test]
    fn the_result_always_parses_back_to_what_was_asked_for() {
        let original = "# hdr\nname: Ardh\nsetting: >-\n  Some prose here.\nselects: []\n";
        let want = json!({
            "name": "Låg Fen — låg",
            "setting": "Line one.\n\nLine three.\n",
            "selects": ["north", "hill-villages"],
            "public": true,
            "nested": { "a": [1, 2, { "b": null }] },
        });
        let out = splice(original, &obj(want.clone())).expect("editable");
        let back: Value = serde_yaml::from_str(&out).unwrap();
        assert_eq!(back, want, "\n{out}");
    }
}
