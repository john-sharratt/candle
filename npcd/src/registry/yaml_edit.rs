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
//! *what* actually changed and emitting one op each: a `Replace` for a changed
//! value, an `Add` for a new key, a `Remove` for a departed one, and nothing at
//! all for what matches, which is the common case and the reason a save is
//! usually a one-line diff.
//!
//! That comparison runs all the way down. A section's `examples:` is sixteen
//! conversations of four turns each, and editing the wording of one turn is a
//! change to `examples[1].turns[0].content` — so that is where the patch routes
//! and the diff is that block scalar. Stopping at the top-level keys instead
//! would replace the value of `examples`, rewriting all sixteen conversations
//! to change one line of one of them.
//!
//! # What happens when the shape changes
//!
//! A sequence that gained or lost an entry has no entry-for-entry
//! correspondence to walk — entry 3 of the new list is not entry 3 of the old
//! one — so that collection is rewritten whole. It is rewritten as *block*
//! YAML, in the key order the file already used, with prose as the literal
//! blocks it was written as and bare scalars left bare. The alternative is one
//! four-thousand-character flow line: correct YAML, and the end of the file's
//! life as something a person reads. Getting this right is what makes "add an
//! example" a diff of the added lines rather than of the file.
//!
//! # Why the result is verified before it is returned
//!
//! Editing text is a place to be wrong quietly. Rather than trust the op set,
//! [`splice`] parses what came out and compares it to what was asked for; if
//! they differ by so much as a field it discards the result and answers `None`.
//! The caller then refuses the save rather than rewriting the document whole —
//! that fallback would cost the file its comments, which is where this module
//! started. A file that says something the author did not is never a possible
//! outcome.

use serde_json::{Map, Value};
use subfeature::{Fragment, Subfeature};
use yamlpatch::{Op, Patch};
use yamlpath::{Component, Document, Route};

/// Rewrite `original` so it holds `next`, changing as little as possible.
///
/// Anything present in both and unchanged keeps its exact original bytes, at
/// any depth. Changed values are replaced in place; keys only in `next` are
/// added and keys only in `original` are removed.
///
/// Returns `None` when the original cannot be edited — it does not parse, it is
/// not a top-level mapping, or the edit failed its own read-back check. The
/// caller must then refuse the save: serialising the document whole is what
/// this exists to prevent.
pub fn splice(original: &str, next: &Map<String, Value>) -> Option<String> {
    match edit(original, next) {
        Ok(out) => Some(out),
        Err(why) => {
            // Why it could not be edited, at debug. The caller reports the
            // consequence — the save is refused — but not the cause, and
            // "which document, and what about it" is the question anybody
            // investigating a refusal will have.
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

    // The same document a second time, as YAML rather than JSON — `Mapping`
    // keeps its keys in the order the file writes them and `serde_json::Map`
    // does not. Nothing is *read* from it; it is the shape a rewritten
    // collection is rendered to match, so an edit comes out looking like the
    // file it went into. See [`block_lines`].
    let shape: serde_yaml::Value =
        serde_yaml::from_str(original).map_err(|e| format!("does not parse: {e}"))?;

    let doc = Document::new(original).map_err(|e| format!("tree-sitter parse: {e}"))?;
    let (patches, touched) = plan(
        &doc,
        &Value::Object(current.clone()),
        &Value::Object(next.clone()),
        &shape,
    )?;
    // Which keys an edit actually touched. A save is supposed to be a one-line
    // diff, so "it patched three keys when I changed one" is the first question
    // worth asking when a file comes out looking different from expected — and
    // the answer is otherwise unobtainable from outside this function.
    if !patches.is_empty() {
        tracing::debug!(keys = ?touched, "yaml edit patching");
    }

    // No patch means no change: hand back the original bytes rather than
    // re-emitting them. `apply_yaml_patches` refuses an empty patch set anyway,
    // and an idle Save must not reformat the repository.
    if patches.is_empty() {
        return Ok(original.to_string());
    }

    /* Applied inside a `catch_unwind`, because it can panic.
     *
     * `yamlpath::Document::block_removal_span` indexes a line table with a
     * range it did not check, and panics in `line-index` for some shapes of
     * key removal — reached from here by a save that drops several keys at
     * once, which is an ordinary thing for the field editor to be asked to do.
     *
     * A panic in an axum handler is a dead request and a stack trace in the
     * log, for an input the caller is allowed to send. Turning it into the
     * refusal this function already has a word for is both the honest answer
     * and the same one every other failure gets.
     *
     * `AssertUnwindSafe` is sound here: the inputs are borrowed immutably, the
     * output is discarded on panic, and nothing outside this call is left
     * half-written. */
    let applied = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        yamlpatch::apply_yaml_patches(&doc, &patches)
    }))
    .map_err(|_| format!("the patcher panicked applying {} patch(es)", patches.len()))?;
    let edited = applied.map_err(|e| format!("applying {} patch(es): {e}", patches.len()))?;
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

/// One op per node that actually differs, and a readable path for each.
///
/// The paths are carried out rather than derived from the patches because
/// `Route` is opaque — and "what did this save actually change" is the question
/// every investigation of a surprising diff starts from.
fn plan<'a>(
    doc: &'a Document,
    current: &Value,
    next: &Value,
    shape: &Shape,
) -> Result<(Vec<Patch<'a>>, Vec<String>), String> {
    let mut patches = Vec::new();
    let mut touched = Vec::new();
    descend(
        doc,
        Route::from(vec![]),
        String::new(),
        current,
        next,
        Some(shape),
        &mut patches,
        &mut touched,
    )?;
    Ok((patches, touched))
}

/// The original document as YAML, used only for the order and spelling a
/// rewritten value should come out in.
type Shape = serde_yaml::Value;

/// Walk both documents together and emit an op at the **deepest** node that
/// differs.
///
/// This is what makes a save reviewable. A section's `examples:` is sixteen
/// conversations of four turns; editing the wording of one turn is a change to
/// `examples[1].turns[0].content` and nothing else, so that is the node the
/// patch routes to and the diff is that block scalar. Replacing the value of
/// `examples` instead — which is what a top-level-keys-only plan can do —
/// rewrites all sixteen, and the one line that changed is then buried in three
/// hundred that did not.
///
/// The descent stops where the *shape* changes rather than the content: a
/// sequence that gained or lost an entry, or a mapping whose key set moved, has
/// no node-for-node correspondence to walk, so the collection is rewritten
/// whole — as block YAML, so an authored list stays an authored list.
fn descend<'a>(
    doc: &'a Document,
    route: Route<'a>,
    path: String,
    have: &Value,
    want: &Value,
    shape: Option<&Shape>,
    patches: &mut Vec<Patch<'a>>,
    touched: &mut Vec<String>,
) -> Result<(), String> {
    // Unchanged. The whole point: no op, so the bytes are untouched and the
    // author's block scalar, wrapping and inline comments survive exactly as
    // written.
    if have == want {
        return Ok(());
    }

    match (have, want) {
        // Same key set: recurse per key, so only the values that moved are
        // touched. A differing key set is handled here too — an arrival is an
        // `Add` to this mapping and a departure a `Remove` from it — because a
        // `PUT` replaces the document, so a key the caller did not send is one
        // they removed.
        (Value::Object(h), Value::Object(w)) => {
            for (key, wv) in w {
                let child = route.with_key(Component::Key(key.clone().into()));
                let child_path = join(&path, key);
                match h.get(key) {
                    Some(hv) => descend(
                        doc,
                        child,
                        child_path,
                        hv,
                        wv,
                        at_key(shape, key),
                        patches,
                        touched,
                    )?,
                    None => {
                        patches.push(Patch {
                            // `Add` routes to the *mapping* that gains the key,
                            // not to the key itself.
                            route: route.clone(),
                            operation: Op::Add {
                                key: key.clone(),
                                value: to_yaml(wv)?,
                            },
                        });
                        touched.push(child_path);
                    }
                }
            }
            for key in h.keys() {
                if !w.contains_key(key) {
                    patches.push(Patch {
                        route: route.with_key(Component::Key(key.clone().into())),
                        operation: Op::Remove,
                    });
                    touched.push(format!("-{}", join(&path, key)));
                }
            }
            Ok(())
        }
        // Same length: recurse per entry. A different length has no
        // correspondence — entry 3 of the new list is not entry 3 of the old
        // one once something was inserted — so it falls through to a rewrite.
        (Value::Array(h), Value::Array(w)) if h.len() == w.len() => {
            for (i, (hv, wv)) in h.iter().zip(w).enumerate() {
                descend(
                    doc,
                    route.with_key(Component::Index(i)),
                    format!("{path}[{i}]"),
                    hv,
                    wv,
                    at_index(shape, i),
                    patches,
                    touched,
                )?;
            }
            Ok(())
        }
        _ => {
            patches.push(Patch {
                operation: if is_collection(want) {
                    rewrite_collection(doc, &route, &path, want, shape)?
                } else {
                    replace_scalar(doc, &route, &path, want)?
                },
                route,
            });
            touched.push(path);
            Ok(())
        }
    }
}

/// `examples` + `turns` → `examples.turns`, and an empty parent stays out of
/// the way so a top-level key is just its own name.
fn join(parent: &str, key: &str) -> String {
    if parent.is_empty() {
        key.to_owned()
    } else {
        format!("{parent}.{key}")
    }
}

/// The shape under a mapping key.
fn at_key<'s>(shape: Option<&'s Shape>, key: &str) -> Option<&'s Shape> {
    shape?.as_mapping()?.get(serde_yaml::Value::from(key))
}

/// The shape of a sequence entry — **the first entry when there is no entry
/// `i`**, because a conversation appended to `examples` should be written the
/// way the fifteen already there are written, not the way a serialiser would
/// choose on its own.
fn at_index(shape: Option<&Shape>, i: usize) -> Option<&Shape> {
    let seq = shape?.as_sequence()?;
    seq.get(i).or_else(|| seq.first())
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
fn rewrite_collection<'a>(
    doc: &'a Document,
    route: &Route<'a>,
    path: &str,
    want: &Value,
    shape: Option<&Shape>,
) -> Result<Op<'a>, String> {
    // `query_exact`, not `query_pretty`: `RewriteFragment` searches within the
    // *exact* feature, which for a key route is the value alone. A `from` that
    // included the key would never match, and the patch fails with "no match
    // for … in feature" rather than doing something wrong quietly.
    let feature = doc
        .query_exact(route)
        .map_err(|e| format!("locating `{path}`: {e}"))?
        .ok_or_else(|| format!("`{path}` has no value to rewrite"))?;
    // Borrowed from the document rather than copied, because `Fragment` keeps
    // the borrow and the patch outlives this call.
    let current = doc.extract(&feature);

    let to = match blocks(want, shape) {
        // Block, at the column the value already starts at.
        //
        // The column comes from the parse rather than from reading a
        // continuation line: for a flat list the two agree, but for a list of
        // mappings the first continuation line is a nested key at a deeper
        // indent, and rendering the entries there produces a document that no
        // longer parses.
        Some(groups) if !is_flow(current) => {
            let indent = " ".repeat(feature.location.point_span.0 .1);
            // A blank line between entries where the author left blank lines
            // between them. The corpus separates its conversations that way and
            // an edit that closed the gaps would rewrite every line of the list.
            let spaced = matches!(want, Value::Array(_)) && current.contains("\n\n");
            let mut lines = Vec::new();
            for (i, group) in groups.into_iter().enumerate() {
                if spaced && i > 0 {
                    lines.push(String::new());
                }
                lines.extend(group);
            }
            indented(&lines, &indent)
        }
        // Flow: what the file already used, and the only form an empty
        // collection has.
        _ => yamlpatch::serialize_flow(&to_yaml(want)?)
            .map_err(|e| format!("flow-rendering `{path}`: {e}"))?,
    };

    Ok(Op::RewriteFragment {
        // `Fragment::new` turns whitespace into `\s+`, so a multi-line block
        // sequence matches its own text across the line breaks.
        from: Subfeature::new(0, Fragment::new(current)),
        to: to.into(),
    })
}

/// Replace a scalar, writing prose as the literal block the corpus writes it
/// as.
///
/// `Op::Replace` renders through `yaml_serde`, which for a multi-line string
/// nested inside a sequence produces text the patcher then rejects as invalid
/// YAML — and a section's turns are multi-line strings nested inside a sequence,
/// so that is the ordinary case rather than an edge. Rendering the block here
/// also means an edited turn stays a `|` block instead of becoming one long
/// line of `\n` escapes.
fn replace_scalar<'a>(
    doc: &'a Document,
    route: &Route<'a>,
    path: &str,
    want: &Value,
) -> Result<Op<'a>, String> {
    let Some((header, body)) = literal_block(want) else {
        return Ok(Op::Replace(to_yaml(want)?));
    };
    // A literal block's content is indented against its **key**, not against
    // where the old value happened to start — `body: x` has the value eight
    // columns in and the key at two.
    let key_col = doc
        .query_pretty(route)
        .map_err(|e| format!("locating `{path}`: {e}"))?
        .location
        .point_span
        .0
         .1;
    let feature = doc
        .query_exact(route)
        .map_err(|e| format!("locating `{path}`: {e}"))?
        .ok_or_else(|| format!("`{path}` has no value to rewrite"))?;
    let current = doc.extract(&feature);

    let indent = " ".repeat(key_col + 2);
    let mut lines = vec![header.to_string()];
    lines.extend(body.into_iter().map(str::to_owned));
    let to = indented(&lines, &indent);

    Ok(Op::RewriteFragment {
        from: Subfeature::new(0, Fragment::new(current)),
        to: to.into(),
    })
}

/// A collection as block YAML, one entry per line, relative to the column the
/// value starts at — the caller indents every line after the first.
///
/// `None` for anything that has no block form here: an empty collection (`[]`
/// and `{}` are the only spellings), or a shape carrying a key or a scalar this
/// cannot render safely. The caller falls back to flow, and the read-back check
/// in [`edit`] refuses the result if either was wrong.
///
/// This exists because the corpus is authored. `examples:` is sixteen
/// conversations written as block sequences of block scalars; the alternative —
/// `serialize_flow` on the whole value — is correct YAML and a single
/// four-thousand-character line, which ends the file's life as something a
/// person reads or reviews.
fn blocks(v: &Value, shape: Option<&Shape>) -> Option<Vec<Vec<String>>> {
    match v {
        Value::Array(items) if !items.is_empty() => {
            let mut out = Vec::new();
            for (i, item) in items.iter().enumerate() {
                let mut lines = block_lines(item, at_index(shape, i))?;
                let first = lines.remove(0);
                let mut group = vec![format!("- {first}")];
                group.extend(lines.into_iter().map(indent_by_two));
                out.push(group);
            }
            Some(out)
        }
        Value::Object(map) if !map.is_empty() => {
            let mut out = Vec::new();
            for key in ordered(map, shape) {
                let val = &map[&key];
                if !plain_key(&key) {
                    return None;
                }
                match literal_block(val) {
                    // A multi-line string as the literal block the author wrote
                    // it as, rather than one line of `\n` escapes.
                    Some((header, body)) => {
                        let mut group = vec![format!("{key}: {header}")];
                        group.extend(body.into_iter().map(|l| indent_by_two(l.to_owned())));
                        out.push(group);
                    }
                    None if is_collection(val) && !is_empty_collection(val) => {
                        let mut group = vec![format!("{key}:")];
                        for lines in blocks(val, at_key(shape, &key))? {
                            group.extend(lines.into_iter().map(indent_by_two));
                        }
                        out.push(group);
                    }
                    None => out.push(vec![format!("{key}: {}", scalar(val)?)]),
                }
            }
            Some(out)
        }
        _ => None,
    }
}

/// The same thing flattened, for a caller that does not care where one entry
/// ends and the next begins.
fn block_lines(v: &Value, shape: Option<&Shape>) -> Option<Vec<String>> {
    match v {
        Value::Array(_) | Value::Object(_) if !is_empty_collection(v) => {
            Some(blocks(v, shape)?.concat())
        }
        // An empty collection has exactly one spelling and it is flow; so does
        // a scalar standing where a collection could have been.
        _ => Some(vec![scalar(v)?]),
    }
}

/// A value on one line — bare where the author could have written it bare,
/// quoted where it needs to be.
///
/// `serialize_flow` quotes every string. That is always correct and, on a
/// rewritten list, always visible: sixteen `note:` lines gaining quotes they
/// never had is a diff over the whole file for an edit to one of them. So a
/// string that YAML reads back as itself is written plain, and everything else
/// goes to the serialiser.
fn scalar(v: &Value) -> Option<String> {
    if let Value::String(s) = v {
        return Some(if plain_scalar(s) {
            s.clone()
        } else {
            // Quoted here rather than by `serialize_flow`, which writes the
            // string `12` as a bare `12` — a value that reads back as a number.
            // The read-back check catches it and the save is then refused, so
            // the difference is between a field that saves and one that cannot.
            //
            // JSON's string escaping is a subset of YAML's double-quoted style
            // (`\n`, `\t`, `\"`, `\\`, `\uXXXX`), so the serialiser for it is
            // the right one and there is no hand-rolled escaping here.
            serde_json::to_string(s).ok()?
        });
    }
    yamlpatch::serialize_flow(&to_yaml(v).ok()?)
        .ok()
        .map(|s| s.trim().to_string())
}

/// Whether a string can be written as a bare YAML scalar and read back
/// unchanged.
///
/// Deliberately narrow: prose that starts with a letter or a digit and holds
/// nothing that would restructure its line. An indicator character in the first
/// position, a `: ` or ` #`, edge whitespace, or a word a YAML reader resolves
/// to a bool or a number all take the quotes.
///
/// A quote **inside** the text does not. `note: A disagreement about the
/// evening's plan` is a plain scalar, and rejecting it over the apostrophe put
/// quotes on a line the author wrote bare — a change to a line nobody edited.
fn plain_scalar(s: &str) -> bool {
    if s.is_empty() || s.contains('\n') || s.trim() != s {
        return false;
    }
    if !s.starts_with(|c: char| c.is_ascii_alphanumeric()) {
        return false;
    }
    if s.contains(": ") || s.contains(" #") || s.ends_with(':') {
        return false;
    }
    if s.chars().any(char::is_control) {
        return false;
    }
    // A bare `12`, `true`, or `null` reads back as something other than a
    // string, which would change the value's type rather than its spelling. The
    // test is the reader itself, so it stays right if the schema does.
    !matches!(
        serde_yaml::from_str::<serde_yaml::Value>(s),
        Ok(serde_yaml::Value::Bool(_) | serde_yaml::Value::Number(_) | serde_yaml::Value::Null)
    )
}

/// A mapping's keys in the order the file writes them, with anything the file
/// does not have after them. Alphabetical is what `serde_json::Map` gives, and
/// `role` before `content` is what the author wrote.
fn ordered(map: &Map<String, Value>, shape: Option<&Shape>) -> Vec<String> {
    let mut out = Vec::with_capacity(map.len());
    if let Some(m) = shape.and_then(Shape::as_mapping) {
        for key in m.keys().filter_map(|k| k.as_str()) {
            if map.contains_key(key) {
                out.push(key.to_owned());
            }
        }
    }
    let rest: Vec<String> = map.keys().filter(|k| !out.contains(k)).cloned().collect();
    out.extend(rest);
    out
}

fn indent_by_two(line: String) -> String {
    if line.is_empty() {
        line
    } else {
        format!("  {line}")
    }
}

/// Lines joined at an indent, leaving blank lines blank rather than turning
/// them into runs of trailing whitespace.
fn indented(lines: &[String], indent: &str) -> String {
    let mut out = String::new();
    for (i, line) in lines.iter().enumerate() {
        if i > 0 {
            out.push('\n');
            if !line.is_empty() {
                out.push_str(indent);
            }
        }
        out.push_str(line);
    }
    out
}

/// A multi-line string as a literal block: its indicator, and its lines.
///
/// `None` where a literal block would not round-trip. Leading whitespace on the
/// first line needs an explicit indentation indicator, trailing whitespace on
/// any line is eaten by the parser, and more than one trailing newline needs
/// `|+` and its own care — all rare enough in prose that a quoted one-liner is
/// the better answer than a renderer that has to be right about them.
fn literal_block(v: &Value) -> Option<(&'static str, Vec<&str>)> {
    let s = v.as_str()?;
    if !s.contains('\n') || s.chars().any(|c| c.is_control() && c != '\n') {
        return None;
    }
    // The indicator is the only thing that says whether the value ends in a
    // newline, so it is chosen from the value rather than assumed.
    let (header, body) = match s.strip_suffix('\n') {
        None => ("|-", s),
        Some(rest) if rest.is_empty() || rest.ends_with('\n') => return None,
        Some(rest) => ("|", rest),
    };
    let lines: Vec<&str> = body.split('\n').collect();
    if lines[0].starts_with(' ') || lines.iter().any(|l| l.ends_with(' ')) {
        return None;
    }
    Some((header, lines))
}

/// A key that can be written bare. Anything else — a space, a colon, something
/// that would read back as a number — sends the whole mapping to flow, where
/// the serialiser decides the quoting.
fn plain_key(k: &str) -> bool {
    !k.is_empty()
        && k.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
        && k.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
}

fn is_empty_collection(v: &Value) -> bool {
    matches!(v, Value::Array(a) if a.is_empty()) || matches!(v, Value::Object(o) if o.is_empty())
}

/// Whether the existing value text is written in flow style.
fn is_flow(current: &str) -> bool {
    let t = current.trim_start();
    t.starts_with('[') || t.starts_with('{')
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

    /// **A save that drops several keys must be refused, not fatal.**
    ///
    /// `yamlpath` panics inside its line table computing the removal span for
    /// some shapes of key removal. Reached from the field editor — which sends
    /// the values it holds, so a document that lost keys is an ordinary
    /// request — that was a panicking HTTP handler. It is a `None` now, which
    /// the caller already renders as `409 cannot_patch`.
    #[test]
    fn a_patcher_panic_is_a_refusal_rather_than_a_crash() {
        let original = "\
id: world
description: |
  Shared knowledge about the setting.
window: 8000
score_threshold: 0.30
gather_scope: shared
budget:
  priority: 70
groups:
  - id: canon
";
        // Everything but two keys removed at once.
        let out = splice(original, &obj(json!({ "id": "world", "window": 9000 })));
        match out {
            // Whichever way the crate behaves, the contract here is the same:
            // an answer, and a correct one.
            Some(text) => {
                let back: Value = serde_yaml::from_str(&text).expect("parses");
                assert_eq!(back, json!({ "id": "world", "window": 9000 }), "{text}");
            }
            None => {}
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

    /// A section's `examples:`, cut to two conversations of two turns. The
    /// shape every response file in the corpus has.
    const EXAMPLES: &str = r#"id: accept_then_move_on

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

  - note: A boundary named plainly.
    turns:
      - role: user
        content: |
          "Can we not talk about it."
      - role: assistant
        thinking: |
          The subject drops.
"#;

    /// **The property that makes the examples editor usable.**
    ///
    /// Editing the wording of one turn must change that turn and nothing else.
    /// A plan that could only route to top-level keys replaced the whole of
    /// `examples`, so a one-word fix arrived as a three-hundred-line diff with
    /// every other conversation rewritten around it — correct YAML, and an
    /// unreviewable change to a file a person wrote.
    #[test]
    fn editing_one_turn_changes_that_turn_and_nothing_else() {
        let mut want: Value = serde_yaml::from_str(EXAMPLES).unwrap();
        want["examples"][1]["turns"][0]["content"] =
            json!("\"Let's talk about something else.\"\n");
        let out = splice(EXAMPLES, &obj(want.clone())).expect("editable");

        assert_eq!(
            serde_yaml::from_str::<Value>(&out).unwrap(),
            want,
            "\n{out}"
        );
        // Every line but the one edited is byte-identical, in place.
        let before: Vec<&str> = EXAMPLES.lines().collect();
        let after: Vec<&str> = out.lines().collect();
        assert_eq!(after.len(), before.len(), "line count moved:\n{out}");
        let moved: Vec<usize> = (0..before.len())
            .filter(|&i| before[i] != after[i])
            .collect();
        assert_eq!(moved.len(), 1, "changed lines {moved:?}:\n{out}");
        assert!(after[moved[0]].contains("something else"), "{out}");
        // Including the comment and the block scalars around it.
        assert!(out.contains("# Provenance lead-ins"), "{out}");
        assert!(out.contains("        content: |\n"), "{out}");
    }

    /// A conversation added to the list rewrites the list — there is no
    /// entry-for-entry correspondence once the length moves — but it comes back
    /// as the block sequence it was, not as one flow line.
    ///
    /// `serialize_flow` on this value is valid YAML and a single
    /// four-thousand-character line. That ends the file's life as something a
    /// person reads, which is the same thing losing the comments would do.
    #[test]
    fn adding_a_conversation_keeps_the_list_a_block() {
        let mut want: Value = serde_yaml::from_str(EXAMPLES).unwrap();
        want["examples"].as_array_mut().unwrap().push(json!({
            "note": "A third.",
            "turns": [{ "role": "user", "content": "\"Something new.\"\n" }],
        }));
        let out = edit(EXAMPLES, &obj(want.clone())).unwrap_or_else(|e| panic!("{e}"));

        assert_eq!(
            serde_yaml::from_str::<Value>(&out).unwrap(),
            want,
            "\n{out}"
        );
        assert!(out.contains("# Provenance lead-ins"), "{out}");
        // A block list of block scalars, at the indent the file already used.
        assert!(out.contains("\n  - note: A third.\n"), "{out}");
        assert!(out.contains("\n      - role: user\n"), "{out}");
        assert!(
            out.contains("        content: |\n          \"Something new.\"\n"),
            "the block scalar flattened:\n{out}"
        );
        assert!(
            !out.contains("\\n"),
            "escaped newlines in the output:\n{out}"
        );

        // **And every line that was already there is still there, unchanged and
        // in order.** The rewrite re-renders the whole list, so this is what
        // says the rendering matches the author's: same key order (`role`
        // before `content`, not the serialiser's alphabetical), same bare
        // scalars, same blank line between conversations. Without it the diff
        // for adding one example is every line of the file.
        let before: Vec<&str> = EXAMPLES.lines().collect();
        let after: Vec<&str> = out.lines().collect();
        assert_eq!(after[..before.len()], before[..], "\n{out}");
        assert_eq!(
            after[before.len()..],
            [
                "",
                "  - note: A third.",
                "    turns:",
                "      - role: user",
                "        content: |",
                "          \"Something new.\""
            ],
            "\n{out}"
        );
    }

    /// A rewritten list must not re-spell the entries it did not change.
    ///
    /// Every one of these is written bare in the corpus, and a renderer that
    /// quoted them would turn "add one example" into a diff over every `note:`
    /// line in the file. The ones that genuinely cannot be written bare are
    /// here too, because guessing wrong in that direction is a document that no
    /// longer says what it said.
    #[test]
    fn a_rewritten_entry_keeps_the_spelling_the_author_used() {
        let bare = [
            "A disagreement about the evening's plan, dropped gracefully.",
            "Wine declined; the glass simply moves on.",
            "Late apology, no toll charged for it.",
            "A fact corrected mid-story, taken without defense.",
            "3 turns, then the decode point",
            // A YAML 1.1 boolean. Under the 1.2 core schema this reads back as
            // the string it is, so it stays bare — and quoting it would be this
            // renderer disagreeing with the serialiser beside it about what
            // needs quotes.
            "yes",
        ];
        let quoted = [
            "12",
            "null",
            "- not a list",
            "#not a comment",
            "a: colon",
            "trailing ",
            "*alias",
            "ends with a colon:",
            "a word # then a comment",
        ];
        for note in bare.iter().chain(&quoted) {
            let want = json!({ "items": [{ "note": note }, { "note": "second" }] });
            let out = edit("items:\n  - note: x\n", &obj(want.clone()))
                .unwrap_or_else(|e| panic!("{note:?}: {e}"));
            assert_eq!(
                serde_yaml::from_str::<Value>(&out).unwrap(),
                want,
                "{note:?} came back wrong:\n{out}"
            );
            let plain = out.contains(&format!("note: {note}\n"));
            assert_eq!(
                plain,
                bare.contains(note),
                "{note:?} was spelled the other way:\n{out}"
            );
        }
    }

    /// The literal-block indicator is chosen from the value, because it is the
    /// only thing that says whether the string ends in a newline. Getting it
    /// wrong is a silent one-character change to authored prose.
    #[test]
    fn a_rewritten_block_scalar_keeps_its_trailing_newline_or_lack_of_one() {
        for (text, indicator) in [("a\nb\n", "|"), ("a\nb", "|-")] {
            let want = json!({ "items": [{ "body": text }] });
            let out = edit("items:\n  - body: x\n", &obj(want.clone()))
                .unwrap_or_else(|e| panic!("{text:?}: {e}"));
            assert!(out.contains(&format!("body: {indicator}")), "{out}");
            assert_eq!(
                serde_yaml::from_str::<Value>(&out).unwrap(),
                want,
                "\n{out}"
            );
        }
    }

    /// Prose the literal block cannot hold — a line ending in a space, which
    /// the parser eats — falls back to a quoted scalar rather than being
    /// written out a character short.
    #[test]
    fn text_a_literal_block_would_damage_is_quoted_instead() {
        let want = json!({ "items": [{ "body": "trailing space \nand more\n" }] });
        let out = splice("items:\n  - body: x\n", &obj(want.clone())).expect("editable");
        assert_eq!(
            serde_yaml::from_str::<Value>(&out).unwrap(),
            want,
            "\n{out}"
        );
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
