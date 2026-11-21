//! The routes that browse, read and edit the authored mind by its addresses.
//!
//! Every mind request resolves its scope in one shared helper, so no handler
//! can forget to apply a world's filter. An address is a corpus name such as
//! `canon/ammo/bolt`, never a path, so nothing on the wire is a file. A field
//! save patches its values into the document already on disk, which keeps the
//! authoring comments above each key. A save that cannot be patched exactly is
//! refused rather than rewritten whole, since a rewrite would cost the file its
//! comments. Deleting a topic removes its own text and keeps what is inside it.

use super::*;

// ── the mind's files ────────────────────────────────────────────────────────

/// Everything a mind request needs, or the refusal that says why not.
///
/// The handlers below all begin the same way — is there a mind, does the path
/// parse, which world is this scoped to — so it happens once. The scope is
/// resolved here too, which is what stops a handler forgetting to apply it.
pub(super) async fn mind_request(
    s: &Arc<Authored>,
    headers: &HeaderMap,
    q: &HashMap<String, String>,
) -> Result<(Option<Address>, Scope), Box<Response>> {
    if !s.mind.is_present() {
        return Err(Box::new(err(
            StatusCode::NOT_FOUND,
            "no_mind",
            "this daemon was started without --mind, so it has no authored content to edit",
        )));
    }
    // `?id=` is an address in the corpus — `canon/ammo/bolt` — not a path. The
    // absent case is the corpus itself.
    let addr = Address::parse(q.get("id").map(String::as_str).unwrap_or_default())
        .map_err(|e| err(StatusCode::BAD_REQUEST, "unknown_address", &e.to_string()))?;

    // `?world=` is optional. Absent means the whole mind, which is the right
    // default for an editor: a world is a lens on one corpus, and somebody
    // editing the corpus should not have to choose a lens first.
    let scope = match q.get("world").map(String::as_str).filter(|w| !w.is_empty()) {
        None => Scope::unscoped(),
        Some(wid) => match seen(s, headers, &s.worlds, WORLDS, wid).await {
            Some(r) => Scope::of_world(&r.body),
            None => {
                return Err(Box::new(err(StatusCode::BAD_REQUEST, "unknown_world", wid)));
            }
        },
    };
    Ok((addr, scope))
}

/// The mind as the caller reads it (§10.3): an admin's own drafts over the
/// published mind; anyone else the published mind. Asked after
/// [`mind_request`], which refuses a daemon with no mind.
pub(super) fn mind_view(s: &Authored, headers: &HeaderMap) -> Box<dyn Shelf> {
    match caller::admin(s, headers) {
        Some(author) => Box::new(author),
        None => Box::new(
            s.mind
                .published()
                .unwrap_or_else(|| Published::folder(PathBuf::new())),
        ),
    }
}

/// The section category of a `responses/` or `moods/` file, from the libraries
/// already in memory.
///
/// The scope needs it to apply a world's `excludes`, and reading it from the
/// loaded library rather than from disk keeps a directory listing a directory
/// listing — otherwise browsing `responses/` would open 596 files to decide
/// what to show.
pub(super) fn category_lookup(s: &Arc<Authored>) -> impl Fn(&MindPath) -> Option<String> + '_ {
    move |path: &MindPath| {
        let area = path.area()?;
        let name = path.name();
        // Only a file directly inside the folder is a section.
        if path.segments().len() != 2 {
            return None;
        }
        let id = name.strip_suffix(".yaml")?;
        let libraries = s.libraries();
        let library = match area {
            "responses" => &libraries.responses,
            "moods" => &libraries.moods,
            _ => return None,
        };
        library
            .sections
            .iter()
            .find(|sec| sec.id == id)
            .map(|sec| sec.category.clone())
    }
}

/// What is inside a place in the corpus.
///
/// With no `?id=`, the corpus itself — the nine sections. A caller never names
/// a directory, so there is no directory to be refused: a folder that is not
/// part of the corpus has no address at all.
pub(super) async fn mind_list(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    let (addr, scope) = match mind_request(&s, &headers, &q).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let view = mind_view(&s, &headers);
    let cat = category_lookup(&s);
    let (id, title, parent, has_text, children) = match &addr {
        None => (
            String::new(),
            "The mind".to_owned(),
            None,
            false,
            Ok(mind_catalog::sections(&*view, &scope, &cat)),
        ),
        Some(a) => (
            a.as_str(),
            a.title(),
            a.parent().map(|p| p.as_str()),
            // Whether *this* has text of its own, which is what lets a topic be
            // opened as well as opened into.
            a.entry_path()
                .is_some_and(|p| view.size(&p.as_str()).is_some()),
            mind_catalog::children(&*view, a, &scope, &cat),
        ),
    };
    match children {
        Ok(nodes) => Json(json!({
            "id": id,
            "title": title,
            "parent": parent,
            "has_text": has_text,
            "scoped": !scope.is_unscoped(),
            "children": nodes.iter().map(mind_catalog::Node::wire).collect::<Vec<_>>(),
        }))
        .into_response(),
        Err(e) => catalog_err(e),
    }
}

/// The addressed thing.
///
/// Every entry handler begins the same way and the scope check is the part that
/// must not be forgotten, so it happens here once. `None` — the corpus itself —
/// is refused: there is no text at the root to read, write or remove.
pub(super) async fn mind_entry_of(
    s: &Arc<Authored>,
    headers: &HeaderMap,
    q: &HashMap<String, String>,
) -> Result<Address, Box<Response>> {
    let (addr, scope) = mind_request(s, headers, q).await?;
    let Some(addr) = addr else {
        return Err(Box::new(err(
            StatusCode::BAD_REQUEST,
            "unknown_address",
            "name something in the mind",
        )));
    };
    let cat = category_lookup(s);
    // Asked of whichever path the address has, so a topic is checked by its
    // tag and an entry by its own file.
    let path = addr.collection_path().or_else(|| addr.entry_path());
    if let Some(p) = path {
        if !scope.admits(&p, &cat) {
            return Err(Box::new(err(
                StatusCode::FORBIDDEN,
                "out_of_scope",
                "this world does not include that",
            )));
        }
    }
    Ok(addr)
}

/// The addressed thing, and the calling admin as its author.
pub(super) async fn mind_write_of(
    s: &Arc<Authored>,
    headers: &HeaderMap,
    q: &HashMap<String, String>,
) -> Result<(Author, Address), Box<Response>> {
    let addr = mind_entry_of(s, headers, q).await?;
    Ok((caller::author(s, headers)?, addr))
}

/// Read the text of an entry, or a topic's overview.
pub(super) async fn mind_entry(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    let addr = match mind_entry_of(&s, &headers, &q).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    match mind_catalog::read(&*mind_view(&s, &headers), &addr) {
        Ok(d) => Json(json!({
            "id": addr.as_str(),
            "title": addr.title(),
            "text": d.text,
            "chars": d.text.chars().count(),
        }))
        .into_response(),
        Err(e) => doc_err(e),
    }
}

/// Write an entry — creating it, or replacing what is there.
///
/// `?new=1` refuses to land on something that exists, which is what "add"
/// needs: a create that overwrote would take somebody's work with no error.
pub(super) async fn put_mind_entry(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Query(q): Query<HashMap<String, String>>,
    Json(body): Json<Value>,
) -> Response {
    let (author, addr) = match mind_write_of(&s, &headers, &q).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let Some(text) = body.get("text").and_then(Value::as_str) else {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_field",
            "`text` is missing or is not a string",
        );
    };
    let must_be_new = matches!(q.get("new").map(String::as_str), Some("1" | "true"));
    let wrote = mind_catalog::write(&author, &addr, text, must_be_new);
    s.wrote(&author).await;
    match wrote {
        Ok(Wrote::Created) => (
            StatusCode::CREATED,
            Json(json!({ "id": addr.as_str(), "title": addr.title(), "created": true })),
        )
            .into_response(),
        Ok(Wrote::Updated) => {
            Json(json!({ "id": addr.as_str(), "title": addr.title(), "created": false }))
                .into_response()
        }
        Err(e) => doc_err(e),
    }
}

/// The projection layers, as the schema declares them.
///
/// Read from the mind's own `projection.yaml` — the same document
/// `settings/projection` edits, through the same reader — so there is no second
/// copy of the nine layers to drift from the first. There was: the console's
/// fixture answered this route with hand-written specs, and by the time anybody
/// looked it had `action` at budget priority 95 where the schema said 100.
/// Nothing could have noticed, because the two were never compared.
///
/// The layers go out in the schema's **own vocabulary** — `name`,
/// `gather_scope`, `budget`, `groups` — rather than translated into a shape of
/// this route's own. A translation is one more place for the two to disagree,
/// and the author's words are the ones the editor shows.
///
/// npcd's share of the host's `/v1/schema/layers`. A daemon with no mind runs
/// the bundled schema ([`BUNDLED`]), so that is the one listed — the layers the
/// engine is actually projecting, not an empty list beside a running nine.
/// Anything that stops the schema being read is the `Err`.
pub fn schema_layers(s: &Authored) -> Result<Vec<Value>, String> {
    // The address, and the keys that say where its parts live. Asking the
    // address rather than writing `layers` here keeps this honest if the schema
    // is ever held under a different key.
    let addr = match Address::parse("settings/projection") {
        Ok(Some(a)) => a,
        _ => return Err("could not address the projection schema".into()),
    };
    let Some((list_key, id_key)) = addr.parts() else {
        return Err("the projection schema declares no parts".into());
    };
    let text = match s.mind.published() {
        Some(published) => {
            mind_catalog::read(&published, &addr)
                .map_err(|e| format!("{e:?}"))?
                .text
        }
        None => BUNDLED.to_string(),
    };
    let items = mind_parts::list(&text, list_key, id_key).map_err(|e| e.to_string())?;
    Ok(items.into_iter().map(|(_, v)| v).collect())
}

/// An entry as editable fields, rather than as its text.
///
/// A document that is not a mapping — a list, a bare scalar — has no fields, and
/// says so with `not_fields` rather than an error the console would show as a
/// failure. It is a fact about the document, and the console offers the text
/// editor instead.
pub(super) async fn mind_fields(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    let addr = match mind_entry_of(&s, &headers, &q).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let doc = match mind_catalog::read(&*mind_view(&s, &headers), &addr) {
        Ok(d) => d,
        Err(e) => return doc_err(e),
    };
    // Whichever key is the address is the one that cannot be edited: `id` for a
    // section, `name` for a projection layer.
    let id_key = addr.part().map(|p| p.id_key).unwrap_or("id");
    match mind_section::parse(&doc.text, id_key) {
        Ok(fields) => Json(json!({
            "id": addr.as_str(),
            "title": addr.title(),
            "fields": fields.iter().map(mind_section::Field::wire).collect::<Vec<_>>(),
        }))
        .into_response(),
        Err(e) => err(
            StatusCode::UNPROCESSABLE_ENTITY,
            "not_fields",
            &e.to_string(),
        ),
    }
}

/// Save an entry from its fields, keeping every comment in the file.
///
/// The values are patched into the document that is already there rather than
/// serialised over it — see [`crate::mind::section`]. Nothing about the file
/// changes except the values that changed, so the authoring notes above each
/// key survive an edit made by somebody who never saw them.
pub(super) async fn put_mind_fields(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Query(q): Query<HashMap<String, String>>,
    Json(body): Json<Value>,
) -> Response {
    let (author, addr) = match mind_write_of(&s, &headers, &q).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let Some(values) = body.get("values").and_then(Value::as_object) else {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_field",
            "`values` is missing or is not an object",
        );
    };
    let wanted = match mind_section::to_document(values) {
        Ok(m) => m,
        Err(why) => return err(StatusCode::BAD_REQUEST, "invalid_field", &why),
    };

    // A *part* — one layer of the projection schema — is written straight
    // through. What it reads as is a rendering of that layer alone, with no
    // comments of its own to protect; the splice that keeps the file's comments
    // happens where they are, when the part goes back into the document that
    // holds it. See [`crate::mind::parts`].
    if addr.part().is_some() {
        let text = match serde_yaml::to_string(&Value::Object(wanted)) {
            Ok(t) => t,
            Err(e) => return err(StatusCode::BAD_REQUEST, "invalid_field", &e.to_string()),
        };
        let wrote = mind_catalog::write(&author, &addr, &text, false);
        s.wrote(&author).await;
        return match wrote {
            Ok(_) => Json(json!({ "id": addr.as_str(), "title": addr.title() })).into_response(),
            Err(e) => doc_err(e),
        };
    }

    let current = match mind_catalog::read(&author, &addr) {
        Ok(d) => d,
        Err(e) => return doc_err(e),
    };
    // `splice` returns `None` when it cannot patch — the document does not
    // parse, or its own read-back check found the result did not say what was
    // asked. Refused rather than falling back to a whole rewrite: the fallback
    // would silently cost the file its comments, which is the one thing this
    // path exists to protect.
    let Some(next) = registry::yaml_edit::splice(&current.text, &wanted) else {
        return err(
            StatusCode::CONFLICT,
            "cannot_patch",
            "this document could not be edited field by field without rewriting it, \
             which would lose its comments — edit it as text instead",
        );
    };
    let wrote = mind_catalog::write(&author, &addr, &next, false);
    s.wrote(&author).await;
    match wrote {
        Ok(_) => Json(json!({ "id": addr.as_str(), "title": addr.title() })).into_response(),
        Err(e) => doc_err(e),
    }
}

/// Remove an entry's text.
///
/// A topic keeps whatever is inside it — those have addresses of their own, and
/// one button must not become a recursive delete.
pub(super) async fn delete_mind_entry(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    let (author, addr) = match mind_write_of(&s, &headers, &q).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let removed = mind_catalog::remove(&author, &addr);
    s.wrote(&author).await;
    match removed {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => doc_err(e),
    }
}
