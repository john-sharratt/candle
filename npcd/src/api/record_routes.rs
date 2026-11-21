//! The routes for authored worlds and personalities, as the caller may see
//! them.
//!
//! A save takes its id from the URL and discards the id in the body. It also
//! strips derived fields such as the living-character count, so a
//! read-edit-write cycle cannot write a stale count into an author's file.
//! Hidden documents are revealed only to an admin: the `reveal` parameter is
//! a request that the role answers, never a grant. An author sees their own
//! drafts over the published records.

use super::*;

/// The console reads `world_id` and `personality_id`; the file knows only its own
/// name. Joining the two here keeps the id out of the authored document, where
/// it would be a second place for the same fact to live and disagree.
pub(crate) fn with_id(key: &str, id: &str, body: &Value) -> Value {
    let mut out = body.clone();
    if let Some(map) = out.as_object_mut() {
        map.insert(key.to_string(), json!(id));
    }
    out
}

/// Attach the count of living characters that name this document.
///
/// A listing figure, not an accounting one: hidden characters are excluded (see
/// [`Npcs::counts_by`]), so it is a floor rather than a total.
pub(super) fn with_count(mut v: Value, counts: &BTreeMap<&str, usize>, id: &str) -> Value {
    if let Some(map) = v.as_object_mut() {
        map.insert(
            "npc_count".to_string(),
            json!(counts.get(id).copied().unwrap_or(0)),
        );
    }
    v
}

/// Whether this request may see hidden documents.
///
/// `?reveal=1` asks; **being an admin is what answers**. The parameter is a
/// request, never a grant — an unauthenticated caller appending it to the URL
/// gets the same listing as one who did not, because the role is read from the
/// gateway's headers on this request and not from anything the client says.
///
/// The console sends it while an admin holds RIGHT ALT, which is the whole of
/// the gesture: the key is a convenience for someone who already has the role,
/// so that a listing being screen-shared does not contain what a screenshot
/// should not. It is discretion, not access control — `GET /v1/world/earth`
/// still answers anyone who knows the id, exactly as it did before.
pub(super) fn revealing(
    s: &Arc<Authored>,
    headers: &HeaderMap,
    q: &HashMap<String, String>,
) -> bool {
    if !matches!(q.get("reveal").map(String::as_str), Some("1" | "true")) {
        return false;
    }
    let id = crate::identity::identify(headers).ok();
    s.roles.of(id.as_ref()).at_least(Role::Admin)
}

/// Every record of `reg`, kept in `folder`, as the caller sees them: an admin
/// sees their own drafts over the published records (§10.3), anyone else the
/// published ones.
pub(super) async fn seen_all(
    s: &Authored,
    headers: &HeaderMap,
    reg: &RwLock<Registry>,
    folder: &str,
) -> Vec<Record> {
    let admin = caller::admin(s, headers);
    let reg = reg.read().await;
    match &admin {
        Some(author) => records::records(author, &reg, folder),
        None => reg.iter().cloned().collect(),
    }
}

/// Record `id` of `reg`, kept in `folder`, as the caller sees it.
pub(super) async fn seen(
    s: &Authored,
    headers: &HeaderMap,
    reg: &RwLock<Registry>,
    folder: &str,
    id: &str,
) -> Option<Record> {
    let admin = caller::admin(s, headers);
    let reg = reg.read().await;
    match &admin {
        Some(author) => records::record(author, &reg, folder, id),
        None => reg.get(id).cloned(),
    }
}

/// Shared save path: the calling admin's write of record `id` of the
/// collection kept in `folder` — a draft over a mind under git, what stands
/// over one that is not (§10.3).
///
/// The id is taken from the URL and the body's own id field is discarded rather
/// than trusted: a document that could name its own file is a document that
/// could name somebody else's.
pub(super) async fn save(
    s: &Arc<Authored>,
    headers: &HeaderMap,
    kind: &str,
    folder: &str,
    key: &str,
    id: &str,
    mut body: Value,
) -> Response {
    let author = match caller::author(s, headers) {
        Ok(a) => a,
        Err(r) => return *r,
    };
    if let Some(map) = body.as_object_mut() {
        map.remove(key);
        // `npc_count` is computed at read time, exactly like the id, and for
        // the same reason must be stripped here rather than by the client: an
        // ordinary read-edit-write cycle sends back everything it was given,
        // so a client that forgets writes a stale count into the author's file
        // and the file grows a field that disagrees with the cast forever
        // after. A caller cannot be the thing that keeps a derived value out of
        // a document.
        map.remove("npc_count");
    } else {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_arguments",
            "the body must be a JSON object",
        );
    }
    match records::save(&author, kind, folder, id, &body) {
        Ok(()) => {
            s.wrote(&author).await;
            Json(with_id(key, id, &body)).into_response()
        }
        Err(e) => registry_err(e),
    }
}

/// Shared remove path, as the calling admin.
pub(super) async fn remove(
    s: &Arc<Authored>,
    headers: &HeaderMap,
    folder: &str,
    missing: &str,
    id: &str,
) -> Response {
    let author = match caller::author(s, headers) {
        Ok(a) => a,
        Err(r) => return *r,
    };
    match records::remove(&author, folder, id) {
        Ok(true) => {
            s.wrote(&author).await;
            StatusCode::NO_CONTENT.into_response()
        }
        Ok(false) => err(StatusCode::NOT_FOUND, missing, id),
        Err(e) => registry_err(e),
    }
}
