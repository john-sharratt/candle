//! The routes that serve a signed-in caller's own account, profile and name.
//!
//! Every handler here answers for the caller's own account. The caller's
//! identity is retrieved rather than re-checked, because the registration
//! already put each route behind its role bar. Ownership is authorization,
//! so the account is keyed by its public handle and never by the provider's
//! subject. Profile writes are checked before anything is merged, so a
//! wrong-typed field is refused instead of being blanked by the store's
//! repair pass. History and restore are separate routes, and a restored
//! revision comes back as a new one, so no revision number is ever reused.

use super::*;

/// The caller's identity, on a route the registration already put behind
/// [`Role::User`].
///
/// The role bar is not re-checked here — `host::guard` enforced it before
/// the handler ran. What this does is *retrieve* the identity, which handlers
/// need for the part a role cannot express: which records are yours. The
/// `require` call is how it gets one safely, and it doubles as a backstop if a
/// route is ever registered at a lower bar than its handler assumes.
pub(super) fn caller(s: &Arc<Authored>, headers: &HeaderMap) -> Result<Identity, Box<Response>> {
    Ok(require(headers, &s.roles, Role::User)?
        .into_identity()
        .expect("`Role::User` is only reachable with an identity"))
}

/// The caller's identity on an account route, which a guest — signed in, not
/// on the role table — reaches too: their own account is the one thing a
/// guest owns (`docs/guest_access.md` §3).
pub(crate) fn account_holder(
    s: &Arc<Authored>,
    headers: &HeaderMap,
) -> Result<Identity, Box<Response>> {
    Ok(require(headers, &s.roles, Role::Guest)?
        .into_identity()
        .expect("`Role::Guest` is only reachable with an identity"))
}

// ── the cast ─────────────────────────────────────────────────────────────────

/// The caller's account id, creating the local record on first sight.
///
/// Every NPC route needs it: ownership is authorization (§8.2), so there is no
/// such thing as an anonymous read of a character.
pub async fn owner_of(
    s: &Arc<Authored>,
    headers: &HeaderMap,
) -> Result<(Identity, String), Box<Response>> {
    let id = caller(s, headers)?;
    let me = account_of(&s.accounts, &id, now_ms()).await.map_err(|e| {
        tracing::error!(error = %e, "account upsert failed");
        Box::new(err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "account_write_failed",
            &e.to_string(),
        ))
    })?;
    // The account's public handle (`u_1a2b3c4d`), not the provider's subject.
    // The subject never leaves `accounts`, and a character record is a durable
    // artifact in a store that outlives this daemon — the last place to write
    // an identity provider's id.
    let owner = me["user_id"].as_str().map(str::to_owned).ok_or_else(|| {
        Box::new(err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "account_malformed",
            "account has no user_id",
        ))
    })?;
    Ok((id, owner))
}

/// The caller's account here, creating the record on first sight — npcd's
/// share of the host's `/v1/me`. The host states the role beside it, from the
/// same table this daemon decides with.
///
/// `Err` is the failed write, said once in the log; the host's answer then
/// carries no account rather than a fabricated one.
pub async fn account(s: &Authored, id: &Identity) -> Result<Value, String> {
    account_of(&s.accounts, id, now_ms()).await.map_err(|e| {
        tracing::error!(error = %e, "account upsert failed");
        e.to_string()
    })
}

pub(super) async fn get_profile(State(s): State<Arc<Authored>>, headers: HeaderMap) -> Response {
    let id = match account_holder(&s, &headers) {
        Ok(id) => id,
        Err(r) => return *r,
    };
    match s.accounts.read().await.get(&id) {
        Some(me) => Json(me["profile"].clone()).into_response(),
        // Signed in but never seen: `/v1/me` is what creates the record.
        None => err(StatusCode::NOT_FOUND, "account_not_found", "no account yet"),
    }
}

pub(super) async fn put_profile(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Json(patch): Json<Value>,
) -> Response {
    let id = match account_holder(&s, &headers) {
        Ok(id) => id,
        Err(r) => return *r,
    };
    // Everything the caller sent is checked before any of it is merged. The
    // store repairs a wrong-typed field by blanking it — which is right for a
    // record read off disk and wrong for a live request, where it would answer
    // 200 while destroying what the author had written.
    let Some(patch) = patch.as_object() else {
        return err(
            StatusCode::BAD_REQUEST,
            "bad_request",
            "expected a JSON object",
        );
    };
    if let Err(e) = accounts::check_patch(patch) {
        return match e {
            PatchError::NotText(field) => err(
                StatusCode::BAD_REQUEST,
                "bad_request",
                &format!("`{field}` must be a string"),
            ),
            PatchError::BadGender => err(
                StatusCode::BAD_REQUEST,
                "bad_gender",
                &format!("expected one of {}", accounts::GENDERS.join(", ")),
            ),
        };
    }
    match s.accounts.write().await.put_profile(&id, patch, now_ms()) {
        Ok(Some(me)) => Json(me["profile"].clone()).into_response(),
        Ok(None) => err(StatusCode::NOT_FOUND, "account_not_found", "no account yet"),
        Err(e) => {
            tracing::error!(error = %e, "profile write failed");
            err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "account_write_failed",
                &e.to_string(),
            )
        }
    }
}

/// Every revision the author has had, live one first.
///
/// Worth its own route rather than a field on `/v1/me`: a profile is revised
/// rarely and read on every page load, so the history is the larger half of the
/// record and almost never the part being asked for.
pub(super) async fn get_profile_history(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
) -> Response {
    let id = match account_holder(&s, &headers) {
        Ok(id) => id,
        Err(r) => return *r,
    };
    match s.accounts.read().await.profile_history(&id) {
        Some(revisions) => Json(json!({ "revisions": revisions })).into_response(),
        None => err(StatusCode::NOT_FOUND, "account_not_found", "no account yet"),
    }
}

/// One revision in full.
///
/// Separate from the index because the index is a chooser and this is the
/// reading: with hundreds of revisions, sending every paragraph in order to
/// render a list of dates is the wrong trade by two orders of magnitude.
pub(super) async fn get_profile_revision(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(rev): Path<u64>,
) -> Response {
    let id = match account_holder(&s, &headers) {
        Ok(id) => id,
        Err(r) => return *r,
    };
    match s.accounts.read().await.profile_revision(&id, rev) {
        Some(v) => Json(v).into_response(),
        None => err(
            StatusCode::NOT_FOUND,
            "revision_not_found",
            "no such revision",
        ),
    }
}

/// Bring a superseded revision back as the live one.
///
/// `POST`, not `PUT`: it is not idempotent. Restoring the same revision twice
/// produces two new revisions, because each is an edit in its own right — the
/// second says "still this" a minute after the first, and both are true.
pub(super) async fn restore_profile(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(rev): Path<u64>,
) -> Response {
    let id = match account_holder(&s, &headers) {
        Ok(id) => id,
        Err(r) => return *r,
    };
    match s.accounts.write().await.restore_profile(&id, rev, now_ms()) {
        Ok(Some(me)) => Json(me["profile"].clone()).into_response(),
        Ok(None) => err(
            StatusCode::NOT_FOUND,
            "revision_not_found",
            "no such revision",
        ),
        Err(e) => {
            tracing::error!(error = %e, "profile restore failed");
            err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "account_write_failed",
                &e.to_string(),
            )
        }
    }
}

/// Set the name characters address this author by.
///
/// Separate from the profile because it fails differently: a profile edit is
/// prose and always succeeds, while a name can be the wrong shape or already
/// somebody else's. 409 is the interesting one — it is the only place in the
/// account surface where one author's data can block another's write, and the
/// console has to be able to say so rather than report a generic failure.
pub(super) async fn put_unique_name(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let id = match account_holder(&s, &headers) {
        Ok(id) => id,
        Err(r) => return *r,
    };
    let Some(name) = body.get("unique_name").and_then(|v| v.as_str()) else {
        return err(
            StatusCode::BAD_REQUEST,
            "bad_request",
            "expected a `unique_name` string",
        );
    };
    match s.accounts.write().await.put_unique_name(&id, name) {
        Ok(me) => Json(me).into_response(),
        Err(NameError::Shape(why)) => err(StatusCode::BAD_REQUEST, "bad_unique_name", why),
        Err(NameError::Taken) => err(
            StatusCode::CONFLICT,
            "name_taken",
            "that name is already in use",
        ),
        Err(NameError::NoAccount) => {
            err(StatusCode::NOT_FOUND, "account_not_found", "no account yet")
        }
        Err(NameError::Io(e)) => {
            tracing::error!(error = %e, "unique name write failed");
            err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "account_write_failed",
                &e.to_string(),
            )
        }
    }
}
