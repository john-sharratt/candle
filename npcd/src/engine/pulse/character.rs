//! The read and journal routes that show what one character is holding, what its
//! next turn would start from, and what its journal says.
//!
//! The system prompt is laid out from the character's latest turn, and the prompt
//! tokens are recomputed at the moment of the call, so an operator sees what the
//! next turn would start from rather than what the last one used. Owner-scoped
//! reads answer 404 until the character has a turn or a conversation, and an admin
//! may read another character with `?all=true`. The journal routes show the held
//! entries and open items, let an operator draft an entry now, and let one entry
//! be forgotten, which rewrites the record a restart rebuilds the journal from.

use super::*;

/// `GET /v1/npc/:nid/system-prompt` — the system prompt the character's latest
/// turn was made under, as `{pieces, text}`.
///
/// Laid out from that turn's own projection, so the identity, place, tool and
/// mission members shown are the ones the engine selected for it. 404
/// `no_system_prompt` until the character has had a turn since the daemon
/// started. Owner-scoped, with `?all=true` for an admin, like the journal.
pub async fn system_prompt(
    State(s): State<Arc<Authored>>,
    Path(nid): Path<String>,
    Query(q): Query<AllQuery>,
    headers: HeaderMap,
) -> Response {
    let nid = match readable(&s, &headers, &nid, &q).await {
        Ok(id) => id,
        Err(r) => return *r,
    };
    let Some(rt) = s.runtime.as_ref() else {
        return no_scheduler();
    };
    let rendered = rt
        .minds
        .read()
        .unwrap()
        .as_ref()
        .and_then(|m| m.system_prompt(nid));
    match rendered {
        Some(r) => Json(r).into_response(),
        None => err(
            StatusCode::NOT_FOUND,
            "no_system_prompt",
            "this character has not had a turn since the daemon started",
        ),
    }
}

/// `GET /v1/npc/:nid/prompt-tokens` — the tokens the character would be given if
/// it decoded this instant, as `{pieces, token_count, token_ids, text}`.
///
/// The projection is recomputed at the moment of the call, so it is what the
/// next turn would start from rather than what the last one used. Each piece is
/// a section, a sealed turn, live template glue or the pending user message,
/// with its token ids, decoded text, and the belief score that selected it —
/// the way to check that a mission, a tool or a turn is really in the context.
/// 404 `no_conversation` for a character with no live conversation. Owner-scoped,
/// with `?all=true` for an admin, like `system-prompt`.
pub async fn prompt_tokens(
    State(s): State<Arc<Authored>>,
    Path(nid): Path<String>,
    Query(q): Query<AllQuery>,
    headers: HeaderMap,
) -> Response {
    let nid = match readable(&s, &headers, &nid, &q).await {
        Ok(id) => id,
        Err(r) => return *r,
    };
    let Some(rt) = s.runtime.as_ref() else {
        return no_scheduler();
    };
    let minds = rt.minds.read().unwrap().as_ref().map(Arc::clone);
    let dump = match minds {
        Some(m) => m.prompt_dump(nid).await,
        None => None,
    };
    match dump {
        Some(d) => Json(d).into_response(),
        None => err(
            StatusCode::NOT_FOUND,
            "no_conversation",
            "this character has no live conversation to read a prompt from",
        ),
    }
}

/// `GET /v1/npc/:nid/journal` — the character's journal and what its drafts have
/// been doing.
///
/// The entries held in memory as the prompt reads them, which of them the next
/// prompt carries, the open items, where the drafting stands, and the last
/// drafts with how each ended and how its claims were checked.
///
/// Owner-scoped like every route about one character. An admin may add
/// `?all=true` to read a journal they do not own, as the pulse feed does; the
/// flag is a request, honoured only for a caller who is one.
pub async fn journal(
    State(s): State<Arc<Authored>>,
    Path(nid): Path<String>,
    Query(q): Query<AllQuery>,
    headers: HeaderMap,
) -> Response {
    let nid = match readable(&s, &headers, &nid, &q).await {
        Ok(id) => id,
        Err(r) => return *r,
    };
    let Some(rt) = s.runtime.as_ref() else {
        return no_scheduler();
    };
    match rt.scheduler.journal_of(nid, JournalState::view) {
        Some(view) => Json(view).into_response(),
        None => err(
            StatusCode::SERVICE_UNAVAILABLE,
            "not_awake",
            "the character exists but is not in the scheduler",
        ),
    }
}

/// `POST /v1/npc/:nid/journal/draft` — draft an entry now, without waiting for
/// the cadence. The draft may still skip the stretch, as any draft may once the
/// journal holds an entry.
///
/// Answers 202 with the turns the draft covers; how it ends shows in the
/// journal's `drafts`. 409 while a draft is already running, when nothing has
/// happened since the last entry, or while the character is suspended.
pub async fn journal_draft(
    State(s): State<Arc<Authored>>,
    Path(nid): Path<String>,
    headers: HeaderMap,
) -> Response {
    let nid = match owned(&s, &headers, &nid).await {
        Ok(id) => id,
        Err(r) => return *r,
    };
    if let Some(r) = suspend::refused(&s, nid) {
        return r;
    }
    let Some(rt) = s.runtime.as_ref() else {
        return no_scheduler();
    };
    let Some(due) = rt.scheduler.journal_begin(nid) else {
        return err(
            StatusCode::CONFLICT,
            "nothing_to_draft",
            "a draft is already running, or nothing has happened since the last entry",
        );
    };
    let covers = json!({
        "from_turn": due.span.from_turn,
        "to_turn": due.span.to_turn,
        "turns": due.span.turns.len(),
    });
    rt.spawn_journal(nid, due);
    (StatusCode::ACCEPTED, Json(covers)).into_response()
}

/// `DELETE /v1/npc/:nid/journal/:eid` — forget one journal entry.
///
/// The entry leaves the journal the character's next prompt is built from, its
/// section is removed from the character's conversation, and the record a
/// restart rebuilds the journal from is rewritten without it. Entry ids are not
/// reused. Answers the journal as it now reads; 404 `no_such_entry` when the
/// character holds no entry by that number (only the newest few are held).
pub async fn journal_forget(
    State(s): State<Arc<Authored>>,
    Path((nid, eid)): Path<(String, u64)>,
    headers: HeaderMap,
) -> Response {
    let nid = match owned(&s, &headers, &nid).await {
        Ok(id) => id,
        Err(r) => return *r,
    };
    let Some(rt) = s.runtime.as_ref() else {
        return no_scheduler();
    };
    let Some((gone, prompt)) = rt.scheduler.journal_forget(nid, &[eid]) else {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "not_awake",
            "the character exists but is not in the scheduler",
        );
    };
    if gone.is_empty() {
        return err(
            StatusCode::NOT_FOUND,
            "no_such_entry",
            "this character holds no journal entry by that number",
        );
    }
    let minds = rt.minds.read().unwrap().as_ref().map(Arc::clone);
    let Some(minds) = minds else {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "no_mind",
            "the entry is forgotten, but the mind is not loaded to drop its section",
        );
    };
    if let Err(e) = minds.hold_journal(nid, &prompt, &gone).await {
        return err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "journal_not_kept",
            &e.to_string(),
        );
    }
    match rt.scheduler.journal_of(nid, JournalState::view) {
        Some(view) => Json(view).into_response(),
        None => err(
            StatusCode::SERVICE_UNAVAILABLE,
            "not_awake",
            "the character exists but is not in the scheduler",
        ),
    }
}
