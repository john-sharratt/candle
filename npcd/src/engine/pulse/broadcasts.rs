//! The admin routes that put one event or one announcement in front of every
//! character at once.
//!
//! A broadcast reaches characters the caller does not own, which every other
//! route on this daemon refuses, so it is admin-only. It goes out at a salience
//! that preempts, because a world-wide event is one nobody is exempt from. Each
//! character reads it on its own world's clock, so the time is resolved per
//! character before the event reaches the scheduler, whose resolver is not async.

use super::*;

#[derive(Debug, Deserialize)]
pub struct BroadcastBody {
    pub(super) line: String,
}

/// `POST /v1/pulse/broadcast` — one event to every character at once.
///
/// Thunder, an alarm, nightfall. Admin-only: it reaches characters the caller
/// does not own, which every other route on this daemon refuses to do.
pub async fn broadcast(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Json(body): Json<BroadcastBody>,
) -> Response {
    if let Err(r) = owner_of(&s, &headers).await {
        return *r;
    }
    let Some(rt) = s.runtime.as_ref() else {
        return no_scheduler();
    };
    let parsed = match slash::parse(&body.line) {
        Ok(p) => p,
        Err(e) => return err(StatusCode::BAD_REQUEST, "bad_command", &e.message()),
    };
    // A broadcast that only some of the cast notices is not a broadcast. The
    // world-wide events worth sending this way are the ones nobody is exempt
    // from, so it goes out at a salience that preempts.
    let salience = Salience::URGENT.max_of(parsed.salience);
    // Each character's own world time, resolved here rather than inside the
    // broadcast: the resolver is async, and the scheduler's is not.
    let mut when: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
    for c in rt.scheduler.census() {
        when.insert(c.npc_id, s.world_ms(c.npc_id));
    }
    // Each character reads it on its own world's clock; there is no single
    // instant to stamp them all with.
    let reached = rt.scheduler.broadcast(
        |id| when.get(&id).copied().unwrap_or(0),
        salience,
        parsed.kind,
    );
    Json(json!({ "delivered": true, "reached": reached, "salience": salience.get() }))
        .into_response()
}

#[derive(Debug, Deserialize)]
pub struct AnnounceBody {
    /// What is put to the world.
    pub(super) text: String,
}

/// The event an `/announce` becomes, before it meets the scheduler.
///
/// Pure, so the trim and the one refusal — an announcement with nothing in it —
/// are tested without a runtime. There is no salience to choose: an announcement
/// to the whole world is [`Salience::URGENT`] by definition, which the handler
/// supplies.
pub(super) fn announce_event(body: AnnounceBody) -> Option<EventKind> {
    let text = body.text.trim();
    (!text.is_empty()).then(|| EventKind::Announcement {
        text: text.to_owned(),
    })
}

/// `POST /v1/pulse/announce` — put one word to the whole world at once.
///
/// The plain, announcement-shaped sibling of [`broadcast`], and the world-scale
/// counterpart of [`direct`]: no `/` notation to parse and no per-character
/// address, just a line every character reads as a word put to everyone. Always
/// [`Salience::URGENT`] — the only announcements worth sending this way are the
/// ones nobody is exempt from, and it rouses even a waiting character, which is
/// how it fits the cast's ambient idle without a mechanism of its own. Admin,
/// like `broadcast`: it reaches characters the caller does not own.
pub async fn announce(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Json(body): Json<AnnounceBody>,
) -> Response {
    if let Err(r) = owner_of(&s, &headers).await {
        return *r;
    }
    let Some(rt) = s.runtime.as_ref() else {
        return no_scheduler();
    };
    let Some(kind) = announce_event(body) else {
        return err(
            StatusCode::BAD_REQUEST,
            "empty_announcement",
            "an announcement needs something in it",
        );
    };
    // What every character will read. Announcement prose does not depend on the
    // clock, so a nominal stamp is enough for the echo.
    let prose = Event::new(0, 0, Salience::URGENT, kind.clone()).prose();
    // Each character reads it on its own world's clock; there is no single
    // instant to stamp them all with — the same resolve `broadcast` makes.
    let mut when: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
    for c in rt.scheduler.census() {
        when.insert(c.npc_id, s.world_ms(c.npc_id));
    }
    let reached = rt.scheduler.broadcast(
        |id| when.get(&id).copied().unwrap_or(0),
        Salience::URGENT,
        kind,
    );
    Json(json!({
        "delivered": true,
        "reached": reached,
        "salience": Salience::URGENT.get(),
        "prose": prose,
    }))
    .into_response()
}
