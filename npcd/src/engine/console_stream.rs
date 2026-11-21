//! Streams a character's acts to a console as they happen, and resumes an
//! attachment from the tick the console already holds.
//!
//! The stream polls the scheduler's own ring rather than a second push path, so
//! it cannot disagree with the Pulse view. A returning console resumes after the
//! last tick it read. That tick is clamped to the newest one, so a tick from before
//! a daemon restart cannot leave a connection silent. A fresh attachment starts at
//! the newest tick, so nobody is replayed a whole afternoon. The stream stays open
//! while a person is in the room and ends when the session goes quiet.

use super::*;

/// How often the stream looks for new ticks.
///
/// **Polled off the scheduler's own ring rather than pushed from the decode
/// loop.** A broadcast channel out of `record_act` would be a second path by
/// which an act becomes observable, and the two would drift — the Pulse view
/// already reads this ring, so a session watching the same rows is watching the
/// same truth. The cost is the latency below, against a character that thinks
/// in seconds.
pub(super) const STREAM_POLL: std::time::Duration = std::time::Duration::from_millis(500);

/// Which tick a console's attachment starts after.
///
/// **`since` is what makes coming back different from starting again.** A
/// conversation outlives the page it is watched from: look at Pulse for a
/// minute and the character goes on acting the whole time. Starting every
/// attachment from the newest tick threw all of that away — you came back to
/// the transcript you left and the minute in between had simply not happened,
/// which is the wrong answer to the one question somebody returning is asking.
///
/// So a console names the last tick it already has and gets what followed.
/// Clamped to `latest`, because a tick from the future — a console that
/// outlived a daemon restart, a hand-typed URL — would otherwise produce a
/// stream that connects and never sends anything, which is indistinguishable
/// from a character that has stopped thinking. Absent or unparseable means a
/// fresh attachment: `latest`, so nobody is replayed a whole afternoon.
///
/// The honest limit is above this function: the scheduler keeps a bounded
/// window, so a console away for longer than that memory gets what is left of
/// the interval rather than all of it.
pub(super) fn resume_from(since: Option<&str>, latest: u64) -> u64 {
    since
        .and_then(|v| v.parse::<u64>().ok())
        .map_or(latest, |t| t.min(latest))
}

/// `GET /v1/interaction/:ix/stream` — what the character does, as it does it.
///
/// Frames are the console's: `open` once, then `act` per act with `tick` at the
/// close of each. There is no `narration` frame, and its absence is honest —
/// this daemon renders an act *as* its own line rather than producing a
/// separate account afterwards, so a narration frame would be prose nothing
/// wrote.
///
/// `?since=<tick>` resumes an attachment where a console left off — see
/// [`resume_from`], which is the difference between coming back to a
/// conversation and starting a new one.
pub(super) async fn stream(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(ix): Path<String>,
    Query(q): Query<BTreeMap<String, String>>,
) -> Response {
    let Some(rt) = s.runtime.as_ref() else {
        return no_engine("streaming an interaction");
    };
    let now = crate::api::now_ms();
    let Some(session) = rt.interactions.get(&ix, now) else {
        return err(StatusCode::NOT_FOUND, "interaction_not_found", &ix);
    };
    if let Err(r) = owned(&s, &headers, &npcs::npc_id_wire(session.npc_id)).await {
        return *r;
    }

    let rt = rt.clone();
    let npc_id = session.npc_id;

    // The newest tick this character has taken. Where a fresh console starts,
    // so attaching to a live session does not replay the whole afternoon.
    let latest: u64 = rt.scheduler.latest_tick_of(npc_id).unwrap_or(0);

    let mut seen: u64 = resume_from(q.get("since").map(String::as_str), latest);

    let open = Event::default().event("open").data(
        json!({
            "interaction_id": ix,
            "mode": session.mode.as_wire(),
            "resume_from": seen,
        })
        .to_string(),
    );

    // A channel and a task rather than a generator: `async-stream` is not a
    // dependency here, and the task ends by itself when the receiver is dropped
    // — a console that navigates away closes the connection, the send fails,
    // and the loop stops. Nothing has to notice the disconnect separately.
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(64);
    tokio::spawn(async move {
        if tx.send(Ok(open)).await.is_err() {
            return;
        }
        loop {
            tokio::time::sleep(STREAM_POLL).await;
            let now = crate::api::now_ms();
            // **Holding this stream open is being in the room.** Idle catches
            // the person who walked off; without this it also caught the one
            // who sat and listened, because it was measured from the last line
            // rather than from the last sign of anybody being there. A console
            // that navigates away closes the connection and this loop ends, so
            // the timeout still does its job the moment nobody is watching.
            rt.interactions.attended(&ix, now);
            // The session going quiet ends the stream, so a console left open
            // is not holding a connection against a conversation that is over.
            if rt.interactions.get(&ix, now).is_none() {
                let _ = tx
                    .send(Ok(Event::default()
                        .event("state")
                        .data(json!({ "state": "ended" }).to_string())))
                    .await;
                return;
            }
            let fresh: Vec<_> = rt
                .scheduler
                .recent_where(512, |t| t.npc_id == npc_id && t.tick > seen);
            for t in fresh {
                seen = seen.max(t.tick);
                for (n, act) in t.acts.iter().enumerate() {
                    // `Act::summary` renders `tool — args`; the console wants
                    // the two apart so it can label one and quote the other.
                    let (tool, intent) = match act.split_once(" — ") {
                        Some((a, b)) => (a, b),
                        None => (act.as_str(), ""),
                    };
                    let frame = Event::default().event("act").data(
                        json!({
                            "act_id": format!("{}-{n}", t.tick),
                            "tick": t.tick,
                            "tool": tool,
                            "intent": intent,
                            "world_ms": t.world_ms,
                        })
                        .to_string(),
                    );
                    if tx.send(Ok(frame)).await.is_err() {
                        return;
                    }
                }
                let close = Event::default()
                    .event("tick")
                    .data(json!({ "tick": t.tick, "acts": t.acts.len() }).to_string());
                if tx.send(Ok(close)).await.is_err() {
                    return;
                }
            }
        }
    });
    Sse::new(tokio_stream::wrappers::ReceiverStream::new(rx))
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}
