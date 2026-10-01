//! `http://local/history` — read-only onto what this body has done and seen
//! (effector design §7.2, C.27).
//!
//! The character's own window onto its past: the stream of what it could make
//! out from where it was standing, oldest first, as JSON. It is a **read** —
//! `GET` only — and, crucially, a *non-destructive* one.
//!
//! # It peeks; it never marks seen
//!
//! The NPC engine drives perception off the same log through the reader's
//! `looked` cursor: a body is handed what it has not yet been shown, and the
//! engine advances the cursor when it delivers it ([`npc_map::World::mark_seen`],
//! via the attention sweep). This route reads that same window with
//! [`npc_map::witness::since`], which takes `&World` and touches nothing — so a
//! character (or an operator) reading its history through the device does not
//! consume the delta the engine is about to deliver, and cannot make the engine
//! skip a moment. `since` is the peek; `mark_seen` is the only thing that spends,
//! and this route never calls it.
//!
//! Because it reads the *unseen-since-looked* window, the answer is the body's
//! recent past — what has happened around it that its mind has not yet folded in
//! — narrowed to what it could legibly witness (`witness::since` applies the
//! room/sight/whisper rules). Its own doings are left out except how a journey
//! it began turned out, which is exactly what `since` already decides.
//!
//! # Pagination
//!
//! `?cursor=<n>` and `?limit=<n>` page the window, oldest first. The cursor is a
//! **position** in the witnessed window — how many events the caller has already
//! consumed — not a tick, and a page returns up to `limit` events starting
//! there. When more remain, `next_cursor` is the position to resume at;
//! otherwise it is `null`. Paginating on position rather than tick is what keeps
//! same-tick events whole: many events are witnessed on one tick, and a `> tick`
//! filter would split them across a page boundary and silently drop the ones
//! past the limit — a position never can.

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use serde::Deserialize;
use serde_json::{json, Value};

use npc_map::witness::{self, Witnessed};
use npc_map::world::World;

use crate::api::err;
use crate::effector::auth::DeviceCaller;
use crate::effector::router::Local;

/// Events returned when no `limit` is given — a bounded recent window rather
/// than the whole log.
const DEFAULT_LIMIT: usize = 50;

/// The most events one page will ever return, whatever `limit` asks for.
const MAX_LIMIT: usize = 200;

/// The page a caller asks for: everything after `cursor`, up to `limit`.
#[derive(Debug, Default, Deserialize)]
pub(crate) struct Page {
    /// Resume at this position in the witnessed window — the count of events
    /// already consumed, taken from a previous page's `next_cursor`. Absent
    /// starts from the beginning. A position, not a tick, so events that share a
    /// tick are never split across a page boundary and lost.
    cursor: Option<u64>,
    /// How many events to return at most. Absent is [`DEFAULT_LIMIT`], and
    /// anything larger than [`MAX_LIMIT`] is capped there.
    limit: Option<usize>,
}

/// `GET /history` — the body's witnessed past, paginated, oldest first.
///
/// A pure read under one lock: [`witness::since`] is non-destructive, so this
/// does not advance the engine's `looked` cursor.
pub(crate) async fn history(
    State(local): State<Local>,
    Extension(caller): Extension<DeviceCaller>,
    Query(page): Query<Page>,
) -> Response {
    let Some((hosted, body)) = local.body_of(&caller) else {
        return no_history();
    };
    let limit = page.limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT);
    let start = page.cursor.unwrap_or(0) as usize;

    let answer = hosted.read(|w| {
        // Non-destructive: `since` reads the log through the reader's cursor and
        // marks nothing. The engine's delta is untouched.
        let seen = witness::since(w, &body);
        let (range, next_cursor) = page_of(seen.len(), start, limit);
        let events: Vec<Value> = seen[range].iter().map(|wit| event_json(w, wit)).collect();
        json!({ "events": events, "next_cursor": next_cursor })
    });
    Json(answer).into_response()
}

/// The slice of a `total`-long witnessed window to return, and where to resume.
///
/// Pagination is **positional** — it never looks at ticks. Many events share a
/// tick, so resuming after a tick (the earlier bug) skipped every same-tick
/// event that fell past the page's `limit`; a position cannot split a tick. The
/// range is `[start, start + limit)` clamped to the window, and `next_cursor` is
/// the position to resume at, or `None` when the window is exhausted. A `start`
/// past the end yields an empty page and no cursor.
fn page_of(total: usize, start: usize, limit: usize) -> (std::ops::Range<usize>, Option<u64>) {
    let start = start.min(total);
    let end = start.saturating_add(limit).min(total);
    let next = (end < total).then_some(end as u64);
    (start..end, next)
}

/// One witnessed event as JSON: when, who a reader would name did it, whether it
/// was in the reader's own room, whether it was the reader's own doing, and the
/// line npcd already narrates it as ([`witness::narrate`]).
fn event_json(world: &World, event: &Witnessed) -> Value {
    let what = witness::narrate(world, std::slice::from_ref(event)).unwrap_or_default();
    json!({
        "at": event.at,
        "who": event.name,
        "here": event.here,
        "mine": event.mine(),
        "what": what,
    })
}

/// The refusal for a caller whose body cannot be resolved to a hosted world —
/// a body that is nowhere has no past to read.
fn no_history() -> Response {
    err(
        StatusCode::NOT_FOUND,
        "no_history",
        "you have no body in a running world, so there is no history to read",
    )
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;

    use axum::body::Body;
    use axum::http::header::AUTHORIZATION;
    use axum::http::{Request, StatusCode};
    use npc_map::world::Where;
    use serde_json::Value;
    use tower::ServiceExt;

    use super::page_of;
    use crate::effector::router::{router, Local};
    use crate::effector::token::{Scope, Tokens};
    use crate::engine::runtime::Runtime;
    use crate::mind::Mind;

    const ROOMS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps");
    const WORLD: &str = "creators-vault";

    // ── the positional pagination, in isolation ─────────────────────────────

    /// Each page is `[start, start + limit)` clamped to the window, and the
    /// cursor resumes exactly where the page stopped — `null` once exhausted.
    #[test]
    fn page_of_returns_bounded_pages_and_resumes_where_it_stopped() {
        assert_eq!(page_of(5, 0, 2), (0..2, Some(2)));
        assert_eq!(page_of(5, 2, 2), (2..4, Some(4)));
        assert_eq!(page_of(5, 4, 2), (4..5, None));
        // A start at or past the end is an empty last page, no cursor.
        assert_eq!(page_of(5, 5, 2), (5..5, None));
        assert_eq!(page_of(5, 9, 2), (5..5, None));
        // The whole window in one page.
        assert_eq!(page_of(3, 0, 50), (0..3, None));
    }

    /// **The regression #3 guards: same-tick events are never dropped at a page
    /// boundary.** Five events all witnessed on one tick — the case tick-based
    /// pagination lost, because it resumed *after* the tick and the next page
    /// filtered every same-tick event out. Positional pagination walks all five
    /// across pages with no loss and no repeat, whatever the limit.
    #[test]
    fn paging_a_window_of_same_tick_events_loses_none() {
        let window = [7u64, 7, 7, 7, 7]; // five events, one shared tick
        for limit in 1..=6 {
            let mut collected: Vec<u64> = Vec::new();
            let mut start = 0usize;
            loop {
                let (range, next) = page_of(window.len(), start, limit);
                collected.extend(window[range].iter().copied());
                match next {
                    Some(n) => start = n as usize,
                    None => break,
                }
            }
            assert_eq!(
                collected, window,
                "limit {limit}: a same-tick event was dropped or repeated"
            );
        }
    }

    // ── the route, end to end over a socket ─────────────────────────────────

    fn tmp() -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "npcd-history-ut-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn daemon() -> (Arc<Runtime>, Arc<Tokens>) {
        let rt = Runtime::new(Mind::new(None), &std::env::temp_dir());
        rt.host(WORLD, Path::new(ROOMS)).expect("the vault loads");
        rt.hold_world(WORLD, true);
        let tokens = Arc::new(Tokens::load(tmp()).expect("a fresh token store"));
        rt.set_tokens(tokens.clone());
        (rt, tokens)
    }

    fn place(rt: &Arc<Runtime>, npc_id: u64, node: &str) {
        let body = Runtime::body_id(npc_id);
        let world = rt.hosted.get(WORLD).expect("hosted");
        world.with(|w| {
            w.enter(
                &body,
                format!("Maker-{npc_id:02}"),
                Where::new("vault-casting", node),
            )
            .expect("a real room");
        });
        rt.bodies.bind(npc_id, WORLD, &body).expect("bound");
    }

    async fn history_page(
        rt: &Arc<Runtime>,
        tokens: &Arc<Tokens>,
        npc_id: u64,
        query: &str,
    ) -> Value {
        let token = tokens.mint(npc_id, Scope::AsNpc).expect("minted");
        let app = router(Local::new(tokens.clone(), rt));
        let req = Request::builder()
            .uri(format!("/history{query}"))
            .header(AUTHORIZATION, format!("Bearer {token}"))
            .body(Body::empty())
            .unwrap();
        let res = app.oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(res.into_body(), 1 << 20)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    /// **The route pages the witnessed window and resumes with `next_cursor`
    /// without loss or repeat.** A reader watches a speaker say five lines;
    /// paging two at a time and following `next_cursor` to the end collects every
    /// line exactly once, and the final page reports `next_cursor: null`.
    #[tokio::test]
    async fn the_route_pages_the_window_and_resumes_without_loss() {
        let (rt, tokens) = daemon();
        place(&rt, 30, "band-one"); // the reader
        place(&rt, 31, "band-one"); // the speaker, same room

        let world = rt.hosted.get(WORLD).expect("hosted");
        let speaker = Runtime::body_id(31);
        let lines = ["alpha", "bravo", "charlie", "delta", "echo"];
        world.with(|w| {
            for line in lines {
                w.say(&speaker, line).expect("said");
            }
        });

        // Page two at a time, following the cursor to the end. Guard the loop so
        // a pagination bug that never terminates fails the test rather than hangs.
        let mut seen_lines: Vec<String> = Vec::new();
        let mut query = "?limit=2".to_string();
        let mut saw_null_cursor = false;
        for _ in 0..20 {
            let page = history_page(&rt, &tokens, 30, &query).await;
            for e in page["events"].as_array().expect("an events array") {
                seen_lines.push(e["what"].as_str().unwrap_or_default().to_string());
            }
            match page["next_cursor"].as_u64() {
                Some(n) => query = format!("?limit=2&cursor={n}"),
                None => {
                    saw_null_cursor = true;
                    break;
                }
            }
        }
        assert!(saw_null_cursor, "pagination never reached a null cursor");

        // Every spoken line was witnessed and returned exactly once.
        for line in lines {
            let hits = seen_lines.iter().filter(|s| s.contains(line)).count();
            assert_eq!(
                hits, 1,
                "`{line}` appeared {hits} times across the pages: {seen_lines:?}"
            );
        }
    }
}
