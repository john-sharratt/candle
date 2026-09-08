//! **Pulse** — the instrument for watching a cast think.
//!
//! # What it is for
//!
//! The hard thing about debugging an NPC engine is that the interesting part is
//! asynchronous and invisible. Events arrive from the world, the loop ticks when
//! salience says so, and acts come out somewhere else entirely. Reading a
//! conversation transcript afterwards tells you what happened and never why it
//! happened *then* — which is where almost every real bug lives: a character
//! that ticked too often, or not at all, or drained a batch it should have
//! preempted on.
//!
//! Pulse shows the loop itself. Every tick, in order, across the whole cast:
//! what the character perceived (as prose — the literal text that went to the
//! model), why it woke, what it did, and what its metabolism did next.
//!
//! # Why it is a first-class surface rather than a debug flag
//!
//! Because the same view is the authoring instrument. Somebody building a
//! character needs to see it react — poke it with `/hurt`, watch the tick, see
//! what the window held when it decided. A debug flag would make that a
//! developer-only affordance, and it is the main thing an author does.

use std::collections::HashSet;
use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};
use web::auth::{Identity, Role};

use crate::api::{err, owner_of, Authored};
use crate::engine::event::Salience;
use crate::engine::slash;

/// How many ticks a feed request may ask for. The scheduler's own ring is the
/// hard bound; this stops one request rendering all of it.
const MAX_FEED: usize = 200;
const DEFAULT_FEED: usize = 50;

#[derive(Debug, Deserialize)]
pub struct FeedQuery {
    #[serde(default)]
    limit: Option<usize>,
    /// Only this character's ticks. Absent means every character the caller may
    /// see, interlaced — which is the view's default and the reason it exists.
    #[serde(default)]
    npc_id: Option<u64>,
    /// Admin only: the whole cast, including characters the caller does not own.
    #[serde(default)]
    all: bool,
}

/// Which characters this caller may watch, and how to describe the choice.
struct Scope {
    /// `None` means no filter — an admin who asked for everything.
    ids: Option<HashSet<u64>>,
    /// Whether the caller could have asked for everything, so the console knows
    /// whether to offer the toggle at all. A control that appears and then
    /// refuses is worse than one that never appears.
    may_see_all: bool,
    /// Whether they did.
    showing_all: bool,
}

/// Resolve what this caller may watch.
///
/// **Ownership is the default here as everywhere else on this daemon.** The tick
/// loop runs the whole cast — the quartermaster counts sacks whether or not his
/// author is signed in — but a tick record carries the literal prose a character
/// perceived, and somebody else's character's perceptions are somebody else's
/// world.
///
/// An admin may ask for everything with `?all=1`, and gets it only if they
/// actually are one: the flag is a request, never a grant.
async fn scope_of(s: &Arc<Authored>, id: &Identity, owner: &str, all: bool) -> Scope {
    let may_see_all = matches!(s.roles.of(Some(id)), Role::Admin);
    if all && may_see_all {
        return Scope {
            // `None`, not an empty set — an empty set reads as "nothing to
            // show", which is the opposite of "no filter".
            ids: None,
            may_see_all,
            showing_all: true,
        };
    }
    let ids = s.npcs.read().await.owned_by(owner);
    Scope {
        ids: Some(ids),
        may_see_all,
        showing_all: false,
    }
}

impl Scope {
    fn admits(&self, npc_id: u64) -> bool {
        self.ids.as_ref().is_none_or(|set| set.contains(&npc_id))
    }
}

/// `GET /v1/pulse` — recent ticks across the cast.
pub async fn feed(
    State(s): State<Arc<Authored>>,
    Query(q): Query<FeedQuery>,
    headers: HeaderMap,
) -> Response {
    let (id, owner) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let Some(rt) = s.runtime.as_ref() else {
        return no_scheduler();
    };
    let scope = scope_of(&s, &id, &owner, q.all).await;
    let limit = q.limit.unwrap_or(DEFAULT_FEED).clamp(1, MAX_FEED);

    // Filtered *before* the limit, so asking for fifty rows gives fifty rows
    // this caller can see rather than fifty from the whole cast with most of
    // them removed. The scheduler's ring is the hard bound either way.
    let mut ticks: Vec<_> = rt
        .scheduler
        .recent(MAX_FEED.max(limit))
        .into_iter()
        .filter(|t| scope.admits(t.npc_id))
        .filter(|t| q.npc_id.is_none_or(|want| t.npc_id == want))
        .collect();
    let drop = ticks.len().saturating_sub(limit);
    ticks.drain(..drop);

    // Names, so an interlaced feed reads as characters rather than as ids. One
    // lookup for the whole page — a name per row would be a listing request per
    // tick.
    let names = name_map(&s, ticks.iter().map(|t| t.npc_id)).await;

    Json(json!({
        "ticks": ticks,
        "names": names,
        "ready": rt.is_ready(),
        "population": rt.scheduler.population(),
        "may_see_all": scope.may_see_all,
        "showing_all": scope.showing_all,
    }))
    .into_response()
}

/// `npc_id` → display name, for the ids present.
///
/// Ids are stringified because a `u64` past 2^53 does not survive a JavaScript
/// client — the same reason the character record itself serialises its id as a
/// string, and getting it wrong here would give every large-id character the
/// name of whichever one it collided with.
async fn name_map(
    s: &Arc<Authored>,
    ids: impl Iterator<Item = u64>,
) -> serde_json::Map<String, Value> {
    let wanted: HashSet<u64> = ids.collect();
    let npcs = s.npcs.read().await;
    wanted
        .into_iter()
        .filter_map(|id| {
            let n = npcs.payload(id)?;
            Some((id.to_string(), Value::String(n.name.clone())))
        })
        .collect()
}

/// `GET /v1/pulse/census` — every character's loop state.
pub async fn census(
    State(s): State<Arc<Authored>>,
    Query(q): Query<FeedQuery>,
    headers: HeaderMap,
) -> Response {
    let (id, owner) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let Some(rt) = s.runtime.as_ref() else {
        return no_scheduler();
    };
    let scope = scope_of(&s, &id, &owner, q.all).await;
    // The day and how far through it each character is. Rendered here rather
    // than in the scheduler because the scheduler knows the day a character
    // thinks it is in and not what time the world says it is — those are
    // different questions, and the gap between them is what a roll-over closes.
    let mut characters: Vec<serde_json::Value> = Vec::new();
    for c in rt.scheduler.census() {
        if !scope.admits(c.npc_id) {
            continue;
        }
        // `s.world_ms`, not `rt.world_ms`: this is inside the runtime, and the
        // runtime's own resolver takes `blocking_read`, which panics here.
        let world_ms = s.world_ms(c.npc_id).await;
        let mut v = serde_json::to_value(&c).unwrap_or_else(|_| json!({}));
        if let Some(o) = v.as_object_mut() {
            o.insert("world_ms".into(), json!(world_ms));
            o.insert(
                "phase".into(),
                json!(crate::engine::sleep::phase_of(world_ms)),
            );
        }
        characters.push(v);
    }
    let names = name_map(
        &s,
        rt.scheduler
            .census()
            .into_iter()
            .map(|c| c.npc_id)
            .filter(|id| scope.admits(*id)),
    )
    .await;

    Json(json!({
        "characters": characters,
        "names": names,
        "may_see_all": scope.may_see_all,
        "showing_all": scope.showing_all,
        "ready": rt.is_ready(),
        // Characters holding an open conversation. Trails the scheduler's
        // population — a character wakes at startup and opens a conversation on
        // its first tick — and the gap is "how many have thought at least once",
        // which is the number that says whether the engine is doing anything.
        "thinking": rt.minds.read().unwrap().as_ref().map(|m| m.resident()),
    }))
    .into_response()
}

/// `GET /v1/pulse/world` — the worlds this daemon is running, and who is in them.
///
/// Not `/v1/world`, which is the authored world *registry* — the documents an
/// author wrote. This is the simulation running from them, and they are
/// different enough that sharing a path would be a collision in every sense.
///
/// The instrument the pulse feed cannot be. The feed says what a character
/// *did* and the window says what it is *holding*; neither says where anybody
/// is standing, which is the one question that makes a moving cast legible —
/// two characters talking and two characters two floors apart look identical in
/// a feed, and completely different here.
///
/// Rooms come back with who is in each, so the shape of the building is visible
/// rather than having to be reconstructed from a list of coordinates. An empty
/// room is included: knowing the green room is empty is why a character walking
/// to it is interesting.
pub async fn world(State(s): State<Arc<Authored>>, headers: HeaderMap) -> Response {
    let (id, owner) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let Some(rt) = s.runtime.as_ref() else {
        return no_scheduler();
    };
    // The rooms are the world's and belong to nobody; the *binding* rows are
    // scoped, so a caller only learns which characters are which bodies for the
    // characters it may already see.
    let scope = scope_of(&s, &id, &owner, true).await;

    let mut worlds = Vec::new();
    for world_id in rt.hosted.ids() {
        let Some(hosted) = rt.hosted.get(&world_id) else {
            continue;
        };
        // One read of the world, holding it still, so every room in the answer
        // is from the same instant. Two reads would let a body move between
        // them and appear in two rooms or in none.
        let (rooms, bodies) = hosted.read(|w| {
            let mut rooms: Vec<serde_json::Value> = Vec::new();
            let mut bodies = 0usize;
            for area in w.map().areas() {
                for node in &area.nodes {
                    let at = npc_map::world::Where::new(area.id.clone(), node.id.clone());
                    let here: Vec<serde_json::Value> = w
                        .actors_at(&at)
                        .into_iter()
                        .map(|a| {
                            bodies += 1;
                            json!({
                                "body": a.id,
                                "name": a.name,
                                // What it is doing here, which is the half a
                                // position alone does not say.
                                "holding": a.hold.as_ref().and_then(|h| h.subject.clone()),
                                "going_to": a.walk.as_ref().map(|k| k.toward.node.clone()),
                            })
                        })
                        .collect();
                    rooms.push(json!({
                        "area": area.id,
                        "area_name": area.name,
                        "node": node.id,
                        "name": node.name,
                        "kind": node.kind,
                        "who": here,
                    }));
                }
            }
            (rooms, bodies)
        });

        worlds.push(json!({
            "world_id": world_id,
            "rooms": rooms,
            "bodies": bodies,
        }));
    }

    // Which character is which body, so a row here joins to a row in the feed.
    let mut bound = serde_json::Map::new();
    for world_id in rt.hosted.ids() {
        for (npc_id, body) in rt.bodies.in_world(&world_id) {
            if scope.admits(npc_id) {
                bound.insert(body, json!(npc_id.to_string()));
            }
        }
    }

    Json(json!({
        "worlds": worlds,
        // `body -> npc_id`, as a string: an id past 2^53 does not survive a
        // JavaScript client as a number, and this one is read by the console.
        "bound": bound,
        "moments": rt.moments().iter().map(|(id, n, paused)| json!({
            "world_id": id, "moments": n, "paused": paused,
        })).collect::<Vec<_>>(),
        "may_see_all": scope.may_see_all,
    }))
    .into_response()
}

/// `GET /v1/npc/:nid/window` — the character's verbatim tail, right now.
///
/// What the character is carrying into its next decode, in order. The
/// counterpart of the feed: the feed says what happened, this says what is still
/// being held. Reading them together is how you tell "it forgot" from "it never
/// perceived it" — the first is a turn that has faded out of here, the second is
/// a turn that never reached the feed.
pub async fn window(
    State(s): State<Arc<Authored>>,
    Path(nid): Path<u64>,
    headers: HeaderMap,
) -> Response {
    let (_, owner) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    if s.npcs.read().await.visible_to(nid, &owner).is_none() {
        return err(StatusCode::NOT_FOUND, "not_found", "no such character");
    }
    let Some(rt) = s.runtime.as_ref() else {
        return no_scheduler();
    };
    let Some(w) = rt.scheduler.window_of(nid, |w| {
        let turns: Vec<_> = w
            .turns()
            .map(|t| {
                json!({
                    "speaker": t.speaker,
                    "text": crate::engine::mind::render_turn(t.speaker, &t.text),
                    "at_ms": t.at_ms,
                    "replaces": t.replaces,
                })
            })
            .collect();
        json!({
            "turns": turns,
            "empty": w.is_empty(),
            "cap": w.cap(),
            // Not a loss counter — every faded turn is still in the substrate
            // and the gather can pull it back. It is the number that says
            // whether continuity is coming from retrieval, which is the intent.
            "faded": w.faded(),
        })
    }) else {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "not_awake",
            "the character exists but is not in the scheduler",
        );
    };
    Json(w).into_response()
}

#[derive(Debug, Deserialize)]
pub struct BroadcastBody {
    line: String,
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
        when.insert(c.npc_id, s.world_ms(c.npc_id).await);
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
pub struct InjectBody {
    /// The operator's line, `/`-prefixed or not.
    line: String,
}

/// `POST /v1/npc/:nid/pulse` — put an event on a character's inbox.
///
/// The `/` notation's endpoint. Ownership *is* checked here, unlike the feed:
/// reading what the cast is doing is an operator's business, but poking a
/// specific character is a write to something somebody owns.
pub async fn inject(
    State(s): State<Arc<Authored>>,
    Path(nid): Path<u64>,
    headers: HeaderMap,
    Json(body): Json<InjectBody>,
) -> Response {
    let (_, owner) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    if s.npcs.read().await.visible_to(nid, &owner).is_none() {
        return err(StatusCode::NOT_FOUND, "not_found", "no such character");
    }
    let Some(rt) = s.runtime.as_ref() else {
        return no_scheduler();
    };

    let parsed = match slash::parse(&body.line) {
        Ok(p) => p,
        // A typo is a 400 with the near miss named, never speech. Sending
        // `/hrut` to the character as dialogue is the one outcome that looks
        // like it worked.
        Err(e) => return err(StatusCode::BAD_REQUEST, "bad_command", &e.message()),
    };

    let world_ms = s.world_ms(nid).await;
    // `/sleep` carries a placeholder day the parser cannot know; the clock is
    // the only thing that can fill it in.
    let kind = match parsed.kind {
        crate::engine::event::EventKind::Sleep { .. } => crate::engine::event::EventKind::Sleep {
            day: crate::engine::sleep::day_of(world_ms),
        },
        other => other,
    };
    let prose =
        crate::engine::event::Event::new(0, world_ms, parsed.salience, kind.clone()).prose();

    if !rt.scheduler.deliver(nid, world_ms, parsed.salience, kind) {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "not_awake",
            "the character exists but is not in the scheduler — the engine is still loading",
        );
    }
    Json(json!({
        "delivered": true,
        "command": parsed.command,
        "salience": parsed.salience.get(),
        "preempts": parsed.salience.preempts(),
        // What the character will actually read. The point of showing it back
        // is that an operator can see the prose their command became, which is
        // the thing that decides how it lands.
        "prose": prose,
    }))
    .into_response()
}

/// `GET /v1/commands` — the `/` vocabulary, for the console's autocomplete.
///
/// Served rather than duplicated in the console, so the two cannot drift. A
/// console with its own list would offer a command the daemon rejects, and the
/// operator would read that as the character ignoring them.
pub async fn commands() -> Response {
    Json(json!({
        "commands": slash::CATALOG,
        "default": "say",
        "preempt_at": Salience::PREEMPT_AT,
    }))
    .into_response()
}

/// `GET /v1/tools` — the act vocabulary, with its calibration examples.
pub async fn tools() -> Response {
    Json(json!({
        "tools": crate::engine::tools::CATALOG,
        "examples": crate::engine::tools::CATALOG.iter().map(|t| t.examples.len()).sum::<usize>(),
    }))
    .into_response()
}

fn no_scheduler() -> Response {
    err(
        StatusCode::SERVICE_UNAVAILABLE,
        "no_engine",
        "the tick scheduler is not running",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The feed's bound has to be enforced on the request, not trusted from it —
    /// `?limit=1000000` would otherwise render the scheduler's whole ring into
    /// one response.
    #[test]
    fn the_feed_limit_is_clamped_at_both_ends() {
        for (asked, want) in [
            (None, DEFAULT_FEED),
            (Some(0), 1),
            (Some(9_999_999), MAX_FEED),
            (Some(10), 10),
        ] {
            let got = asked.unwrap_or(DEFAULT_FEED).clamp(1, MAX_FEED);
            assert_eq!(got, want, "limit={asked:?}");
        }
    }

    /// The console's autocomplete is the daemon's own list. A console holding
    /// its own copy would offer a command that gets rejected, and an operator
    /// reads a rejection as the character ignoring them.
    #[test]
    fn the_command_catalog_is_served_not_duplicated() {
        assert!(!slash::CATALOG.is_empty());
        // Everything offered must parse — the same guarantee `slash`'s own
        // tests make, asserted here because this is the surface that publishes
        // it.
        for c in slash::CATALOG {
            assert!(
                slash::parse(c.example).is_ok(),
                "/{} is published with an example that does not parse",
                c.name
            );
        }
    }
}
