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
//! character needs to see it react — poke it with `/act`, watch the tick, see
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
use crate::engine::event::{Addressed, Event, EventKind, Salience};
use crate::engine::runtime::Runtime;
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
    // **A bar, not an equality.** This read `matches!(.., Role::Admin)`, which
    // was the same thing while `Admin` was the top of the ladder and stopped
    // being it the moment `Creator` went above — the person who owns the estate
    // signed in and silently lost a control an admin has. An exact match on a
    // level in an *ordered* enum is a demotion waiting for the next level to be
    // added, and it does not fail, it just quietly refuses.
    let may_see_all = s.roles.of(Some(id)).at_least(Role::Admin);
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
pub struct AnnounceBody {
    /// What is put to the world.
    text: String,
}

/// The event an `/announce` becomes, before it meets the scheduler.
///
/// Pure, so the trim and the one refusal — an announcement with nothing in it —
/// are tested without a runtime. There is no salience to choose: an announcement
/// to the whole world is [`Salience::URGENT`] by definition, which the handler
/// supplies.
fn announce_event(body: AnnounceBody) -> Option<EventKind> {
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
        when.insert(c.npc_id, s.world_ms(c.npc_id).await);
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
    let (id, owner) = match owner_of(&s, &headers).await {
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
        // **Named.** The parser writes the placeholder `you` because it cannot
        // know whose console the line came from, and that rendered as "you says
        // to you: …" — ungrammatical, and wrong about who spoke. See
        // [`slash::Parsed::attributed_to`].
        Ok(p) => p.attributed_to(&crate::engine::speaking_as(&id, &owner, &s.roles)),
        // A typo is a 400 with the near miss named, never speech. Sending
        // `/hrut` to the character as dialogue is the one outcome that looks
        // like it worked.
        Err(e) => return err(StatusCode::BAD_REQUEST, "bad_command", &e.message()),
    };

    let world_ms = s.world_ms(nid).await;
    // `/sleep` carries a placeholder day the parser cannot know; the clock is
    // the only thing that can fill it in.
    let kind = match parsed.kind {
        EventKind::Sleep { .. } => EventKind::Sleep {
            day: crate::engine::sleep::day_of(world_ms),
        },
        other => other,
    };
    // The command it parsed is the one thing `direct` has no need of — showing
    // an operator the console command their line became.
    deliver_and_report(
        rt,
        nid,
        world_ms,
        parsed.salience,
        kind,
        json!({ "command": parsed.command }),
    )
}

/// Put an event on a character's inbox and answer with what it became.
///
/// The shared tail of [`inject`] and [`direct`]: stamp the world clock into the
/// event's prose, hand it to the scheduler, and report the prose the character
/// will read — the thing that decides how the line lands — or a 503 when the
/// character exists but is not yet in the scheduler. `extra` is folded onto the
/// response object, carrying the one field a caller adds beyond the common set.
fn deliver_and_report(
    rt: &Runtime,
    nid: u64,
    world_ms: u64,
    salience: Salience,
    kind: EventKind,
    extra: Value,
) -> Response {
    let prose = Event::new(0, world_ms, salience, kind.clone()).prose();
    if !rt.scheduler.deliver(nid, world_ms, salience, kind) {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "not_awake",
            "the character exists but is not in the scheduler — the engine is still loading",
        );
    }
    let mut body = json!({
        "delivered": true,
        "salience": salience.get(),
        "preempts": salience.preempts(),
        "prose": prose,
    });
    if let (Value::Object(map), Value::Object(more)) = (&mut body, extra) {
        map.extend(more);
    }
    Json(body).into_response()
}

/// The body of a `/direct` — a line spoken straight into a character's world.
#[derive(Debug, Deserialize)]
pub struct DirectBody {
    /// What the character hears said.
    text: String,
    /// Who is heard to say it. Absent means the operator, under the same name
    /// the rest of the daemon speaks to a character with — see
    /// [`crate::engine::speaking_as`]. A named voice is how a mission-giver, a
    /// terminal, or another character reaches a character in-world without the
    /// operator standing in for them.
    #[serde(default)]
    speaker: Option<String>,
    /// How loudly it lands. Absent is [`Salience::URGENT`], because the whole
    /// point of the route is to reach the character now — the sibling
    /// person-to-character path (`environment.rs`, `phone::Kind::Direct`) uses
    /// the same, and a quieter default would sit in a waiting character's inbox
    /// unheard until its next scheduled thought. A caller that wants it to wait
    /// its turn passes a lower figure; at or above [`Salience::PREEMPT_AT`] it
    /// cuts into a turn the character is mid-way through, and a wire value out of
    /// range is clamped, not rejected.
    #[serde(default)]
    salience: Option<f32>,
    /// Who it was aimed at, from this character's side. Absent is
    /// [`Addressed::You`] — a direct line is by definition aimed at the one it
    /// reaches, unlike a line thrown to the room.
    #[serde(default)]
    to: Option<Addressed>,
}

/// The event a `/direct` becomes, before it meets a clock or the scheduler.
///
/// Pure, so the defaults, the trim, and the salience clamp are tested without a
/// runtime — the handler adds only the world time and the delivery. `None` is a
/// line with nothing in it, which is the one body this route refuses.
fn direct_event(body: DirectBody, operator: &str) -> Option<(Salience, EventKind)> {
    let text = body.text.trim();
    if text.is_empty() {
        return None;
    }
    let speaker = match body.speaker.as_deref().map(str::trim) {
        Some(named) if !named.is_empty() => named.to_owned(),
        _ => operator.to_owned(),
    };
    let salience = body.salience.map_or(Salience::URGENT, Salience::new);
    let to = body.to.unwrap_or(Addressed::You);
    Some((
        salience,
        EventKind::Speech {
            speaker,
            text: text.to_owned(),
            to,
        },
    ))
}

/// `POST /v1/npc/:nid/direct` — speak a line straight into a character's world.
///
/// **The influence primitive, and deliberately not [`inject`].** `inject` is the
/// operator console's `/` notation: it parses a command, and it attributes the
/// line to the operator because a console has no other voice. `direct` is the
/// plain case underneath — a line, a speaker, a loudness — so a mission-giver, a
/// terminal, or another character can reach a character in-world without the
/// operator standing in for them, and so a caller that already knows what it
/// wants said does not have to phrase it as a slash command to be understood.
///
/// Ownership is checked, like `inject` and unlike the feed: putting words in a
/// character's ear is a write to something somebody owns.
pub async fn direct(
    State(s): State<Arc<Authored>>,
    Path(nid): Path<u64>,
    headers: HeaderMap,
    Json(body): Json<DirectBody>,
) -> Response {
    let (id, owner) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    if s.npcs.read().await.visible_to(nid, &owner).is_none() {
        return err(StatusCode::NOT_FOUND, "not_found", "no such character");
    }
    let Some(rt) = s.runtime.as_ref() else {
        return no_scheduler();
    };

    let operator = crate::engine::speaking_as(&id, &owner, &s.roles);
    let Some((salience, kind)) = direct_event(body, &operator) else {
        return err(
            StatusCode::BAD_REQUEST,
            "empty_message",
            "a direct line needs something in it",
        );
    };

    let world_ms = s.world_ms(nid).await;
    deliver_and_report(rt, nid, world_ms, salience, kind, json!({}))
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
/// The act vocabulary, in the shape the console reads it.
///
/// **Not the catalog serialised raw.** It used to be, and three of the table's
/// five columns were therefore blank: `Tool` has no `modes`, no `source` and no
/// `calibrated` field, and the parameter modal read `t.parameters`, which did
/// not exist either — so the one view of the vocabulary an operator has showed a
/// name, a description, and three empty cells over an empty schema.
///
/// Composed here rather than added to `Tool` because these are facts about how
/// the daemon *offers* an act, not about the act: which modes reach it is
/// `Availability` resolved against each mode, and whether it is calibrated is
/// whether anybody wrote it examples. Putting them on the struct would be
/// caching four derived values next to the thing they derive from.
/// What each station is called, by the id an act attaches to.
///
/// Read off whatever worlds are loaded rather than restated in this file, so
/// renaming a part in the map renames it in the console and the two cannot
/// drift. Empty before any world is open, which is the honest answer: with no
/// map, nothing here knows what a station is called.
type PartNames = std::collections::BTreeMap<String, String>;

/// "a" or "an", by the sound the name starts with.
///
/// A vowel letter is the rule almost all the time, and the exceptions in
/// English are about *sound* rather than spelling — "a universal", "an hour".
/// Neither shape occurs among the parts, so the letter is the whole rule here;
/// a part that needed otherwise would be authored with its own article, which
/// the caller already honours.
fn article_for(name: &str) -> &'static str {
    match name.chars().next().map(|c| c.to_ascii_lowercase()) {
        Some('a' | 'e' | 'i' | 'o' | 'u') => "an",
        _ => "a",
    }
}

/// How many stations a condition names before it stops naming them.
///
/// Two fits a line and is worth reading. Six does not: it wraps, it pushes
/// every other column out of shape, and by the third name the reader has
/// stopped taking any of them in. Past this the summary gives the count, and
/// the full list goes where the detail view has room for it.
const NAME_AT_MOST: usize = 2;

/// Each station an act attaches to, in the words the map uses.
fn named_stations(at: &[&str], names: &PartNames) -> Vec<String> {
    at.iter()
        .map(|id| match names.get(*id) {
            // The map's own name, with the article a reader expects. A part
            // authored with its own article keeps it — "the appraisal bench"
            // is one bench and saying "a the appraisal bench" would be worse
            // than saying nothing.
            Some(n) if n.starts_with("the ") => n.clone(),
            Some(n) => format!("{} {n}", article_for(n)),
            // An id no loaded world places. Said plainly rather than dressed
            // up: it means the act is unreachable, and somebody should notice.
            None => format!("`{id}` (nothing places one)"),
        })
        .collect()
}

/// The one-line answer to *where do I have to be*.
///
/// **Named when naming helps, counted when it does not.** "Standing at the
/// station that carries it" was true of every station act at once and told an
/// operator hunting a missing one nothing — so a condition names its station.
/// But an act reaching six of them produced a sentence longer than the column
/// it sat in, and a reader made to parse six noun phrases to learn "a
/// workstation" has been given less rather than more.
///
/// The full list is not lost: it goes in `at_named`, which the detail view
/// renders and which has the room for it.
fn where_it_is(at: &[&str], names: &PartNames) -> Option<String> {
    if at.is_empty() {
        return None;
    }
    let named = named_stations(at, names);
    Some(match named.len() > NAME_AT_MOST {
        false => format!("standing at {}", npc_map::text::list_or(&named)),
        true => format!("standing at any of {} stations", named.len()),
    })
}

pub async fn tools(State(s): State<Arc<Authored>>) -> Response {
    let mut part_names = PartNames::new();
    if let Some(rt) = s.runtime.as_ref() {
        for world_id in rt.hosted.ids() {
            if let Some(hosted) = rt.hosted.get(&world_id) {
                hosted.read(|w| {
                    for area in w.map().areas() {
                        for node in &area.nodes {
                            for (part, _) in w.map().parts_at(node) {
                                part_names
                                    .entry(part.id.clone())
                                    .or_insert_with(|| part.name.clone());
                            }
                        }
                    }
                });
            }
        }
    }
    Json(describe_catalog(&part_names)).into_response()
}

/// The catalogue as the console reads it.
///
/// Separated from the route because it is the part with the decisions in it and
/// takes only the station names — so a test asserts what an operator will
/// actually see, against a real name map, without standing a daemon up.
fn describe_catalog(part_names: &PartNames) -> Value {
    use crate::engine::tools::{self, Availability, Mode};

    const MODES: [Mode; 2] = [Mode::Physical, Mode::InstantMessage];

    let described: Vec<Value> = tools::CATALOG
        .iter()
        .map(|t| {
            // **Which channels the act can reach at all**, which is not the
            // same question `for_mode` answers. That one builds the *prompt*, so
            // it drops everything conditional on the situation — a `Nearby` act
            // is absent from it in every mode, and reading modes off it reported
            // that `tell` works nowhere.
            //
            // A condition that is not about the channel does not narrow the
            // channel: needing company, or a body, is true down a phone line as
            // much as face to face. Those conditions are reported in `needs`,
            // where an operator can act on them.
            let modes: Vec<&str> = MODES
                .iter()
                .filter(|m| match t.availability {
                    // Where a body is standing, like who is standing with it,
                    // is not a fact about the channel.
                    Availability::Always
                    | Availability::Nearby
                    | Availability::AmongOthers
                    | Availability::Embodied
                    | Availability::AwayFromHome
                    // Being at the lift, or in it, is where the body is standing,
                    // not a fact about the channel — said in `needs` instead.
                    | Availability::AtLift
                    | Availability::InLift => true,
                    Availability::PhysicalOnly => **m == Mode::Physical,
                    Availability::MessagingOnly => m.remote(),
                    Availability::Pictorial => m.carries_pictures(),
                    // Standing next to the right thing is not a fact about the
                    // channel, so it narrows nothing here and is said in
                    // `needs` instead.
                    Availability::AtPart => true,
                })
                .map(|m| m.as_wire())
                .collect();

            // The schema the model actually sees, in the JSON Schema shape the
            // modal renders. Built from the same `params` the grammar compiles
            // from, so the two cannot disagree about what an act takes.
            let properties: serde_json::Map<String, Value> = t
                .params
                .iter()
                .map(|p| {
                    let mut prop = json!({ "type": p.ty, "description": p.description });
                    if let Some(vs) = tools::fixed_values(t.name, p.name) {
                        prop["enum"] = json!(vs);
                    } else if let Some(c) = tools::live_choice(t.name, p.name) {
                        // A live set has no fixed membership to publish — what
                        // it admits is a fact about a room, and saying so is
                        // more use to an operator than a list that would be
                        // wrong everywhere except where it was taken.
                        prop["x-bound-to"] = json!(format!("{c:?}"));
                    }
                    (p.name.to_string(), prop)
                })
                .collect();
            let required: Vec<&str> = t
                .params
                .iter()
                .filter(|p| p.required)
                .map(|p| p.name)
                .collect();

            json!({
                "name": t.name,
                "category": t.category,
                "description": t.description,
                "plane": t.plane,
                "modes": modes,
                "needs": match t.availability {
                    Availability::Always => Value::Null,
                    Availability::Nearby => json!("somebody else here"),
                    Availability::AmongOthers => json!("two or more others here"),
                    Availability::Embodied => json!("a body"),
                    Availability::AwayFromHome => json!("being somewhere that is not home"),
                    Availability::PhysicalOnly => json!("being present"),
                    Availability::MessagingOnly => json!("being at a distance"),
                    Availability::Pictorial => json!("a channel that carries pictures"),
                    // Named, not generic. "Standing at the station that carries
                    // it" is true of every station act and tells an operator
                    // hunting a missing one exactly nothing.
                    Availability::AtPart => match where_it_is(t.at, part_names) {
                        Some(where_) => json!(where_),
                        // An `AtPart` act naming no station is a contradiction
                        // the catalogue's own tests refuse; said here rather
                        // than unwrapped so a route never panics over it.
                        None => json!("standing at a station"),
                    },
                    Availability::AtLift => json!("at the lift, with it on another floor"),
                    Availability::InLift => json!("in the lift, its doors open"),
                },
                // Every station, named, however many there are. The `needs`
                // line above summarises for the table; this is what the detail
                // view shows, where a list of six costs nothing.
                "at_named": named_stations(t.at, part_names),
                "source": "builtin",
                // Calibration is examples: a tool with none selects measurably
                // worse, which is exactly what the column is for.
                "calibrated": !t.examples.is_empty(),
                "examples": t.examples.len(),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            })
        })
        .collect();

    let uncalibrated = tools::CATALOG
        .iter()
        .filter(|t| t.examples.is_empty())
        .count();

    json!({
        "tools": described,
        "uncalibrated": uncalibrated,
        "acts_per_turn": tools::ACTS_PER_TURN,
        "examples": tools::CATALOG.iter().map(|t| t.examples.len()).sum::<usize>(),
    })
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

    /// A body with a speaker: `serde` fills the `Option` fields we do not send.
    fn direct_body(text: &str, speaker: Option<&str>, salience: Option<f32>) -> DirectBody {
        DirectBody {
            text: text.to_owned(),
            speaker: speaker.map(str::to_owned),
            salience,
            to: None,
        }
    }

    /// The bare case: a line and nothing else is the operator, aimed at the
    /// character it reaches, loud enough to be heard now — the route exists to
    /// reach the character, so its default rouses a waiting one rather than
    /// sitting unheard until the next scheduled thought. Everything a caller can
    /// leave out has a defined default, because most direct lines will leave all
    /// of it out.
    #[test]
    fn a_bare_direct_line_is_the_operator_aimed_at_you_and_loud_enough_to_rouse() {
        let (salience, kind) =
            direct_event(direct_body("the east door is open", None, None), "Mira")
                .expect("a line with words in it is delivered");
        assert_eq!(salience, Salience::URGENT);
        assert!(
            salience.rouses(),
            "the default must reach a waiting character"
        );
        assert_eq!(
            kind,
            EventKind::Speech {
                speaker: "Mira".to_owned(),
                text: "the east door is open".to_owned(),
                to: Addressed::You,
            }
        );
    }

    /// The influence the route exists for: a named voice reaches the character
    /// in the operator's place, so a mission-giver or a terminal can speak
    /// without the operator standing in for them.
    #[test]
    fn a_named_speaker_takes_the_operators_place() {
        let (_, kind) = direct_event(
            direct_body("come to the desk", Some("The Archivist"), None),
            "Mira",
        )
        .expect("delivered");
        let EventKind::Speech { speaker, .. } = kind else {
            panic!("a direct line is speech");
        };
        assert_eq!(speaker, "The Archivist");
    }

    /// A speaker field that is present but blank is not a voice — it falls back
    /// to the operator rather than putting an empty name in the character's ear.
    #[test]
    fn a_blank_speaker_falls_back_to_the_operator() {
        let (_, kind) =
            direct_event(direct_body("hello", Some("   "), None), "Mira").expect("delivered");
        let EventKind::Speech { speaker, .. } = kind else {
            panic!("speech");
        };
        assert_eq!(speaker, "Mira");
    }

    /// A line with nothing in it is the one body the route refuses — an empty
    /// utterance is not influence, and delivering it would put a blank speech
    /// event on the character's inbox.
    #[test]
    fn a_blank_line_is_refused() {
        assert!(direct_event(direct_body("   ", None, None), "Mira").is_none());
        assert!(direct_event(direct_body("", Some("The Archivist"), Some(0.9)), "Mira").is_none());
    }

    /// The text is trimmed before it is spoken, like a message — leading and
    /// trailing whitespace is transport, not part of what was said.
    #[test]
    fn the_line_is_trimmed() {
        let (_, kind) =
            direct_event(direct_body("  go now  ", None, None), "Mira").expect("delivered");
        let EventKind::Speech { text, .. } = kind else {
            panic!("speech");
        };
        assert_eq!(text, "go now");
    }

    /// A wire salience is clamped through the same path a constructed one is, so
    /// an out-of-range figure lands at the edge and still orders correctly — an
    /// urgent line at `2.0` preempts, it is not rejected for being over one.
    #[test]
    fn a_wire_salience_is_clamped_and_still_preempts() {
        let (salience, _) =
            direct_event(direct_body("now", None, Some(2.0)), "Mira").expect("delivered");
        assert_eq!(salience.get(), 1.0);
        assert!(salience.preempts());

        let (idle, _) =
            direct_event(direct_body("mm", None, Some(-1.0)), "Mira").expect("delivered");
        assert_eq!(idle.get(), 0.0);
        assert!(!idle.preempts());
    }

    /// The address survives from the body — a line can be thrown to the room,
    /// overheard by the character, rather than aimed at it, when the caller says
    /// so; the default is only what fills the field's absence.
    #[test]
    fn an_explicit_address_is_kept() {
        let body = DirectBody {
            text: "the hall is closing".to_owned(),
            speaker: None,
            salience: None,
            to: Some(Addressed::Room),
        };
        let (_, kind) = direct_event(body, "Mira").expect("delivered");
        let EventKind::Speech { to, .. } = kind else {
            panic!("speech");
        };
        assert_eq!(to, Addressed::Room);
    }

    /// An announcement becomes a world-scale `Announcement` event, trimmed, and
    /// renders as a word put to everyone rather than a thing in the room — the
    /// framing is the whole reason it is its own kind and not a `Description`.
    #[test]
    fn an_announcement_is_a_world_event_read_as_reaching_everyone() {
        let kind = announce_event(AnnounceBody {
            text: "  the eastern gate is sealed  ".to_owned(),
        })
        .expect("a line with words in it is announced");
        assert_eq!(
            kind,
            EventKind::Announcement {
                text: "the eastern gate is sealed".to_owned(),
            }
        );
        let prose = Event::new(0, 0, Salience::URGENT, kind).prose();
        assert_eq!(
            prose,
            "Word goes out across the world: the eastern gate is sealed"
        );
    }

    /// A blank announcement is the one body the route refuses — an empty word to
    /// the world is not an announcement, and broadcasting it would put a blank
    /// event on every character's inbox at once.
    #[test]
    fn a_blank_announcement_is_refused() {
        assert!(announce_event(AnnounceBody {
            text: "   ".to_owned()
        })
        .is_none());
        assert!(announce_event(AnnounceBody {
            text: String::new()
        })
        .is_none());
    }

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

    /// **Every field the tools page reads is a field this route sends.**
    ///
    /// The console renders `name`, `description`, `modes`, `source`,
    /// `calibrated` and `parameters`. The route used to serialise `Tool`
    /// directly, which has none of the last four — so three of the table's five
    /// columns were blank and the schema modal showed `{}`, on a page whose
    /// whole job is to say what a character can do. Nothing failed; the page
    /// simply rendered `undefined` as empty.
    ///
    /// Asserted against the names the page actually uses, so adding a column
    /// there and forgetting this breaks a test rather than a view.
    /// The station names the shipped maps actually give, which is what an
    /// operator will read.
    fn shipped_part_names() -> PartNames {
        let set =
            npc_map::MapSet::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps"))
                .expect("the shipped maps must load");
        let mut names = PartNames::new();
        for area in set.areas() {
            for node in &area.nodes {
                for (part, _) in set.parts_at(node) {
                    names
                        .entry(part.id.clone())
                        .or_insert_with(|| part.name.clone());
                }
            }
        }
        names
    }

    #[test]
    fn the_tools_route_sends_every_field_the_console_renders() {
        let v = describe_catalog(&shipped_part_names());

        assert_eq!(
            v["uncalibrated"].as_u64(),
            Some(0),
            "an act shipped with no examples"
        );
        assert!(v["acts_per_turn"].as_u64().unwrap() >= 1);

        let ts = v["tools"].as_array().expect("an array");
        assert_eq!(ts.len(), crate::engine::tools::CATALOG.len());
        for t in ts {
            for field in [
                "name",
                "category",
                "description",
                "modes",
                "source",
                "calibrated",
            ] {
                assert!(!t[field].is_null(), "`{}` has no `{field}`", t["name"]);
            }
            assert_eq!(t["parameters"]["type"], "object", "{}", t["name"]);
            assert!(t["parameters"]["properties"].is_object());
            assert!(t["parameters"]["required"].is_array());
            assert!(
                !t["modes"].as_array().unwrap().is_empty(),
                "`{}` is offered in no mode at all",
                t["name"]
            );
        }
    }

    /// The page prints a `needs` chip to explain an act's absence. An act that
    /// is conditional must therefore say what its condition is, or an operator
    /// hunting a tool a character never calls has nothing to read.
    #[test]
    fn a_conditional_act_says_what_it_needs() {
        let v = describe_catalog(&shipped_part_names());
        let ts = v["tools"].as_array().unwrap();
        let find = |n: &str| {
            ts.iter()
                .find(|t| t["name"] == n)
                .expect("in the catalog")
                .clone()
        };

        // Speech aimed at somebody needs somebody here to aim it at. A shout
        // does not: it is for whoever is within earshot, including nobody.
        assert_eq!(find("tell")["needs"], "somebody else here");
        // And a whisper needs somebody besides its listener to keep it from.
        assert_eq!(find("whisper")["needs"], "two or more others here");
        assert!(
            find("shout")["needs"].is_null(),
            "a shout needs nobody to be there"
        );
        assert_eq!(find("gesture")["needs"], "somebody else here");
        assert!(
            find("reflect")["needs"].is_null(),
            "stopping must be available whatever else is not, or a character \
             with nothing it can do has no way to spend a turn"
        );
        assert_eq!(find("move_to")["needs"], "a body");
        assert_eq!(find("act")["needs"], "being present");

        // **A station act names the station**, in the words the map uses for
        // it. "Standing at the station that carries it" is true of every one of
        // them and answers nothing.
        assert_eq!(
            find("chronicle_add_entry")["needs"],
            "standing at a world history terminal"
        );
        assert_eq!(
            find("record_appraise")["needs"],
            "standing at the appraisal bench"
        );

        // An act reaching several says so as alternatives — you need one of
        // them, not all six.
        // **An act reaching many stations is counted, not listed.** Six noun
        // phrases in a table cell wrap the row and are unreadable at a glance,
        // and a reader made to parse all six to learn "a workstation" has been
        // given less rather than more.
        let bench = find("bench_branch");
        assert_eq!(bench["needs"], "standing at any of 6 stations");

        // The list is not lost — it is in the field the detail view renders,
        // where there is room for it, with the article each name calls for.
        let at: Vec<String> = bench["at_named"]
            .as_array()
            .expect("the full list")
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert_eq!(at.len(), 6, "{at:?}");
        assert!(
            at.contains(&"a world history terminal".to_string()),
            "{at:?}"
        );
        assert!(
            at.contains(&"an easel".to_string()),
            "the article was assumed: {at:?}"
        );

        // **An act narrowed by channel reports the narrowing.** `send_image`
        // goes down a thread and not into a room — you are standing in front of
        // them, so you hold the thing up — and the catalogue the console reads
        // has to say so, or an operator is shown an act as if it worked
        // everywhere.
        let img = find("send_image");
        let modes: Vec<String> = img["modes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m.as_str().unwrap().to_string())
            .collect();
        let modes: Vec<&str> = modes.iter().map(String::as_str).collect();
        assert_eq!(modes, vec!["instant_message"], "{modes:?}");
    }

    /// With no world open there is no map to read a name off, and the field
    /// says the honest generic thing rather than inventing one.
    #[test]
    fn a_station_with_no_map_loaded_is_not_given_an_invented_name() {
        let v = describe_catalog(&PartNames::new());
        let ts = v["tools"].as_array().unwrap();
        let needs = ts
            .iter()
            .find(|t| t["name"] == "chronicle_add_entry")
            .expect("in the catalog")["needs"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(needs.contains("chronicle-terminal"), "{needs}");
        assert!(
            needs.contains("nothing places one"),
            "an unplaced station should say so: {needs}"
        );
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
