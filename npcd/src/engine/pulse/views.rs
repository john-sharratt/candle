//! The read-only pulse views: the feed, the census, the world, a character's
//! window and its scan.
//!
//! The feed is filtered before its limit, so the requested count is the count the
//! caller may see, and it resolves names in one lookup. Ids are sent as base-36
//! strings because a `u64` past 2^53 does not survive a JavaScript client. The
//! world view reads the map once, so every room in the answer is from the same
//! instant. The window shows what a character is still holding, and the feed shows
//! what it did; reading them together tells forgetting apart from never perceiving.

use super::*;

#[derive(Debug, Deserialize)]
pub struct FeedQuery {
    #[serde(default)]
    pub(super) limit: Option<usize>,
    /// Only this character's ticks. Absent means every character the caller may
    /// see, interlaced — which is the view's default and the reason it exists.
    /// A base-36 string off the wire ([`npcs::npc_id_wire`]), not a number —
    /// the console pastes an id straight out of a tick or census row, and
    /// those are base-36.
    #[serde(default)]
    pub(super) npc_id: Option<String>,
    /// Admin only: the whole cast, including characters the caller does not own.
    #[serde(default)]
    pub(super) all: bool,
}

/// Decode the `npc_id` query filter, or say which raw value could not be read.
///
/// A decode failure is reported back to the caller (`Err`, the offending
/// string) rather than silently treated as "no filter" — a mistyped id must
/// not return the whole unfiltered feed with nothing to say why.
pub(super) fn parse_npc_id_filter(raw: Option<&str>) -> Result<Option<u64>, &str> {
    match raw {
        None => Ok(None),
        Some(raw) => npcs::npc_id_of_wire(raw).map(Some).ok_or(raw),
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
    let want = match parse_npc_id_filter(q.npc_id.as_deref()) {
        Ok(want) => want,
        Err(raw) => {
            return err(
                StatusCode::BAD_REQUEST,
                "bad_npc_id",
                &format!("`{raw}` is not a valid base-36 npc_id"),
            )
        }
    };

    // Filtered *before* the limit, so asking for fifty rows gives fifty rows
    // this caller can see rather than fifty from the whole cast with most of
    // them removed. The scheduler's ring is the hard bound either way.
    let mut ticks: Vec<_> = rt.scheduler.recent_where(MAX_FEED.max(limit), |t| {
        scope.admits(t.npc_id) && want.is_none_or(|want| t.npc_id == want)
    });
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
/// Ids are stringified in base-36 because a `u64` past 2^53 does not survive a
/// JavaScript client — the same reason the character record itself serialises
/// its id this way, and getting it wrong here would give every large-id
/// character the name of whichever one it collided with.
pub(super) async fn name_map(
    s: &Arc<Authored>,
    ids: impl Iterator<Item = u64>,
) -> serde_json::Map<String, Value> {
    let wanted: HashSet<u64> = ids.collect();
    let cast = s.npcs.view();
    wanted
        .into_iter()
        .filter_map(|id| {
            let n = cast.payload(id)?;
            Some((npcs::npc_id_wire(id), Value::String(n.name.clone())))
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
    // One census, filtered once, for both the rows and their names.
    let seen: Vec<_> = rt
        .scheduler
        .census()
        .into_iter()
        .filter(|c| scope.admits(c.npc_id))
        .collect();
    let ids: Vec<u64> = seen.iter().map(|c| c.npc_id).collect();
    let mut characters: Vec<serde_json::Value> = Vec::new();
    for c in seen {
        let world_ms = s.world_ms(c.npc_id);
        let mut v = serde_json::to_value(&c).unwrap_or_else(|_| json!({}));
        if let Some(o) = v.as_object_mut() {
            o.insert("world_ms".into(), json!(world_ms));
            o.insert(
                "phase".into(),
                json!(crate::engine::sleep::phase_of(world_ms)),
            );
            // **What its standing task would say now**, whether this member is
            // the one that tells it, and whether it is a drone at rest — held
            // with no mission, its conversation off the card, until the table
            // gives it one.
            o.insert("runs_here".into(), json!(rt.standing.runs(c.npc_id)));
            o.insert("at_rest".into(), json!(rt.scheduler.is_resting(c.npc_id)));
            o.insert("nudge".into(), json!(rt.nudge_for(c.npc_id)));
        }
        characters.push(v);
    }
    let names = name_map(&s, ids.into_iter()).await;

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
                bound.insert(body, json!(npcs::npc_id_wire(npc_id)));
            }
        }
    }

    Json(json!({
        "worlds": worlds,
        // `body -> npc_id`, as a base-36 string: an id past 2^53 does not
        // survive a JavaScript client as a number, and this one is read by
        // the console.
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
    Path(nid): Path<String>,
    headers: HeaderMap,
) -> Response {
    let nid = match owned(&s, &headers, &nid).await {
        Ok(id) => id,
        Err(r) => return *r,
    };
    let Some(rt) = s.runtime.as_ref() else {
        return no_scheduler();
    };
    let Some(w) = rt.scheduler.window_of(nid, |w| {
        let turns: Vec<_> = w
            .turns()
            .map(|t| {
                json!({
                    "id": t.id,
                    "origin": t.origin,
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

/// `GET /v1/npc/:nid/scan` — what the character's `scan` with no place named
/// tells it right now.
///
/// `prose` is the text the character would read, word for word; `survey` is the
/// data it was written from — each station with its address, state, verbs and
/// every parameter's meaning, plus the conversations and recent past — so an
/// operator can check the scan without making the character act.
pub async fn scan(
    State(s): State<Arc<Authored>>,
    Path(nid): Path<String>,
    headers: HeaderMap,
) -> Response {
    let nid = match owned(&s, &headers, &nid).await {
        Ok(id) => id,
        Err(r) => return *r,
    };
    let Some(rt) = s.runtime.as_ref() else {
        return no_scheduler();
    };
    let Some((hosted, body)) = rt.body_of(nid) else {
        return err(
            StatusCode::CONFLICT,
            "not_placed",
            "the character has no body in a running world, so there is nothing to scan",
        );
    };
    let survey = Survey::of(&hosted, &body);
    Json(json!({ "prose": survey.prose(), "survey": survey })).into_response()
}
