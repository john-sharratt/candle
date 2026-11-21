//! The routes that put a line on a character's inbox, either as the operator's
//! console command or as speech spoken directly into its world.
//!
//! Unlike the feed, these check ownership, because putting words in one
//! character's ear is a write to something somebody owns. A `/`-prefixed console
//! line is parsed, and a typo is refused with the near miss named rather than
//! spoken as dialogue. `direct` is the plain influence primitive: a speaker, a
//! loudness and an address, each with a default that makes a bare line aimed at
//! the character and loud enough to rouse it. A blank line is the one body refused.

use super::*;

#[derive(Debug, Deserialize)]
pub struct InjectBody {
    /// The operator's line, `/`-prefixed or not.
    pub(super) line: String,
}

/// `POST /v1/npc/:nid/pulse` — put an event on a character's inbox.
///
/// The `/` notation's endpoint. Ownership *is* checked here, unlike the feed:
/// reading what the cast is doing is an operator's business, but poking a
/// specific character is a write to something somebody owns.
pub async fn inject(
    State(s): State<Arc<Authored>>,
    Path(nid): Path<String>,
    headers: HeaderMap,
    Json(body): Json<InjectBody>,
) -> Response {
    let (id, owner, nid) = match owned_by(&s, &headers, &nid).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
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

    let world_ms = s.world_ms(nid);
    // `/sleep` carries a placeholder day the parser cannot know; the clock is
    // the only thing that can fill it in.
    let kind = match parsed.kind {
        EventKind::Sleep { .. } => EventKind::Sleep {
            day: crate::clock::day_of(world_ms),
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
pub(super) fn deliver_and_report(
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
    pub(super) text: String,
    /// Who is heard to say it. Absent means the operator, under the same name
    /// the rest of the daemon speaks to a character with — see
    /// [`crate::engine::speaking_as`]. A named voice is how a mission-giver, a
    /// terminal, or another character reaches a character in-world without the
    /// operator standing in for them.
    #[serde(default)]
    pub(super) speaker: Option<String>,
    /// How loudly it lands. Absent is [`Salience::URGENT`], because the whole
    /// point of the route is to reach the character now — the sibling
    /// person-to-character path (`environment.rs`, `phone::Kind::Direct`) uses
    /// the same, and a quieter default would sit in a waiting character's inbox
    /// unheard until its next scheduled thought. A caller that wants it to wait
    /// its turn passes a lower figure; at or above [`Salience::PREEMPT_AT`] it
    /// cuts into a turn the character is mid-way through, and a wire value out of
    /// range is clamped, not rejected.
    #[serde(default)]
    pub(super) salience: Option<f32>,
    /// Who it was aimed at, from this character's side. Absent is
    /// [`Addressed::You`] — a direct line is by definition aimed at the one it
    /// reaches, unlike a line thrown to the room.
    #[serde(default)]
    pub(super) to: Option<Addressed>,
}

/// The event a `/direct` becomes, before it meets a clock or the scheduler.
///
/// Pure, so the defaults, the trim, and the salience clamp are tested without a
/// runtime — the handler adds only the world time and the delivery. `None` is a
/// line with nothing in it, which is the one body this route refuses.
pub(super) fn direct_event(body: DirectBody, operator: &str) -> Option<(Salience, EventKind)> {
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
    Path(nid): Path<String>,
    headers: HeaderMap,
    Json(body): Json<DirectBody>,
) -> Response {
    let (id, owner, nid) = match owned_by(&s, &headers, &nid).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
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

    let world_ms = s.world_ms(nid);
    deliver_and_report(rt, nid, world_ms, salience, kind, json!({}))
}
