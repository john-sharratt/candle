//! One character's turn: wait until it is due, perceive, decode, act, and answer.
//!
//! A character's life is one task that waits until due, takes its turn, and, when
//! the turn reflects, waits for the reflection's answer before taking another. A
//! turn waits for the engine and the persona rather than taking anything it cannot
//! think with, and a suspended character takes no turn even when the scheduler has
//! not caught up with its record. Every act a turn makes is recorded and answered
//! within it, so the character reads what its act did on its next turn.

use super::*;

/// How long a character must go without news before the standing task is
/// restated to it.
///
/// **A stretch of quiet, not an empty instant.** Long enough that it cannot land
/// between two turns of a conversation — an exchange runs at roughly the world's
/// moment, and a companion that is thinking, walking a stop, or simply slower
/// than the heartbeat can leave a gap of tens of seconds without the
/// conversation being over. Short enough that a character genuinely left alone
/// does not stand in a room for minutes with nothing to go on.
pub(super) const IDLE_AFTER_MS: u64 = 90_000;

/// How long a character due a turn waits for the engine and its persona when
/// either is not there yet — at boot, before the model has loaded.
pub(super) const UNREADY_RETRY: Duration = Duration::from_secs(2);

/// How long after a failed decode the turn is tried again, on the same events.
pub(super) const DECODE_RETRY_MS: u64 = 5_000;

/// One character's whole life as a task: wait until due, perceive, think, act,
/// and — when a turn reflects — await the reflection's answer before taking
/// another turn. Returns when the character is retired or the daemon stops.
/// Strip an effector `url` to the router path the fast path invokes (effector
/// design §7).
///
/// `http://local` is the only host the device can reach in the fiction, so a
/// leading `http://local` is removed and both `http://local/lift/x` and the bare
/// `/lift/x` address the same route; `http://local` and `http://local/` both map
/// to the near-you index at `/`. A path with no leading slash is given one, so a
/// model that writes `lift/x` still reaches the route.
pub(super) fn device_path(url: &str) -> String {
    let rest = url.trim();
    let rest = rest.strip_prefix("http://local").unwrap_or(rest);
    if rest.is_empty() {
        "/".to_string()
    } else if rest.starts_with('/') {
        rest.to_string()
    } else {
        format!("/{rest}")
    }
}

/// A character's turn, from its start to wherever it ends: a hand-back waits
/// for the end of one in progress.
pub(super) struct Turning<'a>(pub(super) &'a Standing, pub(super) u64);

impl Drop for Turning<'_> {
    fn drop(&mut self) {
        self.0.end_turn(self.1);
    }
}

pub(super) async fn character_loop(rt: Arc<Runtime>, id: u64) {
    // Whether this character's kept entries have been read back from the
    // substrate. Done once, on the first turn the engine is there to ask, and
    // before that turn's conversation opens so it opens scoped to them.
    let mut journal_restored = false;
    loop {
        if rt.stopping() {
            return;
        }
        if !rt.scheduler.wait_due(id).await {
            // Retired: no inbox left to be due.
            return;
        }
        if rt.stopping() {
            return;
        }
        if !journal_restored {
            let minds = rt.minds.read().unwrap().clone();
            if let Some(minds) = minds {
                let kept = minds.restore_journal(id);
                tracing::info!(
                    "npc {id}: journal restored, {} entr(ies)",
                    kept.journal().entries().count()
                );
                rt.scheduler.set_journal(id, kept);
                journal_restored = true;
            }
        }
        let now_ms = rt.scheduler.now_ms();
        {
            // Per character, not once per pass: two characters in
            // different worlds are at different instants, and one of
            // those worlds may be paused.
            let world_ms = rt.world_ms(id);
            // The day boundary is checked before the tick, so a
            // character that has crossed midnight wakes into its new
            // conversation rather than acting once more in yesterday's.
            //
            // The turns the roll clears are kept by the journal state —
            // `roll_day` took them before it cleared the window — and the
            // guardian asks about them like any other stretch, so a day's
            // last stretch is not lost with it.
            if let Some(rolled) = rt.scheduler.roll_day(id, world_ms) {
                let (from, to) = (rolled.from, rolled.to);
                tracing::info!("npc {id}: day {from} → {to}, conversation rolled over");
                rt.scheduler
                    .deliver(id, world_ms, Salience::NORMAL, EventKind::Wake { day: to });
            }
            // The standing task is restated from the world's moment, not
            // here — see [`Runtime::standing_task`].
            let minds = rt.minds.read().unwrap().clone();
            // Before the persona, which reads the journal a closed chapter
            // has just been written into.
            let chapter = rt.turn_chapter(id, world_ms);
            let persona = rt.persona_of_async(id).await;
            // **No engine yet, or no persona: wait for them, and take nothing.**
            // Ticking anyway drained the inbox into a turn that could not run —
            // at every boot, every character told something before the model
            // loaded lost it — and left the character unscheduled with it.
            // Nothing is drained here; the character comes back once there is
            // something to think with.
            let (Some(minds), Some(p)) = (minds, persona) else {
                tokio::time::sleep(UNREADY_RETRY).await;
                continue;
            };
            // **A turn begins here, and a character being handed back to its
            // world takes none**: the member it moves to runs its next one
            // (§9.3). Marked only from here, so a character waiting for an
            // engine is between turns and can be handed back meanwhile.
            // **A suspended character takes no turn**, by its own record: the
            // scheduler's copy of the cast's suspensions parks it between
            // turns, and this is what holds it when that copy has not caught
            // up. Held, it parks in `wait_due` until it is resumed.
            if rt.held_by_record(id) {
                continue;
            }
            if !rt.standing.begin_turn(id) {
                tokio::time::sleep(UNREADY_RETRY).await;
                continue;
            }
            let _turning = Turning(&rt.standing, id);
            let day = crate::clock::day_of(world_ms);
            // What the grammar is built from this turn: the acts that
            // are reachable from where this character stands, and the
            // names it may address. Read here, once, before the decode —
            // the world moves under a decode that takes seconds, and a
            // grammar built halfway through it would be masked to a room
            // that no longer matches the situation the character read.
            //
            // The company is fixed first, so the world answers an addressee
            // against the very set the grammar offers: somebody who walks out
            // during the decode is called after, not refused as "not here".
            rt.begin_decision(id);
            let within = rt.within(id);
            // The standpoint the grammar above was built from, captured at the
            // same pre-decode instant. A device act the character then chooses is
            // reach-checked against this position rather than wherever the
            // metronome has since walked a moving body — so `invoke` on an
            // address the grammar just offered cannot 404 because a leg passed
            // during the decode. See [`crate::effector::auth::PinnedStandpoint`].
            let pinned_at = rt
                .body_of(id)
                .and_then(|(hosted, body)| hosted.read(|w| w.actor(&body).map(|a| a.at.clone())));

            // The tick's three phases, with the decode awaited between
            // the first two and the last — [`Scheduler::begin_tick`] /
            // [`Scheduler::end_tick`] are `tick` split at exactly that
            // seam, so this loop and the scheduler tests run one path. A drone
            // at rest is held first, so `begin_tick` refuses it
            // (`Runtime::settle_rest`).
            rt.settle_rest(id);
            let Some(mut start) = rt.scheduler.begin_tick(id) else {
                rt.end_decision(id);
                continue;
            };
            // What the world opens the turn with, from wherever it runs.
            let opening = rt.begin_turn(id).await;
            // A journey on its mission that ends where it now stands is done, and
            // it is told so, with what comes next — in this tick, so it reads its
            // progress the turn it arrives rather than an unticked step that sends
            // it back to the table to begin again.
            if let Some(text) = opening.arrival {
                start.events.push(Event::new(
                    rt.scheduler.next_seq(),
                    world_ms,
                    Salience::URGENT,
                    EventKind::Nudge { text },
                ));
            }
            // The mission compass — the next step and where whoever it names is
            // now — whenever that has changed since it was last told.
            if let Some(text) = opening.compass {
                start.events.push(Event::new(
                    rt.scheduler.next_seq(),
                    world_ms,
                    Salience::URGENT,
                    EventKind::Nudge { text },
                ));
            }
            // What a failed decode hands back to be read again: everything so
            // far, told once — not the summons below, which is made afresh
            // every tick and would otherwise be read twice.
            let told_once = start.events.len();
            // **A character standing at the table with its work done is told to
            // report it, every tick it stands there.** Appended to the tick in
            // hand rather than `deliver`ed (whose `due_at = 0` would
            // tight-loop): the window is a turn or two before it drifts off.
            // It stops the instant the report is filed (`at_table_report`
            // returns `None`).
            if let Some(text) = opening.summons {
                start.events.push(Event::new(
                    rt.scheduler.next_seq(),
                    world_ms,
                    Salience::URGENT,
                    EventKind::Nudge { text },
                ));
            }
            // **What a person asked and has not been told the outcome of is
            // restated on every turn it is not itself the news of**: a turn
            // begun on a Drone taking up work reads nothing of the request
            // otherwise, and the request is dropped for whatever is in front
            // of the mind. Appended to the tick in hand, as the summons is.
            let fresh = start
                .events
                .iter()
                .any(|e| matches!(e.kind, EventKind::Concluded { .. }));
            let asked = rt.scheduler.asked(id, world_ms);
            // **And the turn that carries it out is waited on**: it decodes
            // at `Urgent`, ahead of the cast, for as long as the request
            // stands, and at the conversation's own band once it is told.
            minds
                .set_band(id, asked.as_ref().map(|_| DecodePriority::Urgent))
                .await;
            if let Some((party, request)) = asked.filter(|_| !fresh) {
                let table = rt.orders_out(id);
                start.events.push(Event::new(
                    rt.scheduler.next_seq(),
                    world_ms,
                    Salience::URGENT,
                    EventKind::Nudge {
                        text: still_asked(&party, &request, table.as_deref()),
                    },
                ));
            }
            let mut awaiting_reflection: Option<(oneshot::Receiver<String>, Owed)> = None;
            // The curated prose the character read, captured out of the think
            // step so the tick record reports it in place of the raw events —
            // what the pulse shows is then what went to the model.
            let mut narrated: Option<String> = None;
            // A decode that failed hands its events back, so the turn that does
            // run reads them — see [`Scheduler::retry`].
            let mut retry: Option<Vec<Event>> = None;
            let done = {
                match minds
                    .think(
                        id,
                        &p.as_persona(),
                        p.mode,
                        day,
                        chapter,
                        &start.events,
                        &within,
                    )
                    .await
                {
                    Ok(t) => {
                        // The curated prose this tick read, for the record.
                        if !t.perception.is_empty() {
                            narrated = Some(t.perception.clone());
                        }
                        // What woke it, beside the acts it then makes: a turn
                        // that reads nothing was brought back by its own last
                        // act, and one that reads something by the world.
                        tracing::info!(
                            target: "npcd::engine::turn",
                            npc = id,
                            read = %t.perception,
                            "a mind's turn began on this"
                        );
                        // Reported, never swallowed: a character failing
                        // to act and one choosing not to look identical
                        // from outside and need completely different
                        // fixes.
                        //
                        // And told to the character, not only to the
                        // log. A rejection it cannot see is a character
                        // acting into silence — it has no reason to do
                        // anything differently, so it makes the same
                        // malformed call every turn for as long as it
                        // runs.
                        // Two lists, because the feed and the character
                        // read different sentences — see `Recorded`.
                        // A rejection is the one case where they agree:
                        // the world's words are all there is.
                        let mut done: Vec<String> = Vec::new();
                        let mut answers: Vec<String> = Vec::new();
                        for r in &t.parsed.rejected {
                            tracing::warn!("npc {id}: act rejected — {r:?}");
                            done.push(r.line());
                            answers.push(r.line());
                        }
                        // A call the grammar could not take is a refusal
                        // like any other, and is read in a turn of its
                        // own — not whenever the room next wakes it.
                        if !t.parsed.rejected.is_empty() {
                            rt.scheduler.think_again(id);
                        }
                        if t.parsed.is_empty() && t.parsed.rejected.is_empty() {
                            // Chose to do nothing, and said so cleanly.
                            // Distinct from a decode that failed, which
                            // logged above.
                            tracing::debug!("npc {id}: no act this tick");
                        }
                        if !t.parsed.narration.is_empty() {
                            tracing::debug!(
                                "npc {id}: narration (not an act) — {}",
                                t.parsed.narration
                            );
                        }
                        // Acts that belong to a body go to the world
                        // they stand in, and the world's verdict — not
                        // the character's intent — decides how each one
                        // is recorded. A refusal is an ordinary
                        // outcome, perceived like any other, so the
                        // character learns it went wrong rather than
                        // believing it worked. `record_act` holds the
                        // rule.
                        // **A reflect the world took is answered by a
                        // reflection** — at most one per turn. Its slot
                        // keeps `body::NO_REFLECTION` until the
                        // reflection answers.
                        let mut reflecting: Option<(Owed, [String; 3])> = None;
                        // What each act came back with and whether the world
                        // took it, in call order: what the loop guard judges
                        // an act that names nothing by.
                        let mut came_back: Vec<(String, bool)> = Vec::new();
                        for a in &t.parsed.acts {
                            // Made in the world the character stands in,
                            // wherever that world runs — see
                            // `Runtime::perform`. A device call, a sitting to
                            // write and a body's act each produce a
                            // `Recorded`, so the feed, the answer, and
                            // everything downstream are one shape.
                            // A post on a group is said in the character's
                            // own words, decoded here before the act goes to
                            // the world; one whose words could not be decoded
                            // is refused, never posted as its intent, and
                            // one saying again what it just said there is
                            // refused before it is posted.
                            let performed = match rt.voice_post(id, a).await {
                                Ok(Post::None) => {
                                    rt.perform_said(
                                        id,
                                        a,
                                        pinned_at.as_ref(),
                                        None,
                                        &ActSource::Mind,
                                    )
                                    .await
                                }
                                Ok(Post::Words(said)) => {
                                    rt.perform_said(
                                        id,
                                        a,
                                        pinned_at.as_ref(),
                                        Some(&said),
                                        &ActSource::Mind,
                                    )
                                    .await
                                }
                                Ok(Post::Repeat) => rt.repeated(a),
                                Err(e) => rt.unvoiced(a, &format!("{e:#}")),
                            };
                            let r = performed.recorded;
                            // An act that moved the mission is answered by
                            // what comes next, and that is read at once.
                            let moved = performed.moved;
                            let took = r.landed;
                            // Intent-carrying acts read back a narration of
                            // what they did, not the bare "You tell X." the
                            // world hands them — and the same line shows in
                            // the feed. Only when the act landed; a narration
                            // failure keeps the plain reply. See
                            // `tools::narrates` / `Minds::narrate_act`.
                            let (feed, answer) = if r.landed
                                && crate::engine::tools::narrates(a.tool)
                            {
                                match minds.narrate_act(id, &p.as_persona(), a, r.departing).await {
                                    Ok(Some(n)) => (format!("{} {LANDED} {n}", a.summary()), n),
                                    _ => (r.feed, r.answer),
                                }
                            } else {
                                (r.feed, r.answer)
                            };
                            done.push(feed);
                            if a.tool == "reflect"
                                && r.landed
                                && reflecting.is_none()
                                && rt.can_reflect()
                            {
                                let arg = |k: &str| {
                                    a.args
                                        .get(k)
                                        .and_then(|v| v.as_str())
                                        .unwrap_or_default()
                                        .to_string()
                                };
                                // Recorded as the act alone — `reflect`
                                // and its thought — and completed with
                                // what came back once it has.
                                let row = a.summary();
                                if let Some(last) = done.last_mut() {
                                    *last = row.clone();
                                }
                                reflecting = Some((
                                    Owed {
                                        answers: Vec::new(),
                                        slot: answers.len(),
                                        row,
                                    },
                                    [arg("situation"), arg("inner_thoughts"), arg("feeling")],
                                ));
                            }
                            came_back.push((answer.clone(), took));
                            answers.push(answer);
                            // A pause that landed stops the character
                            // here. Armed after the act rather than
                            // inside it because going quiet is
                            // scheduling and being seen to stop is the
                            // world's — two halves of one act, and only
                            // this half knows the clock.
                            rt.arm_pause(id, a, now_ms);
                            // And an act the world answered — a read, a
                            // refusal, work on a document, a step of the
                            // mission moved — brings it straight back, so
                            // it has a turn in which to use what it was
                            // told.
                            rt.arm_followup(id, a, took, moved);
                        }
                        // **Fold this decode into the loop guard.** Every
                        // well-formed act, with what it meant, in call
                        // order — an act taken again meaning the same is
                        // struck from the next grammar, and only it. The
                        // strike is *explained* by a nudge that rides the
                        // next turn as a `<tool_response>`. See
                        // [`crate::engine::loopguard`].
                        //
                        // A device call goes in under the verb at its
                        // address: under `invoke`, a loop at one device
                        // struck every device in reach. An act that names
                        // nothing goes in as what it came back with: reading
                        // the cluster's jobs twice is a loop only when the
                        // second reading says what the first did.
                        let taken: Vec<Took> = t
                            .parsed
                            .acts
                            .iter()
                            .zip(&came_back)
                            .map(|(a, (back, landed))| {
                                let took = match body::is_device(a.tool) {
                                    true => {
                                        let path = device_path(
                                            a.args
                                                .get("url")
                                                .and_then(|v| v.as_str())
                                                .unwrap_or(""),
                                        );
                                        // An address it was not offered keeps its
                                        // address in what it meant, so two calls to
                                        // two different unknown devices are never
                                        // one call — and `invoke` itself is struck
                                        // only for the same call made again.
                                        match within
                                            .invokable
                                            .iter()
                                            .find(|i| device_path(&i.url) == path)
                                        {
                                            Some(i) => Took::call(i.act, salient(a)),
                                            None => {
                                                Took::call(a.tool, format!("{path} {}", salient(a)))
                                            }
                                        }
                                    }
                                    false => match salient(a) {
                                        meant if meant.trim().is_empty() => {
                                            Took::call(a.tool, back.as_str())
                                        }
                                        meant => Took::new(a.tool, meant),
                                    },
                                };
                                match *landed {
                                    true => took,
                                    false => took.refused(),
                                }
                            })
                            .collect();
                        // A fight is the one place the guard lets `act`
                        // repeat; `within.hostiles` is what tells it a fight
                        // is on. Read from the turn's own `within`, so a
                        // character shaking a companion in a quiet room is
                        // guarded while one trading blows is not.
                        if rt
                            .loop_guards
                            .record(id, &taken, !within.hostiles.is_empty())
                        {
                            // Pointed at the work: the next step of the
                            // mission, when there is one.
                            let nudge = match rt.mission_compass(id) {
                                Some(next) => format!("{NUDGE}\n{next}"),
                                None => NUDGE.to_string(),
                            };
                            minds.deliver_outcomes(id, vec![nudge]).await;
                            // And a turn to read it in, with the looping
                            // act struck: a loop of speech earns no
                            // follow-up of its own, and the nudge would
                            // otherwise wait on the room.
                            rt.scheduler.think_again(id);
                        }
                        // **And the character is told what came of it.**
                        //
                        // One answer per call it made, riding at the
                        // head of its next turn as `<tool_response>` —
                        // the half of the protocol that was missing.
                        // Without it a character acts and reads the
                        // weather back, which is how every one of them
                        // came to do nothing but reflect.
                        //
                        // The world's verdict, not the feed line: what
                        // a person watching wants to read and what the
                        // character needs to know are different
                        // sentences. See `Recorded`.
                        //
                        // A turn with a reflection in it hands its
                        // answers to the reflection instead, delivered
                        // when its first question is answered — the
                        // await below this tick.
                        match reflecting {
                            Some((owed, [situation, inner, feeling])) => {
                                let rx = rt.spawn_reflection(id, situation, inner, feeling);
                                awaiting_reflection = Some((rx, Owed { answers, ..owed }));
                            }
                            None => minds.deliver_outcomes(id, answers).await,
                        }
                        done
                    }
                    Err(e) => {
                        tracing::warn!(
                            "npc {id}: decode failed — {e:#}; trying again in {}s",
                            DECODE_RETRY_MS / 1000
                        );
                        retry = Some(start.events.iter().take(told_once).cloned().collect());
                        Vec::new()
                    }
                }
            };
            rt.end_decision(id);
            // Report the curated prose as what was perceived, when a narration
            // was produced — the raw events behind it are not what the model
            // read. A quiet tick with no narration keeps its raw perceived.
            if let Some(prose) = narrated {
                start.perceived = vec![prose];
            }
            rt.scheduler.end_tick(id, now_ms, world_ms, start, done);
            if let Some(events) = retry {
                rt.scheduler.retry(id, now_ms, DECODE_RETRY_MS, events);
            }

            // The turn reflected: its answer has to be the next thing
            // this character reads about the act, so the loop does not
            // come back round — and the room cannot give it a turn —
            // until the answer (or the reflection's failure) is in.
            // Everything arriving meanwhile queues in the inbox and is
            // read on the first turn after.
            if let Some((rx, owed)) = awaiting_reflection {
                let line = rx.await.ok();
                rt.settle_reflection(id, owed, line).await;
            }
        }
    }
}
