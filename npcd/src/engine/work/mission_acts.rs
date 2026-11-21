//! Performs the mission acts: taking up a mission, recording progress on it, and
//! reporting it done or stuck, each judged against what the engine can see.
//!
//! A mission is done when the record says so: its document is written and
//! committed, and a step the engine can still see as doable holds the report. A
//! stuck report is refused twice while the next step is within reach, and a
//! refusal counts only toward a way the character actually tried. A rejected
//! draft goes back to be fixed, or is settled as its workflow says once it has
//! failed as often as the workflow allows.

use super::*;

/// Perform a mission act.
///
/// Take one up at the desk, record progress on it wherever the work happened, or
/// report how it went. The rich model is [`crate::engine::mission`]; whose it is
/// is [`crate::sim::missions`]; this moves a mission between those states and
/// hands the character back a line it reads.
pub(super) fn mission(hosted: &Hosted, body: &str, act: &Act) -> Outcome {
    let a = &act.args;
    match act.tool {
        // The table's orders are given by a character that takes none; the
        // table does not offer them to anybody else, and refuses them if asked.
        "order_post" | "order_results" | "mission_write"
            if !hosted.sim(|s| s.missions.is_unordered(body)) =>
        {
            Outcome::Refused(
                "The table's orders are not yours to give: you take your work from it.".into(),
            )
        }
        "order_post" => order_table::post(hosted, body, a),
        "order_results" => order_table::results(hosted, body),
        "mission_write" => write_template::write(hosted, body, a),
        "report_done" => {
            let Some(account) = text(a, "found") else {
                return Outcome::Refused(
                    "You meant to report it done, but did not say what you found.".into(),
                );
            };
            // **A critical step is signed by an authorizer, never by reporting
            // the mission done** (`docs/npc_conversations.md` §7.1, §15).
            let unsigned = hosted.sim(|s| {
                s.missions
                    .active(body)
                    .and_then(|m| m.todo.iter().find(|t| t.critical && !t.done))
                    .map(|t| t.text.clone())
            });
            if let Some(step) = unsigned {
                return Outcome::Refused(format!(
                    "This mission waits on an authorizer's yes to \"{step}\": it is not done \
                     until they sign it. Message them and ask."
                ));
            }
            // **An account says what was found.** A character filed its own
            // name as the account of a reading, six times over; a few words are
            // not an answer anybody can use. Turned back with what the engine
            // saw on the mission, so the next try has the facts in front of it.
            if account.split_whitespace().count() < MIN_ACCOUNT_WORDS {
                let seen = hosted
                    .sim(|s| s.missions.active(body).map(|m| m.observed.clone()))
                    .unwrap_or_default();
                let hint = match seen.is_empty() {
                    true => String::new(),
                    false => format!(" What you saw on it: {}.", seen.join("; ")),
                };
                return Outcome::Refused(format!(
                    "\"{account}\" is not an account of what you found. Say it in a sentence — \
                     what you found, made or concluded.{hint}"
                ));
            }
            // **What is wrong with the document comes before the way to the
            // desk.** The gate puts a document that does not stand back to be
            // written; a reviewer who mended one fault, committed, and was
            // then told only "you have not done this yet: change it and commit
            // it — the desks are on the chronicle" went between the desk and
            // the table with nothing it knew to change, until it gave up stuck.
            if rewriting_next(hosted, body) {
                if let Some(refusal) = report_block(hosted, body) {
                    return Outcome::Refused(refusal);
                }
            }
            // **A step the engine can see, not yet done and still doable, holds
            // the report.** A character sent to two rooms messaged the channel
            // and reported the mission done from the table; the journeys were
            // never made. The way to the step is the refusal.
            if let Some(way) = still_doable(hosted, body) {
                let step = hosted
                    .sim(|s| {
                        s.missions
                            .active(body)?
                            .step_to_do()
                            .map(|t| t.text.clone())
                    })
                    .unwrap_or_default();
                return Outcome::Refused(format!(
                    "You have not done this yet: {step}. {way} Then come back and report it. If \
                     it cannot be done, `invoke` the table's `report_stuck` and say why."
                ));
            }
            // **Work on the record is done when the record says so.** A mission
            // that writes a document is reported done once that document is
            // committed — by this character, while it carried the mission (see
            // [`Mission::written_up`]). Its own word that the writing is
            // finished is the label `docs/asynchronous_mind_hierarchy.md` §7.3
            // says must never stand uncorroborated, and a step ticked on its
            // word is that label: a story was reported done with its write
            // step struck "thwarted" and no document anywhere.
            if let Some(path) = hosted.sim(|s| {
                let m = s.missions.active(body)?;
                (!m.written_up()).then(|| m.work.as_ref().map(|w| w.writes.clone()))?
            }) {
                return Outcome::Refused(format!(
                    "You have not written it yet: {path} is not committed. Go to a desk where \
                     documents are written, `compose` it there — you write it whole with your \
                     brief and what you read in front of you — then `bench_commit` it with one \
                     line saying what it is, and come back and report it. If it cannot be \
                     written, `invoke` this table's `report_stuck` and say why."
                ));
            }
            // **A reading nobody made cannot be reported.** A mission to read a
            // machine was filed as a pass by a character whose own account said
            // it never read it. Reading is a `scan` in the machine's room, which
            // the engine sees; until it has happened, done is not the report.
            if let Some(step) = hosted.sim(|s| {
                let machines: Vec<String> = s.devices.iter().map(|d| d.name.clone()).collect();
                s.missions
                    .active(body)?
                    .unread_step(&machines)
                    .map(str::to_string)
            }) {
                return Outcome::Refused(format!(
                    "You have not done this yet: {step}. Go to the room it is in and `scan` there \
                     — what you scan shows each machine with the state it is in. Then come back \
                     and report it. If you cannot get to it or read it, `invoke` the table's \
                     `report_stuck` and say why."
                ));
            }
            // **The quality gate, while it can still be mended.** An
            // operation's document is held to the engine's checks before its
            // draft or its review may be reported done — see
            // `mission_gen::gates`.
            if let Some(refusal) = report_block(hosted, body) {
                return Outcome::Refused(refusal);
            }
            file_done(hosted, body, &account)
        }
        "report_stuck" => {
            let Some(why) = text(a, "why") else {
                return Outcome::Refused(
                    "You meant to report it stuck, but did not say why.".into(),
                );
            };
            // **Stuck is for a step that cannot be done, and the world can see
            // when one can.** Characters gave missions up as stuck a minute in —
            // the person "nowhere to be found" one room away, the machine "not
            // responding" in a room they never reached. When the next step's
            // room, machine or person is within reach, the desk says where and
            // how instead. Twice, then it takes the report — counting only the
            // turn-aways after which the world refused something the character
            // tried: one told the way that tried it twice and was refused is
            // reporting a real block; one that never tried it is not.
            // A document sent back to be written is stuck only once its
            // faults have been put in front of the Maker: what to change, not
            // the way to a desk.
            let faults = match rewriting_next(hosted, body) {
                true => report_block(hosted, body),
                false => None,
            };
            if let Some(faults) = faults {
                let draft = draft_digest(hosted, body);
                let refused = hosted.with_sim(|s| s.missions.refuse_stuck_over_faults(body, draft));
                if refused <= STUCK_REFUSALS {
                    return Outcome::Refused(format!("It is not stuck yet — {faults}"));
                }
            }
            if let Some(way) = still_doable(hosted, body) {
                let refused = hosted.with_sim(|s| s.missions.refuse_stuck_on_the_way(body));
                if refused <= STUCK_REFUSALS {
                    return Outcome::Refused(format!(
                        "It is not stuck yet — {way} Do that, then come back to the table and \
                         `invoke` its `report_done` with what you found."
                    ));
                }
            }
            // **Nothing is stuck when nothing is left.** A Maker whose story was
            // written, committed and every step signed off reported it stuck —
            // "the mission is done, but I cannot report it" — and the stuck
            // report failed the operation and threw the finished draft away.
            // With every step but the report done and its document committed,
            // the world says the work stands; the report that matches is done.
            if let Some(done) = hosted.sim(|s| finished(s, body)) {
                // **Work with no operation behind it is taken as done.** Its
                // every step is done in the engine's sight and it has no
                // document for a gate to hold, so a refusal saying "report it
                // done" only sent three characters between the two desks for
                // hours — each sure it had reported already, each told it was
                // not stuck. An operation's step has a gate and, at review, a
                // verdict to give, and is told so instead.
                //
                // **Told twice, the third is taken as the done report it
                // matches.** A Maker on a canon check, every step done and its
                // finding "no contradictions" said on the channel, sent
                // `report_stuck` again and again for an hour, refused each time
                // with the two verdicts it could give. The work stands in the
                // engine's sight, so it is filed done — through the gate a done
                // report passes, in the carrier's words.
                let standalone = hosted.sim(|s| {
                    s.missions
                        .active(body)
                        .is_some_and(|m| m.operation().is_none())
                });
                if !standalone {
                    if hosted.with_sim(|s| s.missions.refuse_stuck(body)) <= STUCK_REFUSALS {
                        return Outcome::Refused(done);
                    }
                    if let Some(refusal) = report_block(hosted, body) {
                        return Outcome::Refused(refusal);
                    }
                }
                let account = done_account(hosted, body, &why);
                return file_done(hosted, body, &account);
            }
            hosted.with_sim(
                |s| match s.missions.report(body, Verdict::Fail, &why, None) {
                    Some(m) => {
                        tracing::info!(npc = body, prompt = %m.mission_text(), %why, "mission reported STUCK at the command table");
                        Outcome::Did(format!(
                            "Reported as not done, with your reasons: {}{}",
                            m.mission_text(),
                            back_to_now(&m)
                        ))
                    }
                    None => Outcome::Refused("You are not carrying a mission to report on.".into()),
                },
            )
        }
        "report_rejected" => {
            let Some(why) = text(a, "why") else {
                return Outcome::Refused(
                    "You meant to reject the draft, but did not say why it cannot stand.".into(),
                );
            };
            if why.split_whitespace().count() < MIN_ACCOUNT_WORDS {
                return Outcome::Refused(format!(
                    "\"{why}\" does not say why the draft cannot stand. Say what is wrong with it \
                     that cannot be mended in place."
                ));
            }
            // A review must have read what it rejects.
            if let Some(way) = still_doable(hosted, body) {
                let step = hosted
                    .sim(|s| {
                        s.missions
                            .active(body)?
                            .step_to_do()
                            .map(|t| t.text.clone())
                    })
                    .unwrap_or_default();
                return Outcome::Refused(format!(
                    "You have not done this yet: {step}. {way} Read it, and mend what the table \
                     found, before you judge it — reject only what your mending could not save."
                ));
            }
            let shelf = hosted.shelf_of(body);
            let unsupported = hosted.sim(|s| {
                let m = s.missions.active(body)?;
                m.operation()?;
                rejection::unsupported(&*shelf, m.work.as_ref()?, &why)
            });
            if let Some(refusal) = unsupported {
                tracing::info!(npc = body, %why, "operation rejection refused: no evidence");
                return Outcome::Refused(refusal);
            }
            let rejected = hosted.with_sim(|s| {
                let m = s.missions.reject(body, &why)?;
                // **A rejected draft goes where its workflow sends a
                // rejection** — back to be fixed, with your reasons as what the
                // fix answers to. Only one sent back as often as its workflow
                // allows has failed, and its store is settled as the
                // workflow's `on-failed` says: discarded, or kept unpublished
                // for an operator to reopen. Settled here at once; the
                // generator's loop settles every other failure the same way,
                // and discarding again is harmless.
                let op = m
                    .operation()
                    .and_then(|(id, _)| s.missions.operations().get(id))
                    .cloned();
                let failed = op
                    .as_ref()
                    .filter(|o| o.failed())
                    .map(|o| (o.id, s.missions.operations().on_failed(o)));
                Some((m, failed))
            });
            let Some((m, failed)) = rejected else {
                return Outcome::Refused(
                    "You are not reviewing anybody's draft; there is nothing to reject.".into(),
                );
            };
            let gone = match failed {
                None => "it goes back to be fixed, with your reasons",
                Some((id, OnFailed::Discard)) => {
                    if let Some(stores) = hosted.op_stores() {
                        stores.discard(&Owner::operation(hosted.id(), id));
                    }
                    "the work is dropped, and the record stays as it was"
                }
                Some((_, OnFailed::Keep)) => {
                    "the work is kept, unpublished, for an operator to reopen"
                }
            };
            tracing::info!(npc = body, prompt = %m.mission_text(), %why, "operation REJECTED on review");
            Outcome::Did(format!(
                "Rejected, with your reasons; {gone}. {}{}",
                m.mission_text(),
                back_to_now(&m)
            ))
        }
        other => Outcome::Refused(format!("`{other}` is not a mission act.")),
    }
}

/// File `body`'s mission done with `account` — both the report's notes (how it
/// went) and its answer (what was found): for a mission done, the two are the
/// same line.
pub(super) fn file_done(hosted: &Hosted, body: &str, account: &str) -> Outcome {
    hosted.with_sim(|s| {
        match s
            .missions
            .report(body, Verdict::Pass, account, Some(account.to_string()))
        {
            Some(m) => {
                tracing::info!(npc = body, prompt = %m.mission_text(), %account, "mission reported DONE at the command table");
                // An operation this report finished is the record grown by a
                // document: said once, where it happens.
                if let Some(op) = m
                    .operation()
                    .and_then(|(id, _)| s.missions.operations().get(id))
                    .filter(|o| o.succeeded())
                {
                    tracing::info!(operation = %op.name, document = %op.document, "operation: stands in the record");
                }
                Outcome::Did(format!(
                    "Reported done, and your answer filed: {}{}",
                    m.mission_text(),
                    back_to_now(&m)
                ))
            }
            None => Outcome::Refused("You are not carrying a mission to report on.".into()),
        }
    })
}

/// The account a finished mission is filed done with when its carrier
/// reported it stuck: the carrier's own words, then what the engine saw on it.
///
/// **What the engine saw is not the answer to every question.** Filed in the
/// carrier's place, it answered "read the Concord's era and report its years"
/// with "in the early range: nobody else" — the room's company on arrival,
/// which only a who-is-there errand asks for. The carrier's words say what it
/// found; the sightings are kept beside them as evidence.
pub(super) fn done_account(hosted: &Hosted, body: &str, said: &str) -> String {
    let seen = hosted
        .sim(|s| s.missions.active(body).map(|m| m.observed.clone()))
        .unwrap_or_default();
    let said = said.trim();
    match (said.is_empty(), seen.is_empty()) {
        (_, true) => said.to_string(),
        (true, false) => seen.join("; "),
        (false, false) => format!("{said} (seen: {})", seen.join("; ")),
    }
}

/// Give `body` the table's next mission for it — lodged for it, waiting in
/// the pool, or a routine from the bank — and answer with its brief as the
/// character reads it. `None` when there is nothing to give: a world that
/// draws nothing, with nothing posted.
///
/// **A mission is given, never fetched** (`docs/zen_everywhere.md` §3): the
/// table's authority calls this for a body free to carry
/// (`Runtime::assign_work`), and no act of a character's reaches it. A body
/// already carrying one is given nothing — one at a time.
pub fn take_up(hosted: &Hosted, body: &str) -> Option<String> {
    if hosted.sim(|s| s.missions.is_on_mission(body) || s.missions.is_unordered(body)) {
        return None;
    }
    // The character's own name, so a routine that would send it to visit
    // "the makers here" is not built around visiting itself.
    let me = hosted
        .read(|w| w.actor(body).map(|actor| actor.name.clone()))
        .unwrap_or_default();
    let mission = hosted.with_both(|w, s| {
        let room_name = |place: &str| {
            let (area, node) = place.split_once('/')?;
            w.node(&Where::new(area, node)).map(|n| n.name.clone())
        };
        let reach: Vec<String> = w
            .actor(body)
            .map(|actor| {
                std::iter::once(actor.at.clone())
                    .chain(route::reachable_from(w.map(), &actor.at))
                    .map(|at| format!("{}/{}", at.area, at.node))
                    .collect()
            })
            .unwrap_or_default();
        let material = s.mission_material(&me, &room_name, &reach);
        s.missions.collect(body, &material.facts()).cloned()
    })?;
    tracing::info!(npc = body, prompt = %mission.mission_text(), "mission given by the command table");
    // What an operation's document says in its store as its step is taken up
    // — what a step that must change it is held against (`changed`).
    let found = mission
        .operation()
        .zip(mission.work.as_ref())
        .and_then(|((id, _), w)| Some((id, hosted.op_shelf(id)?.read(&w.writes)?)));
    if let Some((id, text)) = found {
        hosted.with_sim(|s| s.missions.found(id, text));
    }
    Some(mission.standing_text())
}

/// Whether the next step of the body's mission still to do is the writing of
/// its own document — the step the gate puts back when the document does not
/// stand.
pub(super) fn rewriting_next(hosted: &Hosted, body: &str) -> bool {
    hosted
        .sim(|s| {
            let m = s.missions.active(body)?;
            let doc = &m.work.as_ref()?.writes;
            let next = m.next_step()?;
            let text = next.text.trim();
            Some(
                (text.starts_with("write ") || text.starts_with("change "))
                    && text.contains(doc.as_str()),
            )
        })
        .unwrap_or(false)
}

/// A digest of the body's mission document as its shelf holds it now — `0` for
/// one not there — so a rewrite can be told from none.
pub(super) fn draft_digest(hosted: &Hosted, body: &str) -> u64 {
    let path = hosted.sim(|s| Some(s.missions.active(body)?.work.as_ref()?.writes.clone()));
    let text = path.and_then(|p| hosted.shelf_of(body).read(&p));
    let mut h = DefaultHasher::new();
    text.hash(&mut h);
    h.finish()
}

/// The fewest words `report_done` takes as an account of what was found.
pub(super) const MIN_ACCOUNT_WORDS: usize = 4;

/// How many times `report_stuck` is turned away while the next step is within
/// reach before the report is taken anyway.
pub(super) const STUCK_REFUSALS: u32 = 2;

/// How the next step of the body's mission can be done from here, when the
/// engine can see that it can: its room is within reach, its machine stands in
/// a room within reach, or its person is in one. `None` when there is no such
/// step, or what it names is nowhere the body can get to.
pub(super) fn still_doable(hosted: &Hosted, body: &str) -> Option<String> {
    hosted.with_both(|w, s| {
        let step = s
            .missions
            .active(body)?
            .step_to_do()
            .filter(|t| !t.reports)?
            .text
            .clone();
        if let Some(year) = time_step(&step) {
            let here = w.actor(body)?.at.clone();
            let has_machine = |at: &Where| {
                w.node(at).is_some_and(|n| {
                    w.map()
                        .parts_at(n)
                        .any(|(part, _)| part.id == "time-machine")
                })
            };
            if has_machine(&here) {
                return Some(format!(
                    "A time machine is here: `time_travel` naming the year {year}."
                ));
            }
            let name = route::reachable_from(w.map(), &here)
                .iter()
                .filter(|at| has_machine(at))
                .find_map(|at| body::destination_name(w, &here, at))?;
            return Some(format!(
                "You are not working in {year} yet. The time machines are on the time level: \
                 `move_to` {name}, then `time_travel` naming the year {year}."
            ));
        }
        let here = w.actor(body)?.at.clone();
        let reach: Vec<Where> = std::iter::once(here.clone())
            .chain(route::reachable_from(w.map(), &here))
            .collect();
        let room_of = |at: &Where| w.node(at).map(|n| n.name.clone());
        let level_of = |at: &Where| w.map().get(&at.area).map(|a| a.name.clone());
        let place = |at: &Where| format!("{}/{}", at.area, at.node);
        // The name `move_to` takes a room by from here — one act wherever the
        // room is, the world riding the lift for a journey that changes floor.
        let go = |at: &Where| match *at == here {
            true => room_of(at),
            false => body::destination_name(w, &here, at),
        };
        // **A document is read and written at a desk, and the desks can be
        // reached.** Makers gave drafts up as stuck "between the lift and the
        // command room", the chronicle level "not accessible" — one from the
        // landing with the car on its way — and each report failed its
        // operation and threw the work away.
        if DocStep::of(&step).is_some() {
            let desks: Vec<&Where> = reach
                .iter()
                .filter(|at| s.part_offers(&place(at), "file_read"))
                .collect();
            let desk = desks
                .iter()
                .find(|at| at.area == here.area)
                .or_else(|| desks.first())?;
            let room = room_of(desk)?;
            // At the desk already, the way is the act, not a walk to where it
            // stands.
            if **desk == here {
                let write = match s.missions.active(body).is_some_and(|m| m.writes_given()) {
                    true => "`file_write` it exactly as your brief gives it",
                    false => "write it with `compose`",
                };
                return Some(format!(
                    "You are at a desk, in {room}: {write}, then `bench_commit` it."
                ));
            }
            let name = go(desk)?;
            return Some(match desk.area == here.area {
                true => format!("A desk is within reach in {room}: `move_to` it, and work there."),
                false => format!(
                    "The desks are on {on}: `move_to` {name}, and work there.",
                    on = level_of(desk)?
                ),
            });
        }
        let aim = Aim::of(&step)?;
        // A body that moves by attention has no `move_to` and no lift: the way
        // to anywhere is `focus` on it.
        let attends = s.attends(body);
        if let Some(at) = reach.iter().find(|at| {
            room_of(at).is_some_and(|name| aim.is_room(&name, &level_of(at).unwrap_or_default()))
        }) {
            let room = room_of(at)?;
            if attends {
                return Some(format!("{room} is within reach: `focus` on it."));
            }
            let name = go(at)?;
            return Some(match at.area == here.area {
                true => format!("{room} is within reach: `move_to` it."),
                false => format!("{room} is on {}: `move_to` {name}.", level_of(at)?),
            });
        }
        for d in s.devices.iter().filter(|d| aim.is_machine(&d.name)) {
            if let Some(at) = reach.iter().find(|at| place(at) == d.at) {
                let room = room_of(at)?;
                let go = match attends {
                    true => "`focus` on",
                    false => "`move_to`",
                };
                return Some(format!(
                    "{} stands in {room}, within reach: {go} {room}, and you read its state as \
                     you arrive.",
                    d.name
                ));
            }
        }
        for a in w
            .actors()
            .filter(|a| a.id != body && aim.is_person(&a.name))
        {
            if reach.contains(&a.at) {
                let room = room_of(&a.at)?;
                let go = match attends {
                    true => "`focus` on",
                    false => "`move_to`",
                };
                // Room speech is for people; another character is messaged,
                // here or anywhere (`docs/npc_conversations.md` §12).
                return Some(match (a.at == here, people::is_person(&a.id)) {
                    (true, true) => format!("{} is here with you: `ask` or `tell` them.", a.name),
                    (false, true) => format!(
                        "{} is in {room}, within reach: {go} {room} and `ask` or `tell` them — \
                         or `message` them, which reaches them wherever they are.",
                        a.name
                    ),
                    (_, false) => format!("{}: `message` them.", a.name),
                });
            }
        }
        None
    })
}

/// `time_travel`: work in the year `args` names, within the world's history,
/// for as long as the mission the body carries — see
/// [`crate::engine::mission::Mission::travelled`].
pub(super) fn time_travel(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    // A year arrives as text or as a number — the device's body takes either,
    // and a Maker that sent `{"year": 2937}` was told to name it as a number.
    let year = match args.get("year") {
        Some(Value::Number(n)) => n.as_u64().and_then(|y| u32::try_from(y).ok()),
        _ => text(args, "year").and_then(|y| {
            y.chars()
                .filter(|c| c.is_ascii_digit())
                .collect::<String>()
                .parse::<u32>()
                .ok()
        }),
    };
    // **A refusal says what to do, in the mission's own terms.** "Name the year
    // as a number: 2937" sent a Maker whose mission asked for 2950 to ask its
    // colleagues what had happened in 2950 — it read a malformed call as a gap
    // in what it knew, and nobody it asked could see the call either.
    let Some(year) = year else {
        let asked = hosted
            .sim(|s| {
                s.missions
                    .active(body)?
                    .todo
                    .iter()
                    .filter(|t| !t.done)
                    .find_map(|t| time_step(&t.text))
            })
            .map(|y| {
                format!(
                    " Your mission asks you to work in {y}: `time_travel` with the year \"{y}\"."
                )
            })
            .unwrap_or_default();
        return Outcome::Refused(format!(
            "Your call named no year — the machine needs the year written in it.{asked}"
        ));
    };
    let eras = hosted
        .sim(|s| s.bench.files())
        .map(|f| chronology::eras(&f))
        .unwrap_or_default();
    if let (Some(first), Some(now)) = (eras.first(), eras.last()) {
        if year < first.opens {
            return Outcome::Refused(format!(
                "{year} is before the world's history opens, in {} ({}).",
                first.opens, first.title
            ));
        }
        if year > now.opens {
            return Outcome::Refused(format!(
                "{year} is after the present ({}); there is nothing there yet.",
                now.opens
            ));
        }
    }
    let era = chronology::standing_in(&eras, year)
        .last()
        .map(|e| format!(", in {}", e.title))
        .unwrap_or_default();
    hosted.with_sim(|s| match s.missions.travelled(body, year) {
        None => Outcome::Refused(
            "You carry no work to stand in a year for. Take up a mission at the command table \
             first."
                .into(),
        ),
        Some(ticked) => {
            tracing::info!(npc = body, year, "time machine set");
            let mut said = format!(
                "You work in {year} now{era}. Nothing after it comes back to you until you report \
                 the work you carry."
            );
            if ticked {
                let next = s.missions.active(body).and_then(|m| m.next_step());
                said.push('\n');
                said.push_str(&progress_line("The time is set", next, &[]));
            }
            Outcome::Did(said)
        }
    })
}
