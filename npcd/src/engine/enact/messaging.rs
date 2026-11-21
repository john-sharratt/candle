//! The acts that reach somebody who is not in the room: a phone message, a reply, a group,
//! an invitation and a goodbye.
//!
//! A message to one person is a purpose that the character's own conversation with that
//! person carries in its own words. A channel or a group receives the line that was voiced,
//! and so does a reply on a thread. A goodbye on a direct thread ends the conversation and
//! leaves the thread for the next line to reopen, while a goodbye on a group is a departure.
//! A group may not take a name the character already uses, since two identical names make
//! every act that names one ambiguous. Answering by handset also clears a question left
//! standing in the room.

use super::*;

// ── the phone ───────────────────────────────────────────────────────────────
//
// Every one of these reaches somebody who is not here, from wherever the
// character is standing, at a time that is not necessarily now. That is the
// whole difference from `tell`, and it is why they are separate acts.

/// The kind of conversation a `message`'s `via` names.
pub(super) fn via(args: &Map<String, Value>) -> Result<ThreadKind, Outcome> {
    match text(args, "via").as_deref() {
        Some(PRIVATE) => Ok(ThreadKind::Direct),
        Some(CHANNEL_WAY) => Ok(ThreadKind::Group),
        _ => Err(Outcome::Refused(format!(
            "You meant to send something, but not whether `{PRIVATE}`, to one person, or on a \
             `{CHANNEL_WAY}`."
        ))),
    }
}

/// The words a phone act puts on a thread: the line the lease holder voiced
/// for it, carried with the act, and the critical step it asks an authorizer
/// for (`docs/npc_conversations.md` §7.1, §7.2). An act made without one — a
/// program calling a device route with the words it means — says its
/// `intent` as given.
pub(super) fn words(said: Option<&Voiced>, intent: &str) -> Said {
    Said {
        text: said.map_or(intent, |v| v.text.as_str()).trim().to_string(),
        intent: said.map(|_| intent.to_string()),
        requests: said.and_then(|v| v.requests.clone()),
        ..Said::default()
    }
}

/// Send on the thread `to`: `kind` when `message` says how it goes, or a
/// thread waiting on this character for `reply`.
///
/// **A private message is a purpose, not a line.** The character's
/// conversation with that person says it, in its own words, on their thread
/// — so a `message` to one person, from the main loop or as a hand-off from
/// another conversation, hands that conversation what was meant
/// (`docs/npc_conversations.md` §10). A channel or group takes the voiced
/// line itself, as does a `reply` on a thread.
pub(super) fn message(
    hosted: &Hosted,
    body: &str,
    args: &Map<String, Value>,
    kind: Option<ThreadKind>,
    said: Option<&Voiced>,
) -> Outcome {
    let (Some(to), Some(intent)) = (text(args, "to"), text(args, "intent")) else {
        return Outcome::Refused(
            "You meant to send something, but not to which conversation, or not what.".into(),
        );
    };
    if kind == Some(ThreadKind::Direct) {
        return purpose(hosted, body, &to, &intent);
    }
    let sent = hosted.with_sim(|s| {
        let found = match kind {
            Some(kind) => s.threads.by_kind_for(body, kind, &to),
            None => s.threads.by_name_for(body, &to),
        };
        let Some(id) = found.map(|t| t.id.clone()) else {
            return Err(match kind {
                Some(ThreadKind::Group) => {
                    format!("You have no conversation called {to} among your channels.")
                }
                _ => format!("You have no conversation with {to}."),
            });
        };
        let (_, reached) = s.threads.send_on(&id, body, words(said, &intent))?;
        let names: Vec<String> = reached
            .iter()
            .map(|r| s.threads.name_of(r).unwrap_or(r).to_string())
            .collect();
        // **Answering by handset is answering**: a question a body left
        // standing is cleared when it is answered here, as `tell` clears it.
        // Members who are no body in this world had no question standing.
        for asker in &reached {
            s.ledger.answered(body, asker);
        }
        Ok(names)
    });
    match sent {
        Ok(reached) => Outcome::Did(format!(
            "You send it to {to}. {} will see it when they next look.",
            npc_map::text::list(&reached)
        )),
        Err(why) => Outcome::Refused(why),
    }
}

/// What this character means to tell `to`, handed to its conversation with
/// them, which says it in its own words. The authority takes it from
/// [`Sim::purposes`] and delivers it to whichever member holds the
/// character (`authority::perform`).
pub(super) fn purpose(hosted: &Hosted, body: &str, to: &str, intent: &str) -> Outcome {
    hosted.with_sim(|s| {
        let party = s
            .threads
            .by_kind_for(body, ThreadKind::Direct, to)
            .and_then(|t| t.other_ids(body).into_iter().next())
            .or_else(|| s.roster_id(to).map(str::to_string));
        let Some(party) = party else {
            return Outcome::Refused(format!("You have no way to reach {to}."));
        };
        let name = s.roster_name(&party).unwrap_or(to).to_string();
        let me = Who::new(body, s.roster_name(body).unwrap_or(body));
        s.threads.reach(&me, &Who::new(&party, &name));
        s.purposes.push(Purpose {
            from: body.to_string(),
            to: party,
            intent: intent.trim().to_string(),
        });
        Outcome::Did(format!(
            "You mean to tell {name} {}; you will say it in your own words.",
            intent.trim().trim_end_matches('.')
        ))
    })
}

pub(super) fn invite(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let (Some(to), Some(who)) = (text(args, "to"), text(args, "who")) else {
        return Outcome::Refused(
            "You meant to bring somebody in, but not to which conversation, or not who.".into(),
        );
    };
    hosted.with_sim(|s| {
        let Some(id) = s.roster_id(&who).map(str::to_string) else {
            return Outcome::Refused(format!("You have no way to reach {who}."));
        };
        let name = s.roster_name(&id).unwrap_or(&who).to_string();
        match s.threads.invite(body, &to, &Who::new(&id, &name)) {
            Ok(_) => Outcome::Did(format!(
                "{name} is on it. What is said there now reaches all of you."
            )),
            Err(why) => Outcome::Refused(why),
        }
    })
}

pub(super) fn open_group(
    hosted: &Hosted,
    body: &str,
    args: &Map<String, Value>,
    said: Option<&Voiced>,
) -> Outcome {
    let (Some(called), Some(with)) = (text(args, "called"), text(args, "with")) else {
        return Outcome::Refused("A group needs a name and somebody to be on it.".into());
    };
    hosted.with_sim(|s| {
        let mut members = vec![Who::new(body, s.roster_name(body).unwrap_or(body))];
        for name in with.split(',').map(str::trim).filter(|n| !n.is_empty()) {
            let Some(id) = s.roster_id(name).map(str::to_string) else {
                return Outcome::Refused(format!("You have no way to reach {name}."));
            };
            if members.iter().all(|m| m.id != id) {
                let shown = s.roster_name(&id).unwrap_or(name).to_string();
                members.push(Who::new(&id, &shown));
            }
        }
        if members.len() < 2 {
            return Outcome::Refused("A group of one is a note to yourself.".into());
        }
        // **A name this character already uses is refused.** Two threads called
        // the same thing are ambiguous to every act that names one — `message`,
        // `invite` and `sign_off` all resolve by the name the character uses,
        // and would silently pick whichever came first. Worse, the pair became
        // two identical arms in the turn's grammar, which the stencil refuses
        // outright: the whole tree failed to compile, the turn free-decoded,
        // and the character was told its output was not a call. Every turn,
        // permanently, because nothing it could do afterwards removed the
        // duplicate.
        if s.threads.by_name_for(body, &called).is_some() {
            return Outcome::Refused(format!(
                "You are already on something called {called}. Give this one a name of its own, \
                 or say what you meant to say on the one you have."
            ));
        }
        let id = s.threads.open_group(&called, &members);
        if let Some(intent) = text(args, "intent") {
            let _ = s.threads.send_on(&id, body, words(said, &intent));
        }
        let others: Vec<String> = members[1..].iter().map(|m| m.name.clone()).collect();
        Outcome::Did(format!(
            "{called} is open, with {} on it.",
            npc_map::text::list(&others)
        ))
    })
}

/// Saying goodbye on a thread.
///
/// **On a direct thread it ends the conversation and leaves the thread
/// standing**: the parting is said, the conversation is ENDED, and the next
/// line either side says opens it again on the same thread
/// (`docs/npc_conversations.md` §4.3, §11.1). On a group it is leaving: the
/// parting is said, then the character goes.
pub(super) fn sign_off(
    hosted: &Hosted,
    body: &str,
    args: &Map<String, Value>,
    said: Option<&Voiced>,
) -> Outcome {
    let (Some(to), Some(intent)) = (text(args, "to"), text(args, "intent")) else {
        return Outcome::Refused(
            "You meant to go, but not from which conversation, or not why.".into(),
        );
    };
    hosted.with_sim(|s| {
        let Some(t) = s.threads.by_name_for(body, &to) else {
            return Outcome::Refused(format!("You have no conversation with {to}."));
        };
        let (id, kind) = (t.id.clone(), t.kind());
        if let Err(why) = s.threads.send_on(&id, body, words(said, &intent)) {
            return Outcome::Refused(why);
        }
        match kind {
            ThreadKind::Direct => Outcome::Did(format!("You say goodbye to {to}.")),
            ThreadKind::Group => match s.threads.leave(body, &to) {
                Ok(()) => Outcome::Did(format!("You leave {to}, having said why.")),
                Err(why) => Outcome::Refused(why),
            },
        }
    })
}

/// Getting hold of somebody you have no thread with yet: the pair's thread is
/// opened, and what was meant goes to the conversation with them, as a
/// private `message` does.
pub(super) fn reach_out(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let (Some(to), Some(intent)) = (text(args, "to"), text(args, "intent")) else {
        return Outcome::Refused(
            "You meant to reach somebody, but not who, or not what about.".into(),
        );
    };
    purpose(hosted, body, &to, &intent)
}
