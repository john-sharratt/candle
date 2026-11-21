//! Reads what a room or a phone says, and recalls a body home.
//!
//! A read hands back the words a room posts, each attributed to its author and marked as read as
//! it is handed over, so the same board is not offered again. A phone's group or channel is read
//! by the name the character calls it, and the reader sees its own last line there, so it does not
//! say the same thing twice. Recalling a body home while it is already there is refused rather than
//! reported as a success, since a success with no effect is the shape of a loop.

use super::*;

/// Read what something here says.
///
/// # It returns words, and it used to return a switch position
///
/// This was bound to the machines in the room and answered with the mode one was
/// set to — *"You read the accession desk. It stands at `reading`."* A `Device`
/// carries no text, so that was the only thing it *could* answer with, and the
/// answer was the question restated. It also put a raw mode identifier, in
/// backticks, into what a character reads, which is machinery in the prose.
///
/// What a room actually has to read is [`crate::sim::posting`], and that is the
/// only branch left. The set `what` binds to is now what this body has not seen,
/// so an act that arrives here has something to hand back by construction —
/// which is why the empty answer below is a refusal about *timing* rather than a
/// normal outcome.
pub(super) fn read(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let Some(what) = text(args, "what") else {
        return Outcome::Refused("You meant to read something, but not what.".into());
    };
    let place = hosted.standpoint_of(body);
    let me = my_name(hosted, body);
    hosted.with_sim(|s| {
        // Marked read as it is handed over, which is what makes the act
        // terminate: the same board offers this body nothing next turn.
        if let Some(fresh) = s.postings.read(&place, &what, body) {
            if fresh.is_empty() {
                return Outcome::Refused(format!(
                    "You have read everything on {what}. Nothing has been added since."
                ));
            }
            // Attributed, because who wrote a thing is half of what reading it
            // tells you — and a line whose author is this character reads as
            // its own, which is worth knowing before acting on it.
            let said = fresh
                .iter()
                .map(|l| match l.by == me {
                    true => format!("{} (yours)", l.text),
                    false => format!("{} — {}", l.text, l.by),
                })
                .collect::<Vec<_>>()
                .join("; ");
            return Outcome::Did(format!("You read {what}: {said}"));
        }
        // **A group or channel on its phone, by the name it calls it** — or
        // all of them with something unread, asked for as its messages or
        // phone. Read as a board is: marked read as it is handed over, which
        // is what ends the unread reminder (`sim::unread`). A line read here
        // signs off no critical step: only an authorizer's yes to the step a
        // conversation asked them for does (`docs/npc_conversations.md` §7.2).
        if s.has_phone(body) {
            if let Some(said) = read_phone(&mut s.threads, body, &what) {
                // Reading its messages is a step of the reader's round, seen
                // done as it lands.
                if said.happened() {
                    s.missions.read_messages(body);
                }
                return said;
            }
        }
        if what.contains("order") {
            // Marked as this body has now seen them, which is what stops the
            // orders becoming the next thing to be read on a loop.
            if let Some(now) = s.ledger.unheld_unseen_by(body) {
                s.ledger.mark_orders_read(body, &now);
                return Outcome::Did(format!("Standing and unheld: {now}."));
            }
            if !s.ledger.unheld().is_empty() {
                return Outcome::Refused(
                    "You have read the standing orders as they are. Nothing has changed on them."
                        .into(),
                );
            }
        }
        Outcome::Refused(format!("There is nothing here called {what} to read."))
    })
}

/// `what`, read off `me`'s phone: one of its groups or channels by the name
/// it calls it, or every one with something unread when `what` asks for its
/// messages or its phone. `None` when `what` names neither, so the reading
/// goes on to the room.
///
/// **Each thread read shows the reader its own last line there**, so a
/// character answering a channel knows what it already said on it rather
/// than saying it again (`docs/npc_conversations.md` §11.3).
pub(super) fn read_phone(threads: &mut Threads, me: &str, what: &str) -> Option<Outcome> {
    let want = what.trim().to_lowercase();
    let all = want == ALL_MESSAGES
        || matches!(
            want.as_str(),
            "my messages" | "phone" | "my phone" | "handset"
        );
    let names: Vec<String> = match all {
        true => threads
            .unread_threads(me)
            .into_iter()
            .map(|(n, _)| n)
            .collect(),
        false => vec![threads
            .by_kind_for(me, ThreadKind::Group, what)?
            .as_named_to(me)],
    };
    let now = now_ms();
    let mut said = Vec::new();
    for name in &names {
        let Some(t) = threads.by_kind_for(me, ThreadKind::Group, name) else {
            continue;
        };
        let own = t
            .last_said_by(me)
            .map(|m| format!(" (you last said, {}: {})", ago(m.at_ms, now), m.text.trim()));
        let speaker = |id: &str| t.member(id).map_or(id.to_string(), |m| m.name.clone());
        let lines: Vec<String> = t
            .unread(me)
            .iter()
            .map(|m| format!("{}: {}", speaker(&m.from), m.text.trim()))
            .collect();
        let own = own.unwrap_or_default();
        threads.read(me, name).ok()?;
        if lines.is_empty() {
            continue;
        }
        said.push(format!("On {name}: {}{own}", lines.join("; ")));
    }
    Some(match (said.is_empty(), all) {
        (true, true) => Outcome::Refused("Nothing on your phone is unread.".into()),
        (true, false) => Outcome::Refused(format!(
            "You have read everything on {}. Nothing has been said since.",
            names[0]
        )),
        (false, _) => Outcome::Did(format!(
            "{}\nTo answer, `reply` on the conversation it was said on, by that name.",
            said.join("\n")
        )),
    })
}

/// How long ago `then` was, as a character is told it: "just now", "4 min
/// ago", "2 h ago".
pub(super) fn ago(then: u64, now: u64) -> String {
    let s = now.saturating_sub(then) / 1000;
    match s {
        0..=59 => "just now".into(),
        60..=3_599 => format!("{} min ago", s / 60),
        3_600..=86_399 => format!("{} h ago", s / 3_600),
        _ => format!("{} d ago", s / 86_400),
    }
}

pub(super) fn recall(hosted: &Hosted, body: &str) -> Outcome {
    // Where a recall lands is the map's to say, read once at seed — so a body
    // in the vault ports to the command room and one in the waste to the muster
    // hall, without either place being named here.
    let Some(home) = hosted.sim(|s| s.homes().first().cloned()) else {
        return Outcome::Refused("There is nowhere here to be recalled to.".into());
    };
    let Some((area, node)) = home.split_once('/') else {
        return Outcome::Refused("Nothing answers the recall.".into());
    };
    let to = npc_map::world::Where::new(area, node);
    // **Being called home while you are already home is not a thing that can
    // happen.** `World::place` accepts it, changes nothing, and answers "the
    // ground goes out from under you" — a success with no effect, which is the
    // shape every treadmill in this engine has had. Measured live the hour it
    // became reachable: twenty-four of sixty acts across the whole cast, and
    // the characters could see it — "I'm standing here again. The same loop.
    // The same board, the same man."
    if hosted.read(|w| w.actor(body).map(|a| a.at.clone())) == Some(to.clone()) {
        return Outcome::Refused("You are already home.".into());
    }
    match hosted.with(|w| w.place(body, to.clone())) {
        Ok(()) => Outcome::Did("The ground goes out from under you, and you are home.".into()),
        Err(why) => Outcome::Refused(refusal(hosted, &why)),
    }
}
