//! Acts done to somebody in the room or on the phone, with each person named as the world names them.
//!
//! A body's own name and the people in its room are looked up by the name the world wrote for
//! them. A promise or a reminder is made to somebody in the room, so the room hears it, or to the
//! party of the phone thread of that name. An act one body does to another is shown to the room
//! as a body doing something to another. A self-act is shown without the actor's intent, since
//! nobody else can see a feeling, and the actor's own outcome keeps the intent.

use super::*;

/// This body's own name, as the world writes it. Empty for a body that is not
/// in the world, which no live path reaches.
pub(super) fn my_name(hosted: &Hosted, body: &str) -> String {
    hosted.read(|w| w.actor(body).map(|a| a.name.clone()).unwrap_or_default())
}

/// Somebody in this room, by the name the world wrote for them.
pub(super) fn here_named(hosted: &Hosted, body: &str, name: &str) -> Option<String> {
    let want = name.trim().to_lowercase();
    hosted.read(|w| {
        let at = w.actor(body)?.at.clone();
        w.actors()
            .find(|a| a.id != body && a.at == at && a.name.to_lowercase() == want)
            .map(|a| a.id.clone())
    })
}

/// Who `name` is, for a promise or a reminder: somebody standing here, or
/// the party of the pair's thread by that name — a conversation's promise is
/// made on the phone (`docs/npc_conversations.md` §7.1). `Here` when they are
/// in the room, so the room hears it.
pub(super) enum Reached {
    Here(String),
    OnThePhone,
}

pub(super) fn party_named(hosted: &Hosted, body: &str, name: &str) -> Option<Reached> {
    if let Some(id) = here_named(hosted, body, name) {
        return Some(Reached::Here(id));
    }
    hosted
        .sim(|s| {
            s.threads
                .by_kind_for(body, ThreadKind::Direct, name)
                .is_some()
        })
        .then_some(Reached::OnThePhone)
}

/// `act` — one body doing something to another.
///
/// Named `act_on` rather than `act` because [`crate::engine::act::Act`] is the
/// type every tool call arrives as, and a free function called `act` beside it
/// reads as the general thing rather than this one tool.
pub(super) fn act_on(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let (Some(on), Some(intent)) = (text(args, "on"), text(args, "intent")) else {
        return Outcome::Refused(
            "You meant to do something to somebody, but not who, or not what by it.".into(),
        );
    };

    /* **Your own body is a thing you can act on.**
     *
     * Aimed at nobody rather than at yourself: the world refuses a body that
     * addresses itself (`Refused::SpeakingToYourself`), and it is right to —
     * *to* somebody means somebody who receives it. Closing your own wound is
     * seen by everyone standing there and felt by nobody else, which is exactly
     * what an unaimed showing is.
     */
    if on.eq_ignore_ascii_case(SELF) {
        // **The room sees only that a body did something to itself — not the
        // intent.** A self-act's intent is the actor's own, and often
        // introspective ("to feel the shift in pressure"): pasting it into what
        // everybody else perceives both misreports what is observable — nobody
        // sees you *feel* anything — and leaks the actor's first-person reasoning
        // into their window, where the narrator passes it straight through
        // ("Ulysses does it: to feel the shift…"). The actor's own outcome keeps
        // the intent; the room gets the act, and the narrator renders that.
        return match hosted.with(|w| w.show(body, None, "does something to themselves".to_string()))
        {
            Ok(()) => Outcome::Did(format!("You do it to yourself: {intent}.")),
            Err(why) => Outcome::Refused(refusal(hosted, &why)),
        };
    }

    let Some(id) = here_named(hosted, body, &on) else {
        return Outcome::Refused(format!("{on} is not here."));
    };
    // Shown to the room and aimed at one person — the same channel a gesture
    // uses, because what everybody else sees is a body doing something to
    // another body either way. What differs is that this one lands.
    //
    // Neutral about what it was: this act covers steadying somebody and putting
    // them through a railing, and the world does not need to know which. The
    // intent is the act, and the narrator is what turns it into prose.
    // A predicate, and the target named **once**. This passed `"to {on}: …"`
    // while also aiming at `on`, so the room rendered the name twice —
    // "to Wren: steadying her, at Wren". The aiming belongs to `to`; the verb
    // belongs here. See [`npc_map::world::World::show`].
    match hosted.with(|w| w.show(body, Some(&id), format!("does it: {intent}"))) {
        Ok(()) => Outcome::Did(format!(
            "You do it: {intent}. {on} feels it and may refuse."
        )),
        Err(why) => Outcome::Refused(refusal(hosted, &why)),
    }
}

pub(super) fn sleep(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let Some(until) = text(args, "until") else {
        return Outcome::Refused("You meant to stop, but did not say until when.".into());
    };
    hosted.with_sim(|s| s.ledger.sleep(body, &until));
    Outcome::Did(format!(
        "You stop, meaning to be up again {until}. Anything urgent will wake you."
    ))
}
