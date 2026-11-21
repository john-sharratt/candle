//! Moves a body to another room by attention, and calls or rides the lift.
//!
//! A body that moves by attention is taken to another room at once. The lift is called to
//! the floor the body is on, and ridden from a landing to another level. A body on a journey
//! is told that the lift is part of it, since telling it to go to the lift first sent it back
//! the way it had come. The grammar offers these acts only where they apply, and the refusals
//! here cover the paths the grammar does not.

use super::*;

/// Turn a body that moves by attention to another room, at once — see
/// [`crate::engine::acts::FOCUS`]. Grammar-bound to [`rooms`]; the refusals
/// here cover the paths a grammar does not.
pub(super) fn focus(hosted: &Hosted, body: &str, a: &Map<String, Value>) -> Outcome {
    let open = rooms(hosted, body);
    if open.is_empty() {
        return Outcome::Refused(
            "You go where you walk: there is no turning your attention to another room.".into(),
        );
    }
    let Some(want) = text(a, "on") else {
        return Outcome::Refused(
            "You meant to turn your attention somewhere, but not where.".into(),
        );
    };
    let Some((name, to)) = open
        .into_iter()
        .find(|(name, _)| name.eq_ignore_ascii_case(&want))
    else {
        return Outcome::Refused(format!("There is no room called \"{want}\" to attend to."));
    };
    match hosted.with(|w| w.place(body, to)) {
        Ok(()) => Outcome::Did(format!("Your attention is on {name}.")),
        Err(why) => Outcome::Refused(refusal(hosted, &why)),
    }
}

/// The refusal of a lift act away from a landing.
///
/// **It sends nobody to the lift.** "Go to it first" taught the three-act way
/// across floors after `move_to` had come to ride the lift itself: a Maker on
/// its way to the command room asked for the lift mid-walk, was told to go to
/// it, and set off back to the room it had just left. A body on a journey is
/// told the lift is part of it; one standing still is told that `move_to`
/// rides it.
pub(super) fn not_at_the_lift(hosted: &Hosted, body: &str) -> String {
    let going = hosted.read(|w| {
        let toward = w.actor(body)?.walk.as_ref()?.toward.clone();
        w.node(&toward).map(|n| n.name.clone())
    });
    match going {
        Some(room) => format!(
            "You are on your way to {room}, and the lift is part of that journey: you need do \
             nothing to ride it. Carry on, and you are told when you arrive."
        ),
        None => "You are not at the lift, and need not be: `move_to` the room you are going \
                 to, on any level, and you ride the lift there."
            .into(),
    }
}

/// Call the lift to the floor you are on. Grammar-gated to the landing with the
/// car away (see [`tools::Availability::AtLift`]); the refusals here cover the
/// paths a grammar does not — the API, an unarmed decode.
pub(super) fn lift_call(hosted: &Hosted, body: &str) -> Outcome {
    let Some(floor) = hosted.read(|w| w.at_landing(body)) else {
        return Outcome::Refused(not_at_the_lift(hosted, body));
    };
    if hosted.read(|w| w.lift().is_some_and(|l| l.boardable_at(floor))) {
        return Outcome::Refused(
            "The lift is already here, its doors open — step in and ride it with `lift_use`."
                .into(),
        );
    }
    hosted.with(|w| w.call_lift(floor));
    Outcome::Did("You call the lift. It is on its way; wait for its doors to open.".into())
}

/// Ride the lift to another level. Grammar-gated to standing on a landing (see
/// [`tools::Availability::AtLanding`]); with the car elsewhere it is called, and
/// the body waits for it and rides when it comes ([`npc_map::world::World::ride_lift`]).
pub(super) fn lift_use(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let Some(name) = text(args, "floor") else {
        return Outcome::Refused(
            "You meant to ride the lift, but did not say to which level.".into(),
        );
    };
    let Some(from) = hosted.read(|w| w.at_landing(body)) else {
        return Outcome::Refused(not_at_the_lift(hosted, body));
    };
    let Some(dest) = hosted.read(|w| w.floor_named(&name)) else {
        return Outcome::Refused(format!("There is no level called \"{name}\" to ride to."));
    };
    if dest == from {
        return Outcome::Refused("You are already on that level.".into());
    }
    let here = hosted.read(|w| w.lift().is_some_and(|l| l.boardable_at(from)));
    if !hosted.with(|w| w.ride_lift(body, dest)) {
        return Outcome::Refused("The lift would not take you.".into());
    }
    Outcome::Did(match here {
        true => format!("You get in and ride the lift toward {name}. It sets off."),
        false => format!(
            "The lift is on another floor. You call it and wait here; when it comes you get in \
             and it takes you to {name}."
        ),
    })
}
