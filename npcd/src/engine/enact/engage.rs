//! Takes and breaks off a stance toward a thing, and refuses an operation that the
//! device is already in.
//!
//! Breaking off is the one posture that ends a stance rather than setting one. Operating
//! a device into the state it is already in is refused, and the refusal says where the
//! thing already is, since a success that changed nothing could not be told from one that
//! did. The grammar no longer offers the current state. This refusal is the backstop for
//! the API and the harness, which the grammar does not constrain.

use super::*;

pub(super) fn engage(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let Some(posture) = text(args, "posture") else {
        return Outcome::Refused("You meant to fight, but did not say how.".into());
    };
    // Breaking off is the one posture that ends rather than sets a stance.
    if posture == "break off" {
        hosted.with_sim(|s| s.field.clear_stance(body));
        return Outcome::Did("You break off. You are not fighting anybody.".into());
    }
    let place = hosted.standpoint_of(body);
    if let Some(target) = text(args, "target") {
        if hosted.sim(|s| s.field.hostile_by_name_at(&place, &target).is_none()) {
            return Outcome::Refused(format!("{target} is not here to fight."));
        }
    }
    let stance = Stance {
        posture: posture.clone(),
        target: text(args, "target"),
        priority: text(args, "priority"),
        filters: Vec::new(),
    };
    let told = match (&stance.target, &stance.priority) {
        (Some(t), Some(p)) => format!("{posture}, on {t}, and otherwise {p}"),
        (Some(t), None) => format!("{posture}, on {t}"),
        (None, Some(p)) => format!("{posture}, taking {p}"),
        (None, None) => posture.clone(),
    };
    hosted.with_sim(|s| s.field.set_stance(body, stance));
    Outcome::Did(format!(
        "You set yourself: {told}. It stands until you change it."
    ))
}

pub(super) fn operate(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let (Some(what), Some(mode)) = (text(args, "what"), text(args, "mode")) else {
        return Outcome::Refused(
            "You meant to work something, but not what, or not into what state.".into(),
        );
    };
    let place = hosted.standpoint_of(body);
    hosted.with_sim(|s| {
        let Some(d) = s.devices.by_name_at_mut(&place, &what) else {
            return Outcome::Refused(format!("There is nothing here called {what}."));
        };
        if !d.working {
            return Outcome::Refused(format!("{} is not working.", d.name));
        }
        let name = d.name.clone();
        let modes = d.modes.join(", ");
        // **Already there is not a thing that can be done.** `Device::set`
        // accepts the state a machine is currently in, so this reported success
        // and changed nothing — three identical `operate` calls in a row, live,
        // with no way for the character to tell. The grammar no longer offers
        // the current state (`Sim::modes_here`); this is the backstop for the
        // API and the harness, which are not grammar-constrained, and it says
        // *where the thing already is* rather than merely refusing.
        if d.mode.eq_ignore_ascii_case(&mode) {
            return Outcome::Refused(format!("{name} is already {}.", d.mode));
        }
        if d.set(&mode) {
            Outcome::Did(format!("You set {name} to {}.", d.mode))
        } else {
            // The grammar normally makes this unreachable; the API and the
            // harness are not grammar-constrained, so the refusal names what
            // the thing does take rather than merely saying no.
            Outcome::Refused(format!("{name} does not take `{mode}`. It takes: {modes}."))
        }
    })
}
