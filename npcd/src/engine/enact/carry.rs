//! Moves things between a body and the ground or another body, and uses what a body
//! carries.
//!
//! Giving and releasing move an item between holders, and an item that is not carried
//! cannot be moved. Equipping keeps the item on the body. Using an item works it and
//! keeps it when it is gear, and spends it when it is a consumable. Gathering takes an
//! item from the ground into the pack.

use super::*;

pub(super) fn give(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let (Some(what), Some(to)) = (text(args, "what"), text(args, "to")) else {
        return Outcome::Refused(
            "You meant to hand something over, but not what, or not to whom.".into(),
        );
    };
    let Some(id) = here_named(hosted, body, &to) else {
        return Outcome::Refused(format!("{to} is not here."));
    };
    let n = count(args, "count");
    match hosted.with_sim(|s| s.hand_over(body, &id, &what, n)) {
        Ok(moved) => Outcome::Did(format!("You hand {to} {} {}.", moved.count, moved.name)),
        Err(why) => Outcome::Refused(why),
    }
}

pub(super) fn release(hosted: &Hosted, body: &str) -> Outcome {
    if let Some(what) = hosted.with_sim(|s| s.ledger.give_back(body)) {
        return Outcome::Did(format!("You put it back for somebody else: {what}."));
    }
    match hosted.with(|w| w.release(body)) {
        Ok(_) => Outcome::Did("You give up what you were holding.".into()),
        Err(_) => Outcome::Refused("You are not holding anything to give back.".into()),
    }
}

pub(super) fn equip(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let Some(what) = text(args, "what") else {
        return Outcome::Refused("You meant to ready something, but not what.".into());
    };
    hosted.with_sim(|s| {
        let Some(id) = s.pack(body).by_name(&what).map(|i| i.id.clone()) else {
            return Outcome::Refused(format!("You are not carrying {what}."));
        };
        if s.pack_mut(body).equip(&id) {
            Outcome::Did(format!("You ready {what}."))
        } else {
            Outcome::Refused(format!("{what} is not a thing you ready."))
        }
    })
}

pub(super) fn use_it(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let Some(what) = text(args, "what") else {
        return Outcome::Refused("You meant to use something, but not what.".into());
    };
    let on = match text(args, "on") {
        None => None,
        Some(name) => match here_named(hosted, body, &name) {
            Some(_) => Some(name),
            None => return Outcome::Refused(format!("{name} is not here.")),
        },
    };
    hosted.with_sim(|s| {
        let Some(item) = s.pack(body).by_name(&what).cloned() else {
            return Outcome::Refused(format!("You are not carrying {what}."));
        };
        if !item.kind.usable() {
            return Outcome::Refused(format!("{what} is not a thing you use."));
        }
        // Gear is worked and stays; a consumable is spent and goes.
        let spent = item.kind == ItemKind::Consumable;
        if spent {
            s.pack_mut(body).spend(&item.id);
        }
        Outcome::Did(match (on, spent) {
            (Some(who), true) => format!("You use the {} on {who}. It is gone.", item.name),
            (Some(who), false) => format!("You turn the {} on {who}.", item.name),
            (None, true) => format!("You use the {}. It is gone.", item.name),
            (None, false) => format!("You work the {}.", item.name),
        })
    })
}

pub(super) fn gather(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let Some(what) = text(args, "what") else {
        return Outcome::Refused("You meant to work something, but not what.".into());
    };
    let place = hosted.standpoint_of(body);
    hosted.with_sim(|s| match s.field.work(&place, &what) {
        None => Outcome::Refused(format!("There is nothing here called {what} to work.")),
        Some((_, 0)) => Outcome::Refused(format!(
            "{what} is worked out. There is nothing left in it."
        )),
        Some((resource, took)) => {
            s.pack_mut(body).add(Item::new(
                resource.name(),
                resource.name(),
                ItemKind::Resource,
                took,
            ));
            Outcome::Did(format!(
                "You work {what} and come away with {took} {}.",
                resource.name()
            ))
        }
    })
}
