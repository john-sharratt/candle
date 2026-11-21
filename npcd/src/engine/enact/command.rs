//! Commands the tower, and records a verdict on a thing a body has judged.
//!
//! While somebody else is deciding what the tower does about a contact, the tower is
//! theirs to command, and a body that reaches for it is told whose it is. An act that
//! lands settles whatever decisions the body was holding. A verdict is attached to the
//! thing it judges.

use super::*;

/// Command the tower. While somebody else is deciding what it does about
/// something ([`crate::sim::decisions`]) it is theirs to command, and a body
/// reaching for it is told whose it is. An act that lands settles whatever
/// decisions the body was holding.
pub(super) fn command_tower(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let Some(action) = text(args, "action") else {
        return Outcome::Refused("You meant to tell the tower something, but not what.".into());
    };
    hosted.with_both(|world, s| {
        if let Some((decision, holder)) = s.tower_held_against(body) {
            let name = world
                .actor(&holder)
                .map_or(holder.clone(), |a| a.name.clone());
            return Outcome::Refused(format!(
                "{name} is deciding this: {decision}. The tower is theirs to command until it is settled."
            ));
        }
        let outcome = command_the_tower(s, &action, args);
        if matches!(outcome, Outcome::Did(_)) {
            s.settle_tower_decisions(body);
        }
        outcome
    })
}

pub(super) fn command_the_tower(s: &mut Sim, action: &str, args: &Map<String, Value>) -> Outcome {
    {
        let Some(tower) = s.tower.as_mut() else {
            return Outcome::Refused("There is no tower here to command.".into());
        };
        if !tower.actions().contains(&action.to_string()) {
            return Outcome::Refused(format!(
                "The tower cannot {action} just now. It can: {}.",
                tower.actions().join(", ")
            ));
        }
        match action {
            "relocate" => {
                let (Some(x), Some(y)) = (whole(args, "x"), whole(args, "y")) else {
                    return Outcome::Refused("A fold needs somewhere to fold to.".into());
                };
                let (Ok(x), Ok(y)) = (i32::try_from(x), i32::try_from(y)) else {
                    return Outcome::Refused("That is off the map.".into());
                };
                if !on_map(Coord::new(x, y)) {
                    return Outcome::Refused("That is off the map.".into());
                }
                tower.fold_to(Coord::new(x, y));
                Outcome::Did(format!("The tower folds. It stands at {x},{y}."))
            }
            "siege" => {
                let Some(target) = text(args, "target") else {
                    return Outcome::Refused("A siege needs something to besiege.".into());
                };
                tower.draw(Resource::Energy, SIEGE_COST);
                tower.besieging = Some(target.clone());
                tower.posture = Posture::Besieging;
                Outcome::Did(format!("The tower commits against {target}."))
            }
            "lift siege" => {
                let was = tower.besieging.take();
                tower.posture = Posture::Standing;
                Outcome::Did(match was {
                    Some(t) => format!("The tower lifts off {t}."),
                    None => "The siege is lifted.".into(),
                })
            }
            "drill down" => {
                let Some(depth) = whole(args, "depth").filter(|d| *d > 0) else {
                    return Outcome::Refused(
                        "A drill needs a depth to drill to, a whole number of metres.".into(),
                    );
                };
                let Ok(depth) = u32::try_from(depth) else {
                    return Outcome::Refused("That is deeper than the ground goes.".into());
                };
                if !tower.draw(Resource::Energy, DRILL_COST_PER_METRE * depth as u64) {
                    let afford = tower.stock_of(Resource::Energy) / DRILL_COST_PER_METRE;
                    return Outcome::Refused(format!(
                        "The stockpile turns the drill {afford} metres, not {depth}."
                    ));
                }
                tower.depth = depth;
                tower.posture = Posture::DugIn;
                Outcome::Did(format!("The tower drills in, {depth} metres down."))
            }
            "surface" => {
                tower.depth = 0;
                tower.posture = Posture::Standing;
                Outcome::Did("The tower comes back up onto its legs.".into())
            }
            "raise shields" => {
                tower.shields = true;
                Outcome::Did("The shields come up.".into())
            }
            "drop shields" => {
                tower.shields = false;
                Outcome::Did("The shields go down.".into())
            }
            other => Outcome::Refused(format!("Nothing here knows how to {other}.")),
        }
    }
}

pub(super) fn produce(hosted: &Hosted, args: &Map<String, Value>) -> Outcome {
    let Some(what) = text(args, "what") else {
        return Outcome::Refused("You meant to have something made, but not what.".into());
    };
    let n = count(args, "count");
    let queue: u8 = text(args, "queue")
        .and_then(|q| q.parse().ok())
        .unwrap_or(1);
    hosted.with_sim(|s| {
        let Some(tower) = s.tower.as_mut() else {
            return Outcome::Refused("There is nothing here that makes anything.".into());
        };
        match tower.queue(&what, n, queue) {
            Ok(b) => Outcome::Did(format!("{n} of {what} on queue {}.", b.queue)),
            Err(why) => Outcome::Refused(why),
        }
    })
}

pub(super) fn record_verdict(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let (Some(on), Some(judgement)) = (text(args, "on"), text(args, "judgement")) else {
        return Outcome::Refused("A verdict needs a thing judged and a judgement.".into());
    };
    let remedy = text(args, "what_would_change_it");
    hosted.with_sim(|s| {
        s.ledger
            .record_verdict(&on, body, &judgement, remedy.as_deref())
    });
    Outcome::Did(match remedy {
        Some(r) => format!(
            "Your verdict on {on} stands against it: {judgement}. What would change it: {r}."
        ),
        None => format!("Your verdict on {on} stands against it: {judgement}."),
    })
}
