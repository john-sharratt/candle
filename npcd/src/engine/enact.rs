//! Performing the acts that touch what the world *holds* rather than its shape.
//!
//! [`super::body`] performs the acts of speech, attention and movement, which
//! run entirely through `npc_map`. These run through [`crate::sim`] as well:
//! what a body carries, what stands in the room with it, what is in the ground,
//! what the tower can afford.
//!
//! # Every act here changes something, or says why not
//!
//! The rule the catalog is held to — an act that reaches no world is a way to
//! spend a turn and look busy — applies hardest here, because these are the acts
//! with state behind them. Each one either mutates [`crate::sim::Sim`] and
//! reports what changed, or refuses in the second person and says what would
//! have made it work.
//!
//! # Refusals still teach, even where the grammar already prevents the mistake
//!
//! Most bad arguments are unreachable: `gather` binds `what` to the deposits
//! actually here, so a seam that is not here is not a name the decode can emit.
//! The refusals below therefore exist for the paths the grammar does not cover —
//! the API, the harness, a checkpoint whose tokenizer left the tree unarmed —
//! and they are written for a character to read rather than for a log.

use serde_json::{Map, Value};

use crate::engine::act::Act;
use crate::engine::body::Outcome;
use crate::sim::field::Stance;
use crate::sim::item::{Item, Kind as ItemKind};
use crate::world::Hosted;

/// The acts this module performs.
pub fn is_mine(tool: &str) -> bool {
    matches!(
        tool,
        "touch"
            | "sleep"
            | "give"
            | "read"
            | "claim"
            | "release"
            | "equip"
            | "use"
            | "gather"
            | "engage"
            | "operate"
            | "recall"
            | "scan"
            | "command_tower"
            | "produce"
            | "promise"
            | "remind"
            | "record_verdict"
            | "message"
            | "invite"
            | "open_group"
            | "sign_off"
            | "reach_out"
    )
}

/// Perform one act. [`Outcome::NotOfTheBody`] for anything not [`is_mine`].
pub fn perform(hosted: &Hosted, body: &str, act: &Act) -> Outcome {
    let a = &act.args;
    match act.tool {
        "touch" => touch(hosted, body, a),
        "sleep" => sleep(hosted, body, a),
        "give" => give(hosted, body, a),
        "read" => read(hosted, body, a),
        "claim" => claim(hosted, body, a),
        "release" => release(hosted, body),
        "equip" => equip(hosted, body, a),
        "use" => use_it(hosted, body, a),
        "gather" => gather(hosted, body, a),
        "engage" => engage(hosted, body, a),
        "operate" => operate(hosted, body, a),
        "recall" => recall(hosted, body),
        "scan" => scan(hosted, body, a),
        "command_tower" => command_tower(hosted, a),
        "produce" => produce(hosted, a),
        "promise" => promise(hosted, body, a),
        "remind" => remind(hosted, body, a),
        "record_verdict" => record_verdict(hosted, body, a),
        "message" => message(hosted, body, a),
        "invite" => invite(hosted, body, a),
        "open_group" => open_group(hosted, body, a),
        "sign_off" => sign_off(hosted, body, a),
        "reach_out" => reach_out(hosted, body, a),
        _ => Outcome::NotOfTheBody,
    }
}

fn text(args: &Map<String, Value>, key: &str) -> Option<String> {
    let s = args.get(key)?.as_str()?.trim();
    (!s.is_empty()).then(|| s.to_string())
}

/// A count written as a string, which is how every number reaches this catalog.
///
/// The grammar can bound a string and cannot bound a JSON number — enumerating
/// a number's states does not terminate, because JSON values nest without limit
/// — so a count arrives as text and is parsed here. An unparseable one is one,
/// which is what a character that wrote "a few" meant.
fn count(args: &Map<String, Value>, key: &str) -> u32 {
    text(args, key)
        .and_then(|s| s.trim().parse::<u32>().ok())
        .unwrap_or(1)
        .max(1)
}

/// This body's own name, as the world writes it. Empty for a body that is not
/// in the world, which no live path reaches.
fn my_name(hosted: &Hosted, body: &str) -> String {
    hosted.read(|w| w.actor(body).map(|a| a.name.clone()).unwrap_or_default())
}

/// Somebody in this room, by the name the world wrote for them.
fn here_named(hosted: &Hosted, body: &str, name: &str) -> Option<String> {
    let want = name.trim().to_lowercase();
    hosted.read(|w| {
        let at = w.actor(body)?.at.clone();
        w.actors()
            .find(|a| a.id != body && a.at == at && a.name.to_lowercase() == want)
            .map(|a| a.id.clone())
    })
}

fn touch(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let (Some(to), Some(intent)) = (text(args, "to"), text(args, "intent")) else {
        return Outcome::Refused(
            "You meant to reach for somebody, but not who, or not what by it.".into(),
        );
    };
    let Some(id) = here_named(hosted, body, &to) else {
        return Outcome::Refused(format!("{to} is not here."));
    };
    // Shown to the room and aimed at one person — the same channel a gesture
    // uses, because what everybody else sees is a body doing something to
    // another body either way. What differs is that this one lands.
    match hosted.with(|w| w.show(body, Some(&id), format!("reaching for {to}: {intent}"))) {
        Ok(()) => Outcome::Did(format!("You take hold: {intent}. {to} feels it and may refuse.")),
        Err(why) => Outcome::Refused(format!("{why:?}")),
    }
}

fn sleep(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let Some(until) = text(args, "until") else {
        return Outcome::Refused("You meant to stop, but did not say until when.".into());
    };
    hosted.with_sim(|s| s.ledger.sleep(body, &until));
    Outcome::Did(format!(
        "You stop, meaning to be up again {until}. Anything urgent will wake you."
    ))
}

fn give(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let (Some(what), Some(to)) = (text(args, "what"), text(args, "to")) else {
        return Outcome::Refused("You meant to hand something over, but not what, or not to whom.".into());
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

fn read(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let Some(what) = text(args, "what") else {
        return Outcome::Refused("You meant to read something, but not what.".into());
    };
    let place = hosted.place_of(body);
    hosted.sim(|s| {
        if let Some(d) = s.devices.by_name_at(&place, &what) {
            return Outcome::Did(format!(
                "You read {}. It stands at `{}`{}.",
                d.name,
                d.mode,
                if d.working { "" } else { ", and it is not working" }
            ));
        }
        let orders = s.ledger.unheld();
        if !orders.is_empty() && what.contains("order") {
            return Outcome::Did(format!("Standing and unheld: {}.", orders.join("; ")));
        }
        Outcome::Refused(format!("There is nothing here called {what} to read."))
    })
}

fn claim(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let Some(what) = text(args, "what") else {
        return Outcome::Refused("You meant to take something on, but not what.".into());
    };
    let place = hosted.place_of(body);
    // A station first, then the board: both are "taking something on", which is
    // why one act covers them, and the world knows which of the two this is.
    let station = hosted.sim(|s| {
        s.devices
            .by_name_at(&place, &what)
            .filter(|d| d.claimable)
            .map(|d| d.id.clone())
    });
    if station.is_some() {
        return match hosted.with(|w| w.take(body, Some(&what))) {
            Ok(_) => Outcome::Did(format!("You take {what}. It is yours until you leave it.")),
            Err(why) => Outcome::Refused(format!("{why:?}")),
        };
    }
    match hosted.with_sim(|s| s.ledger.take_order(&what, body)) {
        Ok(()) => Outcome::Did(format!("You take it on: {what}. Nobody else can now.")),
        Err(why) => Outcome::Refused(why),
    }
}

fn release(hosted: &Hosted, body: &str) -> Outcome {
    if let Some(what) = hosted.with_sim(|s| s.ledger.give_back(body)) {
        return Outcome::Did(format!("You put it back for somebody else: {what}."));
    }
    match hosted.with(|w| w.release(body)) {
        Ok(_) => Outcome::Did("You give up what you were holding.".into()),
        Err(_) => Outcome::Refused("You are not holding anything to give back.".into()),
    }
}

fn equip(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
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

fn use_it(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
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

fn gather(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let Some(what) = text(args, "what") else {
        return Outcome::Refused("You meant to work something, but not what.".into());
    };
    let place = hosted.place_of(body);
    hosted.with_sim(|s| match s.field.work(&place, &what) {
        None => Outcome::Refused(format!("There is nothing here called {what} to work.")),
        Some((_, 0)) => Outcome::Refused(format!("{what} is worked out. There is nothing left in it.")),
        Some((resource, took)) => {
            s.pack_mut(body).add(Item::new(
                resource.name(),
                resource.name(),
                ItemKind::Resource,
                took,
            ));
            Outcome::Did(format!("You work {what} and come away with {took} {}.", resource.name()))
        }
    })
}

fn engage(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let Some(posture) = text(args, "posture") else {
        return Outcome::Refused("You meant to fight, but did not say how.".into());
    };
    // Breaking off is the one posture that ends rather than sets a stance.
    if posture == "break off" {
        hosted.with_sim(|s| s.field.clear_stance(body));
        return Outcome::Did("You break off. You are not fighting anybody.".into());
    }
    let place = hosted.place_of(body);
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
    Outcome::Did(format!("You set yourself: {told}. It stands until you change it."))
}

fn operate(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let (Some(what), Some(mode)) = (text(args, "what"), text(args, "mode")) else {
        return Outcome::Refused("You meant to work something, but not what, or not into what state.".into());
    };
    let place = hosted.place_of(body);
    hosted.with_sim(|s| {
        let Some(d) = s.devices.by_name_at_mut(&place, &what) else {
            return Outcome::Refused(format!("There is nothing here called {what}."));
        };
        if !d.working {
            return Outcome::Refused(format!("{} is not working.", d.name));
        }
        let name = d.name.clone();
        let modes = d.modes.join(", ");
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

fn recall(hosted: &Hosted, body: &str) -> Outcome {
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
    match hosted.with(|w| w.place(body, to.clone())) {
        Ok(()) => Outcome::Did("The ground goes out from under you, and you are home.".into()),
        Err(why) => Outcome::Refused(format!("{why:?}")),
    }
}

fn scan(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    if let Some(at) = text(args, "at") {
        let seen = hosted.sim(|s| (s.hostiles(&at), s.extractable(&at)));
        return Outcome::Did(match seen {
            (h, d) if h.is_empty() && d.is_empty() => format!("You look at {at}. Nothing moving."),
            (h, d) => format!(
                "You look at {at}. {}{}",
                if h.is_empty() { String::new() } else { format!("Standing there: {}. ", h.join(", ")) },
                if d.is_empty() { String::new() } else { format!("Worth taking: {}.", d.join(", ")) }
            ),
        });
    }
    let (Some(x), Some(y)) = (text(args, "x"), text(args, "y")) else {
        return Outcome::Refused("You meant to look somewhere, but gave neither a place nor a reference.".into());
    };
    let (Ok(x), Ok(y)) = (x.parse::<i32>(), y.parse::<i32>()) else {
        return Outcome::Refused("A grid reference is two numbers.".into());
    };
    // A coordinate is the one argument a character can get well-formed and
    // still wrong, because it is a number rather than a name — so the edge of
    // the map is checked here rather than assumed.
    if !crate::sim::tower::on_map(crate::sim::tower::Coord::new(x, y)) {
        return Outcome::Refused(format!(
            "{x},{y} is off the map. Nothing reaches past {}.",
            crate::sim::tower::EDGE
        ));
    }
    let _ = body;
    Outcome::Did(format!("You put the instruments on {x},{y}. Empty ground, so far as they show."))
}

fn command_tower(hosted: &Hosted, args: &Map<String, Value>) -> Outcome {
    let Some(action) = text(args, "action") else {
        return Outcome::Refused("You meant to tell the tower something, but not what.".into());
    };
    hosted.with_sim(|s| {
        let Some(tower) = s.tower.as_mut() else {
            return Outcome::Refused("There is no tower here to command.".into());
        };
        if !tower.actions().contains(&action) {
            return Outcome::Refused(format!(
                "The tower cannot {action} just now. It can: {}.",
                tower.actions().join(", ")
            ));
        }
        use crate::sim::field::Resource;
        use crate::sim::tower::{Coord, Posture, FOLD_COST, SIEGE_COST};
        match action.as_str() {
            "relocate" => {
                let (Some(x), Some(y)) = (text(args, "x"), text(args, "y")) else {
                    return Outcome::Refused("A fold needs somewhere to fold to.".into());
                };
                let (Ok(x), Ok(y)) = (x.parse::<i32>(), y.parse::<i32>()) else {
                    return Outcome::Refused("A destination is two numbers.".into());
                };
                if !crate::sim::tower::on_map(Coord::new(x, y)) {
                    return Outcome::Refused("That is off the map.".into());
                }
                tower.draw(Resource::Energy, FOLD_COST);
                tower.at = Coord::new(x, y);
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
                let depth: u32 = text(args, "depth").and_then(|d| d.parse().ok()).unwrap_or(20);
                tower.draw(Resource::Energy, crate::sim::tower::DRILL_COST_PER_METRE * depth as u64);
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
    })
}

fn produce(hosted: &Hosted, args: &Map<String, Value>) -> Outcome {
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

fn promise(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let (Some(to), Some(what), Some(by)) = (text(args, "to"), text(args, "what"), text(args, "by"))
    else {
        return Outcome::Refused("A promise needs somebody, something, and a time.".into());
    };
    let Some(id) = here_named(hosted, body, &to) else {
        return Outcome::Refused(format!("{to} is not here."));
    };
    // **Promises are kept under the names the world writes down, not under body
    // ids.** `remind` binds its argument to what somebody owes you, and that
    // set is built from the company list — which is names. Storing ids here and
    // asking by name there meant the set came back empty and `remind` was never
    // offered at all: the act existed, the promise existed, and the two could
    // not see each other. The audit's own rule — one address per person —
    // walked into on the first tool that needed it.
    let me = my_name(hosted, body);
    hosted.with_sim(|s| s.ledger.promise(&me, &to, &what, &by));
    let _ = hosted.with(|w| w.tell(body, &id, format!("promising {what} by {by}")));
    Outcome::Did(format!("You promise {to}: {what}, by {by}. It stands until you keep it."))
}

fn remind(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let (Some(who), Some(which)) = (text(args, "who"), text(args, "which")) else {
        return Outcome::Refused("You meant to remind somebody of something, but not who, or not what.".into());
    };
    let Some(id) = here_named(hosted, body, &who) else {
        return Outcome::Refused(format!("{who} is not here."));
    };
    // By name on both sides — see the note in `promise`.
    let me = my_name(hosted, body);
    if hosted.sim(|s| !s.owed_to(&me, &who).contains(&which)) {
        return Outcome::Refused(format!("{who} never promised you that."));
    }
    match hosted.with(|w| w.tell(body, &id, format!("reminding them of {which}"))) {
        Ok(()) => Outcome::Did(format!("You put {who} in mind of it: {which}.")),
        Err(why) => Outcome::Refused(format!("{why:?}")),
    }
}

fn record_verdict(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let (Some(on), Some(judgement)) = (text(args, "on"), text(args, "judgement")) else {
        return Outcome::Refused("A verdict needs a thing judged and a judgement.".into());
    };
    let remedy = text(args, "what_would_change_it");
    hosted.with_sim(|s| s.ledger.record_verdict(&on, body, &judgement, remedy.as_deref()));
    Outcome::Did(match remedy {
        Some(r) => format!("Your verdict on {on} stands against it: {judgement}. What would change it: {r}."),
        None => format!("Your verdict on {on} stands against it: {judgement}."),
    })
}

// ── the phone ───────────────────────────────────────────────────────────────
//
// Every one of these reaches somebody who is not here, from wherever the
// character is standing, at a time that is not necessarily now. That is the
// whole difference from `say` and `tell`, and it is why they are separate acts.

fn message(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let (Some(to), Some(intent)) = (text(args, "to"), text(args, "intent")) else {
        return Outcome::Refused(
            "You meant to send something, but not to which conversation, or not what.".into(),
        );
    };
    let me = my_name(hosted, body);
    hosted.with_sim(|s| match s.threads.send(&me, &to, &intent) {
        Ok(reached) => Outcome::Did(format!(
            "You send it to {to}: {intent}. {} will see it when they next look.",
            npc_map::text::list(&reached)
        )),
        Err(why) => Outcome::Refused(why),
    })
}

fn invite(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let (Some(to), Some(who)) = (text(args, "to"), text(args, "who")) else {
        return Outcome::Refused(
            "You meant to bring somebody in, but not to which conversation, or not who.".into(),
        );
    };
    let me = my_name(hosted, body);
    hosted.with_sim(|s| match s.threads.invite(&me, &to, &who) {
        Ok(kind) => Outcome::Did(match kind {
            crate::sim::phone::Kind::Group => format!(
                "{who} is on it. What is said there now reaches all of you."
            ),
            crate::sim::phone::Kind::Direct => format!("{who} is on it."),
        }),
        Err(why) => Outcome::Refused(why),
    })
}

fn open_group(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let (Some(called), Some(with)) = (text(args, "called"), text(args, "with")) else {
        return Outcome::Refused("A group needs a name and somebody to be on it.".into());
    };
    let me = my_name(hosted, body);
    let mut members = vec![me.clone()];
    members.extend(
        with.split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty() && *s != me),
    );
    if members.len() < 2 {
        return Outcome::Refused("A group of one is a note to yourself.".into());
    }
    hosted.with_sim(|s| {
        // **A name this character already uses is refused.** Two threads called
        // the same thing are ambiguous to every act that names one — `message`,
        // `invite` and `sign_off` all resolve by the name the character uses,
        // and would silently pick whichever came first. Worse, the pair became
        // two identical arms in the turn's grammar, which the stencil refuses
        // outright: the whole tree failed to compile, the turn free-decoded,
        // and the character was told its output was not a call. Every turn,
        // permanently, because nothing it could do afterwards removed the
        // duplicate.
        if s.threads.by_name_for(&me, &called).is_some() {
            return Outcome::Refused(format!(
                "You are already on something called {called}. Give this one a name of its own, \
                 or say what you meant to say on the one you have."
            ));
        }
        s.threads.open_group(&called, &members);
        if let Some(intent) = text(args, "intent") {
            let _ = s.threads.send(&me, &called, &intent);
        }
        Outcome::Did(format!(
            "{called} is open, with {} on it.",
            npc_map::text::list(&members[1..])
        ))
    })
}

/// Leaving a thread. The thread carries on without you unless nobody is left.
fn sign_off(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let (Some(to), Some(intent)) = (text(args, "to"), text(args, "intent")) else {
        return Outcome::Refused(
            "You meant to go, but not from which conversation, or not why.".into(),
        );
    };
    let me = my_name(hosted, body);
    hosted.with_sim(|s| {
        // Said before leaving, so the parting lands on the thread rather than
        // vanishing with the leaver.
        let _ = s.threads.send(&me, &to, &intent);
        match s.threads.leave(&me, &to) {
            Ok(()) => Outcome::Did(format!("You leave {to}, having said why: {intent}")),
            Err(why) => Outcome::Refused(why),
        }
    })
}

fn reach_out(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let (Some(to), Some(intent)) = (text(args, "to"), text(args, "intent")) else {
        return Outcome::Refused(
            "You meant to reach somebody, but not who, or not what about.".into(),
        );
    };
    let me = my_name(hosted, body);
    hosted.with_sim(|s| {
        s.threads.reach(&me, &to);
        let _ = s.threads.send(&me, &to, &intent);
        Outcome::Did(format!("You get hold of {to}: {intent}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::act::Act;
    use crate::sim::field::Resource;
    use npc_map::world::Where;
    use serde_json::json;

    fn act(tool: &'static str, args: Value) -> Act {
        Act {
            tool,
            args: args.as_object().unwrap().clone(),
        }
    }

    /// A world with two bodies in the ruins, kitted out.
    fn waste() -> Hosted {
        let h = Hosted::load(
            "battle-cities",
            concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps"),
        )
        .expect("the shipped maps must load");
        let at = Where::new("the-waste", "ruins");
        h.with(|w| {
            w.enter("c1", "Wren", at.clone()).unwrap();
            w.enter("c2", "Soren", at).unwrap();
        });
        h.with_sim(|s| {
            crate::sim::seed::outfit(s, "c1");
            crate::sim::seed::outfit(s, "c2");
        });
        h
    }

    #[test]
    fn every_act_in_the_catalog_is_dispatched_by_somebody() {
        // The other half of `every_act_offered_is_one_that_does_something`: that
        // test asks whether an offered act reaches a world, this one asks
        // whether anything in this module names an act that does not exist.
        for name in crate::engine::tools::CATALOG.iter().map(|t| t.name) {
            // `send_image` is the one act performed above this layer: it
            // renders through the image guest and lands in an interaction, so
            // no body reaches a world for it. Named here rather than filtered
            // by category, so a second such act has to be argued for.
            if name == "send_image" {
                continue;
            }
            assert!(
                crate::engine::body::is_of_the_body(name) || is_mine(name),
                "`{name}` is in the catalog and nothing performs it"
            );
        }
    }

    #[test]
    fn giving_moves_the_thing_and_leaves_the_giver_short() {
        let h = waste();
        let before = h.sim(|s| s.pack("c1").get("bolt").unwrap().count);
        let out = perform(&h, "c1", &act("give", json!({"what":"bolt rounds","to":"Soren","count":"20"})));
        assert!(out.happened(), "{out:?}");
        assert_eq!(h.sim(|s| s.pack("c1").get("bolt").unwrap().count), before - 20);
        assert_eq!(h.sim(|s| s.pack("c2").get("bolt").unwrap().count), 80);
    }

    #[test]
    fn giving_more_than_is_carried_moves_nothing() {
        let h = waste();
        let out = perform(&h, "c1", &act("give", json!({"what":"bolt rounds","to":"Soren","count":"999"})));
        assert!(!out.happened());
        assert_eq!(h.sim(|s| s.pack("c1").get("bolt").unwrap().count), 60);
    }

    #[test]
    fn a_stimpak_used_on_somebody_is_spent_and_a_scanner_is_not() {
        let h = waste();
        assert!(perform(&h, "c1", &act("use", json!({"what":"stimpak","on":"Soren"}))).happened());
        assert_eq!(h.sim(|s| s.pack("c1").get("stimpak").unwrap().count), 2);

        assert!(perform(&h, "c1", &act("use", json!({"what":"advanced scanner"}))).happened());
        assert_eq!(
            h.sim(|s| s.pack("c1").get("scanner").unwrap().count),
            1,
            "gear was consumed"
        );
    }

    #[test]
    fn gathering_takes_from_the_ground_and_puts_it_in_the_pack() {
        let h = waste();
        let out = perform(&h, "c1", &act("gather", json!({"what":"the burnt-out carrier"})));
        assert!(out.happened(), "{out:?}");
        assert_eq!(h.sim(|s| s.pack("c1").get("metal").map(|i| i.count)), Some(10));
        assert_eq!(h.sim(|s| s.field.deposit("burnt_carrier").unwrap().remaining), 20);
    }

    #[test]
    fn gathering_something_that_is_not_here_takes_nothing() {
        let h = waste();
        assert!(!perform(&h, "c1", &act("gather", json!({"what":"the ore seam"}))).happened());
        assert!(h.sim(|s| s.pack("c1").get("ore").is_none()));
    }

    #[test]
    fn a_stance_stands_until_it_is_broken_off() {
        let h = waste();
        let out = perform(
            &h,
            "c1",
            &act("engage", json!({"posture":"press","target":"a mech","priority":"nearest"})),
        );
        assert!(out.happened(), "{out:?}");
        assert_eq!(h.sim(|s| s.field.stance("c1").unwrap().posture.clone()), "press");

        assert!(perform(&h, "c1", &act("engage", json!({"posture":"break off"}))).happened());
        assert!(h.sim(|s| s.field.stance("c1").is_none()), "breaking off left a stance");
    }

    #[test]
    fn engaging_something_that_is_not_here_is_refused() {
        let h = waste();
        let out = perform(&h, "c1", &act("engage", json!({"posture":"press","target":"a stinger"})));
        assert!(!out.happened());
        assert!(h.sim(|s| s.field.stance("c1").is_none()));
    }

    #[test]
    fn a_door_takes_its_own_states_and_not_a_turrets() {
        let h = waste();
        h.with(|w| w.place("c1", Where::new("tower-redoubt", "gatehouse")).unwrap());
        let out = perform(&h, "c1", &act("operate", json!({"what":"the blast door","mode":"locked"})));
        assert!(out.happened(), "{out:?}");
        assert_eq!(
            h.sim(|s| s.devices
                .by_name_at("tower-redoubt/gatehouse", "the blast door")
                .unwrap()
                .mode
                .clone()),
            "locked"
        );

        let bad = perform(&h, "c1", &act("operate", json!({"what":"the blast door","mode":"air only"})));
        assert!(!bad.happened());
        assert!(
            bad.line().unwrap().contains("open"),
            "the refusal should name what it does take: {bad:?}"
        );
    }

    #[test]
    fn sleeping_is_recorded_so_something_can_wake_you() {
        let h = waste();
        assert!(perform(&h, "c1", &act("sleep", json!({"until":"dawn"}))).happened());
        assert_eq!(h.sim(|s| s.ledger.asleep("c1").map(str::to_string)), Some("dawn".into()));
    }

    #[test]
    fn a_promise_stands_and_can_then_be_recalled_but_only_the_real_one() {
        let h = waste();
        assert!(perform(
            &h,
            "c1",
            &act("promise", json!({"to":"Soren","what":"the eastern sweep","by":"dusk"}))
        )
        .happened());

        // Soren reminding Wren of it works; reminding of something never
        // promised does not.
        let good = perform(&h, "c2", &act("remind", json!({"who":"Wren","which":"the eastern sweep"})));
        assert!(good.happened(), "{good:?}");
        let bad = perform(&h, "c2", &act("remind", json!({"who":"Wren","which":"the western sweep"})));
        assert!(!bad.happened(), "a promise nobody made was recalled");
    }

    #[test]
    fn a_verdict_attaches_to_the_thing_judged() {
        let h = waste();
        let out = perform(
            &h,
            "c1",
            &act("record_verdict", json!({"on":"the ridge survey","judgement":"it will not do","what_would_change_it":"the count checked"})),
        );
        assert!(out.happened(), "{out:?}");
        assert_eq!(h.sim(|s| s.ledger.verdicts_on("the ridge survey").len()), 1);
    }

    #[test]
    fn a_coordinate_off_the_edge_of_the_world_is_refused_and_says_where_the_edge_is() {
        let h = waste();
        let out = perform(&h, "c1", &act("scan", json!({"x":"999999","y":"0"})));
        assert!(!out.happened());
        assert!(out.line().unwrap().contains("off the map"), "{out:?}");
    }

    #[test]
    fn the_tower_refuses_what_it_cannot_afford_and_names_what_it_can() {
        let h = waste();
        h.with_sim(|s| {
            let t = s.tower.as_mut().unwrap();
            t.draw(Resource::Energy, t.stock_of(Resource::Energy));
        });
        let out = perform(&h, "c1", &act("command_tower", json!({"action":"relocate","x":"10","y":"10"})));
        assert!(!out.happened());
        assert!(out.line().unwrap().contains("cannot relocate"), "{out:?}");
    }

    #[test]
    fn folding_the_tower_moves_it_and_spends_the_energy() {
        let h = waste();
        let before = h.sim(|s| s.tower.as_ref().unwrap().stock_of(Resource::Energy));
        let out = perform(&h, "c1", &act("command_tower", json!({"action":"relocate","x":"-300","y":"180"})));
        assert!(out.happened(), "{out:?}");
        h.sim(|s| {
            let t = s.tower.as_ref().unwrap();
            assert_eq!((t.at.x, t.at.y), (-300, 180));
            assert!(t.stock_of(Resource::Energy) < before);
        });
    }

    #[test]
    fn a_batch_the_stockpile_cannot_cover_is_refused_and_says_what_is_short() {
        let h = waste();
        let out = perform(&h, "c1", &act("produce", json!({"what":"a companion"})));
        assert!(!out.happened());
        assert!(out.line().unwrap().contains("nanobots"), "{out:?}");
    }
}
