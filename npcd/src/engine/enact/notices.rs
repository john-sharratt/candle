//! Leaves words on something in a room, and takes on orders and promises.
//!
//! Posting a notice marks its writer as having read it, so the writer is not offered its own
//! line back. Claiming takes a station and the board in one act, since both are taking something
//! on. Promises are kept under the names the world writes down and not under body ids, because
//! a reminder is offered from the names the company list holds, and a promise stored by id could
//! never be found by name.

use super::*;

/// Leave words on something here, for whoever comes to it next.
///
/// **The writer is marked as having read its own line.** Otherwise posting a
/// notice immediately makes the board unread *for the person who just wrote on
/// it*, so `read` reappears in their grammar, they read their own line back,
/// and the pair of acts is a two-step version of the treadmill this whole
/// change removed. You know what you just wrote.
pub(super) fn post_notice(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let (Some(on), Some(what)) = (text(args, "on"), text(args, "what")) else {
        return Outcome::Refused(
            "You meant to write something down, but not where, or not what.".into(),
        );
    };
    let place = hosted.standpoint_of(body);
    let me = my_name(hosted, body);
    hosted.with_sim(|s| {
        if s.postings.by_name_at(&place, &on).is_none() {
            return Outcome::Refused(format!("There is nothing here called {on} to write on."));
        }
        s.post(&place, &on, &me, &what);
        if let Some(p) = s.postings.by_name_at_mut(&place, &on) {
            p.mark_read(body);
        }
        Outcome::Did(format!(
            "You write it on {on}, where whoever comes next will read it: {what}"
        ))
    })
}

pub(super) fn claim(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let Some(what) = text(args, "what") else {
        return Outcome::Refused("You meant to take something on, but not what.".into());
    };
    let place = hosted.standpoint_of(body);
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
            Err(why) => Outcome::Refused(refusal(hosted, &why)),
        };
    }
    match hosted.with_sim(|s| s.ledger.take_order(&what, body)) {
        Ok(()) => Outcome::Did(format!("You take it on: {what}. Nobody else can now.")),
        Err(why) => Outcome::Refused(order_refusal(hosted, &what, why)),
    }
}

/// What a character is told when an order will not be taken on or handed
/// over: its holder by the name the world knows them by, never a body id.
pub(crate) fn order_refusal(hosted: &Hosted, what: &str, why: OrderRefused) -> String {
    match why {
        OrderRefused::NoSuch => format!("There is no standing order to {what}."),
        OrderRefused::HeldBy(holder) => {
            let name = hosted
                .read(|w| w.actors().find(|a| a.id == holder).map(|a| a.name.clone()))
                .unwrap_or_else(|| "somebody".to_string());
            format!("{name} is holding that one.")
        }
    }
}

pub(super) fn promise(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let (Some(to), Some(what), Some(by)) = (text(args, "to"), text(args, "what"), text(args, "by"))
    else {
        return Outcome::Refused("A promise needs somebody, something, and a time.".into());
    };
    let Some(party) = party_named(hosted, body, &to) else {
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
    // Announced aloud in the room; on the phone the words carry it.
    if let Reached::Here(id) = party {
        let _ = hosted.with(|w| w.tell(body, &id, format!("promising {what} by {by}")));
    }
    Outcome::Did(format!(
        "You promise {to}: {what}, by {by}. It stands until you keep it."
    ))
}

pub(super) fn remind(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let (Some(who), Some(which)) = (text(args, "who"), text(args, "which")) else {
        return Outcome::Refused(
            "You meant to remind somebody of something, but not who, or not what.".into(),
        );
    };
    let Some(party) = party_named(hosted, body, &who) else {
        return Outcome::Refused(format!("{who} is not here."));
    };
    // By name on both sides — see the note in `promise`.
    let me = my_name(hosted, body);
    if hosted.sim(|s| !s.owed_to(&me, &who).contains(&which)) {
        return Outcome::Refused(format!("{who} never promised you that."));
    }
    let said = match party {
        Reached::Here(id) => {
            hosted.with(|w| w.tell(body, &id, format!("reminding them of {which}")))
        }
        Reached::OnThePhone => Ok(()),
    };
    match said {
        Ok(()) => Outcome::Did(format!("You put {who} in mind of it: {which}.")),
        Err(why) => Outcome::Refused(refusal(hosted, &why)),
    }
}
