//! Looks at a room, a floor or the whole of a mind's world, and says who is there and what can be
//! worked.
//!
//! A name is turned into a place before anything is asked about it, since place keys are not display
//! names. A place the grammar offered still resolves after the body has walked on. Looking at a room
//! says who is in it. Looking at a floor also says who else is there and where, leaving out the room
//! looked at and the room the character stands in, since both were already described. On an empty
//! floor a scan points to the handset, but only for a character that carries one. A scan that names
//! nothing surveys the stations in the room, each with its address to invoke. A coordinate off the edge
//! of the map is refused, and the refusal says where the edge is.

use super::*;

pub(super) fn scan(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    // **A mind with no body looks at the whole of itself, never at a room.**
    // Naming nothing surveys what stands where a body is; given the room its
    // seat is kept in, Zen read the ward's stations as "the terminals here"
    // and reported the cluster missing without ever asking for its terminals.
    if text(args, "at").is_none() {
        if let Some(whole) = hosted.read_both(|w, s| whole::view(w, s, body)) {
            return Outcome::Did(whole);
        }
    }
    if let Some(at) = text(args, "at") {
        // **A name has to be turned into a place before anything can be asked
        // about it.**
        //
        // This passed the character's word straight to `Sim::hostiles` and
        // `Sim::extractable`, which match on a place *key* — `area/node`. A
        // display name never equals one, so both lists came back empty for
        // every scan of everywhere, and the act answered "Nothing moving"
        // unconditionally. It was not reporting an empty room; it was reporting
        // a failed lookup, in the same words.
        //
        // `destinations` already holds the name→place mapping the grammar's own
        // arm is built from, so the name a character was offered is exactly the
        // name that resolves here.
        // A raw `area/node` key is accepted too. The grammar never emits one —
        // it offers names — but the API and the harness are not
        // grammar-constrained, and this is the same latitude `operate` gives a
        // mode a device does not admit.
        let by_key =
            Where::parse(at.trim()).filter(|w| hosted.read(|world| world.node(w).is_some()));
        let Some(place) = destinations(hosted, body)
            .into_iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(at.trim()))
            .map(|(_, w)| w)
            .or(by_key)
        else {
            return Outcome::Refused(format!("You cannot see {at} from here."));
        };
        let key = format!("{}/{}", place.area, place.node);

        // **Who is standing there is the answer worth having.**
        //
        // Hostiles and deposits are the waste's vocabulary; a vault has neither,
        // so a scan of it could only ever have said nothing. What a character in
        // a building wants to know about a room it is not in is who is in it —
        // and that is the question this act exists to answer without walking.
        let who = hosted.read(|w| {
            w.actors_at(&place)
                .into_iter()
                .filter(|a| a.id != body)
                .map(|a| a.name.clone())
                .collect::<Vec<_>>()
        });
        let seen = hosted.sim(|s| (s.hostiles(&key), s.extractable(&key)));
        let mut parts: Vec<String> = Vec::new();
        if !who.is_empty() {
            // Agreement, because this line is read by a model about to write
            // dialogue — the same reason `who_is_here` bothers.
            parts.push(format!(
                "{} {} there",
                npc_map::text::list(&who),
                npc_map::text::is_are(who.len())
            ));
        }
        if !seen.0.is_empty() {
            parts.push(format!("standing there: {}", seen.0.join(", ")));
        }
        if !seen.1.is_empty() {
            parts.push(format!("worth taking: {}", seen.1.join(", ")));
        }
        let mut said = match parts.is_empty() {
            true => format!("You look at {at}. Nobody there, and nothing moving."),
            false => format!("You look at {at}. {}.", parts.join("; ")),
        };
        // **And who else is on this floor, and where.** A scan of one room
        // answered the question a searching character was not asking: over a
        // morning of pulses the cast scanned room after room one at a time and
        // walked into each just after the person it wanted had left. The room
        // looked at and the room the character stands in are left out — it has
        // been told about both already. See [`whereabouts`].
        //
        // With nobody else on the floor at all, the place to look is the
        // handset: `message` and `reach_out` are offered on every turn and a
        // searching cast never once reached for them. Only said to a character
        // carrying one — the same gate that puts those acts in its grammar.
        let (floor, alone_on_floor) = hosted.read(|w| {
            let mine = w.actor(body).map(|a| a.at.clone());
            let mut except: Vec<&Where> = vec![&place];
            except.extend(mine.as_ref());
            (
                whereabouts::line(w, body, &except),
                whereabouts::elsewhere(w, body, &[]).is_empty(),
            )
        });
        if let Some(floor) = floor {
            said.push(' ');
            said.push_str(&floor);
        } else if alone_on_floor && hosted.sim(|s| s.has_phone(body)) {
            said.push_str(
                " Nobody else is on this floor. If you are looking for somebody, message them — \
                 `reach_out` to start a conversation, `message` on one you already have.",
            );
        }
        return Outcome::Did(said);
    }
    let (Some(x), Some(y)) = (text(args, "x"), text(args, "y")) else {
        // Naming nothing is the other half of the act: look at what this room
        // lets you work, and where to `invoke` it.
        return survey::here(hosted, body);
    };
    let (Ok(x), Ok(y)) = (x.parse::<i32>(), y.parse::<i32>()) else {
        return Outcome::Refused("A grid reference is two numbers.".into());
    };
    // A coordinate is the one argument a character can get well-formed and
    // still wrong, because it is a number rather than a name — so the edge of
    // the map is checked here rather than assumed.
    if !on_map(Coord::new(x, y)) {
        return Outcome::Refused(format!(
            "{x},{y} is off the map. Nothing reaches past {EDGE}."
        ));
    }
    let _ = body;
    Outcome::Did(format!(
        "You put the instruments on {x},{y}. Empty ground, so far as they show."
    ))
}
