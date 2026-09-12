//! Where everybody else on a floor is, as a line a character reads.
//!
//! **This reverses a choice the map made on purpose.** A node's `habit` is
//! documented as "deliberately not live: it says where to *look* for somebody
//! without saying where anybody is". That left a cast whose standing
//! instruction is to find somebody and talk to them with no way of finding
//! anybody: `scan` sees one named room, perception covers only the room a body
//! stands in, and over a morning of pulses three characters walked two-room
//! circuits for dozens of turns, arriving in rooms the person they wanted had
//! just left. So a character is told who else is on its floor and where — when
//! it arrives on the floor, and when it scans.
//!
//! **The floor, not the world.** That is what somebody walking the corridors
//! could plausibly know, and it keeps the line short on a level of sixty rooms.

use npc_map::text::list;
use npc_map::world::{Where, World};

/// Everybody on `body`'s floor who is not standing in one of `except`, with
/// where they are — `("Yaelis Vayne", "in the barracks")` — ordered by name so
/// the same floor reads the same way twice.
pub fn elsewhere(world: &World, body: &str, except: &[&Where]) -> Vec<(String, String)> {
    let Some(floor) = world.actor(body).map(|a| a.at.area.clone()) else {
        return Vec::new();
    };
    let mut out: Vec<(String, String)> = world
        .actors()
        .filter(|a| a.id != body && a.at.area == floor && !except.contains(&&a.at))
        .map(|a| {
            let at = world
                .node(&a.at)
                .map(|n| format!("{} {}", n.stand().at(), n.name))
                .unwrap_or_else(|| a.at.node.clone());
            (a.name.clone(), at)
        })
        .collect();
    out.sort();
    out
}

/// "Elsewhere on this floor: Wailen Wylde is at the lift and Yaelis Vayne is in
/// the barracks." — or nothing, when nobody else is on the floor outside
/// `except`.
///
/// Nothing rather than "nobody", for the reason every empty field in
/// perception is absent: what an empty floor should prompt depends on who is
/// asking, so the caller phrases it.
pub fn line(world: &World, body: &str, except: &[&Where]) -> Option<String> {
    let people = elsewhere(world, body, except);
    if people.is_empty() {
        return None;
    }
    let clauses: Vec<String> = people
        .iter()
        .map(|(who, at)| format!("{who} is {at}"))
        .collect();
    Some(format!("Elsewhere on this floor: {}.", list(&clauses)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::Hosted;

    fn tower() -> Hosted {
        Hosted::load(
            "battle-cities",
            concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps"),
        )
        .expect("the shipped maps must load")
    }

    fn at(node: &str) -> Where {
        Where::new("tower-redoubt", node)
    }

    /// Each person by name, with the preposition their room takes — *at* the
    /// lift, *in* the barracks — and in a stable order.
    #[test]
    fn a_character_is_told_who_else_is_on_its_floor_and_where() {
        let h = tower();
        h.with(|w| {
            w.enter("c1", "Wren Weaver", at("muster-hall")).unwrap();
            w.enter("c2", "Yaelis Vayne", at("barracks")).unwrap();
            w.enter("c3", "Wailen Wylde", at("core")).unwrap();
        });
        let got = h.read(|w| line(w, "c1", &[&at("muster-hall")]));
        assert_eq!(
            got.as_deref(),
            Some(
                "Elsewhere on this floor: Wailen Wylde is at the lift and Yaelis Vayne is in \
                 the barracks."
            )
        );
    }

    /// Somebody on another floor is not on this one, and an empty floor is no
    /// line at all.
    #[test]
    fn nobody_else_on_the_floor_is_no_line() {
        let h = tower();
        h.with(|w| {
            w.enter("c1", "Wren Weaver", at("muster-hall")).unwrap();
            w.enter("c2", "Soren", Where::new("the-waste", "ruins"))
                .unwrap();
        });
        assert_eq!(h.read(|w| line(w, "c1", &[])), None);
    }

    /// The rooms a caller already reports are left out, so nobody is named
    /// twice.
    #[test]
    fn the_rooms_already_reported_are_left_out() {
        let h = tower();
        h.with(|w| {
            w.enter("c1", "Wren Weaver", at("muster-hall")).unwrap();
            w.enter("c2", "Yaelis Vayne", at("muster-hall")).unwrap();
        });
        assert_eq!(h.read(|w| line(w, "c1", &[&at("muster-hall")])), None);
        let all = h
            .read(|w| line(w, "c1", &[]))
            .expect("not excepted, so named");
        assert!(all.contains("Yaelis Vayne is in the muster hall"), "{all}");
    }
}
