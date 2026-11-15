//! What a mind that *is* its world sees of it: every room with somebody in
//! it, who is there, and what each is doing (`docs/zen_everywhere.md` §4).
//!
//! A body is told the room it stands in. A mind that takes no orders has no
//! body and stands nowhere; the world is its own insides, so what it is told
//! is all of it at once — the one situation, replaced as it changes, never a
//! room at a time. It is carried at idle salience: a view is there to be read
//! when the mind next thinks, and is no reason to think.
//!
//! What *happens* — work taken up, a hand come to rest — is told apart, as it
//! happens ([`crate::engine::runtime::Runtime::tell_whole`]).

use std::collections::BTreeMap;

use npc_map::world::World;

use crate::engine::people;
use crate::sim::Sim;

/// The most of a mission's brief a line of the view carries, in characters.
const BRIEF_SHOWN: usize = 120;

/// What `body` — a mind that takes no orders — sees of the whole world, or
/// `None` for a body that takes them and sees its own room.
pub fn view(world: &World, sim: &Sim, body: &str) -> Option<String> {
    if !sim.missions.is_unordered(body) {
        return None;
    }
    // Room by room, in the map's own order of level then room, so the same
    // world reads the same way twice.
    let mut rooms: BTreeMap<(String, String), Vec<String>> = BTreeMap::new();
    for a in world.actors().filter(|a| a.id != body) {
        let room = world
            .node(&a.at)
            .map_or_else(|| a.at.node.clone(), |n| n.name.clone());
        let level = world
            .map()
            .get(&a.at.area)
            .map_or_else(|| a.at.area.clone(), |area| area.name.clone());
        rooms
            .entry((level, room))
            .or_default()
            .push(format!("{}, {}", a.name, doing(sim, &a.id)));
    }
    let mut s = String::from("Everything inside you, as it stands now:");
    if rooms.is_empty() {
        s.push_str("\nNobody is inside you.");
    }
    for ((level, room), mut who) in rooms {
        who.sort();
        s.push_str(&format!("\n- {room}, {level}: {}", who.join("; ")));
    }
    // What lies outside it is reached through its terminals, and which there
    // are is stated rather than asked for: a mind that had to list them
    // looked around its own rooms instead and reported there were none.
    let terminals = sim.terminals.reachable();
    if !terminals.is_empty() {
        s.push_str(&format!(
            "\nOutside you, reached one at a time with `terminal_connect`: {}.",
            terminals.join(", ")
        ));
    }
    match sim.terminals.connected(body) {
        Some(terminal) => s.push_str(&format!("\nYou are connected to {terminal}.")),
        None => s.push_str("\nYou are connected to none of them."),
    }
    Some(s)
}

/// What the body `id` is doing, as the view says it.
fn doing(sim: &Sim, id: &str) -> String {
    if people::is_person(id) {
        return "a person visiting".to_string();
    }
    match sim.missions.active(id) {
        Some(m) => {
            let brief = shortened(&m.mission_text());
            match m.step_to_do() {
                Some(step) => format!("carrying \"{brief}\" — next: {}", step.text.trim()),
                None => format!("carrying \"{brief}\" — its steps done"),
            }
        }
        None if sim.missions.is_unordered(id) => "keeping its own counsel".to_string(),
        None => "at rest".to_string(),
    }
}

/// The first line of a brief, cut at a word to [`BRIEF_SHOWN`].
fn shortened(brief: &str) -> String {
    let line = brief.lines().next().unwrap_or_default().trim();
    if line.chars().count() <= BRIEF_SHOWN {
        return line.to_string();
    }
    let cut: String = line.chars().take(BRIEF_SHOWN).collect();
    let cut = cut.rsplit_once(' ').map_or(cut.as_str(), |(head, _)| head);
    format!("{cut}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A long brief is cut at a word and marked; a short one is given whole,
    /// and only its first line.
    #[test]
    fn a_brief_is_shown_by_its_first_line_cut_at_a_word() {
        assert_eq!(shortened("Mend the gate.\nThen report."), "Mend the gate.");
        let long = "word ".repeat(60);
        let cut = shortened(&long);
        assert!(cut.ends_with('…'), "{cut}");
        assert!(cut.chars().count() <= BRIEF_SHOWN + 1);
        assert!(!cut.contains("  "));
        assert!(cut.trim_end_matches('…').ends_with("word"));
    }
}
