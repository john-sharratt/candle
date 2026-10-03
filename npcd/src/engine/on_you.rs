//! What a character carries with it, in the survey: its conversations and what it
//! has lately made out.
//!
//! These are the body's own, not the room's, so they appear in every survey
//! wherever the character stands. Each is compressed to one line that still reads
//! as a sentence: a conversation is who it is with and how much is unread (what
//! was said is for `read`, never the survey); the recent past is the witnessed
//! paragraph the narrator already writes.

use npc_map::witness;
use npc_map::world::World;

use crate::sim::phone::{Kind, Thread};

/// How many recent happenings are narrated; older ones are dropped.
const LATELY_SHOWN: usize = 12;

/// One conversation: who it is with and how much is unread.
pub fn thread_line(thread: &Thread, me: &str) -> String {
    let name = thread.as_named_to(me);
    let kind = match thread.kind() {
        Kind::Direct => String::new(),
        Kind::Group => format!(" (a group of {})", thread.members.len()),
    };
    let unread = match thread.unread_for(me) {
        0 => "nothing unread".to_string(),
        n => format!("{n} unread"),
    };
    format!("{name}{kind}: {unread}.")
}

/// The recent past as the character would recall it, or nothing when it has seen
/// nothing since it last looked.
pub fn lately(world: &World, body: &str) -> Option<String> {
    let seen = witness::since(world, body);
    let from = seen.len().saturating_sub(LATELY_SHOWN);
    witness::narrate(world, &seen[from..])
}

/// The "on you" section: conversations first, then the recent past. Nothing at
/// all when the character has neither.
pub fn say(threads: &[String], lately: Option<&str>) -> Option<String> {
    let mut out: Vec<String> = Vec::new();
    if !threads.is_empty() {
        out.push(format!(
            "On your phone:\n{}\nTo read them, `read`.",
            threads
                .iter()
                .map(|l| format!("- {l}"))
                .collect::<Vec<_>>()
                .join("\n")
        ));
    }
    if let Some(lately) = lately {
        out.push(format!("Lately: {lately}"));
    }
    match out.is_empty() {
        true => None,
        false => Some(out.join("\n")),
    }
}

#[cfg(test)]
mod tests {
    use crate::sim::phone::Threads;

    use super::*;

    fn talk() -> Threads {
        let mut t = Threads::new();
        t.reach("Wren", "Pax");
        t.send("Pax", "Wren", "asks where the ledger went").unwrap();
        t
    }

    #[test]
    fn a_direct_thread_reads_with_its_unread_and_none_of_its_content() {
        let t = talk();
        let th = t.of("Wren").into_iter().next().unwrap();
        assert_eq!(thread_line(th, "Wren"), "Pax: 1 unread.");
    }

    #[test]
    fn a_thread_with_nothing_unread_says_so() {
        let mut t = Threads::new();
        t.reach("Wren", "Pax");
        let th = t.of("Wren").into_iter().next().unwrap();
        assert_eq!(thread_line(th, "Wren"), "Pax: nothing unread.");
    }

    #[test]
    fn what_was_said_never_appears_in_the_line() {
        let mut t = talk();
        t.send("Wren", "Pax", "says it is in the vault").unwrap();
        let th = t.of("Wren").into_iter().next().unwrap();
        assert!(!thread_line(th, "Wren").contains("vault"));
    }

    #[test]
    fn a_group_is_named_and_counted() {
        let mut t = Threads::new();
        t.open_group(
            "repair crew",
            &["Wren".into(), "Pax".into(), "Soren".into()],
        );
        let th = t.of("Wren").into_iter().next().unwrap();
        assert!(
            thread_line(th, "Wren").starts_with("repair crew (a group of 3): "),
            "{}",
            thread_line(th, "Wren")
        );
    }

    #[test]
    fn the_section_gives_the_phone_then_the_recent_past() {
        let said = say(&["Pax: 1 unread.".into()], Some("Pax came in.")).unwrap();
        assert_eq!(
            said,
            "On your phone:\n- Pax: 1 unread.\nTo read them, `read`.\nLately: Pax came in."
        );
    }

    #[test]
    fn a_character_with_neither_has_no_section() {
        assert_eq!(say(&[], None), None);
    }
}
