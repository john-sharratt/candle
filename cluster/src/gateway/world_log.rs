//! What a member's worlds add to the coordination log (§12.3): a world's role
//! changing, a handover it waits on, and each character its authority lends,
//! moves or has freed.

use crate::coord::{NpcLeaseLine, WorldLine};

fn role(w: &WorldLine) -> String {
    match w.epoch {
        Some(epoch) if w.role == "authority" => format!("authority (epoch {epoch})"),
        _ => w.role.clone(),
    }
}

/// Each world's role, for a member's first report.
pub fn first(after: &[WorldLine]) -> Vec<String> {
    after
        .iter()
        .map(|w| format!("world {}: {}", w.world, role(w)))
        .collect()
}

/// The lines `after` adds to `before`.
pub fn changes(before: &[WorldLine], after: &[WorldLine]) -> Vec<String> {
    let mut lines = Vec::new();
    for a in after {
        let b = before.iter().find(|b| b.world == a.world);
        match b {
            None => lines.push(format!("world {}: {}", a.world, role(a))),
            Some(b) if b.role != a.role || b.epoch != a.epoch => {
                lines.push(format!("world {}: {} → {}", a.world, role(b), role(a)))
            }
            Some(_) => {}
        }
        if let Some(w) = &a.waiting {
            if b.and_then(|b| b.waiting.as_ref()) != Some(w) {
                lines.push(format!(
                    "world {}: taking it over once {w} is applied",
                    a.world
                ));
            }
        }
        lines.extend(lent(&a.world, b.map_or(&[][..], |b| &b.lent), &a.lent));
    }
    for b in before {
        if !after.iter().any(|a| a.world == b.world) {
            lines.push(format!("world {}: no longer hosted", b.world));
        }
    }
    lines
}

/// What changed in the leases a world's authority lends.
fn lent(world: &str, before: &[NpcLeaseLine], after: &[NpcLeaseLine]) -> Vec<String> {
    let mut lines = Vec::new();
    for a in after {
        let b = before.iter().find(|b| b.npc == a.npc);
        if a.holder.is_empty() {
            if b.is_none_or(|b| !b.holder.is_empty()) {
                lines.push(format!(
                    "world {world}: character {} freed by an admin",
                    a.npc
                ));
            }
        } else if b.is_none_or(|b| b.holder != a.holder) {
            lines.push(format!(
                "world {world}: character {} lent to {}",
                a.npc, a.holder
            ));
        }
        if let Some(next) = &a.next {
            if b.and_then(|b| b.next.as_ref()) != Some(next) {
                lines.push(format!(
                    "world {world}: character {} moves to {next} at its next turn",
                    a.npc
                ));
            }
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    use std::slice::from_ref;

    use super::*;

    fn world(role: &str, epoch: Option<u64>, lent: Vec<NpcLeaseLine>) -> WorldLine {
        WorldLine {
            world: "w1".into(),
            role: role.into(),
            epoch,
            since_ms: 0,
            waiting: None,
            lent,
        }
    }

    fn lease(npc: u64, holder: &str, next: Option<&str>) -> NpcLeaseLine {
        NpcLeaseLine {
            world: "w1".into(),
            npc,
            holder: holder.into(),
            expires_ms: 60_000,
            next: next.map(String::from),
        }
    }

    #[test]
    fn a_world_changing_hands_is_logged() {
        let replica = world("replica", None, vec![]);
        let mut waiting = replica.clone();
        waiting.waiting = Some("0000000000000001 through 42".into());
        let authority = world("authority", Some(9), vec![]);
        assert_eq!(
            first(from_ref(&authority)),
            ["world w1: authority (epoch 9)"]
        );
        assert_eq!(
            changes(from_ref(&replica), from_ref(&waiting)),
            ["world w1: taking it over once 0000000000000001 through 42 is applied"]
        );
        assert_eq!(
            changes(from_ref(&waiting), from_ref(&authority)),
            ["world w1: replica → authority (epoch 9)"]
        );
        assert!(changes(from_ref(&authority), from_ref(&authority)).is_empty());
        assert_eq!(
            changes(from_ref(&authority), &[]),
            ["world w1: no longer hosted"]
        );
    }

    /// A character lent, moving at its next turn, moved, and freed.
    #[test]
    fn leases_lent_moved_and_freed_are_logged() {
        let a = world("authority", Some(9), vec![lease(7, "a", None)]);
        let moving = world("authority", Some(9), vec![lease(7, "a", Some("b"))]);
        let moved = world("authority", Some(9), vec![lease(7, "b", None)]);
        let freed = world("authority", Some(9), vec![lease(7, "", None)]);
        let none = world("authority", Some(9), vec![]);
        assert_eq!(
            changes(from_ref(&none), from_ref(&a)),
            ["world w1: character 7 lent to a"]
        );
        assert_eq!(
            changes(from_ref(&a), from_ref(&moving)),
            ["world w1: character 7 moves to b at its next turn"]
        );
        assert_eq!(
            changes(from_ref(&moving), from_ref(&moved)),
            ["world w1: character 7 lent to b"]
        );
        assert_eq!(
            changes(from_ref(&moved), from_ref(&freed)),
            ["world w1: character 7 freed by an admin"]
        );
    }
}
