//! The load ladder: light, medium and heavy, each a number of concurrent
//! clients holding one conversation apiece through several turns — so a
//! scenario measures both how the cluster spreads conversations and whether a
//! conversation's later turns wait behind other people's.

use crate::tasks::Task;

/// One rung.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scenario {
    pub name: String,
    pub task: Task,
    /// Clients at once, each its own conversation.
    pub clients: usize,
    /// Turns each client sends, one after another.
    pub turns: usize,
    pub max_tokens: usize,
}

/// The rungs named `level` — `light`, `medium` or `heavy` — for `task`.
pub fn rung(level: &str, task: Task) -> Option<Scenario> {
    let (clients, turns) = match level {
        "light" => (2, 2),
        "medium" => (8, 2),
        "heavy" => (24, 2),
        _ => return None,
    };
    Some(Scenario {
        name: format!("{level}-{}", task.name()),
        task,
        clients,
        turns,
        max_tokens: task.max_tokens(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each level is a fixed shape, named for its level and task; anything
    /// else is no rung.
    #[test]
    fn the_rungs_are_fixed() {
        let heavy = rung("heavy", Task::Decode).unwrap();
        assert_eq!(
            (heavy.name.as_str(), heavy.clients, heavy.turns),
            ("heavy-decode", 24, 2)
        );
        assert_eq!(rung("light", Task::Read).unwrap().clients, 2);
        assert_eq!(rung("medium", Task::Read).unwrap().name, "medium-read");
        assert!(rung("extreme", Task::Read).is_none());
    }
}
