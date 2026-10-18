//! Named load patterns: fixed mixes of the ladder's tasks, each part a fixed
//! number of clients starting at a fixed offset, run against one member or
//! the gateway so the same pattern can be compared across a member alone, the
//! other alone, and the cluster.
//!
//! | Pattern | Parts |
//! |---|---|
//! | `heavy-decode` | 24 passthrough clients |
//! | `heavy-read` | 24 zen-code readers |
//! | `blend` | 12 passthrough chats and 12 readers, at once |
//! | `shift` | a 24-client decode wave; 60 s in, a 24-client read burst beside it; 180 s in, a second decode wave |

use std::sync::Arc;
use std::time::Duration;

use tokio::task::JoinSet;
use tokio::time::sleep;

use crate::client::Target;
use crate::ladder::Scenario;
use crate::run::{run, Ran};
use crate::tasks::Task;

/// One part of a pattern: a scenario, and how long after the pattern starts
/// its clients begin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Part {
    pub scenario: Scenario,
    pub start: Duration,
}

/// A named pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pattern {
    pub name: String,
    pub parts: Vec<Part>,
}

fn part(name: &str, task: Task, clients: usize, start_s: u64) -> Part {
    Part {
        scenario: Scenario {
            name: name.to_string(),
            task,
            clients,
            turns: 2,
            max_tokens: task.max_tokens(),
        },
        start: Duration::from_secs(start_s),
    }
}

/// The pattern called `name`.
pub fn pattern(name: &str) -> Option<Pattern> {
    let parts = match name {
        "heavy-decode" => vec![part("heavy-decode", Task::Decode, 24, 0)],
        "heavy-read" => vec![part("heavy-read", Task::Read, 24, 0)],
        "blend" => vec![
            part("blend-chat", Task::Decode, 12, 0),
            part("blend-read", Task::Read, 12, 0),
        ],
        "shift" => vec![
            part("shift-decode-1", Task::Decode, 24, 0),
            part("shift-read", Task::Read, 24, 60),
            part("shift-decode-2", Task::Decode, 24, 180),
        ],
        _ => return None,
    };
    Some(Pattern {
        name: name.to_string(),
        parts,
    })
}

/// Every part of `p` against `target`, each started at its offset, under the
/// run name `run` with its part's name appended; each turn given at most
/// `limit`. The parts' results, in the pattern's order.
pub async fn run_pattern(
    target: Arc<Target>,
    p: &Pattern,
    run_name: &str,
    limit: Duration,
) -> Vec<(Scenario, String, Ran)> {
    let mut parts = JoinSet::new();
    for (i, part) in p.parts.iter().cloned().enumerate() {
        let target = Arc::clone(&target);
        let name = format!("{run_name}-{}", part.scenario.name);
        parts.spawn(async move {
            sleep(part.start).await;
            let ran = run(target, &part.scenario, &name, limit).await;
            (i, part.scenario, name, ran)
        });
    }
    let mut out = Vec::new();
    while let Some(joined) = parts.join_next().await {
        if let Ok(done) = joined {
            out.push(done);
        }
    }
    out.sort_by_key(|(i, ..)| *i);
    out.into_iter().map(|(_, s, n, r)| (s, n, r)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every pattern is a fixed shape; the client counts and offsets are what
    /// makes one run comparable with the next.
    #[test]
    fn the_patterns_are_fixed() {
        let shape = |name: &str| -> Vec<(String, Task, usize, u64)> {
            pattern(name)
                .unwrap()
                .parts
                .into_iter()
                .map(|p| {
                    (
                        p.scenario.name,
                        p.scenario.task,
                        p.scenario.clients,
                        p.start.as_secs(),
                    )
                })
                .collect()
        };
        assert_eq!(
            shape("blend"),
            vec![
                ("blend-chat".to_string(), Task::Decode, 12, 0),
                ("blend-read".to_string(), Task::Read, 12, 0),
            ]
        );
        assert_eq!(
            shape("shift"),
            vec![
                ("shift-decode-1".to_string(), Task::Decode, 24, 0),
                ("shift-read".to_string(), Task::Read, 24, 60),
                ("shift-decode-2".to_string(), Task::Decode, 24, 180),
            ]
        );
        assert_eq!(shape("heavy-read").len(), 1);
        assert!(pattern("everything").is_none());
    }
}
