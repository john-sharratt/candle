//! The steps a task's answer accounts for (`docs/zen_coding_challenge_fixes.md`
//! Cause 3). Recorded on the task's conversation as it starts ([`mark`]), and
//! read by each of its rounds ([`of`]), whose answer the grammar ends with a
//! `Steps:` section of them (`ConversationEngine::ending_with_steps`).

use candle_conversation::projection::TimelineId;
use candle_conversation::ConversationEngine;
use candle_conversation::Result as EngineResult;

/// The metadata key a conversation's steps are recorded under, one per line.
pub const KEY: &str = "answer_steps";

/// The value `steps` are recorded as: one per line, each its first line.
fn value_of(steps: &[String]) -> String {
    steps
        .iter()
        .map(|s| s.lines().next().unwrap_or("").trim())
        .collect::<Vec<_>>()
        .join("\n")
}

/// The steps a value [`value_of`] wrote holds.
fn steps_in(value: &str) -> Vec<String> {
    value
        .lines()
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect()
}

/// Record `steps` as what the conversation `timeline`'s answers account for.
pub fn mark(
    engine: &ConversationEngine,
    timeline: TimelineId,
    steps: &[String],
) -> EngineResult<()> {
    engine.set_conversation_metadata(timeline, KEY, &value_of(steps))
}

/// The steps the conversation `timeline`'s answers account for; empty for one
/// given none.
pub fn of(engine: &ConversationEngine, timeline: TimelineId) -> Vec<String> {
    engine
        .conversation_metadata(timeline)
        .and_then(|m| m.get(KEY).map(|v| steps_in(v)))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Steps are recorded a line each and read back as they were** —
    /// exactly this value; a step's later lines are not its own.
    #[test]
    fn steps_read_back_as_recorded() {
        let steps = vec![
            "Implement it.".to_string(),
            "Run the tests\n  on BlackWorld".to_string(),
        ];
        assert_eq!(value_of(&steps), "Implement it.\nRun the tests");
        assert_eq!(
            steps_in(&value_of(&steps)),
            ["Implement it.", "Run the tests"]
        );
        assert!(steps_in("").is_empty());
    }
    /// **A blank line is no step** — `steps_in` skips empty lines, so a value
    /// with a blank line between two steps reads back as the two steps it
    /// holds, not as three with a blank between them.
    #[test]
    fn a_blank_line_is_no_step() {
        assert_eq!(
            steps_in("Build it.\n\nTest it.\n"),
            ["Build it.", "Test it."]
        );
    }
}
