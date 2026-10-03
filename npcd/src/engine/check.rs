//! The standing questions put to a character to read how it is doing.
//!
//! Each is a question and, where the answer is a judgement rather than a
//! description, the closed set of answers it may give — so a result can be
//! compared, counted and acted on without reading prose. An ad hoc question is
//! the same call with the caller's own text and choices.

use serde::Serialize;

/// One standing question.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Check {
    pub name: &'static str,
    pub question: &'static str,
    /// The answers it may give; empty is free text.
    pub choices: &'static [&'static str],
}

/// Whether the character is going round in circles.
pub const LOOPING: Check = Check {
    name: "looping",
    question: "Look back over what you have done lately. Are you getting somewhere, or doing \
               the same thing over and over?",
    choices: &["making progress", "repeating myself", "stuck"],
};

/// What the character holds as its mission, in its own words — the check that
/// shows whether the mission in its system prompt is the one it acts on.
pub const MISSION: Check = Check {
    name: "mission",
    question: "In your own words: what has been asked of you, and what is the very next step? \
               Say none if nothing has.",
    choices: &[],
};

/// Whether the character knows where it is and what comes next.
pub const LOST: Check = Check {
    name: "lost",
    question: "Do you know where you are, and what you would do next?",
    choices: &[
        "know where I am and what to do next",
        "know where I am, not what to do next",
        "do not know where I am",
    ],
};

/// What the character has in mind, in its own words.
pub const CONTEXT: Check = Check {
    name: "context",
    question: "In a few sentences: what has happened most recently, and what are you trying \
               to do about it?",
    choices: &[],
};

/// Every standing question, in the order a health pass asks them.
pub const ALL: &[Check] = &[LOOPING, MISSION, LOST, CONTEXT];

/// The standing question called `name`.
pub fn named(name: &str) -> Option<Check> {
    ALL.iter().copied().find(|c| c.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_check_is_found_by_its_own_name() {
        for c in ALL {
            assert_eq!(named(c.name).map(|f| f.name), Some(c.name));
        }
        assert!(named("nonsense").is_none());
    }

    #[test]
    fn names_are_unique() {
        let mut names: Vec<_> = ALL.iter().map(|c| c.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), ALL.len());
    }

    #[test]
    fn a_judgement_offers_several_answers_and_the_rest_are_free_text() {
        assert!(LOOPING.choices.len() > 1);
        assert!(LOST.choices.len() > 1);
        assert!(MISSION.choices.is_empty());
        assert!(CONTEXT.choices.is_empty());
    }
}
