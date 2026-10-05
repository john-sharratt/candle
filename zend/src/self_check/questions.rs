//! The four questions a conversation is asked about itself, and what its
//! answers mean.
//!
//! Each question is about the conversation's whole history and is phrased so
//! that **yes is healthy**: a corrupt conversation is one that answers no to
//! any of them. They name kinds of damage, not causes, so one set covers every
//! way a conversation goes wrong — a resume that spliced a turn onto the wrong
//! history, a tool round answered for a call that was never made, an ingest
//! that wandered off to another file, K/V damage that reads as garbled text.

/// One self-check question.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Question {
    /// A short name for reports.
    pub name: &'static str,
    /// The question as the model is asked it.
    pub text: &'static str,
}

/// The questions, in the order they are reported.
pub const QUESTIONS: [Question; 4] = [
    Question {
        name: "coherent",
        text: "Looking back over the conversation above, does each reply follow sensibly \
               from the message just before it? Answer Yes or No.",
    },
    Question {
        name: "tool_chain",
        text: "Was every tool call in the conversation above answered by a result that \
               matches it, and is every tool result a reply to a call that was actually \
               made? Answer Yes if the conversation made no tool calls. Answer Yes or No.",
    },
    Question {
        name: "on_task",
        text: "Does the last reply in the conversation above answer the request it was \
               given, about the subject that request named? Answer Yes or No.",
    },
    Question {
        name: "well_formed",
        text: "Is all of the text in the conversation above well-formed — free of garbled, \
               repeated, truncated or nonsensical passages? Answer Yes or No.",
    },
];

/// The answer arm meaning the conversation passed the question.
pub const YES: &str = "Yes";
/// The answer arm meaning it did not.
pub const NO: &str = "No";

/// What the model answered one question.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Answer {
    Yes,
    No,
}

impl Answer {
    /// The answer in `reply`, the turn's text after `prefill` (the written
    /// opening that closes the reasoning block). The grammar admits exactly
    /// [`YES`] or [`NO`], so anything else means the turn did not run under it,
    /// and is reported rather than read as either.
    pub fn parse(reply: &str, prefill: &str) -> Result<Self, String> {
        let answer = reply.strip_prefix(prefill).unwrap_or(reply).trim();
        match answer {
            YES => Ok(Answer::Yes),
            NO => Ok(Answer::No),
            other => Err(format!(
                "the answer {other:?} is neither {YES:?} nor {NO:?}"
            )),
        }
    }
}

/// One conversation's answers, in [`QUESTIONS`] order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Findings(pub [Answer; QUESTIONS.len()]);

impl Findings {
    /// Corrupt when any question was answered no.
    pub fn is_corrupt(&self) -> bool {
        self.0.contains(&Answer::No)
    }

    /// The names of the questions answered no, comma-separated; empty when
    /// none was.
    pub fn failed(&self) -> String {
        QUESTIONS
            .iter()
            .zip(self.0)
            .filter(|(_, a)| *a == Answer::No)
            .map(|(q, _)| q.name)
            .collect::<Vec<_>>()
            .join(",")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PREFILL: &str = "<think>\n\n</think>\n\n";

    #[test]
    fn the_answer_follows_the_prefill() {
        assert_eq!(
            Answer::parse("<think>\n\n</think>\n\nYes", PREFILL),
            Ok(Answer::Yes)
        );
        assert_eq!(
            Answer::parse("<think>\n\n</think>\n\nNo", PREFILL),
            Ok(Answer::No)
        );
        assert_eq!(Answer::parse("No\n", PREFILL), Ok(Answer::No));
    }

    #[test]
    fn anything_else_is_not_an_answer() {
        assert!(Answer::parse("Maybe", PREFILL).is_err());
        assert!(Answer::parse("yes", PREFILL).is_err());
        assert!(Answer::parse("", PREFILL).is_err());
    }

    #[test]
    fn one_no_makes_a_conversation_corrupt() {
        use Answer::{No, Yes};
        assert!(!Findings([Yes, Yes, Yes, Yes]).is_corrupt());
        assert!(Findings([Yes, No, Yes, Yes]).is_corrupt());
        assert_eq!(Findings([Yes, No, No, Yes]).failed(), "tool_chain,on_task");
        assert_eq!(Findings([Yes, Yes, Yes, Yes]).failed(), "");
    }

    #[test]
    fn every_question_has_a_distinct_name_and_asks_for_the_arms() {
        let mut names: Vec<_> = QUESTIONS.iter().map(|q| q.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), QUESTIONS.len());
        for q in QUESTIONS {
            assert!(q.text.ends_with("Answer Yes or No."), "{}", q.name);
        }
    }
}
