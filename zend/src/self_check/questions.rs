//! The questions a conversation is asked about itself, and what its answers
//! mean.
//!
//! Every question is phrased so that **yes is healthy**: a corrupt conversation
//! is one that answers no to any of them.
//!
//! Each question carries the conversation's record ([`super::transcript`]) and
//! asks about it, so what is judged is the record and nothing else in context.
//!
//! What a conversation is asked depends on what it is. An ingest conversation
//! was given one subject — a file to read, or a folder to summarise — and its
//! questions name that subject. The damage they look for is the damage the
//! ingest actually takes: a final reply about some other file (one of the
//! priming chain's READMEs, most often) or confused about which file it was
//! given, reached by reading outside the subject or by a tool result that
//! answered no call.
//!
//! A dialogue has no assigned subject, so it is asked whether each reply
//! follows from the message before it.

/// What a checked conversation was given to do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Subject {
    /// Read one file and summarise it.
    File { repo: String, path: String },
    /// Summarise one folder from its listing. `path` is empty for a
    /// repository's root.
    Folder { repo: String, path: String },
    /// A conversation with a person.
    Dialogue,
}

impl Subject {
    /// The subject of an ingest conversation from its stored key, which is
    /// `<repo>/<path within the repo>`.
    pub fn file(key: &str) -> Self {
        let (repo, path) = split_key(key);
        Subject::File { repo, path }
    }

    /// See [`Subject::file`]; a folder key may end in `/`.
    pub fn folder(key: &str) -> Self {
        let (repo, path) = split_key(key);
        Subject::Folder { repo, path }
    }

    /// How a question names the subject.
    fn phrase(&self) -> String {
        match self {
            Subject::File { repo, path } => {
                format!("the file `{path}` in the `{repo}` repository")
            }
            Subject::Folder { repo, path } if path.is_empty() => {
                format!("the root folder of the `{repo}` repository")
            }
            Subject::Folder { repo, path } => {
                format!("the folder `{path}` in the `{repo}` repository")
            }
            Subject::Dialogue => String::new(),
        }
    }

    /// The questions this conversation is asked about `record`, in report
    /// order.
    pub fn questions(&self, record: &str) -> Vec<Question> {
        let it = self.phrase();
        let ask = |name: &'static str, preamble: &str, question: String| Question {
            name,
            text: format!("{preamble}\n\n{RECORD_OPEN}\n{record}{RECORD_CLOSE}\n\n{question}"),
        };
        let tool_chain = "Does every TOOL RESULT in the record answer a tool call made in the \
                          ASSISTANT reply just before it? A TOOL RESULT that follows a reply \
                          with no tool call is a No. Answer Yes or No."
            .to_string();
        match self {
            Subject::File { .. } => {
                let preamble = format!(
                    "Below is the record of a conversation in which an assistant was asked \
                     to read {it} and summarise it."
                );
                vec![
                    ask(
                        "summary_of_subject",
                        &preamble,
                        format!(
                            "Is the assistant's final reply in the record a summary of {it} \
                             itself — not of some other file, and not a remark about the tool \
                             results? Answer Yes or No."
                        ),
                    ),
                    ask(
                        "stayed_on_subject",
                        &preamble,
                        format!(
                            "Is every file_read and file_list call in the record for {it}, \
                             and for nothing else? Answer Yes or No."
                        ),
                    ),
                    ask(
                        "knew_its_subject",
                        &preamble,
                        format!(
                            "Does the assistant's final reply treat {it} as the file it was \
                             asked to read — without saying it asked for, expected or was \
                             given some other file, such as a README? Answer Yes or No."
                        ),
                    ),
                    ask("tool_chain", &preamble, tool_chain),
                ]
            }
            Subject::Folder { .. } => {
                let preamble = format!(
                    "Below is the record of a conversation in which an assistant was asked \
                     to summarise {it} in one sentence."
                );
                vec![
                    ask(
                        "summary_of_subject",
                        &preamble,
                        format!(
                            "Is the assistant's final reply in the record a summary of {it} \
                             itself — not of some other folder or file, and not a remark about \
                             the tool results? Answer Yes or No."
                        ),
                    ),
                    ask("tool_chain", &preamble, tool_chain),
                ]
            }
            Subject::Dialogue => vec![ask(
                "replies_follow",
                "Below is the record of a conversation between a person and an assistant.",
                "Does each ASSISTANT reply in the record respond to the message just before \
                 it? Changes of subject made by the person are normal. Answer Yes or No."
                    .to_string(),
            )],
        }
    }
}

/// The lines a question's record sits between.
const RECORD_OPEN: &str = "--- record ---";
const RECORD_CLOSE: &str = "--- end of record ---";

/// `<repo>/<path>` split at its first `/`; a trailing `/` on the path is
/// dropped.
fn split_key(key: &str) -> (String, String) {
    let (repo, path) = key.split_once('/').unwrap_or((key, ""));
    (repo.to_string(), path.trim_end_matches('/').to_string())
}

/// One self-check question.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Question {
    /// A short name for reports.
    pub name: &'static str,
    /// The question as the model is asked it.
    pub text: String,
}

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
    /// opening that closes the reasoning block). The grammar forces the first
    /// token to be [`YES`] or [`NO`] and then hands decoding back to the model,
    /// so the arm is the reply's first word and whatever follows it ("Yes — each
    /// reply …", "No.") is the model explaining itself, not part of the verdict.
    /// A reply that does not open on a whole arm did not run under the grammar,
    /// and is reported rather than read as either.
    pub fn parse(reply: &str, prefill: &str) -> Result<Self, String> {
        let answer = reply.strip_prefix(prefill).unwrap_or(reply).trim_start();
        let opens_on = |arm: &str| {
            answer
                .strip_prefix(arm)
                .is_some_and(|rest| !rest.starts_with(|c: char| c.is_alphanumeric()))
        };
        if opens_on(YES) {
            Ok(Answer::Yes)
        } else if opens_on(NO) {
            Ok(Answer::No)
        } else {
            Err(format!(
                "the answer {answer:?} opens on neither {YES:?} nor {NO:?}"
            ))
        }
    }
}

/// The reply after `prefill`, on one line, for a report.
pub fn reply_text(reply: &str, prefill: &str) -> String {
    reply
        .strip_prefix(prefill)
        .unwrap_or(reply)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// One question's answer, and the reply it came in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verdict {
    pub question: &'static str,
    pub answer: Answer,
    /// The whole reply on one line: the arm, then the model's reason for it.
    pub reply: String,
}

/// One conversation's verdicts, in report order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Findings(pub Vec<Verdict>);

impl Findings {
    /// Corrupt when any question was answered no.
    pub fn is_corrupt(&self) -> bool {
        self.no().next().is_some()
    }

    /// The names of the questions answered no, comma-separated; empty when
    /// none was.
    pub fn failed(&self) -> String {
        self.no().map(|v| v.question).collect::<Vec<_>>().join(",")
    }

    /// Each question answered no with the model's reply, `name: reply`,
    /// separated by ` | `.
    pub fn reasons(&self) -> String {
        self.no()
            .map(|v| format!("{}: {}", v.question, v.reply))
            .collect::<Vec<_>>()
            .join(" | ")
    }

    fn no(&self) -> impl Iterator<Item = &Verdict> {
        self.0.iter().filter(|v| v.answer == Answer::No)
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

    /// The replies the live substrate gave: the arm, then the model's own
    /// continuation once the grammar handed decoding back.
    #[test]
    fn the_verdict_is_the_opening_arm_whatever_follows() {
        assert_eq!(
            Answer::parse("<think>\n\n</think>\n\nYes — each assistant", PREFILL),
            Ok(Answer::Yes)
        );
        assert_eq!(Answer::parse("Yes.", PREFILL), Ok(Answer::Yes));
        assert_eq!(Answer::parse("Yes.\n<tool_call>", PREFILL), Ok(Answer::Yes));
        assert_eq!(Answer::parse("No, there is", PREFILL), Ok(Answer::No));
        assert_eq!(Answer::parse("No. There is", PREFILL), Ok(Answer::No));
    }

    #[test]
    fn anything_else_is_not_an_answer() {
        assert!(Answer::parse("Maybe", PREFILL).is_err());
        assert!(Answer::parse("yes", PREFILL).is_err());
        assert!(Answer::parse("", PREFILL).is_err());
        assert!(Answer::parse("Yesterday", PREFILL).is_err());
        assert!(Answer::parse("Nobody", PREFILL).is_err());
    }

    fn verdict(question: &'static str, answer: Answer, reply: &str) -> Verdict {
        Verdict {
            question,
            answer,
            reply: reply.to_string(),
        }
    }

    #[test]
    fn one_no_makes_a_conversation_corrupt() {
        use Answer::{No, Yes};
        let all_yes = Findings(vec![verdict("a", Yes, "Yes."), verdict("b", Yes, "Yes.")]);
        assert!(!all_yes.is_corrupt());
        assert_eq!(all_yes.failed(), "");
        assert_eq!(all_yes.reasons(), "");
        let two_no = Findings(vec![
            verdict("a", Yes, "Yes."),
            verdict("b", No, "No — it read README.md."),
            verdict("c", No, "No."),
        ]);
        assert!(two_no.is_corrupt());
        assert_eq!(two_no.failed(), "b,c");
        assert_eq!(two_no.reasons(), "b: No — it read README.md. | c: No.");
    }

    #[test]
    fn a_reply_is_reported_on_one_line_without_its_prefill() {
        assert_eq!(
            reply_text(
                "<think>\n\n</think>\n\nNo — the final\nreply  is about X.",
                PREFILL
            ),
            "No — the final reply is about X."
        );
    }

    #[test]
    fn a_stored_key_splits_into_repo_and_path() {
        assert_eq!(
            Subject::file("candle/candle-conversation/src/banned_rows.rs"),
            Subject::File {
                repo: "candle".into(),
                path: "candle-conversation/src/banned_rows.rs".into(),
            }
        );
        assert_eq!(
            Subject::folder("candle/candle-conversation/src/"),
            Subject::Folder {
                repo: "candle".into(),
                path: "candle-conversation/src".into(),
            }
        );
        assert_eq!(
            Subject::folder("battle-cities/"),
            Subject::Folder {
                repo: "battle-cities".into(),
                path: String::new(),
            }
        );
    }

    #[test]
    fn an_ingest_question_names_its_subject() {
        let file = Subject::file("candle/zend/src/main.rs").questions("REC");
        assert_eq!(
            file.iter().map(|q| q.name).collect::<Vec<_>>(),
            [
                "summary_of_subject",
                "stayed_on_subject",
                "knew_its_subject",
                "tool_chain"
            ]
        );
        for q in &file {
            assert!(
                q.text
                    .contains("the file `zend/src/main.rs` in the `candle` repository"),
                "{}",
                q.name
            );
        }
        let root = Subject::folder("candle/").questions("REC");
        assert!(root[0]
            .text
            .contains("the root folder of the `candle` repository"));
        let folder = Subject::folder("candle/docs/").questions("REC");
        assert!(folder[0]
            .text
            .contains("the folder `docs` in the `candle` repository"));
    }

    #[test]
    fn a_question_carries_the_record_between_its_preamble_and_its_ask() {
        let qs = Subject::folder("r/src/").questions("[turn 1]\nUSER: hi\nASSISTANT: ok\n");
        assert_eq!(
            qs[0].text,
            "Below is the record of a conversation in which an assistant was asked to \
             summarise the folder `src` in the `r` repository in one sentence.\n\n\
             --- record ---\n\
             [turn 1]\nUSER: hi\nASSISTANT: ok\n\
             --- end of record ---\n\n\
             Is the assistant's final reply in the record a summary of the folder `src` in \
             the `r` repository itself — not of some other folder or file, and not a remark \
             about the tool results? Answer Yes or No."
        );
    }

    #[test]
    fn every_question_asks_for_the_arms_and_names_are_distinct() {
        for subject in [
            Subject::file("r/a.rs"),
            Subject::folder("r/src/"),
            Subject::Dialogue,
        ] {
            let qs = subject.questions("REC");
            let mut names: Vec<_> = qs.iter().map(|q| q.name).collect();
            names.sort_unstable();
            names.dedup();
            assert_eq!(names.len(), qs.len());
            for q in qs {
                assert!(q.text.ends_with("Answer Yes or No."), "{}", q.name);
            }
        }
    }
}
