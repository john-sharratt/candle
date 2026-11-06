//! The `Steps:` section a task's answer ends with
//! (`docs/zen_coding_challenge_fixes.md` Cause 3), written by the grammar.
//!
//! A task given steps says of each what became of it. Asked to in its brief,
//! a coder left the section out of 3 answers in 8, and each was sent back for
//! a whole extra round to write it. So the section is not the model's to
//! remember: when the answer ends, its end token is dropped and this tree
//! takes over ([`TriggerRegistry::with_tail`]), writing the heading and each
//! step's number and words, and holding each line to an outcome — `done`,
//! `not done` or `could not` — before the model says what shows it. The turn
//! then ends on the tree's own close.
//!
//! [`TriggerRegistry::with_tail`]: super::TriggerRegistry::with_tail

use super::spec::{NodeSpec, TreeSpec};
use super::terminator::Terminator;
use super::tree::FreeTextLimits;

/// The steps tree's label.
pub const STEPS_TREE_LABEL: &str = "answer-steps";

/// What a step's line may say became of it.
pub const OUTCOMES: [&str; 3] = ["done", "not done", "could not"];

/// The most characters of a step's words its line quotes.
const STEP_QUOTE: usize = 80;

/// The most tokens one step's line may run to before the grammar ends it.
const LINE_TOKENS: u32 = 200;

/// `step`'s words as its line quotes them: its first line, cut at
/// [`STEP_QUOTE`] characters with an ellipsis.
fn quote(step: &str) -> String {
    let line = step.lines().next().unwrap_or("").trim();
    match line.char_indices().nth(STEP_QUOTE) {
        Some((at, _)) => format!("{}…", line[..at].trim_end()),
        None => line.to_string(),
    }
}

/// The tree that writes the `Steps:` section for `steps` and then ends the
/// turn with `close_turn`: for each step, `N. <its words> — ` and a choice of
/// [`OUTCOMES`], then `: ` and the model's own line.
pub fn compile_steps_tail(steps: &[String], close_turn: &str) -> TreeSpec {
    let mut spec = TreeSpec::new(STEPS_TREE_LABEL);
    let end = spec.push(NodeSpec::End);
    let mut next = spec.push(NodeSpec::Static {
        text: close_turn.to_string(),
        next: end,
    });
    for (i, step) in steps.iter().enumerate().rev() {
        let line = spec.push(NodeSpec::FreeText {
            term: Terminator::Until {
                marker: "\n",
                tails: &[],
            },
            eos_ends: false,
            limits: FreeTextLimits::prose(LINE_TOKENS),
            close_token: None,
            suppress_close: false,
            next,
        });
        let said = spec.push(NodeSpec::Static {
            text: ": ".to_string(),
            next: line,
        });
        let outcome = spec.push(NodeSpec::Branch {
            arms: OUTCOMES.iter().map(|o| (o.to_string(), said)).collect(),
        });
        let heading = match i {
            0 => "\n\nSteps:\n",
            _ => "",
        };
        next = spec.push(NodeSpec::Static {
            text: format!("{heading}{}. {} — ", i + 1, quote(step)),
            next: outcome,
        });
    }
    spec.root = next;
    spec.bail = close_turn.to_string();
    spec
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::super::compile::compile;
    use super::super::driver::StepMask;
    use super::super::trigger::TriggerRegistry;
    use super::super::vocab::{TestVocab, TokenId, Vocab};
    use super::*;

    const IM_END: TokenId = 300;

    fn steps() -> Vec<String> {
        vec![
            "Implement it in edit_changes.rs.".to_string(),
            "Run the zend-tools tests, every one of them, on a quiet machine and with four \
             threads."
                .to_string(),
        ]
    }

    /// **A step's line quotes its first line, cut with an ellipsis** —
    /// exactly these strings.
    #[test]
    fn a_step_is_quoted_by_its_first_line() {
        assert_eq!(quote("Run it.\n  then more"), "Run it.");
        assert_eq!(
            quote(&steps()[1]),
            "Run the zend-tools tests, every one of them, on a quiet machine and with four…"
        );
    }

    /// **The section is written whole**: the heading and each step's number
    /// and words by the grammar, each outcome chosen from the three, each
    /// line the model's until its newline, and the turn closed — exactly this
    /// text, from a walk that answers `done` and then `could not`.
    #[test]
    fn the_steps_section_is_written_by_the_grammar() {
        let v = TestVocab::new()
            .with_special("<|im_end|>", IM_END)
            .with_end(IM_END);
        let tree = Arc::new(compile(&compile_steps_tail(&steps(), "<|im_end|>"), &v).unwrap());
        let mut d = TriggerRegistry::new().with_tail(tree).tail_driver().unwrap();
        let mut text = String::new();
        let mut said = ["done", "could not"].into_iter();
        let mut line = ["a1b2c3d\n", "no quiet machine\n"].into_iter();
        let mut writing: Vec<u8> = Vec::new();
        let mut ended = false;
        for _ in 0..1000 {
            match d.step() {
                StepMask::Prefill(run) => {
                    ended |= run.contains(&IM_END);
                    text.push_str(&String::from_utf8(v.decode(&run)).unwrap());
                }
                StepMask::Branch(set) => {
                    if writing.is_empty() {
                        writing = said.next().unwrap().as_bytes().to_vec();
                    }
                    let byte = writing.remove(0);
                    assert!(set.tokens().contains(&TokenId::from(byte)), "{text:?}");
                    text.push(byte as char);
                    d.accept(TokenId::from(byte), &[byte]);
                }
                StepMask::Free { .. } => {
                    if writing.is_empty() {
                        writing = line.next().unwrap().as_bytes().to_vec();
                    }
                    let byte = writing.remove(0);
                    text.push(byte as char);
                    d.accept(TokenId::from(byte), &[byte]);
                }
                StepMask::Done => break,
            }
        }
        assert!(ended, "the turn is closed: {text:?}");
        assert_eq!(
            text,
            "\n\nSteps:\n1. Implement it in edit_changes.rs. — done: a1b2c3d\n2. Run the \
             zend-tools tests, every one of them, on a quiet machine and with four… — could \
             not: no quiet machine\n"
        );
    }

    /// **An outcome is one of the three**: at a line's outcome the first
    /// characters offered are those of `done`, `not done` and `could not`.
    #[test]
    fn an_outcome_is_one_of_the_three() {
        let v = TestVocab::new().with_special("<|im_end|>", IM_END);
        let tree = Arc::new(compile(&compile_steps_tail(&steps(), "<|im_end|>"), &v).unwrap());
        let mut d = TriggerRegistry::new().with_tail(tree).tail_driver().unwrap();
        loop {
            match d.step() {
                StepMask::Prefill(_) => continue,
                StepMask::Branch(set) => {
                    let firsts: Vec<u8> = set
                        .tokens()
                        .iter()
                        .filter_map(|&t| v.token_bytes(t).first().copied())
                        .collect();
                    assert_eq!(firsts, b"cdn".to_vec());
                    return;
                }
                _ => panic!("the first choice is the first step's outcome"),
            }
        }
    }
}
