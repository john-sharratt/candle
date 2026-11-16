//! One unit of calibration work, waiting and in flight.

use std::borrow::Cow;

use candle_conversation::{ConversationError, ProjectionEvent, Sequence, TurnHandle, TurnResponse};

use super::exemplar::Calibrated;

/// One unit of calibration work — one conversation, one submission, one resume
/// marker.
///
/// The two kinds differ in how many exemplars a submission produces, which is
/// the whole point of the split: a trajectory is one turn, a question group is
/// one prefill carved into as many turns as it has questions.
pub enum Case<'a> {
    /// One authored trajectory from the tool's definition: prompt +
    /// `<think>…</think>` + the call. Prefilled and sealed on its own.
    Trajectory {
        tool: &'a str,
        /// Borrowed from the definition, which outlives the list — the
        /// authored trajectories are the bulk of the corpus by bytes and
        /// copying them all would be pure waste.
        text: Cow<'a, str>,
        marker: String,
    },
    /// EVERY question exemplar of one tool, prefilled in a single forward and
    /// carved into one turn each.
    Questions {
        tool: &'a str,
        questions: Vec<String>,
        marker: String,
    },
}

impl Case<'_> {
    pub fn tool(&self) -> &str {
        match self {
            Case::Trajectory { tool, .. } | Case::Questions { tool, .. } => tool,
        }
    }

    pub fn marker(&self) -> &str {
        match self {
            Case::Trajectory { marker, .. } | Case::Questions { marker, .. } => marker,
        }
    }

    /// How many provenance exemplars this submission produces — one per turn it
    /// seals. Progress is reported in exemplars rather than submissions so a
    /// question group is not one step of a bar that then jumps.
    pub fn exemplars(&self) -> usize {
        match self {
            Case::Trajectory { .. } => 1,
            Case::Questions { questions, .. } => questions.len(),
        }
    }
}

/// Every case the catalog `defs` asks for: one per trajectory, and one question
/// group per tool that has any question.
///
/// Two kinds of case:
///
///  • a trajectory — the full authored example: prompt + the dialect's
///    assistant header + think→call with projection markers. One submission
///    each: they are long and irregular, and each is staged at its own
///    projection points, which a group's cases do not carry.
///  • a tool's questions — plain questions with an EMPTY assistant turn.
///    Routing happens on the question, so these are exemplars that are only
///    the part a live probe resembles. ALL of a tool's questions go in as ONE
///    submission: they are short, uniform, and all carry the same tool tag and
///    the same pinned selection. Per TOOL rather than per catalog, because a
///    group shares one projection and one selection, and because the whole
///    question corpus in one sequence would consume the entire KV budget.
///
/// Each case carries its content marker, computed once here from the tool's
/// whole projected definition plus the case's text, so editing either
/// regenerates exactly that case. A group has one marker over all of its
/// questions, joined with a separator that cannot occur inside a question, so
/// two different splits cannot hash alike.
pub fn cases<'a, T: Calibrated>(
    defs: &'a [T],
    assistant_start: &'a str,
    user_end: &str,
) -> Vec<Case<'a>> {
    defs.iter()
        .flat_map(|d| {
            let full = d
                .trajectories(assistant_start)
                .map(move |e| Case::Trajectory {
                    tool: d.name(),
                    text: Cow::Borrowed(e),
                    marker: d.calibration_marker(e),
                });
            let questions = d.calibration_questions(assistant_start, user_end);
            let asked = (!questions.is_empty()).then(|| Case::Questions {
                tool: d.name(),
                marker: d.calibration_marker(&questions.join("\u{0}")),
                questions,
            });
            full.chain(asked)
        })
        .collect()
}

/// A calibration submission in flight, awaiting its `Done`.
pub struct InFlight<'a> {
    pub conv: Sequence,
    pub handle: TurnHandle,
    pub tool: &'a str,
    /// Mid-decode reprojection events, accumulated across sweeps — they arrive
    /// interleaved with tokens, so a sweep that finds no terminal event still
    /// has to keep what it drained.
    pub events: Vec<ProjectionEvent>,
    /// A terminal event the idle wait absorbed: the wait parks on every
    /// in-flight handle and consumes exactly one event from whichever speaks
    /// first, so a `Done`/`Error` it takes has to be carried to the retire
    /// sweep rather than lost, and so does the error a failed turn carries.
    pub finished: Option<Result<TurnResponse, ConversationError>>,
    /// A prefilled trajectory is complete by construction (it is the authored
    /// text) and its `Done` carries no decoded text, so it must NOT be gated on
    /// `resp.text` the way a live decode is.
    pub is_prefill: bool,
    /// How many exemplars this submission produces — one for a trajectory, one
    /// per question for a group.
    pub exemplars: usize,
}

#[cfg(test)]
mod tests {
    use super::{cases, Case};
    use crate::calibration::exemplar::tests::Def;
    use crate::calibration::Calibrated;

    const START: &str = "<|im_start|>assistant";
    const END: &str = "<|im_end|>\n";

    #[test]
    fn a_tool_gives_one_case_a_trajectory_and_one_group_of_its_questions() {
        let traj = format!("what time is it?<|im_end|>\n{START}\n<think>the clock</think>");
        let defs = vec![
            Def {
                name: "clock",
                line: "clock()",
                examples: vec![traj.clone(), "is it late?<|im_end|>\n".to_string()],
                questions: vec!["what's the hour".to_string()],
            },
            Def {
                name: "silent",
                line: "silent()",
                examples: Vec::new(),
                questions: Vec::new(),
            },
        ];
        let all = cases(&defs, START, END);
        assert_eq!(all.len(), 2);
        match &all[0] {
            Case::Trajectory { tool, text, marker } => {
                assert_eq!(*tool, "clock");
                assert_eq!(text.as_ref(), traj);
                assert_eq!(marker, &defs[0].calibration_marker(&traj));
            }
            Case::Questions { .. } => panic!("the trajectory comes first"),
        }
        match &all[1] {
            Case::Questions {
                tool,
                questions,
                marker,
            } => {
                assert_eq!(*tool, "clock");
                assert_eq!(questions, &["what's the hour", "is it late?"]);
                assert_eq!(
                    marker,
                    &defs[0].calibration_marker("what's the hour\u{0}is it late?")
                );
            }
            Case::Trajectory { .. } => panic!("the group comes last"),
        }
        assert_eq!(all[0].exemplars(), 1);
        assert_eq!(all[1].exemplars(), 2);
    }
}
