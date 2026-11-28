//! The acts, as the calibration phase reads them (`host::calibration`).
//!
//! A turn shows the acts its room offers that provenance ranks highest
//! ([`tools::install`]), and provenance ranks an act by the turns that carry
//! its tag. This module is where those turns come from: each act's authored
//! examples become trajectories — the situation, the reasoning that picks this
//! act over its neighbours, the call — and each act's situations and
//! [`questions`] become the plain ways it is asked for. The phase that writes
//! them is the host's, the one zend runs over its tool files.
//!
//! | Module | What it holds |
//! |---|---|
//! | [`trajectory`] | one authored example as the turn a character would have decoded |
//! | [`questions`] | the ways an act is asked for, beyond its examples' situations |

pub mod questions;
pub mod trajectory;

use candle_conversation::models::Dialect;
use host::calibration::{Calibrated, Frame};

use crate::engine::beyond;
use crate::engine::tools::{self, Tool};

/// One act, as the calibration phase reads it.
pub struct ActDef {
    /// The act's member name in the live `tools` collection.
    member: String,
    /// The entry a character reads for it.
    line: String,
    /// The authored examples, as trajectories.
    examples: Vec<String>,
    /// The situations it is right in, and the ways it is asked for.
    questions: Vec<String>,
}

impl Calibrated for ActDef {
    fn name(&self) -> &str {
        &self.member
    }

    fn line(&self) -> String {
        self.line.clone()
    }

    fn examples(&self) -> &[String] {
        &self.examples
    }

    fn questions(&self) -> &[String] {
        &self.questions
    }
}

/// Every installed act as a calibration definition.
///
/// An example is calibrated twice, and the two do different work: whole, as
/// the trajectory that ends in the call; and its situation alone, as a
/// question — the part a live turn's opening resembles, before anything has
/// been decoded.
pub fn defs(acts: &[&Tool], dialect: &Dialect) -> Vec<ActDef> {
    acts.iter()
        .map(|t| ActDef {
            member: tools::member_of(t),
            line: tools::entry(t),
            examples: t
                .examples
                .iter()
                .map(|e| trajectory::render(t.name, e, dialect))
                .collect(),
            questions: questions::of(t.name)
                .iter()
                .copied()
                .chain(
                    beyond::acts::of(t.name)
                        .into_iter()
                        .flat_map(|r| r.definition.questions.iter().map(String::as_str)),
                )
                .chain(t.examples.iter().map(|e| e.situation))
                .map(str::to_string)
                .collect(),
        })
        .collect()
}

/// The system prompt a calibration case's one act is shown inside: how a
/// character acts, in the words the mind's own prompt uses, around the act's
/// entry.
pub fn frame(dialect: &Dialect) -> Frame {
    Frame {
        open: format!(
            "{sys_start}HOW YOU ACT\n\nEverything you do, you do by calling an act. The act \
             below is the one open to you now.\n\n",
            sys_start = dialect.system_start,
        ),
        close: format!(
            "\n\nA call is your entire reply, and looks exactly like this:\n<tool_call>\n\
             {{\"name\": <act>, \"arguments\": <its arguments, as a JSON object>}}\n\
             </tool_call>{sys_end}",
            sys_end = dialect.system_end,
        ),
    }
}

#[cfg(test)]
mod tests {
    use candle_conversation::models::Dialect;
    use host::calibration::Calibrated;

    use super::{defs, frame};
    use crate::engine::beyond::acts;
    use crate::engine::terminal::acts::{CLUSTER_STATUS, TERMINAL_CONNECT};
    use crate::engine::tools::{self, CATALOG};

    #[test]
    fn an_act_is_named_by_its_member_and_read_as_its_entry() {
        let dialect = Dialect::chat_ml();
        let all = defs(&[&TERMINAL_CONNECT], &dialect);
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].name(), "tools/terminal_connect");
        assert_eq!(all[0].line(), tools::entry(&TERMINAL_CONNECT));
    }

    #[test]
    fn every_example_is_a_trajectory_and_its_situation_a_question() {
        let dialect = Dialect::chat_ml();
        let all = defs(&[&CLUSTER_STATUS], &dialect);
        let d = &all[0];
        assert_eq!(d.examples().len(), CLUSTER_STATUS.examples.len());
        assert_eq!(
            d.trajectories(dialect.assistant_start).count(),
            CLUSTER_STATUS.examples.len(),
            "every example carries its answer"
        );
        for e in CLUSTER_STATUS.examples {
            assert!(d.questions().iter().any(|q| q == e.situation));
        }
        assert!(d.questions().len() > CLUSTER_STATUS.examples.len());
    }

    /// **An act reached beyond the world is asked for as its tool is**: every
    /// question its definition carries is one it is ranked by, beside its
    /// examples' situations — calibrated on two situations alone, `weather`
    /// lost a forecast question to `web_fetch`.
    #[test]
    fn a_reached_act_is_asked_for_by_its_definitions_questions() {
        let dialect = Dialect::chat_ml();
        let weather = acts::of("weather").unwrap();
        let d = &defs(&[&weather.tool], &dialect)[0];
        for q in &weather.definition.questions {
            assert!(d.questions().contains(q), "{q}");
        }
        assert_eq!(
            d.questions().len(),
            weather.definition.questions.len() + weather.tool.examples.len()
        );
    }

    #[test]
    fn every_act_in_the_catalog_has_something_to_be_ranked_by() {
        let dialect = Dialect::chat_ml();
        let acts: Vec<_> = CATALOG.iter().collect();
        for d in defs(&acts, &dialect) {
            assert!(
                !d.questions().is_empty() && !d.examples().is_empty(),
                "{} has nothing to calibrate",
                d.name()
            );
        }
    }

    #[test]
    fn the_frame_opens_and_closes_the_system_block() {
        let dialect = Dialect::chat_ml();
        let f = frame(&dialect);
        assert!(f.open.starts_with(dialect.system_start));
        assert!(f.close.ends_with(dialect.system_end));
    }
}
