//! What a tool definition gives the calibration phase.

use sha2::{Digest, Sha256};

/// The marker embedded in a trajectory at each projection (wide-Q capture)
/// point. Not a real token — the prefill path strips it and records each
/// occurrence's token offset so a projection fires there, reproducing the
/// decode's per-segment projection sequence.
pub const PROJECTION_MARKER: &str = "<|projection|>";

/// Conversation-metadata key tagging each calibration conversation with its
/// case's marker **at creation**, so it is findable by case on a later load —
/// finished or half-finished. *Done* is signalled separately by **archiving**
/// the conversation once its case seals; a tagged but un-archived conversation
/// is a case that was cut off, so the next load tombstones and regenerates it
/// (the archive is the atomic commit). Changing an example's text changes its
/// marker, so edited examples regenerate automatically.
pub const MARKER_KEY: &str = "calib";

/// One tool, as the calibration phase reads it.
pub trait Calibrated {
    /// The tool's member name in the live `tools` collection. A case's turns
    /// are tagged with it, and the belief scan credits a tagged turn to the
    /// member of that name.
    fn name(&self) -> &str;

    /// The section the model reads when choosing this tool — exactly the text
    /// the live collection shows for it.
    fn line(&self) -> String;

    /// The authored examples: a trajectory (a prompt, the dialect's assistant
    /// header, then think → call with [`PROJECTION_MARKER`]s), or a bare prompt
    /// for a tool whose trajectory is not recorded.
    fn examples(&self) -> &[String];

    /// Extra ways the tool is asked for — one plain question each, no reasoning
    /// and no answer.
    ///
    /// Routing happens on the question. A trajectory is mostly think block and
    /// call, so its question — the part a live probe resembles — is a few
    /// percent of the exemplar; these are entries that are *only* that part, so
    /// a tool can carry many more phrasings than it could afford as examples.
    fn questions(&self) -> &[String];

    /// Content marker for one calibration case: this tool paired with one of
    /// its trajectories or with its question group. Covers **everything that
    /// shapes the exemplar**: the projected [`Self::line`] and the case's text.
    ///
    /// The phase keys its resume on this, so editing a description, a
    /// parameter, an example or a question changes the marker and that case
    /// re-runs against the current text. Hashing the case alone would leave a
    /// reworded tool answering the belief scan with signatures captured against
    /// wording that no longer exists.
    fn calibration_marker(&self, example: &str) -> String {
        let mut h = Sha256::new();
        h.update(self.line().as_bytes());
        h.update([0u8]);
        h.update(example.as_bytes());
        format!("{}|{:x}", self.name(), h.finalize())
    }

    /// The examples that carry a recorded answer: those with an assistant half
    /// after `assistant_start`. Each is one calibration case.
    fn trajectories<'a>(&'a self, assistant_start: &'a str) -> impl Iterator<Item = &'a str> + 'a {
        self.examples()
            .iter()
            .map(String::as_str)
            .filter(move |e| e.contains(assistant_start))
    }

    /// The questions calibrated as this tool's one group: the authored
    /// [`Self::questions`], then every bare-prompt example as the plain
    /// question it is, its trailing `user_end` dropped.
    ///
    /// A bare prompt has no trajectory, and routing happens on the question, so
    /// this is the exemplar it can give without a decode on the load path.
    fn calibration_questions(&self, assistant_start: &str, user_end: &str) -> Vec<String> {
        let end = user_end.trim();
        let bare = self
            .examples()
            .iter()
            .filter(|e| !e.contains(assistant_start))
            .map(|e| {
                let p = e.trim();
                p.strip_suffix(end).unwrap_or(p).trim_end().to_string()
            });
        self.questions().iter().cloned().chain(bare).collect()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::Calibrated;

    pub(crate) struct Def {
        pub name: &'static str,
        pub line: &'static str,
        pub examples: Vec<String>,
        pub questions: Vec<String>,
    }

    impl Calibrated for Def {
        fn name(&self) -> &str {
            self.name
        }
        fn line(&self) -> String {
            self.line.to_string()
        }
        fn examples(&self) -> &[String] {
            &self.examples
        }
        fn questions(&self) -> &[String] {
            &self.questions
        }
    }

    const START: &str = "<|im_start|>assistant";
    const END: &str = "<|im_end|>\n";

    fn def() -> Def {
        Def {
            name: "clock",
            line: "clock() — what time it is",
            examples: vec![
                format!("what time is it?<|im_end|>\n{START}\n<think>the clock</think>"),
                "is it late?<|im_end|>\n".to_string(),
            ],
            questions: vec!["what's the hour".to_string()],
        }
    }

    #[test]
    fn a_marker_is_the_name_and_a_digest_of_the_line_and_the_case() {
        let d = def();
        assert_eq!(
            d.calibration_marker("ex"),
            format!("clock|{}", digest(d.line, "ex"))
        );
        let reworded = Def {
            line: "clock() — the time",
            ..def()
        };
        assert_ne!(
            reworded.calibration_marker("ex"),
            d.calibration_marker("ex")
        );
        assert_ne!(d.calibration_marker("other"), d.calibration_marker("ex"));
    }

    fn digest(line: &str, example: &str) -> String {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(line.as_bytes());
        h.update([0u8]);
        h.update(example.as_bytes());
        format!("{:x}", h.finalize())
    }

    #[test]
    fn only_an_example_with_an_answer_is_a_trajectory() {
        let d = def();
        let t: Vec<&str> = d.trajectories(START).collect();
        assert_eq!(t, vec![d.examples[0].as_str()]);
    }

    #[test]
    fn a_bare_prompt_joins_the_questions_without_its_end_marker() {
        assert_eq!(
            def().calibration_questions(START, END),
            vec!["what's the hour".to_string(), "is it late?".to_string()]
        );
    }
}
