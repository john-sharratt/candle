//! One prompt a unit is asked in its calibration draft, and how many questions
//! it asks for (`docs/provenance_scoped_signatures.md` §2.5).
//!
//! A layer asks several, each for its own kind of question — what a file
//! defines, how it is used, what it is for — because one prompt asked for all
//! of them wrote mostly one kind: on BlackWorld (2026-11-15) a single prompt
//! asking for definitions, callers, dependents and purpose together wrote
//! nearly all "what does X do / why" questions and almost no "what calls X" or
//! "where is X defined" ones. Each prompt carries its own budget, so a kind of
//! question is asked for as often as the file has things to ask it about.

use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::count::QuestionCount;

/// One prompt as the schema declares it.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RawAsk {
    pub name: String,
    pub questions: QuestionCount,
    pub prompt: String,
}

/// One prompt of a calibration draft.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ask {
    /// What kind of question it asks for — `definitions`, `uses`, `purpose`.
    pub name: String,
    /// How many, by the unit's size.
    pub questions: QuestionCount,
    /// What it asks, with `<<n>>`, `<<subject>>`, and for a file subject
    /// `<<file>>`, `<<repo>>` and `<<path>>` filled in.
    pub prompt: String,
}

impl Ask {
    /// The ask a schema declared, or why it cannot be used.
    pub(super) fn of(raw: RawAsk) -> Result<Self, String> {
        if raw.name.trim().is_empty() {
            return Err("a prompt's `name` is empty".into());
        }
        if let Some(why) = raw.questions.invalid() {
            return Err(format!("prompt '{}': `questions`: {why}", raw.name));
        }
        if raw.prompt.trim().is_empty() {
            return Err(format!("prompt '{}': `prompt` is empty", raw.name));
        }
        Ok(Self {
            name: raw.name,
            questions: raw.questions,
            prompt: raw.prompt,
        })
    }

    /// The prompt asking the unit whose subject is `subject` — a file's
    /// workspace-relative path, its repository first, or a day — for
    /// `questions` questions.
    pub fn prompt_for(&self, subject: &str, questions: usize) -> String {
        let (repo, path) = crate::repo_path::split(subject);
        self.prompt
            .replace("<<n>>", &questions.to_string())
            .replace("<<subject>>", subject)
            .replace("<<file>>", subject)
            .replace("<<repo>>", repo)
            .replace("<<path>>", path)
    }

    /// Feed the ask into a calibration marker's digest: its name, its rule and
    /// its template, each closed by a zero byte so no two asks run together.
    pub fn digest_into(&self, h: &mut Sha256) {
        h.update(self.name.as_bytes());
        h.update([0u8]);
        self.questions.digest_into(h);
        h.update(self.prompt.as_bytes());
        h.update([0u8]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(name: &str, prompt: &str) -> RawAsk {
        RawAsk {
            name: name.into(),
            questions: QuestionCount {
                per_tokens: 300,
                min: 4,
                max: 32,
            },
            prompt: prompt.into(),
        }
    }

    #[test]
    fn a_prompt_names_its_count_and_subject() {
        let a = Ask::of(raw(
            "definitions",
            "<<n>> about `<<path>>` in <<repo>> (<<file>>)",
        ))
        .unwrap();
        assert_eq!(
            a.prompt_for("candle/docs/paper.md", 10),
            "10 about `docs/paper.md` in candle (candle/docs/paper.md)"
        );
    }

    #[test]
    fn an_ask_without_a_name_rule_or_prompt_is_refused() {
        assert_eq!(
            Ask::of(raw(" ", "x")).unwrap_err(),
            "a prompt's `name` is empty"
        );
        assert_eq!(
            Ask::of(raw("uses", " ")).unwrap_err(),
            "prompt 'uses': `prompt` is empty"
        );
        let mut bad = raw("uses", "x");
        bad.questions.min = 33;
        assert_eq!(
            Ask::of(bad).unwrap_err(),
            "prompt 'uses': `questions`: `min` must be 0..=`max` (32), not 33"
        );
    }

    /// The name, the rule and the template each move the digest.
    #[test]
    fn each_part_moves_the_digest() {
        let digest = |a: &Ask| {
            let mut h = Sha256::new();
            a.digest_into(&mut h);
            h.finalize().to_vec()
        };
        let a = Ask::of(raw("uses", "x")).unwrap();
        let base = digest(&a);
        let mut renamed = a.clone();
        renamed.name = "callers".into();
        assert_ne!(digest(&renamed), base);
        let mut more = a.clone();
        more.questions.per_tokens = 200;
        assert_ne!(digest(&more), base);
        let mut reworded = a.clone();
        reworded.prompt.push('!');
        assert_ne!(digest(&reworded), base);
        assert_eq!(digest(&a), base);
    }
}
