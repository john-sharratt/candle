//! Which prompts a unit is asked, by what its subject is: a calibration
//! profile is a set of prompts and the paths it is for, matched by regex.
//!
//! A source file is asked what it defines and who calls it, a manifest what it
//! depends on and which feature turns what on, a test what it verifies, a
//! kernel what it computes and which wrapper launches it, a document what it
//! states and decides. One set of prompts for all of them asked a document
//! about every code symbol it mentioned — `ARCHITECTURE.md` wrote 32 questions
//! about symbols other files define, each a vote against the file that defines
//! it — and a small, self-contained source file for callers it does not have
//! (BlackWorld, 2026-11-15). The first profile whose regex matches the subject
//! is the unit's; the last carries none and takes every other.

use regex::Regex;
use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::ask::{Ask, RawAsk};

/// One profile as the schema declares it.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RawProfile {
    pub name: String,
    /// Regexes over the unit's subject — a file's path, its repository
    /// first. Absent for the default profile.
    #[serde(default)]
    pub files: Vec<String>,
    pub prompts: Vec<RawAsk>,
}

/// A calibration profile: its name, the subjects it is for, and its prompts.
#[derive(Debug, Clone)]
pub struct Profile {
    pub name: String,
    /// The regexes as declared — what the marker digests.
    pub patterns: Vec<String>,
    /// Them compiled as one alternation; `None` for the default profile.
    matcher: Option<Regex>,
    pub asks: Vec<Ask>,
}

impl PartialEq for Profile {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.patterns == other.patterns && self.asks == other.asks
    }
}

impl Eq for Profile {}

impl Profile {
    /// The profile a schema declared, or why it cannot be used.
    pub(super) fn of(raw: RawProfile) -> Result<Self, String> {
        let name = raw.name.trim().to_string();
        if name.is_empty() {
            return Err("a profile's `name` is empty".into());
        }
        if raw.prompts.is_empty() {
            return Err(format!("profile '{name}': `prompts` is empty"));
        }
        let asks = raw
            .prompts
            .into_iter()
            .map(Ask::of)
            .collect::<Result<Vec<Ask>, String>>()
            .map_err(|why| format!("profile '{name}': {why}"))?;
        let mut seen: Vec<&str> = Vec::new();
        for a in &asks {
            if seen.contains(&a.name.as_str()) {
                return Err(format!(
                    "profile '{name}': prompt '{}' is declared twice",
                    a.name
                ));
            }
            seen.push(&a.name);
        }
        let matcher = if raw.files.is_empty() {
            None
        } else {
            let alternation = raw
                .files
                .iter()
                .map(|p| format!("(?:{p})"))
                .collect::<Vec<_>>()
                .join("|");
            Some(Regex::new(&alternation).map_err(|e| format!("profile '{name}': `files`: {e}"))?)
        };
        Ok(Self {
            name,
            patterns: raw.files,
            matcher,
            asks,
        })
    }

    /// Whether this profile is the default — the one with no `files`.
    pub fn is_default(&self) -> bool {
        self.matcher.is_none()
    }

    /// Whether a unit whose subject is `subject` takes this profile: one of
    /// its regexes matches, or it is the default.
    pub fn takes(&self, subject: &str) -> bool {
        self.matcher.as_ref().is_none_or(|m| m.is_match(subject))
    }

    /// The most questions this profile can ask one unit, every prompt
    /// together.
    pub fn most(&self) -> usize {
        self.asks.iter().map(|a| a.questions.max).sum()
    }

    /// Feed the profile into a calibration marker's digest: its name, then each
    /// prompt in order. Its regexes are not in it — which files a profile takes
    /// is not what they are asked, and narrowing it must not recalibrate the
    /// files it still takes.
    pub fn digest_into(&self, h: &mut Sha256) {
        h.update(self.name.as_bytes());
        h.update([0u8]);
        for ask in &self.asks {
            ask.digest_into(h);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::ask::AskScope;
    use super::super::count::QuestionCount;
    use super::*;

    fn raw(name: &str, files: &[&str]) -> RawProfile {
        RawProfile {
            name: name.into(),
            files: files.iter().map(|f| f.to_string()).collect(),
            prompts: vec![RawAsk {
                name: "definitions".into(),
                questions: QuestionCount {
                    per_tokens: 300,
                    min: 4,
                    max: 28,
                },
                per: AskScope::Section,
                prompt: "ask <<n>>".into(),
            }],
        }
    }

    /// **A profile takes the subjects one of its regexes matches**, the
    /// default every subject.
    #[test]
    fn a_profile_takes_the_subjects_its_regexes_match() {
        let manifest = Profile::of(raw(
            "manifest",
            &[r"(^|/)Cargo\.toml$", r"(^|/)package\.json$"],
        ))
        .unwrap();
        assert!(manifest.takes("candle/zend/Cargo.toml"));
        assert!(manifest.takes("yenbot/web/package.json"));
        assert!(!manifest.takes("candle/zend/src/cargo_toml.rs"));
        assert!(!manifest.is_default());
        let default = Profile::of(raw("default", &[])).unwrap();
        assert!(default.is_default());
        assert!(default.takes("anything/at/all"));
    }

    #[test]
    fn a_bad_profile_is_refused_and_says_why() {
        assert_eq!(
            Profile::of(raw(" ", &[])).unwrap_err(),
            "a profile's `name` is empty"
        );
        let bad = Profile::of(raw("code", &["("])).unwrap_err();
        assert!(bad.starts_with("profile 'code': `files`: "), "{bad}");
        let mut empty = raw("code", &[]);
        empty.prompts.clear();
        assert_eq!(
            Profile::of(empty).unwrap_err(),
            "profile 'code': `prompts` is empty"
        );
        let mut twice = raw("code", &[]);
        twice.prompts.push(RawAsk {
            name: "definitions".into(),
            questions: QuestionCount {
                per_tokens: 300,
                min: 1,
                max: 2,
            },
            per: AskScope::Section,
            prompt: "x".into(),
        });
        assert_eq!(
            Profile::of(twice).unwrap_err(),
            "profile 'code': prompt 'definitions' is declared twice"
        );
    }

    /// The name and the prompts move the digest; the regexes do not.
    #[test]
    fn the_digest_follows_the_name_and_prompts_not_the_regexes() {
        let digest = |p: &Profile| {
            let mut h = Sha256::new();
            p.digest_into(&mut h);
            h.finalize().to_vec()
        };
        let base = Profile::of(raw("code", &[r"\.rs$"])).unwrap();
        let wider = Profile::of(raw("code", &[r"\.rs$", r"\.py$"])).unwrap();
        assert_eq!(digest(&base), digest(&wider));
        let renamed = Profile::of(raw("source", &[r"\.rs$"])).unwrap();
        assert_ne!(digest(&base), digest(&renamed));
        let mut reworded = base.clone();
        reworded.asks[0].prompt.push('!');
        assert_ne!(digest(&base), digest(&reworded));
    }
}
