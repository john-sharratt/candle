//! A layer's `calibrate:` block — the profiles its conversations are asked by,
//! each a set of prompts for the subjects its path regexes match, how many
//! questions each prompt asks for, and how much of each question's thinking is
//! kept — read from the projection schema it is declared in.
//!
//! ```yaml
//! - name: code_reading
//!   calibrate:
//!     think: 64
//!     profiles:
//!       - name: manifest
//!         files: ['(^|/)Cargo\.toml$']
//!         prompts:
//!           - name: dependencies
//!             questions: { per_tokens: 120, min: 3, max: 16 }
//!             prompt: |
//!               Write <<n>> questions about what `<<path>>` declares …
//!       - name: default
//!         prompts:
//!           - name: definitions
//!             questions: { per_tokens: 300, min: 4, max: 32 }
//!             prompt: |
//!               Write <<n>> questions about what `<<path>>` defines …
//!     guidance:
//!       questions: { per_tokens: 250, min: 8, max: 64 }
//!       prompt: |
//!         Write <<n>> questions a worker might ask itself …
//! ```
//!
//! Nothing here names a layer: any layer whose conversations are units — a
//! file's reading, an analysis of it, a day — may declare one, and each of its
//! units is calibrated the same way ([`super::run`]).

use std::slice;

use candle_conversation::projection::{Draws, LayerSchema};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::ask::{Ask, RawAsk};
use super::count::QuestionCount;
use super::profile::{Profile, RawProfile};

/// The most questions one unit may be asked for, every prompt together. Each
/// is one turn of one prefill, and the calibration's K/V is held lossless
/// until it seals.
pub const MAX_QUESTIONS: usize = 64;

/// The most thinking tokens kept per question: a live probe's trailing
/// window is at most this long (`REPROJECT_MAX_PROBE_TOKENS`).
pub const MAX_THINK: usize = 256;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Raw {
    think: usize,
    profiles: Vec<RawProfile>,
    /// What a unit a drawing group of the layer draws is asked instead.
    #[serde(default)]
    guidance: Option<RawGuidance>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawGuidance {
    questions: QuestionCount,
    prompt: String,
}

/// How a layer's conversations are calibrated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrateSpec {
    /// The layer's name.
    pub layer: String,
    /// The decode budget of each question's thinking ([`super::thinking`]).
    pub think: usize,
    /// What a unit is asked, by its subject: the first profile that takes it,
    /// the last the default ([`super::profile`]).
    pub profiles: Vec<Profile>,
    /// What a unit the layer's drawing groups draw is asked instead, and the
    /// paths they draw (`docs/guidance_layer.md` §2.2). A guide is wanted
    /// at a point in a worker's own reasoning, so its seed questions are the
    /// ones a worker asks itself mid-task, not a user's.
    pub guidance: Option<GuidancePrompt>,
}

/// The prompt for the units a layer's drawing groups draw.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuidancePrompt {
    pub ask: Ask,
    /// What every drawing group of the layer draws.
    pub draws: Vec<Draws>,
}

impl CalibrateSpec {
    /// The layer's spec, `None` when it declares no `calibrate:` block.
    pub fn of(layer: &LayerSchema) -> anyhow::Result<Option<Self>> {
        let Some(value) = &layer.calibrate else {
            return Ok(None);
        };
        let name = &layer.name;
        let raw: Raw = serde_yaml::from_value(value.clone())
            .map_err(|e| anyhow::anyhow!("layer '{name}': its `calibrate:` block: {e}"))?;
        if raw.think == 0 || raw.think > MAX_THINK {
            anyhow::bail!(
                "layer '{name}': `calibrate.think` must be 1..={MAX_THINK}, not {}",
                raw.think
            );
        }
        let profiles = raw
            .profiles
            .into_iter()
            .map(Profile::of)
            .collect::<Result<Vec<Profile>, String>>()
            .map_err(|why| anyhow::anyhow!("layer '{name}': `calibrate.profiles`: {why}"))?;
        let Some(last) = profiles.last() else {
            anyhow::bail!("layer '{name}': `calibrate.profiles` is empty");
        };
        if !last.is_default() {
            anyhow::bail!(
                "layer '{name}': `calibrate.profiles` ends with '{}', which has `files`; the last \
                 profile is the default and has none",
                last.name
            );
        }
        if let Some(early) = profiles[..profiles.len() - 1]
            .iter()
            .find(|p| p.is_default())
        {
            anyhow::bail!(
                "layer '{name}': `calibrate.profiles`: '{}' has no `files` but is not last — every \
                 profile after it would be unreachable",
                early.name
            );
        }
        let mut names: Vec<&str> = Vec::new();
        for p in &profiles {
            if names.contains(&p.name.as_str()) {
                anyhow::bail!(
                    "layer '{name}': `calibrate.profiles`: '{}' is declared twice",
                    p.name
                );
            }
            names.push(&p.name);
            within_the_cap(name, &format!("profiles: '{}'", p.name), p.most())?;
        }
        let draws: Vec<Draws> = layer
            .groups
            .iter()
            .filter_map(|g| g.draws.clone())
            .collect();
        let guidance = match raw.guidance {
            None => None,
            Some(_) if draws.is_empty() => anyhow::bail!(
                "layer '{name}': `calibrate.guidance` is declared but no group of the layer \
                 draws anything"
            ),
            Some(g) => {
                let ask = Ask::of(RawAsk {
                    name: "guidance".into(),
                    questions: g.questions,
                    prompt: g.prompt,
                })
                .map_err(|why| anyhow::anyhow!("layer '{name}': `calibrate.guidance`: {why}"))?;
                within_the_cap(name, "guidance", ask.questions.max)?;
                Some(GuidancePrompt { ask, draws })
            }
        };
        Ok(Some(Self {
            layer: name.clone(),
            think: raw.think,
            profiles,
            guidance,
        }))
    }

    /// The guidance prompt, when `subject` is a unit a drawing group draws.
    fn guidance_for(&self, subject: &str) -> Option<&GuidancePrompt> {
        self.guidance
            .as_ref()
            .filter(|g| g.draws.iter().any(|d| d.takes(subject)))
    }

    /// The profile `subject` is asked by: the first that takes it.
    pub fn profile_for(&self, subject: &str) -> &Profile {
        self.profiles
            .iter()
            .find(|p| p.takes(subject))
            .expect("the last profile is the default, and takes every subject")
    }

    /// What `subject` is asked, in order: the guidance prompt for a unit a
    /// drawing group draws, its profile's prompts for every other.
    pub fn asks_for(&self, subject: &str) -> &[Ask] {
        match self.guidance_for(subject) {
            Some(g) => slice::from_ref(&g.ask),
            None => &self.profile_for(subject).asks,
        }
    }

    /// The calibration marker of the unit committed under `content_key`, whose
    /// subject is `subject`: the layer, a digest of what it is asked, and the
    /// unit's own key — so a changed prompt, its question-count rule, the
    /// profile it is asked by, the thinking budget or the unit's version
    /// calibrates again, and a change to one profile or the guidance prompt
    /// only the units asked it. The unit's own counts are not in it: they
    /// follow from the rules and the content the key names.
    pub fn marker(&self, content_key: &str, subject: &str) -> String {
        let mut h = Sha256::new();
        h.update(self.think.to_le_bytes());
        match self.guidance_for(subject) {
            Some(g) => {
                h.update(b"guidance\0");
                g.ask.digest_into(&mut h);
            }
            None => self.profile_for(subject).digest_into(&mut h),
        }
        let digest: String = h.finalize()[..8]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        format!("{}|{digest}|{content_key}", self.layer)
    }

    /// The content key of the unit `marker` was written for — the part a
    /// changed prompt, count or budget leaves alone.
    pub fn unit_of(marker: &str) -> Option<&str> {
        marker.splitn(3, '|').nth(2)
    }

    /// Whether `marker` is one of this layer's.
    pub fn owns(&self, marker: &str) -> bool {
        marker
            .strip_prefix(self.layer.as_str())
            .is_some_and(|rest| rest.starts_with('|'))
    }
}

/// Refuse a set of prompts whose budgets could together ask one unit for
/// more than [`MAX_QUESTIONS`].
fn within_the_cap(layer: &str, what: &str, most: usize) -> anyhow::Result<()> {
    if most > MAX_QUESTIONS {
        anyhow::bail!(
            "layer '{layer}': `calibrate.{what}` can ask a unit for {most} questions, past the \
             {MAX_QUESTIONS} one calibration holds"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_conversation::projection::Builder;

    fn layer(block: &str) -> LayerSchema {
        let yaml = format!(
            r#"
system_prompt:
  sections:
    - id: frame
      content: "frame"
layers:
  - name: code_reading
    window: 8000
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    score_formula: max
    budget:
      priority: 40
{block}
    groups:
      - id: scopes
        selection: {{ kind: working_set, share: remainder }}
      - id: guidance
        draws: {{ from: scopes, paths: [candle/mcp/src/guides/], except: [README.md] }}
        selection: {{ kind: working_set, share: guidance }}
"#
        );
        Builder::from_yaml(&yaml).unwrap().schema().layers[0].clone()
    }

    /// A profile's YAML: `files` (none for the default) and one prompt per
    /// `(name, max)`.
    fn profile(name: &str, files: &[&str], prompts: &[(&str, usize)]) -> String {
        let mut s = format!("        - name: {name}\n");
        if !files.is_empty() {
            let quoted: Vec<String> = files.iter().map(|f| format!("'{f}'")).collect();
            s += &format!("          files: [{}]\n", quoted.join(", "));
        }
        s += "          prompts:\n";
        for (p, max) in prompts {
            s += &format!(
                "            - name: {p}\n              questions: {{ per_tokens: 300, min: 1, max: \
                 {max} }}\n              prompt: \"{name} {p} <<n>> on `<<path>>`\"\n"
            );
        }
        s
    }

    fn block(profiles: &[String], guidance: bool) -> LayerSchema {
        let mut s = String::from("    calibrate:\n      think: 64\n      profiles:\n");
        for p in profiles {
            s += p;
        }
        if guidance {
            s += "      guidance:\n        questions: { per_tokens: 250, min: 8, max: 64 }\n        \
                  prompt: \"worker <<n>> before `<<path>>`\"\n";
        }
        layer(&s)
    }

    fn spec() -> CalibrateSpec {
        CalibrateSpec::of(&block(
            &[
                profile("manifest", &[r"(^|/)Cargo\.toml$"], &[("dependencies", 16)]),
                profile("code", &[r"\.rs$"], &[("definitions", 32), ("uses", 24)]),
                profile("default", &[], &[("definitions", 32)]),
            ],
            true,
        ))
        .unwrap()
        .expect("declared")
    }

    /// **A unit is asked by the first profile that takes it**, the default
    /// when none does, and a drawn guide by its guidance prompt alone.
    #[test]
    fn a_unit_is_asked_by_the_first_profile_that_takes_it() {
        let s = spec();
        let names = |subject: &str| {
            s.asks_for(subject)
                .iter()
                .map(|a| a.name.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(names("candle/zend/Cargo.toml"), ["dependencies"]);
        assert_eq!(names("candle/zend/src/lib.rs"), ["definitions", "uses"]);
        assert_eq!(names("candle/docs/paper.md"), ["definitions"]);
        assert_eq!(names("candle/mcp/src/guides/deploy.md"), ["guidance"]);
        assert_eq!(s.profile_for("candle/zend/src/lib.rs").name, "code");
        assert_eq!(
            s.asks_for("candle/zend/Cargo.toml")[0].prompt_for("candle/zend/Cargo.toml", 5),
            "manifest dependencies 5 on `zend/Cargo.toml`"
        );
    }

    /// **A change to one profile calibrates only the units it asks**; the
    /// others' markers stand.
    #[test]
    fn a_changed_profile_moves_only_its_own_units_markers() {
        let s = spec();
        let mut reworded = s.clone();
        reworded.profiles[0].asks[0].prompt.push('!');
        let (manifest, code) = ("candle/zend/Cargo.toml", "candle/zend/src/lib.rs");
        assert_ne!(reworded.marker("k", manifest), s.marker("k", manifest));
        assert_eq!(reworded.marker("k", code), s.marker("k", code));
        let mut guided = s.clone();
        guided.guidance.as_mut().unwrap().ask.prompt.push('!');
        let guide = "candle/mcp/src/guides/deploy.md";
        assert_ne!(guided.marker("g", guide), s.marker("g", guide));
        assert_eq!(guided.marker("k", code), s.marker("k", code));
    }

    /// The marker changes with the profile's prompts, their rules and order,
    /// the thinking budget and the unit, and is read back as the layer's.
    #[test]
    fn a_marker_follows_what_is_asked_and_of_which_unit() {
        let s = spec();
        let at = "candle/a.rs";
        let m = s.marker("k1", at);
        assert!(m.starts_with("code_reading|") && m.ends_with("|k1"), "{m}");
        assert_eq!(s.marker("k1", at), m);
        assert_ne!(s.marker("k2", at), m);
        let mut more = s.clone();
        more.profiles[1].asks[1].questions.per_tokens = 200;
        assert_ne!(more.marker("k1", at), m);
        let mut longer = s.clone();
        longer.think = 128;
        assert_ne!(longer.marker("k1", at), m);
        let mut swapped = s.clone();
        swapped.profiles[1].asks.swap(0, 1);
        assert_ne!(swapped.marker("k1", at), m, "the order is what is asked");
        assert_eq!(CalibrateSpec::unit_of(&m), Some("k1"));
        assert_eq!(
            CalibrateSpec::unit_of("code_reading|ab|key:a|b"),
            Some("key:a|b")
        );
        assert_eq!(CalibrateSpec::unit_of("code_reading|ab"), None);
        assert!(s.owns(&m));
        assert!(!s.owns("code_reading_x|ab|k1"));
        assert!(!s.owns("repo_map|ab|k1"));
    }

    fn refused(profiles: &[String]) -> String {
        CalibrateSpec::of(&block(profiles, false))
            .unwrap_err()
            .to_string()
    }

    #[test]
    fn a_bad_set_of_profiles_is_refused_and_says_why() {
        let default = profile("default", &[], &[("definitions", 8)]);
        assert_eq!(
            refused(&[profile("code", &[r"\.rs$"], &[("definitions", 8)])]),
            "layer 'code_reading': `calibrate.profiles` ends with 'code', which has `files`; the \
             last profile is the default and has none"
        );
        assert_eq!(
            refused(&[
                profile("first", &[], &[("definitions", 8)]),
                default.clone()
            ]),
            "layer 'code_reading': `calibrate.profiles`: 'first' has no `files` but is not last — \
             every profile after it would be unreachable"
        );
        assert_eq!(
            refused(&[
                profile("code", &[r"\.rs$"], &[("definitions", 8)]),
                profile("code", &[r"\.py$"], &[("definitions", 8)]),
                default.clone()
            ]),
            "layer 'code_reading': `calibrate.profiles`: 'code' is declared twice"
        );
        assert_eq!(
            refused(&[
                profile("code", &[r"\.rs$"], &[("definitions", 40), ("uses", 25)]),
                default
            ]),
            "layer 'code_reading': `calibrate.profiles: 'code'` can ask a unit for 65 questions, \
             past the 64 one calibration holds"
        );
        assert!(CalibrateSpec::of(&layer("")).unwrap().is_none());
        let empty = layer("    calibrate:\n      think: 64\n      profiles: []\n");
        assert_eq!(
            CalibrateSpec::of(&empty).unwrap_err().to_string(),
            "layer 'code_reading': `calibrate.profiles` is empty"
        );
    }

    /// **A guidance prompt takes the paths the layer's drawing groups draw.**
    #[test]
    fn a_guidance_prompt_takes_the_drawn_paths() {
        let s = spec();
        let g = s.guidance.expect("a guidance prompt");
        assert_eq!(g.ask.name, "guidance");
        assert_eq!(g.draws.len(), 1);
        assert_eq!(g.draws[0].paths, ["candle/mcp/src/guides/"]);
        assert_eq!(g.draws[0].except, ["README.md"]);
    }
}
