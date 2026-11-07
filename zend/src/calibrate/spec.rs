//! A layer's `calibrate:` block — how many questions each of its
//! conversations is asked to write about itself, the prompt that asks, and
//! how much of each question's thinking is kept — read from the projection
//! schema it is declared in.
//!
//! ```yaml
//! - name: code_reading
//!   calibrate:
//!     questions: 24
//!     think: 64
//!     prompt: |
//!       Write <<n>> questions someone might ask whose answer needs `<<path>>` …
//! ```
//!
//! Nothing here names a layer: any layer whose conversations are units — a
//! file's reading, an analysis of it, a day — may declare one, and each of its
//! units is calibrated the same way ([`super::run`]).

use candle_conversation::projection::LayerSchema;
use serde::Deserialize;
use sha2::{Digest, Sha256};

/// The most questions one unit may be asked for. Each is one turn of one
/// prefill, and the calibration's K/V is held lossless until it seals.
pub const MAX_QUESTIONS: usize = 64;

/// The most thinking tokens kept per question: a live probe's trailing
/// window is at most this long (`REPROJECT_MAX_PROBE_TOKENS`).
pub const MAX_THINK: usize = 256;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Raw {
    questions: usize,
    think: usize,
    prompt: String,
    /// What a unit a drawing group of the layer draws is asked instead.
    #[serde(default)]
    guidance_prompt: Option<String>,
}

/// How a layer's conversations are calibrated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrateSpec {
    /// The layer's name.
    pub layer: String,
    /// How many questions each unit is asked to write.
    pub questions: usize,
    /// The decode budget of each question's thinking ([`super::thinking`]).
    pub think: usize,
    /// What it is asked, with `<<n>>`, `<<subject>>`, and for a file subject
    /// `<<file>>`, `<<repo>>` and `<<path>>` filled in.
    pub prompt: String,
    /// What a unit the layer's drawing groups draw is asked instead, and the
    /// paths they draw (`docs/guidance_layer.md` §2.2). A guide is wanted
    /// at a point in a worker's own reasoning, so its seed questions are the
    /// ones a worker asks itself mid-task, not a user's.
    pub guidance: Option<GuidancePrompt>,
}

/// The prompt for the units a layer's drawing groups draw.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuidancePrompt {
    pub prompt: String,
    /// Workspace-relative prefixes, every drawing group's together.
    pub paths: Vec<String>,
}

impl CalibrateSpec {
    /// The layer's spec, `None` when it declares no `calibrate:` block.
    pub fn of(layer: &LayerSchema) -> anyhow::Result<Option<Self>> {
        let Some(value) = &layer.calibrate else {
            return Ok(None);
        };
        let raw: Raw = serde_yaml::from_value(value.clone())
            .map_err(|e| anyhow::anyhow!("layer '{}': its `calibrate:` block: {e}", layer.name))?;
        if raw.questions == 0 || raw.questions > MAX_QUESTIONS {
            anyhow::bail!(
                "layer '{}': `calibrate.questions` must be 1..={MAX_QUESTIONS}, not {}",
                layer.name,
                raw.questions
            );
        }
        if raw.think == 0 || raw.think > MAX_THINK {
            anyhow::bail!(
                "layer '{}': `calibrate.think` must be 1..={MAX_THINK}, not {}",
                layer.name,
                raw.think
            );
        }
        if raw.prompt.trim().is_empty() {
            anyhow::bail!("layer '{}': `calibrate.prompt` is empty", layer.name);
        }
        let paths: Vec<String> = layer
            .groups
            .iter()
            .filter_map(|g| g.draws.as_ref())
            .flat_map(|d| d.paths.iter().cloned())
            .collect();
        let guidance = match raw.guidance_prompt {
            None => None,
            Some(prompt) if prompt.trim().is_empty() => {
                anyhow::bail!(
                    "layer '{}': `calibrate.guidance_prompt` is empty",
                    layer.name
                )
            }
            Some(_) if paths.is_empty() => anyhow::bail!(
                "layer '{}': `calibrate.guidance_prompt` is declared but no group of the layer \
                 draws anything",
                layer.name
            ),
            Some(prompt) => Some(GuidancePrompt { prompt, paths }),
        };
        Ok(Some(Self {
            layer: layer.name.clone(),
            questions: raw.questions,
            think: raw.think,
            prompt: raw.prompt,
            guidance,
        }))
    }

    /// The template `subject` is asked under: the guidance prompt for a unit
    /// a drawing group draws, the layer's prompt for every other.
    fn template_for(&self, subject: &str) -> &str {
        match &self.guidance {
            Some(g) if g.paths.iter().any(|p| subject.starts_with(p.as_str())) => &g.prompt,
            _ => &self.prompt,
        }
    }

    /// The prompt for the unit whose subject is `subject` — a file's
    /// workspace-relative path, its repository first, or a day.
    pub fn prompt_for(&self, subject: &str) -> String {
        fill(self.template_for(subject), self.questions, subject)
    }

    /// The calibration marker of the unit committed under `content_key`, whose
    /// subject is `subject`: the layer, a digest of what it is asked, and the
    /// unit's own key — so a changed prompt, question count, thinking budget
    /// or unit version calibrates again, and a changed guidance prompt only
    /// the units asked it.
    pub fn marker(&self, content_key: &str, subject: &str) -> String {
        let mut h = Sha256::new();
        h.update(self.questions.to_le_bytes());
        h.update(self.think.to_le_bytes());
        h.update([0u8]);
        h.update(self.template_for(subject).as_bytes());
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

/// `template` with the subject's fields and the question count filled in.
fn fill(template: &str, questions: usize, subject: &str) -> String {
    let (repo, path) = crate::repo_path::split(subject);
    template
        .replace("<<n>>", &questions.to_string())
        .replace("<<subject>>", subject)
        .replace("<<file>>", subject)
        .replace("<<repo>>", repo)
        .replace("<<path>>", path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_conversation::projection::Builder;

    fn spec() -> CalibrateSpec {
        CalibrateSpec {
            layer: "code_reading".into(),
            questions: 3,
            think: 64,
            prompt: "<<n>> about `<<path>>` in <<repo>> (<<file>>, <<subject>>)".into(),
            guidance: None,
        }
    }

    fn with_guidance() -> CalibrateSpec {
        CalibrateSpec {
            guidance: Some(GuidancePrompt {
                prompt: "<<n>> a worker asks itself before `<<path>>`".into(),
                paths: vec!["candle/mcp/src/guides/".into(), "journals/".into()],
            }),
            ..spec()
        }
    }

    /// **A drawn unit is asked the guidance prompt**; every other unit the
    /// layer's.
    #[test]
    fn a_drawn_unit_is_asked_the_guidance_prompt() {
        let s = with_guidance();
        assert_eq!(
            s.prompt_for("candle/mcp/src/guides/deploy.md"),
            "3 a worker asks itself before `mcp/src/guides/deploy.md`"
        );
        assert_eq!(
            s.prompt_for("candle/docs/paper.md"),
            "3 about `docs/paper.md` in candle (candle/docs/paper.md, candle/docs/paper.md)"
        );
    }

    /// **A changed guidance prompt calibrates only the units it asks**: a code
    /// file's marker stands.
    #[test]
    fn a_changed_guidance_prompt_moves_only_the_drawn_markers() {
        let s = with_guidance();
        let mut reworded = s.clone();
        reworded.guidance.as_mut().unwrap().prompt.push('!');
        let (guide, code) = ("candle/mcp/src/guides/deploy.md", "candle/docs/paper.md");
        assert_ne!(reworded.marker("g", guide), s.marker("g", guide));
        assert_eq!(reworded.marker("c", code), s.marker("c", code));
        assert_eq!(s.marker("c", code), spec().marker("c", code));
    }

    #[test]
    fn a_prompt_names_its_count_and_subject() {
        assert_eq!(
            spec().prompt_for("candle/docs/paper.md"),
            "3 about `docs/paper.md` in candle (candle/docs/paper.md, candle/docs/paper.md)"
        );
    }

    /// The marker changes with the prompt, the count and the unit, and is
    /// read back as the layer's; another layer's is not.
    #[test]
    fn a_marker_follows_what_is_asked_and_of_which_unit() {
        let s = spec();
        let at = "candle/a.rs";
        let m = s.marker("k1", at);
        assert!(m.starts_with("code_reading|") && m.ends_with("|k1"), "{m}");
        assert_eq!(s.marker("k1", at), m);
        assert_ne!(s.marker("k2", at), m);
        let mut more = s.clone();
        more.questions = 4;
        assert_ne!(more.marker("k1", at), m);
        let mut longer = s.clone();
        longer.think = 128;
        assert_ne!(longer.marker("k1", at), m);
        let mut reworded = s.clone();
        reworded.prompt.push('!');
        assert_ne!(reworded.marker("k1", at), m);
        assert_eq!(CalibrateSpec::unit_of(&m), Some("k1"));
        assert_eq!(CalibrateSpec::unit_of(&longer.marker("k1", at)), Some("k1"));
        assert_eq!(
            CalibrateSpec::unit_of("code_reading|ab|key:a|b"),
            Some("key:a|b")
        );
        assert_eq!(CalibrateSpec::unit_of("code_reading|ab"), None);
        assert!(s.owns(&m));
        assert!(!s.owns("code_reading_x|ab|k1"));
        assert!(!s.owns("repo_map|ab|k1"));
    }

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
        selection: {{ kind: top_k, k: 2 }}
"#
        );
        Builder::from_yaml(&yaml).unwrap().schema().layers[0].clone()
    }

    #[test]
    fn a_block_is_read_and_a_bad_one_refused() {
        let block = |questions: usize, think: &str, prompt: &str| {
            layer(&format!(
                "    calibrate:\n      questions: {questions}\n{think}      prompt: \"{prompt}\""
            ))
        };
        let ok = block(24, "      think: 192\n", "ask <<n>>");
        let s = CalibrateSpec::of(&ok).unwrap().expect("declared");
        assert_eq!(
            (s.layer.as_str(), s.questions, s.think),
            ("code_reading", 24, 192)
        );
        assert!(CalibrateSpec::of(&layer("")).unwrap().is_none());
        assert!(CalibrateSpec::of(&block(0, "      think: 8\n", "x")).is_err());
        assert!(CalibrateSpec::of(&block(4, "      think: 0\n", "x")).is_err());
        assert!(CalibrateSpec::of(&block(4, "      think: 257\n", "x")).is_err());
        assert!(
            CalibrateSpec::of(&block(4, "", "x")).is_err(),
            "think is required"
        );
        assert!(CalibrateSpec::of(&block(4, "      think: 8\n", " ")).is_err());
        let extra = layer(
            "    calibrate:\n      questions: 4\n      think: 8\n      prompt: \"x\"\n      tools: []",
        );
        assert!(CalibrateSpec::of(&extra).is_err());
        let undrawn = layer(
            "    calibrate:\n      questions: 4\n      think: 8\n      prompt: \"x\"\n      \
             guidance_prompt: \"y\"",
        );
        assert!(
            CalibrateSpec::of(&undrawn).is_err(),
            "a guidance prompt with nothing drawn"
        );
    }

    /// **A guidance prompt takes the paths the layer's drawing groups draw.**
    #[test]
    fn a_guidance_prompt_takes_the_drawn_paths() {
        let yaml = r#"
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
    calibrate:
      questions: 4
      think: 8
      prompt: "x"
      guidance_prompt: "y"
    groups:
      - id: scopes
        selection: { kind: working_set, share: remainder }
      - id: guidance
        draws: { from: scopes, paths: [candle/mcp/src/guides/, journals/] }
        selection: { kind: working_set, share: guidance }
"#;
        let layer = Builder::from_yaml(yaml).unwrap().schema().layers[0].clone();
        let s = CalibrateSpec::of(&layer).unwrap().expect("declared");
        assert_eq!(
            s.guidance,
            Some(GuidancePrompt {
                prompt: "y".into(),
                paths: vec!["candle/mcp/src/guides/".into(), "journals/".into()],
            })
        );
    }
}
