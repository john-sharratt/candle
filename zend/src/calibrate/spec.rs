//! A layer's `calibrate:` block — how many questions each of its
//! conversations is asked to write about itself, and the prompt that asks —
//! read from the projection schema it is declared in.
//!
//! ```yaml
//! - name: code_reading
//!   calibrate:
//!     questions: 16
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

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Raw {
    questions: usize,
    prompt: String,
}

/// How a layer's conversations are calibrated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrateSpec {
    /// The layer's name.
    pub layer: String,
    /// How many questions each unit is asked to write.
    pub questions: usize,
    /// What it is asked, with `<<n>>`, `<<subject>>`, and for a file subject
    /// `<<file>>`, `<<repo>>` and `<<path>>` filled in.
    pub prompt: String,
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
        if raw.prompt.trim().is_empty() {
            anyhow::bail!("layer '{}': `calibrate.prompt` is empty", layer.name);
        }
        Ok(Some(Self {
            layer: layer.name.clone(),
            questions: raw.questions,
            prompt: raw.prompt,
        }))
    }

    /// The prompt for the unit whose subject is `subject` — a file's
    /// workspace-relative path, its repository first, or a day.
    pub fn prompt_for(&self, subject: &str) -> String {
        let (repo, path) = crate::repo_path::split(subject);
        self.prompt
            .replace("<<n>>", &self.questions.to_string())
            .replace("<<subject>>", subject)
            .replace("<<file>>", subject)
            .replace("<<repo>>", repo)
            .replace("<<path>>", path)
    }

    /// The calibration marker of the unit committed under `content_key`: the
    /// layer, a digest of what it is asked, and the unit's own key — so a
    /// changed prompt, question count or unit version calibrates again.
    pub fn marker(&self, content_key: &str) -> String {
        let mut h = Sha256::new();
        h.update(self.questions.to_le_bytes());
        h.update([0u8]);
        h.update(self.prompt.as_bytes());
        let digest: String = h.finalize()[..8]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        format!("{}|{digest}|{content_key}", self.layer)
    }

    /// Whether `marker` is one of this layer's.
    pub fn owns(&self, marker: &str) -> bool {
        marker
            .strip_prefix(self.layer.as_str())
            .is_some_and(|rest| rest.starts_with('|'))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> CalibrateSpec {
        CalibrateSpec {
            layer: "code_reading".into(),
            questions: 3,
            prompt: "<<n>> about `<<path>>` in <<repo>> (<<file>>, <<subject>>)".into(),
        }
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
        let m = s.marker("k1");
        assert!(m.starts_with("code_reading|") && m.ends_with("|k1"), "{m}");
        assert_eq!(s.marker("k1"), m);
        assert_ne!(s.marker("k2"), m);
        let mut more = s.clone();
        more.questions = 4;
        assert_ne!(more.marker("k1"), m);
        let mut reworded = s.clone();
        reworded.prompt.push('!');
        assert_ne!(reworded.marker("k1"), m);
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
        candle_conversation::projection::Builder::from_yaml(&yaml)
            .unwrap()
            .schema()
            .layers[0]
            .clone()
    }

    #[test]
    fn a_block_is_read_and_a_bad_one_refused() {
        let ok = layer("    calibrate:\n      questions: 16\n      prompt: \"ask <<n>>\"");
        let s = CalibrateSpec::of(&ok).unwrap().expect("declared");
        assert_eq!((s.layer.as_str(), s.questions), ("code_reading", 16));
        assert!(CalibrateSpec::of(&layer("")).unwrap().is_none());
        let zero = layer("    calibrate:\n      questions: 0\n      prompt: \"x\"");
        assert!(CalibrateSpec::of(&zero).is_err());
        let blank = layer("    calibrate:\n      questions: 4\n      prompt: \" \"");
        assert!(CalibrateSpec::of(&blank).is_err());
        let extra =
            layer("    calibrate:\n      questions: 4\n      prompt: \"x\"\n      tools: []");
        assert!(CalibrateSpec::of(&extra).is_err());
    }
}
