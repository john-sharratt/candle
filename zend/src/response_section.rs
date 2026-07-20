//! Response-section loading — the calibrated section collections (`response`,
//! `mood`, …).
//!
//! A section-collection sink (see [`crate::ingest::section_sinks`]) is filled
//! from `<collection>s/*.yaml`, one file per section, each validated against the
//! response-section schema. The `template` is installed as the collection
//! section's content (its KV seals at base-conv build, exactly like a tool's JSON
//! line); the `examples` are provenance lead-ins that train the section's Q-value
//! selection in the calibration phase.
//!
//! This module owns the parse + validation + `{CHAR_NAME}`/`{USER_NAME}`
//! substitution + install. The examples are parsed and validated here (ready for
//! the calibration pass); this pass installs the templates so the collection is
//! populated with baseline provenance from each template's own prefill.

use std::path::Path;

use candle_conversation::projection::{Builder as ProjectionBuilder, SectionId};
use serde::Deserialize;

/// A turn role in a lead-in example. The schema admits `user` / `assistant` only
/// (`system` framing lives in the section template, not the examples).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SectionRole {
    User,
    Assistant,
}

/// One authored turn in a lead-in example.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Turn {
    pub role: SectionRole,
    /// Authored turn text. `{CHAR_NAME}` substituted at load.
    #[serde(default)]
    pub content: Option<String>,
    /// Third-person reasoning trace (assistant turns only; present on the final
    /// decode-point turn). The loader wraps it in `<think>…</think>`.
    #[serde(default)]
    pub thinking: Option<String>,
}

/// One provenance lead-in: an ordered turn sequence ending at the decode point.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Example {
    /// Optional author note on what this example exercises. Carried for tooling;
    /// not consumed by the loader.
    #[serde(default)]
    #[allow(dead_code)]
    pub note: Option<String>,
    pub turns: Vec<Turn>,
}

/// One selectable response template plus its calibration examples. One file per
/// section under `<collection>s/`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResponseSection {
    pub id: String,
    /// Grouping (e.g. social, combat, communication). Carried for tooling /
    /// calibration bucketing; not consumed by the install pass.
    #[allow(dead_code)]
    pub category: String,
    /// Optional one-line human summary. Carried for tooling; not consumed here.
    #[serde(default)]
    #[allow(dead_code)]
    pub description: Option<String>,
    /// The frozen structural mode — installed as the collection section's content.
    pub template: String,
    pub examples: Vec<Example>,
}

/// Character/user identity used for `{CHAR_NAME}`/`{USER_NAME}` substitution,
/// read from an optional `<workspace>/mind.yaml`.
pub struct Identity {
    pub char_name: String,
    pub user_name: String,
}

impl Identity {
    /// Load `char_name` / `user_name` from `<workspace>/mind.yaml`, falling back
    /// to neutral defaults when the file is absent or a field is unset.
    pub fn load(workspace: &Path) -> Self {
        #[derive(Deserialize, Default)]
        struct Raw {
            #[serde(default)]
            char_name: Option<String>,
            #[serde(default)]
            user_name: Option<String>,
        }
        let raw = std::fs::read_to_string(workspace.join("mind.yaml"))
            .ok()
            .and_then(|t| serde_yaml::from_str::<Raw>(&t).ok())
            .unwrap_or_default();
        Self {
            char_name: raw.char_name.unwrap_or_else(|| "Assistant".to_string()),
            user_name: raw.user_name.unwrap_or_else(|| "User".to_string()),
        }
    }

    fn substitute(&self, s: &str) -> String {
        s.replace("{CHAR_NAME}", &self.char_name)
            .replace("{USER_NAME}", &self.user_name)
    }
}

/// Validate the compiler-enforced invariants the JSON Schema defers: the id must
/// match the filename stem, and each example must end at a decode point — a final
/// `assistant` turn carrying `thinking` and no `content`. Also enforces that
/// `thinking` appears only on assistant turns and every turn has content or
/// thinking (the schema's `anyOf`).
pub fn validate(section: &ResponseSection, stem: &str) -> anyhow::Result<()> {
    if section.id != stem {
        anyhow::bail!("id {:?} must match filename stem {:?}", section.id, stem);
    }
    if section.examples.is_empty() {
        anyhow::bail!("section {:?}: needs at least one example", section.id);
    }
    for (i, ex) in section.examples.iter().enumerate() {
        if ex.turns.len() < 2 {
            anyhow::bail!(
                "section {:?} example {i}: needs at least 2 turns",
                section.id
            );
        }
        for (j, t) in ex.turns.iter().enumerate() {
            if t.content.is_none() && t.thinking.is_none() {
                anyhow::bail!(
                    "section {:?} example {i} turn {j}: needs `content` or `thinking`",
                    section.id
                );
            }
            if t.thinking.is_some() && t.role != SectionRole::Assistant {
                anyhow::bail!(
                    "section {:?} example {i} turn {j}: `thinking` is allowed on assistant turns only",
                    section.id
                );
            }
        }
        let last = ex.turns.last().expect("len >= 2 checked above");
        if last.role != SectionRole::Assistant {
            anyhow::bail!(
                "section {:?} example {i}: final turn must be role=assistant (the decode point)",
                section.id
            );
        }
        if last
            .thinking
            .as_deref()
            .map(str::trim)
            .unwrap_or("")
            .is_empty()
        {
            anyhow::bail!(
                "section {:?} example {i}: final turn must carry `thinking`",
                section.id
            );
        }
        if last.content.is_some() {
            anyhow::bail!(
                "section {:?} example {i}: final (decode-point) turn must have no `content` (the reply is not authored)",
                section.id
            );
        }
    }
    Ok(())
}

/// Load, validate, and substitute every `*.yaml` section under `dir`, in sorted
/// filename order. A file that fails to parse or validate is logged and skipped —
/// one bad section never bricks daemon load. Missing `dir` yields an empty list.
pub fn load_sections(dir: &Path, identity: &Identity) -> Vec<ResponseSection> {
    let mut paths: Vec<std::path::PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("yaml"))
            .collect(),
        Err(_) => return Vec::new(),
    };
    paths.sort();

    let mut out = Vec::new();
    for path in paths {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(file = %path.display(), "response section read failed: {e}");
                continue;
            }
        };
        let mut section: ResponseSection = match serde_yaml::from_str(&text) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(file = %path.display(), "response section parse failed: {e}");
                continue;
            }
        };
        if let Err(e) = validate(&section, &stem) {
            tracing::warn!(file = %path.display(), "response section invalid: {e:#}");
            continue;
        }
        // Substitute identity placeholders across the template and every example.
        section.template = identity.substitute(&section.template);
        for ex in &mut section.examples {
            for t in &mut ex.turns {
                if let Some(c) = &mut t.content {
                    *c = identity.substitute(c);
                }
                if let Some(th) = &mut t.thinking {
                    *th = identity.substitute(th);
                }
            }
        }
        out.push(section);
    }
    out
}

/// Install each section's `template` as a member of `collection` in the shared
/// system prompt, mirroring [`crate::tools::install_tool_catalog`]. The section id
/// is the collection member name; the template is its content (its KV seals at
/// base-conv build, giving the section baseline provenance from its own prefill).
/// Returns the `(id, section_id)` pairs for the calibration pass to key on.
pub fn install_sections(
    builder: &mut ProjectionBuilder,
    collection: &str,
    sections: &[ResponseSection],
) -> anyhow::Result<Vec<(String, SectionId)>> {
    let collection_id = builder
        .id_for_system_collection(collection)
        .ok_or_else(|| {
            anyhow::anyhow!("system prompt missing '{collection}' collection for section install")
        })?;
    let mut out = Vec::with_capacity(sections.len());
    for section in sections {
        let id = builder
            .add_section_to_collection(collection_id, section.id.clone(), &section.template, 100.0)
            .map_err(|e| anyhow::anyhow!("install section {:?}: {e}", section.id))?;
        out.push((section.id.clone(), id));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_yaml() -> &'static str {
        "id: greet_warmly\n\
         category: social\n\
         template: \"{CHAR_NAME} greets {USER_NAME} warmly.\"\n\
         examples:\n\
         \x20 - turns:\n\
         \x20     - role: user\n\
         \x20       content: \"Hello {CHAR_NAME}\"\n\
         \x20     - role: assistant\n\
         \x20       thinking: \"{USER_NAME} greeted {CHAR_NAME}; she answers warmly.\"\n"
    }

    #[test]
    fn parses_validates_and_substitutes() {
        let mut section: ResponseSection = serde_yaml::from_str(valid_yaml()).unwrap();
        validate(&section, "greet_warmly").unwrap();
        let id = Identity {
            char_name: "Aria".to_string(),
            user_name: "John".to_string(),
        };
        section.template = id.substitute(&section.template);
        assert_eq!(section.template, "Aria greets John warmly.");
        assert_eq!(section.category, "social");
        assert_eq!(section.examples.len(), 1);
    }

    #[test]
    fn id_must_match_stem() {
        let section: ResponseSection = serde_yaml::from_str(valid_yaml()).unwrap();
        assert!(validate(&section, "some_other_stem").is_err());
    }

    #[test]
    fn final_turn_must_be_assistant_thinking_no_content() {
        // Final turn carries content → rejected (the reply must not be authored).
        let bad = "id: s\ncategory: c\ntemplate: t\nexamples:\n  - turns:\n      - role: user\n        content: hi\n      - role: assistant\n        thinking: reasons\n        content: authored\n";
        let section: ResponseSection = serde_yaml::from_str(bad).unwrap();
        assert!(validate(&section, "s").is_err());

        // Final turn is user → rejected.
        let bad2 = "id: s\ncategory: c\ntemplate: t\nexamples:\n  - turns:\n      - role: assistant\n        thinking: r\n      - role: user\n        content: hi\n";
        let section2: ResponseSection = serde_yaml::from_str(bad2).unwrap();
        assert!(validate(&section2, "s").is_err());
    }

    #[test]
    fn thinking_on_user_turn_is_rejected() {
        let bad = "id: s\ncategory: c\ntemplate: t\nexamples:\n  - turns:\n      - role: user\n        thinking: nope\n      - role: assistant\n        thinking: r\n";
        let section: ResponseSection = serde_yaml::from_str(bad).unwrap();
        assert!(validate(&section, "s").is_err());
    }

    #[test]
    fn unknown_field_is_rejected() {
        let bad = "id: s\ncategory: c\ntemplate: t\nbogus: x\nexamples:\n  - turns:\n      - role: user\n        content: hi\n      - role: assistant\n        thinking: r\n";
        assert!(serde_yaml::from_str::<ResponseSection>(bad).is_err());
    }
}
