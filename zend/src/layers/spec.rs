//! A layer's `ingest:` block — what one of its units is, what it builds on, and
//! what it is asked — read from the projection schema it is declared in.
//!
//! ```yaml
//! - name: dependency_analysis
//!   dials: { thinking_effort: balanced }
//!   ingest:
//!     unit: file            # one unit per file the layer below holds
//!     after: static_analysis
//!     edges: true           # the answer names the files it depends on
//!     prompt: |
//!       What do the structs, traits and functions in `<<path>>` depend on? …
//! ```
//!
//! The thinking a unit runs with is the layer's own `thinking_effort` dial —
//! the same dial a dialogue turn on the layer would read — so one level names
//! it everywhere and nothing caps it here.

use candle_conversation::projection::LayerSchema;
use candle_conversation::stencil::ThinkMode;
use serde::Deserialize;

/// What one unit of a layer is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnitKind {
    /// One unit per unit of the layer it builds on — per file, all the way
    /// down to `code_reading`.
    File,
    /// One unit per closed day of dialogue — the first day layer reads the
    /// day's conversations, a day layer above it builds on the one below.
    Day,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Raw {
    unit: UnitKind,
    #[serde(default)]
    after: Option<String>,
    prompt: String,
    #[serde(default)]
    edges: bool,
}

/// A layer populated by its own units, each a hidden conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerSpec {
    /// The layer's name.
    pub layer: String,
    pub unit: UnitKind,
    /// The layer each unit builds on: one unit here per unit of it, forked from
    /// that unit so its whole conversation is this one's opening. `None` only
    /// for a day layer reading the day's dialogue itself.
    pub after: Option<String>,
    /// The question each unit is asked, with `<<path>>`, `<<repo>>` and `<<file>>`
    /// (a file unit) or `<<day>>` and `<<transcript>>` (a day unit) filled in.
    pub prompt: String,
    /// Whether the answer names the files the unit's file depends on — the
    /// edges a change is followed along to its dependents.
    pub edges: bool,
    /// The layer's `thinking_effort` dial, as the selection names it.
    pub effort: String,
}

impl LayerSpec {
    /// The layer's spec, `None` when it declares no `ingest:` block.
    pub fn of(layer: &LayerSchema) -> anyhow::Result<Option<Self>> {
        let Some(value) = &layer.ingest else {
            return Ok(None);
        };
        let raw: Raw = serde_yaml::from_value(value.clone())
            .map_err(|e| anyhow::anyhow!("layer '{}': its `ingest:` block: {e}", layer.name))?;
        if raw.unit == UnitKind::File && raw.after.is_none() {
            anyhow::bail!(
                "layer '{}': a file unit builds on a layer below it — name it in `after:`",
                layer.name
            );
        }
        Ok(Some(Self {
            layer: layer.name.clone(),
            unit: raw.unit,
            after: raw.after,
            prompt: raw.prompt,
            edges: raw.edges,
            effort: layer
                .dials
                .get("thinking_effort")
                .unwrap_or("balanced")
                .to_string(),
        }))
    }

    /// The steering the layer's thinking dial resolves to.
    pub fn think_mode(&self) -> ThinkMode {
        match self.effort.as_str() {
            "off" => ThinkMode::Off,
            "quick" => ThinkMode::Quick,
            "deep" => ThinkMode::Deep,
            "exhaustive" => ThinkMode::Exhaustive,
            _ => ThinkMode::Balanced,
        }
    }

    /// The question for the file `file` (workspace-relative, its repository
    /// first).
    pub fn file_prompt(&self, file: &str) -> String {
        let (repo, path) = crate::repo_path::split(file);
        self.prompt
            .replace("<<file>>", file)
            .replace("<<repo>>", repo)
            .replace("<<path>>", path)
    }

    /// The question for `day`, its dialogue rendered as `transcript`.
    pub fn day_prompt(&self, day: &str, transcript: &str) -> String {
        self.prompt
            .replace("<<day>>", day)
            .replace("<<transcript>>", transcript)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(unit: UnitKind) -> LayerSpec {
        LayerSpec {
            layer: "static_analysis".into(),
            unit,
            after: Some("code_reading".into()),
            prompt: "Read `<<path>>` of `<<repo>>` (<<file>>) on <<day>>: <<transcript>>".into(),
            edges: false,
            effort: "deep".into(),
        }
    }

    #[test]
    fn a_file_prompt_names_the_file_three_ways() {
        assert_eq!(
            spec(UnitKind::File).file_prompt("candle/zend/src/lib.rs"),
            "Read `zend/src/lib.rs` of `candle` (candle/zend/src/lib.rs) on <<day>>: <<transcript>>"
        );
    }

    #[test]
    fn a_day_prompt_carries_its_transcript() {
        assert_eq!(
            spec(UnitKind::Day).day_prompt("2026-10-20", "A: hi"),
            "Read `<<path>>` of `<<repo>>` (<<file>>) on 2026-10-20: A: hi"
        );
    }

    #[test]
    fn the_thinking_dial_names_the_steering() {
        assert_eq!(spec(UnitKind::File).think_mode(), ThinkMode::Deep);
        let mut s = spec(UnitKind::File);
        s.effort = "off".into();
        assert_eq!(s.think_mode(), ThinkMode::Off);
    }
}
