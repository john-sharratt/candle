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
//!
//! A layer whose answer depends on the world outside the branches — the
//! threats currently known against what a file uses — may also name the
//! research tools ([`RESEARCH_TOOLS`]) and an `expires_days`: its units search
//! the web, and each is read again within that many days
//! ([`super::expiry`]).

use candle_conversation::projection::LayerSchema;
use candle_conversation::stencil::ThinkMode;
use serde::Deserialize;

use super::expiry::Expiry;
use crate::types::ToolMode;

/// The tools that read the workspace at the unit's own commit and change
/// nothing. A unit is an analysis of what the branches hold, so a tool that
/// writes or runs has no place in one.
pub const READ_ONLY_TOOLS: [&str; 4] = ["file_read", "file_list", "file_search", "file_grep"];

/// The tools that read the web and change nothing: what a layer researching
/// the world outside the branches may add to [`READ_ONLY_TOOLS`]. A unit given
/// one runs with the network granted and nothing else
/// (`super::run::unit_grants`).
///
/// **Search, never fetch.** A unit runs unattended over whatever the branches
/// hold, and anyone who can push a branch writes text it reads. Given the
/// read-only tools and `web_fetch` together, an instruction planted in a file
/// could read the workspace and send it out in a URL to any public host.
/// `web_search` sends only a query, and only to the daemon's configured search
/// provider, so a unit has no channel to a host of the text's choosing.
pub const RESEARCH_TOOLS: [&str; 1] = ["web_search"];

/// The thinking levels a layer's `thinking_effort` dial may name — those
/// [`LayerSpec::think_mode`] steers by.
const EFFORTS: [&str; 5] = ["off", "quick", "balanced", "deep", "exhaustive"];

/// Refuse a layer naming any tool outside [`READ_ONLY_TOOLS`] and
/// [`RESEARCH_TOOLS`].
fn check_tools(layer: &str, tools: &[String]) -> anyhow::Result<()> {
    let allowed = |t: &str| READ_ONLY_TOOLS.contains(&t) || RESEARCH_TOOLS.contains(&t);
    match tools.iter().find(|t| !allowed(t.as_str())) {
        Some(t) => anyhow::bail!(
            "layer '{layer}': `{t}` is not a unit tool — a unit may call only \
             {READ_ONLY_TOOLS:?} and {RESEARCH_TOOLS:?}"
        ),
        None => Ok(()),
    }
}

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
    #[serde(default)]
    tools: Vec<String>,
    #[serde(default)]
    expires_days: Option<u32>,
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
    /// The tools a unit may call to check what it needs — from
    /// [`READ_ONLY_TOOLS`] and [`RESEARCH_TOOLS`]. Empty: the unit answers from
    /// its lineage and the layers below it alone.
    pub tools: Vec<String>,
    /// The layer's `thinking_effort` dial, as the selection names it.
    pub effort: String,
    /// How long each unit stands before it is read again, though nothing it
    /// was built on moved. `None`: a unit stands while what it was built on
    /// does.
    pub expiry: Option<Expiry>,
}

impl LayerSpec {
    /// The layer's spec, `None` when it declares no `ingest:` block.
    pub fn of(layer: &LayerSchema) -> anyhow::Result<Option<Self>> {
        let Some(value) = &layer.ingest else {
            return Ok(None);
        };
        let raw: Raw = serde_yaml::from_value(value.clone())
            .map_err(|e| anyhow::anyhow!("layer '{}': its `ingest:` block: {e}", layer.name))?;
        check_tools(&layer.name, &raw.tools)?;
        let effort = layer
            .dials
            .get("thinking_effort")
            .unwrap_or("balanced")
            .to_string();
        if !EFFORTS.contains(&effort.as_str()) {
            anyhow::bail!(
                "layer '{}': thinking_effort `{effort}` is none of {EFFORTS:?}",
                layer.name
            );
        }
        if raw.unit == UnitKind::File && raw.after.is_none() {
            anyhow::bail!(
                "layer '{}': a file unit builds on a layer below it — name it in `after:`",
                layer.name
            );
        }
        let expiry = match raw.expires_days {
            None => None,
            Some(days) => Some(Expiry::days(days).ok_or_else(|| {
                anyhow::anyhow!(
                    "layer '{}': `expires_days: 0` would expire every unit as it is built",
                    layer.name
                )
            })?),
        };
        Ok(Some(Self {
            layer: layer.name.clone(),
            unit: raw.unit,
            after: raw.after,
            prompt: raw.prompt,
            edges: raw.edges,
            tools: raw.tools,
            effort,
            expiry,
        }))
    }

    /// Whether a unit may reach the web — it was given a research tool.
    pub fn researches(&self) -> bool {
        self.tools
            .iter()
            .any(|t| RESEARCH_TOOLS.contains(&t.as_str()))
    }

    /// The tools mode whose call grammar a unit decodes under: the one whose
    /// catalog holds every tool it was given. Restricted's holds the
    /// read-only tools and no network tool, so a unit that researches decodes
    /// under Comprehensive's — its prompt still lists only its own tools, and
    /// a call of any other is refused unrun (`super::run::only_given`).
    pub fn tool_mode(&self) -> ToolMode {
        if self.researches() {
            ToolMode::Comprehensive
        } else {
            ToolMode::Restricted
        }
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
            tools: Vec::new(),
            expiry: None,
        }
    }

    /// A unit given a research tool reaches the web and decodes under the
    /// grammar that holds it; one given only read-only tools does neither.
    #[test]
    fn a_research_tool_takes_the_unit_to_the_network_grammar() {
        let mut s = spec(UnitKind::File);
        s.tools = vec!["file_read".into(), "file_grep".into()];
        assert!(!s.researches());
        assert_eq!(s.tool_mode(), ToolMode::Restricted);
        s.tools.push("web_search".into());
        assert!(s.researches());
        assert_eq!(s.tool_mode(), ToolMode::Comprehensive);
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

#[cfg(test)]
mod tool_tests {
    use super::*;
    use zend_tools::{registry, Capability};

    /// Every tool a unit may be given is a registered tool — a renamed one
    /// fails here rather than leaving units with a catalog naming nothing.
    #[test]
    fn every_unit_tool_is_registered() {
        for name in READ_ONLY_TOOLS.iter().chain(&RESEARCH_TOOLS) {
            assert!(registry::find(name).is_some(), "{name} is not a tool");
        }
    }

    /// **A research tool needs the network and nothing else**: the grant a
    /// researching unit runs under (`super::super::run::unit_grants`) is the
    /// network alone, so a research tool needing more would be refused.
    #[test]
    fn a_research_tool_needs_only_the_network() {
        for name in RESEARCH_TOOLS {
            let tool = registry::find(name).unwrap();
            assert_eq!(
                tool.requires,
                [Capability::Network],
                "{name} needs more than the network"
            );
        }
    }

    /// A layer naming a tool that writes is refused when the schema loads; one
    /// naming the read-only and research tools is not.
    #[test]
    fn a_layer_naming_a_writing_tool_is_refused() {
        let tools = |names: &[&str]| names.iter().map(|n| n.to_string()).collect::<Vec<_>>();
        assert!(check_tools("static_analysis", &tools(&["file_read", "file_edit"])).is_err());
        assert!(check_tools("security_review", &tools(&["web_search", "http_request"])).is_err());
        // A fetch would give a unit reading the workspace a channel to any host.
        assert!(check_tools("security_review", &tools(&["file_read", "web_fetch"])).is_err());
        assert!(check_tools("static_analysis", &tools(&READ_ONLY_TOOLS)).is_ok());
        assert!(check_tools("security_review", &tools(&RESEARCH_TOOLS)).is_ok());
        assert!(check_tools("static_analysis", &[]).is_ok());
    }
}
