//! A turn's retrieval dials as a caller names them — the `projection`, `seed`
//! and `reprojection` fields of a chat request, and the same arguments of the
//! MCP tools that start a turn — and their mapping onto the turn's selection
//! (`candle_conversation::projection` turn dials).
//!
//! Every dial is optional and an absent one is the production default: full
//! projection, seeds on, and the daemon's configured reprojection.

use candle_conversation::{ProjectionLevel, ReprojectionMode, SelectionState};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// `projection`: how much of the retrieved memory the turn's slot takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectionDial {
    /// Every kind provenance selects from.
    Full,
    /// The journals alone.
    Limited,
    /// The tool catalog alone — and so nothing, with tools off.
    Minimal,
}

/// `reprojection`: when the decode re-selects its working set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReprojectionDial {
    /// At every boundary, inside the think block as in the answer.
    Full,
    /// Not inside the think block: once at its close, and through the answer.
    Held,
    /// Only at the turn's opening.
    Opening,
}

/// The three dials a turn may name.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetrievalDials {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection: Option<ProjectionDial>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reprojection: Option<ReprojectionDial>,
}

impl ProjectionDial {
    fn level(self) -> ProjectionLevel {
        match self {
            Self::Full => ProjectionLevel::Full,
            Self::Limited => ProjectionLevel::Limited,
            Self::Minimal => ProjectionLevel::Minimal,
        }
    }
}

impl ReprojectionDial {
    fn mode(self) -> ReprojectionMode {
        match self {
            Self::Full => ReprojectionMode::Full,
            Self::Held => ReprojectionMode::Held,
            Self::Opening => ReprojectionMode::Opening,
        }
    }
}

impl RetrievalDials {
    /// Set every dial this names on `selection`; an unnamed one stays unset,
    /// and reads as the production default.
    pub fn apply(&self, selection: &mut SelectionState) {
        if let Some(p) = self.projection {
            selection.set_projection_level(p.level());
        }
        if let Some(s) = self.seed {
            selection.set_seeds(s);
        }
        if let Some(r) = self.reprojection {
            selection.set_reprojection_mode(r.mode());
        }
    }

    /// The dials an MCP call's arguments name: `projection`, `seed` and
    /// `reprojection`, each optional. A value of the wrong shape or an unknown
    /// name is refused, never taken for the default.
    pub fn of_args(args: &Value) -> Result<Self, String> {
        let pick = |name: &str| args.get(name).filter(|v| !v.is_null());
        let projection = pick("projection")
            .map(|v| serde_json::from_value(v.clone()))
            .transpose()
            .map_err(|_| "`projection` is `full`, `limited` or `minimal`".to_string())?;
        let seed = match pick("seed") {
            None => None,
            Some(Value::Bool(b)) => Some(*b),
            Some(_) => return Err("`seed` is `true` or `false`".into()),
        };
        let reprojection = pick("reprojection")
            .map(|v| serde_json::from_value(v.clone()))
            .transpose()
            .map_err(|_| "`reprojection` is `full`, `held` or `opening`".to_string())?;
        Ok(Self {
            projection,
            seed,
            reprojection,
        })
    }

    /// `definition`, an MCP tool's, with the three arguments added to its
    /// `inputSchema`'s properties.
    pub fn with_schema(mut definition: Value) -> Value {
        if let Some(properties) = definition
            .pointer_mut("/inputSchema/properties")
            .and_then(Value::as_object_mut)
        {
            for (name, schema) in Self::schema_properties() {
                properties.insert(name.to_string(), schema);
            }
        }
        definition
    }

    /// The JSON schema properties the three arguments take, for an MCP tool's
    /// `inputSchema`.
    fn schema_properties() -> [(&'static str, Value); 3] {
        [
            (
                "projection",
                serde_json::json!({
                    "type": "string",
                    "enum": ["full", "limited", "minimal"],
                    "description": "How much of the memory the turn is shown: `full` (the default) everything provenance selects, `limited` the journals alone, `minimal` none of it — the tool catalog only, so nothing with tools off. Use a lower level to keep a task to what it is given."
                }),
            ),
            (
                "seed",
                serde_json::json!({
                    "type": "boolean",
                    "description": "Whether the knowledge base holds the repositories' root listings and anchor documents — README, ARCHITECTURE, AGENTS, CLAUDE — on every turn (default true)."
                }),
            ),
            (
                "reprojection",
                serde_json::json!({
                    "type": "string",
                    "enum": ["full", "held", "opening"],
                    "description": "When the turn re-selects what it is shown: `full` (the default) as it reasons and answers, `held` not while it thinks, `opening` only once at the start."
                }),
            ),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn no_dial_sets_nothing() {
        let mut s = SelectionState::new();
        RetrievalDials::default().apply(&mut s);
        assert_eq!(s, SelectionState::new());
    }

    #[test]
    fn each_dial_lands_on_the_selection() {
        let mut s = SelectionState::new();
        RetrievalDials {
            projection: Some(ProjectionDial::Minimal),
            seed: Some(false),
            reprojection: Some(ReprojectionDial::Held),
        }
        .apply(&mut s);
        assert_eq!(s.projection_level(), ProjectionLevel::Minimal);
        assert!(!s.seeds());
        assert_eq!(s.reprojection_mode(), Some(ReprojectionMode::Held));
    }

    #[test]
    fn a_request_body_reads_its_dials() {
        let d: RetrievalDials = serde_json::from_value(json!({
            "projection": "limited", "seed": true, "reprojection": "opening"
        }))
        .unwrap();
        assert_eq!(d.projection, Some(ProjectionDial::Limited));
        assert_eq!(d.seed, Some(true));
        assert_eq!(d.reprojection, Some(ReprojectionDial::Opening));
        assert!(serde_json::from_value::<RetrievalDials>(json!({"projection": "some"})).is_err());
    }

    #[test]
    fn mcp_arguments_read_their_dials_or_are_refused() {
        assert_eq!(
            RetrievalDials::of_args(&json!({"text": "hi"})),
            Ok(RetrievalDials::default())
        );
        assert_eq!(
            RetrievalDials::of_args(&json!({"projection": "minimal", "seed": false})),
            Ok(RetrievalDials {
                projection: Some(ProjectionDial::Minimal),
                seed: Some(false),
                reprojection: None,
            })
        );
        assert!(RetrievalDials::of_args(&json!({"projection": "none"})).is_err());
        assert!(RetrievalDials::of_args(&json!({"seed": "no"})).is_err());
        assert!(RetrievalDials::of_args(&json!({"reprojection": 3})).is_err());
    }

    #[test]
    fn a_tool_definition_gains_the_three_arguments() {
        let d = RetrievalDials::with_schema(json!({
            "name": "t",
            "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}
        }));
        let props = d["inputSchema"]["properties"].as_object().unwrap();
        let mut names: Vec<&str> = props.keys().map(String::as_str).collect();
        names.sort_unstable();
        assert_eq!(names, ["projection", "reprojection", "seed", "text"]);
        assert_eq!(
            props["projection"]["enum"],
            json!(["full", "limited", "minimal"])
        );
        assert_eq!(
            props["reprojection"]["enum"],
            json!(["full", "held", "opening"])
        );
        assert_eq!(props["seed"]["type"], json!("boolean"));
    }

    /// A worker record kept before the dials existed reads as no dials.
    #[test]
    fn a_record_without_dials_reads_as_the_default() {
        let d: RetrievalDials = serde_json::from_value(json!({})).unwrap();
        assert_eq!(d, RetrievalDials::default());
        assert_eq!(serde_json::to_value(d).unwrap(), json!({}));
    }
}
