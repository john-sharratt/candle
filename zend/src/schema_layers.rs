//! zend's layers for `GET /v1/schema/layers` (`docs/shared_model_host.md`
//! §3.5): each layer's entry as the projection schema states it — the YAML
//! object, passed through unformatted — with the live fields beside it.
//! Formatting (a flex priority from `budget`, group strings) is the page's.

use std::fs;
use std::path::Path;

use host::substrate_views::LayerView;
use serde_json::{json, Map, Value};

use crate::session::PROJECTION_SCHEMA_TEMPLATE;

/// The `layers:` entries of the schema zend runs for `workspace` — its
/// `projection.yaml` when it has one, the embedded default otherwise — as
/// JSON objects, in declaration order.
pub fn raw_layers(workspace: &Path) -> anyhow::Result<Vec<Map<String, Value>>> {
    let text = fs::read_to_string(workspace.join("projection.yaml"))
        .unwrap_or_else(|_| PROJECTION_SCHEMA_TEMPLATE.to_string());
    parse_layers(&text)
}

fn parse_layers(text: &str) -> anyhow::Result<Vec<Map<String, Value>>> {
    let doc: Value = serde_yaml::from_str(text)?;
    let Some(layers) = doc.get("layers").and_then(Value::as_array) else {
        anyhow::bail!("the projection schema has no `layers:` list");
    };
    Ok(layers
        .iter()
        .filter_map(|l| l.as_object().cloned())
        .collect())
}

/// Each raw entry with the live fields of its layer: `conversation_count`,
/// `tokens`, `disabled` — or the entry alone for a layer the live view does
/// not list (none loaded yet).
pub fn with_live_fields(raw: Vec<Map<String, Value>>, live: &[LayerView]) -> Vec<Value> {
    raw.into_iter()
        .map(|mut entry| {
            let name = entry.get("name").and_then(Value::as_str).unwrap_or("");
            if let Some(l) = live.iter().find(|l| l.name == name) {
                entry.insert("conversation_count".into(), json!(l.conversation_count));
                entry.insert("tokens".into(), json!(l.tokens));
                entry.insert("disabled".into(), json!(l.disabled));
            }
            Value::Object(entry)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The embedded schema's layers come through as its YAML states them.
    #[test]
    fn the_embedded_schema_passes_its_layers_through() {
        let layers = parse_layers(PROJECTION_SCHEMA_TEMPLATE).unwrap();
        let names: Vec<&str> = layers
            .iter()
            .filter_map(|l| l.get("name").and_then(Value::as_str))
            .collect();
        assert!(names.contains(&"dialogue"), "{names:?}");
        let dialogue = layers
            .iter()
            .find(|l| l.get("name").and_then(Value::as_str) == Some("dialogue"))
            .unwrap();
        assert!(dialogue.contains_key("groups"), "unformatted: {dialogue:?}");
    }

    /// The live fields land beside the raw entry of the layer they describe,
    /// and a layer the live view does not list keeps its entry alone.
    #[test]
    fn live_fields_join_their_layer() {
        let raw = parse_layers("layers:\n  - name: a\n    window: 10\n  - name: b\n").unwrap();
        let live = vec![LayerView {
            name: "a".into(),
            description: String::new(),
            window: 10,
            priority: 1.0,
            groups: Vec::new(),
            dials: Vec::new(),
            conversation_count: 3,
            tokens: 900,
            disabled: false,
        }];
        assert_eq!(
            with_live_fields(raw, &live),
            vec![
                json!({"name": "a", "window": 10, "conversation_count": 3, "tokens": 900,
                       "disabled": false}),
                json!({"name": "b"}),
            ]
        );
    }

    #[test]
    fn a_schema_without_layers_is_refused() {
        assert!(parse_layers("system_prompt: []\n").is_err());
    }
}
