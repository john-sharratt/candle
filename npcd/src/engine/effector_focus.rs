//! The effector focus: which resource's schema types a character's next `invoke`.
//!
//! A character that `query`s a resource gets the resource's `OPTIONS` schema back
//! inline, and the same schema is armed as that character's *focus*. On its next
//! turn the whole-turn grammar splices a sub-stencil compiled from the focus's
//! body schema onto the `invoke` branch, so the body it writes can only be a body
//! the resource accepts.
//!
//! Three pieces, each cached on its own key so none of them widens another:
//!
//! - [`FocusTable`] — per character, the current [`Focus`]. Never part of the
//!   `(Deliberation, Within)` frame key: a per-character, per-resource fact in
//!   that key would collapse the frame cache's hit rate.
//! - [`InvokeStencils`] — the body sub-stencil, keyed `(resource, fingerprint)`,
//!   so the same schema compiles once however many characters are focused on it.
//! - [`body_schema_from_options`] / [`resource_id`] — read a focus out of an
//!   `OPTIONS` answer.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use candle_conversation::stencil::{compile_invoke_body_tree, ToolSpec, TreeSpec};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// A character's armed resource: its id, the schema its `invoke` body must
/// match, and a fingerprint of that schema.
#[derive(Debug, Clone, PartialEq)]
pub struct Focus {
    pub resource: String,
    pub body_schema: Value,
    pub fingerprint: String,
}

impl Focus {
    pub fn new(resource: &str, body_schema: Value) -> Self {
        let fingerprint = fingerprint(&body_schema);
        Self {
            resource: resource.to_string(),
            body_schema,
            fingerprint,
        }
    }
}

/// SHA-256 of the schema's compact JSON, in lowercase hex. Two schemas that
/// differ in any field, enum member or `required` entry never share a
/// compiled sub-stencil.
fn fingerprint(schema: &Value) -> String {
    let digest = Sha256::digest(schema.to_string().as_bytes());
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// Each character's current focus.
#[derive(Default)]
pub struct FocusTable {
    foci: Mutex<HashMap<u64, Focus>>,
}

impl FocusTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&self, npc_id: u64, focus: Focus) {
        self.foci.lock().unwrap().insert(npc_id, focus);
    }

    pub fn clear(&self, npc_id: u64) {
        self.foci.lock().unwrap().remove(&npc_id);
    }

    pub fn get(&self, npc_id: u64) -> Option<Focus> {
        self.foci.lock().unwrap().get(&npc_id).cloned()
    }
}

/// The compiled `invoke`-body sub-stencils, keyed `(resource, fingerprint)`.
#[derive(Default)]
pub struct InvokeStencils {
    specs: Mutex<HashMap<(String, String), Arc<TreeSpec>>>,
}

impl InvokeStencils {
    pub fn new() -> Self {
        Self::default()
    }

    /// The sub-stencil for a focus, compiled on first use. `None` when the
    /// schema will not compile, which the caller answers with a free-JSON body.
    pub fn get_or_build(&self, focus: &Focus) -> Option<Arc<TreeSpec>> {
        let key = (focus.resource.clone(), focus.fingerprint.clone());
        if let Some(spec) = self.specs.lock().unwrap().get(&key) {
            return Some(Arc::clone(spec));
        }
        let params = ToolSpec::from_json_schema(&focus.resource, &focus.body_schema).params;
        let spec = match compile_invoke_body_tree(&params) {
            Ok(spec) => Arc::new(spec),
            Err(e) => {
                tracing::warn!(
                    "the invoke body of {} would not compile: {e:#}",
                    focus.resource
                );
                return None;
            }
        };
        // Last writer wins: the key determines the contents.
        self.specs.lock().unwrap().insert(key, Arc::clone(&spec));
        Some(spec)
    }
}

/// The schema an `invoke` body must match, from a resource's `OPTIONS` answer:
/// `methods.POST.body`. `None` when the resource has no POST — a thing whose
/// verbs are listed rather than typed, or a read-only resource.
pub fn body_schema_from_options(options: &Value) -> Option<Value> {
    let body = options.get("methods")?.get("POST")?.get("body")?;
    body.is_object().then(|| body.clone())
}

/// The resource a schema belongs to: the `id` the `OPTIONS` answer names, or the
/// path that was asked when it names none.
pub fn resource_id(options: &Value, path: &str) -> String {
    options
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .unwrap_or(path)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn floor_body() -> Value {
        json!({
            "type": "object",
            "properties": { "floor": { "type": "string", "enum": ["vault", "annex"] } },
            "required": ["floor"],
        })
    }

    #[test]
    fn the_body_schema_is_the_post_body() {
        let options = json!({
            "id": "call",
            "methods": { "POST": { "summary": "s", "body": floor_body() } },
        });
        assert_eq!(body_schema_from_options(&options), Some(floor_body()));
    }

    #[test]
    fn a_resource_without_a_post_arms_nothing() {
        let get_only = json!({ "id": "x", "methods": { "GET": { "returns": "state" } } });
        assert_eq!(body_schema_from_options(&get_only), None);
        assert_eq!(body_schema_from_options(&json!({ "id": "x" })), None);
        let no_body = json!({ "methods": { "POST": { "summary": "s" } } });
        assert_eq!(body_schema_from_options(&no_body), None);
    }

    #[test]
    fn the_resource_is_the_named_id_else_the_path() {
        assert_eq!(resource_id(&json!({ "id": "order-table~0" }), "/p"), "order-table~0");
        assert_eq!(resource_id(&json!({}), "/command/x"), "/command/x");
        assert_eq!(resource_id(&json!({ "id": "" }), "/command/x"), "/command/x");
    }

    #[test]
    fn the_fingerprint_follows_the_schema() {
        let a = Focus::new("r", floor_body());
        let same = Focus::new("r", floor_body());
        let other = Focus::new(
            "r",
            json!({ "type": "object", "properties": { "floor": { "type": "string" } } }),
        );
        assert_eq!(a.fingerprint, same.fingerprint);
        assert_ne!(a.fingerprint, other.fingerprint);
        assert_eq!(a.fingerprint.len(), 64);
        assert_eq!(
            fingerprint(&json!({})),
            "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
        );
    }

    #[test]
    fn the_table_arms_replaces_and_clears_per_character() {
        let table = FocusTable::new();
        assert!(table.get(1).is_none());
        table.set(1, Focus::new("a", floor_body()));
        table.set(2, Focus::new("b", floor_body()));
        table.set(1, Focus::new("c", floor_body()));
        assert_eq!(table.get(1).unwrap().resource, "c");
        assert_eq!(table.get(2).unwrap().resource, "b");
        table.clear(1);
        assert!(table.get(1).is_none());
        assert!(table.get(2).is_some());
    }

    #[test]
    fn one_schema_compiles_once_and_is_shared() {
        let stencils = InvokeStencils::new();
        let focus = Focus::new("call", floor_body());
        let first = stencils.get_or_build(&focus).expect("compiles");
        let second = stencils.get_or_build(&focus).expect("cached");
        assert!(Arc::ptr_eq(&first, &second));
        let other = stencils
            .get_or_build(&Focus::new("use", floor_body()))
            .expect("compiles");
        assert!(!Arc::ptr_eq(&first, &other), "keyed by resource too");
    }
}
