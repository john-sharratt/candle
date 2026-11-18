//! The queues parked on keep while their processes restart
//! (`docs/build_agent.md` §6). An agent about to stop posts its queue,
//! `POST /park`, with its node in the body; its next process takes it back,
//! `POST /unpark {node}`, and keep lets it go as it answers. keep holds a park
//! in memory alone, and refuses to exit while it holds one unless forced.

use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard};

use serde::Deserialize;
use serde_json::Value;

static PARKS: Mutex<BTreeMap<String, Value>> = Mutex::new(BTreeMap::new());

fn parks() -> MutexGuard<'static, BTreeMap<String, Value>> {
    PARKS.lock().unwrap_or_else(|e| e.into_inner())
}

#[derive(Deserialize)]
struct Node {
    node: String,
}

/// Keep the park `body`, a queue under its node, in place of any it had.
pub fn park(body: &str) -> Result<String, String> {
    let park: Value = serde_json::from_str(body).map_err(|e| format!("a park is JSON: {e}"))?;
    let Node { node } =
        serde_json::from_value(park.clone()).map_err(|e| format!("a park names its node: {e}"))?;
    parks().insert(node.clone(), park);
    Ok(node)
}

/// The park of the node `body` names, let go as it is answered: JSON `null`
/// when keep holds none.
pub fn unpark(body: &str) -> Result<String, String> {
    let Node { node } =
        serde_json::from_str(body).map_err(|e| format!("an unpark is {{node}}: {e}"))?;
    Ok(parks().remove(&node).unwrap_or(Value::Null).to_string())
}

/// The nodes whose queues keep holds.
pub fn held() -> Vec<String> {
    parks().keys().cloned().collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// **A park is answered once, to its node, and replaced by a later one.**
    #[test]
    fn a_park_is_taken_back_once_by_its_node() {
        let first = json!({"node": "00000000000000aa", "runs": [{"id": "A"}], "ended": []});
        let later = json!({"node": "00000000000000aa", "runs": [], "ended": []});
        assert_eq!(park(&first.to_string()), Ok("00000000000000aa".into()));
        assert_eq!(park(&later.to_string()), Ok("00000000000000aa".into()));
        assert!(held().contains(&"00000000000000aa".to_string()));
        assert_eq!(
            unpark(r#"{"node": "00000000000000aa"}"#).unwrap(),
            later.to_string()
        );
        assert_eq!(unpark(r#"{"node": "00000000000000aa"}"#).unwrap(), "null");
        assert!(!held().contains(&"00000000000000aa".to_string()));
        assert!(park(r#"{"runs": []}"#).is_err());
        assert!(unpark("x").is_err());
    }
}
