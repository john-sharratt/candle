//! What each backend last listed for each caller (`docs/mcp_app.md` §3).
//!
//! A client does not ask for the tools again for every conversation: claude.ai
//! kept the list it got while npcd restarted (2026-10-27, 07:38) and a chat
//! half an hour later still had no image tools. So a backend that cannot be
//! listed is listed as it last was, and a call to one of its tools meanwhile
//! answers that the backend is down — which says to try again, where a tool
//! missing from the list says nothing.
//!
//! Kept per caller, because a backend lists each person the tools they may
//! call.

use std::collections::HashMap;
use std::sync::RwLock;

use serde_json::Value;

/// The tools each backend last listed for each caller, by caller key
/// (`Caller::key`) and backend index.
#[derive(Default)]
pub struct LastListed(RwLock<HashMap<(String, usize), Vec<Value>>>);

impl LastListed {
    /// `tools` is what `backend` listed for `who` just now.
    pub fn keep(&self, who: &str, backend: usize, tools: &[Value]) {
        self.0
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert((who.to_string(), backend), tools.to_vec());
    }

    /// What `backend` last listed for `who`; `None` when it never has.
    pub fn recall(&self, who: &str, backend: usize) -> Option<Vec<Value>> {
        self.0
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(&(who.to_string(), backend))
            .cloned()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// **A backend down is listed as it last was**, for the caller it was
    /// listed for, and the newest listing is the one recalled.
    #[test]
    fn a_backend_is_recalled_as_it_last_listed_for_that_caller() {
        let last = LastListed::default();
        assert_eq!(last.recall("a", 1), None);

        last.keep("a", 1, &[json!({"name": "image_generate"})]);
        last.keep("a", 1, &[json!({"name": "image_edit"})]);
        last.keep("b", 1, &[json!({"name": "bot_list"})]);

        assert_eq!(
            last.recall("a", 1),
            Some(vec![json!({"name": "image_edit"})])
        );
        assert_eq!(last.recall("b", 1), Some(vec![json!({"name": "bot_list"})]));
        assert_eq!(last.recall("a", 0), None);
    }
}
