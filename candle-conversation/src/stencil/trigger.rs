//! The trigger registry — tokens that start a stencil session.
//!
//! In free decode, each emitted token is checked here; a hit returns the tree to
//! enter at its root.  The `<tool_call>` case is a single special token, so this
//! is one hash lookup per free-decode token.

use std::collections::HashMap;
use std::sync::Arc;

use super::driver::StencilDriver;
use super::session::StencilSession;
use super::tree::StencilTree;
use super::vocab::TokenId;

/// Maps a trigger token to the tree it enters.
#[derive(Debug, Default, Clone)]
pub struct TriggerRegistry {
    by_token: HashMap<TokenId, Arc<StencilTree>>,
}

impl TriggerRegistry {
    pub fn new() -> Self {
        TriggerRegistry {
            by_token: HashMap::new(),
        }
    }

    /// Register `tree` to start when `token` is emitted in free decode.
    /// Replaces any tree already bound to `token` (last write wins).
    pub fn register(&mut self, token: TokenId, tree: Arc<StencilTree>) {
        self.by_token.insert(token, tree);
    }

    /// Return a copy of this registry with `token` bound to `tree`, replacing any
    /// existing binding for that token. The base is untouched — callers that
    /// share a base registry (e.g. the tool-call catalog) per-turn derive a fresh
    /// one for the turn's dial without mutating what other in-flight turns hold.
    ///
    /// **Atomic**: produces a complete registry in one value; no caller ever sees
    /// a half-updated map. **Idempotent**: `with_trigger(t, tree)` applied to a
    /// registry that already binds `t → tree` yields an identical registry, so
    /// re-deriving for the same dial replaces rather than accumulates.
    #[must_use]
    pub fn with_trigger(&self, token: TokenId, tree: Arc<StencilTree>) -> Self {
        let mut by_token = self.by_token.clone();
        by_token.insert(token, tree);
        TriggerRegistry { by_token }
    }

    /// Return a copy of this registry with any binding for `token` removed.
    /// `with_trigger`'s inverse — used to clear a steering trigger when the dial
    /// selects a mode that needs none (e.g. the empty-block leaf).
    #[must_use]
    pub fn without_trigger(&self, token: TokenId) -> Self {
        let mut by_token = self.by_token.clone();
        by_token.remove(&token);
        TriggerRegistry { by_token }
    }

    pub fn is_empty(&self) -> bool {
        self.by_token.is_empty()
    }

    /// If `token` is a trigger, return a fresh session at the tree's root.
    /// Called only in free decode (never inside an active session).
    pub fn on_token(&self, token: TokenId) -> Option<StencilSession> {
        self.by_token
            .get(&token)
            .map(|t| StencilSession::new(Arc::clone(t)))
    }

    /// If `token` is a trigger, return a fresh [`StencilDriver`] walking the
    /// triggered tree — the decode-loop entry point.  An empty registry never
    /// triggers, so a turn with no tools simply free-decodes.
    pub fn driver_for(&self, token: TokenId) -> Option<StencilDriver> {
        self.by_token
            .get(&token)
            .map(|t| StencilDriver::new(Arc::clone(t)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stencil::compile::compile;
    use crate::stencil::spec::{NodeSpec, TreeSpec};
    use crate::stencil::vocab::TestVocab;

    fn tiny_tree() -> Arc<StencilTree> {
        let mut s = TreeSpec::new("t");
        let end = s.push(NodeSpec::End);
        let st = s.push(NodeSpec::Static {
            text: "x".into(),
            next: end,
        });
        s.root = st;
        Arc::new(compile(&s, &TestVocab::new()).unwrap())
    }

    #[test]
    fn triggers_on_registered_token() {
        let mut reg = TriggerRegistry::new();
        assert!(reg.is_empty());
        reg.register(1000, tiny_tree());
        assert!(!reg.is_empty());
        assert!(reg.on_token(1000).is_some());
        assert!(reg.on_token(999).is_none());
    }

    #[test]
    fn with_trigger_is_atomic_and_idempotent() {
        // A base registry (e.g. the tool-call catalog) bound to one token.
        let base = TriggerRegistry::new().with_trigger(1000, tiny_tree());

        // Deriving a per-turn registry leaves the base untouched (atomic: callers
        // sharing `base` never observe the new binding).
        let a = base.with_trigger(151667, tiny_tree());
        assert!(base.on_token(151667).is_none());
        assert!(a.on_token(151667).is_some());
        assert!(a.on_token(1000).is_some()); // base binding carried through

        // Re-deriving for the same dial replaces, never accumulates (idempotent).
        let b = a.with_trigger(151667, tiny_tree());
        assert_eq!(a.by_token.len(), b.by_token.len());
        assert!(b.on_token(151667).is_some());

        // The inverse clears the steering trigger without touching the base one.
        let c = b.without_trigger(151667);
        assert!(c.on_token(151667).is_none());
        assert!(c.on_token(1000).is_some());
    }
}
