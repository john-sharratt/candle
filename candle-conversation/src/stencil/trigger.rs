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
    by_token: HashMap<TokenId, Binding>,
}

/// A trigger's tree, and whether entering it spends the trigger for the turn.
#[derive(Debug, Clone)]
struct Binding {
    tree: Arc<StencilTree>,
    once: bool,
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
        self.by_token.insert(token, Binding { tree, once: false });
    }

    /// [`Self::with_trigger`] for a trigger that fires at most once per turn:
    /// once its tree has been entered, the token is ordinary text for the rest
    /// of the turn — see [`Self::after_firing`].
    ///
    /// The thinking block is the case. It opens a turn; a `<think>` the model
    /// writes later is text — in a reply about think blocks, a quoted tag — and
    /// steering it opened a fresh block at every mention. Measured on a GUI turn
    /// whose answer quoted `<think></think>`: each quote became an injected
    /// empty block, and the model rewrote its sentence around it about twenty
    /// times in one 2383-token reply.
    #[must_use]
    pub fn with_once_trigger(&self, token: TokenId, tree: Arc<StencilTree>) -> Self {
        let mut by_token = self.by_token.clone();
        by_token.insert(token, Binding { tree, once: true });
        TriggerRegistry { by_token }
    }

    /// The registry for the rest of the turn once `token` has fired: without
    /// `token` when it fires once, `None` when firing it changes nothing.
    #[must_use]
    pub fn after_firing(&self, token: TokenId) -> Option<Self> {
        self.by_token
            .get(&token)
            .filter(|b| b.once)
            .map(|_| self.without_trigger(token))
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
        by_token.insert(token, Binding { tree, once: false });
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
            .map(|b| StencilSession::new(Arc::clone(&b.tree)))
    }

    /// If `token` is a trigger, return a fresh [`StencilDriver`] walking the
    /// triggered tree — the decode-loop entry point.  An empty registry never
    /// triggers, so a turn with no tools simply free-decodes.
    pub fn driver_for(&self, token: TokenId) -> Option<StencilDriver> {
        self.by_token
            .get(&token)
            .map(|b| StencilDriver::new(Arc::clone(&b.tree)))
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

    /// A once-trigger (the think block) is spent by firing; a plain one (the
    /// tool call) stays armed for every call the turn makes.
    #[test]
    fn a_once_trigger_is_spent_by_firing_and_the_rest_stay_armed() {
        const TOOL_CALL: TokenId = 1000;
        const THINK: TokenId = 151667;
        let turn = TriggerRegistry::new()
            .with_trigger(TOOL_CALL, tiny_tree())
            .with_once_trigger(THINK, tiny_tree());
        assert!(turn.driver_for(THINK).is_some(), "armed until it fires");

        let rest = turn
            .after_firing(THINK)
            .expect("firing a once-trigger spends it");
        assert!(
            rest.driver_for(THINK).is_none(),
            "text for the rest of the turn"
        );
        assert!(rest.driver_for(TOOL_CALL).is_some(), "the call stays armed");

        assert!(
            rest.after_firing(TOOL_CALL).is_none(),
            "firing a plain trigger changes nothing"
        );
        assert!(
            turn.after_firing(999).is_none(),
            "an unbound token changes nothing"
        );
    }
}
