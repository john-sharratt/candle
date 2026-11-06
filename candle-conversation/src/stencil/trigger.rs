//! The trigger registry — tokens that start a stencil session.
//!
//! In free decode, each emitted token is checked here; a hit returns the tree to
//! enter at its root.  The `<tool_call>` case is a single special token, so this
//! is one hash lookup per free-decode token.

use std::collections::HashMap;
use std::sync::Arc;

use super::driver::StencilDriver;
use super::held_arms::HeldBranches;
use super::held_values::HeldValues;
use super::session::StencilSession;
use super::tree::StencilTree;
use super::vocab::TokenId;

/// Maps a trigger token to the tree it enters.
#[derive(Debug, Default, Clone)]
pub struct TriggerRegistry {
    by_token: HashMap<TokenId, Binding>,
    /// The tree an end token sampled in free decode hands over to, in place of
    /// ending the turn ([`Self::with_tail`]).
    tail: Option<Arc<StencilTree>>,
}

/// A trigger's tree, whether entering it spends the trigger for the turn, and
/// the branches a walk of it is held on (see [`TriggerRegistry::holding`]):
/// `held` once the turn has reasoned for `min_reasoning` tokens, `unreasoned`
/// before (see [`TriggerRegistry::holding_after_reasoning`]).
#[derive(Debug, Clone)]
struct Binding {
    tree: Arc<StencilTree>,
    once: bool,
    held: Option<Arc<HeldBranches>>,
    min_reasoning: u32,
    unreasoned: Option<Arc<HeldBranches>>,
    /// The values the walk's keyed fields are held to (see
    /// [`TriggerRegistry::holding_values`]).
    values: Option<Arc<HeldValues>>,
}

impl Binding {
    fn of(tree: Arc<StencilTree>, once: bool) -> Self {
        Binding {
            tree,
            once,
            held: None,
            min_reasoning: 0,
            unreasoned: None,
            values: None,
        }
    }
}

impl TriggerRegistry {
    pub fn new() -> Self {
        TriggerRegistry {
            by_token: HashMap::new(),
            tail: None,
        }
    }

    /// Register `tree` to start when `token` is emitted in free decode.
    /// Replaces any tree already bound to `token` (last write wins).
    pub fn register(&mut self, token: TokenId, tree: Arc<StencilTree>) {
        self.by_token.insert(token, Binding::of(tree, false));
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
        let mut out = self.clone();
        out.by_token.insert(token, Binding::of(tree, true));
        out
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
        let mut out = self.clone();
        out.by_token.insert(token, Binding::of(tree, false));
        out
    }

    /// A copy of this registry in which an end token sampled in free decode
    /// does not end the turn: it is dropped, and `tree` writes in its place
    /// and ends the turn itself. Once — the tree's own close is not free
    /// decode. A task's `Steps:` section is the case
    /// ([`compile_steps_tail`](super::compile_steps_tail)).
    #[must_use]
    pub fn with_tail(&self, tree: Arc<StencilTree>) -> Self {
        let mut out = self.clone();
        out.tail = Some(tree);
        out
    }

    /// A walk of the tail, when there is one.
    pub fn tail_driver(&self) -> Option<StencilDriver> {
        self.tail
            .as_ref()
            .map(|tree| StencilDriver::new(Arc::clone(tree)))
    }

    /// This registry without its tail: what the rest of a turn whose tail has
    /// fired decodes under.
    #[must_use]
    pub fn without_tail(&self) -> Self {
        let mut out = self.clone();
        out.tail = None;
        out
    }

    /// A copy of this registry in which every walk `token` starts is held on
    /// `held`'s branches ([`StencilDriver::hold_branches`]).
    ///
    /// The per-turn narrowing of a catalog compiled once: a turn given some of
    /// the catalog's tools holds every tool-name branch to theirs
    /// ([`held_branches`](super::held_branches)), so the shared tree is
    /// untouched and the turn can write no other. An unbound `token` leaves the
    /// registry as it is.
    #[must_use]
    pub fn holding(&self, token: TokenId, held: Arc<HeldBranches>) -> Self {
        self.holding_after_reasoning(token, held, 0, None)
    }

    /// [`Self::holding`] for a hold that needs reasoning behind it: a walk
    /// `token` starts once the turn has reasoned for at least `min_reasoning`
    /// tokens is held on `held`, and one started before that on `otherwise`
    /// (on nothing when `None`).
    ///
    /// A note of what the thinking concluded (`docs/zend_journal.md` §5.2) is
    /// the case: forced when there was reasoning to conclude from, and not
    /// when a call follows a few tokens of it.
    #[must_use]
    pub fn holding_after_reasoning(
        &self,
        token: TokenId,
        held: Arc<HeldBranches>,
        min_reasoning: u32,
        otherwise: Option<Arc<HeldBranches>>,
    ) -> Self {
        let mut out = self.clone();
        if let Some(binding) = out.by_token.get_mut(&token) {
            binding.held = Some(held);
            binding.min_reasoning = min_reasoning;
            binding.unreasoned = otherwise;
        }
        out
    }

    /// A copy of this registry in which every walk `token` starts holds its
    /// keyed fields to `values` ([`StencilDriver::hold_values`],
    /// `docs/stencil_tree.md` §9.2). An unbound `token` leaves the registry as
    /// it is.
    #[must_use]
    pub fn holding_values(&self, token: TokenId, values: Arc<HeldValues>) -> Self {
        let mut out = self.clone();
        if let Some(binding) = out.by_token.get_mut(&token) {
            binding.values = Some(values);
        }
        out
    }

    /// The tree `token` starts, when it is a trigger.
    pub fn tree_for(&self, token: TokenId) -> Option<&Arc<StencilTree>> {
        self.by_token.get(&token).map(|b| &b.tree)
    }

    /// Return a copy of this registry with any binding for `token` removed.
    /// `with_trigger`'s inverse — used to clear a steering trigger when the dial
    /// selects a mode that needs none (e.g. the empty-block leaf).
    #[must_use]
    pub fn without_trigger(&self, token: TokenId) -> Self {
        let mut out = self.clone();
        out.by_token.remove(&token);
        out
    }

    pub fn is_empty(&self) -> bool {
        self.by_token.is_empty() && self.tail.is_none()
    }

    /// Whether emitting `token` in free decode starts a walk.
    pub fn is_trigger(&self, token: TokenId) -> bool {
        self.by_token.contains_key(&token)
    }

    /// If `token` is a trigger, return a fresh session at the tree's root.
    /// Called only in free decode (never inside an active session).
    pub fn on_token(&self, token: TokenId) -> Option<StencilSession> {
        self.by_token
            .get(&token)
            .map(|b| StencilSession::new(Arc::clone(&b.tree)))
    }

    /// If `token` is a trigger, return a fresh [`StencilDriver`] walking the
    /// triggered tree, held as a turn that has reasoned enough is.  An empty
    /// registry never triggers, so a turn with no tools simply free-decodes.
    pub fn driver_for(&self, token: TokenId) -> Option<StencilDriver> {
        self.driver_after(token, u32::MAX)
    }

    /// [`Self::driver_for`] in a turn that has reasoned for `reasoned` tokens —
    /// the decode-loop entry point. The walk is held on the binding's `held`
    /// branches once `reasoned` reaches its `min_reasoning`, and on its
    /// `unreasoned` ones before.
    pub fn driver_after(&self, token: TokenId, reasoned: u32) -> Option<StencilDriver> {
        self.by_token.get(&token).map(|b| {
            let mut driver = StencilDriver::new(Arc::clone(&b.tree));
            let held = match reasoned >= b.min_reasoning {
                true => &b.held,
                false => &b.unreasoned,
            };
            if let Some(held) = held {
                driver.hold_branches(Arc::clone(held));
            }
            if let Some(values) = &b.values {
                driver.hold_values(Arc::clone(values));
            }
            driver
        })
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
