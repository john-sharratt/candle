//! The compiled, immutable stencil tree.
//!
//! A flat arena of [`StencilNode`]s addressed by [`NodeId`], rooted at `root`.
//! Built once by [`compile`](super::compile::compile) and shared (`Arc`) across
//! decode sessions.

use super::terminator::Terminator;
use super::trie::TokenTrie;
use super::vocab::TokenId;

/// Arena index of a tree node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

/// One node of a compiled stencil tree.
#[derive(Debug, Clone)]
pub enum StencilNode {
    /// A fixed token run, prefilled atomically. The compiler guarantees no two
    /// `Static` nodes are adjacent (they are fused), so the run is maximal.
    Static { tokens: Vec<TokenId>, next: NodeId },
    /// A constrained choice — the sampler is masked to the trie frontier.
    /// Always ≥2 arms (single-arm branches are folded to `Static`).
    Branch { trie: TokenTrie },
    /// An unconstrained span ending at a terminator.
    FreeText(FreeTextSpan),
    /// Leave stencil mode.
    End,
}

/// A free-text span: a terminator plus optional EOS exit and span-scoped limits.
#[derive(Debug, Clone)]
pub struct FreeTextSpan {
    pub term: Terminator,
    /// When `true`, an EOS sample also ends the span (byte-terminator spans). For
    /// a token-closed span EOS is instead intercepted as a second close trigger
    /// (see `close_token`), so this is left `false` there.
    pub eos_ends: bool,
    pub limits: FreeTextLimits,
    /// When `Some(t)`, sampling token `t` closes the span (in addition to the
    /// terminator and EOS).  This is how a span ends on a delimiter *token*
    /// (e.g. Qwen3's `</think>`) rather than a byte pattern.
    pub close_token: Option<TokenId>,
    /// When `true` and the span closed via `close_token`, that close token is
    /// DROPPED (not committed) and the successor prefills in its place — the
    /// retry continuation that re-steers the thinking block.  When `false`, the
    /// close token is kept (committed normally).  Only meaningful with
    /// `close_token`.
    pub suppress_close: bool,
    pub next: NodeId,
}

/// Span-scoped EOS-style limits, mirroring `SamplingConfig`'s whole-turn ramp.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FreeTextLimits {
    /// Token count at which the close-token logit boost begins ramping. `None`
    /// disables the soft ramp.
    pub ramp_start: Option<u32>,
    /// Token count at which the ramp reaches full `boost`.
    pub ramp_len: u32,
    /// Maximum logit boost added to the close token at full ramp.
    pub boost: f32,
    /// Force-close unconditionally at this count. Always set (runaway guard).
    pub forced_after: u32,
}

impl FreeTextLimits {
    /// A plain JSON string value: a high runaway guard, no soft pressure (the
    /// model reliably emits its own closing quote).
    pub fn json_string() -> Self {
        FreeTextLimits {
            ramp_start: None,
            ramp_len: 0,
            boost: 0.0,
            forced_after: 512,
        }
    }

    /// Any JSON value (number/array/object): a generous runaway guard, since
    /// arrays and objects can be sizeable.
    pub fn json_value() -> Self {
        FreeTextLimits {
            ramp_start: None,
            ramp_len: 0,
            boost: 0.0,
            forced_after: 1024,
        }
    }

    /// A free-prose span: a soft close-token ramp over the back of the span plus
    /// the hard runaway guard, like a normal turn.
    pub fn prose(forced_after: u32) -> Self {
        let ramp_start = (forced_after as f32 * 0.6) as u32;
        FreeTextLimits {
            ramp_start: Some(ramp_start),
            ramp_len: forced_after,
            boost: 4.0,
            forced_after,
        }
    }

    /// A free-flowing thinking span with no soft ramp and a high runaway guard:
    /// the model closes the block (`</think>`) when it is ready.
    pub fn think_flow(forced_after: u32) -> Self {
        FreeTextLimits {
            ramp_start: None,
            ramp_len: 0,
            boost: 0.0,
            forced_after,
        }
    }

    /// The close-token logit boost after `emitted` span tokens (the soft ramp).
    pub fn boost_at(&self, emitted: u32) -> f32 {
        match self.ramp_start {
            Some(start) if emitted >= start && self.ramp_len > start => {
                let frac = (emitted - start) as f32 / (self.ramp_len - start) as f32;
                self.boost * frac.min(1.0)
            }
            _ => 0.0,
        }
    }
}

/// The compiled tree.
#[derive(Debug, Clone)]
pub struct StencilTree {
    nodes: Vec<StencilNode>,
    root: NodeId,
    eos: TokenId,
    fingerprint: u64,
    label: String,
    /// Tokens emitted to gracefully terminate the invocation if an
    /// out-of-grammar token is ever decoded (a failsafe — see the session's
    /// bail path).  Empty ⇒ just exit with no closing tokens.
    bail: Vec<TokenId>,
}

impl StencilTree {
    pub(crate) fn new(
        nodes: Vec<StencilNode>,
        root: NodeId,
        eos: TokenId,
        fingerprint: u64,
        label: String,
        bail: Vec<TokenId>,
    ) -> Self {
        StencilTree {
            nodes,
            root,
            eos,
            fingerprint,
            label,
            bail,
        }
    }

    pub fn root(&self) -> NodeId {
        self.root
    }
    pub fn eos(&self) -> TokenId {
        self.eos
    }
    /// The graceful-termination token sequence (the bail set).
    pub fn bail(&self) -> &[TokenId] {
        &self.bail
    }
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }
    pub fn label(&self) -> &str {
        &self.label
    }
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
    pub fn node(&self, id: NodeId) -> &StencilNode {
        &self.nodes[id.0 as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boost_ramp() {
        let l = FreeTextLimits {
            ramp_start: Some(10),
            ramp_len: 20,
            boost: 8.0,
            forced_after: 30,
        };
        assert_eq!(l.boost_at(5), 0.0); // before ramp
        assert_eq!(l.boost_at(10), 0.0); // at start
        assert_eq!(l.boost_at(15), 4.0); // halfway
        assert_eq!(l.boost_at(20), 8.0); // full
        assert_eq!(l.boost_at(25), 8.0); // clamped
    }

    #[test]
    fn no_ramp_is_zero() {
        let l = FreeTextLimits::json_string();
        assert_eq!(l.boost_at(0), 0.0);
        assert_eq!(l.boost_at(1000), 0.0);
        assert_eq!(l.forced_after, 512);
    }

    #[test]
    fn prose_preset_is_ordered() {
        let l = FreeTextLimits::prose(100);
        assert!(l.ramp_start.unwrap() < l.ramp_len);
        assert!(l.ramp_start.unwrap() < l.forced_after);
    }
}
