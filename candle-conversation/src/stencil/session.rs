//! The runtime walker — a cursor into one tree, no side buffers.
//!
//! `next_action()` says what to do at the current cursor (prefill a static run,
//! mask the next decode to a branch frontier, run a free decode, or exit) and
//! advances past static structure atomically.  `observe(token, bytes)` consumes
//! a decoded token and advances a branch or free-text span.

use std::sync::Arc;

use super::error::WalkError;
use super::mask::AllowedSet;
use super::terminator::{Feed, TerminatorState};
use super::tree::{FreeTextSpan, NodeId, StencilNode, StencilTree};
use super::trie::{Step, TokenTrie, TrieNodeId};
use super::vocab::TokenId;

/// What the scheduler should do next for a sequence in a session.
#[derive(Debug, Clone, PartialEq)]
pub enum StencilAction {
    /// Prefill this static token run (one forward pass, no sampling), then call
    /// `next_action` again.
    Prefill(Vec<TokenId>),
    /// The next decode is masked to this allowed set (a branch frontier).
    MaskedDecode(AllowedSet),
    /// The next decode is free (normal decode — EOS and any close token are
    /// intercepted by the session's `observe`, never banned).  `close_boost` is
    /// added to the span's close token (the soft ramp).
    FreeDecode { close_boost: f32 },
    /// Stencil finished — resume normal decode.
    Exit,
}

/// What `observe` did with a decoded token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Observe {
    /// Still mid-branch or mid-span.
    Continue,
    /// A branch arm completed.
    ArmComplete,
    /// A free-text span closed at its terminator.  `leftover` bytes of the
    /// closing token belong to the next node (a healing signal; 0 = clean).
    SpanClosed { leftover: usize },
    /// A free-text span closed because its `close_token` was sampled, and the
    /// span keeps that token: it is committed normally, then the cursor advances.
    TokenClosedKeep,
    /// A free-text span closed because its `close_token` was sampled, and the
    /// span suppresses it: the just-sampled close token is DROPPED (not
    /// committed) and the cursor advances so the successor prefills the
    /// continuation in its place (the steering retry).
    TokenClosedDrop,
    /// A free-text span hit its hard limit and was force-closed (the integration
    /// injects the canonical close tokens).
    SpanForcedClosed,
    /// A free-text span ended via an EOS sample (`eos_ends`).
    SpanEos,
    /// An out-of-grammar token was decoded (it escaped the mask).  The session
    /// logged it and entered the bail failsafe — the next actions emit the
    /// tree's bail tokens and exit.
    Bailed,
}

enum Cursor {
    At(NodeId),
    InBranch {
        node: NodeId,
        pos: TrieNodeId,
    },
    InFreeText {
        node: NodeId,
        term: TerminatorState,
        emitted: u32,
    },
    /// The failsafe fired — emit the tree's bail tokens, then finish.
    Bailing,
    Done,
}

/// A walk of one tree, attached to a decoding sequence.
pub struct StencilSession {
    tree: Arc<StencilTree>,
    cursor: Cursor,
    /// A lookahead terminator's delimiter, decoded but belonging to the next
    /// node — applied on the following `next_action` (§ push-back).
    pushback: Option<TokenId>,
}

impl StencilSession {
    /// Begin a session at the tree's root.
    pub fn new(tree: Arc<StencilTree>) -> Self {
        let root = tree.root();
        StencilSession {
            tree,
            cursor: Cursor::At(root),
            pushback: None,
        }
    }

    pub fn is_done(&self) -> bool {
        matches!(self.cursor, Cursor::Done)
    }

    pub fn tree(&self) -> &StencilTree {
        &self.tree
    }

    /// Whether the cursor sits in a free-text span where a sampled close ends
    /// the whole block (a TERMINAL span) — the only place the sampler's
    /// hard-cap closing-statement script may play.
    ///
    /// Every think span suppresses its close (the tree injects the real
    /// `</think>` as a static), so suppression alone doesn't distinguish the
    /// cases — what does is the walk AFTER the span: a chain of statics
    /// running to `End` means the block is closing here, while any further
    /// free/branch content means the steering re-opens reasoning (a
    /// deep/exhaustive "But wait, " continuation) and the injected
    /// continuation phrase — not a closing statement — is the bridge.
    ///
    /// Anywhere else — a continuation span, a span that consumes its own
    /// close token (tool-call values), or a cursor not in free text at all
    /// (static prefill, a branch decision) — this is false and a forced close
    /// stays bare.
    pub fn in_terminal_close_span(&self) -> bool {
        let Cursor::InFreeText { node, .. } = self.cursor else {
            return false;
        };
        let StencilNode::FreeText(span) = self.tree.node(node) else {
            return false;
        };
        if !span.suppress_close {
            return false;
        }
        let mut cur = span.next;
        loop {
            match self.tree.node(cur) {
                StencilNode::Static { next, .. } => cur = *next,
                StencilNode::End => return true,
                _ => return false,
            }
        }
    }

    /// What to do at the current cursor.  For static structure this advances the
    /// cursor and returns `Prefill`; for a branch/free-text it leaves the cursor
    /// awaiting a decoded token.
    pub fn next_action(&mut self) -> StencilAction {
        if let Some(tok) = self.pushback.take() {
            return self.apply_pushback(tok);
        }
        match self.cursor {
            Cursor::At(node) => match self.tree.node(node) {
                StencilNode::Static { tokens, next } => {
                    let toks = tokens.clone();
                    self.cursor = Cursor::At(*next);
                    StencilAction::Prefill(toks)
                }
                StencilNode::Branch { trie } => {
                    let pos = trie.root();
                    let set = AllowedSet::from_tokens(trie.frontier(pos));
                    self.cursor = Cursor::InBranch { node, pos };
                    StencilAction::MaskedDecode(set)
                }
                StencilNode::FreeText(span) => {
                    let action = Self::free_decode_action(span, 0);
                    self.cursor = Cursor::InFreeText {
                        node,
                        term: span.term.start(),
                        emitted: 0,
                    };
                    action
                }
                StencilNode::End => {
                    self.cursor = Cursor::Done;
                    StencilAction::Exit
                }
            },
            Cursor::InBranch { node, pos } => {
                let trie = self.branch_trie(node);
                StencilAction::MaskedDecode(AllowedSet::from_tokens(trie.frontier(pos)))
            }
            Cursor::InFreeText { node, emitted, .. } => {
                let span = self.free_span(node);
                Self::free_decode_action(span, emitted)
            }
            Cursor::Bailing => {
                self.cursor = Cursor::Done;
                let bail = self.tree.bail();
                if bail.is_empty() {
                    StencilAction::Exit
                } else {
                    StencilAction::Prefill(bail.to_vec())
                }
            }
            Cursor::Done => StencilAction::Exit,
        }
    }

    /// The free-decode action for `span` given its running `emitted` count.
    /// Decode is normal — EOS and any close token are intercepted by `observe`,
    /// never banned — so the only per-step parameter is the soft close ramp.
    fn free_decode_action(span: &FreeTextSpan, emitted: u32) -> StencilAction {
        StencilAction::FreeDecode {
            close_boost: span.limits.boost_at(emitted),
        }
    }

    /// Enter the bail failsafe: log the out-of-grammar token and arrange for the
    /// next action(s) to emit the tree's bail tokens and exit.
    fn bail(&mut self, token: TokenId, context: &str) {
        tracing::debug!(
            target: "candle_conversation::stencil",
            tree = self.tree.label(),
            token,
            context,
            "stencil: out-of-grammar token decoded — bailing (emitting {} bail token(s))",
            self.tree.bail().len(),
        );
        self.cursor = Cursor::Bailing;
    }

    /// Apply a pushed-back delimiter token to the current node (cursor is
    /// `At(next)`).  For a `Static`, the delimiter is the node's first token
    /// (already decoded), so prefill the rest; for a `Branch`, the delimiter
    /// drives the choice.
    fn apply_pushback(&mut self, tok: TokenId) -> StencilAction {
        let node = match self.cursor {
            Cursor::At(n) => n,
            _ => unreachable!("pushback is only set with cursor At(next)"),
        };
        // Snapshot under one immutable borrow, then mutate.
        enum Out {
            Static {
                rest: Vec<TokenId>,
                next: NodeId,
            },
            BranchDescend {
                node: NodeId,
                pos: TrieNodeId,
                frontier: Vec<TokenId>,
            },
            BranchAccept(NodeId),
            Bad,
        }
        let out = match self.tree.node(node) {
            // The delimiter must be the static's first token for the rest to
            // follow cleanly.  With a real tokenizer the model's delimiter token
            // can differ from the successor's canonical first token (e.g. it
            // emits `}` where the successor opens `}}`); rather than silently
            // miscount bytes, treat that as out-of-grammar and bail.
            StencilNode::Static { tokens, next } if tokens.first() == Some(&tok) => Out::Static {
                rest: tokens.iter().skip(1).copied().collect(),
                next: *next,
            },
            StencilNode::Branch { trie } => match trie.step(trie.root(), tok) {
                Some(Step::Descend(p)) => Out::BranchDescend {
                    node,
                    pos: p,
                    frontier: trie.frontier(p),
                },
                Some(Step::Accept(n)) => Out::BranchAccept(n),
                None => Out::Bad,
            },
            // Static with a non-matching first token, or a free-text/end node:
            // the delimiter isn't legal here.
            StencilNode::Static { .. } | StencilNode::FreeText(_) | StencilNode::End => Out::Bad,
        };
        match out {
            Out::Static { rest, next } => {
                self.cursor = Cursor::At(next);
                if rest.is_empty() {
                    self.next_action()
                } else {
                    StencilAction::Prefill(rest)
                }
            }
            Out::BranchDescend {
                node,
                pos,
                frontier,
            } => {
                self.cursor = Cursor::InBranch { node, pos };
                StencilAction::MaskedDecode(AllowedSet::from_tokens(frontier))
            }
            Out::BranchAccept(n) => {
                self.cursor = Cursor::At(n);
                self.next_action()
            }
            // The pushed-back delimiter isn't legal here — bail gracefully.
            Out::Bad => {
                self.bail(tok, "pushback");
                self.next_action()
            }
        }
    }

    /// Consume a decoded token.  `bytes` are the token's decoded bytes (only
    /// used in a free-text span).
    pub fn observe(&mut self, token: TokenId, bytes: &[u8]) -> Result<Observe, WalkError> {
        match std::mem::replace(&mut self.cursor, Cursor::Done) {
            Cursor::InBranch { node, pos } => {
                let step = self.branch_trie(node).step(pos, token);
                match step {
                    Some(Step::Descend(p)) => {
                        self.cursor = Cursor::InBranch { node, pos: p };
                        Ok(Observe::Continue)
                    }
                    Some(Step::Accept(next)) => {
                        self.cursor = Cursor::At(next);
                        Ok(Observe::ArmComplete)
                    }
                    None => {
                        // The decoded token isn't a legal branch edge — it
                        // escaped the mask.  Bail instead of erroring.
                        self.bail(token, "branch");
                        Ok(Observe::Bailed)
                    }
                }
            }
            Cursor::InFreeText {
                node,
                mut term,
                emitted,
            } => {
                let span = self.free_span(node).clone();
                let emitted = emitted + 1;
                if span.eos_ends && token == self.tree.eos() {
                    self.cursor = Cursor::At(span.next);
                    return Ok(Observe::SpanEos);
                }
                // A close *token* ends the span before any byte terminator runs.
                // A token-closed span closes on EITHER its close token OR EOS —
                // both are intercepted by this normal decode (never banned) and
                // replaced.  `suppress_close` drops the closing token (the successor
                // prefills the continuation in its place); otherwise it is kept
                // (committed).
                if let Some(ct) = span.close_token {
                    if token == ct || token == self.tree.eos() {
                        self.cursor = Cursor::At(span.next);
                        return Ok(if span.suppress_close {
                            Observe::TokenClosedDrop
                        } else {
                            Observe::TokenClosedKeep
                        });
                    }
                }
                match term.feed(bytes) {
                    Feed::Close { consumed } => {
                        self.cursor = Cursor::At(span.next);
                        // A lookahead terminator's delimiter belongs to the next
                        // node.  When it is its own token (consumed == 0, the
                        // byte-level / clean case), push it back so the next node
                        // consumes it instead of re-emitting it.
                        if span.term.is_lookahead() && consumed == 0 {
                            self.pushback = Some(token);
                        }
                        Ok(Observe::SpanClosed {
                            leftover: bytes.len() - consumed,
                        })
                    }
                    Feed::Continue => {
                        if emitted >= span.limits.forced_after {
                            self.cursor = Cursor::At(span.next);
                            Ok(Observe::SpanForcedClosed)
                        } else {
                            self.cursor = Cursor::InFreeText {
                                node,
                                term,
                                emitted,
                            };
                            Ok(Observe::Continue)
                        }
                    }
                }
            }
            other => {
                self.cursor = other;
                Err(WalkError::NotDecoding)
            }
        }
    }

    fn branch_trie(&self, node: NodeId) -> &TokenTrie {
        match self.tree.node(node) {
            StencilNode::Branch { trie } => trie,
            _ => unreachable!("InBranch cursor on a non-Branch node"),
        }
    }

    fn free_span(&self, node: NodeId) -> &FreeTextSpan {
        match self.tree.node(node) {
            StencilNode::FreeText(span) => span,
            _ => unreachable!("InFreeText cursor on a non-FreeText node"),
        }
    }
}
