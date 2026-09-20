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
    /// A free-text span hit its hard limit and was force-closed. The token is
    /// committed with the span's completion after it — the text that finishes
    /// the value as written ([`TerminatorState::completion`]) — exactly as for
    /// an intercepted EOS, which is replaced by that text.
    SpanForcedClosed,
    /// A free-text span ended via an EOS sample (`eos_ends`).
    SpanEos,
    /// A lookahead span closed on a delimiter token its successor does not
    /// continue with — a `]` where the element's `}` comes next. The token is
    /// DROPPED (not committed) and the cursor advances, so the successor writes
    /// the structure in its place.
    ///
    /// Pushed back instead, the token is out of grammar at the successor and
    /// the walk bails with it already committed: live, a Cline `read_files`
    /// call ended `"end_line": 3420]}}}` and was not JSON. Dropped, the
    /// grammar writes the element's `}` and the model chooses again from
    /// what is legal there.
    DelimiterDropped,
    /// A free-text span's token could not be committed as written, and the
    /// session has the bytes to commit in its place
    /// ([`StencilSession::take_rewrite`]): a character escaped inside a string
    /// (`closed: false`), or a value ended at the first byte that could not
    /// continue it and completed as written (`closed: true`).
    Repaired { closed: bool },
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
    /// The bytes to commit in place of the token `observe` just saw, when it
    /// could not be committed as written — see [`Self::take_rewrite`].
    rewrite: Option<Vec<u8>>,
}

impl StencilSession {
    /// Begin a session at the tree's root.
    pub fn new(tree: Arc<StencilTree>) -> Self {
        let root = tree.root();
        StencilSession {
            tree,
            cursor: Cursor::At(root),
            pushback: None,
            rewrite: None,
        }
    }

    /// The bytes to commit in place of the token the last `observe` saw, if it
    /// cannot be committed as written. Empty bytes mean it contributes nothing
    /// and is dropped.
    ///
    /// Set by a free-text span that had to escape a character, end a malformed
    /// value, complete a value cut short by EOS (replacing the EOS), or complete
    /// one at its hard limit (after the token). The caller re-tokenizes these
    /// bytes and commits them in the token's place; the walk has already moved
    /// on as if the model had written them.
    pub fn take_rewrite(&mut self) -> Option<Vec<u8>> {
        self.rewrite.take()
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
    /// free/branch content means more decoding follows and a bare close is
    /// the right forced token.
    ///
    /// Anywhere else — a span that retires into further content, a span that
    /// consumes its own close token (tool-call values), or a cursor not in
    /// free text at all (static prefill, a branch decision) — this is false
    /// and a forced close stays bare.
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
            // follow cleanly. `observe` pushes back only a token that is —
            // anything else it drops — so the bail below is the failsafe it
            // would take to reach it.
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
        self.rewrite = None;
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
                // **EOS never ends a turn the stencil is still steering.**
                //
                // It is intercepted here and the cursor moves on, whatever kind
                // of span this is. What the stencil has not finished emitting,
                // it emits — the structural statics after this span are injected
                // in the EOS's place, so the call is closed rather than
                // abandoned half-written.
                //
                // This used to be reachable only through `eos_ends` or a
                // `close_token`, which coupled "may EOS end this span" to
                // "does this span have a closing token" — two unrelated
                // questions. A byte-terminated span had neither, so EOS fell
                // through to `term.feed`, matched nothing, and left the cursor
                // parked in the value while the decode loop sealed the turn on
                // the EOS it had just sampled. Measured on the persisted
                // substrate: **244 of 259 turns** wrote a complete, correct
                // function block and never closed it, every one discarded.
                //
                // A stencil is a guarantee about what can be emitted; a token
                // that ends the turn from inside one is a hole in it, and the
                // free-decode span was the only place that hole existed.
                // Every end token, not only the canonical one: the decode loop
                // seals on any of them, and one this missed ended a reply from
                // inside an open think block mid-sentence.
                if self.tree.is_end(token) {
                    self.cursor = Cursor::At(span.next);
                    return Ok(match span.eos_ends {
                        // The span is allowed to end this way — the think-steer
                        // tree's final span, whose whole job is to run to EOS.
                        true => Observe::SpanEos,
                        // It is not. The span's terminator never fired, so
                        // whatever finishes the value was never written: the
                        // EOS is replaced by that text — the element is closed
                        // properly rather than running into whatever follows —
                        // or, with nothing to write, swallowed so it never
                        // reaches the sequence (and never seals it).
                        false => {
                            let completion = term.completion();
                            if !completion.is_empty() {
                                self.rewrite = Some(completion);
                            }
                            Observe::TokenClosedDrop
                        }
                    });
                }
                // A close *token* ends the span before any byte terminator runs.
                // `suppress_close` drops the closing token (the successor
                // prefills the continuation in its place); otherwise it is kept
                // (committed).
                if let Some(ct) = span.close_token {
                    if token == ct {
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
                        // consumes it instead of re-emitting it — if the next
                        // node continues with it. If not, drop it and let the
                        // next node write what belongs there.
                        if span.term.is_lookahead() && consumed == 0 {
                            if !self.continues_with(span.next, token) {
                                return Ok(Observe::DelimiterDropped);
                            }
                            self.pushback = Some(token);
                        }
                        Ok(Observe::SpanClosed {
                            leftover: bytes.len() - consumed,
                        })
                    }
                    // The value ended at a byte that could not continue it, and
                    // was completed as written.
                    Feed::Rewrite {
                        bytes: out,
                        closed: true,
                    } => {
                        self.cursor = Cursor::At(span.next);
                        self.rewrite = Some(out);
                        Ok(Observe::Repaired { closed: true })
                    }
                    Feed::Rewrite {
                        bytes: out,
                        closed: false,
                    } => Ok(self.continue_span(node, &span, term, emitted, bytes, Some(out))),
                    Feed::Continue => {
                        Ok(self.continue_span(node, &span, term, emitted, bytes, None))
                    }
                }
            }
            other => {
                self.cursor = other;
                Err(WalkError::NotDecoding)
            }
        }
    }

    /// A span token that did not close the span: stay in it, unless this token
    /// reached the hard limit. `rewritten` is what the token commits in place
    /// of its own `bytes`, when it cannot be committed as written.
    fn continue_span(
        &mut self,
        node: NodeId,
        span: &FreeTextSpan,
        term: TerminatorState,
        emitted: u32,
        bytes: &[u8],
        rewritten: Option<Vec<u8>>,
    ) -> Observe {
        if emitted >= span.limits.forced_after {
            // **Cut short is interrupted too**, and closes the same way the EOS
            // path does: the terminator never fired, so the text that finishes
            // the value was never written, and the tree writes it after the
            // token.
            //
            // This went straight to the successor, which for a function-block
            // value is the next `<parameter=…>` — so a value that ran to the
            // limit left its element open, and a reader bounding it by the
            // first `</parameter>` took the *next* argument's close as its own.
            // A `reflect` whose thoughts ran on came back missing `feeling`, a
            // required argument the grammar had in fact forced, with the
            // feeling itself swallowed into the thoughts.
            let completion = term.completion();
            if rewritten.is_some() || !completion.is_empty() {
                let mut written = rewritten.unwrap_or_else(|| bytes.to_vec());
                written.extend(completion);
                self.rewrite = Some(written);
            }
            self.cursor = Cursor::At(span.next);
            return Observe::SpanForcedClosed;
        }
        self.cursor = Cursor::InFreeText {
            node,
            term,
            emitted,
        };
        match rewritten {
            Some(written) => {
                self.rewrite = Some(written);
                Observe::Repaired { closed: false }
            }
            None => Observe::Continue,
        }
    }

    /// Whether `node` can take `tok` as its first token: a static that opens on
    /// it, or a branch with an arm that does.
    fn continues_with(&self, node: NodeId, tok: TokenId) -> bool {
        match self.tree.node(node) {
            StencilNode::Static { tokens, .. } => tokens.first() == Some(&tok),
            StencilNode::Branch { trie } => trie.step(trie.root(), tok).is_some(),
            StencilNode::FreeText(_) | StencilNode::End => false,
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
