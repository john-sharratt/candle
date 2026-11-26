//! The online walk of a stencil tree: for each step, what the decode loop does next.
//!
//! The driver is the sampler's counterpart to the simulator. Each step yields a static
//! run to prefill, a branch to mask, a span to decode freely, or the end of the walk,
//! and each sampled token is fed back to be committed as written, rewritten or dropped.
//! A held branch is narrowed to the tools the turn was given, so a tool the turn was
//! not given cannot be written. The opening scaffold is walked once to be prefilled,
//! and the driver walks it again to the same position, so the prefill and the driver
//! agree on where the first real choice lies.
//!
//! - `hold`: narrows a branch to the arms the tokens agree with.
//! - `outcome`: decides how a sampled token is committed.
//! - `trace`: tallies the path a walk took.
//! - `tests_arrays`: tests of array and string-close decisions.
//! - `tests_integers`: tests of integer digits and endings.
//! - `tests_strings`: tests of string and free-text values.

use std::sync::Arc;

use super::held_arms::HeldBranches;
use super::held_values::HeldValues;
use super::mask::AllowedSet;
use super::never_twice::NeverTwice;
use super::session::{Observe, StencilAction, StencilSession};
use super::tree::{NodeId, StencilTree};
use super::vocab::TokenId;

/// A live walk of one tree attached to a decoding sequence.
///
/// `Clone` so a speculative step can advance a copy along a drafted block and
/// learn where the block has to stop, without moving the walk the sequence
/// commits through.
#[derive(Clone)]
pub struct StencilDriver {
    session: StencilSession,
    done: bool,
    stats: PathStats,
    /// The arms each held branch is narrowed to, for the whole walk.
    held: Option<Arc<HeldBranches>>,
    /// The tools the turn may call once ([`Self::never_twice`]).
    never_twice: Option<Arc<NeverTwice>>,
    /// The branch being written under a hold, if one is.
    hold: Option<ArmHold>,
}

impl StencilDriver {
    /// Begin a walk at the tree's root.
    pub fn new(tree: Arc<StencilTree>) -> Self {
        StencilDriver {
            session: StencilSession::new(tree),
            done: false,
            stats: PathStats::default(),
            held: None,
            never_twice: None,
            hold: None,
        }
    }

    /// Hold out of every name branch a tool of `tools` the walk has already
    /// named ([`NeverTwice`]): a turn calls each of them at most once.
    pub fn never_twice(&mut self, tools: Arc<NeverTwice>) {
        self.never_twice = Some(tools);
    }

    /// The branch the cursor is in and whether it waits on its first token
    /// ([`StencilSession::in_branch`]).
    pub fn in_branch(&self) -> Option<(NodeId, bool)> {
        self.session.in_branch()
    }

    /// Hold every branch `held` names to its arms: each masked step of such a
    /// branch, from its first token, is narrowed to the tokens a held arm
    /// writes next, so the branch completes on one of them and on nothing else.
    ///
    /// For a tool call these are the tools the turn was given, on every name
    /// branch of the call loop
    /// ([`TriggerRegistry::holding`](super::TriggerRegistry::holding)): the
    /// catalog's grammar is compiled once, and the turn's own tools are a
    /// narrowing of it, so a tool the turn was not given cannot be written. A
    /// frontier that holds none of the held arms' next tokens — arms the branch
    /// does not have — releases the hold, and the step is masked to the whole
    /// frontier as usual.
    pub fn hold_branches(&mut self, held: Arc<HeldBranches>) {
        self.held = Some(held);
    }

    /// Hold every keyed field the walk writes to `values`
    /// ([`StencilSession::hold_values`], `docs/stencil_tree.md` §9.2).
    pub fn hold_values(&mut self, values: Arc<HeldValues>) {
        self.session.hold_values(values);
    }

    /// Hold the tree's first branch ([`StencilTree::first_branch`]) to the
    /// single arm `arm`: the tool the reasoning named
    /// ([`named_arm`](super::named_arm)). Called before the walk starts.
    pub fn steer_first_branch(&mut self, arm: Vec<TokenId>) {
        if let Some(node) = self.tree().first_branch_node() {
            self.hold = ArmHold::new(node, Arc::from(vec![arm]));
        }
    }

    /// The arms the first branch may complete on: those it is held to, when it
    /// is. `None` when nothing holds it.
    pub fn held_arms(&self) -> Option<Vec<Vec<TokenId>>> {
        if let Some(h) = &self.hold {
            return Some(h.live.iter().map(|&i| h.arms[i as usize].clone()).collect());
        }
        let node = self.tree().first_branch_node()?;
        self.held.as_ref()?.arms_at(node).map(|arms| arms.to_vec())
    }

    /// Whether the walk has finished (the caller should drop the driver and
    /// resume free decode).
    pub fn is_done(&self) -> bool {
        self.done
    }

    /// The path the walk took so far — for the steering finish trace.
    pub fn stats(&self) -> PathStats {
        self.stats
    }

    /// The compiled tree being walked.
    pub fn tree(&self) -> &StencilTree {
        self.session.tree()
    }

    /// Whether the walk is inside a free-text span that the next decoded token
    /// continues — see [`StencilSession::mid_free_span`]. While it holds, the
    /// next [`step`](Self::step) is `Free` and nothing is masked.
    pub fn mid_free_span(&self) -> bool {
        !self.done && self.session.mid_free_span()
    }

    /// Whether the cursor sits in a terminal free-text span — the only place
    /// the sampler's hard-cap closing-statement script may play — see
    /// [`StencilSession::in_terminal_close_span`].
    pub fn in_terminal_close_span(&self) -> bool {
        self.session.in_terminal_close_span()
    }

    /// Walk the tree's opening scaffold, for a turn that **begins** inside a
    /// grammar rather than entering one on a trigger token.
    ///
    /// Returns every leading `Static` run concatenated, plus the first action
    /// that actually needs the sampler.  The runs are deterministic — no token
    /// has been observed yet, so the walk from the root cannot branch — which is
    /// what lets the caller seed them into the turn's assistant prefill and then
    /// re-derive the same position here to keep the driver in step with the K/V.
    ///
    /// Prefilling that scaffold rather than decoding it is both cheaper and
    /// stronger: it costs one prefill pass instead of a decode step per token,
    /// and the tokens are *written* rather than chosen, so no sampling outcome
    /// can decline to produce them.  The first sampled token of the turn is then
    /// the first genuine decision the grammar leaves open — a masked branch.
    pub fn opening(&mut self) -> (Vec<TokenId>, StepMask) {
        let mut scaffold = Vec::new();
        loop {
            match self.step() {
                StepMask::Prefill(run) => scaffold.extend_from_slice(&run),
                action => return (scaffold, action),
            }
        }
    }

    /// What to do for the next decode step: prefill a static run, mask a branch,
    /// free-decode a span, or finish.  After a `Prefill` the caller injects the
    /// run and calls `step` again; after `Branch`/`Free` it samples a token under
    /// the constraint and feeds it back via [`accept`](Self::accept).
    pub fn step(&mut self) -> StepMask {
        loop {
            match self.session.next_action() {
                StencilAction::Prefill(toks) => {
                    // An empty run (possible after a fully-consumed push-back)
                    // carries no tokens — skip straight to the next node.
                    if toks.is_empty() {
                        continue;
                    }
                    self.stats.prefills += 1;
                    self.stats.prefill_tokens += toks.len() as u32;
                    return StepMask::Prefill(toks);
                }
                StencilAction::MaskedDecode(set) => {
                    self.stats.branch_tokens += 1;
                    self.enter_hold();
                    if let Some(hold) = &self.hold {
                        let want = hold.narrow(&set);
                        if !want.is_empty() {
                            return StepMask::Branch(AllowedSet::from_tokens(want));
                        }
                        self.hold = None;
                    }
                    return StepMask::Branch(set);
                }
                StencilAction::FreeDecode {
                    close_boost,
                    opens_value,
                    barred,
                } => {
                    self.stats.free_tokens += 1;
                    return StepMask::Free {
                        close_boost,
                        opens_value,
                        barred,
                    };
                }
                StencilAction::Exit => {
                    self.done = true;
                    return StepMask::Done;
                }
            }
        }
    }

    /// At a masked step: keep the hold on the branch the cursor is in, drop one
    /// left on another, and start one when the cursor waits on the first token
    /// of a held branch.
    fn enter_hold(&mut self) {
        let Some((node, at_root)) = self.session.in_branch() else {
            self.hold = None;
            return;
        };
        if self.hold.as_ref().is_some_and(|h| h.node == node) {
            return;
        }
        self.hold = None;
        if !at_root {
            return;
        }
        let held = self.held.as_ref().and_then(|h| h.arms_at(node)).cloned();
        // A tool the turn calls once, already named, is held out of this name
        // branch — narrowing the turn's hold when there is one, the branch's
        // whole catalog when there is not.
        let narrowed = self.never_twice.as_ref().and_then(|once| {
            let arms: Vec<Vec<TokenId>> = match &held {
                Some(arms) => arms.to_vec(),
                None => self.tree().branch(node)?.arms(),
            };
            once.narrow(self.tree(), node, &arms, self.session.named_tools())
        });
        if let Some(arms) = narrowed.or(held) {
            self.hold = ArmHold::new(node, arms);
        }
    }

    /// Feed back the token sampled under the constraint from the preceding
    /// `Branch`/`Free` [`step`](Self::step).  `bytes` are its decoded bytes (used
    /// by free-text terminators).  An out-of-grammar token makes the session bail
    /// — its closing run is then returned as a `Prefill` on the next `step`.
    ///
    /// Returns [`Healed::Rewrite`] when the token is committed as other bytes —
    /// a span that closed strictly inside it, or a repair (see
    /// [`StencilSession::take_rewrite`]). Returns [`Healed::Drop`] when a
    /// `suppress_close` token-closed span ended on this token (the close token
    /// is dropped and the successor prefills a steering continuation), when an
    /// EOS was intercepted inside a span with nothing to close, when a
    /// lookahead value ended on a delimiter the grammar does not continue with,
    /// and when a repair leaves the token nothing to contribute.
    pub fn accept(&mut self, token: TokenId, bytes: &[u8]) -> Healed {
        if self.hold.as_mut().is_some_and(|hold| !hold.take(token)) {
            self.hold = None;
        }
        let observed = self.session.observe(token, bytes);
        let rewrite = self.session.take_rewrite();
        let Ok(observe) = observed else {
            return Healed::No;
        };
        match observe {
            Observe::SpanClosed { leftover } if leftover > 0 && leftover < bytes.len() => {
                self.stats.heals += 1
            }
            // A close signal the span keeps to itself: the token never reaches
            // the sequence, and the tree writes whatever it owes — a steering
            // phrase, or the structure the EOS would otherwise have cut short.
            Observe::TokenClosedDrop => self.stats.intercepted_closes += 1,
            // A wrong delimiter after a value: dropped, and the successor
            // writes the right one.
            Observe::DelimiterDropped => self.stats.dropped_delimiters += 1,
            Observe::Repaired { .. } => self.stats.repairs += 1,
            Observe::Bailed => self.stats.bailed = true,
            _ => {}
        }
        healing(observe, rewrite, bytes)
    }
}

mod hold;
mod outcome;
#[cfg(test)]
mod tests_arrays;
#[cfg(test)]
mod tests_integers;
#[cfg(test)]
mod tests_strings;
mod trace;

use hold::*;
pub use outcome::*;
pub use trace::*;
