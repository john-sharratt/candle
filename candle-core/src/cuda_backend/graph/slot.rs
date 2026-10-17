//! The executables kept for one segment topology (`library`).
//!
//! Every recapture offered here has the same node kinds in the same order; they
//! differ in kernel variants, grids, addresses and scalars. Each shape in steady use — a decode
//! step, a verify block of each length an adaptive draft depth produces —
//! keeps an executable of its own, so a recapture folds into the executable
//! already holding its shape and the driver rewrites only addresses and
//! scalars.
//!
//! Folding into an executable of another shape also succeeds, since the
//! topology matches, but it rewrites grids too and costs several times as much
//! (see [`audit`](super::exec::audit)). So a shape that recurs is given an
//! executable of its own while the library learns, and a shape seen once — a
//! prompt's own length — is folded over another's. How many the slot holds is
//! the library's budget to decide, not a count here.
//!
//! **Once the learn is over, nothing reshapes.** A shape the slot holds folds
//! in place; any other is instantiated for its one launch. Reshaping a held
//! executable to every shape the trial and the lock met instead — the same
//! topology at another row count, often another kernel variant at a node —
//! bought nothing measurable: on `.5` (2026-10-17) a lock that reshaped
//! replayed as few segments as one that ran them once, since what it missed
//! were topologies it did not hold at all.

use std::collections::VecDeque;

use super::exec::{divergence, explaining, Audited, GraphExec, Template};
use super::ComputeStream;
use crate::Result;

/// How many recent shapes a topology remembers, to tell a recurring shape from
/// a one-off.
const SHAPES_REMEMBERED: usize = 32;

/// How a recapture reached its executable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Folded {
    /// Into the executable already holding its shape: arguments rewritten.
    InPlace,
    /// Into an executable of another shape: grids and functions rewritten too.
    Reshaped,
    /// Instantiated afresh and kept.
    Instantiated,
    /// Instantiated for this launch alone: the library does not admit it and
    /// the slot holds no executable it would fold into.
    Transient,
}

/// Whether a fold may keep what it instantiates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Admit {
    /// The learn: a recurring shape is given an executable of its own, and a
    /// one-off is folded over another's.
    Keep,
    /// Nothing is kept or reshaped: into its own shape's executable, or once.
    Never,
}

#[derive(Default)]
pub(super) struct ExecSlot {
    /// Most recently used first.
    execs: Vec<GraphExec>,
    /// The shapes of the last [`SHAPES_REMEMBERED`] recaptures, newest last.
    seen: VecDeque<u64>,
}

impl ExecSlot {
    /// Fold `template` into the executable holding its shape. Failing that:
    ///
    /// - with [`Admit::Keep`], give a recurring shape a new executable, or fold
    ///   a one-off into the first executable that accepts it, most recently
    ///   used first — or instantiate it and keep it;
    /// - with [`Admit::Never`], instantiate it for this launch alone.
    ///
    /// **A one-off never reshapes a `proven` topology's executables**: it runs
    /// once instead. Each of them holds a shape the trial proved, and nothing
    /// past the learn reshapes it back — reshaped by a prompt-length prefill in
    /// a later learn, the steady shape it held was instantiated for every
    /// launch of the lock that followed.
    ///
    /// Returns the executable to launch when it is not kept (`Some`, the
    /// [`Folded::Transient`] case) — otherwise it is [`Self::front`] — how it
    /// got there, and, for an instantiation beside executables the slot
    /// already holds, why none of them would take it ([`Refusal`]) — named only
    /// while the graph's debug log is on, since naming it costs driver calls.
    pub(super) fn fold(
        &mut self,
        template: Template,
        audited: Audited,
        on: &ComputeStream,
        nodes: usize,
        admit: Admit,
        proven: bool,
    ) -> Result<(Option<GraphExec>, Folded, Option<Refusal>)> {
        let Audited {
            shape, topology, ..
        } = audited;
        let recurring = self.seen.contains(&shape);
        self.seen.push_back(shape);
        if self.seen.len() > SHAPES_REMEMBERED {
            self.seen.pop_front();
        }

        let mut outcome = None;
        if let Some(i) = self.execs.iter().position(|e| e.shape() == shape) {
            if self.execs[i].try_fold(&template, on, nodes, shape)? {
                outcome = Some((i, Folded::InPlace));
            }
        }
        // A one-off of a proven topology runs once rather than reshape what the
        // trial proved.
        let one_off_of_proven = proven && !recurring;
        if outcome.is_none() && admit == Admit::Keep && !recurring && !proven {
            for i in 0..self.execs.len() {
                if self.execs[i].try_fold(&template, on, nodes, shape)? {
                    outcome = Some((i, Folded::Reshaped));
                    break;
                }
            }
        }
        let mut refusal = None;
        let folded = match outcome {
            Some((i, how)) => {
                let exec = self.execs.remove(i);
                self.execs.insert(0, exec);
                how
            }
            None => {
                if explaining() {
                    refusal = self.execs.first().map(|last| Refusal {
                        kept_nodes: self.execs.iter().map(GraphExec::node_count).collect(),
                        divergence: divergence(&topology, last.topology()),
                    });
                }
                let exec = GraphExec::instantiate_audited(template, on, nodes, shape, topology)?;
                if admit != Admit::Keep || one_off_of_proven {
                    return Ok((Some(exec), Folded::Transient, refusal));
                }
                self.execs.insert(0, exec);
                Folded::Instantiated
            }
        };
        Ok((None, folded, refusal))
    }

    /// The executable the last kept fold put first.
    pub(super) fn front(&self) -> Option<&GraphExec> {
        self.execs.first()
    }

    /// Its executables, most recently used first.
    pub(super) fn execs(&self) -> &[GraphExec] {
        &self.execs
    }

    /// Whether an executable holds `shape`.
    pub(super) fn holds_shape(&self, shape: u64) -> bool {
        self.execs.iter().any(|e| e.shape() == shape)
    }

    /// Remove the executable at the front — the one the last kept fold put
    /// there.
    pub(super) fn remove_front(&mut self) -> Option<GraphExec> {
        (!self.execs.is_empty()).then(|| self.execs.remove(0))
    }

    /// Executables kept, and their nodes.
    pub(super) fn held(&self) -> (usize, usize) {
        (
            self.execs.len(),
            self.execs.iter().map(GraphExec::node_count).sum(),
        )
    }
}

/// Why a recapture was instantiated beside the executables its topology held.
///
/// Every executable here has the recapture's node kinds in its order, so the
/// driver refusing all of them says something it will not rewrite differs — a
/// dependency edge, a node attribute, a function's context — or that the
/// topology key collided.
pub(super) struct Refusal {
    /// The node counts of the executables offered the fold, most recent first.
    pub(super) kept_nodes: Vec<usize>,
    /// Where the recapture first departs from the most recently used of them,
    /// or `None` when every node agrees — the driver refused the dependency
    /// edges or a node's attributes rather than the kernels.
    pub(super) divergence: Option<String>,
}
