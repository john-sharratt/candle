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
//! executable of its own when the library will keep one, and only a shape seen
//! once — a prompt's own length — is folded over another's. How many the slot
//! holds is the library's budget to decide, not a count here.
//!
//! **Only the learn reshapes.** Once the library stops admitting (`regime`),
//! a recapture folds only into the executable already holding its shape. Folded
//! over another's, it would leave that executable holding a different shape
//! from the one its trial proved — and the shape it was proven for would then
//! miss in turn.

use std::collections::VecDeque;
use std::mem::take;

use super::exec::{divergence, Audited, GraphExec, Template};
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
    /// Instantiated for this launch alone: the library does not admit it —
    /// banned, or no longer learning — and no executable of its shape holds.
    Transient,
}

/// Whether a fold may keep what it instantiates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Admit {
    /// The learn: a recurring shape is given an executable of its own, and a
    /// one-off is folded over another's.
    Keep,
    /// Nothing new is kept: into its own shape's executable, or once.
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
    /// Fold `template` into the executable holding its shape. Failing that,
    /// with [`Admit::Keep`]: give a recurring shape a new executable, or fold a
    /// one-off into the first executable that accepts it, most recently used
    /// first, or instantiate it and keep it. With [`Admit::Never`]: instantiate
    /// it for this launch alone.
    ///
    /// Returns the executable to launch when it is not kept (`Some`, the
    /// [`Folded::Transient`] case) — otherwise it is [`Self::front`] — how it
    /// got there, and, for an instantiation beside executables the slot
    /// already holds, why none of them would take it ([`Refusal`]).
    pub(super) fn fold(
        &mut self,
        template: Template,
        audited: Audited,
        on: &ComputeStream,
        nodes: usize,
        admit: Admit,
    ) -> Result<(Option<GraphExec>, Folded, Option<Refusal>)> {
        let keep = admit == Admit::Keep;
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
        if outcome.is_none() && keep && !recurring {
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
                refusal = self.execs.first().map(|last| Refusal {
                    kept_nodes: self.execs.iter().map(GraphExec::node_count).collect(),
                    divergence: divergence(&topology, last.topology()),
                });
                let exec = GraphExec::instantiate_audited(template, on, nodes, shape, topology)?;
                if !keep {
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

    /// Remove the executables holding `shape`.
    pub(super) fn remove_shape(&mut self, shape: u64) -> Vec<GraphExec> {
        let (gone, kept): (Vec<GraphExec>, Vec<GraphExec>) = take(&mut self.execs)
            .into_iter()
            .partition(|e| e.shape() == shape);
        self.execs = kept;
        gone
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
