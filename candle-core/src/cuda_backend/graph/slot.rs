//! The executables kept for one segment ordinal.
//!
//! A segment's ordinal is its place in its wave, and the engine runs waves of
//! more than one shape — a prompt prefill, a decode step, a draft step, a verify
//! block of each length an adaptive draft depth produces — whose segment `i` is
//! a different graph each. An ordinal therefore keeps one executable per shape
//! in steady use, so a recapture folds into the executable already holding its
//! shape and the driver rewrites only addresses and scalars.
//!
//! Folding into an executable of another shape also succeeds when the topology
//! matches, but it rewrites grids and functions too and costs several times as
//! much (see [`audit`]). So a shape that recurs is given an executable of its
//! own while the ordinal has room, and only a shape seen once — a prompt's own
//! length — is folded over another's.

use std::collections::VecDeque;

use super::exec::{audit, GraphExec, Template};
use super::ComputeStream;
use crate::Result;
use cudarc::driver::sys;

/// How many executables an ordinal keeps — one per shape in steady use. A
/// single sequence's adaptive draft alone runs a draft step and verify blocks
/// of up to nine rows.
const EXECS_PER_SLOT: usize = 16;

/// How many recent shapes an ordinal remembers, to tell a recurring shape from
/// a one-off.
const SHAPES_REMEMBERED: usize = 32;

/// How a recapture reached its executable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Folded {
    /// Into the executable already holding its shape: arguments rewritten.
    InPlace,
    /// Into an executable of another shape: grids and functions rewritten too.
    Reshaped,
    /// Instantiated afresh.
    Instantiated,
}

#[derive(Default)]
pub(super) struct ExecSlot {
    /// Most recently used first.
    execs: Vec<GraphExec>,
    /// The shapes of the last [`SHAPES_REMEMBERED`] recaptures, newest last.
    seen: VecDeque<u64>,
}

impl ExecSlot {
    /// Fold `graph` into the executable holding its shape; failing that, give
    /// a recurring shape a new executable while there is room; failing that,
    /// fold it into the first executable that accepts it, most recently used
    /// first, or instantiate it, dropping the least recently used beyond
    /// [`EXECS_PER_SLOT`]. Returns the executable to launch and how it got
    /// there.
    ///
    /// # Safety
    ///
    /// `graph` must be a valid graph owned by the caller; it is consumed.
    pub(super) unsafe fn fold(
        &mut self,
        graph: sys::CUgraph,
        on: &ComputeStream,
        nodes: usize,
    ) -> Result<(&GraphExec, Folded)> {
        let template = Template(graph);
        let shape = audit(template.0, nodes)?;
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
        if outcome.is_none() && !(recurring && self.execs.len() < EXECS_PER_SLOT) {
            for i in 0..self.execs.len() {
                if self.execs[i].try_fold(&template, on, nodes, shape)? {
                    outcome = Some((i, Folded::Reshaped));
                    break;
                }
            }
        }
        let folded = match outcome {
            Some((i, how)) => {
                let exec = self.execs.remove(i);
                self.execs.insert(0, exec);
                how
            }
            None => {
                let exec = GraphExec::instantiate_audited(template, on, nodes, shape)?;
                self.execs.insert(0, exec);
                self.execs.truncate(EXECS_PER_SLOT);
                Folded::Instantiated
            }
        };
        Ok((&self.execs[0], folded))
    }
}
