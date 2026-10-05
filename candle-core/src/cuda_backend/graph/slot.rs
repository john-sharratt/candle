//! The executables kept for one segment ordinal.
//!
//! A segment's ordinal is its place in its wave, and the engine runs waves of
//! more than one shape — a prompt prefill, a decode or verify step, a draft
//! step — whose segment `i` is a different graph each. An ordinal therefore
//! keeps one executable per shape in steady use, the most recently used first,
//! so a change of shape folds into that shape's executable in place instead of
//! instantiating afresh (tens of milliseconds over a wave's segments).

use super::exec::{audit, GraphExec, Template};
use super::ComputeStream;
use crate::Result;
use cudarc::driver::sys;

/// How many executables an ordinal keeps — one per wave shape in steady use.
const EXECS_PER_SLOT: usize = 4;

#[derive(Default)]
pub(super) struct ExecSlot {
    /// Most recently used first.
    execs: Vec<GraphExec>,
}

impl ExecSlot {
    /// Fold `graph` into the first executable that accepts it in place —
    /// trying, most recently used first, every one with its node count, since
    /// two shapes can share a count — or instantiate it, dropping the least
    /// recently used beyond [`EXECS_PER_SLOT`]. Returns the executable to
    /// launch and whether it was updated in place.
    ///
    /// # Safety
    ///
    /// `graph` must be a valid graph owned by the caller; it is consumed.
    pub(super) unsafe fn fold(
        &mut self,
        graph: sys::CUgraph,
        on: &ComputeStream,
        nodes: usize,
    ) -> Result<(&GraphExec, bool)> {
        let template = Template(graph);
        audit(template.0, nodes)?;
        let mut folded = None;
        for i in 0..self.execs.len() {
            if self.execs[i].try_fold(&template, on, nodes)? {
                folded = Some(i);
                break;
            }
        }
        let updated = match folded {
            Some(i) => {
                let exec = self.execs.remove(i);
                self.execs.insert(0, exec);
                true
            }
            None => {
                let exec = GraphExec::instantiate_audited(template, on, nodes)?;
                self.execs.insert(0, exec);
                self.execs.truncate(EXECS_PER_SLOT);
                false
            }
        };
        Ok((&self.execs[0], updated))
    }
}
