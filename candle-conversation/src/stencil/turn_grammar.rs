//! A grammar a turn begins inside, with the values its calls are held to
//! (`docs/stencil_tree.md` §9.2.2).
//!
//! A turn armed by a trigger takes its registry's held values as the walk
//! starts. A turn that begins inside a grammar is armed before anything is
//! sampled, with no trigger to hand it any, so it carries its own: a file's
//! reading holds its opening call to the one file it reads.

use std::sync::Arc;

use super::driver::StencilDriver;
use super::held_values::HeldValues;
use super::tree::StencilTree;

/// A grammar a turn begins inside, and the values its calls are held to.
#[derive(Debug, Clone)]
pub struct TurnGrammar {
    pub tree: Arc<StencilTree>,
    /// `None` holds nothing: every value span keeps its own rule.
    pub held: Option<Arc<HeldValues>>,
}

impl TurnGrammar {
    /// `tree`, holding nothing.
    pub fn of(tree: Arc<StencilTree>) -> TurnGrammar {
        TurnGrammar { tree, held: None }
    }

    /// The same grammar, its calls held to `held`.
    pub fn holding(self, held: Arc<HeldValues>) -> TurnGrammar {
        TurnGrammar {
            held: Some(held),
            ..self
        }
    }

    /// A driver at the grammar's start, holding its values: the one every
    /// walk of it begins with.
    pub fn driver(&self) -> StencilDriver {
        let mut driver = StencilDriver::new(Arc::clone(&self.tree));
        if let Some(held) = &self.held {
            driver.hold_values(Arc::clone(held));
        }
        driver
    }
}
