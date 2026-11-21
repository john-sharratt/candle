//! Narrows a branch of the grammar to the arms that the tokens so far agree with,
//! so a held branch completes on one of them.
//!
//! A hold keeps the arms a branch is held to and how many of its tokens have been
//! written. Each token keeps only the arms that wrote it, and the hold is spent once
//! no arm agrees or the branch completes. The arms are shared rather than copied, so
//! a hold costs one small index list per call and nothing in the compiled tree.

use super::*;

/// The arms of one branch the walk is held to, token by token — see
/// [`StencilDriver::hold_branches`]. `live` are the arms (indices into `arms`)
/// that every token so far has agreed with, `depth` how many tokens of the
/// branch have been written.
///
/// The arms are shared, not copied: a turn's held arms are found once and every
/// call it opens starts from the same `Arc`, so a hold costs one small index
/// list per call and nothing in the compiled tree.
#[derive(Clone)]
pub(super) struct ArmHold {
    pub(super) node: NodeId,
    pub(super) arms: Arc<[Vec<TokenId>]>,
    pub(super) live: Vec<u32>,
    pub(super) depth: usize,
}

impl ArmHold {
    pub(super) fn new(node: NodeId, arms: Arc<[Vec<TokenId>]>) -> Option<Self> {
        let live: Vec<u32> = (0..arms.len() as u32)
            .filter(|&i| !arms[i as usize].is_empty())
            .collect();
        (!live.is_empty()).then_some(ArmHold {
            node,
            arms,
            live,
            depth: 0,
        })
    }

    /// The tokens of `frontier` some live arm writes next.
    pub(super) fn narrow(&self, frontier: &AllowedSet) -> Vec<TokenId> {
        self.live
            .iter()
            .map(|&i| self.arms[i as usize][self.depth])
            .filter(|&t| frontier.contains(t))
            .collect()
    }

    /// Take `token`: keep the arms that wrote it. `false` once the branch is
    /// complete (an arm ended on it) or no arm agreed — the hold is spent.
    pub(super) fn take(&mut self, token: TokenId) -> bool {
        let depth = self.depth;
        let arms = &self.arms;
        self.live
            .retain(|&i| arms[i as usize].get(depth) == Some(&token));
        self.depth += 1;
        !self.live.is_empty()
            && !self
                .live
                .iter()
                .any(|&i| self.arms[i as usize].len() == self.depth)
    }
}
