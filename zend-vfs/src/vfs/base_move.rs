//! Moving a store's base to another commit, carrying its session changes across.
//!
//! Every changed path is resolved before anything is written, so a refusal from
//! the resolver, or changes that would no longer fit the size cap, leaves the
//! store as it was. The move is held for its whole length, so no change of the
//! conversation's can land between reading the old base and replacing it. When
//! nothing is carried, the old tree is never read, so a base the repository no
//! longer holds is still left behind. Paths left in conflict are reported back
//! to the caller, which settles them before the conversation commits.

use super::*;

impl VfsStore {
    /// Move the store onto `to` — and, given `branch`, onto that branch —
    /// carrying the session's changes across ([`carry`]): every changed path
    /// the old and new trees hold differently, and each of `extra`, is put
    /// to `resolve`, and its answer recorded against the new tree. Returns
    /// every path left in conflict. Everything is resolved first, so a
    /// failure — a refusal from `resolve`, or changes that would no longer
    /// fit the size cap — leaves the store as it was. Only a store over a
    /// branch has a base to move.
    pub fn move_base(
        &self,
        branch: Option<BranchName>,
        to: Base,
        extra: &[String],
        resolve: &mut Resolver<'_>,
    ) -> Result<Vec<String>, VfsError> {
        let Lower::Branch { source, rev } = &self.lower else {
            return Err(VfsError::Unwritable(
                "only a store over a branch has a base to move".into(),
            ));
        };
        let unreadable =
            |e| VfsError::Unreadable(format!("the base could not be read from git: {e}"));
        // Held for the whole move: no change of the conversation's can land
        // between reading the base the chains are made on and replacing it.
        let mut upper = self.upper.write().unwrap();
        // With nothing to carry — no change of the conversation's, and no
        // path the caller asks to reconsider — the old tree is never read, so
        // a base the repository no longer holds is still left behind.
        let nothing_to_carry = upper.chains.is_empty() && extra.is_empty();
        let old_tree = match (&upper.base, nothing_to_carry) {
            (_, true) => Arc::new(Tree::empty()),
            (Some(from), false) => source.tree_of(from).map_err(unreadable)?,
            (None, false) => {
                let rev = rev.read().unwrap().clone();
                let from = source.base_at(&rev).map_err(unreadable)?;
                source.tree_of(&from).map_err(unreadable)?
            }
        };
        let old = View::Branch {
            source,
            tree: old_tree,
        };
        let new = View::Branch {
            source,
            tree: source.tree_of(&to).map_err(unreadable)?,
        };
        let mut chains = upper.chains.clone();
        carry::carry(&mut chains, &old, &new, extra, resolve)?;
        let moved = Upper {
            chains,
            base: Some(to),
        };
        if moved.bytes() > MAX_BYTES {
            return Err(VfsError::Full);
        }
        *upper = moved;
        if let Some(branch) = branch {
            *rev.write().unwrap() = Rev::Branch(branch);
        }
        Ok(Self::conflicted(&upper))
    }

    /// Every path left in conflict by a move of the base, sorted — the
    /// conversation's to settle before it commits.
    pub fn conflicts(&self) -> Vec<String> {
        Self::conflicted(&self.upper.read().unwrap())
    }

    pub(super) fn conflicted(upper: &Upper) -> Vec<String> {
        let mut out: Vec<String> = upper
            .chains
            .iter()
            .filter(|(_, c)| c.conflict)
            .map(|(path, _)| path.clone())
            .collect();
        out.sort();
        out
    }

    /// Whether `norm`, about to hold `content`, stays in conflict: it was,
    /// and a conflict's markers are still in it.
    pub(super) fn still_conflicted(upper: &Upper, norm: &str, content: Option<&str>) -> bool {
        upper.chains.get(norm).is_some_and(|c| c.conflict) && content.is_some_and(has_markers)
    }
}
