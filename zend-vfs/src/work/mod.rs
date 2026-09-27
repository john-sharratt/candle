//! A conversation's work on a branch: committing it, and merging other
//! commits into it.
//!
//! A conversation reads each repository at its own base ([`crate::vfs::Base`])
//! with its uncommitted changes over it, and neither ever lives in a folder.
//! Two operations join that work to the branch everyone shares, and both keep
//! one rule: **nothing anyone wrote is lost**.
//!
//! - **Commit** ([`Committing`]): one commit of the conversation's work,
//!   published to origin whole or not at all. It is refused — with nothing
//!   written anywhere — while a merge's conflicts are unsettled, when the
//!   branch holds commits the conversation does not have, and when origin
//!   moves between the check and the push. The conversation's copy is left
//!   exactly as it was, and a merge is the way on.
//! - **Merge** ([`merge_into`]): another commit — the branch as origin now
//!   holds it, usually — brought into the conversation's copy, as `git merge`
//!   brings it into a working tree. Where both sides changed the same lines,
//!   both are kept between conflict markers in the conversation's copy, and
//!   the path is flagged until the conversation settles it; the next commit
//!   then records the merge.

mod commit;
pub mod held;
mod merge;
mod three_way;

pub use commit::{Committing, Landed, NotCommitted};
pub use merge::{merge_into, Merged};
pub use three_way::{keep_ours, ThreeWay};

use crate::{GitError, VfsError};

/// A refusal from a conversation's file store, as a git operation's error.
fn files(e: VfsError) -> GitError {
    GitError::invalid(e.to_string())
}
