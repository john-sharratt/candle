//! The `git_*` tool family, exercised against real repositories on disk.
//!
//! Split the way the family itself is: what answers questions about a
//! repository, what changes one, and what reaches a remote. The third is
//! separate because it needs a second repository to push to, and because
//! those two tools are the only ones whose effect can leave this machine.

mod harness;

mod reads;
mod remote;
mod writes;
