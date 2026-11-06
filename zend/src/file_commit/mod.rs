//! `file_edit_and_commit`'s work, apart from the connector that calls it
//! (`docs/mcp_file_tools.md` §3).
//!
//! | Module | |
//! |---|---|
//! | [`edits`] | exact `old` → `new` replacements, each naming one place |
//! | [`change`] | the changes a call makes: edit, write, delete, move |
//! | [`commit`] | the changes as one commit on the branch's tip, published whole or not at all |

pub mod change;
pub mod commit;
pub mod edits;
