//! The memory backbone (`docs/memory_backbone.md`): every person's journal
//! tree and their families', kept as YAML in one git repository and read and
//! written by every AI on the estate.
//!
//! | Module | |
//! |---|---|
//! | [`time`], [`period`] | instants, and the calendar stretches windows fold by |
//! | [`slug`] | names in the tree |
//! | [`scope`] | the fixed sections and clients, and which are personal |
//! | [`line`] | one entry: a note, an episode, a fold |
//! | [`files`] | each file's format, read and written exactly |
//! | [`path`] | where a journal is, and how a caller writes it |
//! | [`tree`] | the repository in memory: load, check, render, dirty, folds |
//! | [`actor`] | who writes, and what they may read |
//! | [`ops`] | every change to the tree, checked and applied |
//! | [`view`] | what readers see: index, an opened journal, frames, grep |
//! | [`store`] | the repository on disk: commits, pushes, fetches, replays |
//! | [`error`] | a broken file, a refused operation |

pub mod actor;
pub mod error;
pub mod files;
pub mod line;
pub mod ops;
pub mod path;
pub mod period;
pub mod scope;
pub mod slug;
pub mod store;
pub mod time;
pub mod tree;
pub mod view;

mod yaml;

#[cfg(test)]
mod testing;
