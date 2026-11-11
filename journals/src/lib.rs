//! The memory backbone (`docs/memory_backbone.md`): every person's journal
//! tree and their groups', kept as YAML in one git repository and read and
//! written by every AI on the estate.
//!
//! | Module | |
//! |---|---|
//! | [`time`], [`period`] | instants, and the calendar stretches windows fold by |
//! | [`slug`] | names in the tree |
//! | [`scope`] | the fixed sections and channels, and which are personal |
//! | [`line`] | one entry: a note, an episode, a fold |
//! | [`experience`] | what an episode npcd writes for a bot was to the bot |
//! | [`files`] | each file's format, read and written exactly |
//! | [`path`] | where a journal is, and how a caller writes it |
//! | [`tree`] | the repository in memory: load, check, render, dirty, folds |
//! | [`actor`] | who writes, and what they may read |
//! | [`ops`] | every change to the tree, checked and applied |
//! | [`view`] | what readers see: index, an opened journal, frames, grep |
//! | [`brief`] | the background rebuilds: what is due, its question, its answer |
//! | [`store`] | the repository on disk: commits, pushes, fetches, replays |
//! | [`live`] | a checkout kept open and in step with origin for a service's life |
//! | [`error`] | a broken file, a refused operation |

pub mod actor;
pub mod brief;
pub mod diaries;
pub mod error;
pub mod experience;
pub mod files;
pub mod line;
pub mod live;
pub mod ops;
pub mod path;
pub mod period;
pub mod scope;
pub mod slug;
pub mod store;
pub mod terms;
pub mod time;
pub mod tree;
pub mod view;

mod yaml;

#[cfg(test)]
mod testing;
