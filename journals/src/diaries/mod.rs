//! Diaries (`docs/private_mode.md` §5): topic-based notes kept in the journal
//! repository, each sealed in a file of its own under a key its passphrase
//! alone unwraps. They are no part of the journal tree: the tree's load never
//! reads `diaries/`, so no fold, summary, digest, cue, index, grep or frame
//! does either.
//!
//! | Module | |
//! |---|---|
//! | [`layout`] | where diaries live: `diaries/<id>.bin` and the one salt |
//! | [`crypto`] | passphrase → kek, a diary's key wrapped under it, its file sealed and opened |
//! | [`book`] | what a diary holds once opened: its topic and its entries in time order |

pub mod book;
pub mod crypto;
pub mod layout;
