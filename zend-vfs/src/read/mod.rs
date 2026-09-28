//! Reading repository state. Nothing here writes a ref, an object, the index
//! or the working tree.

pub mod attrs;
pub mod blame;
pub mod blob_reader;
pub mod diff;
pub mod grep;
pub mod head;
pub mod identity;
pub mod log;
pub mod patch;
pub mod record;
pub mod refs;
pub mod remote_branches;
pub mod status;
pub mod tags;
pub mod tree;
