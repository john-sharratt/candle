//! Writing objects and moving refs. Nothing here touches the working tree,
//! the index or `HEAD`.

pub mod apply;
pub mod blobs;
pub mod commit;
pub mod fast_import;
pub mod merge_text;
pub mod merge_tree;
pub mod pick;
pub mod ref_txn;
pub mod scratch;
pub mod tags;
pub mod upstream;
